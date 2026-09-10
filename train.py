from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from ur5e_rl.callbacks import SuccessRateCurriculumCallback
from ur5e_rl.config import ENV_VERSION, STAGES, StageSpec
from ur5e_rl.env import make_env
from ur5e_rl.parallel import build_evaluation_env, build_parallel_env
from ur5e_rl.preflight import run_scripted_grasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UR5e v6 simple-flow：视觉追踪 + 连续真空吸附 SAC"
    )
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene.xml"))
    parser.add_argument("--out", type=Path, default=Path("runs/ur5e_visual_suction_v6"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=12, help="推荐 8~16")
    parser.add_argument(
        "--start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--gradient-steps", type=int, default=-1)
    parser.add_argument(
        "--ent-coef",
        type=str,
        default="auto_0.2",
        help="推荐 auto_0.2；也可传 0.05 这类固定数值。",
    )
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--start-stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--max-stage", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-replay-buffer", type=Path, default=None)
    parser.add_argument("--keep-replay-across-stages", action="store_true")
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--save-replay-on-interrupt", action="store_true")
    parser.add_argument("--allow-existing-out", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="不推荐。默认训练前必须通过 scripted smoke grasp。",
    )
    return parser.parse_args()


def parse_ent_coef(value: str) -> str | float:
    if value.startswith("auto"):
        return value
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError("固定 ent_coef 必须 > 0")
    return numeric


def selected_stages(start_stage: int, max_stage: int) -> tuple[StageSpec, ...]:
    if start_stage > max_stage:
        raise ValueError("--start-stage 不能大于 --max-stage")
    return tuple(stage for stage in STAGES if start_stage <= stage.index <= max_stage)


def validate_runtime(
    scene_path: Path,
    seed: int,
    device: str,
    num_envs: int,
    skip_preflight: bool,
) -> None:
    logical_cores = os.cpu_count() or 1
    if num_envs < 1:
        raise ValueError("--num-envs 必须 >= 1")
    if num_envs > logical_cores:
        print(f"警告：num_envs={num_envs} > logical CPU cores={logical_cores}")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但 torch.cuda.is_available() 为 False")
        gpu_index = torch.cuda.current_device() if device == "cuda" else int(device.split(":", 1)[1])
        print("CUDA GPU:", torch.cuda.get_device_name(gpu_index))
        print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)

    env = make_env(
        scene_path=scene_path,
        stage_name="stage1_static",
        render_mode=None,
        seed=seed,
        viewer_camera="overview_camera",
        training=False,
    )
    try:
        check_env(env, warn=True)
        obs, info = env.reset(seed=seed)
        print("ENV_VERSION:", ENV_VERSION)
        print("action_space:", env.action_space)
        print("observation_space:", env.observation_space)
        print("observation_shape:", obs.shape)
        print("MuJoCo timestep:", env.model.opt.timestep)
        print("table_top_z:", info["table_top_z"])
        print("workspace_min_z:", info["workspace_min_z"])
        if not skip_preflight:
            result = run_scripted_grasp(env, render=False)
            print("preflight:", result)
            if not result.success:
                raise RuntimeError(
                    "scripted preflight 没有完成 pregrasp->contact->held->lift。"
                    "底层物理链未通，禁止浪费 SAC 训练步数。"
                )
    finally:
        env.close()


def write_manifest(out_dir: Path, args: argparse.Namespace, stages: tuple[StageSpec, ...]) -> None:
    manifest = {
        "env_version": ENV_VERSION,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stages": [asdict(stage) for stage in stages],
        "v6_principles": [
            "恢复第1版纯 DLS IK：没有 posture regularization，没有额外软件关节包络。",
            "连续 suction ctrl=max(action[3],0)，减少 SAC 离散阈值探索困难。",
            "固定 horizon；success 一旦达成，余下时间维持最大 staged reward。",
            "reward 使用 reach/contact/held/lift/success 分阶段 max，不再堆大量 progress/event 奖励。",
            "碰桌/碰地先软惩罚，持续碰撞才终止；不再一碰就结束。",
            "训练前 scripted preflight 必须先打通整条物理链。",
        ],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    scene_path = args.scene.resolve()
    out_dir = args.out.resolve()
    seed = int(args.seed)
    num_envs = int(args.num_envs)

    if not scene_path.exists():
        raise FileNotFoundError(scene_path)
    if int(args.start_stage) > 1 and args.resume_model is None:
        raise ValueError("从 Stage2+ 开始必须提供同版本 v6 --resume-model")
    if args.resume_replay_buffer is not None and args.resume_model is None:
        raise ValueError("--resume-replay-buffer 必须与 --resume-model 一起使用")
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.allow_existing_out):
        raise FileExistsError(
            f"输出目录已非空：{out_dir}\n请换新 --out，或确认同一 v6 run 后加 --allow-existing-out。"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(1)

    validate_runtime(
        scene_path=scene_path,
        seed=seed,
        device=str(args.device),
        num_envs=num_envs,
        skip_preflight=bool(args.skip_preflight),
    )
    stages = selected_stages(int(args.start_stage), int(args.max_stage))
    write_manifest(out_dir, args, stages)

    first_stage = stages[0]
    current_env = build_parallel_env(
        scene_path=scene_path,
        stage_name=first_stage.name,
        seed=seed + 1000 * first_stage.index,
        num_envs=num_envs,
        start_method=str(args.start_method),
    )

    if args.resume_model is None:
        model = SAC(
            policy="MlpPolicy",
            env=current_env,
            learning_rate=3e-4,
            buffer_size=int(args.buffer_size),
            learning_starts=int(args.learning_starts),
            batch_size=int(args.batch_size),
            tau=0.005,
            gamma=0.98,
            train_freq=(1, "step"),
            gradient_steps=int(args.gradient_steps),
            ent_coef=parse_ent_coef(str(args.ent_coef)),
            target_entropy="auto",
            policy_kwargs={
                "net_arch": {
                    "pi": [256, 256, 256],
                    "qf": [512, 512, 256],
                }
            },
            tensorboard_log=str(out_dir / "tb"),
            verbose=1,
            seed=seed,
            device=str(args.device),
        )
        reset_num_timesteps_first = True
    else:
        model = SAC.load(
            args.resume_model.resolve(),
            env=current_env,
            device=str(args.device),
            tensorboard_log=str(out_dir / "tb"),
        )
        if args.resume_replay_buffer is not None:
            model.load_replay_buffer(args.resume_replay_buffer.resolve())
        reset_num_timesteps_first = False

    current_stage_name = first_stage.name
    completed_all_selected = True

    try:
        for loop_index, stage in enumerate(stages):
            current_stage_name = stage.name
            if loop_index > 0:
                next_env = build_parallel_env(
                    scene_path=scene_path,
                    stage_name=stage.name,
                    seed=seed + 1000 * stage.index,
                    num_envs=num_envs,
                    start_method=str(args.start_method),
                )
                current_env.close()
                current_env = next_env
                model.set_env(current_env)
                if not bool(args.keep_replay_across_stages) and model.replay_buffer is not None:
                    model.replay_buffer.reset()
                    print(f"[curriculum] {stage.name}: 已清空上一阶段 replay buffer。")

            stage_start_timesteps = int(model.num_timesteps)
            eval_env = build_evaluation_env(
                scene_path=scene_path,
                stage_name=stage.name,
                seed=seed + 10_000 * stage.index,
            )
            checkpoint_callback = CheckpointCallback(
                save_freq=max(1, int(args.save_every) // num_envs),
                save_path=str(out_dir / "checkpoints" / stage.name),
                name_prefix="sac_ur5e_visual_suction_v6",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
            curriculum_callback = SuccessRateCurriculumCallback(
                eval_env=eval_env,
                eval_freq_vec_steps=max(1, int(args.eval_every) // num_envs),
                n_eval_episodes=int(args.eval_episodes),
                eval_seed=seed + 100_000 * stage.index,
                stage_name=stage.name,
                stage_start_timesteps=stage_start_timesteps,
                min_stage_transitions=stage.min_steps,
                promotion_success_rate=stage.promotion_success_rate,
                required_consecutive_evals=stage.required_consecutive_evals,
                max_safety_violation_rate=stage.max_safety_violation_rate,
                best_save_dir=out_dir / "best_success" / stage.name,
                history_path=out_dir / "eval" / stage.name / "history.jsonl",
                verbose=1,
            )

            print(
                f"\n=== {stage.index}:{stage.name} | speed<={stage.max_cube_speed:.3f} m/s | "
                f"spawn_x={stage.cube_spawn_x} spawn_y={stage.cube_spawn_y} | "
                f"min={stage.min_steps:,} max={stage.max_steps:,} | "
                f"success>={stage.promotion_success_rate:.2f} safety<={stage.max_safety_violation_rate:.2f} | "
                f"passes={stage.required_consecutive_evals} ==="
            )

            try:
                model.learn(
                    total_timesteps=stage.max_steps,
                    reset_num_timesteps=(reset_num_timesteps_first if loop_index == 0 else False),
                    callback=[checkpoint_callback, curriculum_callback],
                    progress_bar=True,
                    tb_log_name=stage.name,
                )
            finally:
                eval_env.close()

            model.save(out_dir / stage.name)
            if bool(args.save_replay_buffer):
                model.save_replay_buffer(out_dir / f"{stage.name}_replay_buffer.pkl")

            if not curriculum_callback.promotion_ready:
                completed_all_selected = False
                note = (
                    f"{stage.name} 在最大 {stage.max_steps:,} transitions 内未达到晋级门槛，"
                    "已停止，不自动进入更高难度。\n"
                )
                (out_dir / "CURRICULUM_STOPPED.txt").write_text(note, encoding="utf-8")
                print("[curriculum]", note.strip())
                break

            print(f"[curriculum] {stage.name} 已通过晋级门槛。")

        model.save(out_dir / "final_model")
        if completed_all_selected:
            (out_dir / "CERTIFIED.txt").write_text(
                "所选 v6 课程阶段均通过 success/safety gate。\n", encoding="utf-8"
            )
        print("训练结束：", out_dir / "final_model.zip")

    except KeyboardInterrupt:
        model.save(out_dir / "interrupted_model")
        if bool(args.save_replay_on_interrupt):
            model.save_replay_buffer(out_dir / "interrupted_replay_buffer.pkl")
        (out_dir / "INTERRUPTED.txt").write_text(
            f"用户中断训练。stage={current_stage_name}, num_timesteps={model.num_timesteps}\n",
            encoding="utf-8",
        )
        print(
            f"\n已保存中断点：{out_dir / 'interrupted_model.zip'} "
            f"(stage={current_stage_name}, steps={model.num_timesteps:,})"
        )
    finally:
        current_env.close()


if __name__ == "__main__":
    main()
