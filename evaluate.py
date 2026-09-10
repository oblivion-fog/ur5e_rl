from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ur5e_rl.config import STAGES
from ur5e_rl.env import make_env


@dataclass(frozen=True)
class Result:
    mode: str
    episodes: int
    success_rate: float
    safety_rate: float
    table_rate: float
    cube_rate: float
    floor_rate: float
    pregrasp_rate: float
    ready_rate: float
    contact_rate: float
    held_rate: float
    mean_reward: float
    mean_length: float
    mean_first_success_step: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定种子评估 UR5e v6 策略")
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene.xml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=tuple(stage.name for stage in STAGES),
        required=True,
        help="必须和该模型训练时的 stage 匹配",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="both")
    return parser.parse_args()


def run_mode(args: argparse.Namespace, deterministic: bool) -> Result:
    env = make_env(
        scene_path=args.scene.resolve(),
        stage_name=str(args.stage),
        render_mode=None,
        seed=int(args.seed),
        viewer_camera="overview_camera",
        training=False,
    )
    model = SAC.load(args.model.resolve(), env=env, device=str(args.device))

    success = safety = table = cube = floor = 0
    pregrasp = ready = contact = held = 0
    rewards: list[float] = []
    lengths: list[int] = []
    success_steps: list[int] = []
    try:
        for episode in range(int(args.episodes)):
            obs, _ = env.reset(seed=int(args.seed) + episode)
            ep_reward = 0.0
            ep_length = 0
            info: dict[str, object] = {}
            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                if terminated or truncated:
                    break

            successful = bool(info.get("is_success", False))
            success += int(successful)
            safety += int(bool(info.get("safety_violation", False)))
            table += int(bool(info.get("robot_table_collision", False)))
            cube += int(bool(info.get("robot_cube_collision", False)))
            floor += int(bool(info.get("robot_floor_collision", False)))
            pregrasp += int(bool(info.get("ever_pregrasp", False)))
            ready += int(bool(info.get("ever_ready", False)))
            contact += int(bool(info.get("ever_contact", False)))
            held += int(bool(info.get("ever_held", False)))
            first_step = int(info.get("first_success_step", 0) or 0)
            if successful and first_step > 0:
                success_steps.append(first_step)
            rewards.append(ep_reward)
            lengths.append(ep_length)
    finally:
        env.close()

    n = float(args.episodes)
    return Result(
        mode="deterministic" if deterministic else "stochastic",
        episodes=int(args.episodes),
        success_rate=success / n,
        safety_rate=safety / n,
        table_rate=table / n,
        cube_rate=cube / n,
        floor_rate=floor / n,
        pregrasp_rate=pregrasp / n,
        ready_rate=ready / n,
        contact_rate=contact / n,
        held_rate=held / n,
        mean_reward=float(np.mean(rewards)),
        mean_length=float(np.mean(lengths)),
        mean_first_success_step=(float(np.mean(success_steps)) if success_steps else float("nan")),
    )


def print_result(result: Result) -> None:
    print(
        f"{result.mode:13s} episodes={result.episodes:4d} "
        f"success={result.success_rate:.3f} safety={result.safety_rate:.3f} "
        f"table={result.table_rate:.3f} cube={result.cube_rate:.3f} floor={result.floor_rate:.3f} "
        f"pre={result.pregrasp_rate:.3f} ready={result.ready_rate:.3f} "
        f"contact={result.contact_rate:.3f} held={result.held_rate:.3f} "
        f"reward={result.mean_reward:.2f} len={result.mean_length:.1f} "
        f"first_success={result.mean_first_success_step:.1f}"
    )


def main() -> None:
    args = parse_args()
    if args.mode in ("deterministic", "both"):
        print_result(run_mode(args, deterministic=True))
    if args.mode in ("stochastic", "both"):
        print_result(run_mode(args, deterministic=False))


if __name__ == "__main__":
    main()
