from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from stable_baselines3 import SAC

from ur5e_rl.camera_viewer import CameraPacket, OpenCVCameraProcess
from ur5e_rl.config import STAGES
from ur5e_rl.env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回放 UR5e v6 视觉追踪 + 真空吸附策略")
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene.xml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(stage.name for stage in STAGES), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--mujoco-camera", type=str, default="overview_camera")
    parser.add_argument("--opencv-camera", type=str, default="eef_camera")
    parser.add_argument("--opencv-width", type=int, default=640)
    parser.add_argument("--opencv-height", type=int, default=480)
    parser.add_argument("--no-opencv", action="store_true")
    parser.add_argument("--no-opencv-detection", action="store_true")
    return parser.parse_args()


def to_camera_packet(env: object) -> CameraPacket:
    snapshot = env.snapshot()
    return CameraPacket(snapshot.qpos, snapshot.qvel, snapshot.ctrl, snapshot.time)


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    scene_path = args.scene.resolve()
    env = make_env(
        scene_path=scene_path,
        stage_name=str(args.stage),
        render_mode="human",
        seed=int(args.seed),
        viewer_camera=str(args.mujoco_camera),
        training=False,
    )
    model = SAC.load(args.model.resolve(), env=env, device=str(args.device))

    camera_process: OpenCVCameraProcess | None = None
    if not bool(args.no_opencv):
        camera_process = OpenCVCameraProcess(
            scene_path=scene_path,
            camera_name=str(args.opencv_camera),
            width=int(args.opencv_width),
            height=int(args.opencv_height),
            window_name=f"UR5e camera: {args.opencv_camera}",
            detect_red=not bool(args.no_opencv_detection),
        )
        camera_process.start()

    deterministic = not bool(args.stochastic)
    control_hz = int(env.metadata["render_fps"])
    try:
        for episode in range(int(args.episodes)):
            obs, _ = env.reset(seed=int(args.seed) + episode)
            ep_reward = 0.0
            while True:
                wall_start = time.perf_counter()
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                env.render()
                if camera_process is not None:
                    camera_process.push(to_camera_packet(env))
                    if camera_process.should_stop():
                        return
                if not env.viewer_is_running():
                    return

                remaining = 1.0 / float(control_hz) - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)

                if terminated or truncated:
                    print(
                        f"episode={episode:02d} reward={ep_reward:.2f} "
                        f"success={info.get('is_success', False)} safety={info.get('safety_violation', False)} "
                        f"table={info.get('robot_table_collision', False)} cube={info.get('robot_cube_collision', False)} "
                        f"pre={info.get('ever_pregrasp', False)} ready={info.get('ever_ready', False)} "
                        f"contact={info.get('ever_contact', False)} held={info.get('ever_held', False)} "
                        f"suction={info.get('suction_ctrl', 0.0):.2f} "
                        f"first_success={info.get('first_success_step', 0)} "
                        f"lift={info.get('lift_height', 0.0):.3f}m"
                    )
                    break
    finally:
        if camera_process is not None:
            camera_process.close()
        env.close()


if __name__ == "__main__":
    main()
