from __future__ import annotations

import argparse
from pathlib import Path

from ur5e_rl.config import ENV_VERSION, STAGES
from ur5e_rl.env import make_env
from ur5e_rl.preflight import run_scripted_grasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 v6 场景、观测与整流程物理链")
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene.xml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = args.scene.resolve()
    print("ENV_VERSION:", ENV_VERSION)

    for stage in STAGES:
        env = make_env(
            scene_path=scene,
            stage_name=stage.name,
            render_mode=None,
            seed=42 + stage.index,
            viewer_camera="overview_camera",
            training=False,
        )
        try:
            obs, info = env.reset(seed=42 + stage.index)
            print(
                f"[{stage.name}] obs={obs.shape} table_top={info['table_top_z']:.3f} "
                f"workspace_min_z={info['workspace_min_z']:.3f} "
                f"initial_contact={env.safety_status()}"
            )
        finally:
            env.close()

    env = make_env(
        scene_path=scene,
        stage_name="stage1_static",
        render_mode=None,
        seed=123,
        viewer_camera="overview_camera",
        training=False,
    )
    try:
        result = run_scripted_grasp(env, render=False)
    finally:
        env.close()

    print("preflight:", result)
    if not result.success:
        raise SystemExit(
            "v6 preflight 未通过：请先修复物理/IK 链，不要开始 SAC 训练。"
        )
    print("CHECK_SYSTEM_OK")


if __name__ == "__main__":
    main()
