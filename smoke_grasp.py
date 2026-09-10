from __future__ import annotations

import argparse
from pathlib import Path

from ur5e_rl.env import make_env
from ur5e_rl.preflight import run_scripted_grasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v6 整流程物理自检")
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene.xml"))
    parser.add_argument("--stage", type=str, default="stage1_static")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(
        scene_path=args.scene.resolve(),
        stage_name=str(args.stage),
        render_mode="human" if bool(args.render) else None,
        seed=123,
        viewer_camera="overview_camera",
        training=False,
    )
    try:
        result = run_scripted_grasp(env, render=bool(args.render))
        print("SMOKE_GRASP_SUCCESS=", result.success)
        print("phase=", result.phase)
        print("min_pregrasp_error=", f"{result.min_pregrasp_error:.6f}")
        print("contact_seen=", result.contact_seen)
        print("held_seen=", result.held_seen)
        print("lift_height=", f"{result.lift_height:.4f}")
        print("message=", result.message)
        if not result.success:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
