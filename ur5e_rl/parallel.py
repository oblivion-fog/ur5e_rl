from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from .env import make_env


MONITOR_INFO_KEYS: tuple[str, ...] = (
    "is_success",
    "first_success_step",
    "held",
    "visible",
    "cube_speed",
    "lift_height",
    "pixel_error",
    "pregrasp",
    "suction_ready",
    "contact",
    "suction_ctrl",
    "safety_violation",
    "robot_table_collision",
    "robot_cube_collision",
    "robot_floor_collision",
    "ever_pregrasp",
    "ever_ready",
    "ever_contact",
    "ever_held",
)


def _worker_factory(
    scene_path: Path,
    stage_name: str,
    seed: int,
    rank: int,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return make_env(
            scene_path=scene_path,
            stage_name=stage_name,
            render_mode=None,
            seed=seed + rank,
            viewer_camera="overview_camera",
            training=True,
        )

    return _init


def build_parallel_env(
    scene_path: Path,
    stage_name: str,
    seed: int,
    num_envs: int,
    start_method: str,
) -> VecEnv:
    env_fns = [
        _worker_factory(scene_path, stage_name, seed, rank)
        for rank in range(num_envs)
    ]
    vector_env = SubprocVecEnv(env_fns, start_method=start_method)
    return VecMonitor(vector_env, info_keywords=MONITOR_INFO_KEYS)


def build_evaluation_env(
    scene_path: Path,
    stage_name: str,
    seed: int,
) -> Monitor:
    env = make_env(
        scene_path=scene_path,
        stage_name=stage_name,
        render_mode=None,
        seed=seed,
        viewer_camera="overview_camera",
        training=False,
    )
    return Monitor(env, info_keywords=MONITOR_INFO_KEYS)
