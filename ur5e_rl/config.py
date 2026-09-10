from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


ENV_VERSION = "v6_simple_flow_2026_09_08"


@dataclass(frozen=True)
class StageSpec:
    index: int
    name: str
    max_cube_speed: float
    cube_spawn_x: tuple[float, float]
    cube_spawn_y: tuple[float, float]
    cube_motion_x: tuple[float, float]
    cube_motion_y: tuple[float, float]
    pixel_noise_std: float
    depth_noise_std: float
    vision_dropout_probability: float
    min_steps: int
    max_steps: int
    promotion_success_rate: float
    max_safety_violation_rate: float
    required_consecutive_evals: int


# v6 回到“先证明整条流程能学通，再逐步加难度”的三阶段结构。
# Stage1 仍有少量位置随机化，避免只记住一个固定关节轨迹，但明显比原始全桌面简单。
STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        index=1,
        name="stage1_static",
        max_cube_speed=0.0,
        cube_spawn_x=(-0.06, 0.06),
        cube_spawn_y=(0.54, 0.64),
        cube_motion_x=(-0.24, 0.24),
        cube_motion_y=(0.46, 0.78),
        pixel_noise_std=0.0,
        depth_noise_std=0.0,
        vision_dropout_probability=0.0,
        min_steps=40_000,
        max_steps=300_000,
        promotion_success_rate=0.50,
        max_safety_violation_rate=0.30,
        required_consecutive_evals=2,
    ),
    StageSpec(
        index=2,
        name="stage2_slow",
        max_cube_speed=0.03,
        cube_spawn_x=(-0.15, 0.15),
        cube_spawn_y=(0.49, 0.75),
        cube_motion_x=(-0.25, 0.25),
        cube_motion_y=(0.46, 0.78),
        pixel_noise_std=0.002,
        depth_noise_std=0.001,
        vision_dropout_probability=0.002,
        min_steps=100_000,
        max_steps=600_000,
        promotion_success_rate=0.55,
        max_safety_violation_rate=0.30,
        required_consecutive_evals=2,
    ),
    StageSpec(
        index=3,
        name="stage3_moving",
        max_cube_speed=0.06,
        cube_spawn_x=(-0.16, 0.16),
        cube_spawn_y=(0.49, 0.75),
        cube_motion_x=(-0.25, 0.25),
        cube_motion_y=(0.46, 0.78),
        pixel_noise_std=0.004,
        depth_noise_std=0.002,
        vision_dropout_probability=0.005,
        min_steps=160_000,
        max_steps=900_000,
        promotion_success_rate=0.50,
        max_safety_violation_rate=0.35,
        required_consecutive_evals=2,
    ),
)


@dataclass(frozen=True)
class EnvConfig:
    scene_path: Path
    stage_name: str
    horizon: int
    control_hz: int
    max_cube_speed: float

    # Cartesian DLS controller。v6 不再使用 posture regularization / 额外软件关节包络。
    max_cartesian_step: float
    ik_damping: float
    max_joint_step: float
    orientation_weight: float
    orientation_gain: float

    # 移动物块驱动。
    cube_drive_gain: float
    cube_drive_force_limit: float
    lead_time: float

    # 抓取 / 成功条件。
    success_height: float
    success_hold_steps: int
    contact_threshold: float
    suction_hold_threshold: float

    # 轻量 reward / 安全代价。
    action_penalty: float
    suction_far_penalty: float
    robot_table_contact_penalty: float
    robot_cube_contact_penalty: float
    robot_floor_contact_penalty: float
    persistent_table_collision_steps: int
    persistent_floor_collision_steps: int

    # 仅保留物理工作区边界，不依赖 cube 真值做动作 shield。
    minimum_suction_clearance_above_table: float
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    workspace_z_max: float

    cube_spawn_x: tuple[float, float]
    cube_spawn_y: tuple[float, float]
    cube_motion_x: tuple[float, float]
    cube_motion_y: tuple[float, float]

    camera_width: int
    camera_height: int
    camera_depth_scale: float
    pixel_noise_std: float
    depth_noise_std: float
    vision_dropout_probability: float
    visual_velocity_clip: float


def get_stage(stage_name: str) -> StageSpec:
    for stage in STAGES:
        if stage.name == stage_name:
            return stage
    valid = ", ".join(stage.name for stage in STAGES)
    raise ValueError(f"未知 stage_name={stage_name!r}，可选：{valid}")


def default_env_config(scene_path: Path, stage_name: str) -> EnvConfig:
    stage = get_stage(stage_name)
    return EnvConfig(
        scene_path=scene_path,
        stage_name=stage.name,
        horizon=320,
        control_hz=20,
        max_cube_speed=stage.max_cube_speed,
        max_cartesian_step=0.018,
        ik_damping=0.055,
        max_joint_step=0.10,
        orientation_weight=0.40,
        orientation_gain=0.30,
        cube_drive_gain=12.0,
        cube_drive_force_limit=1.20,
        lead_time=0.15,
        success_height=0.080,
        success_hold_steps=4,
        contact_threshold=0.010,
        suction_hold_threshold=0.10,
        action_penalty=0.002,
        suction_far_penalty=0.010,
        robot_table_contact_penalty=0.15,
        robot_cube_contact_penalty=0.05,
        robot_floor_contact_penalty=0.50,
        persistent_table_collision_steps=8,
        persistent_floor_collision_steps=3,
        minimum_suction_clearance_above_table=0.015,
        workspace_x=(-0.38, 0.38),
        workspace_y=(0.40, 0.84),
        workspace_z_max=0.88,
        cube_spawn_x=stage.cube_spawn_x,
        cube_spawn_y=stage.cube_spawn_y,
        cube_motion_x=stage.cube_motion_x,
        cube_motion_y=stage.cube_motion_y,
        camera_width=320,
        camera_height=240,
        camera_depth_scale=0.80,
        pixel_noise_std=stage.pixel_noise_std,
        depth_noise_std=stage.depth_noise_std,
        vision_dropout_probability=stage.vision_dropout_probability,
        visual_velocity_clip=2.0,
    )


def evaluation_config(config: EnvConfig) -> EnvConfig:
    return replace(
        config,
        pixel_noise_std=0.0,
        depth_noise_std=0.0,
        vision_dropout_probability=0.0,
    )
