from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import UR5eVisualSuctionEnv


@dataclass(frozen=True)
class PreflightResult:
    success: bool
    phase: str
    min_pregrasp_error: float
    contact_seen: bool
    held_seen: bool
    lift_height: float
    message: str


def _move_toward(
    env: UR5eVisualSuctionEnv,
    target: np.ndarray,
    suction_action: float,
    max_steps: int,
    tolerance: float,
    render: bool,
) -> tuple[bool, float, dict[str, object]]:
    min_error = float("inf")
    info: dict[str, object] = {}
    for _ in range(max_steps):
        current = env._suction_world_position()  # 仅用于诊断脚本，不进入 policy observation。
        error_vec = target - current
        error = float(np.linalg.norm(error_vec))
        min_error = min(min_error, error)
        if error <= tolerance:
            return True, min_error, info
        action_xyz = np.clip(
            error_vec / max(float(env._cfg.max_cartesian_step), 1e-9),
            -1.0,
            1.0,
        )
        action = np.asarray(
            [action_xyz[0], action_xyz[1], action_xyz[2], suction_action],
            dtype=np.float32,
        )
        _, _, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        if terminated or truncated:
            return False, min_error, info
    return False, min_error, info


def run_scripted_grasp(
    env: UR5eVisualSuctionEnv,
    render: bool,
) -> PreflightResult:
    """用与 RL 完全相同的 env.step + IK 验证整条物理链。

    这是诊断，不是训练数据，不会给 SAC 任何真值动作提示。
    """
    env.reset(seed=123)
    if render:
        env.render()

    cube_top = env._cube_top_world_position()
    pregrasp = cube_top + np.asarray([0.0, 0.0, 0.085], dtype=np.float64)
    reached, min_err, info = _move_toward(
        env,
        pregrasp,
        suction_action=-1.0,
        max_steps=180,
        tolerance=0.012,
        render=render,
    )
    if not reached:
        return PreflightResult(
            False,
            "pregrasp",
            min_err,
            bool(info.get("contact", False)),
            bool(info.get("held", False)),
            float(info.get("lift_height", 0.0)),
            "无法到达 pregrasp；优先检查 IK / home / workspace，而不是继续训练。",
        )

    # 打开真空并缓慢下降到方块顶面附近。
    contact_seen = False
    held_seen = False
    last_info: dict[str, object] = {}
    for z_offset in np.linspace(0.070, -0.004, 22):
        target = cube_top + np.asarray([0.0, 0.0, float(z_offset)], dtype=np.float64)
        _, _, last_info = _move_toward(
            env,
            target,
            suction_action=1.0,
            max_steps=18,
            tolerance=0.008,
            render=render,
        )
        contact_seen = contact_seen or bool(last_info.get("contact", False))
        held_seen = held_seen or bool(last_info.get("held", False))
        if held_seen:
            break

    if not held_seen:
        return PreflightResult(
            False,
            "contact_or_hold",
            min_err,
            contact_seen,
            held_seen,
            float(last_info.get("lift_height", 0.0)),
            "已到达方块上方，但未形成 held；检查 suction contact / adhesion / held 判据。",
        )

    current = env._suction_world_position()
    lift_target = current + np.asarray([0.0, 0.0, 0.13], dtype=np.float64)
    _, _, last_info = _move_toward(
        env,
        lift_target,
        suction_action=1.0,
        max_steps=140,
        tolerance=0.015,
        render=render,
    )

    # 再保持若干步，让 success_hold 有机会建立。
    for _ in range(12):
        _, _, terminated, truncated, last_info = env.step(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        if render:
            env.render()
        if terminated or truncated:
            break

    success = bool(last_info.get("is_success", False))
    return PreflightResult(
        success,
        "success" if success else "lift",
        min_err,
        contact_seen or bool(last_info.get("contact", False)),
        held_seen or bool(last_info.get("held", False)),
        float(last_info.get("lift_height", 0.0)),
        "整条 IK -> contact -> adhesion -> lift 链路通过。"
        if success
        else "已经 held 但没有达到 success_height；检查抬升 IK / adhesion 强度 / success 判据。",
    )
