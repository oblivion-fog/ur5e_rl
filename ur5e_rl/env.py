from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import EnvConfig, default_env_config, evaluation_config


ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

ROBOT_BODY_NAMES: tuple[str, ...] = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "suction_tool",
)


@dataclass(frozen=True)
class SimulationSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    time: float


@dataclass(frozen=True)
class VisionState:
    u: float
    v: float
    depth: float
    visible: bool


@dataclass(frozen=True)
class ContactStatus:
    robot_table: bool
    robot_cube: bool
    robot_floor: bool

    @property
    def any(self) -> bool:
        return self.robot_table or self.robot_cube or self.robot_floor


class UR5eVisualSuctionEnv(gym.Env[np.ndarray, np.ndarray]):
    """UR5e 末端视觉追踪 + 真空吸附抓取环境（v6 simple-flow）。

    设计原则：
    1. 保留第 1 版已经证明可学习的 4 维动作 [dx, dy, dz, suction]；
    2. Cartesian 控制恢复为纯 DLS Jacobian IK，不再加入 posture regularization；
    3. suction 使用连续控制，避免 SAC 动作经过硬阈值后完全不连续；
    4. reward 借鉴 robosuite / ManiSkill 的 staged reward：reach -> contact -> held -> lift -> success；
    5. 不用 cube 真值做动作 shield，不设置额外软件关节包络；
    6. 碰桌 / 碰地首先是软惩罚，只有持续碰撞才提前失败，避免随机探索一碰就结束；
    7. 成功后不立刻结束 episode，而是在固定 horizon 内持续给最大成功奖励，避免“成功越早总回报反而越低”。

    Policy observation 仍只包含可部署信息：关节状态、末端相机低维特征、触觉和 suction 控制量。
    cube 世界真值只用于 reward、课程和诊断，不进入 observation。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: EnvConfig,
        render_mode: str | None,
        seed: int,
        viewer_camera: str,
    ) -> None:
        super().__init__()
        self._cfg = config
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self._viewer_camera = viewer_camera

        self.model = mujoco.MjModel.from_xml_path(str(config.scene_path))
        self.data = mujoco.MjData(self.model)

        self._frame_skip = max(
            1,
            int(round(1.0 / (float(config.control_hz) * float(self.model.opt.timestep)))),
        )
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = config.control_hz

        self._joint_ids = np.asarray(
            [self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._arm_qpos_adr = np.asarray(
            [int(self.model.jnt_qposadr[joint_id]) for joint_id in self._joint_ids],
            dtype=np.int32,
        )
        self._arm_dof_adr = np.asarray(
            [int(self.model.jnt_dofadr[joint_id]) for joint_id in self._joint_ids],
            dtype=np.int32,
        )
        self._arm_actuator_ids = np.asarray(
            [self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_ACTUATOR_NAMES],
            dtype=np.int32,
        )

        self._suction_actuator_id = self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, "suction")
        self._suction_site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, "suction_site")
        self._cube_top_site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, "cube_top")
        self._cube_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._cube_dof_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        self._home_key_id = self._name2id(mujoco.mjtObj.mjOBJ_KEY, "home")

        self._table_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._floor_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._cube_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self._suction_pad_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "suction_pad")
        self._robot_body_ids = {
            self._name2id(mujoco.mjtObj.mjOBJ_BODY, name) for name in ROBOT_BODY_NAMES
        }

        self._table_top_z = float(
            self.model.geom_pos[self._table_geom_id, 2]
            + self.model.geom_size[self._table_geom_id, 2]
        )
        self._cube_half_height = float(self.model.geom_size[self._cube_geom_id, 2])
        self._cube_rest_center_z = self._table_top_z + self._cube_half_height + 0.001

        self._sensor_slices = {
            "suction_contact": self._sensor_slice("suction_contact"),
            "cube_pixel": self._sensor_slice("cube_pixel"),
            "cube_in_camera": self._sensor_slice("cube_in_camera"),
            "cube_top_in_suction": self._sensor_slice("cube_top_in_suction"),
            "cube_velocity": self._sensor_slice("cube_velocity"),
        }

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._desired_eef_rotation = np.eye(3, dtype=np.float64)
        self._cube_target_velocity = np.zeros(3, dtype=np.float64)
        self._episode_steps = 0
        self._success_hold = 0
        self._first_success_step = 0
        self._ever_success = False
        self._suction_ctrl = 0.0
        self._viewer: Any | None = None
        self._renderer: Any | None = None

        self._previous_vision_state = VisionState(0.0, 0.0, 0.0, False)
        self._visual_features = np.zeros(7, dtype=np.float64)

        self._ever_pregrasp = False
        self._ever_ready = False
        self._ever_contact = False
        self._ever_held = False
        self._ever_table_collision = False
        self._ever_cube_collision = False
        self._ever_floor_collision = False
        self._table_collision_streak = 0
        self._floor_collision_streak = 0

        self._reset_simulation(randomize_cube=False)
        initial_observation = self._observation()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_observation.shape,
            dtype=np.float32,
        )

    def _name2id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise ValueError(f"MJCF 中找不到 {object_type.name}: {name}")
        return object_id

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = self._name2id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        start = int(self.model.sensor_adr[sensor_id])
        dim = int(self.model.sensor_dim[sensor_id])
        return slice(start, start + dim)

    def _sensor(self, name: str) -> np.ndarray:
        return np.asarray(self.data.sensordata[self._sensor_slices[name]], dtype=np.float64).copy()

    def _sample_cube_target_velocity(self) -> np.ndarray:
        max_speed = float(self._cfg.max_cube_speed)
        if max_speed <= 0.0:
            return np.zeros(3, dtype=np.float64)
        speed = self._rng.uniform(0.25 * max_speed, max_speed)
        angle = self._rng.uniform(-np.pi, np.pi)
        return np.asarray([speed * np.cos(angle), speed * np.sin(angle), 0.0], dtype=np.float64)

    def _reset_episode_state(self) -> None:
        self._episode_steps = 0
        self._success_hold = 0
        self._first_success_step = 0
        self._ever_success = False
        self._suction_ctrl = 0.0
        self._ever_pregrasp = False
        self._ever_ready = False
        self._ever_contact = False
        self._ever_held = False
        self._ever_table_collision = False
        self._ever_cube_collision = False
        self._ever_floor_collision = False
        self._table_collision_streak = 0
        self._floor_collision_streak = 0

    def _reset_simulation(self, randomize_cube: bool) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        if randomize_cube:
            cube_x = self._rng.uniform(*self._cfg.cube_spawn_x)
            cube_y = self._rng.uniform(*self._cfg.cube_spawn_y)
            self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = np.asarray(
                [cube_x, cube_y, self._cube_rest_center_z, 1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )

        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.ctrl[self._suction_actuator_id] = 0.0
        for actuator_id, qpos_adr in zip(self._arm_actuator_ids, self._arm_qpos_adr, strict=True):
            self.data.ctrl[actuator_id] = self.data.qpos[qpos_adr]

        mujoco.mj_forward(self.model, self.data)
        self._desired_eef_rotation = np.asarray(
            self.data.site_xmat[self._suction_site_id], dtype=np.float64
        ).reshape(3, 3).copy()

        # 让自由方块稳定落在桌面。
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        self._cube_target_velocity = self._sample_cube_target_velocity()
        self._reset_episode_state()
        self.data.ctrl[self._suction_actuator_id] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        current_vision = self._raw_vision_state(add_noise=False)
        self._previous_vision_state = current_vision
        self._visual_features = self._encode_visual_features(current_vision, current_vision)

    @staticmethod
    def _orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.cross(current[:, 0], desired[:, 0])
            + np.cross(current[:, 1], desired[:, 1])
            + np.cross(current[:, 2], desired[:, 2])
        )

    def _suction_world_position(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self._suction_site_id], dtype=np.float64).copy()

    def _cube_top_world_position(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self._cube_top_site_id], dtype=np.float64).copy()

    def _set_cartesian_action(self, action_xyz: np.ndarray) -> None:
        """恢复第 1 版证明可工作的纯 DLS IK。

        没有 posture regularization，没有额外软件关节范围。只裁剪末端工作区与 actuator 自身 ctrlrange。
        """
        current_position = self._suction_world_position()
        requested_position = current_position + self._cfg.max_cartesian_step * action_xyz
        workspace_low = np.asarray(
            [
                self._cfg.workspace_x[0],
                self._cfg.workspace_y[0],
                self._table_top_z + self._cfg.minimum_suction_clearance_above_table,
            ],
            dtype=np.float64,
        )
        workspace_high = np.asarray(
            [self._cfg.workspace_x[1], self._cfg.workspace_y[1], self._cfg.workspace_z_max],
            dtype=np.float64,
        )
        clipped_position = np.clip(requested_position, workspace_low, workspace_high)
        position_error = clipped_position - current_position

        current_rotation = np.asarray(
            self.data.site_xmat[self._suction_site_id], dtype=np.float64
        ).reshape(3, 3)
        rotation_error = self._orientation_error(current_rotation, self._desired_eef_rotation)

        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            self._suction_site_id,
        )
        arm_jacobian_position = jacobian_position[:, self._arm_dof_adr]
        arm_jacobian_rotation = jacobian_rotation[:, self._arm_dof_adr]

        orientation_weight = float(self._cfg.orientation_weight)
        jacobian = np.vstack(
            [arm_jacobian_position, orientation_weight * arm_jacobian_rotation]
        )
        task_error = np.concatenate(
            [position_error, orientation_weight * self._cfg.orientation_gain * rotation_error]
        )

        damping = float(self._cfg.ik_damping)
        system = jacobian @ jacobian.T + (damping * damping) * np.eye(6, dtype=np.float64)
        joint_delta = jacobian.T @ np.linalg.solve(system, task_error)
        joint_delta = np.clip(
            joint_delta,
            -self._cfg.max_joint_step,
            self._cfg.max_joint_step,
        )

        current_joint_position = self.data.qpos[self._arm_qpos_adr].copy()
        target_joint_position = current_joint_position + joint_delta
        for index, actuator_id in enumerate(self._arm_actuator_ids):
            ctrl_low = float(self.model.actuator_ctrlrange[actuator_id, 0])
            ctrl_high = float(self.model.actuator_ctrlrange[actuator_id, 1])
            self.data.ctrl[actuator_id] = np.clip(
                target_joint_position[index], ctrl_low, ctrl_high
            )

    def _set_suction_action(self, suction_action: float) -> None:
        # v6 使用连续 suction：负值=关闭，正值直接映射到 adhesion ctrl [0,1]。
        # 这样 SAC 的连续动作不会在 0.25 处突然跳变，随机探索时也更容易产生部分吸附。
        self._suction_ctrl = float(np.clip(0.5 * (suction_action + 1.0), 0.0, 1.0))
        self.data.ctrl[self._suction_actuator_id] = self._suction_ctrl

    def _contact_force(self) -> float:
        return float(self._sensor("suction_contact")[0])

    def _cube_top_relative(self) -> np.ndarray:
        return self._sensor("cube_top_in_suction")

    def _is_held(self) -> bool:
        return bool(
            self._suction_ctrl >= self._cfg.suction_hold_threshold
            and self._contact_force() >= self._cfg.contact_threshold
            and float(np.linalg.norm(self._cube_top_relative())) < 0.090
        )

    def _pregrasp_ready(self) -> bool:
        rel = self._cube_top_relative()
        planar = float(np.linalg.norm(rel[:2]))
        z_gap = float(abs(rel[2]))
        return bool(planar < 0.10 and z_gap < 0.20)

    def _suction_ready(self) -> bool:
        rel = self._cube_top_relative()
        planar = float(np.linalg.norm(rel[:2]))
        z_gap = float(abs(rel[2]))
        return bool(planar < 0.055 and z_gap < 0.060)

    def _apply_cube_drive(self) -> None:
        self.data.xfrc_applied[self._cube_body_id, :] = 0.0
        if self._cfg.max_cube_speed <= 0.0 or self._is_held():
            return

        cube_position = np.asarray(self.data.xpos[self._cube_body_id], dtype=np.float64)
        x_min, x_max = self._cfg.cube_motion_x
        y_min, y_max = self._cfg.cube_motion_y

        if cube_position[0] <= x_min and self._cube_target_velocity[0] < 0.0:
            self._cube_target_velocity[0] *= -1.0
        elif cube_position[0] >= x_max and self._cube_target_velocity[0] > 0.0:
            self._cube_target_velocity[0] *= -1.0

        if cube_position[1] <= y_min and self._cube_target_velocity[1] < 0.0:
            self._cube_target_velocity[1] *= -1.0
        elif cube_position[1] >= y_max and self._cube_target_velocity[1] > 0.0:
            self._cube_target_velocity[1] *= -1.0

        cube_velocity = self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 3]
        desired_acceleration = self._cfg.cube_drive_gain * (
            self._cube_target_velocity - cube_velocity
        )
        cube_mass = float(self.model.body_mass[self._cube_body_id])
        force = cube_mass * desired_acceleration
        self.data.xfrc_applied[self._cube_body_id, 0:2] = np.clip(
            force[:2],
            -self._cfg.cube_drive_force_limit,
            self._cfg.cube_drive_force_limit,
        )

    def _raw_vision_state(self, add_noise: bool) -> VisionState:
        pixel = self._sensor("cube_pixel")
        cube_in_camera = self._sensor("cube_in_camera")
        width = float(self._cfg.camera_width)
        height = float(self._cfg.camera_height)

        u = float((pixel[0] - 0.5 * width) / (0.5 * width))
        v = float((pixel[1] - 0.5 * height) / (0.5 * height))
        depth = float(max(0.0, -cube_in_camera[2]) / self._cfg.camera_depth_scale)
        in_front = bool(cube_in_camera[2] < 0.0)
        in_image = bool(0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height)
        visible = in_front and in_image

        if add_noise:
            u += float(self._rng.normal(0.0, self._cfg.pixel_noise_std))
            v += float(self._rng.normal(0.0, self._cfg.pixel_noise_std))
            depth += float(self._rng.normal(0.0, self._cfg.depth_noise_std))
            if self._rng.random() < self._cfg.vision_dropout_probability:
                visible = False

        if not np.isfinite(u):
            u = 2.0
        if not np.isfinite(v):
            v = 2.0
        if not np.isfinite(depth):
            depth = 1.5

        return VisionState(
            u=float(np.clip(u, -2.0, 2.0)),
            v=float(np.clip(v, -2.0, 2.0)),
            depth=float(np.clip(depth, 0.0, 1.5)),
            visible=visible,
        )

    def _encode_visual_features(self, current: VisionState, previous: VisionState) -> np.ndarray:
        velocity_clip = self._cfg.visual_velocity_clip
        du = float(np.clip(current.u - previous.u, -velocity_clip, velocity_clip))
        dv = float(np.clip(current.v - previous.v, -velocity_clip, velocity_clip))
        ddepth = float(np.clip(current.depth - previous.depth, -velocity_clip, velocity_clip))
        if not current.visible:
            du = dv = ddepth = 0.0
        return np.asarray(
            [
                current.u,
                current.v,
                current.depth,
                du,
                dv,
                ddepth,
                1.0 if current.visible else 0.0,
            ],
            dtype=np.float64,
        )

    def _refresh_visual_features(self) -> None:
        current = self._raw_vision_state(add_noise=True)
        self._visual_features = self._encode_visual_features(current, self._previous_vision_state)
        self._previous_vision_state = current

    def _observation(self) -> np.ndarray:
        joint_position = np.clip(self.data.qpos[self._arm_qpos_adr] / np.pi, -2.0, 2.0)
        joint_velocity = np.tanh(self.data.qvel[self._arm_dof_adr] / 2.0)
        normalized_contact = np.tanh(self._contact_force() / 10.0)
        observation = np.concatenate(
            [
                joint_position,
                joint_velocity,
                self._visual_features,
                np.asarray([normalized_contact, self._suction_ctrl], dtype=np.float64),
            ]
        )
        return observation.astype(np.float32)

    def _contact_status(self) -> ContactStatus:
        robot_table = False
        robot_cube = False
        robot_floor = False

        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            if float(contact.dist) > 0.002:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            def other_geom(target: int) -> int | None:
                if geom1 == target:
                    return geom2
                if geom2 == target:
                    return geom1
                return None

            other = other_geom(self._table_geom_id)
            if other is not None:
                body = int(self.model.geom_bodyid[other])
                if body in self._robot_body_ids:
                    robot_table = True

            other = other_geom(self._floor_geom_id)
            if other is not None:
                body = int(self.model.geom_bodyid[other])
                if body in self._robot_body_ids:
                    robot_floor = True

            other = other_geom(self._cube_geom_id)
            if other is not None and other != self._suction_pad_geom_id:
                body = int(self.model.geom_bodyid[other])
                if body in self._robot_body_ids:
                    robot_cube = True

        return ContactStatus(robot_table, robot_cube, robot_floor)

    def _physics_step(self) -> ContactStatus:
        # 每个 RL control step 内做多次物理积分，并把短暂接触 OR 起来。
        remaining = int(self._frame_skip)
        table = cube = floor = False
        while remaining > 0:
            chunk = min(5, remaining)
            mujoco.mj_step(self.model, self.data, nstep=chunk)
            remaining -= chunk
            status = self._contact_status()
            table = table or status.robot_table
            cube = cube or status.robot_cube
            floor = floor or status.robot_floor
        return ContactStatus(table, cube, floor)

    def _reward_and_info(self, action: np.ndarray, contacts: ContactStatus) -> tuple[float, dict[str, Any]]:
        raw_vision = self._raw_vision_state(add_noise=False)
        pixel_error = float(np.linalg.norm([raw_vision.u, raw_vision.v]))

        suction_position = self._suction_world_position()
        cube_top_position = self._cube_top_world_position()
        cube_velocity = self._sensor("cube_velocity")
        intercept_position = cube_top_position + self._cfg.lead_time * cube_velocity

        planar_distance = float(np.linalg.norm(suction_position[:2] - intercept_position[:2]))
        distance = float(np.linalg.norm(self._cube_top_relative()))
        contact_force = self._contact_force()
        contact = bool(contact_force >= self._cfg.contact_threshold)
        held = self._is_held()
        pregrasp = self._pregrasp_ready()
        ready = self._suction_ready()

        cube_center_z = float(self.data.xpos[self._cube_body_id, 2])
        lift_height = max(0.0, cube_center_z - self._cube_rest_center_z)
        lift_fraction = float(np.clip(lift_height / max(self._cfg.success_height, 1e-6), 0.0, 1.0))
        success_now = bool(held and lift_fraction >= 1.0)
        fallen = bool(cube_center_z < self._table_top_z - 0.08)
        cube_speed = float(np.linalg.norm(cube_velocity[:2]))

        self._ever_pregrasp = self._ever_pregrasp or pregrasp
        self._ever_ready = self._ever_ready or ready
        self._ever_contact = self._ever_contact or contact
        self._ever_held = self._ever_held or held

        self._success_hold = self._success_hold + 1 if success_now else 0
        if self._success_hold >= self._cfg.success_hold_steps and not self._ever_success:
            self._ever_success = True
            self._first_success_step = self._episode_steps

        # 借鉴 robosuite 的 staged reward：越接近完整流程，reward 的“层级”越高。
        # 使用 max 而不是无脑相加，避免仅靠多个 dense 项叠出比抓取更高的回报。
        reach_distance = distance
        r_reach = 0.35 * (1.0 - np.tanh(7.0 * reach_distance))
        r_align = 0.10 * (1.0 - np.tanh(10.0 * planar_distance))
        reach_stage = float(r_reach + r_align)
        vacuum_active = (
            self._suction_ctrl >= 0.20
        )

        vacuum_contact = (
            contact
            and vacuum_active
        )

        contact_stage = (
            0.55
            if vacuum_contact
            else 0.0
        )
        held_stage = (0.70 + 0.20 * lift_fraction) if held else 0.0
        success_stage = 1.0 if self._ever_success else 0.0
        reward = max(reach_stage, contact_stage, held_stage, success_stage)

        reward -= self._cfg.action_penalty * float(np.square(action[:3]).sum())
        if self._suction_ctrl > 0.0 and distance > 0.15 and not held:
            reward -= self._cfg.suction_far_penalty * self._suction_ctrl

        if contacts.robot_table:
            reward -= self._cfg.robot_table_contact_penalty
        if contacts.robot_cube:
            reward -= self._cfg.robot_cube_contact_penalty
        if contacts.robot_floor:
            reward -= self._cfg.robot_floor_contact_penalty

        info: dict[str, Any] = {
            "visible": raw_vision.visible,
            "pixel_error": pixel_error,
            "planar_distance": planar_distance,
            "distance_suction_cube_top": distance,
            "pregrasp": pregrasp,
            "suction_ready": ready,
            "contact_force": contact_force,
            "contact": contact,
            "suction_requested": self._suction_ctrl > 0.0,
            "suction_on": self._suction_ctrl > 0.0,
            "suction_ctrl": self._suction_ctrl,
            "held": held,
            "lift_height": lift_height,
            "is_success": self._ever_success,
            "success_now": success_now,
            "first_success_step": self._first_success_step,
            "fallen": fallen,
            "cube_speed": cube_speed,
            "ever_pregrasp": self._ever_pregrasp,
            "ever_ready": self._ever_ready,
            "ever_contact": self._ever_contact,
            "ever_held": self._ever_held,
        }
        return float(reward), info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_simulation(randomize_cube=True)
        return self._observation(), {
            "stage_name": self._cfg.stage_name,
            "table_top_z": self._table_top_z,
            "workspace_min_z": self._table_top_z
            + self._cfg.minimum_suction_clearance_above_table,
            "cube_rest_center_z": self._cube_rest_center_z,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )

        self._set_cartesian_action(clipped_action[:3])
        self._set_suction_action(float(clipped_action[3]))
        self._apply_cube_drive()
        contacts = self._physics_step()
        self.data.xfrc_applied[self._cube_body_id, :] = 0.0

        self._episode_steps += 1
        self._refresh_visual_features()

        self._ever_table_collision = self._ever_table_collision or contacts.robot_table
        self._ever_cube_collision = self._ever_cube_collision or contacts.robot_cube
        self._ever_floor_collision = self._ever_floor_collision or contacts.robot_floor
        self._table_collision_streak = (
            self._table_collision_streak + 1 if contacts.robot_table else 0
        )
        self._floor_collision_streak = (
            self._floor_collision_streak + 1 if contacts.robot_floor else 0
        )

        reward, info = self._reward_and_info(clipped_action, contacts)

        hard_table_failure = (
            self._table_collision_streak >= self._cfg.persistent_table_collision_steps
        )
        hard_floor_failure = (
            self._floor_collision_streak >= self._cfg.persistent_floor_collision_steps
        )
        terminated = bool(hard_table_failure or hard_floor_failure or bool(info["fallen"]))
        truncated = self._episode_steps >= self._cfg.horizon

        # 固定 horizon 是 v6 的关键：成功不会提早缩短回报，越早成功就能在余下时间持续获得 1.0。
        safety_violation = bool(
            self._ever_table_collision or self._ever_cube_collision or self._ever_floor_collision
        )
        info["safety_violation"] = safety_violation
        info["robot_table_collision"] = self._ever_table_collision
        info["robot_cube_collision"] = self._ever_cube_collision
        info["robot_floor_collision"] = self._ever_floor_collision
        info["hard_table_failure"] = hard_table_failure
        info["hard_floor_failure"] = hard_floor_failure
        info["table_collision_streak"] = self._table_collision_streak
        info["success_hold"] = self._success_hold
        info["episode_steps"] = self._episode_steps

        # 兼容旧监控字段；v6 已经没有 Cartesian shield / software joint envelope。
        info["ever_shield"] = False
        info["ever_joint_limit"] = False
        info["shield_intervention_fraction"] = 0.0
        info["joint_limit_intervention_fraction"] = 0.0

        return self._observation(), float(reward), terminated, truncated, info

    def safety_status(self) -> ContactStatus:
        return self._contact_status()

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
            time=float(self.data.time),
        )

    def render_camera(self, camera_name: str, width: int, height: int) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera_name)
        return np.asarray(self._renderer.render()).copy()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                camera_id = int(
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_CAMERA,
                        self._viewer_camera,
                    )
                )
                if camera_id >= 0:
                    self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self._viewer.cam.fixedcamid = camera_id
            if self._viewer.is_running():
                self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            return self.render_camera(
                camera_name="eef_camera",
                width=self._cfg.camera_width,
                height=self._cfg.camera_height,
            )
        return None

    def viewer_is_running(self) -> bool:
        return self._viewer is None or bool(self._viewer.is_running())

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(
    scene_path: Path,
    stage_name: str,
    render_mode: str | None,
    seed: int,
    viewer_camera: str,
    training: bool,
) -> UR5eVisualSuctionEnv:
    config = default_env_config(scene_path=scene_path, stage_name=stage_name)
    if not training:
        config = evaluation_config(config)
    return UR5eVisualSuctionEnv(
        config=config,
        render_mode=render_mode,
        seed=seed,
        viewer_camera=viewer_camera,
    )
