from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class EvalStats:
    step: int
    stage_transitions: int
    episodes: int
    success_rate: float
    safety_violation_rate: float
    robot_table_collision_rate: float
    robot_cube_collision_rate: float
    robot_floor_collision_rate: float
    pregrasp_rate: float
    ready_rate: float
    contact_rate: float
    held_rate: float
    mean_reward: float
    mean_ep_length: float
    mean_first_success_step: float


class SuccessRateCurriculumCallback(BaseCallback):
    """固定种子评估，优先按 success 保存最佳模型，safety 只做次级筛选。"""

    def __init__(
        self,
        eval_env: object,
        eval_freq_vec_steps: int,
        n_eval_episodes: int,
        eval_seed: int,
        stage_name: str,
        stage_start_timesteps: int,
        min_stage_transitions: int,
        promotion_success_rate: float,
        required_consecutive_evals: int,
        max_safety_violation_rate: float,
        best_save_dir: Path,
        history_path: Path,
        verbose: int,
    ) -> None:
        super().__init__(verbose=verbose)
        self._eval_env = eval_env
        self._eval_freq_vec_steps = int(eval_freq_vec_steps)
        self._n_eval_episodes = int(n_eval_episodes)
        self._eval_seed = int(eval_seed)
        self._stage_name = stage_name
        self._stage_start_timesteps = int(stage_start_timesteps)
        self._min_stage_transitions = int(min_stage_transitions)
        self._promotion_success_rate = float(promotion_success_rate)
        self._required_consecutive_evals = int(required_consecutive_evals)
        self._max_safety_violation_rate = float(max_safety_violation_rate)
        self._best_save_dir = best_save_dir
        self._history_path = history_path
        self._best_success = -1.0
        self._best_safety = 1.0
        self._best_reward = -np.inf
        self._consecutive_passes = 0
        self.promotion_ready = False

    def _evaluate(self) -> EvalStats:
        success = safety = table = cube = floor = 0
        pregrasp = ready = contact = held = 0
        rewards: list[float] = []
        lengths: list[int] = []
        first_success_steps: list[int] = []

        for episode in range(self._n_eval_episodes):
            obs, _ = self._eval_env.reset(seed=self._eval_seed + episode)
            episode_reward = 0.0
            episode_length = 0
            info: dict[str, object] = {}
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                if terminated or truncated:
                    break

            successful = bool(info.get("is_success", False))
            unsafe = bool(info.get("safety_violation", False))
            success += int(successful)
            safety += int(unsafe)
            table += int(bool(info.get("robot_table_collision", False)))
            cube += int(bool(info.get("robot_cube_collision", False)))
            floor += int(bool(info.get("robot_floor_collision", False)))
            pregrasp += int(bool(info.get("ever_pregrasp", False)))
            ready += int(bool(info.get("ever_ready", False)))
            contact += int(bool(info.get("ever_contact", False)))
            held += int(bool(info.get("ever_held", False)))
            first_step = int(info.get("first_success_step", 0) or 0)
            if successful and first_step > 0:
                first_success_steps.append(first_step)
            rewards.append(episode_reward)
            lengths.append(episode_length)

        n = float(self._n_eval_episodes)
        return EvalStats(
            step=int(self.model.num_timesteps),
            stage_transitions=int(self.model.num_timesteps) - self._stage_start_timesteps,
            episodes=self._n_eval_episodes,
            success_rate=success / n,
            safety_violation_rate=safety / n,
            robot_table_collision_rate=table / n,
            robot_cube_collision_rate=cube / n,
            robot_floor_collision_rate=floor / n,
            pregrasp_rate=pregrasp / n,
            ready_rate=ready / n,
            contact_rate=contact / n,
            held_rate=held / n,
            mean_reward=float(np.mean(rewards)),
            mean_ep_length=float(np.mean(lengths)),
            mean_first_success_step=(
                float(np.mean(first_success_steps)) if first_success_steps else float("nan")
            ),
        )

    def _record_tensorboard(self, stats: EvalStats) -> None:
        metrics = {
            "eval/success_rate": stats.success_rate,
            "eval/safety_violation_rate": stats.safety_violation_rate,
            "eval/robot_table_collision_rate": stats.robot_table_collision_rate,
            "eval/robot_cube_collision_rate": stats.robot_cube_collision_rate,
            "eval/robot_floor_collision_rate": stats.robot_floor_collision_rate,
            "eval/pregrasp_rate": stats.pregrasp_rate,
            "eval/suction_ready_rate": stats.ready_rate,
            "eval/contact_rate": stats.contact_rate,
            "eval/held_rate": stats.held_rate,
            "eval/mean_reward": stats.mean_reward,
            "eval/mean_ep_length": stats.mean_ep_length,
            "eval/mean_first_success_step": stats.mean_first_success_step,
            "curriculum/stage_transitions": float(stats.stage_transitions),
            "curriculum/consecutive_passes": float(self._consecutive_passes),
        }
        for key, value in metrics.items():
            self.logger.record(key, value)
        self.logger.dump(step=stats.step)

    def _append_history(self, stats: EvalStats) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(stats), ensure_ascii=False) + "\n")

    def _maybe_save_best(self, stats: EvalStats) -> None:
        # v6：success 永远是第一优先级。相同 success 时，先选更安全，再看 mean_reward。
        better = bool(
            stats.success_rate > self._best_success + 1e-12
            or (
                abs(stats.success_rate - self._best_success) <= 1e-12
                and stats.safety_violation_rate < self._best_safety - 1e-12
            )
            or (
                abs(stats.success_rate - self._best_success) <= 1e-12
                and abs(stats.safety_violation_rate - self._best_safety) <= 1e-12
                and stats.mean_reward > self._best_reward
            )
        )
        if not better:
            return

        self._best_success = stats.success_rate
        self._best_safety = stats.safety_violation_rate
        self._best_reward = stats.mean_reward
        self._best_save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(self._best_save_dir / "best_success_model")
        (self._best_save_dir / "best_metrics.json").write_text(
            json.dumps(asdict(stats), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _on_step(self) -> bool:
        if self._eval_freq_vec_steps <= 0 or self.n_calls % self._eval_freq_vec_steps != 0:
            return True

        stats = self._evaluate()
        self._maybe_save_best(stats)

        eligible = stats.stage_transitions >= self._min_stage_transitions
        pass_eval = bool(
            eligible
            and stats.success_rate >= self._promotion_success_rate
            and stats.safety_violation_rate <= self._max_safety_violation_rate
        )
        self._consecutive_passes = self._consecutive_passes + 1 if pass_eval else 0
        self._record_tensorboard(stats)
        self._append_history(stats)

        if self.verbose > 0:
            print(
                f"[eval:{self._stage_name}] step={stats.step:,} "
                f"stage_steps={stats.stage_transitions:,} "
                f"success={stats.success_rate:.3f} safety={stats.safety_violation_rate:.3f} "
                f"table={stats.robot_table_collision_rate:.3f} cube={stats.robot_cube_collision_rate:.3f} "
                f"pre={stats.pregrasp_rate:.3f} ready={stats.ready_rate:.3f} "
                f"contact={stats.contact_rate:.3f} held={stats.held_rate:.3f} "
                f"reward={stats.mean_reward:.2f} len={stats.mean_ep_length:.1f} "
                f"first_success={stats.mean_first_success_step:.1f} "
                f"passes={self._consecutive_passes}/{self._required_consecutive_evals}"
            )

        if self._consecutive_passes >= self._required_consecutive_evals:
            self.promotion_ready = True
            if self.verbose > 0:
                print(f"[curriculum] {self._stage_name} 已达到晋级条件。")
            return False
        return True
