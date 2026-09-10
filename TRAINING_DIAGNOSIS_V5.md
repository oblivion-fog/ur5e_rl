# v5 success=0 的核心诊断

## 现象

用户反馈 v5 Stage1 在物块几乎位于末端正下方的情况下，约 160k transitions 仍然：

```text
success=0
passes=0/2
```

同时 v5 的 `smoke_grasp.py --render` 在完全不使用 RL 时就失败：

```text
phase=pregrasp reached=False
RuntimeError: 无法到达 pregrasp
```

## 判断

这足以证明：至少在该 v5 运行版本里，**底层 Cartesian controller 本身就没有打通**。

SAC policy 输出的是 Cartesian delta；如果 deterministic 脚本使用同一 IK 都不能到达目标，则训练步数再多也不会自然产生 contact / held / success。

## 与最初成功版本的关键差异

最初版本：

- 纯 DLS Jacobian IK；
- 没有 posture regularization；
- 没有额外 software joint envelope；
- suction 是简单直接控制；
- dense reward 直接奖励 reach/contact/held/lift；
- 没有一碰就终止的复杂安全逻辑。

v4/v5 逐步增加了约束与 reward 结构，使学习问题变得更复杂。

## v6 处理方式

不是继续给 v5 打补丁，而是：

- 以最初成功版控制器为基线；
- 保留用户当前更合理的桌面/物块场景；
- 只保留软碰撞惩罚和持续碰撞终止；
- reward 改成清晰的 staged max；
- 训练前强制 scripted preflight。
