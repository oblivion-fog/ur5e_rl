# UR5e 强化学习物品分拣执行方案（v7_sorting 路线）

> 基于当前仓库 v6 Simple-Flow、已通过的 `stage1_static` 抓取模型，以及对成熟 Pick-and-Place / Sorting 任务结构的对照分析。
>
> 最终目标：从“单物体找到 → 吸附 → 抬升”升级为“多物品识别/选择 → 抓取 → 运输 → 放置到对应箱子 → 释放 → 继续下一件”的完整强化学习分拣流程。

## 1. 当前阶段与路线调整

当前 v6 的原课程为：

```text
stage1_static
→ stage2_slow
→ stage3_moving
```

其中 Stage2/3 通过 `_apply_cube_drive()` 主动物块移动，主要验证：

1. 策略是否真正使用视觉反馈，而不是记住固定开环轨迹；
2. 末端能否对目标位置变化进行闭环修正；
3. 为未来传送带/动态目标抓取保留技术基础。

这些目标本身合理，但它们不是“静态桌面多物品分拣”的最短主线。

因此建议：

- 冻结当前 v6 Stage1 作为 **Pick-Lift baseline**；
- 暂停把 `stage2_slow / stage3_moving` 作为主课程；
- 将移动目标保留为后期 `conveyor_sorting` 扩展；
- 新版本以 `v7_sorting` 为主线，先补齐 **Place / Release / Correct Bin / Multi-object Selection / Sequential Sorting**。

---

## 2. 最终系统应该是什么样

推荐系统分为五层：

```text
场景 / 任务层
多种物品 + 多个分类箱 + class→bin 映射
        ↓
感知层
全局相机：发现桌上所有物体/箱子
腕部相机：抓取前精细对准
        ↓
目标管理层
当前抓哪个？应该送到哪个箱？
        ↓
低层强化学习操作策略
Goal-conditioned SAC
[dx, dy, dz, suction]
        ↓
UR5e / MuJoCo
Cartesian action → DLS IK → 6 joint targets
suction action → adhesion actuator
```

### 为什么推荐双相机

当前 `eef_camera` 很适合抓取前最后几十厘米的视觉伺服，但 home 位姿下不适合承担全桌面多目标搜索。

推荐：

- `sorting_camera`：固定俯视/斜视全局相机，用于发现物品、分类、定位箱子；
- `eef_camera`：继续负责近距离精细对准。

如果真实平台最终只有腕部相机，也能实现，但需要加入“主动扫描 + 记忆”，会把问题升级为 POMDP，第一版不建议这样做。

---

## 3. 为什么不建议立即做端到端 RGB SAC

当前成功版本是 21 维低维 observation + MLP SAC。若直接将 640×480 RGB 输入 SAC，会同时引入：

- 视觉特征学习；
- 多物体检测；
- 类别识别；
- 抓取；
- 搬运；
- 放置；
- 长时程 credit assignment。

这会把已经稳定的控制问题重新变成极难的端到端任务。

推荐先定义统一感知接口：

```text
TargetObjectObservation
- visible
- u, v, depth
- relative xyz
- class_id
- confidence
- size/features

TargetBinObservation
- visible
- u, v, depth / relative xyz
- bin_id
```

MuJoCo 训练阶段可以先由仿真真值生成同样格式；之后再替换为 OpenCV / YOLO / segmentation / depth 模块。RL 层只消费统一接口，不关心数据来自仿真真值还是视觉模型。

---

# 4. 推荐课程设计

## Stage 0 — `baseline_pick_lift`

状态：**已完成**。

当前 v6 Stage1 已证明：

```text
找到单物块
→ ready
→ contact
→ held
→ lift
→ success
```

最终两次 deterministic eval 达到 100% success、0% safety。

作用：冻结可复现 baseline。后续版本必须保留这个回归测试，防止新增放置功能后破坏抓取能力。

---

## Stage 1 — `pick_place_fixed_bin`

这是下一步真正应该实现的阶段。

### 场景

```text
1 个可吸附物体
1 个固定目标箱
```

物体在小范围随机位置生成，目标箱固定在工作空间另一侧。

### 任务流程

```text
抓取
→ 抬升
→ 搬运到目标箱上方
→ 下降
→ 关闭 suction
→ 物体落入箱内
→ 稳定保持 N 步
→ SUCCESS
```

### 成功判定

不能只看 TCP 是否到达箱子上方。必须同时满足：

```text
object_center ∈ target_bin_inner_bounds
AND suction 已释放
AND held = False
AND object 速度低于阈值
AND 连续保持 N steps
```

### Reward

推荐 staged reward：

```text
reach target object        0.00 ~ 0.15
grasp / held               0.25
lift                       0.35 ~ 0.45
transport                  0.45 ~ 0.65
hover over target bin      0.65 ~ 0.75
inside correct bin         0.85
released + settled         1.00
```

原则：

- 普通 contact 不能等同 grasp；
- 未 held 时不能获得 transport reward；
- wrong-bin 必须明显低于正确 placement；
- 成功奖励必须始终是全流程最高层；
- 避免大量 dense 项无条件累加。

建议晋级门槛：

```text
deterministic success >= 0.90
wrong_bin <= 0.01
drop_rate <= 0.05
safety <= 0.05
连续 3 次 eval
```

---

## Stage 2 — `goal_conditioned_multibin`

### 场景

```text
1 个物体
2~3 个不同目标箱
```

每个 episode 随机指定 `desired_bin_id`。

### 目标

让 policy 理解：同一个物体并不总去固定位置，而是根据 goal 决定放置位置。

推荐 observation 逐步改成 goal-conditioned 形式：

```python
{
    "observation": [...robot, target_object, target_bin, suction/contact...],
    "achieved_goal": [...],
    "desired_goal": [...]
}
```

这一阶段可测试 `SAC + HerReplayBuffer`，提高 goal-conditioned placement 的失败轨迹利用率。

建议晋级：

```text
总体 success >= 0.90
每个 bin success >= 0.85
wrong_bin <= 0.02
```

---

## Stage 3 — `multi_object_target_pick`

### 场景

```text
2~3 个物体
1 个目标箱
每轮明确指定 target_object_id
```

### 目标

从“抓看到的唯一物体”升级为“多个候选中抓指定物体”。

当前仓库不具备这一能力，因为 XML sensor 和 observation 都硬编码唯一 `cube`。

推荐全局相机产生固定 K 个 object slots：

```text
slot_0: class, u, v, depth, visible
slot_1: class, u, v, depth, visible
slot_2: class, u, v, depth, visible
```

第一版由 Task Manager 选出 target，再把目标对象特征送给低层 SAC。不要一开始让同一个 MLP SAC 同时完成多目标注意力和连续控制。

错误抓取：

```text
held_object_id != target_object_id
→ wrong_pick = True
```

给予明显负奖励，但早期探索阶段不建议第一次 wrong pick 就立刻终止整个 episode。

---

## Stage 4 — `single_cycle_sort`

### 场景

```text
3 类物体
3 个对应箱
每个 episode 指定 1 个待分拣目标
```

例如：

```text
class_0 → bin_0
class_1 → bin_1
class_2 → bin_2
```

目标：

```text
识别/选择目标
→ 抓取正确物体
→ 找到正确箱
→ 搬运
→ 释放
```

类别到箱子的规则第一版建议由任务层提供，不要强迫低层 SAC 通过 trial-and-error 重新发现分类规则。

---

## Stage 5 — `sequential_sort`

### 场景

```text
3~5 个物体
2~3 个箱子
一个 episode 内全部处理
```

推荐第一版使用：

```text
Task Manager
选择 next unsorted object
查 class→bin 映射
生成 target_object / target_bin
        ↓
同一个 goal-conditioned SAC
完成 pick→place→release
        ↓
标记 sorted
继续下一件
```

最终的机械臂运动仍由 RL 完成，同时系统比“一张网络包办所有决策”更稳定、可解释。

如果研究目标要求“先抓哪件”也必须由 RL 决定，可在低层稳定之后增加：

```text
High-level PPO / DQN：选择 object slot
Low-level SAC：连续 manipulation
```

不要把两个时间尺度硬塞进一个 flat SAC。

---

## Stage 6 — `robust_sorting`

加入 domain randomization：

- 物体质量；
- 摩擦系数；
- 尺寸；
- 朝向；
- 相机轻微位姿；
- 光照；
- 颜色；
- 遮挡；
- suction gain；
- 传感器噪声；
- 初始物体间距。

目的：减少策略只适用于一个完美仿真状态的过拟合。

---

## Stage 7（可选）— `conveyor_sorting`

只有静态 sorting 稳定后再启用传送带/移动物体。

当前 v6 已有：

```text
cube_velocity
_apply_cube_drive()
lead_time
```

届时可以重新利用，形成真正的动态分拣扩展。

---

# 5. 推荐的软件结构改造

当前 `env.py` 同时负责 MuJoCo、IK、视觉、reward、suction、碰撞、移动物体、render，进入 sorting 后会过于臃肿。

建议 v7 拆为：

```text
ur5e_rl/
├── robot/
│   ├── controller.py       # DLS IK / joint control
│   └── suction.py          # suction/contact/held
├── perception/
│   ├── interfaces.py       # 感知结果类型
│   ├── sim_perception.py   # MuJoCo 真值→统一接口
│   └── vision_perception.py# OpenCV/检测/深度
├── tasks/
│   ├── sorting_task.py     # object→bin、任务状态
│   ├── predicates.py       # grasped/in_bin/released/settled
│   └── rewards.py          # staged reward
├── envs/
│   ├── pick_lift_env.py    # v6 回归基准
│   └── sorting_env.py      # 新分拣环境
└── training/
    ├── curriculum.py
    └── callbacks.py
```

边界原则：

- Robot controller 不知道“某类物品该去哪个箱”；
- Perception 不计算 reward；
- Reward 不控制机器人；
- Task Manager 不实现 IK；
- Env 负责把这些组件串起来。

---

# 6. Scene / MJCF 改造

## 6.1 多物体

从单一：

```text
cube
```

升级为：

```text
object_0
object_1
object_2
...
```

每个物体至少有：

```text
freejoint
collision geom
visual geom
center site
grasp/top site
class metadata（Python registry）
```

第一批物品建议仍然选择吸盘友好的刚体：

- cube；
- 圆柱罐；
- 矮长方体盒。

不要一开始加入软袋、多孔体、复杂凹面物品。

## 6.2 Bin

每个 bin 至少包含：

```text
bottom
4 walls
bin_center site
bin_inner_bounds
```

`in_bin()` 应判断物体是否真正进入内腔，而不是只看与一个目标点的距离。

## 6.3 Camera

保留：

```text
eef_camera
```

新增：

```text
sorting_camera
```

---

# 7. Observation 正式设计

低层 SAC 推荐使用：

```text
6 joint positions
6 joint velocities

target object:
    u, v, depth
    du, dv, ddepth
    visible
    class one-hot / embedding
    estimated size

target bin:
    u, v, depth / relative xyz
    visible
    bin one-hot

contact force
suction ctrl
held flag
```

目标箱必须显式进入 observation，否则相同机器人状态面对不同目标箱时，policy 无法判断应该往哪个方向移动。

---

# 8. Reward 设计原则

定义物理阶段：

```text
P0 search/reach
P1 grasp
P2 lift
P3 transport
P4 hover
P5 lower
P6 release
P7 settled-in-correct-bin
```

推荐后阶段只在前阶段成立后激活：

```text
r_reach
if held:
    r_lift
if sufficiently_lifted:
    r_transport
if over_target_bin:
    r_lower
if inside_bin:
    r_release
if released_and_settled:
    success = 1
```

额外只保留小量：

```text
action penalty
collision penalty
wrong-pick penalty
wrong-bin penalty
drop penalty
```

---

# 9. 算法组合建议

## 低层连续控制

继续使用 SAC，因为当前 4 维连续动作已经实际成功。

## Goal-conditioned placement

建议测试：

```text
SAC + HerReplayBuffer
```

特别适用于“把物体送到指定空间目标”的 Stage2。

## 多物体连续任务

优先：

```text
Task Manager + goal-conditioned SAC
```

低层稳定以后，再评估：

```text
High-level PPO/DQN selector
+
Low-level SAC manipulation
```

---

# 10. 新增训练指标

分拣任务至少记录：

```text
pick_success_rate
wrong_pick_rate
held_rate
drop_rate
transport_success_rate
correct_bin_rate
wrong_bin_rate
release_success_rate
settled_success_rate
full_sort_success_rate
mean_cycles_per_episode
mean_steps_per_item
regrasp_count
collision_rate
```

最终最重要的不是“某一次抓取成功”，而是：

```text
全部 N 件物品正确入箱的 episode 比例
```

---

# 11. 测试策略

## A. 物理 preflight

脚本直接执行：

```text
pick→lift→transport→place→release
```

脚本都失败时，禁止开始 RL。

## B. 单技能回归

每次新版本必须重新验证旧任务：

```text
pick-lift >= 95%
```

## C. RL 独立评估

训练 callback 固定 seed 只用于晋级；最终报告必须使用未参与训练 callback 的新 seed，并分别测试 deterministic/stochastic。

---

# 12. 推荐实施里程碑

### Milestone A — 冻结 v6 baseline

保存当前 best model、eval history、Stage1 报告，并建议打 tag：

```text
v6-pick-lift-baseline
```

### Milestone B — v7 Stage1

```text
单物体 → 单固定箱
```

只新增 transport/release/in-bin success。

### Milestone C — Goal-conditioned Multi-bin

```text
单物体 → 2~3 个箱
```

### Milestone D — Multi-object Target Pick

```text
2~3 个候选物体 → 抓指定目标
```

### Milestone E — Single-cycle Sorting

```text
多类物体 × 多箱，一次完成一个目标
```

### Milestone F — Sequential Sorting

```text
一个 episode 内全部分拣完成
```

### Milestone G — Robust / Dynamic

最后再加入 domain randomization、遮挡、视觉噪声、移动目标和传送带。

---

# 13. 结论

当前项目已经完成最难的基础突破之一：**单物体视觉抓取与真空抬升已经能通过 SAC 稳定学习。**

下一步最合理的主线不是继续强化“追移动 cube”，而是：

```text
Pick
↓
Pick-and-Place
↓
Goal-conditioned Place
↓
Multi-object Target Pick
↓
Single-cycle Sort
↓
Sequential Sorting
↓
Robust / Conveyor Sorting
```

这样每次只增加一个关键能力，失败时能明确定位问题发生在抓取、搬运、放置、目标条件、分类还是长时程控制，避免再次出现早期版本“同时改太多变量、训练失败后很难判断原因”的情况。
