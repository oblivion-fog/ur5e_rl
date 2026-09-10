# UR5e Visual Tracking + Vacuum Suction RL v6 Simple-Flow

这是针对前几个版本训练问题重新收敛后的正式版本。

核心目标只有一个：让 UR5e 学会完整的

```text
末端相机看到物块
→ 追踪 / 接近
→ suction pad 接触
→ 真空吸附
→ held
→ 抬升
→ success
```

而不是为了“安全”不断向策略添加越来越多的硬规则。

## 为什么需要 v6

最初版本虽然存在机械臂撞桌、推动物块等 reward hacking，但 Stage1/Stage2 已经证明了当前 MuJoCo suction + SAC 结构**能够产生成功抓取**。

之后的 v4/v5 为了修复作弊同时增加了很多约束。最关键的是 v5 的 IK 加入了 posture regularization，随后 `smoke_grasp.py` 在完全不使用 RL 的情况下都出现：

```text
phase=pregrasp reached=False
```

因此 v5 长时间 `success=0` 的首要解释不是“训练不够”，而是底层 action -> IK -> contact 链本身就被控制器限制住了。

v6 以最初已成功版本为控制基线，只保留必要修复。

## v6 与 v5 的主要区别

### 1. 恢复纯 DLS IK

没有：

- posture regularization
- null-space home 强制回正
- 额外 software joint envelope
- cube 真值下降 shield

只保留：

- 末端工作区裁剪
- UR5e actuator 自身 ctrlrange
- 末端姿态弱保持

### 2. 连续真空控制

SAC 第 4 维动作：

```python
suction_ctrl = clip(action[3], 0, 1)
```

因此：

- action <= 0：关闭；
- action > 0：连续真空强度；
- 不再在 `0.25` 阈值处突然 OFF/ON。

MuJoCo `adhesion` 仍然依赖 suction tool 与物体的 contact；pad 的 `gap=0.003` 提供几毫米接触候选区域。

### 3. staged reward

单步 reward 不是几十个项相加，而是取当前已经达到的最高阶段：

```text
reach/align          0 ~ 0.45
contact              0.55
held + lift          0.70 ~ 0.90
success              1.00
```

只靠靠近/悬停不能刷到比抓取更高的阶段 reward。

### 4. success 不立即结束 episode

horizon 固定 320。

一旦成功：

```text
ever_success = True
```

之后剩余时间保持 success stage reward=1.0。

这样越早成功总回报越高，不会再出现“失败 episode 活得久反而 reward 更高”的问题。

### 5. 碰撞改成软约束

瞬时 robot-table / robot-cube contact 不立即终止。

- table：扣分；连续 8 个 control step 撞桌才失败；
- floor：连续 3 个 control step 才失败；
- robot body 碰 cube：轻扣分，但不立即终止；
- suction pad 正常接触 cube 完全允许。

这保留了探索空间，同时抑制最初版本长时间用 shoulder 顶桌子的策略。

## 环境结构

```text
12 x MuJoCo CPU subprocesses
             ↓
        observations
             ↓
     SAC replay buffer
             ↓
      PyTorch CUDA GPU
             ↓
          actions
```

动作：

```text
[dx, dy, dz, suction]
```

观测不包含 cube 世界真值，只包含：

- 6 关节位置；
- 6 关节速度；
- eef camera 的 u/v/depth 与视觉速度；
- visible；
- suction touch；
- suction ctrl。

## 训练阶段

### Stage1 — stage1_static

静止物块，位置只做中等范围随机化：

```text
x ∈ [-0.06, 0.06]
y ∈ [ 0.54, 0.64]
```

目标：先证明整套流程稳定出现。

### Stage2 — stage2_slow

```text
cube speed <= 0.03 m/s
```

扩大位置范围。

### Stage3 — stage3_moving

```text
cube speed <= 0.06 m/s
```

最终移动目标。

## 安装

```bash
pip install -r requirements.txt
```

UR5e OBJ 资源仍放在：

```text
assets/
```

如果你的 assets 已在旧工程，直接复制即可。

## 训练前必须做的检查

### 1. 系统检查

```bash
python check_system.py
```

必须最后出现：

```text
CHECK_SYSTEM_OK
```

它不仅加载 XML，还会执行完整 scripted grasp preflight。

### 2. 可视化物理自检

```bash
python smoke_grasp.py --render
```

必须出现：

```text
SMOKE_GRASP_SUCCESS= True
```

如果这里失败，不要开始 SAC。

## 第一次训练建议

先只跑 Stage1：

```bash
python train.py \
  --max-stage 1 \
  --num-envs 12 \
  --device cuda \
  --out runs/ur5e_visual_suction_v6_stage1 \
  2>&1 | tee train_v6.log
```

v6 默认：

```text
learning_starts = 20,000
batch_size      = 256
ent_coef        = auto_0.2
gamma           = 0.98
```

不要加载 v4/v5 模型或 replay buffer，因为 IK、suction action 和 reward 语义都改变了。

## 重点看什么

不要首先看 reward。

技能链：

```text
pregrasp
→ ready
→ contact
→ held
→ success
```

Stage1 正常时应当先出现 contact，然后 held，再出现 success。

如果 80k~120k 后仍然：

```text
contact=0
held=0
success=0
```

停止训练并重新跑 `smoke_grasp.py`，不要再等到 300k。

## 评估

```bash
python evaluate.py \
  --model runs/ur5e_visual_suction_v6_stage1/best_success/stage1_static/best_success_model.zip \
  --stage stage1_static \
  --episodes 100 \
  --mode both
```

同时比较 deterministic 和 stochastic。

## 回放

```bash
python play.py \
  --model runs/ur5e_visual_suction_v6_stage1/best_success/stage1_static/best_success_model.zip \
  --stage stage1_static
```

默认 MuJoCo overview + OpenCV eef camera。

## 参考设计

v6 的 reward / episode 设计参考了这些成熟机器人操作环境的共性做法：

- robosuite Lift / PickPlace：reach、grasp、lift 等 staged reward；
- robosuite：机器人操作环境默认固定 horizon，而不是一成功立刻缩短 episode；
- ManiSkill：推荐 normalized dense reward，并通过阶段条件激活后续 reward；
- MuJoCo adhesion actuator：吸附力通过目标 body 的 contact 注入，`gap` 可允许有限距离内的吸附候选；
- Stable-Baselines3 SAC：`auto_0.2` 使用自动 entropy 调整并指定初始 entropy coefficient，避免固定很小的熵过早限制探索。

完整修改原因见 `V6_CHANGES.md`。
