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

## 当前训练状态（2026-09-10）

**Stage1 `stage1_static` 已通过 curriculum gate。**

最后两次 40-episode deterministic eval：

| Step | Success | Safety | Ready | Contact | Held | Mean reward | First success step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 239,904 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 278.57 | 117.7 |
| 259,896 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 286.89 | 84.95 |

Stage1 自动晋级条件为：

```text
success >= 0.50
safety  <= 0.30
连续通过 2 次 eval
```

训练器在 `259,896 / 300,000` transitions 时主动判定 Stage1 通过并结束训练。

训练过程中一个重要现象是：`contact/held` 很早就出现，但稳定 lift / success 到约 240k 才真正形成。因此，后续不能仅因为早期 success=0 就判断训练失败；应该结合 `ready → contact → held → success` 整条技能链判断。

完整 Stage1 分析、风险与 Stage2 计划见：

- [`STAGE1_EVALUATION.md`](STAGE1_EVALUATION.md)
- 训练诊断产物：[`diagnostic/stage1_static/`](diagnostic/stage1_static/)

当前推荐流程：

```text
Stage1 已完成
    ↓
独立 200-episode 泛化评估
    ↓
可视化 play 检查是否存在异常 shortcut
    ↓
继承 Stage1 best model 进入 Stage2
```

目前**不建议为了 v6.1 重新训练 Stage1**。本次真实训练已经证明 v6 policy 可以自行学出 `contact → held → lift → success`，保持同一 v6 动作和 reward 语义继续 curriculum 更有利于连续性。

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
- 不再在固定阈值处突然 OFF/ON。

MuJoCo `adhesion` 仍然依赖 suction tool 与物体的 contact；pad 的 `gap=0.003` 提供几毫米接触候选区域。

### 3. Staged reward

单步 reward 不再由大量 progress/event 项叠加，而是取当前已经达到的最高阶段：

```text
reach/align          0 ~ 0.45
contact              0.55
held + lift          0.70 ~ 0.90
success              1.00
```

只靠靠近/悬停不能刷到比完整抓取更高的阶段 reward。

### 4. Success 不立即结束 episode

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

### Stage1 — `stage1_static` ✅ 已通过

静止物块，位置做中等范围随机化：

```text
x ∈ [-0.06, 0.06]
y ∈ [ 0.54, 0.64]
```

最终内部 eval：

```text
success = 1.000
safety  = 0.000
ready   = 1.000
contact = 1.000
held    = 1.000
```

推荐模型：

```text
diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip
```

### Stage2 — `stage2_slow` ← 下一阶段

```text
cube speed <= 0.03 m/s
spawn_x ∈ [-0.15, 0.15]
spawn_y ∈ [ 0.49, 0.75]
```

同时加入少量 pixel/depth noise 与 vision dropout。

自动晋级 gate：

```text
success >= 0.55
safety  <= 0.30
连续通过 2 次 eval
```

### Stage3 — `stage3_moving`

```text
cube speed <= 0.06 m/s
```

最终移动目标阶段。

## 安装

```bash
pip install -r requirements.txt
```

UR5e OBJ 资源仍放在：

```text
assets/
```

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

## Stage1 独立泛化评估（推荐先执行）

Curriculum callback 使用固定的 40 个 eval seeds。进入 Stage2 前建议换一组 seed，独立评估 200 episodes：

```bash
python evaluate.py \
  --model diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip \
  --stage stage1_static \
  --episodes 200 \
  --seed 300000 \
  --mode both \
  --device cuda
```

建议额外工程 gate：

```text
deterministic success >= 0.90
deterministic safety  <= 0.05
stochastic success    >= 0.70
```

这些不是代码当前的自动门槛，而是进入 Stage2 前更严格的独立验证建议。

## 可视化回放 Stage1

```bash
python play.py \
  --model diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip \
  --stage stage1_static
```

重点检查是否还有 arm/wrist 推物块、长时间压桌、吸住后剧烈摆动等行为。

## 下一步：训练 Stage2

在独立评估和可视化回放没有暴露明显问题后，直接继承 Stage1 best model：

```bash
python train.py \
  --start-stage 2 \
  --max-stage 2 \
  --resume-model diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip \
  --num-envs 12 \
  --device cuda \
  --out runs/ur5e_visual_suction_v6_stage2 \
  2>&1 | tee train_v6_stage2.log
```

**不要加载 Stage1 replay buffer。** Stage2 的目标位置、运动速度和视觉噪声分布都发生变化，保留已经学好的网络权重即可。

v6 默认：

```text
learning_starts = 20,000
batch_size      = 256
ent_coef        = auto_0.2
gamma           = 0.98
```

## Stage2 重点看什么

不要首先看 reward。优先看：

```text
pregrasp
→ ready
→ contact
→ held
→ success
```

判断方式：

- `pre/ready` 高但 `contact` 低：移动目标截获或下降有问题；
- `contact` 高但 `held` 低：suction / 接触稳定性成为瓶颈；
- `held` 高但 `success` 低：主要是移动目标下的 lift 策略问题；
- success 上升同时 safety 上升：警惕推块或碰撞型 shortcut；
- reward 上升但 success 不升：重新检查 reward alignment。

Stage1 的历史说明：如果 `contact/held` 已经很高而 success 仍为 0，不要仅根据 success 过早停止；这可能意味着策略正在学习最后的 lift 阶段。反之，如果很长时间 `contact=0`、`held=0`、`success=0`，应优先检查控制或物理链路。

## 评估 Stage2

Stage2 通过后运行：

```bash
python evaluate.py \
  --model runs/ur5e_visual_suction_v6_stage2/best_success/stage2_slow/best_success_model.zip \
  --stage stage2_slow \
  --episodes 200 \
  --seed 400000 \
  --mode both \
  --device cuda
```

再根据独立评估决定是否进入 Stage3。

## 参考设计

v6 的 reward / episode 设计参考这些成熟机器人操作环境的共性做法：

- robosuite Lift / PickPlace：reach、grasp、lift 等 staged reward；
- robosuite：机器人操作环境默认固定 horizon，而不是一成功立刻缩短 episode；
- ManiSkill：推荐 normalized dense reward，并通过阶段条件激活后续 reward；
- MuJoCo adhesion actuator：吸附力通过目标 body 的 contact 注入，`gap` 可允许有限距离内的吸附候选；
- Stable-Baselines3 SAC：`auto_0.2` 使用自动 entropy 调整并指定初始 entropy coefficient。

完整 v6 修改原因见 `V6_CHANGES.md`；Stage1 实际训练评估见 [`STAGE1_EVALUATION.md`](STAGE1_EVALUATION.md)。
