# UR5e Visual Tracking + Vacuum Suction RL v6 Simple-Flow

这是针对前几个版本训练问题重新收敛后的正式 Pick-and-Lift 基线版本。

当前核心能力：

```text
末端相机看到物块
→ 追踪 / 接近
→ suction pad 接触
→ 真空吸附
→ held
→ 抬升
→ success
```

当前 v6 的主要价值不是继续无限扩展单物块任务，而是作为后续 **Pick-and-Place / Multi-object Sorting** 的稳定抓取基座。

## 当前训练状态（2026-09-10）

**Stage1 `stage1_static` 已通过 curriculum gate。**

最后两次 40-episode deterministic eval：

| Step | Success | Safety | Ready | Contact | Held | Mean reward | First success step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 239,904 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 278.57 | 117.7 |
| 259,896 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 286.89 | 84.95 |

Stage1 自动晋级条件：

```text
success >= 0.50
safety  <= 0.30
连续通过 2 次 eval
```

训练器在 `259,896 / 300,000` transitions 时主动判定 Stage1 通过并结束训练。

训练过程中一个重要现象是：`contact/held` 很早就出现，但稳定 lift / success 到约 240k 才真正形成。因此不能仅因为早期 success=0 就判断训练失败，必须结合 `ready → contact → held → success` 整条技能链。

### Stage1 文档与产物

- [`STAGE1_EVALUATION.md`](STAGE1_EVALUATION.md)：Stage1 完整训练评估；
- [`diagnostic/stage1_static/`](diagnostic/stage1_static/)：日志、eval history、best/final model；
- [`UR5E_REPOSITORY_TECHNICAL_ANATOMY_REPORT.md`](UR5E_REPOSITORY_TECHNICAL_ANATOMY_REPORT.md)：当前仓库逐模块技术解剖；
- [`UR5E_SORTING_EXECUTION_PLAN.md`](UR5E_SORTING_EXECUTION_PLAN.md)：从当前 Pick-Lift 升级到多物品分拣的正式执行路线。

---

# 项目路线更新：从“移动物块”转向“完整分拣”

原 v6 Curriculum 定义了：

```text
stage1_static
→ stage2_slow
→ stage3_moving
```

其中 `stage2_slow / stage3_moving` 的作用是验证动态目标视觉跟踪、截获能力，并为未来传送带抓取做准备。

在项目最终目标明确为：

```text
桌面多个不同物品
→ 识别 / 选择目标
→ 抓取
→ 抬升
→ 搬运
→ 放入对应箱子
→ 释放
→ 继续下一件
→ 全部分拣完成
```

之后，**移动物块不再作为当前主线下一阶段**。

推荐主路线改为：

```text
v6 Stage1: Pick-Lift baseline               ✅ 已完成
        ↓
v7 Stage1: single object → fixed bin       ← 下一正式开发阶段
        ↓
v7 Stage2: single object → multi-bin goal
        ↓
v7 Stage3: multi-object target pick
        ↓
v7 Stage4: single-cycle sorting
        ↓
v7 Stage5: sequential sorting
        ↓
v7 Stage6: domain randomization / robustness
        ↓
v7 Stage7: moving/conveyor sorting         可选最终扩展
```

因此当前 `stage2_slow / stage3_moving` **代码保留，但降级为动态目标 benchmark / 后期扩展**。不建议现在把主要训练算力继续投入移动 cube，而应该先补齐 `transport → place → release → correct-bin success`。

详细设计见 [`UR5E_SORTING_EXECUTION_PLAN.md`](UR5E_SORTING_EXECUTION_PLAN.md)。

---

## 当前系统结构

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

当前 observation 为 21 维，主要包含：

- 6 个关节位置；
- 6 个关节速度；
- `eef_camera` 对唯一 cube 的 `u/v/depth`；
- 对应视觉变化 `du/dv/ddepth`；
- visible；
- suction touch；
- suction ctrl。

重要说明：当前 SAC **并不是直接从 RGB 中识别多个物体**。`scene.xml` 中的 `camprojection` / `framepos` sensor 直接提供唯一 cube 的低维视觉状态；`camera_viewer.py` 中的红色 HSV 检测目前仅用于 `play.py` 可视化窗口，不进入 policy observation。

这也是后续多物品 sorting 必须新增 perception / target-object / target-bin 接口的原因。

---

## 为什么需要 v6

最初版本虽然存在机械臂撞桌、推动物块等 reward hacking，但早期实验已经证明当前 MuJoCo suction + SAC 结构可以产生成功抓取。

之后 v4/v5 为了修复作弊同时增加较多约束。最关键的问题之一是 v5 的 IK posture regularization 与 Cartesian 目标竞争，导致 `smoke_grasp.py` 在不使用 RL 的情况下都曾出现：

```text
phase=pregrasp reached=False
```

因此 v6 回到更简单的控制链：

```text
SAC Cartesian action
→ workspace clip
→ pure DLS IK
→ UR5e actuator
```

只保留必要的安全和物理约束。

---

## v6 主要设计

### 1. Pure DLS IK

没有：

- posture regularization；
- null-space home 强制回正；
- cube 真值下降 shield；
- 额外 software joint envelope。

只保留：

- 末端工作区裁剪；
- UR5e actuator 自身 ctrlrange；
- 较弱末端朝向保持。

### 2. 连续真空控制

当前 `env.py` 实际实现为：

```python
suction_ctrl = 0.5 * (action[3] + 1.0)
```

因此：

```text
action=-1 → suction=0%
action= 0 → suction=50%
action=+1 → suction=100%
```

MuJoCo `adhesion` 仍依赖 suction tool 与物体 contact；不会隔空把目标吸过来。

### 3. Staged reward

当前 reward 以任务阶段为主，并使用 `max` 抑制多个中间 dense 项叠加造成 reward hacking：

```text
reach / align
→ vacuum contact
→ held
→ lift
→ success
```

普通 contact 不等于 grasp；只有 contact 与有效 suction 同时成立才进入 vacuum-contact stage。

### 4. 固定 Horizon

当前 horizon 为 320。

成功以后 `ever_success=True`，episode 不立即结束，剩余时间保持最高 success stage reward。这样越早成功，总 return 越高。

### 5. 软碰撞约束

- 瞬时 robot-table / robot-cube contact 不立即 terminate；
- table / floor 只有持续碰撞才 hard failure；
- suction pad 正常接触 cube 允许；
- safety 仍单独统计，避免只根据 reward 选择模型。

---

## 训练前检查

### 系统检查

```bash
python check_system.py
```

必须看到：

```text
CHECK_SYSTEM_OK
```

### 物理链 smoke test

```bash
python smoke_grasp.py --render
```

必须看到：

```text
SMOKE_GRASP_SUCCESS= True
```

它验证同一个 `env.step()` 链上的：

```text
DLS IK
→ contact
→ adhesion
→ held
→ lift
→ success
```

如果 scripted flow 失败，不应继续浪费 SAC training steps。

---

## Stage1 独立评估

Curriculum callback 使用固定 eval seeds。建议用不同 seed 额外做 200-episode 泛化测试：

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

这些是额外工程建议，不是当前代码自动 gate。

---

## Stage1 可视化回放

```bash
python play.py \
  --model diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip \
  --stage stage1_static
```

重点检查：

- 是否有 arm/wrist 推物块；
- 是否长期压桌；
- 吸住后是否剧烈摆动；
- 是否真的完成视觉接近 → suction → held → lift。

---

# 下一正式开发目标：v7_sorting Stage1

不要直接上多物品。

下一步只增加一个新能力：**Place**。

场景：

```text
1 个静态物体
+
1 个固定目标箱
```

目标：

```text
Pick
→ Lift
→ Transport
→ Hover above bin
→ Descend
→ Suction OFF
→ Object inside bin
→ Released
→ Settled
→ SUCCESS
```

成功必须是物理意义上的：

```text
object 位于 bin 内部
AND held=False
AND suction 已释放
AND object 速度足够小
AND 连续稳定若干 step
```

这个阶段成功以后，再增加多箱 Goal Conditioning；再之后才增加多物品目标选择。

详细里程碑、Observation、Reward、HER、多目标感知和最终 Sequential Sorting 设计见：

[`UR5E_SORTING_EXECUTION_PLAN.md`](UR5E_SORTING_EXECUTION_PLAN.md)

当前仓库各模块、DLS IK、SAC、Replay Buffer、21 维 Observation、MuJoCo sensor、parallel、callbacks、preflight、evaluate、play 和 camera viewer 的详细解释见：

[`UR5E_REPOSITORY_TECHNICAL_ANATOMY_REPORT.md`](UR5E_REPOSITORY_TECHNICAL_ANATOMY_REPORT.md)

---

## 安装

```bash
pip install -r requirements.txt
```

UR5e OBJ 资源位于：

```text
assets/
```

如果本地缺失，可以使用：

```bash
python scripts/prepare_assets.py
```

---

## 当前推荐模型

```text
diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip
```

这个模型作为当前 Pick-Lift baseline，建议后续冻结并保留，用于 v7 每次改动后的单技能回归测试。
