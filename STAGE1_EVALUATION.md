# Stage1 训练评估与后续计划

> 评估对象：UR5e Visual Tracking + Vacuum Suction RL **v6 Simple-Flow**  
> Stage：`stage1_static`  
> 训练产物：`diagnostic/stage1_static/`  
> 评估日期：2026-09-10

## 1. 结论

Stage1 已经**正式通过课程晋级门槛**，并且最终模型表现明显高于配置中的最低晋级要求。

最终两次固定种子 deterministic eval：

| Step | Success | Safety | Table | Cube | Ready | Contact | Held | Mean reward | First success step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 239,904 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 278.57 | 117.7 |
| 259,896 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 286.89 | 84.95 |

Stage1 配置要求为：

- `success_rate >= 0.50`
- `safety_violation_rate <= 0.30`
- 连续通过 2 次 eval

因此训练在 `259,896 / 300,000` transitions 时由 curriculum callback 主动停止，并标记为通过。

**判断：Stage1 已经学会完整的“视觉接近 → ready → contact → held → lift → success”流程，可以进入下一阶段。**

---

## 2. 训练过程分析

### 2.1 早期：很快学会接触和吸附，但不会抬升

20k 时：

- `success=0`
- `ready=0.65`
- `contact=0.55`
- `held=0.55`

40k 后，`ready/contact/held` 基本已经达到 1.0，但 success 仍长期为 0。

这说明 Stage1 的主要瓶颈并不是“看不到物块”或“吸盘不能吸”，而是：

```text
reach / align
    ↓
contact
    ↓
held
    ↓
[长期停留在这里]
    ↓
lift
    ↓
success
```

模型很早就掌握了接近和吸附，但还没有形成稳定的抬升策略。

### 2.2 中期：出现第一次完整成功，但策略仍不稳定

179,928 steps：

- `success=0.125`
- `first_success_step=228.4`
- `safety=0.575`

说明完整流程第一次真正被 SAC 学到，但行为仍然非常不稳定。

199,920 steps 又退化到：

- `success=0`
- `safety=0.95`
- `robot_cube_collision=0.875`

因此，单次出现 success 不能直接认为已经收敛。

### 2.3 后期：约 240k 发生明显策略突破

239,904 steps：

- success 从 0 突然提升到 1.0
- safety 降到 0
- ready/contact/held 全部 1.0
- 首次成功平均为 117.7 steps

259,896 steps：

- success 继续保持 1.0
- safety 继续保持 0
- 首次成功平均进一步下降到 84.95 steps

相较上一轮 eval，首次成功时间减少约 32.75 control steps，说明策略不仅保持成功，而且完成流程的效率仍在改善。

这两次连续完美 eval 满足课程晋级条件，因此 Stage1 正常结束。

---

## 3. 对 v6 的判断

Stage1 的结果证明以下 v6 设计已经具备可学习性：

- pure DLS Cartesian IK 可以到达抓取区域；
- 末端 camera-based 低维视觉观测足以支持 Stage1；
- MuJoCo contact + adhesion + held 物理链可以被策略利用；
- SAC 最终能够自主学出 lift；
- 当前 reward 最终能够把策略从 contact/held 推向完整 success；
- 软碰撞约束没有阻止最终策略形成安全抓取行为。

因此，**目前不建议为了 v6.1 的 suction curriculum 重新训练 Stage1**。

v6.1 最初是为了解决“可能只会撞而不吸”的风险而设计，但这次 v6 Stage1 已经用真实训练结果证明：policy 最终能够自行学习 contact → held → lift → success，而且最后两次 eval 都达到 100% success / 0% safety。

在没有新的 Stage2 证据表明 suction 学习再次成为瓶颈之前，继续保持 v6 动作语义和 reward 语义更有利于 curriculum 连续性。

---

## 4. 当前仍需注意的风险

### 4.1 Callback eval 使用固定的 40 个种子

训练期间反复使用固定 eval seeds，有利于不同 checkpoint 公平比较，但最终模型仍需要一轮**独立种子评估**，避免只根据课程内部评估做结论。

### 4.2 240k 前波动较大

训练历史中曾出现：

- success 短暂出现后消失；
- robot-cube collision 明显升高；
- safety 从低值突然升到 0.95。

因此进入 Stage2 后不能只看单次 eval。仍应坚持至少连续两次评估，并同时看 success 与 safety。

### 4.3 Stage2 比 Stage1 难很多

Stage2 将同时增加：

- cube 最大速度到 `0.03 m/s`；
- spawn 范围扩大；
- pixel/depth noise；
- 少量 vision dropout。

所以刚进入 Stage2 时 success 明显下降是正常现象，不应立即判断模型退化。

---

## 5. 下一步：先做 Stage1 独立泛化评估

建议使用与 curriculum eval 不同的 seed，运行 200 episodes，同时测试 deterministic 与 stochastic policy：

```bash
python evaluate.py \
  --model diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip \
  --stage stage1_static \
  --episodes 200 \
  --seed 300000 \
  --mode both \
  --device cuda
```

建议的工程判断标准（不是当前代码中的自动 gate）：

- deterministic success：建议 `>= 0.90`
- deterministic safety：建议 `<= 0.05`
- stochastic success：建议 `>= 0.70`
- 不应出现持续 table/floor collision

如果 deterministic 明显低于 0.90，先不要进入 Stage2，应检查新 seed 下失败类型。

如果 deterministic 保持较高成功率而 stochastic 略低，可以继续 Stage2，因为正式部署通常使用 deterministic policy。

---

## 6. 可视化检查

在进入 Stage2 前，建议至少运行多次：

```bash
python play.py \
  --model diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip \
  --stage stage1_static
```

重点观察：

- 是否仍存在用 wrist / arm 推物块的行为；
- suction pad 是否主要从上方接近；
- held 后是否立即产生稳定抬升；
- 成功后是否出现剧烈摆动或重新砸向桌面；
- 不同随机初始位置是否都能完成抓取。

只要行为与日志一致，就可以进入 Stage2。

---

## 7. Stage2 推荐启动方式

使用 Stage1 的 best-success model 继续训练，但**不要加载 Stage1 replay buffer**：

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

原因：

- actor / critic 权重保留 Stage1 已学到的视觉接近、吸附和抬升技能；
- Stage2 状态分布发生明显变化，不保留旧 replay 更干净；
- 当前 `train.py` 从 Stage2 单独启动时本身不会加载旧 replay，除非显式传入 `--resume-replay-buffer`。

---

## 8. Stage2 应重点监控什么

进入 Stage2 后，不要只看 reward。按以下顺序判断：

```text
pregrasp
→ ready
→ contact
→ held
→ success
```

重点解释：

- `pre/ready` 高但 `contact` 低：移动目标截获 / 下降存在问题；
- `contact` 高但 `held` 低：suction / 接触稳定性成为瓶颈；
- `held` 高但 `success` 低：主要是追踪后的 lift 策略问题；
- success 上升同时 safety 上升：可能出现推块或碰撞型 shortcut；
- reward 上升但 success 不升：需要再次检查 reward alignment。

Stage2 当前自动 gate：

- `success >= 0.55`
- `safety <= 0.30`
- 连续 2 次 eval

达到后再进入 Stage3。

---

## 9. 模型文件说明

当前仓库中：

```text
diagnostic/stage1_static/final_model.zip
diagnostic/stage1_static/stage1_static.zip
diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip
```

三者当前对应同一个 Git blob 内容，因此本次 Stage1 最终模型和 best-success 模型实际上相同。

推荐后续统一使用：

```text
diagnostic/stage1_static/best_success/stage1_static/best_success_model.zip
```

这样模型和 `best_metrics.json` 可以保持在同一目录中，来源最清晰。

---

## 10. 当前项目状态

```text
Stage1 static
    ✓ pregrasp
    ✓ ready
    ✓ contact
    ✓ held
    ✓ lift
    ✓ success = 100%（最后两次内部 eval）
    ✓ safety = 0%（最后两次内部 eval）
    ↓
Independent Stage1 evaluation
    ↓
Stage2 slow moving target
    ↓
Stage3 moving target
```

当前建议状态：**Stage1 已完成，先做独立评估，随后进入 Stage2；暂不修改 v6 核心算法。**
