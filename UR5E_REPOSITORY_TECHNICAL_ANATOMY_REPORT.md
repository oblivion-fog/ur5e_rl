# UR5e Visual Tracking + Vacuum Suction RL 项目技术解剖报告

> 仓库：`oblivion-fog/ur5e_rl`
>
> 本报告解释当前仓库各模块做什么、为什么这么做、数据如何流动、强化学习算法怎样工作，以及当前架构距离“多物品分拣”还差哪些关键能力。
>
> 当前主体版本：`v6_simple_flow_2026_09_08`。

## 1. 当前项目已经实现什么

当前系统已经能够训练 UR5e 完成：

```text
机械臂回到 home
↓
末端相机获得目标物块位置/深度特征
↓
SAC 输出 [dx, dy, dz, suction]
↓
Cartesian 位移经过 DLS IK 转成 6 个关节目标
↓
吸盘接近物块
↓
contact + adhesion 建立吸附
↓
held
↓
向上抬升
↓
达到 success_height
↓
SUCCESS
```

当前 v6 Stage1 最后两次 40-episode deterministic eval 均达到：

```text
success = 1.0
safety = 0.0
ready = 1.0
contact = 1.0
held = 1.0
```

所以当前仓库最准确的定位是：

> 一个已经实际验证成功的“末端视觉 + Cartesian SAC + 真空吸附”的单物体 Pick-and-Lift 基座。

它还不是完整 Pick-and-Place，更不是 Multi-object Sorting。

---

## 2. 整体数据流

```text
scene.xml + ur5e.xml
        ↓
MuJoCo MjModel / MjData
        ↓
UR5eVisualSuctionEnv
        ├── observation ─────────────→ SAC Actor
        │                                ↓
        │                      [dx,dy,dz,suction]
        │                                ↓
        │                           DLS Jacobian IK
        │                                ↓
        │                       6 joint position ctrl
        │
        ├── suction action ─────────→ adhesion ctrl
        ├── MuJoCo physics
        ├── sensors / contacts
        └── reward / success
                    ↓
               replay buffer
                    ↓
               SAC critic/actor
```

训练时再包一层：

```text
12 个 SubprocVecEnv workers
           ↓
并行采集 transitions
           ↓
SAC replay buffer
           ↓
PyTorch / CUDA
```

---

## 3. `ur5e.xml`：机器人本体

这个文件定义：

1. UR5e 六关节和连杆；
2. 可视 mesh；
3. 简化 collision geom；
4. 六个关节执行器；
5. 自定义 suction tool；
6. suction pad；
7. wrist/end-effector camera；
8. MuJoCo adhesion actuator。

### 3.1 六轴 UR5e

关节：

```text
shoulder_pan
shoulder_lift
elbow
wrist_1
wrist_2
wrist_3
```

当前 RL **不直接输出六个关节角**。Policy 输出末端 Cartesian 小位移，环境使用 DLS IK 计算每个关节该怎么配合。

### 3.2 suction tool

`suction_tool` 安装在 `wrist_3_link` 后方，包含：

- `suction_mount`：只负责显示；
- `suction_pad`：参与实际接触/吸附；
- `suction_site`：作为 TCP、IK 和距离计算参考；
- `suction_touch_site`：触觉传感区域；
- `eef_camera`：末端相机。

### 3.3 adhesion

```xml
<adhesion name="suction" body="suction_tool" ctrlrange="0 1" gain="30"/>
```

通俗理解：

> 当吸盘表面和目标物体形成有效 contact 时，`ctrl` 决定吸附力强度。

它并不是“隔空磁铁”，必须先有合理接触候选。

---

## 4. `scene.xml`：实验场景

`scene.xml` include `ur5e.xml`，然后加入：

- gravity；
- floor；
- table；
- 当前唯一一个可抓 cube；
- overview camera；
- MuJoCo sensors；
- home keyframe。

### 4.1 桌子

当前桌面顶部约：

```text
z = 0.40 m
```

这比早期版本更合理，主要为了避免 UR5e 上臂把桌边当“支点”产生 reward hacking。

### 4.2 cube

当前只有一个：

```text
body name="cube"
```

它使用 freejoint，因此可以在桌面自由运动，也能在 suction 吸附后被抬起。

### 4.3 传感器

关键 sensor：

```text
suction_contact
cube_pixel
cube_in_camera
cube_top_in_suction
cube_velocity
```

尤其：

```xml
<camprojection name="cube_pixel" site="cube_center" camera="eef_camera"/>
```

这意味着当前 policy 并不是从 RGB 图像中自己“识别红色方块”，而是 MuJoCo 已经直接计算出了唯一 cube 在末端相机中的像素投影。

这是当前系统训练效率高的重要原因，也是未来多物体分拣必须重构的地方。

---

## 5. `ur5e_rl/config.py`：参数和 Curriculum

`config.py` 相当于项目的统一实验参数表。

### 5.1 `StageSpec`

每个 stage 定义：

- 名称；
- cube 最大速度；
- 初始随机位置；
- 移动范围；
- vision noise；
- 最少/最多训练步数；
- promotion success；
- safety gate；
- 连续通过 eval 次数。

当前三阶段：

```text
stage1_static
stage2_slow
stage3_moving
```

Stage2/3 的“移动”主要用于动态视觉 tracking benchmark，不等同于 sorting。

### 5.2 `EnvConfig`

主要控制：

```text
horizon
control_hz
max_cartesian_step
ik_damping
max_joint_step
orientation_weight/orientation_gain
cube drive
success height
contact threshold
suction threshold
collision penalties
workspace
camera resolution/noise
```

统一集中配置的好处是：算法逻辑和实验参数不会混在一起。

---

## 6. `ur5e_rl/env.py`：整个系统核心

当前这个文件承担了最多职责：

```text
Gymnasium Env
+ MuJoCo 初始化
+ reset
+ observation
+ visual features
+ DLS IK
+ suction
+ moving cube
+ contacts/safety
+ reward
+ success
+ render
```

当前单物体任务还能维护，但进入多物体 sorting 后必须拆分职责。

---

## 7. Action Space

当前：

```python
Box(-1, 1, shape=(4,))
```

四维含义：

```text
a0 = TCP x 增量
a1 = TCP y 增量
a2 = TCP z 增量
a3 = suction control
```

前三维不是关节角。

例如：

```text
[0, 0, -1, ...]
```

表示让吸盘下一 control step 尽可能向下移动。

这比直接让 SAC 输出 6 个关节动作更简单，因为 RL 不需要自己学习整套逆运动学。

---

## 8. DLS Jacobian IK 原理

机械臂微分运动可近似写成：

```text
Δx = J Δq
```

其中：

- `Δx`：末端希望产生的小位移/姿态变化；
- `J`：当前姿态下的 Jacobian；
- `Δq`：六关节应该怎么变化。

直接对 Jacobian 求逆在奇异姿态附近容易数值爆炸，因此 v6 使用 Damped Least Squares：

```text
Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ Δx
```

`λ` 是 damping。

通俗理解：

> Jacobian 告诉系统“每个关节动一点，末端会怎么动”；DLS 则反过来计算“末端想往这里走，六个关节应该怎么配合”。

### v5 为什么曾经失败

v5 一度把 posture regularization 直接加入主任务，使“末端去目标”和“关节回 home”竞争，导致 scripted smoke test 都无法到达 pregrasp。

v6 恢复 pure DLS 后，底层 IK→contact→adhesion→lift 链路重新打通。

---

## 9. Orientation 保持

系统不是让吸盘随便翻转。

reset 后保存 home 时末端旋转矩阵：

```text
_desired_eef_rotation
```

每一步计算当前姿态和目标姿态误差，再把 rotation Jacobian 与 position Jacobian 一起求解。

`orientation_weight` 较低，所以策略是：

```text
第一优先：位置跟踪
第二优先：尽量保持吸盘朝下
```

这很适合桌面吸盘操作。

---

## 10. suction control

当前代码将 SAC 第 4 维 `[-1,1]` 映射到 `[0,1]`：

```python
0.5 * (suction_action + 1.0)
```

因此：

```text
-1 → 0%
 0 → 50%
+1 → 100%
```

这是一个值得注意的代码/README 版本漂移点：旧 README 曾描述过 `clip(action, 0, 1)`。后续新版本应统一文档和 action semantics。

---

## 11. held 判定

`_is_held()` 同时要求：

```text
suction_ctrl 足够大
AND contact_force 足够大
AND cube_top 与 suction_site 足够近
```

所以：

```text
contact=True
```

并不等于：

```text
held=True
```

这对诊断“只是撞到了”还是“真的吸住了”非常重要。

---

## 12. Observation Space：21 维

当前 observation：

```text
6 joint positions
+ 6 joint velocities
+ 7 visual features
+ 1 normalized contact
+ 1 suction ctrl
= 21 dims
```

### 12.1 视觉 7 维

```text
u
v
depth
du
dv
ddepth
visible
```

`u/v` 是相对于图像中心归一化后的目标位置。

```text
u ≈ 0, v ≈ 0
```

表示目标大致在相机中心。

`du/dv/ddepth` 是连续 control step 的视觉变化，为动态目标提供趋势信息。

---

## 13. OpenCV 检测不是 RL 感知

`ur5e_rl/camera_viewer.py` 中存在 `_detect_red_object()`：

```text
BGR
→ HSV
→ 红色阈值
→ morphology
→ contours
→ bbox / center
```

但它只是 `play.py` 的可视化叠加。

**它没有作为 observation 输入 SAC。**

所以当前 policy 学到的是：

> “唯一指定 cube 的 MuJoCo 投影在哪里？”

不是：

> “画面里有多个不同物体，我要识别其中哪一个。”

这就是多物品 sorting 与当前单物体任务之间最大的架构鸿沟之一。

---

## 14. `_apply_cube_drive()`：物块为什么移动

Stage2/3 中，环境会：

1. 采样二维目标速度；
2. 读取 cube 当前速度；
3. 计算速度误差；
4. 用类似 P 控制生成 desired acceleration；
5. 通过 `F = m a` 算水平力；
6. 用 `xfrc_applied` 施加到 cube；
7. 到边缘后反向。

如果 cube 已被 held，则停止外部 drive。

这是一种动态目标 tracking benchmark，而不是分拣必须能力。

---

## 15. Reward

当前 reward 采用 staged reward。

### Reach / Align

距离越近：

```text
1 - tanh(k * distance)
```

越大。

### Vacuum Contact

当前只有：

```text
contact AND vacuum_active
```

才进入 contact stage，避免重现早期“反复撞物块却不开吸盘”的局部最优。

### Held / Lift

真正 held 后进入更高 reward；lift_height 越接近目标高度，reward 越高。

### Success

一旦 `_ever_success=True`：

```text
success stage = 1.0
```

### 为什么用 `max`

如果把：

```text
visible + reach + pregrasp + contact + ...
```

全部无条件相加，agent 很可能找到某个中间状态刷大量 reward。

使用 staged/max 的目标是保证：

> 完整技能阶段永远比中间状态更有价值。

---

## 16. Success 判定

当前 success：

```text
held
AND lift_height >= success_height
AND 连续保持 success_hold_steps
```

成功后 episode 不立即结束。

固定 horizon 下，越早成功，剩余越多 step 都能拿最高 success reward，因此自然鼓励更快完成。

---

## 17. Safety

当前不是“一碰就死”。

系统记录：

```text
robot_table_collision
robot_cube_collision
robot_floor_collision
```

但策略是：

- 瞬时碰撞先扣 reward；
- table 连续多 step 才 hard failure；
- floor 连续多 step 才 hard failure；
- robot body 碰 cube 计 safety，但不一定立即 terminate；
- suction pad 正常碰 cube 不当作 robot_cube_collision。

这是 v6 相比早期过强 shield 更利于探索的地方。

---

## 18. `ur5e_rl/parallel.py`

这个模块负责把单环境复制成多个独立进程。

当前：

```text
SubprocVecEnv
```

每个 worker 都有独立 MuJoCo model/data 和随机 seed。

外层 `VecMonitor` 记录 episode 指标。

所以训练结构是：

```text
12 x MuJoCo CPU sampling
+
PyTorch CUDA SAC training
```

MuJoCo physics 主要在 CPU；GPU 主要用于 Actor/Critic 神经网络更新。

---

## 19. `train.py`

这是训练总调度器。

主要职责：

1. 解析命令行参数；
2. 检查 CUDA；
3. `check_env`；
4. 运行 scripted preflight；
5. 创建并行环境；
6. 创建/恢复 SAC；
7. 按 Stage 循环；
8. 保存 checkpoint；
9. 定期 eval；
10. 判断 curriculum 晋级；
11. 保存 final model。

### 当前 SAC 主要超参数

```text
learning_rate = 3e-4
buffer_size = 1,000,000
learning_starts = 20,000
batch_size = 256
tau = 0.005
gamma = 0.98
ent_coef = auto_0.2
```

---

## 20. 为什么使用 SAC

SAC 很适合当前任务，因为：

```text
动作是连续的
环境物理仿真成本不低
需要高样本利用率
```

### Actor

输入 observation，输出动作分布。

### Critic

通常有两个 Q 网络：

```text
Q(s,a)
```

估计当前状态下执行动作的长期价值。

### Entropy

SAC 不只追求 reward，还主动保持一定随机探索。

`ent_coef="auto_0.2"` 表示 entropy coefficient 自动调整，并使用 0.2 作为初始值。

---

## 21. Replay Buffer

SAC 是 off-policy。

每一步存：

```text
(s, a, r, s', done)
```

进 replay buffer。

神经网络之后随机抽 batch 重复学习这些 transition，所以同一条 MuJoCo 数据不会只用一次。

---

## 22. `ur5e_rl/callbacks.py`

这是 curriculum 的“考试老师”。

每隔固定 transitions：

1. deterministic eval；
2. 统计 success/safety/collision；
3. 统计 pregrasp/ready/contact/held；
4. 写 TensorBoard；
5. 写 `history.jsonl`；
6. 保存 best model；
7. 判断是否达到 promotion gate。

### Best model 排序原则

```text
success
→ safety
→ reward
```

这是合理的，因为不能只按 reward 选模型，否则可能把 reward hacking 策略当成最好模型。

---

## 23. `ur5e_rl/preflight.py`

这是非常重要的工程保护层。

它不使用 RL，而是脚本式执行：

```text
pregrasp
→ suction on
→ descend
→ contact/held
→ lift
→ success
```

关键点是它仍然调用同一个：

```text
env.step()
```

所以 DLS IK、contact、adhesion、success 判定与 RL 完全相同。

意义：

> 如果 scripted grasp 都失败，就不应该浪费几十万 RL transitions。

v5 的 IK 问题就是靠它真正定位出来的。

---

## 24. `check_system.py`

自动检查入口。

它会：

- 构造各 Stage env；
- reset；
- 检查 observation；
- 输出 table/workspace；
- 检查初始 contact；
- 再运行 scripted grasp。

最后出现：

```text
CHECK_SYSTEM_OK
```

才说明底层物理链可以训练。

---

## 25. `smoke_grasp.py`

用于人工快速诊断：

```bash
python smoke_grasp.py
python smoke_grasp.py --render
```

输出：

```text
min_pregrasp_error
contact_seen
held_seen
lift_height
success
```

你之前得到：

```text
SMOKE_GRASP_SUCCESS=True
```

这是一个非常重要的证据：

> IK → contact → adhesion → lift 链路本身没有根本故障。

---

## 26. `evaluate.py`

训练结束后的独立评估器。

支持：

```text
deterministic
stochastic
both
```

与 callback 的区别：

- callback：训练过程中固定 seed 的晋级考试；
- `evaluate.py`：训练后可以换全新 seed 做独立泛化测试。

---

## 27. `play.py`

用于观察模型真实行为。

同时运行：

```text
MuJoCo overview viewer
+
OpenCV eef camera window
```

每步：

```text
model.predict
→ env.step
→ render
→ camera process
```

episode 结束会打印：

```text
reward
success
safety
contact
held
suction
first_success
lift
```

这个工具对于识别“日志很好看，但实际动作很奇怪”的 shortcut 非常有价值。

---

## 28. `camera_viewer.py`

OpenCV GUI 和 MuJoCo 主循环放在一起容易互相阻塞，所以这里单独启动一个进程。

数据流：

```text
主进程
  ↓ CameraPacket
Queue(maxsize=1)
  ↓
OpenCV camera worker
```

只保留最新 packet 是正确的，因为实时显示没有必要把旧帧排队补播。

---

## 29. `scripts/prepare_assets.py`

用于准备 MuJoCo Menagerie UR5e mesh。

可以：

```text
从本地已有 assets 复制
```

或者：

```text
从官方 Menagerie 下载
```

这样 mesh 资源准备与训练逻辑解耦。

---

## 30. `requirements.txt`

主要依赖：

```text
mujoco              物理仿真
gymnasium           RL Env API
stable-baselines3   SAC / VecEnv / Monitor
torch               神经网络 / CUDA
numpy               数值计算
opencv-python       相机显示 / 简单检测
tensorboard         训练曲线
tqdm / rich         终端输出
```

---

## 31. Stage1 训练过程真正说明了什么

Stage1 history 中：

```text
20k:
contact ≈ 0.55
held ≈ 0.55
success = 0
```

之后很长一段：

```text
contact ≈ 1
held ≈ 1
success = 0
```

说明策略很早就学会了：

```text
接近
→ 接触
→ 吸住
```

真正困难的是：

```text
held
→ 稳定 lift
→ 达到 success_height
```

约 180k 才首次出现少量成功，240k 后才稳定突破到 100%。

这说明训练中不能只看 reward，也不能因为 success 前期为 0 就立即认定“完全没学到”。应该结合完整技能链判断。

---

# 32. 当前仓库距离多物品 Sorting 还缺什么

## 32.1 Scene 只有一个物体

代码全部是：

```text
_cube_body_id
_cube_geom_id
_cube_top_site_id
```

多物体必须改成 object registry/list。

## 32.2 Sensor 硬编码唯一 cube

```text
cube_pixel
cube_in_camera
cube_top_in_suction
cube_velocity
```

无法直接描述 3 个不同物体。

## 32.3 Observation 没有类别/目标 ID

当前 21 维不包含：

```text
class_id
object_id
target_object_id
bin_id
target_bin_id
```

所以 policy 没有“该抓谁”的信息。

## 32.4 没有 Target Bin

当前 reward 终点只是：

```text
lift_height
```

没有：

```text
transport
hover
descend into bin
release
settled
correct bin
```

## 32.5 没有 Release 技能

当前拿起来就算成功。

Sorting 必须学会在正确位置：

```text
suction OFF
```

并确认物体真正留在箱中。

## 32.6 没有 Multi-object Selection

即便 XML 多放几个 object，现有 policy 也只收到硬编码单个 cube 的视觉坐标。

## 32.7 Wrist Camera 不适合单独做全局搜索

`eef_camera` 很适合最终精对准，但多物体全局发现推荐增加固定 `sorting_camera`。

## 32.8 没有长任务状态

最终一个 episode 需要知道：

```text
object0 sorted
object1 unsorted
object2 unsorted
```

当前环境没有 object mask / task state。

---

# 33. Goal-conditioned Policy 为什么必要

未来同一个机器人状态：

```text
机械臂正拿着某个物体
```

可能要求：

```text
送左箱
```

也可能要求：

```text
送右箱
```

如果 observation 不包含目标箱，policy 无法知道哪种动作才正确。

所以未来应显式输入：

```text
target_object
target_bin / desired_goal
```

这是 Goal-conditioned RL 的核心。

---

# 34. HER 为什么值得测试

目标放置任务比“抬高 8 cm”更稀疏。

早期策略可能：

```text
成功抓起
→ 搬到错误位置
→ episode fail
```

HER 的思想是：

> 虽然没有达到原目标，但它实际到达了某个位置；将这个实际位置重新当成一个虚拟 goal，这条失败轨迹仍有学习价值。

因此 HER 很适合后续单物体 goal-conditioned placement。

但最好在单物体多箱阶段引入，不要和多物体选择同时增加。

---

# 35. 最终算法边界建议

不要把“强化学习项目”理解成：

> 分类、检测、任务规则和机械臂 20 Hz 控制都必须塞进一个 SAC 网络。

更稳健的第一版系统应是：

```text
视觉检测/分类
↓
Task Manager：class→bin
↓
Goal-conditioned SAC
↓
抓取/搬运/放置
```

强化学习仍然负责最关键的连续机器人操作。

如果研究目标要求“先抓哪件也必须由 RL 决定”，再增加：

```text
High-level PPO/DQN selector
+
Low-level SAC manipulation
```

这比一个 flat SAC 同时处理两个时间尺度更合理。

---

# 36. 当前代码的版本技术债

`config.py` 仍标：

```text
v6_simple_flow_2026_09_08
```

但 `env.py` 已经使用：

```text
0.5 * (suction_action + 1)
```

并且 contact stage 已绑定 `vacuum_active`。

因此当前代码实际上吸收了一部分后续修正，而版本号/README 曾有残留旧描述。

建议在进入 v7 前：

1. 冻结当前成功 baseline；
2. 建议打 tag `v6-pick-lift-baseline`；
3. 不再继续往 v6 主线叠功能；
4. 新建 v7 sorting branch/version；
5. action / reward semantics 改变时同步升级版本号。

---

# 37. 一句话理解整个仓库

如果把整个项目想象成一个人：

- `scene.xml`：桌子和物体摆在哪里；
- `ur5e.xml`：机器人的骨骼、肌肉、吸盘和眼睛；
- `env.py`：神经系统，翻译动作、整理传感器、计算 reward；
- `config.py`：训练计划和参数表；
- `SAC`：通过不断试错学习的“大脑”；
- `parallel.py`：同时开启多个训练场；
- `callbacks.py`：定期考试和晋级；
- `preflight.py`：开课前体检；
- `evaluate.py`：毕业考试；
- `play.py`：现场演示；
- `camera_viewer.py`：给人看腕部相机；
- `prepare_assets.py`：准备机器人 mesh。

而后续 `v7_sorting` 要给这个“已经会抓东西的大脑”增加四类新能力：

```text
该抓哪个？
该送去哪？
什么时候放？
还有哪些物品没处理？
```

这四个问题解决以后，项目才会真正从单物体 Pick-and-Lift 演进为完整 Multi-object Sorting 系统。
