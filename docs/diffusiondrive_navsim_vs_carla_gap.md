# DiffusionDrive: NAVSIM 原版与 CARLA Port 的差异清单

**kimi k2.6-code-preview挑的刺。**

> **目的**：建立并固化对"原版 NAVSIM 假设"与"当前 CARLA port 实现"之间差异的精确认知，防止后续迭代中遗漏关键分布 mismatch。
> **基线代码**：
> - 原版：`./DiffusionDrive/navsim/agents/diffusiondrive/` (commit 以当前工作区为准)
> - CARLA Port：`./carla_garage/team_code/diffusiondrive_agent.py`、`./carla_garage/team_code/diffusiondrive/`
>
> **状态说明**：本文最初用于记录 port 早期 gap。`99x10x2` anchor、XY-only trajectory head、7 维 `status_feature`、空间 checkpoint PID 等当前主线事实，以 `diffusiondrive_anchor_adaptation.md`、`diffusiondrive_training.md` 和 `dd_train_todo.md` 为准。

---

## 1. 相机输入（最致命的分布差异）

### 原版 NAVSIM
- **多相机拼接**：使用 `cam_l0` + `cam_f0` + `cam_r0` 三张图像。
- **预处理流程**：
  1. 裁剪：`l0` 和 `r0` 去掉上下各 28px、左右各 416px；`f0` 只去掉上下 28px。
  2. 水平拼接：将 `l0 + f0 + r0` 沿宽度方向拼接。
  3. Resize 到 `(1024, 256)`。
  4. `transforms.ToTensor()` 归一化到 `[0, 1]`。
- **来源文件**：`DiffusionDrive/navsim/agents/diffusiondrive/transfuser_features.py:55-75`

### 当前 CARLA Port
- **单前视相机**：仅使用 Leaderboard 注册的 `rgb_front`。
- **预处理流程**：
  1. `t_u.crop_array()` 裁剪（默认输出 `384×1024`，基于 `GlobalConfig.cropped_height/width`）。
  2. 手动注入 **JPEG artifact**：`cv2.imencode('.jpg')` → `cv2.imdecode()`。
  3. 在 `run_step()` 中 `/255.0` 后再用 `F.interpolate` resize 到 `(dd_config.camera_height, dd_config.camera_width)` 即 `(256, 1024)`。
- **来源文件**：`carla_garage/team_code/diffusiondrive_agent.py:315-328` 及 `:513-514`

### 差异影响
- **若直接加载原版 NAVSIM checkpoint 进行推理**：模型 backbone 接收的图像特征分布完全不同（单视图 vs 拼接宽视图）。这是当前最大的输入分布 mismatch，会显著降低性能。
- **JPEG artifact 的争议**：CARLA garage 的 `sensor_agent` 加入 JPEG 是为了匹配训练数据保存格式；但原版 NAVSIM 训练数据是直接由原始图像 `resize + ToTensor`，没有 JPEG 这一步。因此 JPEG 对 DiffusionDrive 不一定是正向的。

### 建议
1. 若坚持单相机输入，应在 CARLA 上重新训练 backbone 权重，或冻结 backbone 进行微调。
2. 若希望零样本利用原版权重，需要复现三相机拼接逻辑（CARLA Leaderboard 中注册多个相机并做拼接）。
3. 明确是否保留 JPEG artifact；若重新训练，应与训练侧的图像保存格式保持一致。

---

## 2. LiDAR 时序与空间处理

### 原版 NAVSIM
- **单帧无缓冲**：只使用 `agent_input.lidars[-1]` 单帧点云。
- **无坐标重对齐**：开环数据本身就是 ego 坐标系，不需要历史帧变换。
- **BEV 生成**：直接调用 histogram splat，逻辑与 `carla_garage` 基本一致。默认配置是 `pixels_per_meter=4`、`[-32m,32m]` 范围，得到 `256x256` BEV histogram；默认 `use_ground_plane=False`，实际输入通常是一层 above-ground BEV。
- **LiDAR backbone**：原版默认 `lidar_architecture="resnet34"`，不是 syb / garage 旧模型中常见的 `regnety_032`。

### 当前 CARLA Port
- **半帧拼接**：每步只收到半帧 LiDAR，用 `lidar_last` 与当前帧拼接成完整扫描。
- **多帧 Buffer**：`lidar_buffer` 长度为 `lidar_seq_len * data_save_freq`；未满时持续刹车等待。
- **历史帧 Realign**：若 `config.realign_lidar=True` 且 `lidar_seq_len>1`，使用 `align_lidar()` 将历史帧点云变换到当前 ego 坐标系。
- **模型侧 LiDAR encoder**：当前 DiffusionDrive config 默认仍是 `lidar_architecture="resnet34"`；`GlobalConfig.lidar_architecture="regnety_032"` 没有自动同步到 `DiffusionDriveConfig`。因此当前 CARLA DiffusionDrive 更接近原版 NAVSIM 的 LiDAR backbone，而不是 syb 的 backbone。
- **来源文件**：`carla_garage/team_code/diffusiondrive_agent.py:444-510`

### 差异影响
- **CARLA 端反而更完整**：LiDAR 时序链路已经基本对齐甚至超出原版。只要坐标变换（`lidar_to_ego_coordinate` + `align_lidar`）与 BEV 坐标系假设一致，这一块不是瓶颈。
- **风险点**：`lidar_to_ego_coordinate()` 将点云旋转平移到 ego 坐标系后，`data.lidar_to_histogram_features()` 中的 `splat_points` 有 `overhead_splat.T`（x/y 轴 swap）。需要确认这个 transpose 与 `align_lidar` 的组合在时序帧上不会引入方向错位（参见第 7 节）。
- **zero-LiDAR 诊断边界**：`DIFFUSIONDRIVE_ZERO_LIDAR=1` 只置零模型侧 BEV 输入，不关闭 raw LiDAR safety-box。20-route Z0/Z1 结果显示 zero model-LiDAR 反而提升 DS / RC 并降低 forced creep，这优先指向模型侧 BEV LiDAR 分支的 domain gap 或监督不足，而不是证明 safety-box 应该移除。
- **backbone 切换边界**：`regnety_032` 是 `timm` 2D CNN 名称，不是 LiDAR 表示方式。若要对齐 syb，应作为 full retrain ablation；仅 warm-start 当前 ResNet34 checkpoint 没有意义。

---

## 3. 轨迹表示：CARLA 主线已升级为 `99x10x2` / XY-only

### 原版 NAVSIM
- `plan_anchor` 形状：`(20, 8, 2)`，仅含 `(x, y)`。
- `norm_odo` / `denorm_odo` 处理 **3 维**：`(x, y, heading)`。
  - `x`: `2*(x + 1.2)/56.9 - 1`
  - `y`: `2*(y + 20)/46 - 1`
  - `heading`: `2*(heading + 2)/3.9 - 1`
- 模型输出 `poses_reg` 为 `(bs, 20, 8, 3)`，最终 best mode 取 `(x, y, heading)`。

### 当前 CARLA Port
- 当前主线 `plan_anchor` 已升级为 `99x10x2`，对应 `4-0-0-1910-tracked_clusters_anchor.npy`。
- 这份新 anchor 来源于 `4-0-0-1910-tracked_clusters.json` 中的 `99` 个 cluster；每个 cluster 的 `mu` 都是一条拉直后的 `10` 个空间 checkpoint `(x, y)` 轨迹，也就是一个 `20D` 向量。
- 当前模型、loss、训练 target 和推理控制入口已经围绕 `99x10x2` / XY-only 轨迹组织。
- **`model.py` 中的 `norm_odo` / `denorm_odo` 被改成了 2D**，去掉了 heading 的归一化：
  ```python
  def norm_odo(self, odo_info_fut):
      # 仅处理 x, y
      return torch.cat([x_normed, y_normed], dim=-1)  # 2D
  ```
- **`DiffMotionPlanningRefinementModule.plan_reg_branch` 当前输出 `ego_fut_ts * 2`**，即 DD CARLA 主线不再预测 heading。
- **在 Agent 侧**，`diffusiondrive_agent.py` 将模型输出的 `(x, y)` waypoints 送入空间 checkpoint PID。
- **来源文件**：
  - `carla_garage/team_code/diffusiondrive/model.py:432-447`
  - `carla_garage/team_code/diffusiondrive/model.py:501-555` (forward_test)
  - `carla_garage/team_code/diffusiondrive_agent.py:525`

### 差异影响
1. **若加载原版权重**：原版 `norm_odo` 期望 3D 输入，而 CARLA 版 `norm_odo` 只接受 2D；旧 checkpoint 的 trajectory head / anchor 相关权重应预期 shape mismatch。
2. **若合并原版更新**：NAVSIM 原版仍保留 heading 维度和 `20x8x2` anchor 假设，和当前 CARLA 主线会形成明确 merge conflict。
3. **闭环控制**：当前 target / anchor 是空间 checkpoint，不是 fixed-time trajectory；因此推理侧默认空间 PID 不再按 waypoint index 时间间隔估计 desired speed。

### 建议
- **当前决策**：CARLA 侧主线采用 `99x10x2` / XY-only，不再保留 heading 轨迹 head。
- **后续方向**：若需要更稳的闭环速度控制，优先新增 speed head 或单独训练控制相关输出，而不是把当前空间 checkpoint 重新解释成时间轨迹。

---

## 4. `status_feature` 构建

### 原版 NAVSIM
- 10 维向量：
  - `driving_command`：6 维 one-hot（`[0,1,2,3,4,5]` 经 `command_to_one_hot` 映射）
  - `ego_velocity`：2 维（`[v_x, v_y]`）
  - `ego_acceleration`：2 维（`[a_x, a_y]`）

### 当前 CARLA Port
- 当前 `DiffusionDriveAgent` 推理代码路径里，显式构造的是：
  - `command`：CARLA 6 维 one-hot
  - `speed`：1 维原始速度
- 另外，在 `carla_garage/team_code/model.py` 的原 garage 模型中，还存在独立的 `extra_sensors` 机制：
  - 若 `use_velocity=True`，拼接 `velocity_normalization(ego_vel)`，贡献 `1` 维
  - 若 `use_discrete_command=True`，拼接 `command`，贡献 `6` 维
  - 然后把拼接结果送入 `extra_sensor_encoder`
- 因此这里需要明确区分：
  - `DiffusionDriveAgent` 当前使用的是显式 `status_feature = command_one_hot(6)+speed(1)`
  - `extra_sensors` 是另一套可选输入分支，不应被表述成固定的“command 6+1 维”
- **来源文件**：`carla_garage/team_code/diffusiondrive_agent.py:379-390`

### 差异影响
- 形状不再强行对齐 NAVSIM 原版 10 维状态；当前 CARLA 主线以重新训练为前提，采用 7 维 status。
- 之前若把 `command` 和 `extra_sensors` 写成固定的 `6+1`，会误导后续训练设计，因为真实代码里 `extra_sensors` 是可选分支，维度取决于 `use_velocity` 和 `use_discrete_command` 的组合。
- 训练和推理默认都使用当前 command；旧 `commands[-2]` 一拍延迟只作为推理侧 ablation。

---

## 5. 辅助头（Auxiliary Heads）与损失

### 原版 NAVSIM
- 训练时同时监督：
  - `trajectory`（主任务）
  - `agent_states` / `agent_labels`（agent detection，Hungarian matching loss）
  - `bev_semantic_map`（BEV 语义分割，CrossEntropy）
- 损失权重由 `TransfuserConfig` 控制。
- 特征构造：`bev_semantic_map` 的 ground truth 需要 `nuPlan map_api` 和 `Annotations`（`transfuser_features.py:197-289`）。

### 当前 CARLA Port
- **模型推理时仍会输出** `agent_states`、`agent_labels`、`bev_semantic_map`（`model.py:126-136`）。
- **但 Agent 侧完全不使用这些输出**：`diffusiondrive_agent.py` 只读取 `outputs['trajectory']`。
- **损失模块已迁移**：`modules/multimodal_loss.py` 和 `LossComputer` 已存在，但仅在 `model.forward_train()` 中被调用；当前 inference 只走 `forward_test`。

### 差异影响
- **若重新训练，BEV semantic head 的数据来源是空白**：原版依赖 `nuPlan` 的 `SemanticMapLayer`，CARLA 端没有对应实现。如果要开启 `bev_semantic_loss`，需要：
  - 要么关闭该 loss（`config.use_bev_semantic = False`）；
  - 要么在 CARLA 中重建 BEV label 生成逻辑（可用 OpenDRIVE 地图或 CARLA 的语义 LiDAR/ 相机）。
- **Agent detection head 的数据格式不兼容**：`LossComputer` 期望的 `targets["agent_states"]` 格式与 `carla_garage/team_code/data.py` 中解析的 bounding box 格式不同。若要在 CARLA 训练该头，需要重写 target builder 或适配 `data.py` 的 box 逻辑。
- **对 LiDAR 分支的影响**：原版的 LiDAR BEV feature 在训练时还有 BEV semantic / agent detection 等辅助监督约束；当前 CARLA 主线主要训练 trajectory + SpeedHead-v1。这会让 LiDAR 分支更容易学成闭环噪声源，也是 zero-LiDAR 诊断变好的一个合理解释。

### 建议
- 在 CARLA 上重新训练时，**第一阶段建议只保留 trajectory loss**，关闭 BEV semantic 和 agent detection，以降低复杂度。
- 若后续继续保留模型侧 LiDAR，优先考虑两个方向：先做 no-LiDAR full retrain 得到干净对照；再决定是否接入 BEV / agent auxiliary supervision 或 syb-style `regnety_032` backbone。

---

## 6. 配置体系：双 Config 与手动同步

### 原版 NAVSIM
- 单个 `TransfuserConfig` 管理所有模型、传感器、训练参数。

### 当前 CARLA Port
- **双 Config 并行**：
  - `GlobalConfig`：CARLA 运行时配置（传感器位姿、PID 参数、LiDAR buffer 长度等，体量巨大）。
  - `DiffusionDriveConfig`：模型专用配置（backbone 参数、diffusion 参数、anchor 路径等）。
- **手动同步点**（位于 `diffusiondrive_agent.py:setup()`）：
  ```python
  self.dd_config.lidar_min_x = self.config.min_x
  self.dd_config.lidar_max_x = self.config.max_x
  self.dd_config.lidar_min_y = self.config.min_y
  self.dd_config.lidar_max_y = self.config.max_y
  self.dd_config.lidar_resolution_height = self.config.lidar_resolution_height
  self.dd_config.lidar_resolution_width = self.config.lidar_resolution_width
  self.dd_config.lidar_seq_len = self.config.lidar_seq_len
  self.dd_config.use_ground_plane = self.config.use_ground_plane
  ```
- **其余字段**（如 `camera_height`、`pixels_per_meter`、`tf_d_model` 等）在 `DiffusionDriveConfig` 中写死，未与 `GlobalConfig` 同步。

### 差异影响
- 维护成本高，容易遗漏。例如：若修改 `GlobalConfig.cropped_height`，`DiffusionDriveConfig.camera_height` 不会自动跟随，导致图像 resize 后的尺寸与 backbone 期望的 token grid（`img_vert_anchors = camera_height // 32`）错位。

### 建议
- 如 `dd_todo.md` 所述，新增显式 builder，例如 `build_dd_config(global_config, overrides=None)`，将所有映射集中到一处维护。
- 不要尝试让 `DiffusionDriveConfig` 继承 `GlobalConfig`（原因在 `dd_todo.md` 中已有充分说明）。

---

## 7. 代码中的关键妥协（缺少内联注释）

以下代码位置存在"为了跑起来而做的妥协"，但**缺少解释性注释**，容易成为后续 debug 的陷阱：

| 位置 | 妥协内容 | 应补充的注释 |
|---|---|---|
| `diffusiondrive_agent.py:387-388` | `vel = [[speed, 0.0]]`, `acc = [[accel, 0.0]]` | 说明原版 NAVSIM 的 velocity/acceleration 是 2D 向量，CARLA 端仅用单帧差分近似，且 y 轴分量置零。 |
| `diffusiondrive_agent.py:525` | `waypoints = traj[:, :, :2]` | 说明当前模型输出是 3D (x,y,heading)，但 PID 控制器只消费 2D，heading 被显式丢弃。 |
| `diffusiondrive_agent.py:317-318` | JPEG encode/decode | 说明这是复用 `sensor_agent` 的 trick，但对 DiffusionDrive 未必必要，因为原版训练数据没有经过 JPEG。 |
| `model.py:433-447` | `norm_odo` 只处理 2D | 说明这是 CARLA 适配修改：anchor 是 2D，但模型头仍输出 3D，因此此处仅 norm x,y。 |
| `model.py:545-551` (forward_test) | `x_start = poses_reg[...,:2]` 送入 scheduler | 说明 heading 未参与 diffusion 去噪迭代，仅由 MLP 直接预测。 |

---

## 8. 当前已冻结 / 已验证的决策

以下早期阻塞点已被当前 baseline-basic 主线明确：

1. **相机输入策略**：第一阶段采用 CARLA-native 单前视，保留 JPEG artifact，默认不做 ImageNet normalization。
2. **轨迹维度**：正式采用 `99x10x2` anchor；主轨迹只预测 XY，不预测 heading。
3. **轨迹 target 语义**：默认 `spatial_path`，按 `2.5m, 3.5m, ..., 11.5m` 空间距离重采样，不再把 target 当作 fixed-time trajectory。
4. **状态输入**：训练 / 推理共用 `command_one_hot(6)+speed(1)` 的 7 维 `status_feature`。
5. **Stop Sign Controller**：`DiffusionDriveAgent` 保留 actor-based privileged controller，但 baseline-basic / sensor-only 主线默认 `STOP_CONTROL=0`。
6. **闭环定位滤波**：UKF 已加 covariance 正定保护和 measurement reset，避免 `filterpy` 的 `LinAlgError` 直接导致 agent crash。
7. **训练配置**：full baseline-basic 已用 B2D Full scenario-balanced 全量 manifest、4 卡 L40 / DDP、`epochs=100` 跑完。

---

## 9. 结论

- **LiDAR 时序、UKF、RoutePlanner、PID 控制、基础推理链路**已经接通；UKF 数值稳定性问题已加保护。
- **baseline-basic 开环很强但闭环仍弱**：full-scenario open-loop `l1_mean=0.0192`，但 Bench2Drive 220 closed-loop 只有 `DS=44.81`、`RC=79.48`、`NDS=35.52`。
- **当前最大问题已经从训练链路转为闭环行为**：route deviation、blocked、低速和 collisions 是主要失败模式。
- **sensor / control gap 仍需显式处理**：B2D Full raw sensor 与在线 garage sensor suite 的 FOV / pose / LiDAR 外参 gap 仍可能影响闭环；空间 checkpoint target 和 PID 控制之间也需要继续调参或引入显式 speed/control head。
- 建议的推进顺序：
  1. 对 220 条闭环结果按 route deviation / blocked / timeout / collision 做失败聚类。
  2. 做 command delay、空间 PID、stuck / creep / safety box 的闭环 A/B。
  3. 验证 sensor contract gap，必要时做推理侧对齐或 finetune。
  4. 再考虑显式 speed/control head、辅助头、sensor-only stop sign 和持续学习方法。
