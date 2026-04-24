# DiffusionDrive: NAVSIM 原版与 CARLA Port 的差异清单

**kimi k2.6-code-preview挑的刺。**

> **目的**：建立并固化对"原版 NAVSIM 假设"与"当前 CARLA port 实现"之间差异的精确认知，防止后续迭代中遗漏关键分布 mismatch。
> **基线代码**：
> - 原版：`./DiffusionDrive/navsim/agents/diffusiondrive/` (commit 以当前工作区为准)
> - CARLA Port：`./carla_garage/team_code/diffusiondrive_agent.py`、`./carla_garage/team_code/diffusiondrive/`

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
- **BEV 生成**：直接调用 histogram splat，逻辑与 `carla_garage` 基本一致。

### 当前 CARLA Port
- **半帧拼接**：每步只收到半帧 LiDAR，用 `lidar_last` 与当前帧拼接成完整扫描。
- **多帧 Buffer**：`lidar_buffer` 长度为 `lidar_seq_len * data_save_freq`；未满时持续刹车等待。
- **历史帧 Realign**：若 `config.realign_lidar=True` 且 `lidar_seq_len>1`，使用 `align_lidar()` 将历史帧点云变换到当前 ego 坐标系。
- **来源文件**：`carla_garage/team_code/diffusiondrive_agent.py:444-510`

### 差异影响
- **CARLA 端反而更完整**：LiDAR 时序链路已经基本对齐甚至超出原版。只要坐标变换（`lidar_to_ego_coordinate` + `align_lidar`）与 BEV 坐标系假设一致，这一块不是瓶颈。
- **风险点**：`lidar_to_ego_coordinate()` 将点云旋转平移到 ego 坐标系后，`data.lidar_to_histogram_features()` 中的 `splat_points` 有 `overhead_splat.T`（x/y 轴 swap）。需要确认这个 transpose 与 `align_lidar` 的组合在时序帧上不会引入方向错位（参见第 7 节）。

---

## 3. 轨迹表示：处于 "2D anchor + 3D 输出" 的半对齐状态

### 原版 NAVSIM
- `plan_anchor` 形状：`(20, 8, 2)`，仅含 `(x, y)`。
- `norm_odo` / `denorm_odo` 处理 **3 维**：`(x, y, heading)`。
  - `x`: `2*(x + 1.2)/56.9 - 1`
  - `y`: `2*(y + 20)/46 - 1`
  - `heading`: `2*(heading + 2)/3.9 - 1`
- 模型输出 `poses_reg` 为 `(bs, 20, 8, 3)`，最终 best mode 取 `(x, y, heading)`。

### 当前 CARLA Port
- `plan_anchor` 同样是 `(20, 8, 2)`。
- 仓库中另外已经提取出一份新的聚类 anchor：`4-0-0-1910-tracked_clusters_anchor.npy`，其 shape 为 `99x10x2`。
- 这份新 anchor 来源于 `4-0-0-1910-tracked_clusters.json` 中的 `99` 个 cluster；每个 cluster 的 `mu` 都是一条拉直后的 `10` 个路点 `(x, y)` 轨迹，也就是一个 `20D` 向量。
- 由于当前运行链路仍按 `20x8x2` 组织 mode 数和时间步长度，这份 `99x10x2` anchor 目前不能直接替换现有 `plan_anchor.npy`。
- **但 `model.py` 中的 `norm_odo` / `denorm_odo` 被改成了 2D**，去掉了 heading 的归一化：
  ```python
  def norm_odo(self, odo_info_fut):
      # 仅处理 x, y
      return torch.cat([x_normed, y_normed], dim=-1)  # 2D
  ```
- **然而 `DiffMotionPlanningRefinementModule.plan_reg_branch` 仍然输出 `ego_fut_ts * 3`**，即 `poses_reg` 仍是 `(bs, 20, 8, 3)`。
- **在 Agent 侧**，`diffusiondrive_agent.py` 只取 `traj[:, :, :2]` 送入 PID 控制器，heading 维度被直接丢弃。
- **来源文件**：
  - `carla_garage/team_code/diffusiondrive/model.py:432-447`
  - `carla_garage/team_code/diffusiondrive/model.py:501-555` (forward_test)
  - `carla_garage/team_code/diffusiondrive_agent.py:525`

### 差异影响
1. **Diffusion Scheduler 的行为不一致**：在 `forward_test` 中，`x_start = poses_reg[...,:2]` 被 `norm_odo` 后送入 `DDIMScheduler.step()`。但模型内部仍然计算并输出 heading，这个 heading 维度既没有被 `norm_odo` 约束，也没有参与 diffusion 的去噪迭代。它只是在 `CustomTransformerDecoderLayer` 中通过 `tanh() * np.pi` 截断。
2. **若加载原版权重**：原版 `norm_odo` 期望 3D 输入，而 CARLA 版 `norm_odo` 只接受 2D。虽然当前 CARLA `model.py` 的 `norm_odo` 被手动改了，但如果未来需要合并原版更新，这个差异会成为 merge conflict 的隐患。
3. **若重新训练**：必须统一决定采用 `8×2`（仅 x,y）还是 `8×3`（x,y,heading）。
   - 若采用 `8×2`：应将 `plan_reg_branch` 的最后一层改为 `ego_fut_ts * 2`，并同步修改 `CustomTransformerDecoderLayer` 和 `LossComputer`。
   - 若采用 `8×3`：应恢复 `norm_odo` 的 heading 分支，并在 Agent 侧利用 heading 做更贴合轨迹朝向的控制。
4. **若接入新聚类 anchor**：还会额外引入 `20 -> 99` 个 mode 和 `8 -> 10` 个 pose 两处接口变化，影响轨迹头、loss、checkpoint 对齐和控制链路。

### 建议
- **立即明确决策**：在 CARLA 侧最终采用 `8x2` 还是 `8x3`，并写成配置项。当前不应长期停留在“半对齐”状态。
- **单独明确新 anchor 策略**：是先离线重采样得到兼容版 `20x8x2`，还是系统性升级到 `99x10x2`。

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
  - `velocity`：2 维 `[[speed, 0.0]]`
  - `acceleration`：2 维 `[[accel, 0.0]]`，其中 `accel = (speed - prev_speed) / carla_frame_rate`
- 另外，在 `carla_garage/team_code/model.py` 的原 garage 模型中，还存在独立的 `extra_sensors` 机制：
  - 若 `use_velocity=True`，拼接 `velocity_normalization(ego_vel)`，贡献 `1` 维
  - 若 `use_discrete_command=True`，拼接 `command`，贡献 `6` 维
  - 然后把拼接结果送入 `extra_sensor_encoder`
- 因此这里需要明确区分：
  - `DiffusionDriveAgent` 当前使用的是显式 `status_feature`
  - `extra_sensors` 是另一套可选输入分支，不应被表述成固定的“command 6+1 维”
- **来源文件**：`carla_garage/team_code/diffusiondrive_agent.py:379-390`

### 差异影响
- 形状一致，但 **CARLA 端的加速度是单帧数值差分**，原版 NAVSIM 的加速度来自开环数据集的原始记录（通常是车辆动力学模型或 IMU 直接输出）。这个差异在重新训练时是否重要，尚未评估。
- 之前若把 `command` 和 `extra_sensors` 写成固定的 `6+1`，会误导后续训练设计，因为真实代码里 `extra_sensors` 是可选分支，维度取决于 `use_velocity` 和 `use_discrete_command` 的组合。
- `dd_todo.md` 已明确将 `status_feature` 对齐列为"不作为近期 TODO"（因为计划重新训练）。这是合理决策，但应在训练准备阶段重新评估。

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

### 建议
- 在 CARLA 上重新训练时，**第一阶段建议只保留 trajectory loss**，关闭 BEV semantic 和 agent detection，以降低复杂度。
- 若后续需要利用辅助头进行可视化或 safety check，再逐步接入。

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

## 8. 待明确的问题清单（决策阻塞点）

在进入 CARLA 训练或系统调参之前，以下问题必须有明确答案：

1. **相机输入策略**：单前视？三相机拼接？是否保留 JPEG？
2. **轨迹维度**：统一采用 `8x2` 还是 `8x3`？
3. **训练时辅助头**：是否关闭 `bev_semantic` 和 `agent_detection`？若开启，label 如何生成？
4. **LiDAR / BEV 对齐验证**：`lidar_to_ego_coordinate()` 与 `align_lidar()` 的组合在时序帧上是否与 BEV 坐标系完全一致？建议增加可视化验证。
5. **Diffusion 超参数**：`trunc_timesteps=8`、`step_num=2` 等默认设置是否适合 CARLA 闭环分布？是否应参数化以便 grid search？
6. **Stop Sign Controller**：是否从 `sensor_agent.py` 迁移到 `DiffusionDriveAgent`？
7. **新聚类 anchor**：是否正式切到 `99x10x2`，以及如何同步处理 mode 数、时间步长度和 checkpoint 兼容问题？

---

## 9. 结论

- **LiDAR 时序、UKF、RoutePlanner、PID 控制、基础推理链路**已经接通，不是当前主要瓶颈。
- **最大的输入分布 mismatch 是相机**（单前视 vs 三相机拼接），但 `dd_todo.md` 中未明确记录。
- **最大的模型内部债务是轨迹表示的半对齐**（2D `norm_odo` + 3D 输出），这会影响 diffusion 行为的可预测性，也阻碍了训练设计的统一。
- 建议的推进顺序：
  1. 明确相机输入策略和轨迹维度决策。
  2. 验证 LiDAR / BEV 对齐。
  3. 收敛 Config 映射并补齐关键注释。
  4. 规划 CARLA 训练时的辅助头取舍。
  5. 再考虑 ensemble、stop sign controller 等增强项。
