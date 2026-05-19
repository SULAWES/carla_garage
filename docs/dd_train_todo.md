# DiffusionDrive CARLA Training TODO

本文档记录面向后续 CARLA 训练的事项。格式参考 `dd_todo.md`，但关注点从“当前推理 agent 还缺什么”切换为“训练和训练后闭环需要先定义和验证什么”。

**更新日期**: 2026-05-19
**判断基线**:

- 当前计划是在 CARLA 上重新训练 DiffusionDrive
- 因此训练侧优先级高于继续对齐 navsim checkpoint 语义
- 训练文档同时覆盖训练前定义、训练中实验项、训练后闭环验证项
- 需要区分 `status_feature` 和 `extra_sensors` 两套输入机制
- 当前 `DiffusionDriveAgent` 显式构造的是 `status_feature = command(6) + velocity(2) + acceleration(2)`
- `carla_garage/team_code/model.py` 中的 `extra_sensors` 是可选分支：按配置拼接 `velocity(1)` 与 / 或 `discrete_command(6)`，再经 `extra_sensor_encoder` 编码
- 当前新提取的聚类 anchor 文件 `4-0-0-1910-tracked_clusters_anchor.npy` 语义为 `99x10x2`
- 其来源 JSON 中每个 cluster 的 `mu` 都是一条拉直后的 `10` 个路点 `(x, y)` 轨迹，即一个 `20D` 向量

---

## TODO List

### P0 级别（训练前必须明确）

- [ ] **训练输入定义冻结**
  - [x] 明确相机输入方案：第一阶段采用 CARLA-native 单前视，不采用 NAVSIM 三相机拼接
  - [x] 明确图像裁剪 / resize / normalize 方案：`512x1024 -> crop_array 384x1024 -> resize 256x1024 -> [0,1]`，默认无 ImageNet normalization
  - [x] 明确 LiDAR 输入通道定义：沿用 garage histogram，默认 `use_ground_plane=False`，每帧 `1` 个 above-split 通道；多帧时按 `lidar_seq_len` 拼接
  - [ ] 明确 `status_feature` 的最终定义
  - [ ] 明确是否保留独立 `extra_sensors` 分支，以及其输入组成

- [ ] **训练标签与轨迹定义冻结**
  - [ ] 修正 raw Bench2Drive `anno["theta"]` 到 CARLA ego yaw 的转换；当前 target builder 未做 `preprocess_compass()` 等价处理
  - [ ] 用 `bounding_boxes[0]["world2ego"]` / ego pose 矩阵复核 `ego_relative_xy()`，并落盘 target 可视化/统计
  - [ ] 明确未来轨迹采样方式
  - [ ] 明确 raw Bench2Drive frame 间隔、`future_stride`、`trajectory_sampling.interval_length`、anchor 时间间隔和 PID waypoint 时间间隔的一致关系
  - [ ] 明确 anchor 生成方式与数组形状
  - [x] 校验最新 `99x10x2` anchor 与源 JSON 一致，并记录直接升级路线，见 `diffusiondrive_anchor_adaptation.md`
  - [x] 明确 heading 不由轨迹 head 预测，主轨迹接口固定为 XY
  - [x] 明确采用新聚类 anchor：`99` 个 mode，每个 mode 对应 `10x2` 轨迹点

- [ ] **数据集生成规范**
  - [ ] 明确需要采哪些传感器
  - [ ] 明确 raw Bench2Drive 传感器几何是否作为训练主线，还是重新采集 garage sensor suite 数据
  - [ ] 明确 camera FOV / pose、LiDAR pose / yaw / range 与在线 agent 的一致性要求
  - [ ] 明确数据保存频率和时序长度
  - [ ] 明确训练 / 验证 / 测试切分方式

- [x] **最小训练入口**
  - [x] 新增 `team_code/train_diffusiondrive.py`
  - [x] 新增 raw Bench2Drive 数据集 helper `team_code/diffusiondrive/carla_native_dataset.py`
  - [x] 支持 trajectory loss-only 单机训练
  - [x] 用 Bench2Drive mini 跑通 1-step training smoke

- [x] **第一版训练配置落盘**
  - [x] 训练入口支持 optimizer / scheduler / warmup 配置
  - [x] 训练入口支持 trajectory loss weights、focal alpha/gamma 和 diffusion timestep 配置
  - [x] 训练入口支持 validation split 参数、checkpoint resume 和 `latest.pth`
  - [x] 输出目录落盘 `training_config.json`，记录 CLI、DiffusionDrive config、数据 split、预处理和 status feature schema
  - [ ] 后续仍需把实验配置从 CLI-only 进一步整理成可复用 config 文件或 launch preset

### P1 级别（直接影响训练效果）

- [ ] **`status_feature` 重定义**
  - [ ] 明确 command 维度和语义
  - [ ] 明确训练使用当前 command 还是推理侧延迟一拍的 `commands[-2]`
  - [ ] 明确 velocity / acceleration 的取值方式
  - [ ] 统一 acceleration：训练当前用 IMU `acceleration[0]`，推理当前用 speed finite difference
  - [ ] 明确是否做归一化
  - [ ] 明确 `status_feature` 和 `extra_sensors` 是否并存，还是合并为单一状态输入

- [ ] **图像预处理方案验证**
  - [x] 记录当前推理侧图像预处理事实，见 `diffusiondrive_input_preprocessing.md`
  - [x] 冻结第一阶段 CARLA-native 图像预处理主线：单前视、去底部裁剪、`[0,1]`、默认 JPEG artifact、无 ImageNet normalization
  - [ ] 验证 raw Bench2Drive `1600x900, fov=70, camera x=0.8,z=1.6` resize 到 garage `512x1024, fov=110, camera x=-1.5,z=2.0` 是否可接受
  - [ ] 评估 JPEG artifact 是否需要保留
  - [ ] 评估是否需要 ImageNet mean/std normalization
  - [ ] 评估单前视与多相机的收益差异

- [ ] **LiDAR 表征方案验证**
  - [ ] 验证 raw Bench2Drive LiDAR `x=-0.39,z=1.84,yaw=0,range=85` 与 garage 在线 LiDAR `x=0,z=2.5,yaw=-90` 的训练/推理 gap
  - [ ] 明确是否始终使用多帧 LiDAR
  - [ ] 明确 histogram 参数是否沿用当前 garage 设置
  - [ ] 验证时序 realign 对训练标签的一致性

- [ ] **轨迹表示统一**
  - [x] 明确最终采用 `99x10x2` 轨迹表示
  - [x] 训练与推理统一使用同一套 anchor / 归一化 / 输出语义
  - [x] 明确不在轨迹 head 中保留 heading；如后续需要 heading-aware 控制，另行设计监督与接口
  - [x] 明确新聚类结果直接作为 `99x10x2` 正式 anchor，不再重采样 / 截断为兼容格式

- [ ] **loss / optimization 定标**
  - [ ] 99 modes 下重新评估 focal alpha/gamma、`trajectory_cls_weight`、`trajectory_reg_weight`
  - [ ] 评估外层 `trajectory_weight` 接入后总梯度尺度，记录 grad norm 和 loss breakdown
  - [ ] 明确 image encoder 是否使用更小 LR / freeze BN / freeze backbone warmup
  - [ ] 小 batch 训练前确认 BatchNorm 策略：frozen BN、SyncBN、或足够大的 effective batch

### P2 级别（训练后闭环增强）

- [ ] **辅助头训练策略**
  - [ ] 评估是否保留 `agent_states / agent_labels / bev_semantic_map`
  - [ ] 明确多任务损失是否继续保留

- [ ] **评测与控制闭环设计**
  - [ ] 明确训练完成后继续用 waypoint PID 还是改控制器
  - [ ] 若轨迹间隔采用 0.5s 或 1.0s，重写 `_control_pid()` 中 desired speed 的 waypoint 索引/时间差计算
  - [ ] 明确训练指标与 leaderboard 指标的对齐方式

- [ ] **训练工程可复现性**
  - [ ] `--resume-file` 当前只从 checkpoint 的下一个 epoch 继续，不恢复 dataloader epoch 内位置；长训前决定是否需要精确断点恢复
  - [ ] 明确 full training 的 AMP / DDP / gradient accumulation 策略
  - [ ] 给 full 数据 smoke 增加吞吐量、显存、grad norm、target 分布统计

- [ ] **闭环调参与可观测性**
  - [ ] 调 `safety_box_*` 阈值，适配训练后模型的闭环行为
  - [ ] 评估 `stuck_threshold` / `creep_duration` 是否需要联调
  - [ ] 收敛 creep / safety box 相关日志输出，避免长跑日志过噪

---

## 当前事实

### 已完成部分

#### 1. 推理侧已具备可运行闭环

当前推理 agent 已经可以：

- 接收 CARLA 传感器
- 构建 DiffusionDrive 输入
- 输出轨迹
- 通过 PID 执行

这意味着训练侧后续不需要再从零定义推理入口，而是可以直接围绕已有 agent 迭代。

#### 2. LiDAR 时序链路已在推理侧接入

当前推理侧已经接入：

- 半帧 LiDAR 拼接
- `lidar_buffer`
- 历史 LiDAR realign
- stuck detection
- safety box

这为后续训练时是否采用多帧 LiDAR，以及训练后如何做闭环调参，提供了直接参考。

#### 3. 新聚类 anchor 已提取，但尚未接入当前链路

当前仓库根目录下已存在：

- `4-0-0-1910-tracked_clusters.json`
- `4-0-0-1910-tracked_clusters_anchor.npy`

其中：

- cluster 数为 `99`
- 每个 cluster 的 `mu` 长度为 `20`
- 语义上等价于一条拉直后的 `10` 个路点 `(x, y)` 轨迹
- 提取后的 anchor 文件 shape 为 `99x10x2`

这说明新的聚类结果已经可供训练侧使用；当前决策是直接将推理 / 训练接口升级到 `99x10x2`，而不是生成兼容版 `20x8x2`。

#### 4. 2026-05-19 训练风险扫描结论

本次扫描没有改代码，只对现有训练链路做静态检查和 mini 数据统计。结论是：full training 前不能只跑 smoke，需要先修正 target 坐标和时间语义，否则长训结果不可用。

**最高优先级问题**：

- `carla_native_dataset.py` 当前直接用 `anno["theta"]` 构造 `ego_relative_xy()`。但在线推理侧会先对 IMU compass 做 `preprocess_compass()`，等价于 `theta - pi/2`。在 Bench2Drive mini 上，当前 target 与 anno 中 ego `world2ego` 矩阵计算结果的误差约为：
  - mean error: `17.05m`
  - p95 error: `56.50m`
  - 若先做 `theta - pi/2`，误差降到 mean `0.87m`、p95 `1.70m`
- raw Bench2Drive frame 从 bbox 位移 / speed 估计约为 `0.1s` 间隔。当前 `future_stride=10` 更接近 `1.0s` 采样，而 DiffusionDrive config / anchor / 控制器仍按另一套 waypoint 时间语义工作。
- raw Bench2Drive 传感器与 garage 在线 agent 传感器几何差异很大：
  - Bench2Drive front camera: `1600x900, fov=70, x=0.8,z=1.6`
  - garage front camera: `1024x512, fov=110, x=-1.5,z=2.0`
  - Bench2Drive LiDAR: `x=-0.39,z=1.84,yaw=0,range=85`
  - garage LiDAR: `x=0,z=2.5,yaw=-90`
- `status_feature` 中 acceleration 当前训练/推理不一致。训练 helper 用 IMU `acceleration[0]`，推理 agent 用 speed finite difference；mini 上二者相关性约 `0.23`。
- command 当前训练用 `command_far`，推理用 `commands[-2]`，存在一拍时序差异。

**直接建议**：

1. 先修 target builder，至少让 `ego_relative_xy()` 与 `world2ego` 矩阵在 mini/full smoke 上对齐。
2. 冻结轨迹采样间隔：明确 `future_stride`、anchor interval、`trajectory_sampling.interval_length` 和 PID waypoint interval。
3. 决定训练主线到底是 raw Bench2Drive 传感器几何，还是重采 garage sensor suite 数据；不要让 resize/crop 掩盖 FOV/pose gap。
4. 再跑 full 数据小 smoke 和吞吐量测试。

---

## P0 级别问题（训练前必须明确）

### 1. `status_feature` 需要重新定义并冻结

**问题描述**：

既然后续计划在 CARLA 上重新训练，那么当前推理侧的 `status_feature` 不应被视为最终答案，而应在训练前重新定义。

**至少要明确**：

- command 用几维
- command 的类别语义是什么
- velocity / acceleration 是否保留 2D
- 是否加入更多状态量
- 是否做归一化
- 是否保留独立 `extra_sensors` 分支
- 若保留，`extra_sensors` 中具体拼哪些量：`velocity(1)`、`discrete_command(6)`，还是别的输入

**当前代码事实**：

- `DiffusionDriveAgent` 当前显式构造的是 `status_feature = command(6) + velocity(2) + acceleration(2)`
- `carla_garage/team_code/model.py` 中的 `extra_sensors` 则是另一套可选输入分支：
  - 若 `use_velocity=True`，加入 `velocity_normalization(ego_vel)`，贡献 `1` 维
  - 若 `use_discrete_command=True`，加入 `command`，贡献 `6` 维
  - 两者拼接后送入 `extra_sensor_encoder`
- 因此这里真正需要冻结的是：训练侧到底采用哪一套输入接口，而不是把两者误写成固定的 `6+1`

**结论**：

这项不该继续围绕 navsim checkpoint 修补，而应该直接按 CARLA 训练目标重新设计。

### 2. 相机输入方案需要冻结

**问题描述**：

当前推理侧使用单前视相机，但原版 navsim 更接近多相机拼接输入。

**已冻结的第一阶段方案**：

- 继续用 CARLA-native 单前视相机。
- 不按 NAVSIM 三相机拼接作为当前主线。
- 在线传感器保持 `512x1024`，裁剪为 `384x1024`，再 resize 到模型输入 `256x1024`。
- 训练侧必须与推理侧保持同一套裁剪、resize 和 normalization 语义。
- 多相机拼接保留为后续单独实验，不能混入第一阶段训练基线。

### 3. 数据规范需要尽早固定

**问题描述**：

如果训练数据的采集规范和推理时假设反复变化，后续实验会很难归因。

**需要明确**：

- 数据保存频率
- LiDAR 时序长度
- 相机布局
- 轨迹标签定义
- anchor 生成方式
- 是否围绕新的 `99x10x2` 聚类 anchor 组织训练数据与标签

---

## P1 级别问题（影响训练效果）

### 4. 图像预处理与 LiDAR 表征要做小规模消融

**问题描述**：

在 CARLA 上重新训练前，最好先把最可能影响效果的输入因素拆开验证。

**建议优先做的消融**：

1. 单前视 vs 多相机
2. 当前 resize / crop 方案 vs 备选方案
3. 单帧 LiDAR vs 多帧 LiDAR
4. 是否保留 JPEG artifact

### 5. 轨迹表示需要在训练前统一

**问题描述**：

当前实现已经将主轨迹接口统一为 `99x10x2`。对于继续训练来说，训练标签也应按 `10x2` XY 轨迹组织；若数据构建阶段暂时保留 heading，loss 只会使用前两维 XY。

另外，新提取出的聚类 anchor `4-0-0-1910-tracked_clusters_anchor.npy` 目前是 `99x10x2`。它对应的原始 JSON 里，每个 cluster 的 `mu` 都是一条拉直后的 `10` 点轨迹，也就是一个 `20D` 向量；这已作为新的正式 anchor 接口。

**已明确**：

- 训练目标采用 2D XY 轨迹
- anchor、归一化、loss、输出头和控制入口统一为 `99x10x2`
- 控制器当前只消费 XY waypoint，不利用 heading
- 新聚类 anchor 直接作为正式训练 anchor
- `trajectory_sampling.num_poses`、loss、控制链路和 checkpoint 策略已按 `99x10x2` 同步升级
- 当前不再生成兼容版 anchor；后续训练直接围绕 `99x10x2` 组织

### 6. 训练后闭环参数需要重新调

**问题描述**：

即使模型重新训练后效果更好，闭环侧也未必能直接沿用当前阈值和日志策略。

**需要重点关注**：

- `safety_box_*` 阈值需要按新模型行为重新调
- `stuck_threshold` / `creep_duration` 需要和模型停车/起步风格联调
- 当前 creep / safety box 相关 `print` 适合调试，但长时间评测时可能日志过噪
- 如果目标评测环境是 Bench2Drive，还要额外考虑 `AgentBlockedTest` 更早触发的问题

**建议**：

- 训练后单独做闭环调参阶段，不要把训练提升和阈值变化混在一起评估
- 如需长时间批量评测，考虑将这类日志改为可配置 debug 开关或限频输出
- 如果跑 Bench2Drive，优先确认 stuck recovery 的触发时机是否明显早于 blocked timeout
- 不要直接沿用当前 `stuck_threshold = 1100` 的默认值去跑 Bench2Drive 闭环评测

---

## P2 级别问题（中长期增强）

### 7. 辅助头和多任务训练是否保留

**问题描述**：

当前模型还带有：

- `agent_states`
- `agent_labels`
- `bev_semantic_map`

**需要判断**：

- 这些头是保留作为正则与辅助监督
- 还是在 CARLA 训练中裁掉以简化训练目标

### 8. 评测闭环需要提前考虑

**问题描述**：

训练目标如果和最后上线的执行链路不一致，效果容易失真。

**需要明确**：

- 最终是否仍然通过 waypoint PID 执行
- 如果是，训练轨迹和控制器参数要如何协同调
- 是否要引入辅助头帮助安全或可视化

---

## 建议的推进顺序

1. 冻结训练输入定义
2. 冻结轨迹标签与 anchor 方案
3. 固定数据采集规范
4. 做最小规模输入消融
5. 开始正式训练
6. 训练后单独调 `safety_box_*`、stuck recovery 和日志策略
7. 最后再考虑控制器与辅助头增强

---

## 相关文档

- `docs/dd_todo.md`: 推理侧后续事项
- `docs/diffusiondrive_agent_explained.md`: 当前 agent 的实现说明
- `docs/diffusiondrive_run.md`: 通用运行方法
- `docs/run.md`: 机器/环境相关运行备忘
