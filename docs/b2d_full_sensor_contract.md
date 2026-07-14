# Bench2Drive Full Sensor Contract

本文档记录当前 DiffusionDrive 训练主线使用的 Bench2Drive Full 数据约定，以及它和在线 `DiffusionDriveAgent` sensor suite 之间仍需显式处理的 gap。

## 当前决策

- 后续训练和模型修改以 Bench2Drive Full 为主要训练数据分布。
- 当前 raw Bench2Drive / B2D Full loader 不是只用于 smoke 或预训练；它是当前 baseline-basic 和 baseline-condition-v1 的主训练入口。
- 因此训练配置、文档和实验命名应围绕 `b2d_full_raw` 语义记录，而不是默认假设它已经和在线 garage sensor suite 完全一致。

## B2D Full Raw 输入事实

当前数据 helper 支持两种 Bench2Drive route 目录结构。

早期 / mini smoke 格式：

- `camera/rgb_front/*.jpg`
- `lidar/*.laz`
- `anno/*.json.gz`

B2D Full 原生格式：

- `rgb/*.jpg`
- `lidar/*.laz`
- `measurements/*.json.gz`
- `records.json.gz` / `results.json.gz` 是 route 级元数据，当前不作为逐帧训练标注读取

远端 Full 数据通常是 scenario / route 两层结构，例如 `carla_dataset/Accident/Town13_.../`；训练时应使用 `--route-glob "*/*"`。

当前已知 raw sensor 几何记录：

- front camera: `1600x900, fov=70, x=0.8, y=0.0, z=1.6`
- LiDAR: `x=-0.39, y=0.0, z=1.84, yaw=0, range=85`
- LiDAR collection attributes: `channels=64, points_per_second=600000, upper_fov=10, lower_fov=-30, dropoff_general_rate=0, dropoff_intensity_limit=0, dropoff_zero_intensity=0`

注意：远端当前用于 full training 的 B2D Full 图像文件实际观测为 `512x1024`，不是文档早期假设的 `1600x900` tensor。`1600x900, fov=70` 仍应作为原始采集 provenance 记录，但训练脚本的 `--b2d-source-image-height/width` 默认已修正为 `512/1024`。

当前 dataset 会把 raw front image 转成模型输入：

- source image size: `512x1024`
- resize 到在线 garage sensor size: `512x1024`
- `crop_array(): 512x1024 -> 384x1024`
- 默认模型输入: `384x1024`
- 数值范围: `[0,1]`，随后可选 ImageNet mean/std normalization
- 默认保留 JPEG artifact
- 训练侧 `--image-normalization none|imagenet`，推理侧 `DIFFUSIONDRIVE_IMAGE_NORMALIZATION=none|imagenet`

## 在线 Garage Sensor Suite

当前 `DiffusionDriveAgent` 在线推理仍使用 garage sensor contract：

- front camera: `512x1024, fov=110, x=-1.5, y=0.0, z=2.0`
- LiDAR: `x=0.0, y=0.0, z=2.5, yaw=-90, range=85`
- LiDAR wrapper attributes: `channels=64, points_per_second=600000, upper_fov=10, lower_fov=-30, dropoff_general_rate=0.45, dropoff_intensity_limit=0.8, dropoff_zero_intensity=0.4`
- LiDAR histogram: `256x256`, `pixels_per_meter=4.0`
- 继续复用半帧拼接、多帧 buffer、realign、stuck / safety box 和 PID 控制。

## LiDAR 表示与 Backbone

当前 DiffusionDrive LiDAR 表示更接近原版 NAVSIM，而不是 syb 的旧 garage 模型：

- raw LiDAR / `.laz` 点云会先转成 BEV histogram。
- 默认 BEV 范围为 `[-32m, 32m] x [-32m, 32m]`，`pixels_per_meter=4`，分辨率 `256x256`。
- 默认 `use_ground_plane=False`，模型侧通常只收到一层 above-ground BEV feature。
- `lidar_seq_len=1`，所以当前模型不消费更久历史 BEV；半帧拼接只是为了得到当前完整扫描。
- 当前 CARLA DiffusionDrive 的 `DiffusionDriveConfig.lidar_architecture` 默认是 `resnet34`。syb 常用的 `regnety_032` 是另一个 `timm` 2D CNN backbone，不是 LiDAR 专用表示；切换它需要 full retrain。

需要区分两条 LiDAR 使用路径：

- 模型侧 LiDAR：BEV histogram 作为 `lidar_feature` 输入 DiffusionDrive backbone。
- runtime LiDAR：raw / buffered LiDAR 用于 safety-box、stuck / creep 等规则逻辑。

`DIFFUSIONDRIVE_ZERO_LIDAR=1` 只置零模型侧 `lidar_feature`，不关闭 runtime safety-box raw LiDAR。因此 zero-LiDAR 诊断变好时，优先说明模型侧 BEV LiDAR 分支存在 train-vs-online contract gap、fusion 噪声或监督不足风险，不能直接推出 safety-box 应该关闭。项目最终路线仍应使用 LiDAR；zero/no-LiDAR 只作为诊断上界和归因工具。

当前有两个互补诊断入口：

- `tools/inspect_diffusiondrive_lidar_contract.py`：从 B2D `.laz` 读取保存的 full scan，用共享 schema 生成 `records.jsonl`、`summary.json` 和受上限约束的 NPZ dump。
- `DIFFUSIONDRIVE_DEBUG_LIDAR_CONTRACT=1`：在线记录 `current_half`、`previous_half_aligned`、`full_scan`、`model_input` 四个 stage；`DIFFUSIONDRIVE_LIDAR_DEBUG_INTERVAL` 控制统计频率，`DIFFUSIONDRIVE_LIDAR_DUMP_INTERVAL` 和 `DIFFUSIONDRIVE_LIDAR_DUMP_MAX` 控制数组 dump。

二者统计 raw / finite / in-BEV / below-split / above-split / model point count，z 分位数，front/back/left/right occupancy，以及 BEV nonzero / mean / max / saturation 和逐通道汇总。先比较分布，再决定是否修改坐标、过滤或融合；不要只凭单张 BEV 图判断 contract。

## 已知 Gap

B2D Full raw sensor 和在线 garage sensor suite 不完全一致：

- camera FOV 不同：`70` vs `110`
- camera pose 不同：B2D raw 在车前方，garage 在线相机在 `x=-1.5,z=2.0`
- raw provenance / source tensor 记录需要区分：采集几何记录为 `1600x900`，当前训练文件实际为 `512x1024`
- LiDAR nominal range / channels / points-per-second / vertical FOV 相同，但传感器 pose / yaw、ray origin 和 dropoff 参数不同；尤其 collection 禁用 dropoff，而在线 wrapper 使用非零 dropoff，可能造成显著点密度差异
- 当前 resize / crop 只能对齐 tensor shape，不能消除真实几何 gap

当前结论不是“raw B2D 已经等价于在线 sensor suite”，而是：

- 训练主线以 B2D Full raw 分布为准。
- 在线闭环 sensor gap 作为后续推理一致性 / domain gap 问题显式记录和验证。

## 时间语义

当前 B2D Full 相邻 annotation frame interval 估计约为 `0.1s`。这个时间间隔现在只用于解释 future ego path 搜索窗口和 legacy time target，不再定义默认 trajectory target 的点间隔。

- `dataset_mode`: `b2d_full_raw`
- `assumed_frame_interval_seconds`: `0.1`
- `target_mode`: `spatial_path`
- `future_stride_frames`: 默认 `10`，仅用于 `target_mode=future_ego_time` legacy 路径
- `trajectory_sampling_interval_seconds`: 来自 `DiffusionDriveConfig.trajectory_sampling.interval_length`
- `anchor_interval_seconds_assumption`: `null`

## Trajectory Target 与 Anchor 语义

当前采用方案 B：训练 target 与 `99x10x2` anchor 统一为空间 checkpoint 语义。

anchor 估计语义：

- 第一个点距离 ego 约 `2.5m`
- 后续相邻点约 `1.0m`
- 10 个点覆盖约 `2.5m -> 11.5m`
- 不是固定 `0.5s` 或 `1.0s` 的 time-based future trajectory

当前 dataset target 构造：

- 样本发现阶段要求 `spatial_path` 至少存在下一帧 annotation，避免 route 末尾完全无未来 ego path 时纯靠 command 生成训练标签
- 从当前 frame 后续 ego vehicle world location 构成 future ego path；Full 原生 `measurements` 使用 `pos_global` 和 `ego_matrix`，其中 `ego_matrix` 会取逆得到 `world2ego`
- 将 future path 转到当前 ego frame
- 按距离重采样为 `2.5m, 3.5m, ..., 11.5m`
- 若 future path 不足，沿最后路径方向外推；若几乎无路径方向，则用 `command_far / command_near` 方向外推

注意：`DiffusionDriveConfig.trajectory_sampling.interval_length` 目前仍保留为兼容 DiffusionDrive config 的字段，不应被解释为当前 target / anchor 的真实时间间隔。后续如需更干净的配置体系，应将空间 checkpoint 采样从 `TrajectorySampling` 中拆出来。

## Training Config 记录

`team_code/train_diffusiondrive.py` 会在 `training_config.json` 中记录：

- `data.dataset_mode`
- `target`
- `time_semantics`
- `anchor.path / shape / num_modes`
- `sensor_contract`
- `preprocessing.source_image_size / online_sensor_size / model_image_size`
- `status_feature`
- `route_condition_token`
- `speed_head`

这些字段用于保证 checkpoint 和实验结果可以追溯到当时的 sensor / time / anchor 假设。

## 后续建议

1. full baseline-basic 已完成，B2D Full raw + scenario-balanced manifest 能支持全量训练，且 open-loop all-scenarios `l1_mean=0.0192`。
2. 当前 closed-loop Bench2Drive 220 只有 `DS=44.81`、`RC=79.48`、`NDS=35.52`，说明仅靠 B2D Full open-loop 轨迹误差不能保证闭环表现。
3. 后续应抽样对比 raw image、preprocessed image、LiDAR BEV、target trajectory 和在线闭环失败帧，重点排查 route deviation / blocked / collisions 与 sensor contract gap 的关系。
4. `baseline-condition-v1` 已加入 `SpeedHead-v1` 和 `route_condition_token`，下一步应先重训并检查 open-loop trajectory / speed metrics，再跑 20-route 闭环。
5. 针对空间 checkpoint target，推理侧仍保留空间 PID fallback；`DIFFUSIONDRIVE_USE_SPEED_HEAD=1` 可切到 predicted-speed longitudinal controller。
6. 若在线闭环性能受 sensor gap 影响，再单独决定推理 sensor contract 是否向 B2D Full raw 对齐，或是否加入显式 domain adaptation / finetune。
