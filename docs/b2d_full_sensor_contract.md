# Bench2Drive Full Sensor Contract

本文档记录当前 DiffusionDrive 训练主线使用的 Bench2Drive Full 数据约定，以及它和在线 `DiffusionDriveAgent` sensor suite 之间仍需显式处理的 gap。

## 当前决策

- 后续训练和模型修改以 Bench2Drive Full 为主要训练数据分布。
- 当前 raw Bench2Drive / B2D Full loader 不是只用于 smoke 或预训练；它是当前 trajectory-only baseline 的主训练入口。
- 因此训练配置、文档和实验命名应围绕 `b2d_full_raw` 语义记录，而不是默认假设它已经和在线 garage sensor suite 完全一致。

## B2D Full Raw 输入事实

当前数据 helper 读取原始 Bench2Drive route 目录：

- `camera/rgb_front/*.jpg`
- `lidar/*.laz`
- `anno/*.json.gz`

当前已知 raw sensor 几何：

- front camera: `1600x900, fov=70, x=0.8, y=0.0, z=1.6`
- LiDAR: `x=-0.39, y=0.0, z=1.84, yaw=0, range=85`

当前 dataset 会把 raw front image 转成模型输入：

- raw image source size: `900x1600`
- resize 到在线 garage sensor size: `512x1024`
- `crop_array(): 512x1024 -> 384x1024`
- resize 到模型输入: `256x1024`
- 数值范围: `[0,1]`
- 默认保留 JPEG artifact
- 默认无 ImageNet normalization

## 在线 Garage Sensor Suite

当前 `DiffusionDriveAgent` 在线推理仍使用 garage sensor contract：

- front camera: `512x1024, fov=110, x=-1.5, y=0.0, z=2.0`
- LiDAR: `x=0.0, y=0.0, z=2.5, yaw=-90`
- LiDAR histogram: `256x256`, `pixels_per_meter=4.0`
- 继续复用半帧拼接、多帧 buffer、realign、stuck / safety box 和 PID 控制。

## 已知 Gap

B2D Full raw sensor 和在线 garage sensor suite 不完全一致：

- camera FOV 不同：`70` vs `110`
- camera pose 不同：B2D raw 在车前方，garage 在线相机在 `x=-1.5,z=2.0`
- raw image aspect / source size 不同：`900x1600` vs 在线 `512x1024`
- LiDAR pose / yaw / range 不同
- 当前 resize / crop 只能对齐 tensor shape，不能消除真实几何 gap

当前结论不是“raw B2D 已经等价于在线 sensor suite”，而是：

- 训练主线以 B2D Full raw 分布为准。
- 在线闭环 sensor gap 作为后续推理一致性 / domain gap 问题显式记录和验证。

## 时间语义

当前训练入口默认记录以下假设：

- `dataset_mode`: `b2d_full_raw`
- `assumed_frame_interval_seconds`: `0.1`
- `future_stride_frames`: 默认 `10`
- `target_interval_seconds`: `future_stride * assumed_frame_interval`
- `trajectory_sampling_interval_seconds`: 来自 `DiffusionDriveConfig.trajectory_sampling.interval_length`
- `anchor_interval_seconds_assumption`: 当前同 `trajectory_sampling.interval_length`

注意：轨迹时间语义仍未冻结。full training 前仍需明确：

- B2D Full frame interval 是否稳定为 `0.1s`
- target stride、anchor interval、`trajectory_sampling.interval_length` 是否一致
- PID desired speed 的 waypoint 间隔假设是否需要改写

## Training Config 记录

`team_code/train_diffusiondrive.py` 会在 `training_config.json` 中记录：

- `data.dataset_mode`
- `time_semantics`
- `anchor.path / shape / num_modes`
- `sensor_contract`
- `preprocessing.source_image_size / online_sensor_size / model_image_size`
- `status_feature`

这些字段用于保证 checkpoint 和实验结果可以追溯到当时的 sensor / time / anchor 假设。

## 后续建议

1. full training 前先在 B2D Full 上跑小 smoke 和吞吐量测试。
2. 抽样可视化 raw image、preprocessed image、LiDAR BEV 和 target trajectory。
3. 固定时间语义后，再决定是否调整 `trajectory_sampling.interval_length` 和 PID waypoint interval。
4. 若在线闭环性能受 sensor gap 影响，再单独决定推理 sensor contract 是否向 B2D Full raw 对齐，或是否加入显式 domain adaptation / finetune。
