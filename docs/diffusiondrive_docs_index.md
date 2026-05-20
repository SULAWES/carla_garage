# DiffusionDrive CARLA 文档索引

本文档作为当前 DiffusionDrive on CARLA 工作的轻量入口，索引其它更详细的文档。

## 文档索引

### 当前主线文档

- [dd_todo.md](dd_todo.md)：当前 DiffusionDriveAgent 后续事项和优先级。
- [diffusiondrive_lidar_bev_alignment_issues.md](diffusiondrive_lidar_bev_alignment_issues.md)：LiDAR / BEV 对齐从 `v3` 到 `v10` 的详细排查记录和实验结论。
- [diffusiondrive_agent_explained.md](diffusiondrive_agent_explained.md)：当前 CARLA 侧 DiffusionDriveAgent 结构说明。
- [diffusiondrive_navsim_vs_carla_gap.md](diffusiondrive_navsim_vs_carla_gap.md)：NAVSIM 原版 DiffusionDrive 与 CARLA Garage 环境差异。
- [diffusiondrive_input_preprocessing.md](diffusiondrive_input_preprocessing.md)：当前 DiffusionDrive 输入预处理核对和训练前决策项。
- [b2d_full_sensor_contract.md](b2d_full_sensor_contract.md)：Bench2Drive Full 主训练分布的 sensor / time / preprocessing contract。
- [diffusiondrive_anchor_adaptation.md](diffusiondrive_anchor_adaptation.md)：最新 `99x10x2` anchor 的适配路线。
- [diffusiondrive_training.md](diffusiondrive_training.md)：当前 CARLA-native DiffusionDrive 训练入口和 smoke 命令。
- [diffusiondrive_local_env.md](diffusiondrive_local_env.md)：当前本机 `garage_2` 环境和 smoke test 结果。

### 背景和工程说明

- [coordinate_systems.md](coordinate_systems.md)：坐标系相关说明。
- [engineering.md](engineering.md)：工程实现和维护说明。
- [run.md](run.md)：仓库通用运行说明。
- [ltr.md](ltr.md)：LTR / 远程实验相关记录。
- [history.md](history.md)：历史记录。
- [done.md](done.md)：已完成事项。

### 训练和后续扩展

- [dd_train_todo.md](dd_train_todo.md)：DiffusionDrive 后续训练事项。
- [additional_features.md](additional_features.md)：额外功能记录。
- [common_mistakes_in_benchmarking_ad.md](common_mistakes_in_benchmarking_ad.md)：自动驾驶 benchmark 常见误区。

### 旧文档

旧版分析和过时方案已放在 [outdated/](outdated/) 下，仅用于追溯，不作为当前实现判断基线。
