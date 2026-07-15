# DiffusionDrive CARLA 文档索引

本文档作为当前 DiffusionDrive on CARLA 工作的轻量入口，索引其它更详细的文档。

## 文档索引

### 当前主线文档

- [dd_todo.md](dd_todo.md)：当前 DiffusionDriveAgent 后续事项和优先级。
- [diffusiondrive_lidar_bev_alignment_issues.md](diffusiondrive_lidar_bev_alignment_issues.md)：LiDAR / BEV 对齐从 `v3` 到 `v9` 的详细排查记录、实验结论和当前优先级边界。
- [diffusiondrive_agent_explained.md](diffusiondrive_agent_explained.md)：当前 CARLA 侧 DiffusionDriveAgent 结构说明。
- [diffusiondrive_navsim_vs_carla_gap.md](diffusiondrive_navsim_vs_carla_gap.md)：NAVSIM 原版 DiffusionDrive 与 CARLA Garage 环境差异。
- [diffusiondrive_input_preprocessing.md](diffusiondrive_input_preprocessing.md)：当前 DiffusionDrive 输入预处理核对和训练前决策项。
- [b2d_full_sensor_contract.md](b2d_full_sensor_contract.md)：Bench2Drive Full 主训练分布的 sensor / time / preprocessing contract。
- [diffusiondrive_lidar_diagnostics_20260715.md](diffusiondrive_lidar_diagnostics_20260715.md)：condition-v1 配对开环 LiDAR 归因、train-vs-online contract、fixed 5-seed、spatial target v2 和下一步决策门槛。
- [diffusiondrive_anchor_adaptation.md](diffusiondrive_anchor_adaptation.md)：最新 `99x10x2` anchor 的适配路线。
- [diffusiondrive_training.md](diffusiondrive_training.md)：当前 CARLA-native DiffusionDrive 训练入口和 smoke 命令。
- [diffusiondrive_remote_training_progress.md](diffusiondrive_remote_training_progress.md)：远端 B2D Full 训练阶段、baseline-basic open-loop / closed-loop、20-route 闭环 A/B 和高误差样本诊断。
- [diffusiondrive_baseline_next_steps.md](diffusiondrive_baseline_next_steps.md)：baseline-basic 之后的 `baseline-condition-v1` 主线、`SpeedHead-v1`、route condition token、PID / LiDAR 诊断和评测门槛。
- [diffusiondrive_local_env.md](diffusiondrive_local_env.md)：当前本机 `garage_2` 环境和 smoke test 结果。

### 本地结果归档

- `/home/HeavenlySU/sitp_workspace/dd_logs/EVAL_CSV_INDEX.md`：本地下载的远端 eval artifact 索引。
- `/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/README.md`：本地 open-loop / closed-loop 汇总说明。
- `/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/summary.csv`：per-stage / per-scene open-loop 汇总。
- `/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/aggregate_summary.csv`：six-scene 和 baseline full-scenario open-loop 聚合。
- `/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_closed_loop_summary.csv`：baseline-basic Bench2Drive 220 闭环摘要。
- `/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_ablation_20routes_summary.csv`：baseline-basic 20-route 闭环 A/B 指标汇总。
- `/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_ablation_20routes_creep_summary.csv`：baseline-basic 20-route creep / safety-box 日志统计。
- `/home/HeavenlySU/sitp_workspace/dd_logs/lidar_diagnostics/report/lidar_diagnostics_report.html`：2026-07-15 LiDAR 配对开环、online contract 和时序轨迹诊断报告。

### 背景和工程说明

- [coordinate_systems.md](coordinate_systems.md)：坐标系相关说明。
- [engineering.md](engineering.md)：工程实现和维护说明。
- [run.md](run.md)：早期手工 CARLA / leaderboard debug 命令备忘；当前训练和评测命令优先看主线 DiffusionDrive 文档。
- [ltr.md](ltr.md)：LTR / 远程实验相关记录。
- [history.md](history.md)：历史记录。
- [done.md](done.md)：已完成事项。

### 训练和后续扩展

- [dd_train_todo.md](dd_train_todo.md)：DiffusionDrive 后续训练事项。
- [additional_features.md](additional_features.md)：额外功能记录。
- [common_mistakes_in_benchmarking_ad.md](common_mistakes_in_benchmarking_ad.md)：自动驾驶 benchmark 常见误区。

### 旧文档

旧版分析和过时方案已放在 [outdated/](outdated/) 下，仅用于追溯，不作为当前实现判断基线。
