# 已完成工作记录

**更新日期**: 2026-04-26

本文档只记录已经完成的工作，详细问题分析见 `docs/lidar_bev_alignment_issues.md`，后续事项见 `docs/dd_todo.md`。

## DiffusionDrive CARLA 链路

- 已梳理当前 `DiffusionDriveAgent` 推理链路，并将旧结论迁移到 `docs/outdated/`。
- 已确认当前运行基线以 `carla_garage/team_code/diffusiondrive_agent.py` 为准。
- 已接入并验证基础 leaderboard 推理链路，包括传感器输入、RoutePlanner、UKF 状态估计、模型加载、轨迹输出和 PID 控制。
- 已增强 checkpoint 加载逻辑，支持常见容器字段、前缀清理、shape 匹配加载，并输出 missing / unexpected / mismatch 摘要。
- 已梳理 NAVSIM DiffusionDrive 与 CARLA 运行链路之间的主要差异，形成 `docs/diffusiondrive_navsim_vs_carla_gap.md`。
- 已整理运行方式、工程注意事项和 agent 解释文档，包括 `docs/diffusiondrive_run.md`、`docs/engineering.md`、`docs/diffusiondrive_agent_explained.md`。

## 运行时功能补齐

- 已接入 stuck detection，包括 `stuck_detector`、`force_move` / creep 逻辑和 `creep_throttle` 配置。
- 已接入 safety box，包括前方 LiDAR 安全框过滤、emergency stop 逻辑和相关阈值配置。
- 已接入基于 CARLA world stop sign actor 的 stop sign controller，不再依赖旧版 bbox stop sign 检测头。
- 已梳理 `status_feature` 与 `extra_sensors` 的职责边界，明确当前 agent 显式构造的是 command、velocity、acceleration 组合。

## LiDAR 时序与对齐

- 已实现 LiDAR 半帧拼接逻辑，包括 `lidar_last`、`align_lidar()` 和完整扫描生成。
- 已实现 LiDAR 多帧 buffer、buffer 未填满等待、历史帧 realign 到当前坐标系，以及 `realign_lidar` 配置。
- 已添加合成验证脚本 `tools/validate_lidar_bev_alignment.py`。
- 已验证在线 `DiffusionDriveAgent.align_lidar()` 与离线 `CARLA_Data.align()` 当前公式一致。
- 已验证 `DiffusionDriveAgent` 的负索引历史帧配对优于 `sensor_agent.py` 的旧正索引写法。

## LiDAR / BEV 对齐实验

- 已迭代 `tools/render_lidar_bev_alignment_v3.py` 到 `tools/render_lidar_bev_alignment_v9.py`，覆盖动态 masking、局部搜索、粗到细搜索、固定 ROI、common-support、位移先验和距离分桶诊断。
- 已完成 v5、v8、v9 的 batch_100 对比，以及 v9 的完整批量输出。
- 已确认固定 ROI 和单纯扩大搜索范围不是有效方向；v9 更适合诊断，但还不是最终替代方案。
- 已确认主要残留问题集中在近场 `0-8m` raw IoU 退化，中远场整体表现更稳定。

## 批量诊断工具

- 已添加 `tools/analyze_lidar_bev_residuals.py`，用于统计 residual `dx/dy/norm`、history、town、scenario 等分布。
- 已添加 `tools/make_lidar_bev_contact_sheet.py`，用于生成 top scenario 和 worst gain 的 HTML contact sheet。
- 已生成 `lidar_bev_v9_batch_100` 与 `lidar_bev_v9_batch_100_1` 的 summary 和 contact sheet。
- 已用 residual 统计排除明显全局固定 `dx/dy` 偏移，当前不优先按外参或坐标系常量偏差修复。

## 当前结论

- 当前共享变换公式在线与离线一致，但在合成 canonical SE(2) 检查中仍有差异。
- 现有证据不支持把问题主因判断为全局外参偏移或固定坐标系偏差。
- v9 是更保守、更适合诊断的版本，但还不是最终替代方案。
- 下一步重点应放在近场 `0-8m` raw/dynamic 一致性约束，而不是继续扩大搜索范围或优先修 SE(2) 语义。
