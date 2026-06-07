# 已完成工作记录

**更新日期**: 2026-06-07

本文档只记录已经完成的工作，详细问题分析见 `docs/diffusiondrive_lidar_bev_alignment_issues.md`，后续事项见 `docs/dd_todo.md`。

## DiffusionDrive CARLA 链路

- 已梳理当前 `DiffusionDriveAgent` 推理链路，并将旧结论迁移到 `docs/outdated/`。
- 已确认当前运行基线以 `carla_garage/team_code/diffusiondrive_agent.py` 为准。
- 已接入并验证基础 leaderboard 推理链路，包括传感器输入、RoutePlanner、UKF 状态估计、模型加载、轨迹输出和 PID 控制。
- 已增强 checkpoint 加载逻辑，支持常见容器字段、前缀清理、shape 匹配加载，并输出 missing / unexpected / mismatch 摘要。
- 已梳理 NAVSIM DiffusionDrive 与 CARLA 运行链路之间的主要差异，形成 `docs/diffusiondrive_navsim_vs_carla_gap.md`。
- 已整理运行方式、工程注意事项和 agent 解释文档，包括 `docs/run.md`、`docs/engineering.md`、`docs/diffusiondrive_agent_explained.md`。
- 已完成 full baseline-basic 训练：4 卡 L40 / DDP、B2D Full scenario-balanced 全量 manifest、`epochs=100`、per-GPU batch 64、`lr=6e-4`、`image_encoder_lr_mult=0.5`。
- 已完成 baseline-basic full-scenario open-loop 汇总和 Bench2Drive 220 sensor-only closed-loop 汇总，并下载到本地 `dd_logs` 镜像路径。
- 已完成 baseline-basic 20-route 闭环 A/B 初筛，并将结果同步到 `dd_logs/eval_summaries/baseline_basic_ablation_20routes_summary.csv` 与 `baseline_basic_ablation_20routes_creep_summary.csv`。
- 已完成 20-route sensor / LiDAR 诊断补充：Z0/Z1 zero-LiDAR 只置零模型侧 `lidar_feature`、不关闭 raw LiDAR safety-box；结果显示当前模型侧 LiDAR BEV 可能是负贡献。

## 运行时功能补齐

- 已接入 stuck detection，包括 `stuck_detector`、`force_move` / creep 逻辑和 `creep_throttle` 配置。
- 已接入 safety box，包括前方 LiDAR 安全框过滤、emergency stop 逻辑和相关阈值配置。
- 已接入基于 CARLA world stop sign actor 的 stop sign controller，不再依赖旧版 bbox stop sign 检测头；baseline-basic / sensor-only 主线默认 `STOP_CONTROL=0`，privileged 规则停车 ablation 需显式开启。
- 已梳理 `status_feature` 与 `extra_sensors` 的职责边界；当前 DiffusionDrive 训练和推理共用 `command_one_hot(6) + speed(1)` 的 7 维 `status_feature`，不再使用旧的 velocity / acceleration 组合。
- 已给 `DiffusionDriveAgent` 的 UKF 状态估计加入 covariance 正定保护和 measurement reset，避免 `filterpy` 在 `P` 非正定时让闭环 route 直接 agent crash。

## Baseline-Basic 评测归档

- Open-loop legacy six-scene：`l1_mean=0.0092`、`ade_mean=0.0152`、`fde_mean=0.0234`。
- Open-loop all-scenarios：39 scenes、39844 samples、`l1_mean=0.0192`、`ade_mean=0.0303`、`fde_mean=0.0483`。
- Closed-loop Bench2Drive 220：`DS=44.8074`、`RC=79.4774`、`NDS=35.5193`。
- Closed-loop status：`Completed=118`、`Perfect=1`、`Failed - Agent deviated from the route=69`、`Failed - Agent got blocked=27`、`Failed - Agent timed out=5`。
- 当前结论：baseline-basic 开环轨迹误差很低，但闭环仍是弱 baseline，主要失败模式是 route deviation、blocked、低速和 collisions。
- 20-route A/B 当前结论：空间 PID speed tuning 是最明显正向方向；`A8_pid_6_2p5` 更均衡，`A9_pid_7_3` 的 DS/NDS 最高但 collision / timeout 风险更高。单独调 stuck threshold 不够稳定，creep / safety-box 需要单独用日志计数观察。

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
- 已明确当前 DiffusionDrive 模型侧 LiDAR 仍是 NAVSIM-style 单帧 BEV histogram + `resnet34` encoder，不是 syb 的 `regnety_032`；切换 backbone 或 no-LiDAR 都应作为 full retrain ablation。

## 当前结论

- 当前共享变换公式在线与离线一致，但在合成 canonical SE(2) 检查中仍有差异。
- 现有证据不支持把问题主因判断为全局外参偏移或固定坐标系偏差。
- v9 是更保守、更适合诊断的版本，但还不是最终替代方案。
- 旧 v3-v9 多帧 BEV residual refinement 不是当前 baseline 主线，因为 `lidar_seq_len=1`，模型不消费更久历史 BEV。
- 下一步 LiDAR 方向优先做 no-LiDAR full retrain；若保留 LiDAR，再考虑 `regnety_032` full retrain 或补 `agent_states / agent_labels / bev_semantic_map` auxiliary supervision。近场 `0-8m` raw/dynamic 一致性约束保留为后续重启多帧 BEV refinement 时的诊断项。
