# DiffusionDrive LiDAR Attribution Diagnostics (2026-07-15)

本文档记录 `baseline-condition-v1` 第一批 LiDAR 归因结果。目标不是证明“LiDAR 有害”或“zero-LiDAR 更好”，而是把模型是否使用 LiDAR、离线与在线 contract 是否一致、闭环灾难性分叉是否可复现这三个问题拆开。

## 数据与产物

远端原始结果：

```text
/share/home/u19666033/ltr/dd_logs/lidar_diagnostics/b2d_train_4096
/share/home/u19666033/ltr/dd_logs/lidar_diagnostics/c2_online_contract
/share/home/u19666033/ltr/dd_logs/lidar_diagnostics/openloop_ablation
```

本地镜像和汇总：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/lidar_diagnostics/
/home/HeavenlySU/sitp_workspace/dd_logs/lidar_diagnostics/report/lidar_diagnostics_report.html
/home/HeavenlySU/sitp_workspace/dd_logs/lidar_diagnostics/report/analysis_summary.json
```

三组开环 CSV 均有 `39,432` 个样本，样本 key 完全配对。统计使用相同 checkpoint、manifest 和预处理；仅改变模型侧 LiDAR 输入。shuffle donor 是确定性的全数据集循环错配。

## 配对开环归因

| LiDAR mode | Mean L1 | Mean ADE | Mean FDE |
|---|---:|---:|---:|
| `original` | 0.02047 | 0.03211 | 0.05227 |
| `zero` | 0.31397 | 0.50525 | 0.86122 |
| `shuffle` | 0.58965 | 0.95248 | 1.64013 |

配对结果：

- `original` 在 `91.50%` 的样本上优于 `zero`，在 `95.91%` 的样本上优于 `shuffle`。
- `zero` 在 `69.21%` 的样本上优于 `shuffle`，说明错误 LiDAR 比缺失 LiDAR 更糟。
- 按 scenario/route/frame 聚合后的 `2,331` 个 route group 中，`original` 有 `99.70%` 优于 `zero`。
- 以 route group 为 cluster 的 bootstrap 中，`original-zero` mean-L1 差的 95% CI 为 `[-0.3538, -0.2351]`。
- 39 个 scenario 的 `original-zero` 差值全部为负。

因此可以排除“模型完全忽略 LiDAR”。在 B2D raw 分布上，正确 LiDAR 是强正向输入；shuffle 最差也说明模型确实使用了点云与场景的对应关系。闭环 C2 的局部负向不能外推成“LiDAR 全局负贡献”。

## 轨迹时间连续性

对同 route 相邻、frame gap 不超过 10 的 `37,101` 个 transition 做了 endpoint 跳变统计：

| Metric | Result |
|---|---:|
| target endpoint jump `>5m` | `1,271` / `3.43%` |
| target endpoint jump `>12m` | `119` / `0.32%` |
| original prediction jump `>5m` | `3.47%` |
| original prediction jump `>12m` | `0.35%` |
| zero prediction jump `>5m` | `0.60%` |
| zero prediction jump `>12m` | `0.011%` |
| `P(pred jump >5m | target jump >5m)` | `95.67%` |

target jump `>12m` 后一帧的 original mean L1 达到 `1.52m`。另有 `170` 个 target jump `>5m` 发生在 `speed<0.1m/s` 且 route-condition delta `<0.25m` 的情况下，占所有 `>5m` target jump 的 `13.38%`；这类样本不能用 command 或 route token 突变解释。

代码上，`build_spatial_path_target()` 最多向后读取 120 帧；路径不足时，`_resample_polyline_by_distance()` 会沿最后一个有效线段外推。车辆长时间静止时，窗口末端的微小位移方向或最后有效线段可能翻转，进而把 `2.5m...11.5m` target 外推到相反方向。该问题需要单独修复，不能归因给 LiDAR。

同时，`TrajectoryHead.forward_test()` 当前每次 forward 都从新的 `torch.randn()` 开始扩散去噪。zero-LiDAR 的预测显著更平滑，既可能来自模态被置零后模型退化为稳定先验，也可能包含随机初始噪声触发 mode switching 的影响。现有数据尚不能区分两者。

## Train-vs-online Contract

角度覆盖检查没有发现 gross yaw 或半扫描丢失：

- B2D full scan 与 online 拼接后的 `full_scan` 的 1-degree occupied-bin median 均为 `100%`，最大角度 gap median 均为 `0`。
- online `current_half` occupied-bin median 为 `50.28%`，最大 gap median 为 `179 deg`。
- `previous_half_aligned` occupied-bin median 为 `52.78%`，最大 gap median 为 `170 deg`。

online full scan 的前向区域主要来自上一 tick 的 half scan，后向区域主要来自当前 half scan。因此静态 ego alignment 虽可补偿车体运动，却不能把移动 actor 的上一 tick 点云变成当前时刻。这是当前最明确的在线时序 contract 差异。

场景匹配后的 median 比值也不是统一常数：

| Route / scenario | Model points online/train | Occupancy online/train | BEV mean online/train | 结果摘要 |
|---|---:|---:|---:|---|
| 24 / HighwayCutIn | 1.15 | 1.35 | 1.34 | DS 81.46，完成 |
| 50 / MergerIntoSlowTraffic | 0.91 | 1.04 | 1.03 | DS 54.61，vehicle collision |
| 139 / InterurbanActorFlow | 3.29 | 1.22 | 1.75 | DS 96.02，完成 |
| 153 / DynamicObjectCrossing | 1.80 | 3.07 | 2.49 | DS 27.12，RC 50.12，collision/offroad/blocked |
| 167 / YieldToEmergencyVehicle | 0.82 | 0.61 | 0.75 | DS 78.78，完成 |
| 185 / StaticCutIn | 2.98 | 1.92 | 1.85 | DS 79.88，完成 |

route 153 在失败前已经明显偏离训练分布，是值得继续追踪的正例；但 route 139 和 185 在更大密度差异下仍能完成，而 route 50 几乎匹配仍发生碰撞。因此 contract drift 是真实因素，但其幅度与闭环伤害不单调，不能仅靠全局 density normalization 解释或修复。

route 167/185 的本次 contract run 与原 C2 结果相比已不再出现相同灾难性分叉，也说明单次 CARLA route 结果包含不可忽略的随机性。

## 修正后的结论

1. 模型在 B2D raw 开环上强依赖且正确利用 LiDAR，LiDAR 不是无效分支。
2. online half-scan 拼接没有明显角度错误，但存在“前向动态物体点云晚一 tick”的时序不对称，以及强场景相关的密度/占用漂移。
3. C2/L1 的闭环差异不能只归因于 LiDAR；随机扩散初始噪声、target 外推不连续和 CARLA 单次运行随机性都是当前混杂因素。
4. zero-LiDAR 仍只作为诊断上界。项目正式路线应保留 LiDAR。

## 下一步顺序

1. **先做确定性推理诊断。** `random/fixed/zero/seeded` 初始 diffusion noise 和最终 mode/top-2/margin/entropy/endpoint 入口已实现；下一步用 5 个 fixed seed 重跑开环，并对 route 24/50/139/153 做重复闭环，回答灾难性分叉是否随 seed/mode 切换。
2. **修复并重建 spatial target。** 对低速停驻段引入稳定方向来源和连续性检查，禁止用窗口末端微小位移直接决定 11.5m 外推方向；重建 manifest 后重新统计 target jump。
3. **再做 online temporal LiDAR 诊断。** 优先比较当前 half-scan concat、仅当前 half、对动态区域降权等方案；同时做 above/below channel ablation。
4. **最后进入训练结构改造。** 在归因稳定后评估 conservative fusion gate、受控 modality dropout 和 BEV/agent auxiliary supervision。此时再决定是否需要 density alignment 或新 LiDAR backbone。

进入 fusion full retrain 的门槛是：固定 seed 下灾难样本可复现、target jump 已显著下降，并且至少一个 LiDAR channel/temporal ablation 能稳定解释 route-level 改善。否则新结构会把多个问题混在一次昂贵重训中。

### 确定性推理实现状态

- `fixed` 使用局部 `torch.Generator` 每次重建同一 batch-invariant template，不消耗全局 RNG。
- `seeded` 只接受显式 `features["diffusion_noise"]`；开环按 scenario/route/frame、在线按 route/step 生成稳定 key，避免静默依赖调用顺序。
- 默认 `random` 保持旧行为；新增输出均为无参数诊断 tensor，不改变 checkpoint strict-load contract。
- 在线 `records.jsonl` 会先把上一 tick endpoint 对齐到当前 ego frame，再判断 `>5m` jump。
- 本地 mini fixed/seeded 两种模式均完成 forward/loss/backward，并在重复 inference 中得到 `max_abs_diff=0.0`。远端 5-seed 与重点 route 结果仍待运行。
