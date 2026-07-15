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

根因已在 `MergerIntoSlowTrafficV2/Town12_Rep0_968_25...` 原始 annotation 上确认：车辆长时间停驻时，窗口末端 `frame 236 -> 237` 约 `1.5mm` 的反向量化抖动超过旧 `1mm` 去重阈值，被当成最后有效 segment，并被放大外推到 `11.5m`。随着 120 帧窗口滑动，末端 segment 会在前后方向之间切换。

代码已改为 `arc_length_stable_extrapolation_v2`：已观测路径内的弧长插值不变；外推使用至少 `0.5m` baseline 的 trailing displacement，未来窗口总位移不足时改用 route condition。旧 manifest 会因缺少新 `spatial_target` schema 被 loader 明确拒绝。该已知 route 的 8 个旧 `>12m` jump 在 v2 重算后全部降为 `0.002m...0.058m`。

又对旧统计中 `>12m` 最多的前五条 route 做了 raw annotation 配对重算，覆盖 `77/119` 个全量大跳变：77 个事件全部降到 `<12m`，新 jump p95 为 `0.083m`。仅 2 个仍 `>5m`，它们分别伴随 `85m/146m` route-condition delta、明显减速和 command/阶段切换，应保留为真实 transition。

2026-07-16 已完成 v2 全量 `39,432` 样本配对门控。样本 key/order 完全一致，duplicate/missing/extra 和 condition/speed/brake 等 invariant mismatch 均为 0；`>12m` jump 从旧版 `119` 降到 `19`，`>5m` 从 `1,271` 降到 `1,162`。剩余 19 条中有 11 条来自 `InterurbanAdvancedActorFlow`，多伴随车速和 future path length 急降；trace 显示 trailing displacement 与 route fallback cosine 最低到 `-0.94`。

因此 contract 继续升级为 `arc_length_route_aligned_extrapolation_v3`：可靠 trailing displacement 除了至少 `0.5m`，还必须与 route fallback cosine `>=0`。只对需要外推的方向选择增加 guard，不修改已观测路径内的弧长插值。对上述 19 条 trace 本地反事实重算后，`>12m` 降到 7 条；剩余 7 条全部伴随 `35m...250m` route-condition delta。v3 full manifest 仍需通过调度作业重建并做实际全量复核。

同时，`TrajectoryHead.forward_test()` 历史行为会在每次 forward 从新的 `torch.randn()` 开始扩散去噪。zero-LiDAR 的预测显著更平滑，既可能来自模态被置零后模型退化为稳定先验，也可能包含随机初始噪声触发 mode switching 的影响，因此补做了 5-seed 配对诊断。

## Fixed 5-seed 推理结果

fixed seed `0...4` 各完成 `39,432` 个严格配对样本，结果下载到：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/lidar_diagnostics/diffusion_noise_fixed_5seed/
```

关键结果：

- fixed 五个 seed 的 mean L1 平均为 `0.01936`，单 seed 范围为 `0.01875...0.02106`；历史 random run 为 `0.02047`，位于正常 seed 波动范围内。
- 同一样本跨 seed endpoint 最大两两距离的 p95/p99 仅为 `0.052m/0.099m`。虽然只有 `11.38%` 样本五次都选择相同 mode id，但多数 mode id 变化并不产生几何灾难。
- `L1>1m` 的五 seed 并集为 `127` 个样本，交集仅 `29`；说明随机初始噪声会改变尾部样本是否越过阈值，但不是共同异常的主因。
- `127` 个并集样本中 `123` 个位于存在 target jump `>12m` 的 route，`93` 个发生在该 jump 前后两个采样间隔内；top-5 route 占并集的 `74.8%`，top-10 占 `95.3%`。
- 固定噪声相对 random 只让 prediction temporal jump `>5m` 减少约 `21.8%`，而 target jump 分布完全不变。主要不连续来自旧 target contract，diffusion noise 是次级放大因素。

因此不能把 mode id 变化本身当成错误。后续应先重建稳定 target 并重训，再用固定 seed 做模型和 LiDAR ablation；重点 route 的多 attempt 闭环仍用于量化 CARLA 环境方差。

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
3. 5-seed 开环表明随机扩散初始噪声只影响一部分尾部归属；旧 target 外推不连续是更主要、可直接修复的混杂因素。
4. zero-LiDAR 仍只作为诊断上界。项目正式路线应保留 LiDAR。

## 下一步顺序

1. **重建并验证 spatial target。** v2 全量门控已完成并暴露反向 trailing displacement；下一步通过调度作业新建 v3 full soft-clean 配对 manifest，重新统计 `>5m/>12m` jump，并确认剩余事件均由大幅 route-condition 切换解释。
2. **完成重点 route 方差诊断。** 5-seed all-scenarios 开环已完成；route 24/50/139/153 的 fixed seed 多 attempt 闭环仍待跑，用于区分模型 seed 与 CARLA 环境方差。
3. **再做 online temporal LiDAR 诊断。** 优先比较当前 half-scan concat、仅当前 half、对动态区域降权等方案；同时做 above/below channel ablation。
4. **最后进入训练结构改造。** 用 v3 target full retrain 后评估 conservative fusion gate、受控 modality dropout 和 BEV/agent auxiliary supervision。此时再决定是否需要 density alignment 或新 LiDAR backbone。

进入 fusion full retrain 的门槛是：固定 seed 下灾难样本可复现、target jump 已显著下降，并且至少一个 LiDAR channel/temporal ablation 能稳定解释 route-level 改善。否则新结构会把多个问题混在一次昂贵重训中。

### 确定性推理实现状态

- `fixed` 使用局部 `torch.Generator` 每次重建同一 batch-invariant template，不消耗全局 RNG。
- `seeded` 只接受显式 `features["diffusion_noise"]`；开环按 scenario/route/frame、在线按 route/step 生成稳定 key，避免静默依赖调用顺序。
- 默认 `random` 保持旧行为；新增输出均为无参数诊断 tensor，不改变 checkpoint strict-load contract。
- 在线 `records.jsonl` 会先把上一 tick endpoint 对齐到当前 ego frame，再判断 `>5m` jump。
- 本地 mini fixed/seeded 两种模式均完成 forward/loss/backward，并在重复 inference 中得到 `max_abs_diff=0.0`。
- 远端 fixed seed `0...4` all-scenarios 已完成并通过样本 key、noise mode/seed、LiDAR mode 和 invariant columns 校验；重点 route 多 attempt 尚待运行。
