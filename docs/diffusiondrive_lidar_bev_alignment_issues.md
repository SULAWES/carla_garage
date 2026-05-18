# LiDAR BEV 对齐结论

本文档记录离线 LiDAR BEV 对齐排查过程、阶段性结论、已否定方向、仍然失败的样本，以及下一步建议。

## 总体判断

当前最重要的结论：

- measurement-based alignment 整体是有效的，`v3` 批量结果已经证明平均 IoU 有正收益。
- 动态目标会污染 occupancy IoU，必须在评价时剔除动态目标区域。
- 单纯扩大 residual 搜索范围会诱导 scorer 发散，尤其容易出现不可信的大平移。
- 固定矩形 ROI 已被否定，不能继续围绕固定 ROI 调参。
- common-support 能减少坏区域污染，但单独使用会在 support 很小时虚高。
- residual-motion prior 是有效组件，可以显著压住大位移 outlier。
- 当前没有证据表明存在全局固定外参或坐标系偏移，问题更像是评价目标和局部场景导致的不稳定。
- 距离分桶显示主要风险在近场 `0-8m`，中远场 `16-32m` 的 raw gain 反而稳定为正。

当前不建议直接做的事：

- 不建议继续盲目扩大 `dx/dy/yaw` 搜索范围。
- 不建议继续调固定矩形 ROI。
- 不建议只最大化 common-support IoU。
- 不建议假设存在全局固定 `dx/dy` 偏移并直接修外参。

当前更合理的方向：

- 保留 `v9` 的 residual-motion prior。
- 在下一版 scorer 中提高近场 `0-8m` 的 raw/dynamic 一致性权重。
- 对 top scenario 的 contact sheet 做人工观察，确认近场变差是遮挡、动态物、稀疏点云，还是搜索目标误导。

## 当前结论

离线 LiDAR BEV 对齐分析目前有七层结论：

- `v3`：在计算 IoU 时剔除动态目标区域
- `v4`：在 measurement 初值附近做单阶段 `dx/dy/yaw` refinement
- `v5`：将 refinement 改成 coarse-to-fine 两阶段搜索
- `v6`：尝试固定矩形 ROI 打分
- `v7`：尝试基于当前帧和候选历史帧 occupancy 的 common-support 区域打分
- `v8`：在 `v7` 基础上加入 residual-motion prior，惩罚过大的 refinement 位移
- `v9`：统一 before/initial_after/after 的 support 评价口径，并在 support 太小时回退 dynamic-only

探索路径可以概括为：

- 先用 `v3` 修正动态目标污染，确认对齐整体有效。
- 再用 `v4/v5` 验证坏例子是否存在局部可修正 residual。
- 然后用 `v6` 测试固定 ROI，结果证明固定 ROI 会误导 scorer。
- 再用 `v7` 测试 common-support，发现能翻正 masked 指标但会诱导大平移。
- 接着用 `v8` 加位移先验，解决大平移，但暴露 support 口径不公平和小 support 虚高。
- 最后用 `v9` 统一评价口径并加入 fallback，同时增加距离分桶诊断，定位到近场 raw 指标变差。

对应脚本为：

- [tools/render_lidar_bev_alignment_v3.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/render_lidar_bev_alignment_v3.py)
- [tools/render_lidar_bev_alignment_v4.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/render_lidar_bev_alignment_v4.py)
- [tools/render_lidar_bev_alignment_v5.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/render_lidar_bev_alignment_v5.py)
- [tools/render_lidar_bev_alignment_v6.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/render_lidar_bev_alignment_v6.py)
- [tools/render_lidar_bev_alignment_v7.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/render_lidar_bev_alignment_v7.py)
- [tools/render_lidar_bev_alignment_v8.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/render_lidar_bev_alignment_v8.py)
- [tools/render_lidar_bev_alignment_v9.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/render_lidar_bev_alignment_v9.py)
- [tools/analyze_lidar_bev_residuals.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/analyze_lidar_bev_residuals.py)
- [tools/make_lidar_bev_contact_sheet.py](/home/heavenlysu/sitp_workspace/carla_garage/tools/make_lidar_bev_contact_sheet.py)

`v3` 的批量统计结果表明：

- 原始指标的平均增益：
  `mean_iou_gain_raw = 0.07026892615635823`
- 剔除动态目标后的平均增益：
  `mean_iou_gain = 0.08790455719210802`

这说明：

- LiDAR 历史帧经过对齐后，整体上确实提升了与当前帧的 BEV 一致性
- 旧指标低估了静态场景上的真实对齐收益
- 动态目标运动会污染原始 occupancy IoU

进一步对失败样本做 `v4/v5` 局部 refinement 后，可以确认：

- measurement 初值附近确实存在可修正的平移误差
- 局部 refinement 能显著改善最差样本
- 但继续单纯放大搜索范围，收益已经有限
- 目前主要问题不再像是“最优点还在搜索范围外面”
- `yaw` 在这些坏例子里不是主导误差，主修正量主要集中在平移，尤其是 `dy`

`v6` 的固定矩形 ROI 不应继续作为主方向：

- 默认 ROI 和收窄后的 A/B ROI 都让两个坏例子的 masked gain 进一步恶化
- 固定 ROI 会把打分目标强行限制到错误区域，容易诱导 refinement 选出不合理的大平移
- 当前结论是固定矩形 ROI 已被否定，后续不应继续围绕它调参

`v7` 的 common-support 打分比固定 ROI 更接近正确方向，但还不能作为最终对齐结果：

- `Town12_Rep0_2488_0_route0_11_08_02_55_17`, `frame=35`, `history=5`
  - `v7`: `masked_gain = 0.02155371373728293`
  - `raw_gain = -0.1143449668584853`
  - `dx = 3.799999952316284`, `dy = 0.75`
  - `support_pixels = 4002`

- `Town12_Rep0_2488_1_route0_11_08_17_28_24`, `frame=38`, `history=5`
  - `v7`: `masked_gain = 0.10058318547124515`
  - `raw_gain = -0.04788894332650179`
  - `dx = -4.5`, `dy = 0.8500000238418579`
  - `support_pixels = 5098`

这说明 common-support 区域可以把两个坏例子的 masked IoU 翻正，但 refinement 选出的 `dx` 偏大，且第二个样本已经贴到有效搜索边界。因此 `v7` 目前更适合作为诊断指标，不能直接作为最终历史帧变换策略。

`v8` 是目前更可信的方向：

- 保留 `v7` 的 common-support 打分
- 默认加入 `score = support_iou - 0.01 * (dx^2 + dy^2)`
- 默认限制 `max_total_translation_norm = 2.0`
- 目标是允许 `v5` 中约 `0.8m` 的合理修正，同时压制 `v7` 中 `3.8m/4.5m` 级别的大位移解

两条坏例子的 `v8` 结果如下：

- `Town12_Rep0_2488_0_route0_11_08_02_55_17`, `frame=35`, `history=5`
  - `v8`: `masked_gain = 0.0524648118605377`
  - `raw_gain = -0.07013995171224305`
  - `dx = 0.05000000074505806`, `dy = 0.8500000238418579`
  - `translation_norm = 0.8514693421407872`
  - `rejected_candidates = 892`

- `Town12_Rep0_2488_1_route0_11_08_17_28_24`, `frame=38`, `history=5`
  - `v8`: `masked_gain = 0.0591520139030873`
  - `raw_gain = -0.046310534539907994`
  - `dx = 0.0`, `dy = 0.800000011920929`
  - `translation_norm = 0.800000011920929`
  - `rejected_candidates = 892`

这说明 `v8` 同时满足两个目标：一方面通过 common-support 让 masked IoU 翻正，另一方面把 residual translation 压回了和 `v5` 一致的合理范围。因此 `v8` 比 `v7` 更适合作为下一步批量验证版本。

`batch_100` 对比结果表明，`v8` 的位移先验有效，但当前指标口径还不能直接证明 `v8` 优于 `v5`：

- 样本数：`600 = 100 routes * 2 frames * 3 history`
- `v5 mean_iou_gain = 0.14045930654581848`
- `v8 mean_iou_gain = 0.2790834153693468`
- `v5 positive_rate = 0.9533333333333334`
- `v8 positive_rate = 0.9983333333333333`
- `v5 translation_norm mean = 0.4472907111717987`, `p95 = 3.252499997615812`, `max = 6.121478644212499`
- `v8 translation_norm mean = 0.16512446594784147`, `p95 = 0.949999988079071`, `max = 2.0`

这些结果说明 `v8` 确实压住了 `v5/v7` 中不可信的大位移。但这里有一个重要 caveat：`v8` 的 `after_iou` 使用 common-support mask，而 `before_iou` 仍是普通 dynamic mask，所以 `v8 iou_gain` 会被抬高，不能直接和 `v5 iou_gain` 做公平比较。

更可靠的 paired raw 指标显示：

- `v5 mean_raw_gain = 0.11567376091741809`
- `v8 mean_raw_gain = 0.10858453390820565`
- `v8 - v5 mean_raw_gain_delta = -0.0070892270092124295`
- `raw_v8_better_rate = 0.05`
- `raw_v8_worse_rate = 0.21166666666666667`
- `raw_delta < -0.05` 的样本数为 `20 / 600`

这说明 `v8` 目前还不能直接全量替代 `v5`。它解决了“大位移搜索发散”的问题，但 common-support 指标在 support 很小的样本上会虚高。例如有些样本 `support_pixels` 只有 `61` 到 `352`，masked gain 很高，但 raw gain 明显变差。

当前批量结论：

- `v8` 的 residual-motion prior 是有效组件，应保留
- common-support mask 需要增加最小 support 约束，或者在 support 太小时回退到 `v5`/dynamic-only 评分
- 后续不应直接扩大到全量，而应先修正指标口径

`v9` 针对 `batch_100` 暴露的问题做了两个修正：

- `before_iou`、`initial_after_iou`、`after_iou` 都使用同一个评价函数，避免 `before` 是 dynamic-only、`after` 是 common-support 的不公平比较
- 默认 `min_common_support_pixels = 1000`，当 common support 小于该阈值时回退到 dynamic-only 评分

`v9` 仍然保留 `v8` 的 residual-motion prior：

- `translation_penalty_weight = 0.01`
- `max_total_translation_norm = 2.0`

`v9` summary 中新增的关键字段：

- `before_score_mode`
- `initial_after_score_mode`
- `after_score_mode`
- `before_support_pixels`
- `initial_after_support_pixels`
- `support_pixels`
- `before_iou_dynamic`
- `after_iou_dynamic`
- `before_iou_support`
- `after_iou_support`
- `iou_gain_0_8m`
- `iou_gain_8_16m`
- `iou_gain_16_24m`
- `iou_gain_24_32m`
- `iou_gain_raw_0_8m`
- `iou_gain_raw_8_16m`
- `iou_gain_raw_16_24m`
- `iou_gain_raw_24_32m`

`v9 batch_100` 验证时优先检查了：

- `mean_iou_gain` 是否不再明显虚高
- `mean_raw_gain` 是否不低于 `v5`
- `score_mode = dynamic_fallback` 的比例是否合理
- `support_pixels < 1000` 的坏样本是否不再出现 masked gain 高但 raw gain 大幅负的情况

`v9 batch_100` 结果表明，`v9` 修正了 `v8` 的主要虚高问题，但整体还不能证明优于 `v5`：

- 样本数：`600 = 100 routes * 2 frames * 3 history`
- `v9 mean_iou_gain = 0.14168927757920569`
- `v5 mean_iou_gain = 0.14045930654581848`
- `v9 mean_raw_gain = 0.11235761933794228`
- `v5 mean_raw_gain = 0.11567376091741809`
- `v9 positive_rate = 0.94`
- `v5 positive_rate = 0.9533333333333334`
- `v9 bad_rate(iou_gain < -0.05) = 0.016666666666666666`
- `v5 bad_rate(iou_gain < -0.05) = 0.006666666666666667`
- `v9 translation_norm mean = 0.1549054046755023`, `p95 = 0.8505116484999548`, `max = 2.0`
- `v5 translation_norm mean = 0.4472907111717987`, `p95 = 3.252499997615812`, `max = 6.121478644212499`

`v9` 的 score mode 分布：

- `after_score_mode = common_support`: `596 / 600`
- `after_score_mode = dynamic_fallback`: `4 / 600`
- `before_score_mode = common_support`: `580 / 600`
- `before_score_mode = dynamic_fallback`: `20 / 600`

这说明：

- `v9` 成功压住了不可信的大位移，位移分布明显优于 `v5`
- `v9` 把 `v8` 中小 support 高 masked gain、raw gain 大幅负的问题压下来了
- 但 `v9` 的 mean raw gain 仍略低于 `v5`
- `v9` 的坏样本比例反而高于 `v5`
- 因此 `v9` 目前更像是“更保守、更稳定的诊断版本”，还不能作为最终替代 `v5` 的方案

当前更合理的后续方向是继续保留 `v9` 的位移先验，但 refine 选择目标需要同时约束 common-support 和 dynamic/raw 一致性，避免只优化 support 区域后牺牲更大范围 occupancy。

## 可视化与距离分桶

已新增 contact sheet 脚本，用于按 scenario 汇总最值得看的样本：

```bash
python tools/make_lidar_bev_contact_sheet.py \
  --summary /home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100/summary.json \
  --out-html /home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100/contact_sheet_top_scenarios.html \
  --top-scenarios 5 \
  --per-scenario 8 \
  --sort-by norm
```

旧版 `v9 batch_100` 曾只下载 `summary.json`，因此最初的 contact sheet 只能列出缺失 PNG 的预期路径。完整图片下载后，已重新生成可直接查看的 HTML。

旧版目录下已生成：

- `/home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100/contact_sheet_top_scenarios.html`

新版目录下已生成：

- `/home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100_1/contact_sheet_top_scenarios.html`
- `/home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100_1/contact_sheet_worst_gain.html`

已在 `v9` 中加入距离分桶诊断字段。默认分桶为：

- `0-8m`
- `8-16m`
- `16-24m`
- `24-32m`

这些字段只用于诊断，不改变 refinement 搜索目标。`lidar_bev_v9_batch_100_1` 已使用这些字段定位失败主要来自近场，而不是全范围一致下降。

`lidar_bev_v9_batch_100_1` 是加入距离分桶后的完整 v9 重跑结果。它与旧 v9 的整体指标基本一致，说明距离分桶字段没有改变 refinement 行为：

- `v9_new mean_iou_gain = 0.1416891522039797`
- `v9_old mean_iou_gain = 0.14168927757920569`
- `v9_new mean_raw_gain = 0.11235754829812201`
- `v9_old mean_raw_gain = 0.11235761933794228`

距离分桶暴露了更关键的问题：当前主要风险在近场。

- `0-8m`: `mean_gain = 0.10573794669446021`, `mean_raw_gain = -0.07138553283272947`, `raw_bad_rate = 0.44333333333333336`
- `8-16m`: `mean_gain = 0.14806590442514989`, `mean_raw_gain = 0.06335946994218003`, `raw_bad_rate = 0.15166666666666667`
- `16-24m`: `mean_gain = 0.17474196322923435`, `mean_raw_gain = 0.141191947603299`, `raw_bad_rate = 0.04`
- `24-32m`: `mean_gain = 0.16900641484945575`, `mean_raw_gain = 0.14026674701060388`, `raw_bad_rate = 0.006666666666666667`

这说明：

- v9 的全局 masked gain 看起来略正，但近场 raw occupancy 反而平均变差
- 中远场收益是正的，尤其 `16-32m` 很稳定
- 当前不应该继续单纯提高 common-support 或全图 IoU，而应该把近场 raw/dynamic 一致性作为硬约束或更高权重

加入距离分桶后已生成新版 contact sheet：

- `/home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100_1/contact_sheet_top_scenarios.html`
- `/home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100_1/contact_sheet_worst_gain.html`

## 系统性误差排查

基于 `v5/v8/v9 batch_100` 的 residual `dx/dy` 分布，目前没有看到全局固定平移误差。

`v9` 的整体残差如下：

- `mean_dx = -0.03808333340411385`
- `median_dx = 0.0`
- `mean_dy = -0.005583333180596431`
- `median_dy = 0.0`
- `mean_norm = 0.1549054046755023`
- `p95_norm = 0.8505116484999548`
- `max_norm = 2.0`

按 history 分组：

- `history=1`: `mean_dx = -0.05449999952688813`, `mean_dy = 0.009250000026077032`, `mean_norm = 0.1050786859480983`
- `history=3`: `mean_dx = 0.0034999992325901986`, `mean_dy = -0.0072500001825392246`, `mean_norm = 0.12087739739400295`
- `history=5`: `mean_dx = -0.06324999991804361`, `mean_dy = -0.0187499993853271`, `mean_norm = 0.23876013068440557`

按 Town 分组：

- `Town12`: `mean_dx = -0.06509433953829531`, `mean_dy = -0.007547169630913614`
- `Town13`: `mean_dx = -0.007624113720887941`, `mean_dy = -0.0033687942047068414`

这说明：

- 之前两个重点坏例子中出现的 `dy ~= 0.8m` 不是全局系统偏差
- `history` 增大时 residual norm 会变大，但 signed `dx/dy` 没有稳定朝同一方向漂移
- `v5` 中的大 residual 主要是少数 route 的搜索发散，尤其表现为负 `dx` 大 outlier
- `v9` 把这些 outlier 压住后，仍剩下一些场景级残差，但不像统一坐标系或外参错误

更像存在问题的是评价目标，而不是 measurement transform 存在一个全局固定偏移。

当前残差较大的 scenario 主要包括：

- `Town12_Rep0_1498`: `mean_norm = 0.7178023374040906`, `mean_dx = -0.41666666852931183`, `mean_dy = -0.11666667088866234`
- `Town12_Rep0_729`: `mean_norm = 0.516666658843557`, `mean_dx = -0.4249999901900689`, `mean_dy = -0.09166666865348816`
- `Town12_Rep0_3817`: `mean_norm = 0.5084875996985095`, `mean_dx = -0.4749999977648258`, `mean_dy = -0.008333333457509676`
- `Town13_Rep0_1073`: `mean_norm = 0.4963147653447017`, `mean_dx = 0.11944444311989678`, `mean_dy = 0.13611111210452187`
- `Town12_Rep0_2882`: `mean_norm = 0.4821183818540708`, `mean_dx = -0.41666666604578495`, `mean_dy = -0.016666666915019352`

后续如果继续排查系统误差，应优先做 scenario 级可视化，而不是假设全局外参或坐标系偏移。

## 下一步建议

下一步优先级如下：

- 先人工查看 `/home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100_1/contact_sheet_top_scenarios.html` 和 `/home/heavenlysu/sitp_workspace/lidar_bev_v9_batch_100_1/contact_sheet_worst_gain.html`，确认近场变差的视觉模式。
- 如果近场确实被 refinement 伤害，下一版 scorer 应加入 `0-8m` raw/dynamic gain 约束。
- 如果近场问题主要来自动态物体或遮挡，应优先改 mask 或过滤策略，而不是继续调 residual search。
- 如果某些 scenario 有稳定局部偏差，应做 scenario 级专项分析，不应上升为全局外参修正。
- 批量验证仍使用 `batch_100` 作为第一关，只有同时满足 raw gain 不下降、bad rate 不上升、translation norm 不发散，才扩大到更多 route。

下一版候选方向：

- `v10` 可以保留 `v9` 的 residual-motion prior 和 support fallback。
- `v10` 的 score 应同时考虑 common-support IoU、dynamic IoU、近场 raw/dynamic IoU。
- `v10` 应把 `0-8m` raw/dynamic gain 作为硬约束或高权重项，避免中远场收益掩盖近场退化。


## 仍然失败的部分

即使剔除了动态目标，并加入局部 refinement，仍然存在对齐后 raw IoU 变差的样本。

当前最典型的失败样本包括：

- `Town12_Rep0_2488_0_route0_11_08_02_55_17`
- `Town12_Rep0_2488_1_route0_11_08_17_28_24`

其中重点样本如下：

- `Town12_Rep0_2488_0_route0_11_08_02_55_17`, `frame=35`, `history=5`
  - `v3`: `masked_gain = -0.2243713463315527`
  - `v4`: `masked_gain = -0.09075109069354934`
  - `v5`: `masked_gain = -0.08266024156334817`
  - `v7`: `masked_gain = 0.02155371373728293`, `raw_gain = -0.1143449668584853`
  - `v8`: `masked_gain = 0.0524648118605377`, `raw_gain = -0.07013995171224305`

- `Town12_Rep0_2488_1_route0_11_08_17_28_24`, `frame=38`, `history=5`
  - `v3`: `masked_gain = -0.21242163916148643`
  - `v4`: `masked_gain = -0.05401660537122796`
  - `v5`: `masked_gain = -0.05463022115993085`
  - `v7`: `masked_gain = 0.10058318547124515`, `raw_gain = -0.04788894332650179`
  - `v8`: `masked_gain = 0.0591520139030873`, `raw_gain = -0.046310534539907994`

这些结果说明：

- `v4/v5` 已经证明这类样本存在局部可修正误差
- `v6` 证明固定矩形 ROI 会引入新的错误，不适合作为当前方案
- `v7` 证明 common-support 打分能减少坏区域污染，但也可能诱导过大的 `dx`
- `v8` 证明加入 residual-motion prior 后，可以保留 common-support 的收益，同时避免明显不可信的大平移
- `batch_100` 证明 `v8` 的 masked 指标存在口径不一致和小 support 虚高问题
- `v9` 已修正指标口径，并压住大位移，但 batch 结果仍未优于 `v5`
- residual 统计没有显示全局固定平移误差，问题更像是场景级残差和评价目标不稳定
- 目前的全图 occupancy IoU 目标函数仍然不够稳定，无法把这些样本完全翻正
- 下一步更合理的方向是在 `v9` 基础上加入 dynamic/raw 一致性约束，再重新跑 `batch_100`

这些 route 仍然是后续人工排查和新指标测试的优先对象。
