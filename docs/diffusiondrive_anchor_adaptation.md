# DiffusionDrive Anchor 适配方案

本文档记录当前 anchor 文件事实、源文件一致性校验、现有模型接口约束，以及直接升级到最新聚类 anchor 的实现路线。

## 当前文件事实

仓库工作区当前有三份相关文件：

- `/home/HeavenlySU/sitp_workspace/plan_anchor.npy`
  - NPY header: `shape = (20, 8, 2)`
  - 当前 CARLA port 默认兼容格式。
- `/home/HeavenlySU/sitp_workspace/4-0-0-1910-tracked_clusters_anchor.npy`
  - NPY header: `shape = (99, 10, 2)`
  - 最新聚类 anchor。
- `/home/HeavenlySU/sitp_workspace/4-0-0-1910-tracked_clusters.json`
  - `99` 个 cluster。
  - 每个 cluster 包含 `cluster_id`、`mu`、`var`。
  - 每个 `mu` 长度为 `20`，语义是拉直后的 `10` 个 `(x, y)` 空间 checkpoint 点。

## Anchor 语义估计

当前没有拿到完整生成日志，但从 anchor 几何和 syb / garage DPMM 脚本可以推断：

- anchor 来自旧 garage / syb 的 `route` checkpoint 聚类，而不是 fixed-time future ego trajectory 聚类。
- `syb_carla_garage/my_dpmm_model/b2d_traj_fit_dpmm_by_ability.py` 中聚类输入是 `batch["route"].reshape(...)`。
- 旧 dataset 的 `smooth_path()` / `iterative_line_interpolation()` 逻辑是第一个 route point 约 `2.5m`，之后每个点间隔 `1.0m`。
- 当前 `99x10x2` anchor 的相邻点距离中位数约 `1.0m`，第一个点距离约 `2.5m`，最后一个点约 `11.5m`。

因此当前 anchor 应按空间 route/checkpoint anchor 理解，不应解释为 `0.5s` 或 `1.0s` 的 time-based trajectory anchor。

## 源文件一致性校验

已用不依赖 numpy 的 NPY reader 对 `4-0-0-1910-tracked_clusters_anchor.npy` 和 `4-0-0-1910-tracked_clusters.json` 做逐项校验：

- JSON cluster 数：`99`
- cluster id 前后范围示例：`[1, 2, 3] ... [100, 101, 102]`
- NPY shape：`(99, 10, 2)`
- JSON 期望 shape：`(99, 10, 2)`
- flat value 数量：`1980 == 1980`
- 最大绝对误差：`0.0`
- `>1e-9` mismatch 数：`0`

结论：当前 `.npy` anchor 与 JSON 中的 `mu` 完全一致，可作为迁移基线。

## 当前代码约束

迁移前推理链路默认假设：

- `DiffusionDriveConfig.trajectory_sampling.num_poses = 8`
- `config_adapter.validate_diffusiondrive_config()` 要求 anchor 是 `(20, 8, 2)`
- `TrajectoryHead.plan_anchor` 从 `DIFFUSIONDRIVE_ANCHOR_PATH` 读取
- `norm_odo()` / `denorm_odo()` 当前只处理 `(x, y)`
- 模型最终仍输出 `(x, y, heading)`，agent 侧只取 `(x, y)` 给 PID

迁移前模型内部对 mode 数的情况比较微妙：

- `TrajectoryHead.forward_*()` 中很多张量会跟随 `plan_anchor.shape[0]` 得到实际 mode 数。
- `DiffMotionPlanningRefinementModule.forward()` 也从 `traj_feature.shape[1]` 读取实际 mode 数。
- 当时代码里仍有 `ego_fut_mode=20`、注释、校验和 checkpoint 假设写死为 20。
- 因此 `99` 个 mode 理论上不一定需要重写所有层，但 checkpoint 兼容、state dict 中的 `plan_anchor` 形状、分类分支语义和训练标签都必须重新确认。

pose 数更硬：

- `GridSampleCrossBEVAttention` 的 `num_points` 来自 `num_poses`。
- `DiffMotionPlanningRefinementModule.plan_reg_branch` 输出 `ego_fut_ts * 3`。
- gather、loss 和 PID 都按输出时间步数运行。
- 所以要支持 `10` 个 pose，必须同步改 `trajectory_sampling`、config 校验、训练 target 长度和控制侧假设。

## 当前选择：正式升级为 `99x10x2`

目标：训练和推理都围绕最新 anchor 重新定义。

当前实现已经将轨迹主接口统一为 `99x10x2`：

- `DiffusionDriveConfig.num_anchor_modes` 跟随 anchor 第一维，当前为 `99`。
- `trajectory_sampling.time_horizon` 当前仍随 pose 数和兼容字段 `interval_length` 得到 `10 * 0.5s = 5.0s`，但这不代表当前 anchor 的真实时间语义。
- `DiffMotionPlanningRefinementModule.plan_reg_branch` 输出 `ego_fut_ts * 2`。
- `TrajectoryHead.forward_train()/forward_test()`、loss、agent PID 入口均使用 `(x, y)`，不再传递 heading。

已同步修改：

1. `DiffusionDriveConfig`
   - 支持 `trajectory_sampling.num_poses = 10`。
   - 增加显式 `num_anchor_modes`，并由 anchor shape 推导。
2. `config_adapter.validate_diffusiondrive_config()`
   - 不再硬编码 mode 必须为 `20`。
   - 校验 mode 数和 pose 数与 config 一致。
3. `TrajectoryHead` / decoder
   - 将 decoder 的 mode 数改为 `config.num_anchor_modes`。
   - 将 `plan_anchor_encoder` 的输入维度从固定 `512` 改成 `64 * num_poses`，以支持 `10` pose。
   - 增加 anchor pose 数与 `trajectory_sampling.num_poses` 的显式校验。
   - 更新误导性 shape 注释。
4. Loss / target builder
   - `LossComputer` 的注释和 shape 假设改成动态 mode / pose。
   - loss 仅监督 XY；若训练 target 暂时保留 heading，也只取前两维。
   - 默认 CARLA / B2D Full 训练 target 已切换为 `spatial_path`，按 `2.5m, 3.5m, ..., 11.5m` 空间距离重采样到 `10` 个 pose。
5. 控制器
   - `DiffusionDriveAgent` 默认启用空间 checkpoint PID，不再通过 `carla_fps // (wp_dilation * data_save_freq)` 把 waypoint index 解释成 0.5s / 1.0s 时间点。
   - 当前空间 PID 用路径横向偏移 / endpoint turn ratio 在 slow / fast 目标速度之间插值，转向仍按满足 `aim_distance` 的空间 waypoint 选取 aim point。
   - 旧的 time-index desired speed 逻辑仍可通过 `DIFFUSIONDRIVE_SPATIAL_PID=0` 作为 A/B fallback。
6. Checkpoint 策略
   - 原 `20x8x2` checkpoint 不能无缝代表 `99x10x2` 分类语义。
   - 可以部分加载 backbone / transformer，trajectory head 和 anchor 相关参数应预期 mismatch 或重新训练。

优点：

- 保留最新聚类结果完整信息。
- 训练目标、anchor、输出 horizon 可以一次性统一。

缺点：

- 已经不是纯推理侧替换；需要训练侧同步。
- 和原 NAVSIM checkpoint 兼容性弱。
- 需要重新验证控制器、loss、评估和日志。

## 运行注意

升级后应直接将：

```bash
export DIFFUSIONDRIVE_ANCHOR_PATH=/home/HeavenlySU/sitp_workspace/4-0-0-1910-tracked_clusters_anchor.npy
```

预期 checkpoint 加载时，旧 `20x8x2` 权重中与 trajectory head / plan anchor / pose attention 相关的若干 tensor 会 shape mismatch 并被跳过。这是接口升级后的正常现象，但也意味着旧 checkpoint 不能作为最终性能判断基线；正式效果需要配套 CARLA 训练或至少针对新 head 微调。
