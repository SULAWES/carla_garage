# DiffusionDrive 后续优化建议

本文档记录 `DiffusionDriveAgent` 当前可运行版本的主要性能瓶颈，以及后续建议的优化方向。目标不是“先能跑”，而是逐步把输入分布、时序建模、控制执行和权重对齐做扎实。

## 1. 当前实现概况

当前链路已经完整：

1. CARLA 传感器进入 `DiffusionDriveAgent`
2. `tick()` 生成图像、LiDAR、UKF 状态、导航指令
3. `run_step()` 构造 `camera_feature / lidar_feature / status_feature`
4. `V2TransfuserModel` 输出未来轨迹
5. `_control_pid()` 把轨迹转成控制量

这套流程工程上已经正确，但效果较差，主要因为当前实现为了优先打通链路，做了较多简化。

## 2. 优化优先级

### P0：必须优先做

#### 2.1 Checkpoint 对齐

当前最基础但也最容易被忽略的问题，是 checkpoint 是否真的和本地模型结构对齐。

需要关注：

- checkpoint 是否带有 `agent.` / `model.` / `_transfuser_model.` / `module.` 等包装前缀
- 本地 `V2TransfuserModel` 是否和训练时模块命名一致
- 是否存在 shape mismatch
- 是否有大批关键权重根本没有加载进去

如果这一层不对，后续调输入和控制意义不大。

当前已确认的一点是：

- 真实 checkpoint 几乎全部加载成功
- 唯一未对齐的是 `status` 编码层
- checkpoint 中 `_status_encoding.weight` 形状为 `(256, 8)`
- 当前 CARLA 版本模型定义为 `(256, 10)`

这说明当前主要矛盾不是主干权重缺失，而是 `status_feature` 的定义和训练时不一致。

已从原始训练代码确认：

- 训练侧 `status_feature = driving_command + ego_velocity(2) + ego_acceleration(2)`
- 因此 `driving_command` 是 4 维，整条状态向量总共 8 维

当前已进一步确认，这 4 维 `driving_command` 的语义为：

- dim 0: `left`
- dim 1: `straight`
- dim 2: `right`
- dim 3: `unknown`

其中：

- `left/right` 不仅包含转弯，也包含 lane change 和 sharp curve
- `unknown` 是 NAVSIM 官方定义中的第 4 类，可用于训练时过滤不确定样本
- 但当前这份仓库代码里，并没有默认启用基于 `unknown` 的筛样逻辑，`driving_command` 只是被直接送入模型

因此后续改动时，不能只把维度改回 8，还必须确认 CARLA 的 6 类 command 应该如何映射到这 4 类 NAVSIM command。

#### 2.2 多帧 LiDAR 时序

当前 `DiffusionDriveAgent` 只使用单帧 LiDAR histogram，没有移植 `sensor_agent.py` 中完整的：

- `lidar_buffer`
- 半帧补全
- `align_lidar`
- 时序重对齐

这会直接损失动态目标和局部几何的稳定性。对 TransFuser / DiffusionDrive 这类方法，这是高收益项。

#### 2.3 预处理和训练保持一致

需要严格核对训练时的：

- 图像裁剪方式
- resize 方式
- 是否使用 ImageNet mean/std normalization
- 是否保留 jpeg artifact
- LiDAR histogram 参数和坐标轴约定
- `status_feature` 的定义和归一化

当前“维度对得上”不等于“分布对得上”。

### P1：次高优先级

#### 2.4 `status_feature` 重构

当前状态向量是：

- `command(6)`
- `velocity = [speed, 0]`
- `acceleration = [a_long, 0]`

这是一个可运行的简化版，但不一定等于训练侧使用的状态定义。应核查训练代码，必要时改成真正一致的：

- 2D velocity
- 2D acceleration
- heading / yaw rate
- 归一化后的 command/state

当前已确认：

- 训练侧不是 6 维 command，而是 4 维 `driving_command`
- 当前 CARLA 侧的 `command(6) + velocity(2) + acceleration(2)` 会导致 `_status_encoding` 这一层无法吃到 checkpoint 权重
- 这 4 维 command 的官方语义是 `left / straight / right / unknown`

#### 2.5 控制器调优

当前控制器是 classic PID，适合“先跑通”，不一定适合 diffusion 输出轨迹。

优先调这些参数：

- `aim_distance_fast`
- `aim_distance_slow`
- `aim_distance_threshold`
- `brake_speed`
- `brake_ratio`
- 目标速度估计窗口

要先区分“轨迹本身差”还是“轨迹还行但控制器执行差”。

#### 2.6 diffusion 推理步数

当前 `forward_test()` 的 `step_num = 2`，这是明显偏快的设置，更像是为了先保证推理可用。

建议逐步试验：

- `step_num = 5`
- `step_num = 10`

一般会提升轨迹平滑性和模式选择稳定性，但推理耗时会增加。

### P2：中长期优化

#### 2.7 Route conditioning 增强

当前模型输入中主要用了 command，没有充分使用 route 几何信息。

后续可考虑：

- 将 `target_point` 直接作为模型输入
- 加入 `target_point_next`
- 引入 route polyline 或 route token

尤其在复杂路口和连续转向场景下，这通常比单独 one-hot command 更有效。

#### 2.8 多相机视角

当前只用了前视相机。如果训练侧本来依赖多视角，当前感知范围天然受限。

如果后续要追求更高上限，可以引入：

- 左前/右前拼接
- 更宽 FOV
- 与训练侧一致的多相机布局

#### 2.9 利用辅助头

如果模型本身还保留：

- agent box
- BEV semantic
- 其他辅助输出

可以将它们用于：

- 训练时多任务约束
- 推理时安全检查
- debug 可视化和误差分析

## 3. 建议的实施顺序

建议按下面顺序推进，避免同时改太多导致无法归因：

1. 完成 checkpoint key/shape 对齐，并记录加载摘要
2. 对齐训练侧输入预处理
3. 移植多帧 LiDAR 与时序对齐
4. 校准 `status_feature`
5. 调 PID 控制器
6. 增大 diffusion 推理步数
7. 再尝试 route conditioning / 多相机

## 4. 建议的实验方式

每次只改一类因素，并记录：

- route 完成率
- 平均速度
- 越线/碰撞/闯灯类型
- 轨迹可视化
- checkpoint 加载摘要

推荐先做三类消融：

1. 单帧 LiDAR vs 多帧 LiDAR
2. 当前状态向量 vs 训练侧严格一致状态向量
3. `step_num = 2/5/10`

这样最容易定位收益来源。
