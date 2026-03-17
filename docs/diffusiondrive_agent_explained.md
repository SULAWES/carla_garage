# `diffusiondrive_agent.py` 说明

本文档解释 [`diffusiondrive_agent.py`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py) 的职责、执行流程、关键数据结构，以及它和 `sensor_agent.py` 的关系。

## 1. 这个 agent 在做什么

`DiffusionDriveAgent` 是一个给 CARLA leaderboard 用的推理 agent。它的工作可以概括成一条链路：

1. 从 CARLA 读取前视相机、LiDAR、IMU、GNSS、速度计数据。
2. 用 `RoutePlanner` 和 UKF 估计车辆当前状态，并生成导航指令。
3. 把 CARLA 传感器数据整理成 DiffusionDrive 需要的三类输入：
   - `camera_feature`
   - `lidar_feature`
   - `status_feature`
4. 调用 `V2TransfuserModel` 输出未来轨迹。
5. 取轨迹前两维 `(x, y)` 作为 waypoints，交给 PID 控制器生成 `steer / throttle / brake`。

这个实现本质上是：

- 感知和轨迹预测用 DiffusionDrive
- 路由、状态滤波和低层控制复用 `carla_garage` 现有逻辑

## 2. 入口与生命周期

### 2.1 入口函数

文件中的 `get_entry_point()` 返回 `DiffusionDriveAgent`，这是 leaderboard 加载 agent 的入口。

位置：

- [`diffusiondrive_agent.py:35`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L35)

### 2.2 `setup()`

`setup()` 负责完成一次评测 route 前的初始化。

位置：

- [`diffusiondrive_agent.py:46`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L46)

它主要做了几件事：

- 创建 CARLA 侧配置 `GlobalConfig`
- 创建 DiffusionDrive 侧配置 `DiffusionDriveConfig`
- 把 CARLA 的 LiDAR 边界、分辨率等参数同步到 `dd_config`
- 从环境变量读取：
  - `DIFFUSIONDRIVE_ANCHOR_PATH`
  - `DIFFUSIONDRIVE_CHECKPOINT`
  - `DIFFUSIONDRIVE_BACKBONE_PATH`
- 实例化 `V2TransfuserModel`
- 加载 checkpoint
- 初始化 PID 控制器
- 初始化 UKF 和状态缓存

### 2.3 `run_step()`

`run_step()` 是每个仿真 step 调用一次的主循环。

位置：

- [`diffusiondrive_agent.py:362`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L362)

主流程如下：

1. 第一次进入时调用 `_init()` 建立 route planner，并返回刹车控制。
2. 后续每一步先调用 `tick()` 处理原始传感器。
3. 把 LiDAR 转成 histogram BEV。
4. 把图像归一化到 `[0,1]` 并 resize 到 DiffusionDrive 配置尺寸。
5. 通过 `_build_status()` 生成状态向量。
6. 调用模型预测轨迹。
7. 提取 `(x, y)` waypoints，并通过 `_control_pid()` 生成控制量。

### 2.4 `destroy()`

`destroy()` 用于 route 结束后的清理。

位置：

- [`diffusiondrive_agent.py:405`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L405)

这里专门覆写了 `destroy(self, results=None)`，原因是本地 evaluator 会调用 `destroy(results)`；如果不覆写，Python 会落到基类 `AutonomousAgent.destroy(self)`，导致参数不匹配报错。

## 3. 关键函数解释

### 3.1 `_load_checkpoint()`

位置：

- [`diffusiondrive_agent.py:129`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L129)

作用：

- 加载 `.pth/.ckpt`
- 兼容常见前缀：
  - `agent.`
  - `model.`
  - `module.`
- 用 `strict=False` 加载，便于移植初期做权重对齐

如果你后续要严格检查模型结构一致性，这里可以改成 `strict=True`，或者显式打印并处理 key mapping。

### 3.2 `_init()`

位置：

- [`diffusiondrive_agent.py:157`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L157)

作用：

- 计算地理参考点 `lat_ref / lon_ref`
- 构建 `RoutePlanner`
- 把全局路线写入 planner

这个函数基本沿用了 `sensor_agent.py` 的思路，优先使用 CARLA map 的地理坐标转换；如果失败，再退回到数值估计。

### 3.3 `sensors()`

位置：

- [`diffusiondrive_agent.py:191`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L191)

声明了 agent 需要的传感器：

- 前视 RGB 相机 `rgb_front`
- `imu`
- `gps`
- `speed`
- `lidar`

这里没有像原始 `sensor_agent` 那样支持更多相机，也没有启用额外 debug sensor。

### 3.4 `tick()`

位置：

- [`diffusiondrive_agent.py:243`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L243)

这是输入预处理的核心。

它做了几件事：

- 图像处理：
  - 读取 `rgb_front`
  - 人为做一次 jpeg encode/decode，尽量贴近训练时的数据分布
  - BGR -> RGB
  - 调用 `t_u.crop_array()` 裁剪
  - 转成 `C,H,W` 的 tensor
- LiDAR 处理：
  - 用 `t_u.lidar_to_ego_coordinate()` 把点云转换到 ego 坐标系
- 状态估计：
  - 使用 UKF 融合 `gps + compass + speed`
  - 更新 `state_log`
- 导航处理：
  - 用 `RoutePlanner` 输出 route point
  - 生成 one-hot command
  - 生成 ego 坐标系下的 `target_point`

注意：

- 当前命令缓存使用 `self.commands[-2]`，这和 `sensor_agent` 的处理保持一致，用来减小指令抖动。

### 3.5 `_build_status()`

位置：

- [`diffusiondrive_agent.py:308`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L308)

这个函数把 CARLA 当前状态拼成 DiffusionDrive 的 `status_feature`。

当前定义是：

- `command`: 6 维 one-hot
- `velocity`: 2 维，实际填的是 `[speed, 0.0]`
- `acceleration`: 2 维，实际填的是 `[longitudinal_accel, 0.0]`

其中加速度是通过当前速度和上一帧速度差分得到的：

- `accel = (speed - prev_speed) / dt`

所以当前 `status_feature` 不是一个完整的 2D 动力学状态，而是一个“简化版状态输入”。

### 3.6 `_control_pid()`

位置：

- [`diffusiondrive_agent.py:321`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L321)

这个函数把模型输出轨迹转成控制信号。

逻辑基本继承自 `carla_garage` 现有 waypoint PID：

- 用未来 waypoint 间距估计目标速度
- 根据 `brake_speed` 和 `brake_ratio` 判断是否刹车
- 用纵向 PID 控 throttle
- 选取满足 `aim_distance` 的 waypoint 作为转向目标
- 用横向 PID 输出 steer

也就是说，这个 agent 当前不是“直接输出控制量”，而是“输出轨迹，再用经典控制器执行”。

## 4. 模型输入输出是怎么接起来的

在 `run_step()` 里，模型输入是这样组织的：

- `camera_feature`
  - 来自 `tick_data['rgb']`
  - 除以 `255.0`
  - resize 到 `(dd_config.camera_height, dd_config.camera_width)`
- `lidar_feature`
  - 来自 `CARLA_Data.lidar_to_histogram_features()`
- `status_feature`
  - 来自 `_build_status()`

对应位置：

- [`diffusiondrive_agent.py:374`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L374)
- [`diffusiondrive_agent.py:384`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L384)
- [`diffusiondrive_agent.py:386`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L386)

模型输出中，当前真正用于控制的只有：

- `outputs['trajectory']`

然后只取前两维：

- `traj[:, :, :2]`

位置：

- [`diffusiondrive_agent.py:392`](/home/heavenlysu/sitp_workspace/carla_garage/team_code/diffusiondrive_agent.py#L392)

这意味着：

- 轨迹中的 heading 当前没有进入控制器
- agent box / semantic 等头虽然模型可能会输出，但目前未被本 agent 使用

## 5. 与 `sensor_agent.py` 的关系

这个文件并不是完全从零写的，它实际上是“部分复用 `sensor_agent.py` 的外围逻辑 + 替换中间模型”。

保留的部分：

- 传感器组织方式
- `RoutePlanner`
- UKF 状态估计
- command 缓存逻辑
- waypoint PID 控制思路

替换掉的部分：

- 不再用原 `LidarCenterNet`
- 不再走 `pred_wp / pred_checkpoint / pred_target_speed` 这套输出头
- 改为直接调用 `V2TransfuserModel`
- 改为使用 DiffusionDrive 的 anchor + diffusion trajectory head

可以把它理解为：

- `sensor_agent` 提供 CARLA 接入范式
- `diffusiondrive_agent` 把中间的感知/规划模型替换成了 DiffusionDrive

## 6. 当前实现的限制

### 6.1 只使用单前视相机

当前只注册了一个 `rgb_front`，没有拼接多相机视角。

### 6.2 LiDAR 目前是单帧 histogram

当前实现没有把 `sensor_agent` 里的历史 LiDAR buffer、半帧对齐、realign 逻辑完整搬过来，因此多帧时序信息目前没有真正发挥出来。

### 6.3 `status_feature` 是简化版

横向速度和横向加速度目前直接填 `0`，这对模型是否足够，需要以后结合训练配置继续确认。

### 6.4 控制器仍然是 classic PID

DiffusionDrive 当前只负责轨迹生成，真正执行层仍是 garage 原有 PID。这样实现简单稳妥，但和“端到端直接输出控制”的范式不是一回事。

## 7. 后续最可能继续改的地方

如果后面要继续迭代这个 agent，优先关注这几块：

1. 多帧 LiDAR 与时序对齐
2. `status_feature` 的定义是否与训练侧完全一致
3. 图像归一化是否需要 mean/std normalization
4. 是否要利用模型的 heading / 检测 / 语义输出
5. 是否要把 PID 控制替换成更贴合 DiffusionDrive 设定的执行器

