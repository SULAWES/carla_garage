# `diffusiondrive_agent.py` 说明

本文档解释 `carla_garage/team_code/diffusiondrive_agent.py` 的职责、执行流程、关键数据结构，以及它和 `sensor_agent.py` 的关系。

## 1. 这个 agent 在做什么

`DiffusionDriveAgent` 是一个给 CARLA leaderboard 用的推理 agent。当前链路可以概括为：

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

### 2.2 `setup()`

`setup()` 负责完成一次评测 route 前的初始化。主要工作包括：

- 创建 CARLA 侧配置 `GlobalConfig`
- 创建 DiffusionDrive 侧配置 `DiffusionDriveConfig`
- 把 CARLA 的 LiDAR 边界、分辨率、`lidar_seq_len`、`use_ground_plane` 等参数同步到 `dd_config`
- 从环境变量读取：
  - `DIFFUSIONDRIVE_ANCHOR_PATH`
  - `DIFFUSIONDRIVE_CHECKPOINT`
  - `DIFFUSIONDRIVE_BACKBONE_PATH`
  - `DIFFUSIONDRIVE_COMMAND_DELAY`
  - `DIFFUSIONDRIVE_SPATIAL_PID`
  - `DIFFUSIONDRIVE_DEBUG_CONTROL`
  - `DIFFUSIONDRIVE_DEBUG_INTERVAL`
- 实例化 `V2TransfuserModel`
- 加载 checkpoint
- 初始化 PID 控制器
- 初始化 UKF、状态缓存、LiDAR buffer

补充说明：

- 当前主线 anchor 是 `4-0-0-1910-tracked_clusters_anchor.npy`，shape 为 `99x10x2`。
- 这份 anchor 来自 `99` 个 cluster，每个 cluster 的 `mu` 都是一条拉直后的 `10` 个 `(x, y)` 空间 checkpoint 轨迹。
- 当前模型、loss、训练 target 和推理控制入口都已按 `99x10x2` / XY-only 轨迹组织。

### 2.3 `run_step()`

`run_step()` 是每个仿真 step 调用一次的主循环。当前主流程是：

1. 第一次进入时调用 `_init()` 建立 route planner，并返回刹车控制。
2. 后续每一步先调用 `tick()` 处理原始传感器。
3. 将上一帧半帧 LiDAR 对齐到当前车体坐标系。
4. 将当前半帧与上一半帧拼接成完整扫描，并写入 `lidar_buffer`。
5. 在 buffer 未填满时持续刹车等待。
6. 从 buffer 中取出单帧或多帧 LiDAR，必要时将历史帧 realign 到当前坐标系。
7. 把图像归一化到 `[0,1]` 并 resize 到 DiffusionDrive 配置尺寸。
8. 通过 `_build_status()` 生成状态向量。
9. 调用模型预测轨迹。
10. 提取 `(x, y)` waypoints，并通过 `_control_pid()` 生成控制量。

### 2.4 `destroy()`

`destroy()` 用于 route 结束后的清理。

这里专门覆写了 `destroy(self, results=None)`，原因是本地 evaluator 会调用 `destroy(results)`；如果不覆写，Python 会落到基类 `AutonomousAgent.destroy(self)`，导致参数不匹配报错。

## 3. 关键函数解释

### 3.1 `_load_checkpoint()`

作用：

- 加载 `.pth/.ckpt`
- 兼容常见前缀：
  - `agent.`
  - `model.`
  - `module.`
  - `_transfuser_model.`
- 只加载 key 和 shape 都匹配的参数
- 打印 missing / unexpected / shape mismatch 摘要，方便权重对齐

### 3.2 `_init()`

作用：

- 计算地理参考点 `lat_ref / lon_ref`
- 构建 `RoutePlanner`
- 把全局路线写入 planner

这个函数基本沿用了 `sensor_agent.py` 的思路，优先使用 CARLA map 的地理坐标转换；如果失败，再退回到数值估计。

### 3.3 `sensors()`

当前注册的传感器有：

- 前视 RGB 相机 `rgb_front`
- `imu`
- `gps`
- `speed`
- `lidar`

这里没有像 navsim 原版那样拼接多相机，也没有像 `sensor_agent.py` 那样开启额外 debug sensor。

### 3.4 `tick()`

这是输入预处理的核心。

它做了几件事：

- 图像处理：
  - 读取 `rgb_front`
  - 做一次 jpeg encode/decode，尽量贴近训练时的数据分布
  - BGR -> RGB
  - 调用 `t_u.crop_array()` 裁剪
  - 转成 `C,H,W` tensor
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

- DD 默认使用当前 `far_command.value` 构造 `status_feature`，与训练侧当前 command 语义对齐。
- 旧 garage / `sensor_agent.py` 的 `self.commands[-2]` 一拍延迟逻辑仍可通过 `DIFFUSIONDRIVE_COMMAND_DELAY=1` 启用，用于闭环 A/B。

### 3.5 `_build_status()`

这个函数把 CARLA 当前状态拼成 DiffusionDrive 的 `status_feature`。

当前 `DiffusionDriveAgent` 的实现里，这个函数显式构造的是：

- `command`: 6 维 one-hot
- `speed`: 1 维原始速度

需要注意不要把它和 `carla_garage/team_code/model.py` 中的 `extra_sensors` 机制混在一起：

- `DiffusionDriveAgent` 当前走的是显式 `status_feature`
- `extra_sensors` 是 garage 原模型里的另一套可选输入分支，会按配置拼接 `velocity(1)` 和 / 或 `discrete_command(6)`，再送入 `extra_sensor_encoder`

这是一份面向当前 CARLA 推理链路的状态输入定义。当前 DD 主线已经在训练和推理侧共用 `diffusiondrive.status` 中的 builder，不再保留独立 `extra_sensors` 分支。

### 3.6 `_control_pid()`

这个函数把模型输出轨迹转成控制信号。

逻辑基本继承自 `carla_garage` 现有 waypoint PID：

- 默认按空间 checkpoint 几何估计目标速度，不把 waypoint index 当作固定时间间隔
- 根据 `brake_speed` 和 `brake_ratio` 判断是否刹车
- 用纵向 PID 控 throttle
- 选取满足 `aim_distance` 的 waypoint 作为转向目标
- 用横向 PID 输出 steer

旧 time-index desired speed 逻辑仍可通过 `DIFFUSIONDRIVE_SPATIAL_PID=0` 启用。默认空间 PID 的速度估计只是一版保守闭环控制启发式，后续仍建议通过 CARLA / Bench2Drive 闭环评测调参，或改为显式 speed head。

也就是说，这个 agent 当前不是“直接输出控制量”，而是“输出轨迹，再用经典控制器执行”。

### 3.7 控制调试日志

闭环 A/B 时可设置：

```bash
export DIFFUSIONDRIVE_DEBUG_CONTROL=1
export DIFFUSIONDRIVE_DEBUG_INTERVAL=20
```

日志会按 step 间隔打印当前 / delayed / 实际使用 command、PID mode、desired speed、turn ratio、endpoint distance、aim waypoint、最终 control、stuck / force_move / stop sign 状态。默认关闭，避免长跑日志过噪。

远端闭环运行时需要同步 `team_code/diffusiondrive_agent.py` 和 `team_code/config.py`。新版 agent 会读取 `GlobalConfig.diffusiondrive_spatial_pid*` 参数；如果只同步 agent 而没有同步 config，setup 阶段会报 `GlobalConfig` 缺少 `diffusiondrive_spatial_pid`。

## 4. 模型输入输出是怎么接起来的

在 `run_step()` 里，模型输入是这样组织的：

- `camera_feature`
  - 来自 `tick_data['rgb']`
  - 除以 `255.0`
  - resize 到 `(dd_config.camera_height, dd_config.camera_width)`
- `lidar_feature`
  - 来自 `CARLA_Data.lidar_to_histogram_features()`
  - 可来自单帧或多帧 LiDAR buffer
- `status_feature`
  - 来自 `_build_status()`

模型输出中，当前真正用于控制的只有：

- `outputs['trajectory']`

然后只取前两维：

- `traj[:, :, :2]`

这意味着：

- 轨迹中的 heading 当前没有进入控制器
- agent box / semantic 等头虽然模型会输出，但目前未被本 agent 使用

## 5. 与 `sensor_agent.py` 的关系

这个文件并不是完全从零写的，它实际上是“部分复用 `sensor_agent.py` 的外围逻辑 + 替换中间模型”。

保留的部分：

- 传感器组织方式
- `RoutePlanner`
- UKF 状态估计
- command 缓存逻辑
- waypoint PID 控制思路
- LiDAR 半帧拼接、多帧 buffer 和 realign 的基本做法

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

### 6.2 缺少安全与恢复逻辑

当前还没有迁入 `sensor_agent.py` 里的：

- stuck detection
- creep / force move
- safety box
- stop sign controller

所以当前版本更偏“先跑通模型链路”，而不是“补齐全部运行时保护”。

### 6.3 双 config 仍然并存

当前同时维护 `GlobalConfig` 和 `DiffusionDriveConfig` 两个配置对象，模型输入相关参数需要手动同步，维护成本偏高。

### 6.4 控制器仍然是 classic PID

DiffusionDrive 当前只负责轨迹生成，真正执行层仍是 garage 原有 PID。这样实现简单稳妥，但和“端到端直接输出控制”的范式不是一回事。

## 7. 后续最可能继续改的地方

如果后面要继续迭代这个 agent，优先关注这几块：

1. stuck detection / safety box / stop sign 等运行时保护逻辑
2. 图像归一化是否需要更严格对齐训练侧
3. 是否要利用模型的 heading / 检测 / 语义输出
4. 是否要把 PID 控制替换成更贴合 DiffusionDrive 设定的执行器
5. 是否要收敛成单一配置体系
