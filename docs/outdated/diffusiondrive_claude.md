# DiffusionDrive CARLA 移植问题分析

本文档是 2026-03-30 对 `diffusiondrive_agent.py` 的阶段性问题分析快照。

说明：

- 该文档中的部分结论已经被后续代码修正
- 尤其是 LiDAR 半帧拼接、多帧 buffer、初始化等待等问题，当前代码已不再符合本文档描述
- 阅读时请优先参考当前文档：
  - `docs/dd_todo.md`
  - `docs/diffusiondrive_agent_explained.md`

原文保留如下，供历史回溯使用。

---

# DiffusionDrive CARLA 移植问题分析

本文档记录通过对比 `diffusiondrive_agent.py` 和 `sensor_agent.py` 以及原始 DiffusionDrive 代码发现的关键问题。

**分析日期**: 2026-03-30
**分析范围**: 运行时逻辑、架构设计、与 CARLA 其他 agent 的对比

---

## TODO List

### P0 级别（必须修复才能正常运行）

- [x] **LiDAR 半帧拼接逻辑** - 已完成 (2026-03-30)
  - [x] 添加 `lidar_buffer` 和 `lidar_last`
  - [x] 实现 `align_lidar()` 方法
  - [x] 实现半帧拼接和完整扫描生成
  - [x] 添加初始化等待逻辑

- [x] **LiDAR 多帧缓冲和时序重对齐** - 已完成 (2026-03-30)
  - [x] 实现 LiDAR buffer 填充逻辑
  - [x] 实现多帧索引计算
  - [x] 实现历史帧时序重对齐
  - [x] 支持 `realign_lidar` 配置

- [ ] **Config 参数整合**
  - [ ] 统一 `GlobalConfig` 和 `DiffusionDriveConfig`
  - [ ] 或让 `DiffusionDriveConfig` 继承 `GlobalConfig`

### P1 级别（影响性能和稳定性）

- [ ] **Stuck Detection**
  - [ ] 添加 `stuck_detector` 计数器
  - [ ] 实现 `force_move` 机制
  - [ ] 添加 `creep_throttle` 控制

- [ ] **Safety Box**
  - [ ] 实现前方障碍物检测
  - [ ] 添加 emergency stop 逻辑
  - [ ] 配置 safety box 边界参数

- [ ] **LiDAR 坐标系验证**
  - [ ] 验证 `lidar_to_ego_coordinate()` 和 `align_lidar()` 一致性
  - [ ] 确认 BEV 特征对齐正确

### P2 级别（增强功能）

- [ ] **Stop Sign Controller**
  - [ ] 添加 stop sign buffer
  - [ ] 实现停止标志检测逻辑
  - [ ] 添加 `update_stop_box()` 方法

- [ ] **Model Ensemble**
  - [ ] 支持加载多个模型文件
  - [ ] 实现 ensemble 推理
  - [ ] 添加 NMS 后处理

- [ ] **Config 重构**
  - [ ] 统一配置管理
  - [ ] 减少手动参数同步

---

## P0 级别问题（必须修复才能正常运行）

### 1. 缺失完整的 LiDAR 时序处理逻辑 ⚠️

**问题描述**：

`sensor_agent.py` 使用完整的半帧拼接 + 多帧缓冲 + 时序重对齐流程，而 `diffusiondrive_agent.py` 只使用单帧 histogram，完全跳过了这些关键逻辑。

**sensor_agent.py 的完整流程**：

```python
# 1. 每帧只获得半个 LiDAR 扫描（CARLA 限制）
# 2. 用 align_lidar() 将上一帧的半扫描对齐到当前坐标系
lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last,
                               ego_x, ego_y, ego_theta)

# 3. 拼接成完整扫描
lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)

# 4. 存入 lidar_buffer
self.lidar_buffer.append(lidar_full)

# 5. 等待 buffer 填满后才开始推理
if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
    return carla.VehicleControl(0.0, 0.0, 1.0)

# 6. 推理时对历史帧做 realign_lidar 对齐到当前坐标系
for i in lidar_indices:
    lidar_point_cloud = deepcopy(self.lidar_buffer[-(i + 1)])
    if self.config.realign_lidar and self.config.lidar_seq_len > 1:
        lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta,
                                             ego_x, ego_y, ego_theta)
```

**diffusiondrive_agent.py 的当前实现**：

```python
# diffusiondrive_agent.py:331
result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])
# 直接使用单帧，没有半帧拼接、没有缓冲、没有重对齐
```

**影响**：

- LiDAR 数据只有一半（CARLA 每帧只返回 180° 扫描）
- 丢失时序信息，无法感知动态物体运动
- 与训练时的输入分布严重不一致
- 如果配置 `lidar_seq_len > 1`，模型期望多帧输入但实际只有单帧

**相关代码位置**：

- `sensor_agent.py:422-494`
- `sensor_agent.py:769` (`align_lidar` 函数定义)
- `diffusiondrive_agent.py:331`

---

### 2. 初始化阶段控制逻辑不完整 ⚠️

**问题描述**：

`sensor_agent.py` 在 LiDAR buffer 未填满时会持续返回刹车控制并继续累积数据，而 `diffusiondrive_agent.py` 只在第一帧刹车，之后立即开始推理。

**sensor_agent.py 逻辑**：

```python
# 第一帧：初始化并刹车
if not self.initialized:
    self._init()
    self.lidar_last = deepcopy(tick_data['lidar'])
    return carla.VehicleControl(0.0, 0.0, 1.0)

# 后续帧：等待 buffer 填满
if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
    self.lidar_last = deepcopy(tick_data['lidar'])
    return carla.VehicleControl(0.0, 0.0, 1.0)
```

**diffusiondrive_agent.py 逻辑**：

```python
# 只在第一帧刹车，之后立即推理
if not self.initialized:
    self._init()
    self.tick(input_data)
    return carla.VehicleControl(0.0, 0.0, 1.0)
# 第二帧就开始正常推理，但 LiDAR buffer 可能还是空的
```

**影响**：

- 如果配置 `lidar_seq_len > 1`，会因为 buffer 未填满而崩溃或使用错误数据
- 初始帧的推理质量差，可能导致启动时的异常行为

**相关代码位置**：

- `sensor_agent.py:416-462`
- `diffusiondrive_agent.py:430-435`

---

### 3. Config 参数不完整 ⚠️

**问题描述**：

`DiffusionDriveConfig` 缺少很多 CARLA 运行必需的参数，导致需要同时维护两个 config 对象并手动同步参数。

**缺失的关键参数**：

```python
# GlobalConfig 有但 DiffusionDriveConfig 没有：
- carla_fps / carla_frame_rate  # 用于 UKF 和控制器时间步长
- data_save_freq  # 用于 LiDAR buffer 索引计算
- realign_lidar  # 是否重对齐历史 LiDAR
- stuck_threshold / creep_duration  # 卡住检测参数
- safety_box_x_min/max, safety_box_y_min/max, safety_box_z_min/max  # 安全检查边界
- inital_frames_delay  # 初始延迟帧数
- creep_throttle  # 强制移动时的油门
```

**当前实现**：

```python
# diffusiondrive_agent.py:58-71
self.config = GlobalConfig()  # CARLA 配置
self.dd_config = DiffusionDriveConfig()  # 模型配置

# 手动同步参数
self.dd_config.lidar_min_x = self.config.min_x
self.dd_config.lidar_max_x = self.config.max_x
self.dd_config.lidar_min_y = self.config.min_y
self.dd_config.lidar_max_y = self.config.max_y
# ... 容易遗漏或出错
```

**影响**：

- 参数同步容易出错
- 代码可维护性差
- 难以复用 CARLA garage 的现有配置

**相关代码位置**：

- `diffusiondrive_agent.py:58-71`
- `config.py` (GlobalConfig 定义)
- `diffusiondrive/config.py` (DiffusionDriveConfig 定义)

---

## P1 级别问题（影响性能和稳定性）

### 4. 缺少 Stuck Detection 和 Safety Box ⚠️

**问题描述**：

`sensor_agent.py` 有完整的卡住检测和安全检查机制，`diffusiondrive_agent.py` 完全没有这些保护逻辑。

**sensor_agent.py 的保护机制**：

```python
# 卡住检测
if gt_velocity < 0.1:
    self.stuck_detector += 1
else:
    self.stuck_detector = 0

if self.stuck_detector > self.config.stuck_threshold:
    self.force_move = self.config.creep_duration

# 强制移动时的安全检查
if self.force_move > 0:
    safety_box = deepcopy(self.lidar_buffer[-1])
    # 过滤出前方安全框内的点
    safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
    safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]
    # ... x, y 轴过滤
    emergency_stop = (len(safety_box) > 0)

    if not emergency_stop:
        throttle = max(self.config.creep_throttle, throttle)
        self.force_move -= 1
```

**影响**：

- 车辆卡住时无法自动恢复
- 可能导致 `AgentBlockedTest` 失败（CARLA 规定超过 170 帧不动会判定为阻塞）
- 缺少安全检查，强制移动时可能撞到障碍物

**相关代码位置**：

- `sensor_agent.py:629-666`
- `diffusiondrive_agent.py`: 完全缺失

---

### 5. LiDAR 坐标系处理可能不一致 ⚠️

**问题描述**：

`sensor_agent.py` 使用 `align_lidar()` 做坐标变换，`diffusiondrive_agent.py` 使用 `lidar_to_ego_coordinate()`，两者的实现可能不同。

**需要验证**：

```python
# diffusiondrive_agent.py 使用
t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

# sensor_agent.py 使用
self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)
```

**潜在影响**：

- 如果两者实现不同，LiDAR 坐标系可能不一致
- 可能导致 BEV 特征错位

**相关代码位置**：

- `sensor_agent.py:769` (`align_lidar` 定义)
- `transfuser_utils.py` (`lidar_to_ego_coordinate` 定义)
- `diffusiondrive_agent.py:331`

---

## P2 级别问题（增强功能）

### 6. 缺少 Stop Sign Controller

**问题描述**：

`sensor_agent.py` 有专门的停止标志检测和控制逻辑，`diffusiondrive_agent.py` 完全没有。

**影响**：

- 在有停止标志的场景中可能违规
- 影响 LAV benchmark 等需要停止标志处理的评测

**相关代码位置**：

- `sensor_agent.py:126-131, 568-569, 668-671, 697-720`

---

### 7. 双 Config 设计混乱

**问题描述**：

同时维护 `GlobalConfig` 和 `DiffusionDriveConfig` 两个配置对象，参数同步容易出错。

**建议**：

- 让 `DiffusionDriveConfig` 继承 `GlobalConfig`
- 或者统一使用一个 config，通过命名空间区分模型参数和运行参数

---

### 8. 缺少 Ensemble 支持

**问题描述**：

`sensor_agent.py` 支持多模型集成（`self.nets` 是列表，可以加载多个 `.pth` 文件），`diffusiondrive_agent.py` 只支持单模型。

**影响**：

- 无法使用模型集成提升性能
- 与 CARLA garage 的标准做法不一致

**相关代码位置**：

- `sensor_agent.py:134-153` (加载多个模型)
- `sensor_agent.py:515-556` (ensemble 推理)
- `diffusiondrive_agent.py:86-94` (只加载单个模型)

---

## 架构对比总结

| 功能 | sensor_agent.py | diffusiondrive_agent.py | 状态 |
|------|----------------|------------------------|------|
| LiDAR 半帧拼接 | ✅ | ❌ | **缺失** |
| LiDAR 多帧缓冲 | ✅ | ❌ | **缺失** |
| LiDAR 时序重对齐 | ✅ | ❌ | **缺失** |
| 初始化等待逻辑 | ✅ | ❌ | **缺失** |
| Stuck Detection | ✅ | ❌ | **缺失** |
| Safety Box | ✅ | ❌ | **缺失** |
| Stop Sign Controller | ✅ | ❌ | **缺失** |
| Model Ensemble | ✅ | ❌ | **缺失** |
| UKF 状态估计 | ✅ | ✅ | 已实现 |
| Route Planner | ✅ | ✅ | 已实现 |
| PID 控制器 | ✅ | ✅ | 已实现 |

---

## 修复优先级建议

**立即修复（P0）**：

1. 实现完整的 LiDAR 半帧拼接逻辑
2. 实现 LiDAR buffer 和初始化等待逻辑
3. 修复 config 参数缺失问题

**尽快修复（P1）**：

4. 添加 stuck detection 和 safety box
5. 验证并统一 LiDAR 坐标变换

**后续增强（P2）**：

6. 添加 stop sign controller
7. 重构 config 设计
8. 支持模型 ensemble

---

## 相关文档

- `diffusiondrive_port_progress.md`: 移植进度记录
- `diffusiondrive_optimization_notes.md`: 优化建议
- `diffusiondrive_agent_explained.md`: Agent 实现说明
