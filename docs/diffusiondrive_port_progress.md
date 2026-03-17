# DiffusionDrive -> CARLA 移植进度与待补充项

本文档描述 `DiffusionDrive`（DiffusionDrive/TransFuser backbone + diffusion trajectory head）的 **CARLA 推理侧移植** 已完成的工作、当前实现假设，以及建议继续补充/验证的点。

> 说明：目前只把代码“接上并能跑通逻辑链路”的部分写好；未在本环境实际运行验证。

## 已完成

### 1) 代码结构与模块落地

已将 DiffusionDrive 相关代码迁移到 `carla_garage/team_code/diffusiondrive/` 下，并去除了对 `navsim/nuplan` 的 import 依赖（改为 CARLA 侧最小配置与枚举）。

- 主要文件
  - `carla_garage/team_code/diffusiondrive/config.py`：`DiffusionDriveConfig` + `TrajectorySampling`（简化版）
  - `carla_garage/team_code/diffusiondrive/backbone.py`：TransFuser backbone（timm）
  - `carla_garage/team_code/diffusiondrive/model.py`：`V2TransfuserModel`（含 diffusion trajectory head）
  - `carla_garage/team_code/diffusiondrive/enums.py`：`StateSE2Index`、`BoundingBox2DIndex`
  - `carla_garage/team_code/diffusiondrive/modules/*`：blocks / loss / scheduler / conditional_unet 等

### 2) CARLA agent 已接入（推理）

新增 CARLA leaderboard agent：

- `carla_garage/team_code/diffusiondrive_agent.py`

核心行为：

- 传感器与 UKF/RoutePlanner 逻辑：基本复用 `team_code/sensor_agent.py` 的做法（减少引入新的坐标系坑）。
- 特征构建（喂给 DiffusionDrive 模型）：
  - `camera_feature`：来自 `rgb_front`，按 `transfuser_utils.crop_array` 裁剪后，归一化到 `[0,1]` 并 resize 到 `256x1024`
  - `lidar_feature`：使用 `CARLA_Data.lidar_to_histogram_features` 得到 BEV histogram（`(C,H,W)`）
  - `status_feature`：拼接 `command(6)` + `velocity(2)` + `acceleration(2)`，其中加速度用速度差分估计（纵向为主，横向填 0）
- 模型输出：
  - 使用 `outputs['trajectory']`（形状预期 `B x T x 3`），取 `x,y` 作为 waypoints
- 控制：
  - 用 CARLA garage 现有的 PID 方式对 waypoints 进行控制（在 `diffusiondrive_agent.py` 内实现了 `_control_pid`，逻辑对齐 `team_code/model.py#control_pid`）

### 3) Anchor 路径接入（必须）

agent 通过环境变量读取 anchor：

- `DIFFUSIONDRIVE_ANCHOR_PATH`：plan anchor `.npy`（必须）

并写了显式报错：未提供会 `raise RuntimeError`，避免 silent failure。

### 4) 权重加载（可选，但建议提供）

agent 支持加载 checkpoint：

- `DIFFUSIONDRIVE_CHECKPOINT`：`.pth/.ckpt` 等

加载时会尝试清理常见前缀：`agent.` / `model.` / `module.`，并使用 `strict=False` 加载，打印 missing/unexpected keys 以便你做权重对齐。

## 当前实现的关键假设/约束

### 1) Anchor 的数组形状/含义

当前移植版按 DiffusionDrive 常见设置：anchor 只包含 `(x, y)`，heading 由网络的轨迹 head 预测。

- 推荐形状：`(num_mode, num_poses, 2)`，例如 `(20, 8, 2)`
- 含义：20 个 mode，每个 mode 8 个未来点的 (x,y)

### 2) 相机输入尺寸

DiffusionDrive 默认按 `camera_height=256, camera_width=1024` 设计；CARLA garage 的默认相机是 `512x1024` 且启用裁剪（裁到 `384x1024`）。

当前 agent 流程：

1) 先按 garage 的 `crop_array()` 裁剪
2) 再 resize 到 `256x1024`

这能保证形状对齐，但数值分布是否最优需要后续用训练设置核对（见“待补充”）。

### 3) status_feature 的定义与尺度

DiffusionDrive 原实现的 status 维度是 `4 + 2 + 2`（明显不是 CARLA garage 的 6-class command），此处我改成：

- `command_dim=6`
- `velocity_dim=2`（只填 `speed, 0`）
- `accel_dim=2`（只填 `a_long, 0`）

并把 `model.py` 的 `self._status_encoding` 输入维度改为 `config.status_dim`。

这保证了维度一致，但“速度/加速度的具体定义、归一化方式”仍需你用原 DiffusionDrive 训练侧做一致化。

## 建议继续补充/验证的点（TODO）

### A. 多帧 LiDAR（lidar_seq_len > 1）

当前 agent 只用单帧 LiDAR histogram。若你要对齐 garage 的多帧/重对齐逻辑，需要把 `SensorAgent` 中的：

- `lidar_buffer`、半帧 LiDAR 对齐、`realign_lidar` 等逻辑

迁移到 `diffusiondrive_agent.py`，并把输入拼成 `C = lidar_seq_len`（或 `2*lidar_seq_len`）通道喂给 backbone。

### B. 输入归一化/预处理与训练对齐

建议核对以下点与 DiffusionDrive 训练侧是否一致：

- 相机是否需要 ImageNet mean/std normalization（当前没有做）
- JPEG artifact 是否需要（当前有做，复用 garage 推理流程）
- LiDAR histogram 的坐标轴方向/是否 transpose（当前复用 garage 的实现）

### C. 控制器与评测指标对齐

目前是“DiffusionDrive 输出 trajectory -> PID”。如果你希望更贴近 DiffusionDrive 原方法/论文设置，可能需要：

- 重新设计控制器（例如直接回归控制量、或基于轨迹做 MPC/LQR 等）
- 或复用 garage 中的 direct controller（如果你打算新增 `target_speed/checkpoints` 输出）

### D. 模型输出与任务头裁剪

`V2TransfuserModel` 里还包含 agent box / bev semantic 等头，CARLA 推理中目前不使用它们的输出。

后续可以考虑：

- 关闭不需要的 head（节省显存/时间）
- 或将 `agent_states/labels` 映射到 CARLA 的 bbox 格式做可视化与 safety check

### E. Checkpoint key 对齐与严谨加载

当前是 `strict=False`，适合早期快速迭代。等你确定最终结构后，建议：

- 逐步收紧 strict
- 或写一份显式的 key mapping 脚本（确保 backbone/heads 对齐）
