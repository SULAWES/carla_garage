# DiffusionDrive Baseline Next Steps

本文档集中记录 baseline-basic 之后讨论出的工程判断和下一步实验方向。目标是先得到一个更可信的 `baseline-sensor-aligned`，再在其上做持续学习或更复杂的模型改进。

## 当前判断

- `baseline-basic` 是当前朴素 CARLA-native port：B2D Full raw train、garage online sensor suite、当前 crop/JPEG/preprocessing、`spatial_path` target、trajectory-only、sensor-only 且 `STOP_CONTROL=0`。
- 当前 full baseline-basic open-loop 很强：all-scenarios `l1_mean=0.0192`；但 Bench2Drive 220 closed-loop 只有 `DS=44.81`、`RC=79.48`、`NDS=35.52`。
- 这说明主要问题已经从训练链路转向闭环执行：sensor contract gap、空间 checkpoint 到控制的速度语义 gap、低速/creep/safety-box、route deviation、collisions。
- 完全重训成本可接受，因此后续不必只在当前 checkpoint 上做推理侧小调参；更合理的是重训一个 sensor / preprocessing 更一致的强 baseline。

## 实验原则

- `baseline-basic` 作为当前已完成结果保留，不再回头修改其定义；后续新结果用新的 run name 和文档条目记录。
- `baseline-sensor-aligned` 仍然属于 baseline，不应混入持续学习、hard-case loss weighting、privileged stop-sign controller 或复杂 auxiliary task。
- 每次只改一个主要因素。sensor geometry、crop、JPEG、normalization、PID、safety box、LiDAR ablation 应拆开验证，避免一个好结果解释不清。
- 20-route A/B 必须尽量复用同一组 routes、同一份 checkpoint、同一套记录字段。否则 route 难度和基础设施波动会掩盖控制或 sensor 修改的真实收益。
- 闭环结论不能只看聚合 `DS/NDS/RC`。要同时看 route-level status、collision、timeout、blocked、route deviation、`MinSpeedTest`、forced creep 和 safety-box stop。
- 因为全量训练比 220-route 闭环评测更便宜，策略应是：先用 open-loop 和 20-route 筛掉明显错误方向，再把少数候选放到 220-route。

## 命名约定

保留当前结果为：

```text
baseline-basic
```

建议新线命名为：

```text
baseline-sensor-aligned
```

含义是：仍然是 baseline，不引入持续学习方法，不使用 privileged stop-sign controller，不启用 hard-case tuned checkpoint，但显式减少训练和在线推理之间的 sensor / preprocessing gap。

## Sensor Contract 方向

当前最大传感器 gap 是 camera：

```text
B2D Full raw camera: 1600x900, fov=70,  x=0.8,  z=1.6
online garage camera: 1024x512, fov=110, x=-1.5, z=2.0
```

当前训练只是把 raw B2D 图像 resize 到 online size，再 crop/resize 到模型输入。这只对齐 tensor shape，不能消除真实几何 gap。特别是 raw image 是 `1600x900`，训练中先压到 `1024x512`，aspect ratio 从 `16:9` 变为 `2:1`；如果在线直接申请 `1024x512, fov=70`，它也只是 B2D-like，不应被解释为完全等价于 raw B2D camera。

还有一个容易混淆的点：`baseline-basic` 的训练图像本身已经来自 B2D Full raw camera，因此训练侧视觉分布已经更接近 `fov=70, x=0.8, z=1.6`。真正不一致的是 closed-loop 在线推理仍向 CARLA 申请 garage camera。也就是说，第一步 sensor-aligned 实验应优先改在线 sensor suite，而不是马上重训。

### 当前代码开关状态

训练侧图像链路在 `team_code/diffusiondrive/carla_native_dataset.py`：

```text
rgb/*.jpg
-> cv2.imread()
-> resize to GlobalConfig.camera_width x GlobalConfig.camera_height = 1024x512
-> optional JPEG artifact
-> crop_array(GlobalConfig): 512x1024 -> 384x1024
-> interpolate to --model-image-height x --model-image-width = 256x1024
-> [0,1] tensor
```

训练命令当前已支持：

```text
--model-image-height
--model-image-width
--no-jpeg-artifact
--b2d-source-image-height
--b2d-source-image-width
```

注意：

- `--b2d-source-image-height/width` 目前主要用于记录 `training_config.json`，实际像素处理以读到的图片和 `GlobalConfig.camera_width/height` 为准。
- 当前训练侧没有 CLI 参数直接覆盖 `GlobalConfig.crop_image`、`camera_fov`、`camera_pos` 或 `camera_height/width`。
- 当前训练侧没有 ImageNet normalization 路径，默认就是 `[0,1]`。

推理侧 sensor 定义在 `DiffusionDriveAgent.sensors()`，实际使用：

```text
GlobalConfig.camera_pos
GlobalConfig.camera_rot_0
GlobalConfig.camera_width
GlobalConfig.camera_height
GlobalConfig.camera_fov
GlobalConfig.lidar_pos
GlobalConfig.lidar_rot
```

推理侧在线 env 已支持：

```text
DIFFUSIONDRIVE_JPEG_ARTIFACT
DIFFUSIONDRIVE_IMAGE_NORMALIZATION
DIFFUSIONDRIVE_CAMERA_FOV
DIFFUSIONDRIVE_CAMERA_POS
DIFFUSIONDRIVE_CAMERA_ROT
DIFFUSIONDRIVE_CAMERA_WIDTH
DIFFUSIONDRIVE_CAMERA_HEIGHT
DIFFUSIONDRIVE_LIDAR_POS
DIFFUSIONDRIVE_LIDAR_ROT
DIFFUSIONDRIVE_CROP_IMAGE
DIFFUSIONDRIVE_MODEL_IMAGE_HEIGHT
DIFFUSIONDRIVE_MODEL_IMAGE_WIDTH
DIFFUSIONDRIVE_ZERO_LIDAR
DIFFUSIONDRIVE_DEBUG_ROUTE
DIFFUSIONDRIVE_ROUTE_DEBUG_DISTANCE_WARN
DIFFUSIONDRIVE_ROUTE_DEBUG_ANGLE_WARN_DEG
DIFFUSIONDRIVE_DEBUG_SAFETY_BOX
```

这些 override 在 `DiffusionDriveAgent.setup()` 中创建 `GlobalConfig()` 后、构造 `CARLA_Data` 和 `dd_config` 之前应用；model image size 在构建 `V2TransfuserModel` 前写入 `dd_config.camera_height/width` 并重新调用 `__post_init__()`。因此在线 closed-loop A/B 已经可以只通过 shell env 覆盖 camera / LiDAR / crop / model input size，并且 agent 启动时会打印最终 sensor config 和 model image size。

注意：训练侧 `train_diffusiondrive.py` 还没有补完整的 crop / camera metadata CLI。`S1/S1a/S1b/S2` 这类“当前 checkpoint 的在线部署诊断”已经可跑；`S4/S5` 这类 full retrain / no-crop retrain 仍需要先补训练参数化并把最终 preprocessing snapshot 写入 `training_config.json`。

在线 camera-aligned 诊断的典型 override：

```text
DIFFUSIONDRIVE_CAMERA_FOV=70
DIFFUSIONDRIVE_CAMERA_POS=0.8,0.0,1.6
DIFFUSIONDRIVE_CAMERA_ROT=0.0,0.0,0.0
DIFFUSIONDRIVE_CROP_IMAGE=0/1
DIFFUSIONDRIVE_MODEL_IMAGE_HEIGHT=256
DIFFUSIONDRIVE_MODEL_IMAGE_WIDTH=1024 或 512
```

实现边界：

- 在线推理：已实现 camera / LiDAR / crop / model image size override，确保 `sensors()`、`CARLA_Data`、`tick()` 和 `build_diffusiondrive_config()` 使用同一套配置。
- 训练：仍需给 `train_diffusiondrive.py` 增加显式 crop / camera metadata 参数。其中 camera FOV / pose 不改变已保存 raw 图片像素，但会影响 run 的 sensor contract 记录。
- 日志：agent 启动时已打印最终 camera FOV / pose / resolution / crop / model image size；closed-loop result 目录仍建议额外保存同样信息，方便论文归档。

### 为什么优先看 camera

- camera 是模型最主要的语义输入，FOV 从 `70` 到 `110` 的差异会显著改变物体尺度、车道线弯曲程度和远近场分布。
- camera 位置从 `x=0.8,z=1.6` 到 `x=-1.5,z=2.0`，等价于从更靠前、更低的视角切到更靠后、更高的视角；即使 resize 后 tensor shape 一样，路口、停止线、前车和边界在图像中的位置也会变。
- 当前 open-loop 是在 raw B2D 传感器分布上评估，closed-loop 是 online garage sensor suite。二者差距越大，越容易出现 open-loop 很低、closed-loop 不迁移的情况。
- 先用当前 checkpoint 只改 online camera geometry 做 20-route，是为了隔离“部署传感器差异”本身。如果不重训也有收益，说明 B2D-like online camera 是值得继续验证的方向。

### 推荐实验矩阵

0. **`S0_baseline_reference`**

   目的：固定对照组。使用当前 baseline-basic checkpoint 和当前 garage online sensor suite，不改 camera、不改 crop、不改 JPEG。控制器使用当前最稳的 PID 候选，例如 A8/A9 或新跑出的插值 PID。

   必须记录：

   ```text
   checkpoint
   PID env
   STOP_CONTROL
   JPEG / normalization
   route list
   output dir
   ```

1. **`S1_camera_b2d_like_checkpoint`**

   只改在线 camera 几何，暂不重训。这里的 `B2D-like` 是“更接近 B2D raw”，不是完全等价：

   ```text
   camera_fov = 70
   camera_pos = [0.8, 0.0, 1.6]
   camera_rot = [0.0, 0.0, 0.0]  # run 前需从 B2D sensor metadata / 采集配置确认
   resolution = 1024x512
   preprocessing = current crop + JPEG
   model input = 256x1024
   checkpoint = baseline-basic
   ```

   这是最重要的第一组 sensor 实验。它利用的事实是：baseline-basic 已经在 B2D raw camera 图像上训练，closed-loop 只需要把在线相机改成更接近训练分布。

   预期信号：

   - route deviation、blocked 或低速相关问题减少。
   - collision 不明显增加。
   - 20-route 的 DS / NDS 至少接近 A8/A9，或者在坏 route 上有清楚改善。

   如果这一步有收益，说明 online sensor gap 是有效方向。此时不一定需要立即重训，因为训练图像本来就是 B2D raw；更应该先确认这个收益在 20-route 上稳定，再决定是否用 clean run name 重新训练或直接把它作为 `baseline-sensor-aligned-inference`。

   必须保存 preprocessing snapshot，至少包括：

   ```text
   online raw CARLA frame
   crop 后图像
   model input resize 后图像
   同 route 的 baseline S0 对应帧
   ```

   否则很难判断收益来自 FOV / pose，还是来自 aspect ratio、裁剪和插值共同造成的视觉变化。

2. **`S1a_camera_fov_only` 和 `S1b_camera_pose_only`**

   如果 `S1` 有明显变化，但不清楚收益来自 FOV 还是 pose，可做两个更小的拆分：

   ```text
   S1a: camera_fov = 70,  camera_pos 保持 [-1.5, 0.0, 2.0]
   S1b: camera_fov = 110, camera_pos = [0.8, 0.0, 1.6]
   ```

   判断方式：

   - 若 `S1a` 接近 `S1`，主要问题是 FOV。
   - 若 `S1b` 接近 `S1`，主要问题是 camera mounting position。
   - 若二者都一般但 `S1` 好，说明 FOV 和 pose 共同作用，后续不应只保留其中一个。

3. **`S2_nocrop_checkpoint_diagnostic`**

   只做诊断，不作为正式结果。使用 baseline-basic checkpoint，但把 online preprocessing 改为不 crop：

   ```text
   crop_image = False
   CARLA sensor = 1024x512
   model input = 256x1024
   checkpoint = baseline-basic
   ```

   这会把未裁剪的 `512x1024` 图像直接插值成 `256x1024`，与训练分布不一致，因此不能用它的绝对分数下结论。它只回答一个问题：底部近场视觉信息是否可能缓解低速、blocked、creep 或 safety-box stop。

4. **`S3_baseline_sensor_aligned_currentcrop_clean`**

   可选 clean retrain。在线和训练记录都采用更接近 B2D raw 的 camera geometry，但保留当前 crop / model image size。

   需要明确：如果训练像素链路、manifest、seed 以外都不变，这个实验的训练 tensor 与 baseline-basic 基本相同，主要价值是得到一个命名和配置都干净的 `baseline-sensor-aligned`，而不是提供新的实验变量。因此它的优先级低于 `S1/S1a/S1b`。只有当 `S1` 证明 B2D-like online camera 有稳定收益，或者论文/归档需要一个配置命名干净的复现实验时，再考虑花 full training。它应尽量保持如下条件不变：

   ```text
   target_mode = spatial_path
   status_schema = command_one_hot(6) + speed(1)
   trajectory_loss_only
   STOP_CONTROL = 0
   JPEG = 1
   normalization = none
   PID = 当前最稳候选
   ```

   预期上 open-loop 不应明显变化。如果 open-loop 大幅变化，应优先检查训练代码是否无意改变了 crop、image size、JPEG 或 normalization。

5. **`S4_baseline_sensor_aligned_nocrop_train`**

   目的：验证当前 `crop_array()` 去掉底部近场信息是否伤害低速、起步、跟车和障碍交互。

   ```text
   crop_image = False
   model input = 256x1024
   JPEG = 1
   normalization = none
   ```

   该实验需要训练和推理同时关闭 crop。只改推理 crop 是诊断，正式结论必须全量重训。

   这个实验重点看低速和近场交互：

   - 如果 `MinSpeedTest`、blocked、forced creep、safety-box stop 变好，说明底部近场视觉信息对闭环有用。
   - 如果 collision 或 route deviation 上升，说明底部区域可能带来 hood / ego-car / artifacts 干扰，或者当前模型容量和 token grid 不适合直接消费未裁剪图像。
   - 如果 open-loop 明显变差，不应直接进 220-route，应先检查输入分辨率、crop 标志和训练日志中的 image shape。

6. **`S5_baseline_sensor_aligned_256x512_train`**

   目的：验证单前视是否不适合强行拉成 `256x1024` 的 4:1 宽图。`256x512` 更接近单前视自然宽高比，也更省算力。该实验会改变模型 token grid，必须全量重训。

   推荐固定：

   ```text
   CARLA sensor = 1024x512
   crop_image = 按 S4 结果决定
   model input = 256x512
   checkpoint = 从头训练，不加载 256x1024 checkpoint
   ```

   当前训练脚本支持 `--model-image-width 512`，但推理侧 `DiffusionDriveConfig.camera_width` 默认仍是 `1024`，没有 env override。因此在正式跑该实验前必须补齐 runtime model image size 配置，否则会出现 checkpoint shape 或输入尺寸不一致。

   该方向的风险是横向视野被压缩后，路口和大曲率转弯的上下文不足。它适合作为第三个 sensor-aligned full train，而不是第一个。

### Sensor 实验记录要求

每个 sensor A/B 至少记录以下字段，避免之后无法解释结果：

```text
experiment_name:
checkpoint:
camera_fov:
camera_pos:
camera_rot:
camera_rot_source:
camera_resolution:
crop_image:
cropped_size:
model_image_size:
jpeg_artifact:
image_normalization:
lidar_pos:
lidar_rot:
PID env:
route_list:
open_loop_result:
closed_loop_result:
notes:
```

如果只跑 20-route，建议另外保存 3 到 5 条典型 route 的前处理图像快照：原始 CARLA frame、crop 后图像、模型输入 resize 后图像。优先看 route deviation、低速 blocked、collision 前几秒，而不是随机帧。

### 不建议现在切到三相机

原版 NAVSIM DiffusionDrive 使用多相机视角，但当前 CARLA-native baseline 的数据、在线 sensor suite、训练代码和推理 contract 都是围绕单前视冻结的。三相机会引入新的数据生成、模型输入接口、显存和在线同步问题。除非后续明确要做 `baseline-multicam`，否则不要把它混入 `baseline-sensor-aligned`。

### 暂不优先做

- 不要一次同时改 camera、LiDAR、PID、safety box。
- 不要直接把 JPEG 和 normalization 混在 sensor geometry 实验中一起改。`DIFFUSIONDRIVE_JPEG_ARTIFACT=0` 可以单独做诊断，但当前 checkpoint 是 JPEG-on 路径训练的，不能预期一定更好。
- 不要把 `--b2d-source-image-height/width` 当成真正的 resize 开关；它目前不是。
- 不要直接跑 `256x512` 推理，除非已经确认 runtime model config 和 checkpoint 都是 `256x512`。

## PID / Control 方向

当前模型输出空间 checkpoint，不是 fixed-time trajectory；desired speed 只能由空间 PID 启发式估计。20-route A/B 已证明 PID 是主要闭环瓶颈之一。

已知结果：

- `A8_pid_6_2p5` 更均衡：`DS=42.9973`、`RC=78.6635`、`NDS=34.0060`。
- `A9_pid_7_3` 分数最高：`DS=46.0220`、`RC=77.2035`、`NDS=35.7785`，但 collision 和 timeout 风险更高。
- 单独调 stuck threshold 不够稳定，不能作为主线突破口。

### 指标如何解读

- `RC` 高但 `DS/NDS` 低，通常表示路线能推进，但 infractions 或驾驶质量扣分严重。
- `MinSpeedTest` 好转不一定代表整体好转。如果同时 collision / timeout 增加，可能只是车开得更激进。
- `Completed` route 也可能质量很差，应看每条 route 的 infraction、route completion 和 system/game time。
- `Ratio(Game/System)` 极低时，先把它当作基础设施或 CARLA runtime 异常记录，不宜直接用来判断模型控制好坏。
- `blocked`、`timed out`、大量 `Detected agent being stuck` 和 `Creeping stopped by safety box` 更接近低速控制 / safety-box / 近场感知问题。

### 正在跑 / 优先等待的实验

```text
speed_fast = 6.5
speed_slow = 2.75
```

```text
speed_fast = 7.0
speed_slow = 2.5
```

判断逻辑：

- 如果 `7.0/2.5` 比 A9 少 collision / timeout，同时保留低 `MinSpeedTest` penalty，则它适合作为后续 sensor-aligned baseline 的默认 PID。
- 如果 `6.5/2.75` 更稳，则先用它作为强 baseline 控制配置。

### 下一步可试

如果上述两组仍然在复杂 route 上碰撞或 timeout，优先调 turn threshold，而不是继续盲目增减整体速度：

```text
speed_fast = 7.0
speed_slow = 2.5
turn_threshold = 0.20
sharp_turn_threshold = 0.45
```

这个实验的意图是让中等转弯更早进入 slow speed，重点观察 `212/194/203/43` 等高速度退化 route。

如果转弯阈值变小后 route deviation 减少但 `MinSpeedTest` 变差，说明速度上限不是唯一问题，可能需要让 desired speed 同时依赖曲率和前方安全框，而不是只用一个 turn ratio 分段。

### 后续控制改动

可考虑加入 safety-box-aware speed cap，但不能简单用 `front_safety_box_nonempty` 触发。当前已经观察到大量 `Creeping stopped by safety box`，其中既可能是真障碍，也可能是近场点云、坐标或阈值误触发。如果只要 safety box 非空就降速，可能会进一步加重低速和 blocked。

更合理的实现应先记录统计，再用 gate 触发：

```text
if safety_box_active_for_consecutive_ticks >= N
   and safety_box_point_count >= K
   and nearest_front_point_distance < X
   and ego_speed_or_desired_speed is above a small threshold:
    desired_speed = min(desired_speed, 1.5~2.0)
```

候选初值可以先从日志分布里定，不建议拍脑袋固定。这个改动仍属于规则控制，不是模型改进。它针对的是 A9 中“速度提升后低速问题缓解，但复杂交互里碰撞 / timeout 上升”的现象；若 gate 过松，它会把低速问题重新带回来。

每条 route 建议记录结构化 control summary：

- desired speed mean / p95
- turn ratio mean / p95
- brake ticks
- throttle mean
- low-speed ticks
- forced creep ticks
- safety-box stop ticks
- final status / DS / RC / infractions

这些字段的价值在于把“失败原因”从 leaderboard 最终状态拆开。例如同样是 `blocked`，可能是模型轨迹不动、PID desired speed 过低、safety box 误停、或者前车交互导致长时间 creep。没有控制日志时，这些情况在最终 JSON 里很难区分。

## Route-Aware Guard / Blend 方向

route deviation 是 baseline-basic 的主要失败模式之一，因此除了 sensor 和 PID，也需要保留一条轻量 route-aware 控制线。这里的 route-aware 指使用 `RoutePlanner` / global route corridor 这种 leaderboard 已允许的路线信息，不使用 actor、stop sign、traffic light 等 privileged object state，因此仍可放在 sensor-only baseline 的工程控制范围内。

### 目标

不要直接覆盖模型预测轨迹，而是在高风险时做软约束：

- aim point 偏离 route corridor 太多时，降低 desired speed。
- 预测轨迹方向和 route tangent 明显冲突时，将控制目标向 route waypoint 做小比例 blend。
- junction / obstacle route 中保留足够自由度，避免把合法绕行强行拉回 route centerline。

### 推荐先做诊断

先在 debug log 中记录：

```text
predicted aim point
nearest route point
distance_to_route_corridor
angle_to_route_tangent
command
desired_speed
steer
route_deviation_warning
final route status
```

重点看 `deviated` route、route completion 接近但 InRouteTest failure 的 route，以及低速长时间偏离 route 的 route。只有确认偏离发生在模型/控制 aim point 已明显离开 route corridor 时，才实现 guard。

### 候选实现

第一版优先做 speed guard，而不是直接改轨迹：

```text
if distance_to_route_corridor > D_high:
    desired_speed = min(desired_speed, route_guard_speed)
elif distance_to_route_corridor > D_mid and abs(angle_to_route_tangent) > A:
    desired_speed = min(desired_speed, route_guard_speed)
```

如果 speed guard 能减少 route deviation 但导致 blocked / low-speed 上升，再试轻量 blend：

```text
blend_weight = clamp((distance_to_route_corridor - D_mid) / (D_high - D_mid), 0, W_max)
aim_point = (1 - blend_weight) * model_aim_point + blend_weight * route_aim_point
```

注意事项：

- `W_max` 应小，避免把模型轨迹直接改成 route follower。
- 在 obstacle / construction / accident two-way 等需要绕行的场景，应通过 route command、横向偏差持续时间或低速状态限制触发，避免压掉绕障能力。
- 这个方向应和 PID / sensor 实验分开 A/B，不要和 camera-aligned 同时改。

## LiDAR BEV 对齐遗留项

以前搁置的 v3-v9 LiDAR BEV 对齐工作，主要研究的是多帧历史 LiDAR BEV 的 residual alignment / refinement。当前 baseline 主线实际并没有使用这套 refinement。

### 当前实际使用路径

训练侧：

```text
lidar/*.laz -> lidar_to_histogram_features() -> BEV histogram -> model
```

推理侧：

```text
current half scan + aligned previous half scan -> full scan -> lidar_to_histogram_features() -> model
```

当前 `GlobalConfig.lidar_seq_len = 1`。因此：

- 模型不消费更久历史帧。
- `realign_lidar=True` 只有在 `lidar_seq_len > 1` 时才会影响历史帧。
- v5/v8/v9 中讨论的 `history=1/3/5` 多帧 BEV refinement 现在不是当前模型输入主路径。

### 对当前 baseline 的影响判断

- 旧的多帧 LiDAR BEV residual refinement 没完成，不应被视作当前 baseline 结果差的主要原因。
- 但 LiDAR 坐标和近场点云仍可能影响闭环，尤其是 creep 时的 safety box。
- 目前大量 `Creeping stopped by safety box` 说明 safety-box runtime 行为值得单独诊断。

当前真正需要关注的是：

```text
B2D raw LiDAR: x=-0.39, z=1.84, yaw=0, range=85
online LiDAR:  x=0.0,   z=2.5,  yaw=-90
```

以及 online safety box 是否因为近场点云 / 坐标 / 阈值问题误触发。

### 什么时候它会变成高优先级

LiDAR / safety-box 应在以下情况升为主线问题：

- camera-aligned 和 PID 调参后，blocked / creep 仍然占主要失败。
- `Creeping stopped by safety box` 在坏 route 中高频出现，并且点云统计显示近场框内经常有异常点。
- zero-LiDAR 诊断没有显著恶化，甚至减少 blocked / creep，说明当前 BEV 可能带来 domain gap 噪声。
- collision 主要发生在低速近场交互，而不是高速转弯或路线偏离。

### 推荐诊断

1. 对 `212/194/203/101/43` 等坏 route，在每次 `Creeping stopped by safety box` 时记录：

   ```text
   safety_box point count
   min/max x/y/z
   nearest point distance
   ego speed
   desired speed
   brake / throttle
   ```

2. 在 blocked / collision 前 5 秒保存：

   ```text
   LiDAR BEV
   safety box rectangle
   predicted trajectory
   route direction
   ```

3. 做诊断性 LiDAR ablation。推理侧已支持 runtime override，让 `lidar_feature` 置零：

   ```text
   DIFFUSIONDRIVE_ZERO_LIDAR=1
   ```

   目的不是作为正式结果，而是判断当前 LiDAR BEV 对闭环是正贡献、弱贡献，还是存在 domain gap 噪声。

### 暂不建议

- 暂不建议直接重启 v10 residual refinement。
- 暂不建议在当前 baseline 中启用 `lidar_seq_len > 1`。
- 只有当后续决定重训多帧 LiDAR 模型时，v9/v10 才重新成为主线问题。
- 如果要重启多帧 LiDAR，应先明确目标是“模型输入多帧历史 BEV”还是“仅 safety-box runtime 更稳”。前者需要训练接口和 checkpoint 全部变化，后者更像工程诊断和规则修正。

## Time / Speed Semantics

当前预测 target 是空间 checkpoint，没有时间语义。这确实是低速、起步、creep 和跟车问题的根源之一。

但该点先放到 baseline 之后：

- baseline 阶段先通过 sensor contract + PID / safety-box 调参得到稳定结果。
- 后续改进阶段再考虑 speed head、arrival-time profile 或 time-based trajectory。

推荐后续方向是“空间路径 + 时间剖面”，而不是直接把当前空间 anchor 硬解释成 fixed-time trajectory。

### 为什么暂时不改

时间语义确实更接近根因，但它会改变训练目标、loss、模型输出和控制接口。现在还在建立论文 baseline，如果此时加入 speed head 或 arrival-time head，baseline 和后续方法的边界会变得不清楚。更稳妥的路线是：

1. 先把 sensor-only baseline 做到可解释、可复现。
2. 再把显式时间 / 速度预测作为后续改进方法。
3. 最后用同一套闭环门槛证明它解决了低速、creep 或跟车问题。

## 评测门槛

每个新全量训练不应立即跑 220 条 closed-loop。建议顺序：

1. open-loop all-scenarios
2. 20-route closed-loop
3. 满足门槛后再跑 Bench2Drive 220

20-route 进入 220 的建议门槛：

- open-loop 不明显劣化。
- 20-route DS / NDS 至少接近 A8/A9。
- collision 不比 A9 更高。
- `212/194/203/43` 不继续明显退化。
- forced creep ticks 和 safety-box stop ticks 不明显上升。

### 每次实验至少记录

```text
experiment_name:
checkpoint:
train_manifest:
val_manifest:
sensor_geometry:
preprocessing:
PID env:
STOP_CONTROL:
ZERO_LIDAR:
sensor_override_implemented:
preprocessing_snapshot_saved:
route_guard_enabled:
safety_box_speed_cap_enabled:
closed_loop_routes:
open_loop_output:
closed_loop_output:
summary_csv:
notes:
```

建议把这些信息写进 run 目录的 `notes.md` 或 `training_config.json` 附近。后面做论文表格时，最容易遗漏的不是分数，而是某个 run 是否启用了特殊 env。

### 当前不应混入 baseline 的项

- hard-left 或其他 hard-case sample weighting。
- privileged stop-sign controller。
- auxiliary heads，包括 `agent_states / agent_labels / bev_semantic_map`。
- 显式 speed head、time head、arrival-time profile。
- 多帧 LiDAR BEV refinement。
- 针对 20-route 过拟合的 route-specific PID。

## 当前推荐优先级

1. 等当前两个 PID 实验结果。
2. 同步新版 `team_code/diffusiondrive_agent.py` 到远端，确认启动日志打印最终 sensor config / model image size / zero-LiDAR 状态。
3. 用当前 checkpoint 跑 `S1_camera_b2d_like_checkpoint` 的 20-route。
4. 若 `S1` 有收益，再视需要跑 `S1a/S1b` 拆分 FOV 和 pose。
5. `S3_baseline_sensor_aligned_currentcrop_clean` 不作为立即优先项；只有在 `S1` 收益稳定或论文归档需要时再做 clean retrain。
6. 用当前 checkpoint 跑 `S2_nocrop_checkpoint_diagnostic`，只判断是否值得全量 nocrop 重训。
7. 若 crop 诊断有收益，重训 `S4_baseline_sensor_aligned_nocrop_train`。
8. 视结果决定是否重训 `S5_baseline_sensor_aligned_256x512_train`；在线 runtime model image size override 已实现，但 full retrain 前仍要补训练侧 model / preprocessing 参数化并确认 checkpoint shape。
9. 打开 `DIFFUSIONDRIVE_DEBUG_ROUTE=1` 跑坏 route，确认 aim point / route corridor 偏离是否是主要触发原因；warning 阈值可用 `DIFFUSIONDRIVE_ROUTE_DEBUG_DISTANCE_WARN` 和 `DIFFUSIONDRIVE_ROUTE_DEBUG_ANGLE_WARN_DEG` 调整。再决定是否实现 route-aware speed guard 或轻量 blend。
10. 打开 `DIFFUSIONDRIVE_DEBUG_SAFETY_BOX=1` 和必要时 `DIFFUSIONDRIVE_ZERO_LIDAR=1` 做 LiDAR 近场诊断；safety-box speed cap 必须先用连续 tick / point count / nearest distance gate，不要用 nonempty 直接触发，也暂不把多帧 BEV refinement 放回主线。
