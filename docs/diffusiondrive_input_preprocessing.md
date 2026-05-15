# DiffusionDrive 输入预处理核对

本文档记录当前 `DiffusionDriveAgent` 的输入预处理事实、与 NAVSIM 原版 DiffusionDrive 的差异，以及后续 CARLA 训练前需要冻结的决策。

## 当前推理侧事实

### 相机

当前 CARLA port 只注册并使用一个前视 RGB 相机：

- 传感器尺寸：`GlobalConfig.camera_height x camera_width = 512 x 1024`
- 相机位姿：`camera_pos = [-1.5, 0.0, 2.0]`，`camera_rot_0 = [0.0, 0.0, 0.0]`
- FOV：`110`

当前 `DiffusionDriveAgent.tick()` 的图像处理顺序是：

1. 从 `rgb_front` 读取 `BGR` 图像。
2. 可选 JPEG encode/decode，默认开启，用于保留现有 garage 评测路径的图像压缩 artifact。
3. `BGR -> RGB`。
4. 调用 `t_u.crop_array()`，即从原图顶部开始截取 `384 x 1024`，去掉底部 `128px`。
5. 转成 `C,H,W`，保持像素值 `0..255`。
6. 推理前除以 `255.0`，resize 到 DiffusionDrive 模型输入 `256 x 1024`。
7. 默认不做 ImageNet mean/std normalization。

当前可用运行时开关：

- `DIFFUSIONDRIVE_JPEG_ARTIFACT`
  - 默认：`1`
  - `1`：保留 JPEG encode/decode，保持当前闭环基线。
  - `0`：跳过 JPEG artifact，便于和 NAVSIM 原版输入路径做对照。
- `DIFFUSIONDRIVE_IMAGE_NORMALIZATION`
  - 默认：`none`
  - `none`：输入为 `[0, 1]`，与 NAVSIM 原版 `transforms.ToTensor()` 一致。
  - `imagenet`：对 `[0, 1]` 输入应用 ImageNet mean/std，便于做 CARLA 训练消融。

### LiDAR

当前 LiDAR 处理继续复用 garage 侧实现：

- `t_u.lidar_to_ego_coordinate()` 先把 CARLA LiDAR 点云转到 ego 坐标。
- 在线半帧拼接生成完整扫描。
- `lidar_buffer` 支持 `lidar_seq_len * data_save_freq` 的多帧缓存。
- 当 `realign_lidar=True` 且 `lidar_seq_len>1` 时，将历史帧 realign 到当前 ego 坐标。
- `CARLA_Data.lidar_to_histogram_features()` 生成 BEV histogram。
- 默认 `use_ground_plane=False`，因此每帧 LiDAR 贡献 `1` 个 above-split 通道。

LiDAR histogram 参数当前由 `GlobalConfig -> DiffusionDriveConfig` 映射：

- `min_x/max_x/min_y/max_y = -32/32/-32/32`
- `pixels_per_meter = 4.0`
- `lidar_resolution = 256 x 256`
- `hist_max_per_pixel = 5`
- `lidar_split_height = 0.2`
- `max_height_lidar = 100.0`

## 与 NAVSIM 原版的差异

### 相机差异

NAVSIM 原版 `TransfuserFeatureBuilder._get_camera_feature()` 使用三相机拼接：

- `cam_l0`：裁掉上下 `28px`，左右各 `416px`
- `cam_f0`：裁掉上下 `28px`
- `cam_r0`：裁掉上下 `28px`，左右各 `416px`
- 横向拼接后 resize 到 `1024 x 256`
- `transforms.ToTensor()` 得到 `[0, 1]`
- 不注入 JPEG artifact
- 不做 ImageNet mean/std normalization

当前 CARLA port 的主要差异是：

- 单前视相机，不是三相机拼接。
- 裁剪策略不同：CARLA 当前去掉底部区域；NAVSIM 原版去掉上下边缘。
- 当前默认保留 JPEG artifact；NAVSIM 原版没有。
- 两者默认都不是 ImageNet mean/std normalization，而是 `[0, 1]` 输入。

因此，若继续加载原版 NAVSIM checkpoint，最大分布差异仍是相机视野和裁剪方式，而不是数值归一化。

### LiDAR 差异

NAVSIM 原版只使用最新单帧 LiDAR，并直接用 histogram 特征。当前 CARLA port 为适应 leaderboard 在线传感器，额外加入：

- 半帧拼接
- 多帧 buffer
- 历史帧 realign

这部分当前是工程适配，不应简单按 NAVSIM 单帧路径回退。后续重点仍是验证 BEV 对齐和近场稳定性。

## 当前结论

1. 当前闭环基线应继续默认使用：
   - 单前视相机
   - JPEG artifact 开启
   - `crop_array(): 512x1024 -> 384x1024`
   - resize 到 `256x1024`
   - `[0, 1]` normalization

2. 如果目标是零样本或少量微调原版 NAVSIM checkpoint，优先怀疑相机视野分布 mismatch：
   - 单前视 vs 三相机拼接
   - CARLA 去底部裁剪 vs NAVSIM 上下裁剪
   - JPEG artifact 是否引入额外退化

3. 如果目标是在 CARLA 上重新训练，训练侧必须与推理侧冻结同一套预处理。建议第一阶段先保留当前单前视路径，并用新增运行时开关做小规模消融：
   - `DIFFUSIONDRIVE_JPEG_ARTIFACT=1/0`
   - `DIFFUSIONDRIVE_IMAGE_NORMALIZATION=none/imagenet`

4. 多相机拼接不是简单预处理开关，需要同时改 leaderboard 传感器注册、图像拼接、训练数据采集和 checkpoint 兼容策略，应作为单独决策项处理。

## 后续待冻结决策

- 是否正式继续单前视，还是升级到 CARLA 多相机拼接。
- 若继续单前视，是否保持当前去底部裁剪，还是改成更接近 NAVSIM 的上下裁剪。
- CARLA 重新训练时是否保留 JPEG artifact。
- CARLA 重新训练时是否引入 ImageNet mean/std normalization。
- 训练数据加载、在线推理和可视化调试是否共用同一份预处理 helper，减少未来分叉。

## 最小测试计划

当前输入预处理改动只引入运行时开关，默认行为不变。建议按以下顺序验证：

1. 静态检查：
   ```bash
   cd /home/HeavenlySU/sitp_workspace/carla_garage
   python -m py_compile team_code/diffusiondrive_agent.py
   python -c "import sys; sys.path.insert(0, 'team_code'); import diffusiondrive_agent; print(diffusiondrive_agent.get_entry_point())"
   ```

2. 默认闭环 smoke test：
   ```bash
   export DIFFUSIONDRIVE_JPEG_ARTIFACT=1
   export DIFFUSIONDRIVE_IMAGE_NORMALIZATION=none
   ```
   目标是确认短 route 能正常进入模型推理，日志显示 JPEG artifact 开启、normalization 为 `none`。

3. 预处理开关 smoke test：
   ```bash
   export DIFFUSIONDRIVE_JPEG_ARTIFACT=0
   export DIFFUSIONDRIVE_IMAGE_NORMALIZATION=none
   ```
   ```bash
   export DIFFUSIONDRIVE_JPEG_ARTIFACT=1
   export DIFFUSIONDRIVE_IMAGE_NORMALIZATION=imagenet
   ```
   目标先只验证不崩溃和 shape 正确，不用立刻判断性能优劣。

4. 小规模 A/B：
   - 同一 checkpoint、同一 route、同一 seed / traffic 条件。
   - 比较默认、JPEG off、ImageNet normalize 三组。
   - 记录 route 是否完成、是否更早碰撞或卡死、control 是否抖动、creep / safety box / stop sign 触发次数是否异常。

5. 训练前离线统计：
   - 抽样保存 resize 前后图像。
   - 对 JPEG on/off 的输入均值、方差和简单特征分布做对比。
   - 若考虑多相机，先做视野覆盖可视化，再决定是否改传感器注册和训练数据规范。
