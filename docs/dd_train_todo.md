# DiffusionDrive CARLA Training TODO

本文档记录面向后续 CARLA 训练的事项。格式参考 `dd_todo.md`，但关注点从“当前推理 agent 还缺什么”切换为“训练和训练后闭环需要先定义和验证什么”。

**更新日期**: 2026-07-15
**判断基线**:

- 当前计划是在 CARLA 上重新训练 DiffusionDrive
- 因此训练侧优先级高于继续对齐 navsim checkpoint 语义
- full `baseline-basic` 已在远端使用 4 卡 L40 / DDP 完成训练：B2D Full scenario-balanced 全量 manifest、`epochs=100`、per-GPU `batch_size=64`、global batch `256`、`lr=6e-4`、`image_encoder_lr_mult=0.5`、无 hard-case weighting、无 Stage5/6 tuned checkpoint 初始化
- baseline-basic 开环 / 闭环结果已完成并下载到本地 `dd_logs` 镜像路径；open-loop six-scene `l1_mean=0.0092`，all-scenarios `l1_mean=0.0192`，Bench2Drive 220 closed-loop `DS=44.81`、`RC=79.48`、`NDS=35.52`
- baseline-basic 已完成一组 20-route 闭环 A/B；当前最明显正向方向是空间 PID 速度参数，`A8_pid_6_2p5` 更均衡，`A9_pid_7_3` DS/NDS 最高但 collision / timeout 风险更高
- condition-v1 C2/L1 对照、`original/zero/shuffle` 配对开环和第一批 train-vs-online contract 已完成：正确 LiDAR 在 B2D raw 上显著优于 zero/shuffle，online full scan 无 gross yaw/覆盖错误，但前向动态点云晚一 tick 且密度漂移场景相关。当前先排除随机 diffusion mode switching 和 `spatial_path` target 不连续，再做 channel/fusion 结构改造；zero-LiDAR 仍只作为诊断上界
- 训练文档同时覆盖训练前定义、训练中实验项、训练后闭环验证项
- 需要区分 `status_feature` 和 `extra_sensors` 两套输入机制
- 当前 `DiffusionDriveAgent` 显式构造的是 `status_feature = command_one_hot(6) + speed(1)`
- `carla_garage/team_code/model.py` 中的 `extra_sensors` 是可选分支：按配置拼接 `velocity(1)` 与 / 或 `discrete_command(6)`，再经 `extra_sensor_encoder` 编码
- `syb_carla_garage` 里的 `extra_sensors` 关键设计是 `speed(1) + command_one_hot(6)`，并作为低维条件 token 接入 transformer decoder；这个思想已吸收到 DiffusionDrive 的 `status_feature`，但不在 DD 中额外保留一套独立 `extra_sensors` 分支
- 当前新提取的聚类 anchor 文件 `4-0-0-1910-tracked_clusters_anchor.npy` 语义为 `99x10x2` 空间 route/checkpoint anchor
- 其来源 JSON 中每个 cluster 的 `mu` 都是一条拉直后的 `10` 个路点 `(x, y)` 轨迹，即一个 `20D` 向量
- 当前训练和后续修改以 Bench2Drive Full raw 数据为主训练分布；sensor / time / preprocessing contract 见 `b2d_full_sensor_contract.md`

---

## TODO List

### P0 级别（训练前必须明确）

- [ ] **训练输入定义冻结**
  - [x] 明确相机输入方案：第一阶段采用 CARLA-native 单前视，不采用 NAVSIM 三相机拼接
  - [x] 明确图像裁剪 / resize / normalize 方案：`512x1024 -> crop_array 384x1024 -> model input 384x1024 -> [0,1] -> optional ImageNet normalization`
  - [x] 明确 LiDAR 输入通道定义：沿用 garage histogram，默认 `use_ground_plane=False`，每帧 `1` 个 above-split 通道；多帧时按 `lidar_seq_len` 拼接
  - [x] 记录旧 `status_feature` 事实：`command_one_hot(6) + velocity(speed,0) + acceleration(accel_x,0)`，共 `10` 维
  - [x] 记录 `extra_sensors` 来源：garage / syb 旧模型中为 `speed(1) + command_one_hot(6)` 的低维条件 token
  - [x] 明确 DiffusionDrive 不新增独立 `extra_sensors` 分支；将其思想合并进 `status_feature`
  - [x] 将 `status_feature` 迁移到推荐的 `command_one_hot(6) + speed(1)`，共 `7` 维

- [ ] **训练标签与轨迹定义冻结**
  - [x] 修正 raw Bench2Drive target 坐标系：优先使用 ego vehicle `world2ego` 矩阵，缺失时 fallback 到 `preprocess_compass()` 等价处理后的 `x/y/theta`
  - [x] 用 ego vehicle `world2ego` / `location` 抽样复核 `ego_relative_xy()`；mini 抽样 92 对 frame 的矩阵目标误差为 `0`
  - [x] 明确默认 trajectory target 采用 `spatial_path`：从 future ego path 按 `2.5m + 1.0m` 空间距离重采样
  - [x] 明确 anchor 是空间 checkpoint 语义，不是固定时间间隔 trajectory
  - [x] 明确 `trajectory_sampling.interval_length` 当前保留为兼容字段，不代表默认 `spatial_path` target 的真实时间间隔
  - [x] 推理侧默认改用空间 checkpoint PID，不再按 0.5s / 1.0s waypoint index 估计 desired speed
  - [x] 明确 anchor 生成方式与数组形状
  - [x] 校验最新 `99x10x2` anchor 与源 JSON 一致，并记录直接升级路线，见 `diffusiondrive_anchor_adaptation.md`
  - [x] 明确 heading 不由轨迹 head 预测，主轨迹接口固定为 XY
  - [x] 明确采用新聚类 anchor：`99` 个 mode，每个 mode 对应 `10x2` 轨迹点

- [ ] **数据集生成规范**
  - [x] 明确当前训练主线以 B2D Full raw 数据为主，不把它仅作为 smoke / 预训练材料
  - [x] 初步记录 B2D Full raw camera / LiDAR sensor contract，见 `b2d_full_sensor_contract.md`
  - [x] dataset helper 兼容 B2D Full 原生 `rgb + measurements` route 结构，以及早期 `camera/rgb_front + anno` 结构
  - [x] 增加 scenario-balanced discovery：`--balanced-scenarios` 按 `route_dir.parent.name` 分桶并 round-robin 合并，`--max-samples-per-scenario` 限制每类场景样本数
  - [ ] 明确 B2D Full raw sensor 与在线 `DiffusionDriveAgent` sensor suite 的 gap 是否需要在推理侧对齐或单独做 domain adaptation
  - [ ] 明确 camera FOV / pose、LiDAR pose / yaw / range 与在线 agent 的一致性验证标准
  - [ ] 明确数据保存频率和时序长度
  - [ ] 明确正式训练 / 验证 / 测试切分方式；当前已有多 scenario eval-only probe，`NonSignalizedJunctionLeftTurn` 对顺序截断较敏感

- [x] **最小训练入口**
  - [x] 新增 `team_code/train_diffusiondrive.py`
  - [x] 新增 raw Bench2Drive 数据集 helper `team_code/diffusiondrive/carla_native_dataset.py`
  - [x] 支持 trajectory loss-only 单机训练
  - [x] 用 Bench2Drive mini 跑通 1-step training smoke

- [x] **第一版训练配置落盘**
  - [x] 训练入口支持 optimizer / scheduler / warmup 配置
  - [x] 训练入口支持 `--image-encoder-lr-mult`，可按原版 DiffusionDrive 给 `image_encoder` 参数组使用 `0.5x` 学习率
  - [x] 训练入口支持 `torchrun` / DDP 多卡训练，训练集使用 `DistributedSampler`，rank0 负责 validation / checkpoint / 配置落盘
  - [x] 训练入口支持 trajectory loss weights、focal alpha/gamma 和 diffusion timestep 配置
  - [x] 训练入口支持 validation split 参数、checkpoint resume 和 `latest.pth`
  - [x] 训练入口支持 `--eval-only`，可从训练 checkpoint 只加载 `model` 跑验证集
  - [x] 训练入口支持 scenario-balanced 采样参数，并在日志中打印 scenario sample counts
  - [x] 训练入口支持 low-speed left-turn hard-case loss weighting：`command=1 && speed<0.1 && abs(target_end_y)>4`
  - [x] 训练入口会落盘 sample distribution JSON，统计 command、speed bin、`abs(target_end_y)` bin 和 hard-case 数量
  - [x] 训练入口支持 sample manifest JSONL，缓存 route/frame、command、speed 和 trajectory target，减少每个 epoch 反复解未来 annotation
  - [x] sample manifest 读取时校验 header 中的数据选择和 target 语义参数，避免 baseline 长训误用旧缓存
  - [x] 新增 `tools/build_diffusiondrive_manifest.py`，支持 CPU-only 作业并行预构建 train / val manifest，避免 GPU 作业等待缓存生成
  - [x] manifest builder 支持 `--quality-filter none|soft_clean|syb_clean`，可直接构建 full / soft-clean / syb-clean 训练样本缓存
  - [x] `tools/inspect_diffusiondrive_eval_errors.py` 支持 `--sample-manifest`，eval CSV 诊断可复用训练 / 验证同一份缓存样本列表
  - [x] `tools/inspect_diffusiondrive_eval_errors.py` 已补齐 `--skip-first-frames`、`--model-image-height/width` 和 `--image-normalization`，避免 condition-v1 诊断脚本沿用旧 `256x1024` 默认
  - [x] DataLoader worker 会限制 OpenCV / torch 内部线程，并支持 `--prefetch-factor` 与可选 `--persistent-workers`
  - [x] 远端 CPU 瓶颈优化已落到文档主线：manifest 只缓存样本元数据和 trajectory target，不缓存图像 / LiDAR；DataLoader worker 配合 `OMP_NUM_THREADS=1` 等环境变量避免在 7 CPU 核限制下过度抢线程
  - [x] 输出目录落盘 `training_config.json`，记录 CLI、DiffusionDrive config、数据 split、预处理和 status feature schema
  - [x] `training_config.json` 记录 B2D Full dataset mode、target mode、空间 checkpoint 采样、frame interval 假设、anchor shape 和 sensor contract
  - [x] condition-v1 manifest loader 默认要求 `target_speed_label` 与 `route_condition_feature` metadata，不再对旧 manifest 静默逐样本 fallback
  - [x] 已完成 full baseline-basic 训练，作为论文 baseline 和后续持续学习工作的干净基础；Stage5 / Stage6 tuned hard-weight checkpoint 只作为 ablation / 改进参考
  - [x] 已完成 baseline-basic full-scenario open-loop 和 Bench2Drive 220 closed-loop 汇总，并归档到本地 `dd_logs/eval_summaries`
  - [ ] 后续仍需把实验配置从 CLI-only 进一步整理成可复用 config 文件或 launch preset

### P1 级别（直接影响训练效果）

- [ ] **`status_feature` 重定义**
  - [x] 明确当前 command 维度为 CARLA `command_one_hot(6)`
  - [x] 明确 syb / garage `extra_sensors` 对应的是 `speed(1) + command_one_hot(6)`，不是 DD 旧实现的 `10` 维 status
  - [x] 明确设计方向：DD 只保留 `status_feature` 入口，不额外引入旧模型式独立 `extra_sensors` 分支
  - [x] 实现推荐 schema：`status_feature = command_one_hot(6) + speed(1)`
  - [x] 训练 / 推理默认统一使用当前 command；推理侧旧 `commands[-2]` 延迟逻辑保留为 `DIFFUSIONDRIVE_COMMAND_DELAY=1` fallback
  - [x] 将训练侧和推理侧改为共用同一份 status builder，避免 schema 分叉
  - [x] 迁移时删除 / 废弃 acceleration 输入；旧实现中训练用 IMU `acceleration[0]`、推理用 speed finite difference，二者不一致
  - [x] `baseline-condition-v1` 中保留该 7 维输入作为 `status_token`，不恢复旧 `extra_sensors`
  - [ ] 明确 speed 是否归一化；syb 旧模型通过 `BatchNorm1d(1, affine=False)` 处理速度，DD 迁移时需要单独决策

- [ ] **`SpeedHead-v1` 显式速度 / 刹车语义**
  - [x] 扩展 raw B2D sample discovery / manifest，缓存 `target_speed`、`brake`，并在 header 中记录 speed label schema
  - [x] dataset batch 的 `targets` 已输出 `target_speed`、`brake`、`target_speed_twohot`、`target_speed_class`、`target_speed_label_valid`
  - [x] 采用 syb 风格 speed bins 作为第一版：`[0.0, 4.0, 8.0, 10.0, 13.8889, 16.0, 17.7778, 20.0]`，其中 `0.0` 类承担 brake / stop 语义
  - [x] 在 DiffusionDrive 模型中新增独立 speed query + MLP head，不改 diffusion trajectory head
  - [x] 训练入口新增 `--speed-loss-weight`、`target_speed_loss`、`target_speed_accuracy`、`target_speed_brake_accuracy`、`target_speed_l1`
  - [x] 推理侧新增 `DIFFUSIONDRIVE_USE_SPEED_HEAD=1` predicted-speed longitudinal controller，保留当前 spatial PID desired speed 作为 fallback / ablation
  - [x] 记录 `C1_speedhead` direct controller 失败：route 00 起步阶段持续 class 0 / desired speed 0 / brake 1，导致车辆锁死，route 00 得分约 `2.35`，不应继续跑 direct speed-head 20-route / 220-route
  - [x] 2026-06-11 已按 syb 思路将 speed-head 推理接法改为 uncertainty-weighted speed conversion：默认只有 `p(class0)>0.9` 才强制 `desired_speed=0`，否则用 `sum(prob * target_speeds)`；`DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT=0` 保留 argmax 诊断
  - [ ] 新版 speed-head 先重跑 route 00 / 24 sanity；如果仍有低速锁死，再升级为 spatial PID 主控的 gated speed cap / brake gate
  - [ ] 将 run 命名为 `baseline-condition-v1` 或更具体的 `speedhead_v1_*`，避免和已完成的 `baseline-basic` 混淆

- [x] **两个 condition token 的 route conditioning**
  - [x] 模型接口改为两个低维 condition token：`status_token=command_one_hot(6)+speed(1)`，`route_condition_token=target_point(2)+target_point_next(2)`
  - [x] 不做 11 维 flat `status_feature`，也不恢复旧 garage / syb 的独立 `extra_sensors` 分支
  - [x] 训练侧从 measurements / manifest 读取并缓存 `target_point`、`target_point_next`
  - [x] 推理侧复用 `DiffusionDriveAgent.tick()` 已计算的 ego-frame `target_point`、`target_point_next`
  - [x] 在 `training_config.json` 中记录 condition token schema 和维度
  - [x] 已明确该改动改变模型接口，必须 full retrain；如 warm-start baseline-basic，需要显式跳过新增层并在 run notes 中标注

- [ ] **图像预处理方案验证**
  - [x] 记录当前推理侧图像预处理事实，见 `diffusiondrive_input_preprocessing.md`
  - [x] `baseline-condition-v1` 输入主线改为单前视、去底部裁剪、`384x1024` 模型输入、默认 JPEG artifact，并支持训练/推理 ImageNet normalization 开关
  - [x] 明确 B2D Full raw 是当前主训练分布，见 `b2d_full_sensor_contract.md`
  - [x] 修正 B2D source image size 记录：当前远端训练文件实际为 `512x1024`，训练参数默认已改为 `--b2d-source-image-height 512 --b2d-source-image-width 1024`
  - [ ] 评估 JPEG artifact 是否需要保留
  - [ ] 评估 `--image-normalization imagenet` 对 open-loop / closed-loop 的影响
  - [ ] 评估单前视与多相机的收益差异

- [ ] **LiDAR 表征方案验证**
  - [x] 明确当前 CARLA DiffusionDrive 模型侧 LiDAR 是 NAVSIM-style 单帧 BEV histogram + CNN encoder，默认 `DiffusionDriveConfig.lidar_architecture="resnet34"`，不是 syb 的 `regnety_032`
  - [x] 明确 `regnety_032` 是 `timm` 2D CNN backbone 名称，不是 LiDAR 专用表示；切换它需要 full retrain，不能用当前 ResNet34 checkpoint warm-start 得到正式结论
  - [x] 记录 zero-LiDAR 诊断边界：`DIFFUSIONDRIVE_ZERO_LIDAR=1` 只置零模型侧 `lidar_feature`，不关闭 safety-box raw LiDAR
  - [x] 记录 condition-v1 L0 诊断：共同 19 条 route 上 zero model-LiDAR `DS=43.58 / RC=82.10 / IP=0.509`，相对 C0 common19 的 `DS=39.07 / RC=86.31 / IP=0.433` 提升 DS/IP 但降低 RC，并新增 6 条 route deviation
  - [x] 记录 condition-v1 L1 诊断：A9 PID + zero model-LiDAR full20 `DS=50.77 / RC=88.97 / NDS=34.95 / IP=0.544`，`12 completed / 5 deviated / 3 blocked`，stuck 从 C0 的 `3158` 降到 `783`，safety-box stop 从 `12179` 降到 `8132`
  - [x] 补 `condition-v1 + A9 PID + LiDAR ON` 对照：C2 full20 `DS=41.00 / RC=94.40 / IP=0.422 / NDS=19.56`，对比同 A9 的 L1 zero-LiDAR 为 `DS -9.77 / RC +5.44 / IP -0.122 / NDS -15.39`；LiDAR-on 提升 completion / 减少终止性 deviation，但碰撞、越线、低速和 blocked 更差
  - [x] 把 no-LiDAR / zero-LiDAR 定位为诊断上界，而不是正式主线；项目最终路线仍应使用 LiDAR
  - [x] `inspect_diffusiondrive_eval_errors.py` 支持 `--lidar-mode original|zero|shuffle`，并在 shuffle CSV 中记录确定性 donor 样本，用于同一 checkpoint / manifest 的开环 LiDAR 归因
  - [x] 实现共享 LiDAR BEV contract 统计 / dump，对比 B2D `.laz -> histogram` 与 online `half-scan concat -> histogram` 的点数、保留率、z 分位数、非零率、通道均值 / 饱和率和左右前后 occupancy
  - [x] 完成 `39,432` 样本配对开环：original/zero/shuffle mean L1 为 `0.02047/0.31397/0.58965`，确认模型在 B2D raw 上正确利用 LiDAR，而不是忽略该分支
  - [x] 完成第一批 online contract 对比：拼接 full scan angular coverage 正常；主要剩余问题是前向动态点云晚一 tick，以及 route 153 等场景的 density/occupancy drift
  - [x] 增加确定性 diffusion inference 入口：`random/fixed/zero/seeded` noise，输出 mode、top-1/top-2 margin、entropy 和 endpoint；fixed 不依赖 batch size，seeded 按稳定 sample key 生成显式 noise
  - [x] 运行 fixed seed 0-4 all-scenarios 开环：五组各 39,432 样本，mean L1 范围 `0.01875...0.02106`，确认随机 noise 是次级尾部因素
  - [ ] 对 route 24/50/139/153 跑 fixed seed 多 attempt 闭环，量化 CARLA attempt 方差
  - [x] 实现 `arc_length_stable_extrapolation_v2`：外推使用至少 `0.5m` trailing displacement，不足时使用 route condition；旧 manifest schema 明确失效
  - [x] 完成 `39,432` 样本 v2 配对门控：key/order 和 condition/speed/brake 不变量零差异，`>12m` jump 从 `119` 降到 `19`，`>5m` 从 `1,271` 降到 `1,162`
  - [x] 实现 `arc_length_route_aligned_extrapolation_v3`：可靠 trailing displacement 还必须与 route condition cosine `>=0`；19 条剩余 trace 的本地反事实重算将 `>12m` 降到 7，且剩余事件 route-condition delta 均 `>=35m`
  - [x] 完成 `39,432` 样本 v3 capped 配对门控：`>12m=7`、`>5m=1,151`，没有新阈值越界，低速且 condition 稳定的 `>12m` 为 0；剩余 7 条的 condition delta 为 `35m...250m`
  - [x] 构建无 scenario cap 的正式 v3 train manifest 并完成全量门控：`113,008` 样本、`5,304` routes、`107,704` transitions，key/order 与全部 invariant 零差异；`>12m 269 -> 10`、低速稳定 `>12m 238 -> 0`，`>5m 2,916 -> 2,650`、低速稳定 `>5m 379 -> 80`
  - [x] 构建匹配 `--balanced-scenarios --val-max-samples 1024` 的 v3 `NonSignalizedJunctionLeftTurn` validation manifest：`946` transitions，`>12m=0`、`>5m=62`，无低速稳定事件
  - [x] 用正式 v3 train/val manifest 完成 4 卡 full retrain：`100 epochs / 44,200 steps`，`--num-workers 12 --prefetch-factor 2` 全程无 OOM、worker kill、NaN 或 traceback；run 为 `soft_clean_skip25_imgnet_v3target_ddp4_bs64x4_lr6e-4_ep100`
  - [ ] 让旧 condition-v1 与新 v3 checkpoint 在同一 v3 eval manifest、同一确定性 diffusion noise 下做配对开环；之后检查旧版 `L1>1m` route 和 80 条低速稳定中等跳变附近的尾部误差
  - [ ] 做模型侧 LiDAR channel ablation：above-only、below-only、all-zero、original，确认负贡献来自哪个通道或整体 distribution gap
  - [ ] 在确定性推理、target 连续性和 channel/temporal 归因后，评估 LiDAR fusion gate / dropout full retrain，避免把多个问题混入一次训练
  - [ ] 补 `agent_states / agent_labels / bev_semantic_map` auxiliary supervision，让 LiDAR 分支学习可解释的近场几何 / 动态体语义，而不是只靠 trajectory loss 间接学习
  - [ ] 如继续对齐 syb，再评估 `regnety_032` full retrain；不要只把当前 ResNet34 checkpoint warm-start 到 RegNet
  - [ ] 验证 raw Bench2Drive LiDAR `x=-0.39,z=1.84,yaw=0,range=85` 与 garage 在线 LiDAR `x=0,z=2.5,yaw=-90` 的训练/推理 gap
  - [ ] 暂不把多帧 LiDAR 作为当前主线；只有单帧 LiDAR 的 contract / channel / fusion 结论明确后，再决定是否启用 `lidar_seq_len>1`
  - [ ] 明确 histogram 参数是否沿用当前 garage 设置
  - [ ] 验证时序 realign 对训练标签的一致性

- [ ] **轨迹表示统一**
  - [x] 明确最终采用 `99x10x2` 轨迹表示
  - [x] 训练与推理统一使用同一套 anchor / 归一化 / 输出语义
  - [x] 明确不在轨迹 head 中保留 heading；如后续需要 heading-aware 控制，另行设计监督与接口
  - [x] 明确新聚类结果直接作为 `99x10x2` 正式 anchor，不再重采样 / 截断为兼容格式

- [ ] **loss / optimization 定标**
  - [x] 增加针对 `NonSignalizedJunctionLeftTurn` 已知高误差模式的可选样本级 loss weighting，默认关闭，validation / eval-only 保持未加权
  - [x] full baseline-basic 原版对齐实验使用 `image_encoder_lr_mult=0.5`，避免 `lr=6e-4` 直接作用到全部 image encoder 参数
  - [ ] 99 modes 下重新评估 focal alpha/gamma、`trajectory_cls_weight`、`trajectory_reg_weight`
  - [ ] 评估外层 `trajectory_weight` 接入后总梯度尺度，记录 grad norm 和 loss breakdown
  - [x] 明确 image encoder 采用更小 LR；freeze BN / freeze backbone warmup 暂不启用
  - [ ] 小 batch 训练前确认 BatchNorm 策略：frozen BN、SyncBN、或足够大的 effective batch

### P2 级别（训练后闭环增强）

- [ ] **辅助头训练策略**
  - [ ] 评估是否保留 `agent_states / agent_labels / bev_semantic_map`
  - [ ] 若保留模型侧 LiDAR，优先评估这些辅助头是否能给 BEV / obstacle representation 更强监督，避免 LiDAR 分支在 trajectory / speed 监督不足时成为闭环噪声
  - [ ] 明确多任务损失是否继续保留

- [ ] **评测与控制闭环设计**
  - [x] baseline-basic 已完成一次 sensor-only Bench2Drive 220 闭环，确认当前空间 PID 可以完成评测但闭环表现较弱
  - [x] 当前 target / anchor 是空间 checkpoint 语义，`DiffusionDriveAgent` 默认空间 PID 已不再用 waypoint 时间索引估计 desired speed
  - [x] 已确认当前开环 `l1 / ade / fde` 与 leaderboard closed-loop 指标不充分对齐：baseline-basic 开环极低误差仍只得到 `DS=44.81`
  - [x] 已完成 20-route 闭环 A/B 初筛：空间 PID speed tuning 优于单独 stuck-threshold sweep
  - [ ] 设计面向闭环 failure modes 的训练 / 调参指标，重点覆盖 route deviation、blocked、低速和 collisions

- [ ] **训练工程可复现性**
  - [ ] `--resume-file` 当前只从 checkpoint 的下一个 epoch 继续，不恢复 dataloader epoch 内位置；长训前决定是否需要精确断点恢复
  - [x] 明确 DDP 策略：用 `torchrun` 自动启用，训练集 `DistributedSampler`，rank0 负责 validation / checkpoint；checkpoint 保存普通非 `module.` state dict
  - [ ] 明确 full training 的 AMP / gradient accumulation 策略
  - [ ] 给 full 数据 smoke 增加吞吐量、显存、grad norm、target 分布统计

- [ ] **闭环调参与可观测性**
  - [x] `DiffusionDriveAgent` UKF 已加入 covariance 正定保护和 measurement reset，避免部分 route 在前几秒因 `numpy.linalg.LinAlgError` 直接 agent crash
  - [x] `DIFFUSIONDRIVE_STUCK_THRESHOLD` / `DIFFUSIONDRIVE_CREEP_DURATION` / `DIFFUSIONDRIVE_CREEP_THROTTLE` 已支持 env 覆盖，并已重跑 `_real` stuck-threshold ablation
  - [x] 已确认 creep 不能仅从 leaderboard aggregate 判断，需要统计 `Detected agent being stuck` 与 `Creeping stopped by safety box` 日志
  - [ ] 调 `safety_box_*` 阈值，适配训练后模型的闭环行为
  - [ ] 评估 `stuck_threshold` / `creep_duration` 是否需要和空间 PID speed 联调
  - [ ] 收敛 creep / safety box 相关日志输出，避免长跑日志过噪

---

## 当前事实

### 已完成部分

#### 1. 推理侧已具备可运行闭环

当前推理 agent 已经可以：

- 接收 CARLA 传感器
- 构建 DiffusionDrive 输入
- 输出轨迹
- 通过 PID 执行

这意味着训练侧后续不需要再从零定义推理入口，而是可以直接围绕已有 agent 迭代。

#### 2. LiDAR 时序链路已在推理侧接入

当前推理侧已经接入：

- 半帧 LiDAR 拼接
- `lidar_buffer`
- 历史 LiDAR realign
- stuck detection
- safety box

这为后续训练后如何做闭环调参提供了直接参考。多帧 LiDAR 目前不再是默认下一步；应先确认单帧 BEV LiDAR 在训练 / 推理之间的 contract 是否一致，并用 zero/no-LiDAR 作为诊断对照，而不是把移除 LiDAR 当最终路线。

#### 3. 新聚类 anchor 已提取并接入当前链路

当前仓库根目录下已存在：

- `4-0-0-1910-tracked_clusters.json`
- `4-0-0-1910-tracked_clusters_anchor.npy`

其中：

- cluster 数为 `99`
- 每个 cluster 的 `mu` 长度为 `20`
- 语义上等价于一条拉直后的 `10` 个路点 `(x, y)` 轨迹
- 提取后的 anchor 文件 shape 为 `99x10x2`

这说明新的聚类结果已经可供训练侧使用；当前决策是直接将推理 / 训练接口升级到 `99x10x2`，而不是生成兼容版 `20x8x2`。

#### 4. 2026-05-19 训练风险扫描结论

本次扫描当时没有改代码，只对训练链路做静态检查和 mini 数据统计。后续已修正 target 坐标主路径；full training 前仍需冻结时间语义，否则长训结果不可用。

**最高优先级问题**：

- `carla_native_dataset.py` 曾直接用 `anno["theta"]` 构造 `ego_relative_xy()`。但在线推理侧会先对 IMU compass 做 `preprocess_compass()`，等价于 `theta - pi/2`。在 Bench2Drive mini 上，旧 target 与 anno 中 ego `world2ego` 矩阵计算结果的误差约为：
  - mean error: `17.05m`
  - p95 error: `56.50m`
  - 若先做 `theta - pi/2`，误差降到 mean `0.87m`、p95 `1.70m`
- raw Bench2Drive frame 从 bbox 位移 / speed 估计约为 `0.1s` 间隔。默认 target 已不再使用 `future_stride=10` 的 fixed-time 采样，而是从 future ego path 空间重采样为 `2.5m, 3.5m, ..., 11.5m` checkpoint。
- raw Bench2Drive 传感器与 garage 在线 agent 传感器几何差异很大：
  - Bench2Drive front camera: `1600x900, fov=70, x=0.8,z=1.6`
  - garage front camera: `1024x512, fov=110, x=-1.5,z=2.0`
  - Bench2Drive LiDAR: `x=-0.39,z=1.84,yaw=0,range=85`
  - garage LiDAR: `x=0,z=2.5,yaw=-90`
- 旧 `status_feature` 中 acceleration 训练/推理不一致。训练 helper 用 IMU `acceleration[0]`，推理 agent 用 speed finite difference；mini 上二者相关性约 `0.23`。当前已移除 acceleration 输入。
- command 训练 / 推理默认统一为当前 command；推理侧旧 `commands[-2]` 一拍延迟保留为环境变量 fallback。

**直接建议**：

1. 在 full 数据上复核 target builder，使 `ego_relative_xy()` 与 `world2ego` 矩阵继续保持对齐。
2. 复核空间 checkpoint target builder 在 Full 数据上的 target 分布，并确认 `trajectory_sampling.interval_length` 是否只保留为兼容字段。
3. 训练主线已确定为 B2D Full raw；继续显式记录 raw sensor 与在线 garage sensor suite 的 FOV / pose / LiDAR 外参 gap。
4. 再跑 full 数据小 smoke 和吞吐量测试。

---

## P0 级别问题（训练前必须明确）

### 1. `status_feature` 需要重新定义并冻结

**问题描述**：

既然后续计划在 CARLA 上重新训练，推理侧的 `status_feature` 已按当前训练主线重新定义。这里最容易混淆的是：DiffusionDrive 当前的 `status_feature` 和 garage / syb 旧模型里的 `extra_sensors` 不是同一个接口。

**至少要明确**：

- command 用 6 维 one-hot，默认训练 / 推理都使用当前 command；旧 `commands[-2]` 延迟只作为推理侧 ablation
- speed 是否归一化
- acceleration 已移除
- syb / garage `extra_sensors` 的 7 维设计已吸收到 DD 的 `status_feature`

**当前代码事实**：

- `DiffusionDriveAgent` 当前显式构造的是 `status_feature = command_one_hot(6) + speed(1)`
- 训练侧与推理侧共用 `diffusiondrive.status` 中的 status builder
- `carla_garage/team_code/model.py` 中的 `extra_sensors` 则是另一套可选输入分支：
  - 若 `use_velocity=True`，加入 `velocity_normalization(ego_vel)`，贡献 `1` 维
  - 若 `use_discrete_command=True`，加入 `command`，贡献 `6` 维
  - 两者拼接后送入 `extra_sensor_encoder`
- `syb_carla_garage` 里默认关注的是 `use_velocity=1`、`use_discrete_command=True`、`use_tp=True`、`tp_attention=True`，其中 `extra_sensors` 本质是 `speed(1) + command_one_hot(6)`，再投影为 decoder token
- 因此这里真正需要冻结的是：DD 采用哪个低维条件 schema，而不是把 `status_feature` 和 `extra_sensors` 当成两套并存输入

**结论**：

当前方向是：DiffusionDrive 只保留 `status_feature` 入口，不新增独立 `extra_sensors` 分支；把 syb / garage `extra_sensors` 的核心设计吸收到 `status_feature = command_one_hot(6) + speed(1)`。这会把旧 `10` 维 status 改成 `7` 维，`_status_encoding` 需要重建，旧 checkpoint 中这一层按 shape mismatch 跳过即可。

### 2. 相机输入方案需要冻结

**问题描述**：

当前推理侧使用单前视相机，但原版 navsim 更接近多相机拼接输入。

**已冻结的第一阶段方案**：

- 继续用 CARLA-native 单前视相机。
- 不按 NAVSIM 三相机拼接作为当前主线。
- 在线传感器保持 `512x1024`，裁剪为 `384x1024`，当前 `baseline-condition-v1` 默认模型输入也是 `384x1024`。
- 训练侧必须与推理侧保持同一套裁剪、resize 和 normalization 语义。
- 多相机拼接保留为后续单独实验，不能混入第一阶段训练基线。

### 3. 数据规范需要尽早固定

**问题描述**：

如果训练数据的采集规范和推理时假设反复变化，后续实验会很难归因。

**需要明确**：

- 数据保存频率
- LiDAR 时序长度
- 相机布局
- 轨迹标签定义
- anchor 生成方式
- 是否围绕新的 `99x10x2` 聚类 anchor 组织训练数据与标签

---

## P1 级别问题（影响训练效果）

### 4. 图像预处理与 LiDAR 表征要做小规模消融

**问题描述**：

在 CARLA 上重新训练前，最好先把最可能影响效果的输入因素拆开验证。

**建议优先做的消融**：

1. 单前视 vs 多相机
2. 当前 resize / crop 方案 vs 备选方案
3. LiDAR BEV contract / channel ablation / fusion gate vs 当前单帧 BEV LiDAR
4. 是否保留 JPEG artifact

多帧 LiDAR 暂不作为当前优先消融。Z0/Z1 和 condition-v1 L0 说明模型侧 BEV LiDAR 接入不稳定；在确认单帧 LiDAR 的 train-vs-online BEV contract、通道贡献和 fusion 强度前，直接增加历史帧会让归因更复杂。

### 5. 轨迹表示需要在训练前统一

**问题描述**：

当前实现已经将主轨迹接口统一为 `99x10x2`。对于继续训练来说，训练标签也应按 `10x2` XY 轨迹组织；若数据构建阶段暂时保留 heading，loss 只会使用前两维 XY。

另外，新提取出的聚类 anchor `4-0-0-1910-tracked_clusters_anchor.npy` 目前是 `99x10x2`。它对应的原始 JSON 里，每个 cluster 的 `mu` 都是一条拉直后的 `10` 点轨迹，也就是一个 `20D` 向量；这已作为新的正式 anchor 接口。

**已明确**：

- 训练目标采用 2D XY 轨迹
- anchor、归一化、loss、输出头和控制入口统一为 `99x10x2`
- 控制器当前只消费 XY waypoint，不利用 heading
- 新聚类 anchor 直接作为正式训练 anchor
- `trajectory_sampling.num_poses`、loss、控制链路和 checkpoint 策略已按 `99x10x2` 同步升级
- 当前不再生成兼容版 anchor；后续训练直接围绕 `99x10x2` 组织

### 6. 训练后闭环参数需要重新调

**问题描述**：

即使模型重新训练后效果更好，闭环侧也未必能直接沿用当前阈值和日志策略。

**需要重点关注**：

- `safety_box_*` 阈值需要按新模型行为重新调
- `stuck_threshold` / `creep_duration` 需要和模型停车/起步风格、空间 PID speed 联调
- 当前 creep / safety box 相关 `print` 适合调试，但长时间评测时可能日志过噪
- 如果目标评测环境是 Bench2Drive，还要额外考虑 `AgentBlockedTest` 更早触发的问题
- 当前 20-route 结果显示，A8/A9 减少了 safety-box stop loops，但 A9 在个别 route 上 forced creep ticks 更多；creep 需要作为独立 debug 指标，不应只看 DS/RC

**建议**：

- 训练后单独做闭环调参阶段，不要把训练提升和阈值变化混在一起评估
- 如需长时间批量评测，考虑将这类日志改为可配置 debug 开关或限频输出
- 如果跑 Bench2Drive，优先确认 stuck recovery 的触发时机是否明显早于 blocked timeout
- 不要直接沿用当前 `stuck_threshold = 1100` 的默认值去跑 Bench2Drive 闭环评测；也不要只调 stuck threshold，应优先联调空间 PID speed、safety box 和 creep 参数

---

## P2 级别问题（中长期增强）

### 7. 辅助头和多任务训练是否保留

**问题描述**：

当前模型还带有：

- `agent_states`
- `agent_labels`
- `bev_semantic_map`

**需要判断**：

- 这些头是保留作为正则与辅助监督
- 还是在 CARLA 训练中裁掉以简化训练目标

### 8. 评测闭环需要提前考虑

**问题描述**：

训练目标如果和最后上线的执行链路不一致，效果容易失真。

**需要明确**：

- 最终是否仍然通过 waypoint PID 执行
- 如果是，训练轨迹和控制器参数要如何协同调
- 是否要引入辅助头帮助安全或可视化

---

## 建议的推进顺序

1. `baseline-condition-v1` soft-clean + skip-first full retrain 已完成；闭环先用 `DIFFUSIONDRIVE_USE_SPEED_HEAD=0` 的 spatial PID fallback 评估 trajectory + route token。
2. 旧 `C1_speedhead` direct controller 路线已失败；2026-06-11 已接入 syb-style uncertainty-weighted conversion，新版应先重跑 route 00 / 24 sanity，再决定是否跑 20-route。
3. open-loop all-scenarios 检查 trajectory / speed metrics，尤其是 `target_speed_l1`、brake accuracy 和坏场景 top-k。
4. 结合 Z0/Z1 和 condition-v1 L0 结果，优先修复 LiDAR 接入：先做 LiDAR BEV contract 统计 / dump、channel ablation，再评估 fusion gate / dropout 和 auxiliary supervision。
5. no-LiDAR / zero-LiDAR 只作为诊断上界；项目最终路线仍应保留 LiDAR。若要换 `regnety_032`，必须作为 full retrain ablation，不要直接 warm-start 当前 ResNet34 checkpoint。
6. 训练后只把少数候选放到固定 20-route 闭环，接近或超过 A8/A9/A12/Z1 后再跑 220-route。
7. 闭环阶段继续记录 `safety_box_*`、stuck / creep 和 route deviation debug，但不再密集扫 PID 插值。
8. 最后再考虑持续学习、route-aware guard / blend、sensor-only stop sign 和更复杂控制头。

---

## 相关文档

- `docs/dd_todo.md`: 推理侧后续事项
- `docs/diffusiondrive_agent_explained.md`: 当前 agent 的实现说明
- `docs/run.md`: 早期手工 CARLA / leaderboard debug 命令备忘
- `docs/diffusiondrive_local_env.md`: 本地 smoke / 环境备忘
