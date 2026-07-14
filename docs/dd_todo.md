# DiffusionDrive CARLA TODO

本文档记录基于当前代码状态整理出的 `DiffusionDriveAgent` 后续事项。格式参考旧版 `diffusiondrive_claude.md`，但结论以现在的实现为准。

**更新日期**: 2026-07-15
**判断基线**:

- 当前代码状态以 `carla_garage/team_code/diffusiondrive_agent.py` 为准
- 旧版问题分析已移入 `docs/outdated/`
- `status_feature` 已按 CARLA 重新训练主线迁移为 7 维 schema，不再以 NAVSIM checkpoint 对齐为目标
- 当前模型侧 LiDAR 是 NAVSIM-style 单帧 BEV histogram + `resnet34` encoder；syb 常用的 `regnety_032` 是 `timm` 2D CNN backbone，不是 LiDAR 专用表示，切换需要 full retrain
- `DIFFUSIONDRIVE_ZERO_LIDAR=1` 只置零模型 `lidar_feature`，不关闭 raw LiDAR safety-box。C2/L1 对照已完成；后续 `39,432` 样本配对开环显示 original/zero/shuffle mean L1 为 `0.02047/0.31397/0.58965`，证明模型在 B2D raw 上正确使用 LiDAR。online full scan 未发现 gross yaw/覆盖错误，但前向动态点云晚一 tick、density drift 场景相关。正式主线保留 LiDAR，先排除 diffusion inference 随机性和 spatial target 不连续，再做 channel/fusion/supervision
- 需要区分两套机制：
- `DiffusionDriveAgent` 当前显式构造的是 `status_feature = command_one_hot(6) + speed(1)`
- `carla_garage/team_code/model.py` 中另有可选 `extra_sensors` 分支，会按配置拼接 `velocity(1)` 与 `discrete_command(6)` 后再编码；它不是固定的“command 6+1 维”

---

## TODO List

### P0 级别（优先补齐运行时稳定性）

- [x] **LiDAR 半帧拼接逻辑** - 已完成
  - [x] 添加 `lidar_last`
  - [x] 实现 `align_lidar()` 方法
  - [x] 实现半帧 LiDAR 拼接生成完整扫描

- [x] **LiDAR 多帧缓冲和时序重对齐** - 已完成
  - [x] 添加 `lidar_buffer`
  - [x] 实现 buffer 未填满前持续等待
  - [x] 实现历史帧时序重对齐
  - [x] 支持 `realign_lidar` 配置

- [x] **Stuck Detection** - 已完成
  - [x] 添加 `stuck_detector` 计数器
  - [x] 实现 `force_move` / creep 机制
  - [x] 接入 `creep_throttle`

- [x] **Safety Box** - 已完成
  - [x] 实现前方 LiDAR 安全框过滤
  - [x] 添加 emergency stop 逻辑
  - [x] 接入 `safety_box_*` 配置

### P1 级别（影响性能与可维护性）

- [x] **Stop Sign Controller**
  - [x] 接入基于 CARLA world stop sign actor 的运行时 stop sign controller
  - [x] 不再依赖 `sensor_agent.py` 的 bbox stop sign 检测头
  - [x] 明确当前选择 actor-based controller 是阶段性工程取舍：先保证 DD 推理链路和规则停车可运行，而不是同时迁移 garage 原模型的 stop sign bbox 检测接口
  - [x] baseline-basic / sensor-only 主线默认关闭 actor-based stop sign controller；需要 privileged ablation 时显式设置 `STOP_CONTROL=1`
  - [ ] 迁移为 sensor-only 的 bbox / route-aware stop sign 逻辑
  - [ ] 评估是否需要进一步迁移为 bbox / route-aware stop sign box 更新逻辑

- [ ] **LiDAR / BEV 对齐验证与近场修复**
  - [x] 添加合成验证脚本 `tools/validate_lidar_bev_alignment.py`
  - [x] 验证在线 `DiffusionDriveAgent.align_lidar()` 与离线 `CARLA_Data.align()` 当前实现一致
  - [x] 验证 `DiffusionDriveAgent` 的负索引历史帧配对优于 `sensor_agent.py` 的旧正索引写法
  - [x] 用 v3-v9 离线脚本验证动态目标遮挡、粗搜索范围、固定 ROI、common-support、位移先验等路径
  - [x] 用 batch residual 统计排除明显全局固定 `dx/dy` 偏移；当前不优先按外参或坐标系常量偏差修复
  - [x] 新增批量诊断与 contact sheet 输出，便于检查 top scenario 和 worst gain case
  - [x] 明确当前 baseline 不消费更久历史 BEV，v3-v9 residual refinement 不是当前主线瓶颈
  - [x] 明确 zero-LiDAR 只诊断模型侧 BEV LiDAR 分支，不能用于判断 safety-box 是否该关闭
  - [x] 记录 condition-v1 L0 zero model-LiDAR 诊断：共同 19 条 route 上 `DS=43.58` 高于 C0 common19 的 `39.07`，但 route completion 降低且 route deviation 从 `0` 增至 `6`，说明 LiDAR 问题不能简单归结为“移除 LiDAR”
  - [x] 记录 condition-v1 L1 zero model-LiDAR 诊断：A9 PID full20 `DS=50.77 / RC=88.97 / NDS=34.95`，当前最强 condition-v1 候选，但需要 A9 LiDAR-on 对照拆分收益来源
  - [x] 补 `condition-v1 + A9 PID + LiDAR ON` 固定 20-route C2 对照：20/20 有效，`DS=41.00 / RC=94.40 / IP=0.422 / NDS=19.56`；同 A9 下 zero-LiDAR 的 L1 为 `50.77 / 88.97 / 0.544 / 34.95`，确认 LiDAR 分支有效但局部安全 / penalty 贡献高度不稳定
  - [x] 增加在线 LiDAR BEV dump / 统计：共享 schema 记录 point count / model 保留率 / z 分位数 / nonzero / channel mean/max / saturation / front-back-left-right occupancy，并拆分 `current_half`、`previous_half_aligned`、`full_scan`、`model_input`；事件和周期 dump 均有限频 / 上限
  - [x] 完成配对开环归因：original 显著优于 zero，zero 又优于 shuffle，排除“模型忽略 LiDAR”和“LiDAR 全局负贡献”
  - [x] 完成第一批 train-vs-online contract：full scan angular coverage 正常；确认前向动态物体点云晚一 tick，且密度/占用 drift 是场景相关而非固定 scale
  - [ ] 第一优先增加确定性 diffusion inference：固定/zero/多 seed noise，并记录 trajectory mode、margin、entropy、endpoint；用多 seed 开环和重点 route 重复闭环量化 mode switching
  - [ ] 修复并验证低速停驻段 `spatial_path` target 的方向连续性，避免未来窗口末端微小位移触发长距离反向外推
  - [ ] 增加模型侧 LiDAR channel ablation：above / below / all-zero / original，定位是地面通道、障碍通道还是整体 BEV contract 有害
  - [ ] 在随机性、target 连续性和 temporal/channel 归因完成后评估 LiDAR fusion gate / dropout，保留 LiDAR 输入并降低未校准分支的早期污染
  - [ ] 若继续出现负贡献，优先补 `bev_semantic_map`、`agent_states`、`agent_labels` 等辅助监督，而不是把 zero-LiDAR 当正式方案
  - [ ] 人工检查 `lidar_bev_v9_batch_100_1` 的 contact sheet，确认近场 `0-8m` raw IoU 退化主要来自动态目标、遮挡还是 scorer 偏置
  - [ ] 当前优先级高于 no-LiDAR final 方案；no-LiDAR / zero-LiDAR 只能作为诊断对照，若后续重启多帧 BEV，再考虑下一版 scorer 加入近场 `0-8m` raw/dynamic 一致性约束并用 batch_100 对比 v5/v9
  - [ ] 低优先级：如果后续仍怀疑 pose 语义，再基于真实 route log 对比当前共享变换公式与标准 SE(2)，不要作为当前主线

- [ ] **Config 体系收敛**
  - [x] 评估 `GlobalConfig` 与 `DiffusionDriveConfig` 的职责边界
  - [x] 新增 `diffusiondrive/config_adapter.py`，集中管理 garage runtime config 到 DD model config 的映射与基础校验
  - [x] 减少 `setup()` 中的手动参数同步
  - [ ] 明确模型参数与运行参数的统一组织方式

- [ ] **输入预处理核对**
  - [x] 核对并记录当前图像裁剪 / resize / normalize 事实，见 `diffusiondrive_input_preprocessing.md`
  - [x] 增加 JPEG artifact 和 ImageNet normalization 的推理侧运行时消融开关，默认保持当前行为
  - [ ] 冻结后续 CARLA 训练的图像裁剪 / resize / normalize 方案
  - [ ] 用小规模闭环或离线特征统计评估 JPEG artifact 是否需要保留
  - [ ] 核对单前视输入是否满足当前实验目标

- [ ] **`SpeedHead-v1` 推理接入**
  - [x] 从模型输出读取 `target_speed_logits` / brake 类概率
  - [x] `DIFFUSIONDRIVE_USE_SPEED_HEAD=1` 时将纵向 desired speed 切到 predicted target speed；默认 `0` 保留当前 spatial PID fallback
  - [x] 在 debug log 中记录 predicted target speed、brake prob、最终 throttle/brake、fallback 状态
  - [x] 2026-06-09 初筛发现 direct speed-head controller 不可用：`C1_speedhead` 在 route 00 起步阶段持续预测 `speed_head_class=0`、`desired_speed=0`、`brake=1`，导致车辆锁死，route 00 得分约 `2.35`
  - [x] 2026-06-11 已按 syb 思路修复 speed-head 标量速度转换：默认 `DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT=1`，仅当 `p(class0) > DIFFUSIONDRIVE_SPEED_HEAD_BRAKE_THRESHOLD`（默认 `0.9`）才强制 `desired_speed=0`，否则用概率期望速度
  - [ ] 新版 speed-head 先重跑 route 00 / 24 sanity；通过后再跑固定 20-route，不能把旧 `C1_speedhead` 结果当作新版结论
  - [ ] 如 syb-style direct 仍不稳定，再将 speed head 推理改成 gated speed cap / brake gate，而不是直接覆盖 spatial PID desired speed
  - [ ] 闭环评测优先固定 20-route；只有 fallback 控制下接近或超过 A8/A9/A12 时再跑 220-route

- [x] **Route condition token 推理接入**
  - [x] 保留 `status_token=command_one_hot(6)+speed(1)`
  - [x] 新增 `route_condition_token=target_point(2)+target_point_next(2)`，复用 `tick()` 已计算的 ego-frame route target
  - [x] 不恢复旧 `extra_sensors` 分支，不把 route 信息简单塞成 11 维 flat status
  - [x] 在 agent 启动日志中打印 route condition 输入值开关；`DIFFUSIONDRIVE_USE_ROUTE_CONDITION=0` 只做 zero-route-token ablation，不恢复旧模型接口
  - [x] 闭环 checkpoint 加载默认拒绝 missing / shape mismatch，并校验 checkpoint preprocessing metadata；只有显式设置 `DIFFUSIONDRIVE_ALLOW_PARTIAL_CHECKPOINT=1` / `DIFFUSIONDRIVE_ALLOW_PREPROCESS_MISMATCH=1` 才允许诊断性绕过

- [ ] **轨迹表示一致性**
  - [x] 明确 CARLA 侧训练 / 推理统一使用 `99x10x2` 轨迹表示，不在轨迹 head 中预测 heading
  - [x] 核对 `plan_anchor`、`norm_odo()/denorm_odo()` 与最终 `trajectory` 输出的维度语义是否一致
  - [x] 消除当前“2D anchor + 3D 输出”的半对齐状态
  - [x] 校验新聚类 anchor `4-0-0-1910-tracked_clusters_anchor.npy` 与源 JSON 完全一致
  - [x] 记录新 anchor 直接升级路线，见 `diffusiondrive_anchor_adaptation.md`
  - [x] 选择适配方案：直接同步修改 `trajectory_sampling.num_poses`、模型轨迹头和控制链路以支持 `99x10x2`
  - [x] 明确 `99` 个 mode 需要同步调整当前依赖 `20` 个 anchor mode 的模块与权重兼容性
  - [x] 用新 anchor 跑 DiffusionDrive 模型实例化 / checkpoint 加载 smoke test

### P2 级别（增强项）

- [ ] **辅助头利用**
  - [ ] 评估 `agent_states / agent_labels / bev_semantic_map` 的调试价值
  - [ ] 评估是否用于 safety check 或可视化

- [ ] **Model Ensemble**
  - [ ] 支持加载多个模型文件
  - [ ] 实现 ensemble 推理
  - [ ] 评估是否需要 NMS 或后处理

- [ ] **Truncated Diffusion 超参数重调**
  - [ ] 评估训练时截断噪声 timestep 范围是否仍适合 CARLA 分布
  - [ ] 评估推理时 `step_num` / `trunc_timesteps` / scheduler 相关设置是否需要重新搜索
  - [ ] 在速度、稳定性和轨迹质量之间重新权衡截断 diffusion 的配置

---

## 当前事实

### 已完成部分

#### 1. LiDAR 半帧拼接和时序处理已接入

旧版文档中“DiffusionDrive 仍然只有单帧 LiDAR histogram”的结论已经过时。

当前 `DiffusionDriveAgent` 已经具备：

- `lidar_last`
- 半帧 LiDAR 拼接
- `lidar_buffer`
- buffer 未填满前持续刹车等待
- 历史 LiDAR realign 到当前坐标系

因此，LiDAR 时序链路已经基本接上，当前不再把它列为 TODO。

#### 2. 基础推理链路已打通

当前已经具备完整的 leaderboard 推理链路：

- 传感器接入
- RoutePlanner + UKF 状态估计
- DiffusionDrive 模型加载
- 轨迹输出
- waypoint PID 控制

#### 3. Checkpoint 加载对齐能力已增强

当前 checkpoint 加载已支持：

- 提取常见容器字段
- 清理常见前缀
- 只加载 key 和 shape 都匹配的参数
- 打印 missing / unexpected / shape mismatch 摘要

这足以支撑后续继续迭代或切换新权重。

---

## P0 级别事项（已完成后的验证与调参）

### 1. Stuck Detection 和 Safety Box 已接入，后续重点是调参与验证

**当前状态**：

`DiffusionDriveAgent` 已经接入：

- stuck detection
- `force_move` / creep
- safety box

因此，这一项已经不再属于“缺失功能”，而是进入“效果验证与参数调优”阶段。

**当前关注点**：

- `safety_box_*` 阈值是否过紧或过松
- `stuck_threshold` / `creep_duration` 是否适合当前 route 分布
- creep 是否会在跟车、红灯起步或贴近障碍物时误触发
- creep 不能只从 leaderboard aggregate 指标判断，需要同时统计 `Detected agent being stuck` 和 `Creeping stopped by safety box` 日志

**建议**：

- 用简单 route 先做闭环验证
- 单独记录 creep 触发次数和 emergency stop 次数
- 根据实测结果微调 `safety_box_*`、`stuck_threshold`、`creep_duration`
- 当前 20-route 结果显示，单独调 stuck threshold 不如空间 PID 速度参数有效；A10/A11/A12 后纯 PID 插值边际收益下降，后续应把 creep / safety-box 触发计数作为重训候选的辅助指标，而不是继续密集扫控制参数

---

## P1 级别问题（影响性能和维护）

### 2. Stop Sign Controller 当前是 actor-based 阶段性方案

**问题描述**：

`DiffusionDriveAgent` 已经接入 stop sign controller，但当前实现不是 `sensor_agent.py` 的 bbox stop sign 检测头路线，而是基于 CARLA world actor 的运行时规则 controller。

当前方案在 `_init()` 中通过 `CarlaDataProvider.get_hero_actor()` 获取 ego actor，并用 `RunStopSign(self.hero_actor.get_world())` 跟踪当前 route-relevant stop sign。每步运行时通过 `stop_sign_criteria.tick(hero_actor)` 和 `target_stop_sign`，结合 ego 到 stop sign trigger volume 的距离、当前速度、等待 tick 数，决定是否强制刹车。

选择这个方案的原因是：当前 DiffusionDrive port 的核心目标是先把 NAVSIM DD 的模型推理、LiDAR 时序、轨迹输出和 PID 控制接进 CARLA leaderboard；原 `sensor_agent.py` 的 stop sign 逻辑依赖 garage 原模型的 bbox 输出格式和 stop sign 类别，而当前 DD 模型没有直接产出 `bb[7] == 3` 这种 stop sign bbox 接口。若直接迁移旧逻辑，需要额外补齐 stop sign 监督、检测头类别、bbox 格式对齐、NMS、buffer 运动补偿和 OBB 相交验证，工作量和风险都不属于“基础推理链路接通”本身。

**影响**：

- 好处：stop sign 规则停车与 DD 模型推理解耦，不依赖当前模型是否学会或检测到 stop sign，便于先验证基础闭环运行。
- 风险：actor-based controller 直接读取 CARLA world actor，可能被视为 privileged 信息；如果后续目标是严格 sensor-only leaderboard 合规，需要改回基于传感器/模型输出的方案。
- 当前实现可通过 `STOP_CONTROL` 开关独立启停，适合做 ablation 和规则合规性对比；baseline-basic / sensor-only 主线默认值是 `STOP_CONTROL=0`，只有 privileged ablation 才显式设为 `1`。

**后续方向**：

- baseline-basic / sensor-only 结果默认不允许使用 CARLA world actor stop sign controller。
- 如果只做 privileged 规则停车 ablation，可以显式设置 `STOP_CONTROL=1` 保留 actor-based controller，并在结果中单独标注。
- 如果目标是严格 sensor-only leaderboard，优先迁移为 bbox / route-aware stop sign box 更新逻辑，或在后续 CARLA 训练中显式加入 stop sign 感知与监督。

### 3. 双 Config 设计仍然分裂

**问题描述**：

当前同时维护：

- `GlobalConfig`
- `DiffusionDriveConfig`

并且在 `setup()` 中要手动同步一批参数到 `dd_config`。

**影响**：

- 维护成本高
- 参数容易遗漏
- 后续训练和推理配置不易统一
- `status_feature` 与 `extra_sensors` 的边界也需要一起固化，避免把当前 `DiffusionDriveAgent` 的状态输入和 garage 原模型的可选 `extra_sensors` 分支混为一谈

**推荐方向**：

- 保留 `GlobalConfig` 作为 CARLA 运行时配置
- 保留 `DiffusionDriveConfig` 作为模型配置
- 使用显式 builder / adapter：`build_diffusiondrive_config(global_config, overrides=None)`
- 将当前 `setup()` 里的手动字段同步集中到 `diffusiondrive/config_adapter.py` 维护
- 在该 builder / adapter 中显式声明 `status_feature` 各子项维度，以及是否存在独立 `extra_sensors` 分支，避免后续 checkpoint、训练配置和推理侧对输入接口理解不一致

**不推荐方案**：

- 不建议让 `DiffusionDriveConfig` 继承 `GlobalConfig`

**原因**：

- `GlobalConfig` 体量过大，包含大量 autopilot、控制、数据采集和历史训练参数
- `DiffusionDriveConfig` 未来还需要服务 CARLA 训练，不适合被运行时细节污染
- 当前真正的问题是“映射分散且手写”，而不是“存在两个 config 对象”本身

### 4. 输入分布仍可能与训练侧不完全一致

**问题描述**：

即使 LiDAR 时序已接入，当前输入仍有一些可能影响性能的差异：

- 单前视相机 vs 原版多相机拼接
- 图像裁剪 / resize / normalize 的细节
- JPEG artifact 是否保留
- LiDAR 坐标变换与 BEV 对齐是否完全正确

**影响**：

- 这些问题更偏性能和稳定性，而不是“能不能跑”
- 如果后续准备在 CARLA 上重新训练，这些项应该尽早固化

**补充说明**：

- 当前 `DiffusionDriveAgent` 文档应按 `command_one_hot(6) + speed(1)` 描述 `status_feature`
- `extra_sensors` 是 `carla_garage/team_code/model.py` 中的另一套可选输入机制：若开启，会将 `velocity(1)` 和 / 或 `discrete_command(6)` 拼接后送入 `extra_sensor_encoder`
- 因此后续训练设计里已明确：只保留 `status_feature`，不额外引入独立 `extra_sensors` 分支

### 5. 轨迹表示已明确为 `99x10x2`

**问题描述**：

仓库中新提取出的聚类 anchor 文件 `4-0-0-1910-tracked_clusters_anchor.npy` 当前为 `99x10x2`。当前决策是不再生成兼容版 `20x8x2`，而是直接将推理 / 训练接口升级到 `99x10x2`。

当前实现中 `plan_anchor`、`norm_odo()/denorm_odo()`、模型解码输出、loss 和 agent 控制入口都统一为二维 `(x, y)` 轨迹；heading 不再由轨迹 head 预测。

**影响**：

- 新 anchor 直接接入会同时引入 mode 数和时序长度两处接口变化，影响模型结构、checkpoint 对齐和控制逻辑
- 旧 `20x8x2` checkpoint 的 trajectory head 相关权重会出现 shape mismatch，不能作为最终性能判断基线
- 后续训练标签需要按 `10x2` 组织，若原始标签含 heading，loss 只使用前两维 XY

**推荐方向**：

- 训练与推理统一使用同一套 `99x10x2` 轨迹表示
- 保持控制链路只消费 XY waypoint；如未来需要 heading-aware 控制，另行设计独立监督
- 新 anchor 接入策略已定：系统性升级为 `99x10x2`

---

## P2 级别问题（增强功能）

### 5. 模型辅助头尚未利用

**问题描述**：

当前控制只使用：

- `outputs['trajectory']`

而模型还能输出：

- `agent_states`
- `agent_labels`
- `bev_semantic_map`

**潜在价值**：

- debug 可视化
- safety check
- 误差分析

### 6. Truncated Diffusion 超参数仍沿用原版默认值

**问题描述**：

当前 diffusion 轨迹头中的截断噪声范围、推理步数和 scheduler 设置基本沿用了原版 NAVSIM DiffusionDrive 的默认假设，并未针对 CARLA 数据分布和闭环需求重新搜索。

**影响**：

- 当前配置未必是 CARLA 上最优的速度 / 质量折中
- 可能影响轨迹平滑性、模态选择稳定性和推理耗时
- 如果后续开始 CARLA 重训或系统调参，这会成为一个重要可调轴

**推荐方向**：

- 将训练截断 timestep 范围参数化
- 将推理 `step_num`、`trunc_timesteps`、scheduler 相关设置参数化
- 结合闭环表现而不是只看离线 loss 做调参

### 6. 缺少 Ensemble 支持

**问题描述**：

当前 `DiffusionDriveAgent` 只支持单模型加载，没有像 `sensor_agent.py` 那样支持 ensemble。

**影响**：

- 不能直接复用 garage 的多模型推理思路
- 不是当前主线阻塞，但如果后续想追求上限，可以再考虑

---

## 明确不作为当前 TODO 的事项

### 7. `status_feature` 对齐

**当前决策**：

已按 CARLA 重训主线处理。

**原因**：

- 当前计划是在 CARLA 上重新训练
- 因此 `status_feature` 直接采用 `command_one_hot(6) + speed(1)`，旧 checkpoint 的 `_status_encoding` shape mismatch 按预期跳过

**结论**：

训练和推理共用 `diffusiondrive.status` 的 status builder，避免 schema 分叉。

---

## 建议的推进顺序

1. 基于已完成的 baseline-basic 220 条闭环结果，优先分析 route deviation、blocked、低速和 collisions 的具体触发场景。
2. 保留 `STOP_CONTROL=0` 作为 sensor-only 主线；`STOP_CONTROL=1` 只作为 privileged stop-sign ablation。

注意：早期命名为 `A5_stuck120` / `A6_stuck170` / `A7_stuck300` 的 20-route ablation 是在 `DIFFUSIONDRIVE_STUCK_THRESHOLD` 尚未被代码读取时跑出的，不能解释为 stuck threshold 对比，只能作为重复运行 / 随机性参考。后续 `_real` 后缀重跑已经有效：`A5_stuck120_real` / `A6_stuck170_real` / `A7_stuck300_real` 表明单独调 stuck threshold 不是主要突破口，`A8_pid_6_2p5` 和 `A9_pid_7_3` 的空间 PID 速度参数收益更明显。
3. A10/A11/A12 和 S1/S2 后，纯 PID / camera geometry 小实验的边际收益已下降；后续闭环预算优先留给重训后的少数候选，而不是继续做密集 PID 插值。
4. 结合 Z0/Z1 和 condition-v1 L0 结果，下一步模型侧 LiDAR 优先做 BEV contract 统计 / dump、channel ablation、fusion gate / dropout 和 auxiliary supervision；zero/no-LiDAR 只作为诊断上界，不作为最终主线。
5. 若后续切换到 `regnety_032`，应作为 full retrain ablation；不要只把当前 ResNet34 checkpoint warm-start 到 RegNet。
6. 将 stop sign controller 从 privileged actor-based ablation 迁移到 sensor-only 的 bbox / route-aware 方案，或在论文中仅作为 privileged ablation 单列。
7. 再考虑闭环导向数据增强、持续学习方法、route-aware guard / blend 或更复杂的控制头。

---

## 相关文档

- `docs/diffusiondrive_agent_explained.md`: 当前 agent 的实现说明
- `docs/run.md`: 早期手工 CARLA / leaderboard debug 命令备忘
- `docs/diffusiondrive_local_env.md`: 本地 smoke / 环境备忘
- `docs/outdated/diffusiondrive_claude.md`: 旧版问题分析归档
