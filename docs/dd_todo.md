# DiffusionDrive CARLA TODO

本文档记录基于当前代码状态整理出的 `DiffusionDriveAgent` 后续事项。格式参考旧版 `diffusiondrive_claude.md`，但结论以现在的实现为准。

**更新日期**: 2026-04-26
**判断基线**:

- 当前代码状态以 `carla_garage/team_code/diffusiondrive_agent.py` 为准
- 旧版问题分析已移入 `docs/outdated/`
- `status_feature` 暂不作为近期对齐项，因为后续计划在 CARLA 上重新训练
- 需要区分两套机制：
- `DiffusionDriveAgent` 当前显式构造的是 `status_feature = command(6) + velocity(2) + acceleration(2)`
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
  - [ ] 评估 actor-based controller 是否符合目标 leaderboard / sensor-only 约束；若不符合，需要迁移为 sensor-only 的 bbox / route-aware stop sign 逻辑
  - [ ] 评估是否需要进一步迁移为 bbox / route-aware stop sign box 更新逻辑

- [ ] **LiDAR / BEV 对齐验证与近场修复**
  - [x] 添加合成验证脚本 `tools/validate_lidar_bev_alignment.py`
  - [x] 验证在线 `DiffusionDriveAgent.align_lidar()` 与离线 `CARLA_Data.align()` 当前实现一致
  - [x] 验证 `DiffusionDriveAgent` 的负索引历史帧配对优于 `sensor_agent.py` 的旧正索引写法
  - [x] 用 v3-v9 离线脚本验证动态目标遮挡、粗搜索范围、固定 ROI、common-support、位移先验等路径
  - [x] 用 batch residual 统计排除明显全局固定 `dx/dy` 偏移；当前不优先按外参或坐标系常量偏差修复
  - [x] 新增批量诊断与 contact sheet 输出，便于检查 top scenario 和 worst gain case
  - [ ] 人工检查 `lidar_bev_v9_batch_100_1` 的 contact sheet，确认近场 `0-8m` raw IoU 退化主要来自动态目标、遮挡还是 scorer 偏置
  - [ ] 下一版 scorer 优先加入近场 `0-8m` raw/dynamic 一致性约束，再用 batch_100 对比 v5/v9
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

- [ ] **轨迹表示一致性**
  - [ ] 明确 CARLA 侧训练 / 推理统一使用 `8x2` 还是 `8x3(x, y, heading)` 轨迹表示
  - [ ] 核对 `plan_anchor`、`norm_odo()/denorm_odo()` 与最终 `trajectory` 输出的维度语义是否一致
  - [ ] 评估是否需要恢复 heading-aware 训练以减少当前“2D anchor + 3D 输出”的半对齐状态
  - [ ] 处理新聚类 anchor `4-0-0-1910-tracked_clusters_anchor.npy` 的时序长度不匹配问题
  - [ ] 当前新 anchor shape 为 `99x10x2`，现有运行链路使用的 anchor shape 为 `20x8x2`，不能直接替换
  - [x] 记录新 anchor 适配路线，见 `diffusiondrive_anchor_adaptation.md`
  - [ ] 选择适配方案：将 `10` 个 pose 重采样 / 截断到 `8` 个 pose，或同步修改 `trajectory_sampling.num_poses`、模型轨迹头和控制链路以支持 `10` 个 pose
  - [ ] 明确 `99` 个 mode 是否需要同步调整当前依赖 `20` 个 anchor mode 的模块与权重兼容性

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

**建议**：

- 用简单 route 先做闭环验证
- 单独记录 creep 触发次数和 emergency stop 次数
- 根据实测结果微调 `safety_box_*`、`stuck_threshold`、`creep_duration`

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
- 当前实现可通过 `STOP_CONTROL` 开关独立启停，适合做 ablation 和规则合规性对比。

**后续方向**：

- 明确当前实验目标是否允许使用 CARLA world actor stop sign controller。
- 如果只做工程闭环和模型训练前验证，可以继续保留 actor-based controller。
- 如果目标是严格 sensor-only leaderboard，优先迁移为 bbox / route-aware stop sign box 更新逻辑，或在 CARLA 重训时显式加入 stop sign 感知与监督。

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

- 当前 `DiffusionDriveAgent` 文档应继续按 `command(6) + velocity(2) + acceleration(2)` 描述 `status_feature`
- `extra_sensors` 是 `carla_garage/team_code/model.py` 中的另一套可选输入机制：若开启，会将 `velocity(1)` 和 / 或 `discrete_command(6)` 拼接后送入 `extra_sensor_encoder`
- 因此后续训练设计里需要明确：是只保留 `status_feature`，还是同时引入独立 `extra_sensors` 分支

### 5. 轨迹表示目前仍处于“部分改成 2D”状态

**问题描述**：

当前 CARLA 版为了适配现有 anchor，已经把 `plan_anchor` 相关处理改成了二维 `(x, y)` 假设；但模型解码输出和原版方法仍然保留了 `(x, y, heading)` 轨迹结构。

另外，仓库中新提取出的聚类 anchor 文件 `4-0-0-1910-tracked_clusters_anchor.npy` 当前为 `99x10x2`，与现有推理所用的 `20x8x2` anchor 设定不一致，因此目前只能作为候选输入，不能直接替换 `plan_anchor.npy`。

**影响**：

- 当前实现能跑，但轨迹表示在 anchor、归一化和最终输出之间不是完全统一的
- 这会增加后续训练设计和调试成本
- 如果未来希望引入更贴合轨迹朝向的控制或分析，这个不一致会变成阻碍
- 新 anchor 若直接接入，会同时引入 mode 数和时序长度两处接口变化，影响模型结构、checkpoint 对齐和控制逻辑

**推荐方向**：

- 明确 CARLA 侧最终采用 `8x2` 还是 `8x3`
- 训练与推理统一使用同一套轨迹表示
- 如果准备在 CARLA 上系统训练，优先评估恢复 heading-aware 监督
- 单独决定新 anchor 的接入策略：先做离线重采样得到 `20x8x2` 兼容版，或系统性升级为 `99x10x2`

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

暂时不作为近期任务。

**原因**：

- 当前计划是在 CARLA 上重新训练
- 现阶段没必要为了对齐 navsim checkpoint 去专门重构 `status_feature`

**结论**：

后续如果进入训练阶段，应直接以 CARLA 训练配置重新定义并固定这部分输入，而不是围绕现有 navsim checkpoint 做局部修补。

---

## 建议的推进顺序

1. 补 stuck detection / creep / safety box
2. 视需要补 stop sign controller
3. 验证 LiDAR 坐标系与 BEV 对齐
4. 核对图像预处理与训练设置
5. 再考虑是否利用辅助头或做 ensemble

---

## 相关文档

- `docs/diffusiondrive_agent_explained.md`: 当前 agent 的实现说明
- `docs/diffusiondrive_run.md`: 通用运行方法
- `docs/run.md`: 机器/环境相关运行备忘
- `docs/outdated/diffusiondrive_claude.md`: 旧版问题分析归档
