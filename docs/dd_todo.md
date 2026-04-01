# DiffusionDrive CARLA TODO

本文档记录基于当前代码状态整理出的 `DiffusionDriveAgent` 后续事项。格式参考旧版 `diffusiondrive_claude.md`，但结论以现在的实现为准。

**更新日期**: 2026-04-01
**判断基线**:

- 当前代码状态以 `carla_garage/team_code/diffusiondrive_agent.py` 为准
- 旧版问题分析已移入 `docs/outdated/`
- `status_feature` 暂不作为近期对齐项，因为后续计划在 CARLA 上重新训练

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

- [ ] **Stuck Detection**
  - [ ] 添加 `stuck_detector` 计数器
  - [ ] 实现 `force_move` / creep 机制
  - [ ] 接入 `creep_throttle`

- [ ] **Safety Box**
  - [ ] 实现前方 LiDAR 安全框过滤
  - [ ] 添加 emergency stop 逻辑
  - [ ] 接入 `safety_box_*` 配置

### P1 级别（影响性能与可维护性）

- [ ] **Stop Sign Controller**
  - [ ] 添加 stop sign buffer
  - [ ] 迁移停止标志检测逻辑
  - [ ] 评估是否需要 `update_stop_box()`

- [ ] **LiDAR / BEV 对齐验证**
  - [ ] 验证 `lidar_to_ego_coordinate()` 与 `align_lidar()` 的组合是否一致
  - [ ] 确认历史帧 realign 后的 BEV 对齐正确
  - [ ] 必要时增加可视化或调试输出

- [ ] **Config 体系收敛**
  - [ ] 评估 `GlobalConfig` 与 `DiffusionDriveConfig` 的职责边界
  - [ ] 减少 `setup()` 中的手动参数同步
  - [ ] 明确模型参数与运行参数的统一组织方式

- [ ] **输入预处理核对**
  - [ ] 核对图像裁剪 / resize / normalize 是否符合后续训练计划
  - [ ] 核对 JPEG artifact 是否需要保留
  - [ ] 核对单前视输入是否满足当前实验目标

- [ ] **轨迹表示一致性**
  - [ ] 明确 CARLA 侧训练 / 推理统一使用 `8x2` 还是 `8x3(x, y, heading)` 轨迹表示
  - [ ] 核对 `plan_anchor`、`norm_odo()/denorm_odo()` 与最终 `trajectory` 输出的维度语义是否一致
  - [ ] 评估是否需要恢复 heading-aware 训练以减少当前“2D anchor + 3D 输出”的半对齐状态

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

## P0 级别问题（优先补齐）

### 1. 缺少 Stuck Detection 和 Safety Box ⚠️

**问题描述**：

当前 `DiffusionDriveAgent` 还没有迁入 `sensor_agent.py` 中的 stuck detection、creep 和 safety box 逻辑。

**当前影响**：

- 车辆卡住时缺少自动恢复机制
- 更容易触发 `AgentBlockedTest`
- 即使后续加入 creep，如果没有 safety box，也缺少前方障碍物保护

**结论**：

这一项比模型结构问题更优先，因为它直接影响 route 能否稳定跑完。

---

## P1 级别问题（影响性能和维护）

### 2. 缺少 Stop Sign Controller

**问题描述**：

`sensor_agent.py` 中有 stop sign 相关逻辑，`DiffusionDriveAgent` 目前仍未迁入。

**影响**：

- 在带 stop sign 的场景里，当前 agent 更依赖模型自身轨迹是否学会停下
- 如果目标是提高规则合规性，这块值得补

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

**推荐方向**：

- 保留 `GlobalConfig` 作为 CARLA 运行时配置
- 保留 `DiffusionDriveConfig` 作为模型配置
- 新增显式 builder / adapter，例如 `build_dd_config(global_config, overrides=None)`
- 将当前 `setup()` 里的手动字段同步集中到一处维护

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

### 5. 轨迹表示目前仍处于“部分改成 2D”状态

**问题描述**：

当前 CARLA 版为了适配现有 anchor，已经把 `plan_anchor` 相关处理改成了二维 `(x, y)` 假设；但模型解码输出和原版方法仍然保留了 `(x, y, heading)` 轨迹结构。

**影响**：

- 当前实现能跑，但轨迹表示在 anchor、归一化和最终输出之间不是完全统一的
- 这会增加后续训练设计和调试成本
- 如果未来希望引入更贴合轨迹朝向的控制或分析，这个不一致会变成阻碍

**推荐方向**：

- 明确 CARLA 侧最终采用 `8x2` 还是 `8x3`
- 训练与推理统一使用同一套轨迹表示
- 如果准备在 CARLA 上系统训练，优先评估恢复 heading-aware 监督

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
