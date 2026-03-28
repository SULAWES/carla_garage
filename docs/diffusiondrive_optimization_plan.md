# DiffusionDrive 优化计划

本文档用于记录后续开发计划和当前进度。状态定义如下：

- `DONE`：已完成
- `IN_PROGRESS`：正在做
- `TODO`：尚未开始
- `BLOCKED`：被外部条件阻塞

## 当前状态

更新时间：2026-03-18

### 已完成

- `DONE` 基础移植完成，`DiffusionDriveAgent` 已可在 leaderboard 正常评测
- `DONE` 生命周期问题修复，支持 `destroy(results)` 清理流程
- `DONE` anchor 改为 `(20, 8, 2)` 使用方式
- `DONE` checkpoint 对齐增强
  - 自动提取常见 checkpoint 容器字段
  - 自动剥离常见包装前缀：`agent / model / module / _transfuser_model`
  - 仅加载 key 和 shape 都匹配的参数
  - 输出加载摘要，便于排查缺失权重和 shape mismatch
- `DONE` 识别出当前 checkpoint 的唯一结构性不匹配
  - `_status_encoding.weight: ckpt(256, 8) != model(256, 10)`
  - 已确认问题来源是 `status_feature` 定义不一致，而不是大规模权重未加载
- `DONE` 确认训练侧 `driving_command` 的 4 维语义
  - `left / straight / right / unknown`
  - `left/right` 还覆盖 lane change 和 sharp curve
  - `unknown` 是官方定义的第 4 类，但当前仓库默认没有启用基于 `unknown` 的筛样逻辑

### 进行中

- `IN_PROGRESS` 核对现有 checkpoint 的真实加载覆盖率
  - 下一步需要结合实际 checkpoint 日志确认关键模块是否完整加载
- `BLOCKED` 设计 CARLA 6 类 command 到 NAVSIM 4 类 command 的映射规则
  - 训练侧总维度已确认是 `4 + 2 + 2 = 8`
  - 该问题暂时搁置，后续如果继续沿用当前 checkpoint 再恢复处理
  - 当前决策：暂时搁置，不作为下一阶段优先开发项

### 待做

- `TODO` 移植 `sensor_agent.py` 的多帧 LiDAR buffer 与对齐逻辑
- `TODO` 对齐训练侧图像/LiDAR/status 预处理
- `TODO` 重构 `status_feature`，与训练定义严格一致
  - 先完成 CARLA 6 类 command 到训练侧 4 类 command 的映射规则，再改代码
- `TODO` 重新调优 PID 控制参数
- `TODO` 尝试提高 diffusion 推理步数
- `TODO` 评估 route conditioning 增强
- `TODO` 评估多相机输入

## 下一步建议

建议先做这两项：

1. 读取一次真实 checkpoint 的加载摘要，确认哪些核心参数还没对齐
2. 开始移植多帧 LiDAR 时序逻辑

这两项通常能最快带来可见收益。

## 暂时搁置

- `PAUSED` 当前 checkpoint 的 `status_feature` 维度/command 语义不一致问题
  - 原因：如果后续改为基于当前扩散头重新训练新模型，这个问题可以在训练数据与模型定义阶段一并解决
  - 当前结论：继续使用现有 checkpoint 时，它是性能问题；重新训练时，则应直接用新的状态定义重新训练 `_status_encoding`
