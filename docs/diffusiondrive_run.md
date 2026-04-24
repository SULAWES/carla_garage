# DiffusionDrive CARLA 运行方法（Leaderboard）

本文档说明如何在 `carla_garage` 中使用 `DiffusionDriveAgent` 进行评测或调试。

> 说明：本文档保留通用运行方法。与当前机器、集群、路径强绑定的个人运行备忘仍放在 `docs/run.md`，该文件不在这里重复。

## 1. Agent 入口

- Agent 文件：`carla_garage/team_code/diffusiondrive_agent.py`
- `get_entry_point()` 返回：`DiffusionDriveAgent`

## 2. 必要环境变量

在运行 evaluator 前设置以下环境变量：

- `DIFFUSIONDRIVE_ANCHOR_PATH`（必须）
  - plan anchor `.npy`
  - 当前可直接运行的推荐形状：`(20, 8, 2)`
  - 仓库中另有一份新提取的聚类 anchor `4-0-0-1910-tracked_clusters_anchor.npy`，shape 为 `99x10x2`
  - 该文件来自 `99` 个聚类中心，每个 cluster 的 `mu` 是一条拉直后的 `10` 个 `(x, y)` 路点轨迹，也就是一个 `20D` 向量
  - 由于当前 agent 默认链路仍按 `20x8x2` 组织，这份 `99x10x2` anchor 目前不能直接替换现有运行 anchor
- `DIFFUSIONDRIVE_CHECKPOINT`（可选，但通常建议提供）
  - DiffusionDrive 权重 `.pth/.ckpt`
- `DIFFUSIONDRIVE_BACKBONE_PATH`（可选）
  - timm backbone 权重
  - 不设置时，agent 会尝试读取当前工作目录下的 `pytorch_model.bin`

最小示例：

```bash
export DIFFUSIONDRIVE_ANCHOR_PATH=/abs/path/to/anchor.npy
export DIFFUSIONDRIVE_CHECKPOINT=/abs/path/to/diffusiondrive.ckpt
export DIFFUSIONDRIVE_BACKBONE_PATH=/abs/path/to/pytorch_model.bin
```

运行 evaluator 前，还需要按 `carla_garage` 的常规方式配置：

- `CARLA_ROOT`
- `SCENARIO_RUNNER_ROOT`
- `LEADERBOARD_ROOT`
- `PYTHONPATH`

这些环境变量的机器相关写法请参考 `docs/run.md`。

## 3. 运行（本地 evaluator）

常见做法是把 `--agent` 指向本 agent 文件：

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS
```

如果需要指定 routes：

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS \
  --routes /abs/path/to/routes.xml
```

如果你跑 MAP track：

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track MAP
```

说明：

- 具体还需要你按 `carla_garage` 的 leaderboard 配置补齐 `routes`、`repetitions`、`port`、`checkpoint` 等参数。
- 如果你已经有自己的运行脚本，只需要替换 `--agent` 路径即可。

## 4. 运行（标准 evaluator）

如果你使用非 local 的 evaluator，同样替换 `--agent`：

```bash
python leaderboard/leaderboard/leaderboard_evaluator.py \
  --agent team_code/diffusiondrive_agent.py
```

## 5. 常见问题排查

### A) 启动即报 anchor 相关错误

`DIFFUSIONDRIVE_ANCHOR_PATH` 未设置会直接报错；另外请确认 `.npy` 形状与当前模型假设一致。

特别注意：

1. 当前默认可运行链路假设的是 `20x8x2`
2. `4-0-0-1910-tracked_clusters_anchor.npy` 虽然格式正确，但它是 `99x10x2`
3. 若直接把 `DIFFUSIONDRIVE_ANCHOR_PATH` 指向这份新文件，通常还需要同步处理 mode 数、时间步长度以及相关 checkpoint 兼容问题

### B) 权重加载后 missing / unexpected / shape mismatch 较多

先确认三件事：

1. checkpoint 对应的网络结构是否与当前 `V2TransfuserModel` 一致
2. checkpoint 是否带有包装前缀，例如 `agent.` / `model.` / `module.` / `_transfuser_model.`
3. 当前权重是否本来就不是按 CARLA 侧输入定义训练出来的

当前 agent 会打印加载摘要，优先以摘要为准，不要只看是否“加载成功”。

### C) 缺少依赖

DiffusionDrive 模块会依赖一些 Python 包，例如：

- `diffusers`
- `timm`
- `einops`

如果环境缺包，需要在运行环境里补齐。
