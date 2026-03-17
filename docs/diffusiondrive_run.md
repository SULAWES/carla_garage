# DiffusionDrive CARLA 运行方法（Leaderboard）

本文档说明如何在 `carla_garage` 中使用 `DiffusionDriveAgent` 进行评测/调试。

> 注意：我只写了代码与运行说明；未在本环境实际运行验证。

## 1. Agent 入口

- Agent 文件：`carla_garage/team_code/diffusiondrive_agent.py`
- `get_entry_point()` 返回：`DiffusionDriveAgent`

## 2. 必要环境变量

在运行 evaluator 前设置以下环境变量：

- `DIFFUSIONDRIVE_ANCHOR_PATH`（必须）：plan anchor `.npy`
  - 推荐 `(20,8,2)`（只包含 `x,y`；heading 由网络预测）
- `DIFFUSIONDRIVE_CHECKPOINT`（强烈建议）：DiffusionDrive 权重 `.pth/.ckpt`
  - 支持常见前缀清理：`agent.` / `model.` / `module.`
- `DIFFUSIONDRIVE_BACKBONE_PATH`（可选）：timm backbone 权重
  - 不设时会尝试使用工作目录下的 `pytorch_model.bin`（如果存在）

示例：

```bash
export DIFFUSIONDRIVE_ANCHOR_PATH=/abs/path/to/anchor.npy
export DIFFUSIONDRIVE_CHECKPOINT=/abs/path/to/diffusiondrive.ckpt
export DIFFUSIONDRIVE_BACKBONE_PATH=/abs/path/to/pytorch_model.bin
```

export DIFFUSIONDRIVE_ANCHOR_PATH=/home/sulawesi/sitp_workspace/plan_anchor.npy
export DIFFUSIONDRIVE_CHECKPOINT=/home/sulawesi/sitp_workspace/diffusiondrive_navsim_88p1_PDMS
export DIFFUSIONDRIVE_BACKBONE_PATH=/home/sulawesi/sitp_workspace/pytorch_model.bin

export CARLA_ROOT=/home/sulawesi/sitp_workspace/carla_garage/carla
export WORK_DIR=/home/sulawesi/sitp_workspace/carla_garage
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

## 3. 运行（本地 evaluator）

`carla_garage` 自带 leaderboard，本地跑通常用 local evaluator（与仓库内 `README.md` 的用法一致）。

常见做法是把 `--agent` 指向本 agent 文件：

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS
```

```bash
python3 /home/sulawesi/sitp_workspace/carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS \
  --routes /home/sulawesi/sitp_workspace/carla_garage/leaderboard/data/debug.xml \
  
```

如果你跑 MAP track：

```bash
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track MAP
```

说明：

- 具体还需要你按 `carla_garage` 的 leaderboard 配置补齐 routes / repetitions / port 等参数。
- 如果你已经有自己的运行脚本（例如 slurm 脚本或 evaluate 脚本），只需要替换 agent 路径即可。

## 4. 运行（标准 evaluator）

如果你使用非 local 的 evaluator，同样替换 `--agent`：

```bash
python leaderboard/leaderboard/leaderboard_evaluator.py \
  --agent team_code/diffusiondrive_agent.py
```

## 5. 常见问题排查

### A) 启动即报 anchor 相关错误

`DIFFUSIONDRIVE_ANCHOR_PATH` 未设置会直接报错；另外请确认 `.npy` 形状与配置一致（默认按 `(20,8,2)` 读取）。

### B) 权重加载后 missing/unexpected keys 很多

这是预期的“早期移植状态”。建议你：

1) 确认 checkpoint 对应的网络结构是否与当前 `V2TransfuserModel` 一致
2) 观察打印的 missing/unexpected keys，做 key mapping 或修改模块命名

### C) 依赖缺失

DiffusionDrive 模块会依赖一些 python 包（如 `diffusers`, `timm`, `einops` 等）。如果环境缺包，需要在你的运行环境里安装。
