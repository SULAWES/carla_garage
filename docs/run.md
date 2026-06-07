> 注意：本文是早期手工 CARLA / leaderboard debug 命令备忘，包含旧 NAVSIM checkpoint 路径和双窗口 / tmux 流程。当前 DiffusionDrive 训练、开环评估、闭环脚本和 baseline-condition-v1 进度以 `docs/diffusiondrive_training.md`、`docs/diffusiondrive_remote_training_progress.md`、`docs/diffusiondrive_baseline_next_steps.md` 和 `tools/run_baseline_basic_closed_loop.sh` 为准。不要把本文命令直接当作当前 full baseline 或 condition-v1 的标准命令。

申请GPU
srun -p L40 -J ltr_debug -w gpu4013 -N 1 -n 1 --gres=gpu:l40:1 --cpus-per-task=6 --pty /bin/bash

启动tmux

加载PYTHONPATH

```bash
./ltr_python_path.sh
```

进入carla_root

```bash
cd ~/syb/carla_0_9_15
conda activate ltr_garage_2
./CarlaUE4.sh -RenderOffScreen -nosound
```

新开窗口

```bash
cd ~/ltr/carla_garage
conda activate ltr_garage_2
```

```bash
export DIFFUSIONDRIVE_ANCHOR_PATH=/share/home/u19666033/ltr/plan_anchor.npy
export DIFFUSIONDRIVE_CHECKPOINT=/share/home/u19666033/ltr/diffusiondrive_navsim_88p1_PDMS
export DIFFUSIONDRIVE_BACKBONE_PATH=/share/home/u19666033/ltr/pytorch_model.bin

export CARLA_ROOT=/share/home/u19666033/syb/carla_0_9_15
export WORK_DIR=/share/home/u19666033/ltr/carla_garage
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export WORK_DIR=/share/home/u19666033/ltr/carla_garage
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
```

```bash
python3 /share/home/u19666033/ltr/carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS \
  --routes /share/home/u19666033/ltr/carla_garage/leaderboard/data/debug.xml \
  --checkpoint /share/home/u19666033/ltr/
```

```bash
python /share/home/u19666033/ltr/carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS
```
