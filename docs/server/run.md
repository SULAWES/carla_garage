申请GPU

```bash
srun -p L40 -J ltr_debug -w gpu4013 -N 1 -n 1 --gres=gpu:l40:1 --cpus-per-task=6 --pty /bin/bash
srun -p L40 -J ltr_debug -w gpu4008 -N 1 -n 1 --gres=gpu:l40:1 --cpus-per-task=6 --pty /bin/bash

srun -p amd -J ltr_cpu -N 1 -n 1 --cpus-per-task=12 --pty /bin/bash

```

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

export SAVE_PATH=${WORK_DIR}/results
```

```bash
python3 /share/home/u19666033/ltr/carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS \
  --routes /share/home/u19666033/ltr/carla_garage/leaderboard/data/debug.xml \
  --checkpoint /share/home/u19666033/ltr/diffusiondrive_navsim_88p1_PDMS
```

```bash
python /share/home/u19666033/ltr/carla_garage/leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent team_code/diffusiondrive_agent.py \
  --track SENSORS
```
/share/home/u19666033/ltr/output.md

排查

```bash
lsb_release -a
uname -r
nvidia-smi
ss -lntp | grep -E '2000|2001'
```

```bash
# 1) 基础环境
lsb_release -a
uname -r
nvidia-smi
vulkaninfo | head -n 80

# 2) 端口占用
ss -lntp | grep -E '2000|2001'

# 3) 干净环境启动
unset DISPLAY
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low

# 4) 另一个终端判断是否真活着
pgrep -af CarlaUE4
ss -lntp | grep 2000
nvidia-smi

# 5) 客户端探活
python3 - <<'PY'
import carla
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)
world = client.get_world()
print("connected:", world.get_map().name)
PY

# 6) 找日志/崩溃文件
find ~/.config -type f | grep -E 'CarlaUE4\.log|Diagnostics\.txt|CrashContext'
find . -type f | grep -E 'CarlaUE4\.log|Diagnostics\.txt|CrashContext'
```

hf download

```bash
hf download hustvl/DiffusionDrive diffusiondrive_navsim_88p1_PDMS
```


mkdir -p /share/home/u19666033/ltr/logs/closed_loop_ab

pkill -f CarlaUE4 || true
sleep 5

${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 \
  > /share/home/u19666033/ltr/logs/closed_loop_ab/carla.log 2>&1 &

export CARLA_PID=$!
sleep 30

export DIFFUSIONDRIVE_CHECKPOINT=/share/home/u19666033/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth

export DIFFUSIONDRIVE_SPATIAL_PID=1
export DIFFUSIONDRIVE_COMMAND_DELAY=0

python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
  --agent ${WORK_DIR}/team_code/diffusiondrive_agent.py \
  --track SENSORS \
  --routes ${ROUTES} \
  --checkpoint /share/home/u19666033/ltr/logs/closed_loop_ab/default_result.json \
  2>&1 | tee /share/home/u19666033/ltr/logs/closed_loop_ab/default.log
