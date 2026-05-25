# DiffusionDrive Remote Training Progress

本文档记录在远端服务器上进行的 Bench2Drive Full / CARLA-native DiffusionDrive 训练进展和诊断结论。

## 远端环境

- 登录入口：`ssh tj_server`
- 主要代码目录：`~/ltr/carla_garage`
- B2D Full 数据目录：`/share/home/u19666033/djy/carla_dataset`
- conda 环境：`ltr_garage_2`
- GPU 分区：`L40`
- 典型交互式作业命令：

```bash
srun -p L40 -J ltr_debug -w gpu4013 -N 1 -n 1 \
  --gres=gpu:l40:1 \
  --cpus-per-task=6 \
  --pty /bin/bash
```

数据结构为 `scenario/route` 两层，例如：

```text
carla_dataset/
  Accident/
    Town13_.../
      rgb/*.jpg
      lidar/*.laz
      measurements/*.json.gz
      records.json.gz
      results.json.gz
```

逐帧训练标注来自 `measurements/*.json.gz`；`records.json.gz` / `results.json.gz` 是 route 级元数据。

## 代码能力现状

当前远端训练依赖以下能力：

- dataset helper 兼容两类 route 结构：
  - 早期 / mini：`camera/rgb_front + lidar + anno`
  - B2D Full 原生：`rgb + lidar + measurements`
- 默认 target mode 是 `spatial_path`：
  - 按 `2.5m, 3.5m, ..., 11.5m` 从 future ego path 空间重采样
  - anchor 语义按空间 route/checkpoint 处理，不按固定时间间隔处理
- status feature 已迁移为 `command_one_hot(6) + speed(1)`，共 `7` 维
- `train_diffusiondrive.py` 支持：
  - `--eval-only`
  - `--balanced-scenarios`
  - `--max-samples-per-scenario`
  - Full 原生 `rgb + measurements` 结构
- `tools/inspect_diffusiondrive_eval_errors.py` 可输出 per-sample `l1 / ade / fde` 和 top-k 高误差 route/frame
- 训练入口已加入 CPU / IO 优化：
  - `--sample-manifest` / `--val-sample-manifest` 把 route/frame、command、speed 和 trajectory target 缓存为 JSONL，避免长训中每个 epoch 反复读取未来 `measurements/*.json.gz`
  - DataLoader worker 初始化时限制 OpenCV / torch 内部线程，并支持 `--prefetch-factor` 与可选 `--persistent-workers`
  - 在学校 1 GPU 最多 7 CPU 核限制下，远端训练优先用 `--num-workers 5/6`，并在 shell 里设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`、`OPENCV_NUM_THREADS=1`

## 远端 Smoke

Full 原生结构 smoke 已跑通：

```bash
python team_code/train_diffusiondrive.py \
  --root-dir /share/home/u19666033/djy/carla_dataset \
  --route-glob "*/*" \
  --logdir ~/ltr/dd_logs/full_smoke \
  --id full_smoke_l40 \
  --epochs 1 \
  --batch-size 1 \
  --max-samples 8 \
  --max-steps 2 \
  --frame-sampling 50 \
  --num-workers 0 \
  --device cuda:0 \
  --load-file ""
```

结果：

```text
Dataset samples: 8
epoch=0 step=1 lr=0.0001 loss=192.5114 avg=192.5114 trajectory_unweighted=16.0426
Saved checkpoint: ~/ltr/dd_logs/full_smoke/full_smoke_l40/checkpoint_epoch000_step0000002.pth
```

结论：Full 原生 `scenario/route -> rgb/lidar/measurements -> spatial_path target -> forward/backward -> checkpoint` 链路已通。

## Stage1: 顺序截断训练

训练配置：

- 输出目录：`~/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4`
- checkpoint：`latest.pth` / `checkpoint_epoch019_step0005120.pth`
- `batch-size=16`
- `frame-sampling=5`
- `max-samples=4096`
- `epochs=20`
- `scheduler=cosine`
- `min-lr=1e-6`

训练末期：

```text
epoch=19 step=5120 lr=1e-06 loss=17.2832 avg=25.2835 trajectory_unweighted=1.4403
```

多场景 eval-only：

| Scenario | trajectory_unweighted |
|---|---:|
| Accident | 1.9680 |
| HighwayCutIn | 1.7679 |
| InterurbanActorFlow | 3.9680 |
| NonSignalizedJunctionLeftTurn | 11.3806 |
| ParkingCutIn | 3.2051 |
| VehicleTurningRoute | 4.3801 |

结论：

- Stage1 能稳定收敛。
- `NonSignalizedJunctionLeftTurn` 明显偏高。
- 单独 finetune `NonSignalizedJunctionLeftTurn` 后，验证 `trajectory_unweighted` 降到约 `4.4306`，说明它不是完全学不了，主要和场景覆盖 / 数据分布有关。

## Stage2: Scenario-Balanced 训练

为避免 `--max-samples` 按目录排序截断导致前几个 scenario 过度占比，加入 scenario-balanced discovery：

- `--balanced-scenarios`
- `--max-samples-per-scenario 256`

训练配置：

```bash
python team_code/train_diffusiondrive.py \
  --root-dir /share/home/u19666033/djy/carla_dataset \
  --route-glob "*/*" \
  --logdir ~/ltr/dd_logs/full_stage2 \
  --id balanced_spatial_path_bs16_256ps \
  --epochs 20 \
  --batch-size 16 \
  --frame-sampling 5 \
  --balanced-scenarios \
  --max-samples-per-scenario 256 \
  --num-workers 6 \
  --scheduler cosine \
  --warmup-steps 1000 \
  --min-lr 1e-6 \
  --save-every-steps 1000 \
  --log-every 50 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --device cuda:0 \
  --load-file ~/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4/latest.pth
```

训练末期：

```text
epoch=19 step=6200 lr=1.01423e-06 loss=24.7669 avg=18.4123 trajectory_unweighted=2.0639
Saved checkpoint: ~/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/checkpoint_epoch019_step0006240.pth
```

多场景 eval-only：

| Scenario | Stage1 | Stage2 |
|---|---:|---:|
| Accident | 1.9680 | 1.3760 |
| HighwayCutIn | 1.7679 | 0.8122 |
| InterurbanActorFlow | 3.9680 | 2.0905 |
| NonSignalizedJunctionLeftTurn | 11.3806 | 8.3305 |
| ParkingCutIn | 3.2051 | 1.8166 |
| VehicleTurningRoute | 4.3801 | 1.2720 |

结论：

- balanced sampling 明显改善大多数场景。
- `NonSignalizedJunctionLeftTurn` 也下降约 `27%`，但仍显著高于其它场景。
- 后续不能只靠顺序截断训练；balanced sampling 应作为 B2D Full 主线默认策略之一。

## NonSignalizedJunctionLeftTurn 误差诊断

使用 `tools/inspect_diffusiondrive_eval_errors.py` 对 Stage2 checkpoint 在 `NonSignalizedJunctionLeftTurn` 上做 per-sample 诊断。

CSV：`/home/HeavenlySU/sitp_workspace/nsj_left_errors.csv`

总体统计：

```text
rows: 1024
l1 mean: 0.6884
l1 median: 0.0658
l1 p90: 2.4620
l1 p95: 4.3950
l1 max: 7.3713
```

按 command：

| Command | Samples | mean L1 | median L1 | max L1 |
|---:|---:|---:|---:|---:|
| 1 | 400 | 1.6582 | 0.7026 | 7.3713 |
| 2 | 11 | 0.4860 | 0.4515 | 1.5211 |
| 3 | 12 | 0.0451 | 0.0396 | 0.0749 |
| 4 | 601 | 0.0595 | 0.0354 | 1.2478 |

高误差样本定义为 `l1 > 2`：

```text
high-error count: 118
command=1: 118 / 118
speed < 0.1: 108 / 118
abs(target_end_y) > 4: 117 / 118
三者同时满足: 107 / 118
```

最典型错误：

```text
route: Town12_Rep0_3388_0_route0_11_08_18_28_21
frame: 350
speed: 0.0
command: 1
target_end: (-1.46, -11.38)
pred_end:   (11.43, -0.16)
l1: 7.371
```

route 聚合中，`Town12_Rep0_3388_0_route0_11_08_18_28_21` 是明显异常 route：

```text
mean l1: 3.10
median l1: 3.15
max l1: 7.371
n: 168
l1 > 2: 107
```

结论：

- `NonSignalizedJunctionLeftTurn` 不是全局都差。
- 高误差高度集中于 `command=1`、静止 / 低速、target 终点横向偏移大的左转帧。
- 模型在这些样本上倾向输出近似直行轨迹，而 target 是强左转轨迹。

## 当前判断

当前 trajectory-only baseline 已经证明：

- B2D Full 原生数据能训练。
- `spatial_path` target 和 `99x10x2` anchor 能收敛。
- scenario-balanced 采样有效。
- 主要短板已经从“训练链路是否能跑”转为“复杂路口左转 / 静止后起步转向的条件建模和采样权重”。

## Stage3 / Stage4: Hard-Case Weighting 与 Manifest 长训

Stage3 使用 Stage2 checkpoint 初始化，并对 `command=1 && speed<0.1 && abs(target_end_y)>4` 的低速左转 hard case 做 loss weighting。`hard_left_weight5_bs32_512ps` 在 `NonSignalizedJunctionLeftTurn` 的 per-sample CSV 诊断上相对 Stage2 有小幅改善：

| Metric | Stage2 | Stage3 weight5 |
|---|---:|---:|
| NSJ L1 mean | 0.6884 | 0.6062 |
| NSJ L1 p95 | 4.3950 | 4.1710 |
| NSJ hard-case L1 mean | 3.7465 | 3.4214 |
| NSJ `l1 > 2` count | 118 | 106 |

同时抽查 `VehicleTurningRoute` 和 `HighwayCutIn` 未见明显退化，其中 `HighwayCutIn` 的 L1 mean 约 `0.0282`，`VehicleTurningRoute` 的 L1 mean 约 `0.0743`。

Stage4 主要用于验证新加入的 manifest / DataLoader CPU 优化能支撑更长训练：

- `--sample-manifest /share/home/u19666033/ltr/dd_cache/full_stage4_train_1024ps_fs5_spatial.jsonl`
- `--val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl`
- `--max-samples-per-scenario 1024`
- `--batch-size 64`
- `--num-workers 6`
- `--prefetch-factor 2`
- `--dataset-stats-max-samples 0`

这一路线工程上能减少 target 构造带来的 CPU 压力，但当前 Stage4 中途验证 `validation step=9000 trajectory_unweighted=2.1736`，弱于 Stage3 在 step 3000 的 `1.3921`。因此 Stage4 暂时只作为“缓存优化已接入长训命令”的工程记录，不应直接当作效果最佳 checkpoint；下一步更应考虑 hard-case oversampling，而不是单纯继续扩大每场景样本数或拉长训练。

## 建议下一步

优先级从高到低：

1. 使用已新增的 hard-case loss weighting 做小实验：
   - `--hard-left-turn-stop-loss-weight` 会对 `command=1 && speed<0.1 && abs(target_end_y)>4` 样本加权。
   - 先从 Stage2 checkpoint 短训，观察 `NonSignalizedJunctionLeftTurn` 是否继续下降，以及其它场景是否受损。
2. 优先实现 hard-case oversampling：
   - 当前 weighting 能改善但 hard case 在 batch 中仍然稀疏，Stage4 日志里常见 `hard_left_turn_stop=0-3/64`。
   - 建议在 dataset 层按同一 hard-case predicate 复制训练样本，validation / eval-only 保持原始分布。
3. 使用已新增的 sample distribution JSON 确认 hard cases 是否进入训练：
   - 脚本会落盘 command、speed bin、`abs(target_end_y)` bin 和 hard-case 数量。
   - 用 `--dataset-stats-max-samples` 控制统计样本数；默认最多统计 `4096` 个样本。
4. 可视化 top error 样本：
   - raw image
   - target trajectory
   - predicted trajectory
   - speed / command
5. 再考虑扩大 `max-samples-per-scenario` 到 `512 / 1024` 做更长 balanced stage，但需要用 manifest 缓存降低 CPU 开销，并保留 NSJ eval / CSV 诊断。
6. 进入 CARLA 闭环前，先确认远端已同时同步 `team_code/diffusiondrive_agent.py` 和 `team_code/config.py`，否则新版 agent 会因缺少 `GlobalConfig.diffusiondrive_spatial_pid*` 参数在 setup 阶段失败。然后使用默认空间 PID 做 smoke，并 A/B `DIFFUSIONDRIVE_SPATIAL_PID=0` 的旧 time-index fallback；同时可打开 `DIFFUSIONDRIVE_DEBUG_CONTROL=1`、`DIFFUSIONDRIVE_DEBUG_INTERVAL=20` 记录 command / desired speed / turn ratio / aim waypoint / control，重点观察路口低速转弯、停车起步和 emergency stop 触发。
