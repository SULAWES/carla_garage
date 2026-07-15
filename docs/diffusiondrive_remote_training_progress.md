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

Stage1/2 的 trajectory-only baseline 已经证明：

- B2D Full 原生数据能训练。
- `spatial_path` target 和 `99x10x2` anchor 能收敛。
- scenario-balanced 采样有效。
- 主要短板已经从“训练链路是否能跑”转为“复杂路口左转 / 静止后起步转向的条件建模和采样权重”。
- Stage5 / Stage6 的 tuned hard-weight 路线在六场景 eval 上已经把平均预测误差压到较低水平，但它们使用 hard-case weighting / tuned 初始化，不适合作为论文里的 `baseline-basic`。
- 论文基线主线随后切换为干净的 `baseline-basic`：B2D Full scenario-balanced 全量训练、`spatial_path` target、`99x10x2` anchor、`command_one_hot(6)+speed(1)` status、sample manifest 缓存、无 hard-case weighting、无 hard-case oversampling、无持续学习方法。

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

这一路线工程上能减少 target 构造带来的 CPU 压力，但当前 Stage4 中途验证 `validation step=9000 trajectory_unweighted=2.1736`，弱于 Stage3 在 step 3000 的 `1.3921`。因此 Stage4 暂时只作为“缓存优化已接入长训命令”的工程记录，不应直接当作效果最佳 checkpoint。hard-case oversampling 仍可作为后续改进方向，但当前论文 baseline 主线先转为 full `baseline-basic`，避免把 tuned hard-case 策略混入基础 baseline。

## Baseline-Basic Full Training: Completed

全量 `baseline-basic` 已在远端使用 4 卡 L40 / DDP 完成训练，用于论文 baseline 和后续持续学习方法的初始化基础。这个实验刻意不使用 Stage5 / Stage6 tuned checkpoint，也不使用 hard-case weighting。

数据缓存：

- Train manifest：`/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl`
- 构建参数：`--root-dir /share/home/u19666033/djy/carla_dataset --route-glob "*/*" --frame-sampling 5 --balanced-scenarios`
- 未使用 `--max-samples-per-scenario`，因此是 scenario-balanced discovery 下的全量 manifest
- Val manifest：`/share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl`

训练输出：

- Logdir：`/share/home/u19666033/ltr/dd_logs/full_baseline_basic`
- Run id：`origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5`

核心参数：

```text
epochs=100
batch_size=64 per GPU
global_batch_size=256
lr=6e-4
weight_decay=1e-4
optimizer=AdamW
scheduler=cosine
min_lr=1e-6
warmup_steps=3 * steps_per_epoch
image_encoder_lr_mult=0.5
grad_clip_norm=0
num_workers=4
prefetch_factor=2
hard_left_turn_stop_loss_weight=1.0
load_file=""
hardware=4x L40
distributed=torchrun/DDP
```

与原版 DiffusionDrive 对齐情况：

- 已对齐：`max_epochs=100`、per-GPU `batch_size=64`、`AdamW`、`lr=6e-4`、`weight_decay=1e-4`、`min_lr=1e-6`、`warmup_epochs=3` 的等价 step warmup、`image_encoder` 使用 `0.5x` 学习率。
- 已启用：`torchrun` / DDP 4 卡训练。DDP 下 `--batch-size` 是 per-GPU batch，global batch 为 `batch_size * WORLD_SIZE`，本 run global batch 为 `256`。
- 尚未对齐：AMP / `16-mixed`、原版 Lightning 训练框架。

## Post-Training Evaluation Status

baseline-basic 训练完成后，已完成 full-scenario 开环诊断和 Bench2Drive 220 sensor-only 闭环评测。结果已下载到本地镜像路径：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/full_baseline_basic/origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5/
```

本地汇总文件：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_open_loop_summary.csv
/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_closed_loop_summary.csv
/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/aggregate_summary.csv
```

开环结果：

| Scope | Scenes | Samples | L1 mean | L1 median | L1 p95 | L1 p99 | L1 > 2 | ADE mean | FDE mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy six-scene | 6 | 6144 | 0.0092 | 0.0023 | 0.0176 | 0.0305 | 4 | 0.0152 | 0.0234 |
| all scenarios | 39 | 39844 | 0.0192 | 0.0028 | 0.0188 | 0.0391 | 74 | 0.0303 | 0.0483 |

闭环结果：

| Metric | Value |
|---|---:|
| Avg. driving score | 44.8074 |
| Avg. route completion | 79.4774 |
| Avg. infraction penalty | 0.5334 |
| Avg. normalized DS | 35.5193 |
| Avg. speed km/h | 4.6448 |

闭环 status 分布：

| Status | Count |
|---|---:|
| Completed | 118 |
| Perfect | 1 |
| Failed - Agent deviated from the route | 69 |
| Failed - Agent got blocked | 27 |
| Failed - Agent timed out | 5 |

当前判断：

1. baseline-basic 是有效的论文基础 baseline：训练配置干净，不混入 Stage5 / Stage6 hard-weight tuning 或持续学习方法。
2. full baseline-basic 的开环轨迹误差显著低于 Stage5 / Stage6 的旧六场景 tuned checkpoint，但闭环只达到弱 baseline 水平。
3. 开环提升没有干净迁移到闭环，主要失败模式是 route deviation、vehicle blockage、低速和 collisions。
4. 闭环评测使用 sensor-only 协议，默认 `STOP_CONTROL=0`；如果显式开启 `STOP_CONTROL=1`，结果必须标注为 privileged stop-sign ablation。
5. 部分 route 曾在前几秒因 `filterpy` UKF covariance 非正定触发 `numpy.linalg.LinAlgError`，现已在 `DiffusionDriveAgent` 加入 `P` 对称化 / jitter / measurement reset 保护；后续遇到 `[DiffusionDriveUKF] reset` 日志时应按定位滤波重置记录，不再视作模型预测失败。
6. 闭环结果 skip 逻辑不能只看 `status=Failed`。正常完成但驾驶失败的 route 也可能是 `Failed`，应以 `_checkpoint.progress`、records 是否存在、以及是否属于 setup/crash 类状态判断。

## baseline-basic 20-route closed-loop ablations

为定位 full baseline-basic 的闭环失败原因，已在一组约 20 条代表 route 上做小规模闭环 A/B。结果已同步到本地：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/full_baseline_basic/ablation_20routes/
/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_ablation_20routes_summary.csv
/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_ablation_20routes_creep_summary.csv
```

有效实验摘要：

| Experiment | Main setting | DS | RC | NDS | Completed / Perfect | Deviated | Blocked | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `A0_baseline` | default sensor-only baseline | 37.6321 | 76.1615 | 25.8009 | 6 / 0 | 6 | 7 | 1 |
| `A5_stuck120_real` | `DIFFUSIONDRIVE_STUCK_THRESHOLD=120` | 39.0805 | 77.8525 | 28.5186 | 6 / 0 | 4 | 10 | 0 |
| `A6_stuck170_real` | `DIFFUSIONDRIVE_STUCK_THRESHOLD=170` | 39.6621 | 75.2455 | 28.8684 | 6 / 0 | 9 | 5 | 0 |
| `A7_stuck300_real` | `DIFFUSIONDRIVE_STUCK_THRESHOLD=300` | 40.3743 | 75.9430 | 29.2390 | 6 / 0 | 8 | 4 | 2 |
| `A8_pid_6_2p5` | `SPATIAL_PID_SPEED_FAST=6.0`, `SPATIAL_PID_SPEED_SLOW=2.5` | 42.9973 | 78.6635 | 34.0060 | 9 / 0 | 5 | 5 | 1 |
| `A9_pid_7_3` | `SPATIAL_PID_SPEED_FAST=7.0`, `SPATIAL_PID_SPEED_SLOW=3.0` | 46.0220 | 77.2035 | 35.7785 | 7 / 1 | 4 | 6 | 2 |

当前判断：

1. 空间 PID 速度参数是目前最明显的正向调参方向。`A8_pid_6_2p5` 更均衡，`A9_pid_7_3` 的 DS/NDS 最高但 vehicle collision 和 timeout 风险更高。
2. 单独调 stuck threshold 不够稳定：`120` 增加 blocked，`170` 将 blocked 交换成 deviation，`300` 引入 timeout。
3. 早期 `A5_stuck120` / `A6_stuck170` / `A7_stuck300` 是在 `DIFFUSIONDRIVE_STUCK_THRESHOLD` 尚未被 agent 读取时跑出的，不能作为 stuck-threshold 结论；只有 `_real` 后缀的重跑结果有效。
4. route `167`、`185`、`102` 在 A8/A9 下提升明显；`212`、`194`、`203`、`101` 是高速度 PID 下的主要退化 route，需要继续用 debug log 排查 route deviation / safety box / interaction。

### Creep / safety-box 观测

Bench2Drive leaderboard 没有独立的 `creep` 指标。creep 只能通过两类信号观察：

- 间接指标：`MinSpeedTest`、`AgentBlockedTest`、timeout status 和 route completion。
- 直接日志：`Detected agent being stuck` 表示 forced creep 真正给出最小 throttle；`Creeping stopped by safety box` 表示 force-move 已触发但前方 LiDAR safety box 非空，因此被强制停下。

当前 20-route 日志计数：

| Experiment | Routes with creep/safety signal | Forced creep ticks | Safety-box stop ticks |
|---|---:|---:|---:|
| `A0_baseline` | 8 | 795 | 21273 |
| `A5_stuck120_real` | 12 | 5546 | 34588 |
| `A6_stuck170_real` | 9 | 1202 | 24837 |
| `A7_stuck300_real` | 10 | 1591 | 29213 |
| `A8_pid_6_2p5` | 8 | 1343 | 16186 |
| `A9_pid_7_3` | 8 | 4623 | 16326 |

这说明 creeping 在上述实验中确实有体现，但不应只从 aggregate DS/RC 推断。A8/A9 明显减少了 safety-box stop loops，但 A9 在个别 route 上有更多实际 forced creep ticks；后续如果继续调闭环，应把 creep/safety-box 触发次数作为独立 debug 指标记录。

### Sensor / LiDAR 20-route diagnostics

后续又在同一类 20-route 诊断集上补了 sensor / LiDAR ablation。结果位于：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/full_baseline_basic/ablation_20routes_sensor/
```

关键 zero-LiDAR 结果：

| Experiment | Main setting | DS | RC | Infraction penalty | Blocked | Deviated | Timeout | Forced creep ticks | Safety-box stop ticks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Z0_zero_lidar_safety_debug` | zero model LiDAR, baseline-like control | 46.7179 | 78.7435 | 0.5754 | 3 | 8 | 0 | 72 | 9733 |
| `Z1_A9_pid_7_3_zero_lidar_safety_debug` | zero model LiDAR + A9 PID | 53.6419 | 80.9960 | 0.6049 | 3 | 6 | 1 | 0 | 10937 |

解释边界：

- `DIFFUSIONDRIVE_ZERO_LIDAR=1` 只置零模型侧 `lidar_feature`，不关闭 raw LiDAR safety-box。
- 因此 Z0/Z1 更好，优先说明当前模型侧 LiDAR BEV 分支可能是 domain-gap 噪声或监督不足，而不是说明 safety-box 应该关闭。
- 当前 DiffusionDrive LiDAR backbone 是 NAVSIM-style `resnet34` BEV encoder，不是 syb 的 `regnety_032`。若要对齐 syb，应作为 full retrain ablation。
- 项目要求使用 LiDAR，因此 zero-LiDAR / no-LiDAR 只能作为诊断和上界，不应作为最终主线。下一步 LiDAR 主线优先级应调整为：先验证训练 / 推理 LiDAR BEV contract，再做 channel ablation、LiDAR fusion gate / dropout，以及 `agent_states / agent_labels / bev_semantic_map` auxiliary supervision。

### Condition-v1 LiDAR diagnostic

2026-07-14 已拉取 `baseline-condition-v1` 上的 L0 / L1 / C2 诊断结果：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/full_baseline_condition_v1/ablation_20routes_lidar/
  L0_CV1_A8_zero_model_lidar_fallback/
  L1_CV1_A9_zero_model_lidar_fallback/
  C2_A9_pid_7_3_fallback/
```

运行条件：

- checkpoint：`full_baseline_condition_v1/soft_clean_skip25_imgnet_ddp4_bs64x4_lr6e-4_ep100/latest.pth`
- route token：on
- speed-head longitudinal controller：off
- PID：L0 使用 A8 spatial PID，`speed_fast=6.0`、`speed_slow=2.5`；L1 使用 A9 spatial PID，`speed_fast=7.0`、`speed_slow=3.0`
- LiDAR：L0 / L1 使用 `DIFFUSIONDRIVE_ZERO_LIDAR=1`；C2 使用 `DIFFUSIONDRIVE_ZERO_LIDAR=0`。三者都保留 raw LiDAR safety-box

聚合结果：

| Experiment | Valid routes | DS | RC | Infraction penalty | NDS | Completed | Deviated | Blocked | Stuck logs | Safety-box stop logs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0_A8_pid_6_2p5_fallback` common19 | 19 | 39.0702 | 86.3089 | 0.4334 | n/a | 13 | 0 | 6 | n/a | n/a |
| `C0_A8_pid_6_2p5_fallback` full20 | 20 | 41.8398 | 86.9935 | 0.4590 | 27.4672 | 14 | 0 | 6 | 3158 | 12179 |
| `L0_CV1_A8_zero_model_lidar_fallback` | 19 | 43.5831 | 82.1032 | 0.5095 | 30.2713 | 8 | 6 | 5 | 2712 | 10845 |
| `L1_CV1_A9_zero_model_lidar_fallback` common19 | 19 | 48.5136 | 88.3847 | 0.5239 | n/a | 11 | 5 | 3 | n/a | n/a |
| `L1_CV1_A9_zero_model_lidar_fallback` full20 | 20 | 50.7682 | 88.9655 | 0.5445 | 34.9458 | 12 | 5 | 3 | 783 | 8132 |
| `C2_A9_pid_7_3_fallback` full20 | 20 | 40.9991 | 94.4010 | 0.4224 | 19.5552 | 16 | 0 | 4 | 2834 | 10268 |

L0 的完整 20-route 中 route `50` 没有生成 result JSON；`bench2drive_50_attempt2_eval.log` 显示 `SensorReceivedNoData: A sensor took too long to send their data`，应视为基础设施 / 传感器超时，不能计入模型成绩。L1 / C2 均有 20/20 有效 route。C2 每条 route 只有一次有效 attempt，没有 setup / agent / simulation crash；日志确认与 L1 使用相同 checkpoint、ImageNet normalization、JPEG artifact、route token、A9 PID 和严格 `769/769` tensor 加载，模型侧 LiDAR ON/OFF 是有效主变量。

结论：

- L0 延续了 Z0/Z1 的方向：zero model-LiDAR 后低速 / stuck / safety-box stop 有缓和，`DS` 和 infraction penalty 改善。
- 但 L0 也引入了明显 route deviation：共同 19 条 route 中从 C0 的 `0` 条 route-deviated 变为 L0 的 `6` 条。
- C0 -> C2 在 LiDAR-on 条件下只改变 A8 -> A9 PID：`DS -0.84`、`RC +7.41`、`IP -0.0366`、`NDS -7.91`、平均速度 `+1.09 km/h`，completed 从 `14` 增到 `16`、blocked 从 `6` 降到 `4`。因此 A9 的主要收益是推进速度和 route completion，不是处罚质量。
- C2 -> L1 在同一 A9 PID 下只把模型 LiDAR 置零：`DS +9.77`、`RC -5.44`、`IP +0.1221`、`NDS +15.39`、平均速度 `+1.04 km/h`。zero model-LiDAR 降低 completion / 增加终止性 route deviation，但显著改善碰撞、越线、低速和归一化处罚。
- C2 相对 L1 的 raw event 为 vehicle collision `20 vs 17`、layout collision `5 vs 4`、outside-lane `10 vs 6`、min-speed `44 vs 35`、vehicle-blocked `5 vs 3`；stuck / safety-stop 也从 `783/8132` 升到 `2834/10268`。
- 逐 route 配对中 C2 对 L1 为 `8` 胜 `12` 负，DS 中位差只有 `-0.93`，但 route `50/94/153/167/185` 出现 `-37.26/-38.22/-67.89/-44.33/-32.21` 的灾难性分叉；route `139` 则反向提升 `+46.23`。稳定 route `00/04/24/212` 基本一致，说明 LiDAR 分支不是被模型忽略，而是在少数交互场景中产生强且不稳定的影响。
- 当前结论不是“LiDAR 一律有害”：LiDAR-on 将 failed route-deviation 从 `5` 降到 `0`，并把 completed 从 `12` 提到 `16`；但它显著恶化局部安全 / 处罚质量。项目要求保留 LiDAR，下一步应先用已实现的 train-vs-online contract dump 和 `original/zero/shuffle` 开环归因定位分布 / fusion 问题，再做 channel ablation、fusion gate / dropout 和 auxiliary supervision。

### Paired open-loop and online LiDAR contract diagnostics (2026-07-15)

已拉取并分析：

```text
/share/home/u19666033/ltr/dd_logs/lidar_diagnostics/b2d_train_4096
/share/home/u19666033/ltr/dd_logs/lidar_diagnostics/c2_online_contract
/share/home/u19666033/ltr/dd_logs/lidar_diagnostics/openloop_ablation
```

`original/zero/shuffle` 各有 `39,432` 个严格配对样本。结果为：

| LiDAR mode | Mean L1 | Mean ADE | Mean FDE |
|---|---:|---:|---:|
| `original` | 0.02047 | 0.03211 | 0.05227 |
| `zero` | 0.31397 | 0.50525 | 0.86122 |
| `shuffle` | 0.58965 | 0.95248 | 1.64013 |

`original` 在 `91.50%` 的样本上优于 `zero`，在 `95.91%` 上优于 `shuffle`；39 个 scenario 的 original-zero 差值全部为负。由此确认模型在 B2D raw 分布上强依赖且正确使用 LiDAR，不能把 C2 闭环差异解释为“LiDAR 没学到”或“LiDAR 全局有害”。

在线 angle contract 没发现 gross yaw / half-scan 缺失：B2D full scan 和 online 拼接 full scan 的 1-degree angular coverage median 都是 `100%`。但 online 前向区域主要来自上一 tick，当前 half 主要覆盖后向区域；这会让移动 actor 的前向点云天然晚一 tick。场景匹配统计还显示 density/occupancy drift 高度场景相关：route 153 在失败前 model points / occupancy / BEV mean 分别约为训练分布的 `1.80x/3.07x/2.49x`，但 route 139 在 `3.29x/1.22x/1.75x` 下仍以 DS 96.02 完成，route 50 几乎匹配仍发生碰撞。contract drift 是因素，但不是单调充分解释。

另一个独立问题是 spatial target / prediction 的时序跳变。在 `37,101` 个相邻 transition 中，target endpoint jump `>5m` 占 `3.43%`，其中 `95.67%` 同时触发 original prediction jump `>5m`；`170` 个大跳变发生在低速且 route condition 基本不变时。当前 target 路径不足会沿最后有效线段外推，长时间静止时微小位移方向可能翻转。与此同时，`TrajectoryHead.forward_test()` 每次 forward 使用新的随机初始噪声，可能引入 mode switching。

因此下一步顺序修正为：先实现固定/zero/多 seed diffusion noise 与 mode/margin 日志，确认随机推理对灾难分叉的贡献；并行修复 spatial target 连续性；随后再做 online half-scan temporal/channel ablation。fusion gate/dropout 和 auxiliary supervision 放在上述归因稳定之后。完整记录见 `docs/diffusiondrive_lidar_diagnostics_20260715.md`。

同日已完成确定性推理入口：模型支持 `random/fixed/zero/seeded` noise，开环 CSV 输出 mode/top-2/margin/entropy/endpoint，在线 agent 写独立 trajectory JSONL 并在比较 endpoint 前做 ego-frame 对齐。本地 fixed/seeded mini 重复推理均得到 `max_abs_diff=0.0`；远端 fixed seed 0-4 与 route 24/50/139/153 多 attempt 尚待运行。

### Fixed 5-seed 与 spatial target v2（2026-07-15）

远端 fixed seed `0...4` all-scenarios 开环已完成，每组均为 `39,432` 个严格配对样本。五组 mean L1 平均 `0.01936`，范围 `0.01875...0.02106`；历史 random run 为 `0.02047`。跨 seed endpoint 最大两两距离 p95/p99 仅 `0.052m/0.099m`。`L1>1m` 并集 `127`、交集 `29`，其中 `123/127` 位于存在 target jump `>12m` 的 route，`93/127` 位于 jump 前后两个采样间隔内。结论是 diffusion noise 会改变部分尾部归属，但旧 target 不连续是主要来源。

随后对高频异常 route `MergerIntoSlowTrafficV2/Town12_Rep0_968_25...` 下钻原始 annotation。旧实现会把窗口末端约 `1.5mm` 的反向 pose 量化抖动当成最后有效 segment，再外推到 `11.5m`；窗口滑动时方向翻转。已实现 `arc_length_stable_extrapolation_v2`：外推优先使用至少 `0.5m` baseline 的 trailing displacement，不足时使用 route condition；已观测路径内的弧长插值不变。该 route 的 8 个旧 `>12m` jump 重算后全部降为 `0.002m...0.058m`。

跨场景抽取旧 `>12m` 统计前五条 route 后，v2 将 `77/77` 个事件全部降到 `<12m`，新 jump p95 为 `0.083m`。仅 2 个仍 `>5m`，均伴随 `>85m` route-condition delta 和明显速度/command 阶段变化，不属于停驻抖动。该结果覆盖全量 119 个旧大跳变中的 77 个，支持保留 `0.5m` 稳定 baseline。

同时新增 `tools/inspect_diffusiondrive_spatial_target_continuity.py`，输出全量 transition、旧缓存与当前 contract 的重算对比，以及 future world pose / ego-frame polyline / 外推来源 trace。manifest header 新增 `spatial_target` schema，旧缓存会被训练 loader 明确拒绝。当前尚未完成 full soft-clean v2 manifest 重建和全局 jump 复核；完成后才能启动基于 v2 target 的正式 full retrain。route 24/50/139/153 多 attempt 闭环也仍待运行。
