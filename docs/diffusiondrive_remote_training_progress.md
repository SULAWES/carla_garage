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
- Stage5 / Stage6 的 tuned hard-weight 路线在六场景 eval 上已经把平均预测误差压到较低水平，但它们使用 hard-case weighting / tuned 初始化，不适合作为论文里的 `baseline-basic`。
- 当前论文基线主线已切换为干净的 `baseline-basic`：B2D Full scenario-balanced 全量训练、`spatial_path` target、`99x10x2` anchor、`command_one_hot(6)+speed(1)` status、sample manifest 缓存、无 hard-case weighting、无 hard-case oversampling、无持续学习方法。

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

## Baseline-Basic Full Training: Running

当前正在远端挂全量 `baseline-basic` 训练，用于论文 baseline 和后续持续学习方法的初始化基础。这个实验刻意不使用 Stage5 / Stage6 tuned checkpoint，也不使用 hard-case weighting。

数据缓存：

- Train manifest：`/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl`
- 构建参数：`--root-dir /share/home/u19666033/djy/carla_dataset --route-glob "*/*" --frame-sampling 5 --balanced-scenarios`
- 未使用 `--max-samples-per-scenario`，因此是 scenario-balanced discovery 下的全量 manifest
- Val manifest：`/share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl`

训练输出：

- Logdir：`/share/home/u19666033/ltr/dd_logs/full_baseline_basic`
- Run id：`origlike_bs64_lr6e-4_ep100_fs5_spatial_imgenc0p5`

核心参数：

```text
epochs=100
batch_size=64
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
```

与原版 DiffusionDrive 对齐情况：

- 已对齐：`max_epochs=100`、`batch_size=64`、`AdamW`、`lr=6e-4`、`weight_decay=1e-4`、`min_lr=1e-6`、`warmup_epochs=3` 的等价 step warmup、`image_encoder` 使用 `0.5x` 学习率。
- 尚未对齐：AMP / `16-mixed`、DDP、多卡 global batch、原版 Lightning 训练框架。

当前应重点观察：

- warmup 前 3 个 epoch 内 loss 是否平稳下降，不要过早按前几个 step 判断。
- `lr` 与 `image_encoder_lr` 是否分别打印为约 `6e-4` 与 `3e-4`。
- `trajectory_unweighted` 与 validation loss 在 warmup 后是否继续下降。
- GPU 利用率是否仍周期性掉低，CPU / 内存是否出现单调增长到 OOM。
- 若 `lr=6e-4` 出现明显 loss 爆炸或 NaN，下一版保留 `epochs=100` 和 `image_encoder_lr_mult=0.5`，优先回退到 `lr=1e-4` 做保守 baseline。

## 后续建议

当前先等待 full baseline-basic 的训练曲线和 checkpoint。拿到结果后按同一六场景 eval CSV 汇总平均 `l1 / ade / fde`，再决定：

1. 是否保留原版对齐版作为论文 baseline-basic。
2. 是否需要再跑一个 `lr=1e-4` 的保守 baseline-basic 对照。
3. Stage5 / Stage6 tuned hard-weight 路线只作为改进 / ablation 参考，不和 baseline-basic 混用。
4. 进入 CARLA 闭环前，确认使用 sensor-only 设置，尤其 `STOP_CONTROL=0`；如果显式开启 `STOP_CONTROL=1`，结果必须标注为 privileged stop-sign ablation。
