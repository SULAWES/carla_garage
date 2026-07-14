# DiffusionDrive CARLA Training

本文档记录当前第一版 Bench2Drive Full / CARLA-native DiffusionDrive 训练入口。

## 当前入口

- 训练脚本：`team_code/train_diffusiondrive.py`
- 数据集 helper：`team_code/diffusiondrive/carla_native_dataset.py`
- 支持数据格式：原始 Bench2Drive / Bench2Drive Full route 目录
  - `camera/rgb_front/*.jpg`
  - `lidar/*.laz`
  - `anno/*.json.gz`
  - 或 B2D Full 原生 `rgb/*.jpg`、`lidar/*.laz`、`measurements/*.json.gz`

当前训练和后续主要修改以 B2D Full raw 数据为主训练分布。sensor / time / preprocessing contract 见 `b2d_full_sensor_contract.md`。

当前训练 trajectory head 主路径，并新增 `SpeedHead-v1` 纵向速度 / 刹车监督。模型输出的 `trajectory_loss` 是 trajectory head 内部的未加外层权重损失；训练脚本会再乘 `DiffusionDriveConfig.trajectory_weight`。`SpeedHead-v1` 使用 `target_speed_twohot` 做 masked soft cross entropy，再乘 `DiffusionDriveConfig.speed_loss_weight`。当前仍不接入 `agent_states / agent_labels / bev_semantic_map` 辅助 loss。

当前 LiDAR 分支继承的是 NAVSIM-style 单帧 BEV histogram + CNN encoder：`lidar/*.laz` 被栅格化成 `256x256` BEV histogram，再送入 `DiffusionDriveConfig.lidar_architecture` 指定的 `timm` backbone。CARLA DiffusionDrive 默认仍是 `resnet34`，不是 syb / garage 旧模型常用的 `regnety_032`。`regnety_032` 只是另一个 2D CNN backbone，切换它会改变大量权重 shape，应作为新的 full retrain ablation，而不是当前 checkpoint 的部署开关。

训练入口会在输出目录写入：

- `training_config.json`：CLI 参数、DiffusionDrive config、数据 split、B2D Full sensor contract、时间语义、anchor shape、预处理和 status feature schema
- `checkpoint_epoch*_step*.pth`：周期或结束 checkpoint
- `latest.pth`：最近一次保存的 checkpoint，供 `--resume-file` 使用

可配置项已经覆盖第一阶段 full training 需要的基础参数：

- optimizer：`--optimizer adamw`、`--lr`、`--weight-decay`、`--image-encoder-lr-mult`
- scheduler：`--scheduler none|cosine|multistep`、`--warmup-steps`、`--lr-steps`、`--lr-gamma`、`--min-lr`；其中 `--lr-steps` 是按 optimizer step 计数的逗号分隔 milestone
- loss：`--trajectory-weight`、`--speed-loss-weight`、`--trajectory-cls-weight`、`--trajectory-reg-weight`、`--trajectory-focal-alpha`、`--trajectory-focal-gamma`
- diffusion：`--diffusion-num-train-timesteps`、`--diffusion-train-timestep-min/max`、`--diffusion-infer-step-num`、`--diffusion-infer-timestep-span`、`--diffusion-infer-trunc-timesteps`
- data split：`--val-root-dir`、`--val-route-glob`、`--val-frame-sampling`、`--val-max-samples`、`--val-every-steps`、`--max-val-steps`
- dataset / target / time contract：`--dataset-mode`、`--target-mode`、`--spatial-target-first-distance`、`--spatial-target-interval`、`--spatial-target-max-future-frames`、`--assumed-frame-interval`、`--future-stride`、`--skip-first-frames`、`--b2d-source-image-height`、`--b2d-source-image-width`
- scenario balancing：`--balanced-scenarios`、`--max-samples-per-scenario`
- distributed：`--distributed auto|none|ddp`，默认 `auto`，用 `torchrun` 启动且 `WORLD_SIZE>1` 时自动启用 DDP
- hard-case weighting：`--hard-left-turn-stop-loss-weight`、`--hard-left-turn-command`、`--hard-left-turn-speed-threshold`、`--hard-left-turn-y-threshold`
- sample distribution stats：`--dataset-stats-max-samples` 会落盘 `train_sample_distribution.json` / `validation_sample_distribution.json` / `eval_sample_distribution.json`
- dataloader / manifest：`--sample-manifest`、`--val-sample-manifest`、`--rebuild-sample-manifest`、`--prefetch-factor`、`--persistent-workers`；CPU-only manifest builder 支持 `--quality-filter none|soft_clean|syb_clean`
- preprocessing：`--model-image-height`、`--model-image-width`、`--image-normalization none|imagenet`、`--no-jpeg-artifact`
- evaluation：`--eval-only` 会只加载模型并跑评估，不进入训练循环；训练 checkpoint 用 `--resume-file` 严格加载 `model`，普通权重 / NAVSIM checkpoint 可用 `--load-file` 部分加载；闭环 agent 默认拒绝部分加载或 preprocessing metadata mismatch

## Optimizer Param Groups

训练入口支持按原版 DiffusionDrive 的 optimizer 习惯给 image encoder 使用较小学习率：

```bash
--image-encoder-lr-mult 0.5
```

参数名包含 `image_encoder` 的参数会进入单独 param group，学习率为 `--lr * --image-encoder-lr-mult`；其余参数使用 `--lr`。默认值是 `1.0`，表示保持旧行为。原版对齐的 baseline-basic 建议使用 `--lr 6e-4 --weight-decay 1e-4 --image-encoder-lr-mult 0.5`；训练日志会同时打印 `lr` 和 `image_encoder_lr`，`training_config.json` 也会记录这组 optimizer 设置。

## Multi-GPU DDP

训练入口支持 PyTorch DDP。单卡命令保持不变；多卡用 `torchrun` 启动，脚本会从 `LOCAL_RANK / RANK / WORLD_SIZE` 自动设置设备、初始化 process group，并给训练集使用 `DistributedSampler`：

```bash
torchrun --standalone --nproc_per_node=4 team_code/train_diffusiondrive.py ...
```

DDP 下 `--batch-size` 是每张 GPU / 每个进程的 batch size，实际 global batch size 为 `batch_size * WORLD_SIZE`。scheduler、`--warmup-steps`、`--val-every-steps`、`--save-every-steps` 都按 optimizer step 计数；由于 DDP 每个 epoch 的 optimizer step 数约为 `ceil(samples / global_batch_size)`，计算 warmup 和保存间隔时要用 global batch size。rank0 负责写 `training_config.json`、sample distribution、validation 和 checkpoint；checkpoint 保存的是未包 DDP 的普通 model state dict，后续单卡或多卡都可以 resume。当前 DDP 使用 `find_unused_parameters=True`，并在 DDP 模式下对 auxiliary outputs 加 `0.0 * output.sum()` dummy term，使未监督 heads 产生零梯度；这不改变数值 loss，只是避免未监督 auxiliary branches 在 DDP 下触发 unused-branch 问题。

## Completed Baseline-Basic Run

full `baseline-basic` 已在远端使用 4 卡 L40 / DDP 完成训练，目标是作为论文 baseline 和后续持续学习方法的干净起点。该实验使用 B2D Full scenario-balanced 全量 train manifest，不使用 Stage5 / Stage6 tuned checkpoint，不启用 hard-case weighting：

```text
logdir=/share/home/u19666033/ltr/dd_logs/full_baseline_basic
id=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
train_manifest=/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl
val_manifest=/share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl
epochs=100
batch_size=64 per GPU
global_batch_size=256
lr=6e-4
weight_decay=1e-4
image_encoder_lr_mult=0.5
hard_left_turn_stop_loss_weight=1.0
load_file=""
```

这个 run 已尽量对齐原版 DiffusionDrive 的主要训练预算和优化器设置，并使用 `torchrun` / DDP 跑完；当前训练入口尚未对齐 AMP。训练结果、开环诊断和闭环进展记录在 `diffusiondrive_remote_training_progress.md`。

本地下载的结果镜像位于：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/full_baseline_basic/origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5/
```

结果摘要：

- Open-loop legacy six-scene：`l1_mean=0.0092`、`ade_mean=0.0152`、`fde_mean=0.0234`
- Open-loop all-scenarios：`l1_mean=0.0192`、`ade_mean=0.0303`、`fde_mean=0.0483`
- Closed-loop Bench2Drive 220：`DS=44.8074`、`RC=79.4774`、`NDS=35.5193`
- Closed-loop status：`Completed=118`、`Perfect=1`、`deviated=69`、`blocked=27`、`timed out=5`

这组结果说明 full baseline-basic 在开环轨迹误差上很强，但闭环仍是弱 baseline，后续持续学习或控制改进需要以闭环 failure modes 为主要诊断对象。

### 20-route closed-loop ablations

baseline-basic 后续已完成一组 20-route 闭环 A/B，用于初步定位控制侧 failure modes。结果归档在：

```text
/home/HeavenlySU/sitp_workspace/dd_logs/full_baseline_basic/ablation_20routes/
/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_ablation_20routes_summary.csv
/home/HeavenlySU/sitp_workspace/dd_logs/eval_summaries/baseline_basic_ablation_20routes_creep_summary.csv
```

当前结论：

- 空间 PID 速度参数是最明显的正向调参方向。`A8_pid_6_2p5` 更均衡：`DS=42.9973`、`RC=78.6635`、`NDS=34.0060`；`A9_pid_7_3` 分数最高：`DS=46.0220`、`RC=77.2035`、`NDS=35.7785`，但 collision 和 timeout 风险更高。
- 单独调 stuck threshold 不够稳定：`A5_stuck120_real` blocked 增加，`A6_stuck170_real` 将 blocked 换成 route deviation，`A7_stuck300_real` 引入 timeout。
- 早期 `A5_stuck120` / `A6_stuck170` / `A7_stuck300` 不是有效 stuck-threshold A/B，因为当时 agent 尚未读取 `DIFFUSIONDRIVE_STUCK_THRESHOLD`；只使用 `_real` 后缀结果做 stuck 结论。
- creeping 不在 leaderboard 中作为独立指标出现，需要结合日志中 `Detected agent being stuck` / `Creeping stopped by safety box` 与 `MinSpeedTest`、`AgentBlockedTest`、timeout 一起分析。A8/A9 减少 safety-box stop loops，但 A9 在个别 route 上实际 forced creep ticks 更多。

## Status Feature 与 Extra Sensors

当前 DiffusionDrive 训练入口使用 `7` 维 `status_feature`：

```text
command_one_hot(6) + speed(1)
```

训练侧当前由 `carla_native_dataset.build_status_feature()` 构造；推理侧由 `DiffusionDriveAgent._build_status()` 构造。两者底层共用 `diffusiondrive.status` 中的 status builder，避免训练和推理 schema 分叉。

需要和 garage / syb 旧模型里的 `extra_sensors` 区分：

- `extra_sensors` 是旧 `team_code/model.py` 的可选分支，不是 DiffusionDrive 当前接口。
- syb 仓库里重点关注的 `extra_sensors` 本质是 `speed(1) + command_one_hot(6)`，再经 `extra_sensor_encoder` 投影成 decoder token。
- DiffusionDrive 已经有 `status_feature -> _status_encoding` 这个低维条件 token 入口，因此不建议再额外引入一套独立 `extra_sensors` 分支。

当前迁移结果：

```text
status_feature = command_one_hot(6) + speed(1)
```

也就是把 syb / garage `extra_sensors` 的核心设计吸收到 DiffusionDrive 的 `status_feature` 中。迁移后 `_status_encoding` 从 `Linear(10, 256)` 变成 `Linear(7, 256)`，旧 checkpoint 中这一层按 shape mismatch 跳过即可。

`baseline-condition-v1` 已不恢复旧 `extra_sensors`，也不把 route 目标点直接拼成 `11` 维 flat status。当前模型接口使用两个 condition token：

```text
status_token = command_one_hot(6) + speed(1)
route_condition_token = target_point(2) + target_point_next(2)
```

`status_token` 继续复用当前 status builder；`route_condition_token` 由训练侧从 measurements / manifest 读取并缓存 `target_point`、`target_point_next`，推理侧复用 `DiffusionDriveAgent.tick()` 计算出的 ego-frame route target。该改动改变模型接口和 checkpoint 兼容性，应作为 full retrain 实验处理。`DIFFUSIONDRIVE_USE_ROUTE_CONDITION=0` 只会把 route token 的输入值置零做 ablation，不会移除模型里的 route token 或恢复旧 checkpoint 结构。

`SpeedHead-v1` 已接入：dataset / manifest 输出 `target_speed`、`brake`、`target_speed_twohot`、`target_speed_class`、`target_speed_label_valid`；模型新增独立 speed query 和 MLP head；训练记录 `target_speed_loss`、`target_speed_accuracy`、`target_speed_brake_accuracy`、`target_speed_l1`。推理侧新增 `DIFFUSIONDRIVE_USE_SPEED_HEAD=1` predicted-speed longitudinal controller，默认关闭，空间 PID desired speed 仍是 fallback。注意：2026-06-09 的旧 `C1_speedhead` 闭环初筛显示 direct controller 在 route 00 起步阶段持续预测 class 0 并全刹锁死，不能作为新版 speed-head 结论继续外推。2026-06-11 起，推理侧默认改为 syb-style uncertainty-weighted speed conversion：`DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT=1` 时仅当 class-0 概率超过 `DIFFUSIONDRIVE_SPEED_HEAD_BRAKE_THRESHOLD=0.9` 才强制目标速度为 0，否则使用 `sum(prob * target_speeds)`；`DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT=0` 切回 argmax 诊断。

## 本机 mini smoke

```bash
cd /home/HeavenlySU/sitp_workspace/carla_garage
conda run -n garage_2 python team_code/train_diffusiondrive.py \
  --root-dir Bench2Drive/Bench2Drive-mini-extracted \
  --logdir /tmp/dd_train_smoke \
  --id smoke \
  --epochs 1 \
  --batch-size 1 \
  --max-samples 2 \
  --max-steps 1 \
  --frame-sampling 20 \
  --num-workers 0 \
  --load-file ""
```

最近一次 smoke 结果：

- `Dataset samples: 2`
- `status_feature`: `{"schema": "command_one_hot(6)+speed(1)", "dim": 7, "normalized": false}`
- `data.dataset_mode`: `b2d_full_raw`
- `target.mode`: `spatial_path`
- `target.spatial_path`: `first=2.5m`, `interval=1.0m`, `last=11.5m`
- `anchor`: `shape=[99,10,2]`, `semantics=spatial route/checkpoint anchor`
- `epoch=0 step=1 loss=287.6801`
- `trajectory_unweighted=23.9733`
- `trajectory_loss_0=11.8513`
- `trajectory_loss_1=12.1220`
- checkpoint：`/tmp/dd_train_spatial_target_smoke/smoke/checkpoint_epoch000_step0000001.pth`

带验证集 smoke：

```bash
cd /home/HeavenlySU/sitp_workspace/carla_garage
conda run -n garage_2 python team_code/train_diffusiondrive.py \
  --root-dir Bench2Drive/Bench2Drive-mini-extracted \
  --val-root-dir Bench2Drive/Bench2Drive-mini-extracted \
  --logdir /tmp/dd_train_smoke_config \
  --id smoke_val \
  --epochs 1 \
  --batch-size 1 \
  --max-samples 2 \
  --max-steps 1 \
  --val-max-samples 1 \
  --max-val-steps 1 \
  --val-every-steps 1 \
  --frame-sampling 20 \
  --num-workers 0 \
  --load-file ""
```

最近一次验证输出：

- `validation step=1 loss=3079.8933`
- `trajectory_unweighted=256.6578`

resume 示例：

```bash
conda run -n garage_2 python team_code/train_diffusiondrive.py \
  --root-dir Bench2Drive/Bench2Drive-mini-extracted \
  --logdir /tmp/dd_train_smoke_config \
  --id smoke_resume \
  --epochs 2 \
  --batch-size 1 \
  --max-samples 2 \
  --max-steps 2 \
  --frame-sampling 20 \
  --num-workers 0 \
  --resume-file /tmp/dd_train_smoke_config/smoke_val/latest.pth
```

eval-only 示例：

```bash
conda run -n garage_2 python team_code/train_diffusiondrive.py \
  --root-dir Bench2Drive/Bench2Drive-mini-extracted \
  --val-root-dir Bench2Drive/Bench2Drive-mini-extracted \
  --logdir /tmp/dd_train_smoke_config \
  --id eval_only \
  --eval-only \
  --batch-size 1 \
  --val-max-samples 1 \
  --max-val-steps 1 \
  --frame-sampling 20 \
  --num-workers 0 \
  --resume-file /tmp/dd_train_smoke_config/smoke_val/latest.pth
```

`--eval-only` 的评估数据优先使用 `--val-root-dir`；未提供时会回退到 `--root-dir`。输出形如：

```text
eval loss=... trajectory_unweighted=... steps=... trajectory_loss_0=... trajectory_loss_1=...
```

## Full 数据集示例

远程 B2D Full 数据集只需要替换 `--root-dir` 和 `--logdir`。如果数据是 `scenario/route` 两层结构，例如 `carla_dataset/Accident/Town13_.../`，需要加 `--route-glob "*/*"`。默认 `--dataset-mode b2d_full_raw`、`--target-mode spatial_path`、`--assumed-frame-interval 0.1`，训练配置会记录这些假设：

```bash
conda run -n ltr_garage_2 python team_code/train_diffusiondrive.py \
  --root-dir /path/to/carla_dataset \
  --route-glob "*/*" \
  --logdir /path/to/logs \
  --id dd_carla_native_v1 \
  --epochs 20 \
  --batch-size 4 \
  --frame-sampling 5 \
  --num-workers 6 \
  --scheduler cosine \
  --warmup-steps 1000 \
  --save-every-steps 1000
```

如需从 NAVSIM checkpoint 初始化：

```bash
--load-file /home/HeavenlySU/sitp_workspace/diffusiondrive_navsim_88p1_PDMS
```

脚本会按 key 和 shape 部分加载 checkpoint；当前 `99x10x2` trajectory head 相关 mismatch 属于预期。

Full checkpoint eval-only 示例：

```bash
conda run -n ltr_garage_2 python team_code/train_diffusiondrive.py \
  --root-dir /share/home/u19666033/djy/carla_dataset \
  --val-root-dir /share/home/u19666033/djy/carla_dataset/Accident \
  --route-glob "*/*" \
  --val-route-glob "*" \
  --logdir ~/ltr/dd_logs/full_eval \
  --id accident_eval \
  --eval-only \
  --batch-size 16 \
  --val-frame-sampling 10 \
  --val-max-samples 512 \
  --max-val-steps 50 \
  --num-workers 6 \
  --device cuda:0 \
  --resume-file ~/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4/latest.pth
```

## Scenario-Balanced Training

远端 B2D Full 数据是 `scenario/route` 两层结构时，普通 `--max-samples` 会按排序后的 route 顺序截断，容易偏向前几个 scenario。`NonSignalizedJunctionLeftTurn` 的单场景 finetune probe 已显示该类场景主要是覆盖不足：stage1 eval `trajectory_unweighted=11.3806`，单场景 finetune 后验证降到约 `4.4306`。

为避免顺序截断偏置，训练入口支持：

- `--balanced-scenarios`：按 `route_dir.parent.name` 分桶，并 round-robin 合并各 scenario 样本
- `--max-samples-per-scenario`：每个 scenario 最多保留的样本数

quick balanced stage 示例：

```bash
conda run -n ltr_garage_2 python team_code/train_diffusiondrive.py \
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
  --load-file ""
```

训练日志会打印 `Dataset scenario samples`，`training_config.json` 会记录 `balanced_scenarios` 和 `max_samples_per_scenario`。

## Hard-Case Loss Weighting

针对 Stage2 诊断中仍然偏高的 `NonSignalizedJunctionLeftTurn` 静止 / 低速左转样本，训练入口支持对一类 hard case 做 trajectory loss 样本加权：

```bash
--hard-left-turn-stop-loss-weight 3.0 \
--hard-left-turn-command 1 \
--hard-left-turn-speed-threshold 0.1 \
--hard-left-turn-y-threshold 4.0
```

匹配条件是 `command == 1 && speed < 0.1 && abs(target_end_y) > 4.0`。权重只用于训练集；validation 和 eval-only 的 loss 保持未加权，便于和旧实验直接对比。训练日志会打印当前 batch 的 `hard_left_turn_stop` 数量和 `mean_sample_weight`，`training_config.json` 会记录 hard-case criteria。

启动时脚本会额外抽样统计 command、speed bin、`abs(target_end_y)` bin 和 hard-case 数量，并写入输出目录的 sample distribution JSON。默认最多统计 `4096` 个样本；可用 `--dataset-stats-max-samples 0` 关闭，或调大以覆盖完整数据集。

## Sample Manifest and CPU Bottlenecks

raw B2D Full loader 的主要 CPU / IO 压力来自反复解 `json.gz` 构造 `spatial_path` target、解 `.laz` 并生成 LiDAR histogram，以及图像解码 / resize。训练入口现在支持 sample manifest，先把每个样本的 route/frame、`status_feature` 所需 command/speed 和 trajectory target 落盘为 JSONL，后续训练可直接读取 manifest，避免每个 epoch 反复扫描未来 annotation：

```bash
--sample-manifest /share/home/u19666033/ltr/dd_cache/full_stage3_train_manifest.jsonl
```

manifest 当前缓存的是样本发现与 target 构造结果，格式版本为 `diffusiondrive_sample_manifest_v1`，每行包含绝对 `route_dir`、`frame`、`command`、`speed` 和 `trajectory`。它不缓存图像 tensor 或 LiDAR histogram，所以训练时仍会在线解码 `rgb/*.jpg` 和 `lidar/*.laz`；优化重点是避免每个 epoch 为了构造空间轨迹 target 重复读取大量未来 `measurements/*.json.gz`。

如果 manifest 不存在，脚本会在 dataset 初始化时创建；如果已存在，会先读取 header 并校验当前 dataset 参数。校验覆盖 `target-mode / num-poses / future-stride / spatial target 参数 / frame-sampling / balanced-scenarios / max-samples-per-scenario / route-glob / root-dir / sample_count` 等字段；不匹配会直接报错，避免 baseline 长训静默使用错误分布。manifest 与数据选择和 target 语义绑定，至少要按 `root-dir / route-glob / frame-sampling / target-mode / spatial target 参数 / balanced-scenarios / max-samples-per-scenario` 区分命名；这些参数变化后应换一个 manifest 文件或强制重建。需要强制重建时加：

```bash
--rebuild-sample-manifest
```

验证集或 eval-only 使用不同 root 时，应单独指定：

```bash
--val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_manifest.jsonl
```

DataLoader worker 会在初始化时限制 OpenCV / torch 内部线程，避免在 7 CPU 核限制下过度抢占。可用 `--prefetch-factor 2` 调整预取；`--persistent-workers` 默认关闭，建议只在远端长训确认稳定后开启。

在远端 1 GPU 最多 7 CPU 核的限制下，推荐组合是：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

python team_code/train_diffusiondrive.py \
  ... \
  --num-workers 5 \
  --prefetch-factor 2 \
  --sample-manifest /share/home/u19666033/ltr/dd_cache/full_train_fs5_spatial.jsonl \
  --val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl
```

可以把 `--num-workers` 试到 `6`，但不要超过作业实际申请的 CPU 核数。`--persistent-workers` 会减少 epoch 切换时重启 worker 的开销，但如果远端长训中遇到 dataloader 卡住或退出不干净，先保持默认关闭。

如果实验室空闲 CPU 核较多，可以先用 CPU-only 作业预构建 manifest，再启动 GPU 训练。这样 GPU 作业只读取已生成的 JSONL，不在训练开始时长时间占 GPU 等待 target 构建。专用脚本是：

```bash
python tools/build_diffusiondrive_manifest.py \
  --root-dir /share/home/u19666033/djy/carla_dataset \
  --route-glob "*/*" \
  --output-manifest /share/home/u19666033/ltr/dd_cache/full_condition_v1_train_soft_clean_fs5_skip25.jsonl \
  --frame-sampling 5 \
  --skip-first-frames 25 \
  --balanced-scenarios \
  --quality-filter soft_clean \
  --num-workers 32 \
  --rebuild \
  --verify-load
```

manifest builder 可选 route-level quality filter：

```bash
--quality-filter none       # 默认，保留所有可构造样本的 route
--quality-filter soft_clean # 去掉 missing results / FAILED_ / hard failed status route
--quality-filter syb_clean  # soft_clean 基础上，只保留 score=100 或仅 min-speed infractions 的 route
```

`soft_clean` 在当前 B2D Full 统计中几乎等价于 full，只去掉少量 `missing_results`，适合作为下一轮 `baseline-condition-v1` 主线。`syb_clean` 会明显削减 brake / class-0 / hard interaction 场景样本，建议先作为 ablation，而不是默认主线。filter 写入 manifest header 的 `quality_filter` 字段；训练时仍通过 `--sample-manifest` 读取该 JSONL。

验证集 manifest 单独构建；如果训练 manifest 使用了 `--skip-first-frames 25`，验证 manifest 也建议显式使用同样参数。若确实要评估不跳过开头帧，可在训练 / eval 命令中加 `--val-skip-first-frames 0` 并使用对应 manifest。

```bash
python tools/build_diffusiondrive_manifest.py \
  --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
  --route-glob "*" \
  --output-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial_skip25.jsonl \
  --frame-sampling 5 \
  --skip-first-frames 25 \
  --max-samples 1024 \
  --num-workers 16 \
  --rebuild \
  --verify-load
```

`--num-workers` 是 CPU 多进程数，只用于并行读取 annotation 和构造 trajectory target / route condition / speed labels；不要把它和训练 DataLoader 的 `--num-workers` 混淆。预构建 manifest 时的 `root-dir / route-glob / frame-sampling / skip-first-frames / target-mode / spatial target 参数 / balanced-scenarios / max-samples-per-scenario` 必须和后续 GPU 训练保持一致。当前 condition-v1 loader 要求 manifest header 和每条 record 都包含 `target_speed_label` 与 `route_condition_feature`；旧 manifest 不再静默 fallback 到逐样本读取 annotation，应直接重建。

## Eval Error Inspection

`tools/inspect_diffusiondrive_eval_errors.py` 用于定位高误差样本。它输出的是模型推理轨迹和 target 之间的 per-sample 诊断误差，不是训练里的 batch-reduced focal + regression loss。

示例：

```bash
conda run -n ltr_garage_2 python tools/inspect_diffusiondrive_eval_errors.py \
  --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
  --route-glob "*" \
  --sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial_skip25.jsonl \
  --checkpoint ~/ltr/dd_logs/full_baseline_condition_v1/soft_clean_fs5_skip25_imgnet/latest.pth \
  --top-k 50 \
  --max-samples 1024 \
  --frame-sampling 5 \
  --skip-first-frames 25 \
  --model-image-height 384 \
  --model-image-width 1024 \
  --image-normalization imagenet \
  --lidar-mode original \
  --diffusion-noise-mode fixed \
  --diffusion-noise-seed 0 \
  --batch-size 16 \
  --num-workers 6 \
  --device cuda:0 \
  --output-csv ~/ltr/dd_logs/full_baseline_condition_v1/soft_clean_fs5_skip25_imgnet/nsj_left_errors.csv
```

输出包括 `l1 / ade / fde` 的 mean / median / p90 / p95 / max，以及 top-k 样本的 `scenario / route / frame / speed / command / target_path_length / target_end / pred_end`。

`--lidar-mode` 支持三种模型侧 LiDAR 开环归因方式：

- `original`：使用样本自身的 LiDAR，作为正常对照。
- `zero`：将模型 LiDAR tensor 全部置零，不影响图像和其他输入。
- `shuffle`：按 `--seed` 构造确定性的全数据集循环错配，每个样本使用另一个样本的 LiDAR；CSV 会记录 donor 的 sample index、scenario、route 和 frame。若 `shuffle` 与 `zero` 都优于 `original`，更支持当前 LiDAR 分支提供了有害信息；若 `shuffle` 明显差于 `zero`，则模型确实依赖 LiDAR，但输入语义或跨模态对齐可能不正确。

三种模式应使用相同 checkpoint、manifest、sample 顺序和 seed，并分别写入不同 CSV；不要把三种模式混在同一输出文件。

`--diffusion-noise-mode` 独立控制 diffusion trajectory head 的初始噪声：

- `random`：保持历史行为，每次 forward 从全局 torch RNG 取新噪声。
- `fixed`：用 `--diffusion-noise-seed` 生成一份 `[1,num_modes,num_poses,2]` template，再扩展到 batch；同一 seed 对所有样本相同，并且不依赖 batch size。
- `zero`：使用全零初始噪声，只作分布外诊断，不能直接作为正式推理方案。
- `seeded`：按 `base_seed + scenario + route + frame` 的稳定 hash 在 Dataset worker 中生成显式逐样本噪声；同一样本不受 batch size、worker 数和读取顺序影响。

`--seed` 仍控制全局诊断 RNG 和 LiDAR shuffle donor；它不再代替 `--diffusion-noise-seed`。CSV 额外记录 noise mode/seed/key、最终 trajectory mode、top-2 mode/probability/endpoint、top-1 margin 和 normalized entropy。模型默认仍为 `random`，checkpoint tensor schema 不变。

闭环可通过以下 env 使用同一 noise contract：

```bash
DIFFUSIONDRIVE_DIFFUSION_NOISE_MODE=fixed \
DIFFUSIONDRIVE_DIFFUSION_NOISE_SEED=0 \
DIFFUSIONDRIVE_DEBUG_TRAJECTORY=1 \
DIFFUSIONDRIVE_TRAJECTORY_DEBUG_INTERVAL=1 \
DIFFUSIONDRIVE_TRAJECTORY_JUMP_WARN=5 \
bash tools/run_baseline_basic_closed_loop.sh
```

在线诊断默认写到 `${OUT}/trajectory_diagnostics/<route_timestamp_pid>/records.jsonl`。每条记录包含 mode/margin/entropy、noise key、ego state、top-2 endpoint，以及把上一 tick endpoint 对齐到当前 ego frame 后计算的 `aligned_endpoint_jump`。`seeded` 在线模式使用 `START_IDX + step` 作为稳定 key；要研究跨 tick mode flicker 时应优先使用 `fixed`。

## LiDAR Contract Inspection

`tools/inspect_diffusiondrive_lidar_contract.py` 使用与在线 agent 相同的统计 schema，直接检查 B2D `.laz -> histogram` 数据契约，不做模型 forward。建议先在 full eval manifest 上抽取 scenario-balanced 的 4096 个样本：

```bash
conda run -n ltr_garage_2 python tools/inspect_diffusiondrive_lidar_contract.py \
  --root-dir /share/home/u19666033/djy/carla_dataset \
  --route-glob "*/*" \
  --sample-manifest /share/home/u19666033/ltr/dd_cache/full_condition_v1_eval_soft_clean_1024ps_fs5_skip25_spatial.jsonl \
  --output-dir /share/home/u19666033/ltr/dd_logs/full_baseline_condition_v1/lidar_contract/b2d_raw_soft_clean_4096 \
  --max-records 4096 \
  --frame-sampling 5 \
  --skip-first-frames 25 \
  --target-mode spatial_path \
  --balanced-scenarios \
  --max-samples-per-scenario 1024 \
  --dump-every 256 \
  --max-dumps 16 \
  --log-every 100
```

输出目录包含逐样本 `records.jsonl`、按 stage 和 BEV channel 汇总的 `summary.json`，以及受 `--max-dumps` 限制的 `dumps/*.npz` 与对应 metadata。统计覆盖 raw / model point count、有限点和 BEV 范围保留率、below/above split 点数、z 分位数、front/back/left/right occupancy、BEV nonzero / mean / max / saturation 及逐通道分布。

在线闭环使用同一 schema，但额外拆成 `current_half`、`previous_half_aligned`、`full_scan` 和 `model_input` 四个 stage：

```bash
DIFFUSIONDRIVE_DEBUG_LIDAR_CONTRACT=1 \
DIFFUSIONDRIVE_LIDAR_DEBUG_INTERVAL=20 \
DIFFUSIONDRIVE_LIDAR_DUMP_INTERVAL=200 \
DIFFUSIONDRIVE_LIDAR_DUMP_MAX=30 \
DIFFUSIONDRIVE_LIDAR_HISTORY_STEPS=100 \
bash tools/run_baseline_basic_closed_loop.sh
```

默认输出到 `${OUT}/lidar_diagnostics/<route_timestamp_pid>/`，也可用 `DIFFUSIONDRIVE_LIDAR_DEBUG_DIR` 改根目录。周期性 dump 默认关闭；creep / safety-box stop 会在 rate limit 和 dump 上限内保存事件上下文，`destroy()` 会补写 route 终止前的 recent history。正式 C2 对照不要开启该诊断，单独选 route 00 / 24 或 blocked route 做 dump，避免磁盘 I/O 干扰闭环时序。

## 当前限制

- 远端当前 B2D Full 训练图像实际观测为 `512x1024`；旧 raw sensor 元数据中的 `1600x900, fov=70` 只能作为 provenance 记录，不应再写成当前 tensor 的实际 source size。dataset 仍会先 resize 到在线 garage sensor size `512x1024`，再执行 `crop_array -> 384x1024`，当前默认模型输入保持 `384x1024`。这会对齐模型输入 shape，但不消除 B2D Full raw sensor 与在线 garage sensor suite 的 FOV / pose / LiDAR 外参 gap。
- dataset helper 兼容早期 `camera/rgb_front + anno` 和 Full 原生 `rgb + measurements` 两种 route 结构；Full 原生逐帧标注来自 `measurements/*.json.gz`，不是 `records.json.gz`。
- 默认 trajectory target 已切换为 `spatial_path`：从 future ego path 中按 `2.5m, 3.5m, ..., 11.5m` 空间距离重采样，使 target 与当前 `99x10x2` anchor 的空间 checkpoint 语义一致。
- `spatial_path` 样本发现至少要求下一帧 annotation 存在；future path 不足覆盖 `11.5m` 时才沿路径末段或 command 方向外推。
- `future_stride` 只用于 `--target-mode future_ego_time` legacy 路径；默认训练不再把 target 点解释为固定时间间隔。
- `trajectory_sampling.interval_length` 目前仍保留为 DiffusionDrive config 兼容字段，不代表当前空间 checkpoint target 的真实时间间隔。
- 推理侧 `DiffusionDriveAgent` 默认启用空间 checkpoint PID，不再从 waypoint index 的 0.5s / 1.0s 时间假设估计 desired speed；可用 `DIFFUSIONDRIVE_SPATIAL_PID=0` 临时回到旧逻辑做 A/B。
- 空间 PID 的速度和转弯阈值已支持 env 覆盖：`DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST`、`DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW`、`DIFFUSIONDRIVE_SPATIAL_PID_TURN_THRESHOLD`、`DIFFUSIONDRIVE_SPATIAL_PID_SHARP_TURN_THRESHOLD`；未设置时继续使用 `GlobalConfig` 默认值。
- 推理侧 predicted-speed controller 已支持 `DIFFUSIONDRIVE_USE_SPEED_HEAD=1`，用模型 `target_speed_logits` 转出的目标速度替代空间 PID 的 longitudinal desired speed；默认 `0` 保持空间 PID fallback。默认转换方式与 syb 对齐：`DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT=1`、`DIFFUSIONDRIVE_SPEED_HEAD_BRAKE_THRESHOLD=0.9`，class 0 概率未超过阈值时用概率期望速度，超过阈值才强制 `desired_speed=0`；`DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT=0` 为 argmax 诊断。闭环 debug 会打印 `speed_head_class / speed_head_selection / speed_head_desired / speed_head_expected / speed_head_brake_prob / speed_head_max_prob`。旧 `C1_speedhead` route 00 class-0 全刹锁死是修复前结果，新版必须重新跑 00 / 24 sanity 后再评价。
- 闭环 agent 默认要求 checkpoint 与当前模型结构完全匹配，并校验训练 checkpoint 中的 `image_normalization / model_image_height / model_image_width` metadata。仅在 warm-start 或部署诊断时设置 `DIFFUSIONDRIVE_ALLOW_PARTIAL_CHECKPOINT=1` 或 `DIFFUSIONDRIVE_ALLOW_PREPROCESS_MISMATCH=1`，正式结果中必须记录这些 env。
- 推理侧默认使用当前 `far_command.value` 构造 `status_feature`，与训练侧当前 command 语义对齐；可用 `DIFFUSIONDRIVE_COMMAND_DELAY=1` 启用旧 garage / `sensor_agent.py` 的 `commands[-2]` 延迟逻辑做 A/B。
- 推理侧新增 `DIFFUSIONDRIVE_LOW_SPEED_STEER=1` 闭环 A/B 开关：低速近似静止但未 brake 时保留横向 PID angle，默认 `0` 以保持 baseline 行为；该开关用于验证低速起步直行是否导致 route deviation。
- stuck recovery 已支持 env 覆盖：`DIFFUSIONDRIVE_STUCK_THRESHOLD`、`DIFFUSIONDRIVE_CREEP_DURATION`、`DIFFUSIONDRIVE_CREEP_THROTTLE`；未设置时继续使用 `GlobalConfig` 默认值。
- 推理侧已支持在线 sensor / model override：`DIFFUSIONDRIVE_CAMERA_FOV`、`DIFFUSIONDRIVE_CAMERA_POS`、`DIFFUSIONDRIVE_CAMERA_ROT`、`DIFFUSIONDRIVE_CAMERA_WIDTH`、`DIFFUSIONDRIVE_CAMERA_HEIGHT`、`DIFFUSIONDRIVE_LIDAR_POS`、`DIFFUSIONDRIVE_LIDAR_ROT`、`DIFFUSIONDRIVE_CROP_IMAGE`、`DIFFUSIONDRIVE_MODEL_IMAGE_HEIGHT`、`DIFFUSIONDRIVE_MODEL_IMAGE_WIDTH`。这些只影响 closed-loop online agent；训练侧 crop / camera metadata CLI 仍需后续补齐。
- 推理侧已支持 `DIFFUSIONDRIVE_ZERO_LIDAR=1`，用于把模型 LiDAR BEV 输入置零做诊断性 ablation；该开关不关闭 raw LiDAR safety-box。20-route Z0/Z1 和 condition-v1 L0 诊断显示 zero model-LiDAR 反而优于对应对照，说明当前模型侧 LiDAR 分支接入不稳定；但项目最终路线仍要求使用 LiDAR，因此 zero/no-LiDAR 只能作为诊断上界。正式 baseline 应优先验证 train-vs-online LiDAR BEV contract、做 channel ablation、fusion gate / dropout 和更强 LiDAR auxiliary supervision。
- 推理侧 UKF 已加入 covariance 正定保护和 measurement reset，避免 `filterpy` 在 `P` 非正定时直接导致 agent crash。
- 闭环 A/B 建议打开 `DIFFUSIONDRIVE_DEBUG_CONTROL=1` 和 `DIFFUSIONDRIVE_DEBUG_INTERVAL=20`，观察 command、desired speed、turn ratio、aim waypoint、angle reset、control、stuck / force_move / stop sign。若排查 route deviation 或 creep / safety-box，可额外打开 `DIFFUSIONDRIVE_DEBUG_ROUTE=1`、`DIFFUSIONDRIVE_DEBUG_SAFETY_BOX=1`；route warning 阈值可用 `DIFFUSIONDRIVE_ROUTE_DEBUG_DISTANCE_WARN`、`DIFFUSIONDRIVE_ROUTE_DEBUG_ANGLE_WARN_DEG` 调整。
- 远端闭环 A/B 前必须同时同步 `team_code/diffusiondrive_agent.py` 和 `team_code/config.py`；新版 agent 依赖 `GlobalConfig.diffusiondrive_spatial_pid*` 默认参数。
- 已支持 `torchrun` / DDP 多卡训练；暂不支持 AMP / EMA。
- 暂不训练 auxiliary heads。原版 NAVSIM 会监督 `agent_states / agent_labels / bev_semantic_map`，这可能也是当前 LiDAR BEV 分支闭环负贡献的原因之一；若后续继续使用 LiDAR，优先考虑补 auxiliary supervision，并用 zero/no-LiDAR 只做诊断对照。
- `--resume-file` 会恢复 model / optimizer / scheduler / global step，并从 checkpoint 记录的下一个 epoch 继续；中途 step checkpoint 恢复时不会恢复 dataloader 在 epoch 内的位置。
- `status_feature` 已迁移到 `command_one_hot(6) + speed(1)` 的 `7` 维 schema；speed 暂不归一化。
