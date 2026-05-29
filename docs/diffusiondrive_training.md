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

当前只训练 trajectory head 主路径。模型输出的 `trajectory_loss` 是 trajectory head 内部的未加外层权重损失；训练脚本会再乘 `DiffusionDriveConfig.trajectory_weight` 得到反传用的总 loss。当前不接入 `agent_states / agent_labels / bev_semantic_map` 辅助 loss。

训练入口会在输出目录写入：

- `training_config.json`：CLI 参数、DiffusionDrive config、数据 split、B2D Full sensor contract、时间语义、anchor shape、预处理和 status feature schema
- `checkpoint_epoch*_step*.pth`：周期或结束 checkpoint
- `latest.pth`：最近一次保存的 checkpoint，供 `--resume-file` 使用

可配置项已经覆盖第一阶段 full training 需要的基础参数：

- optimizer：`--optimizer adamw`、`--lr`、`--weight-decay`、`--image-encoder-lr-mult`
- scheduler：`--scheduler none|cosine|multistep`、`--warmup-steps`、`--lr-steps`、`--lr-gamma`、`--min-lr`；其中 `--lr-steps` 是按 optimizer step 计数的逗号分隔 milestone
- loss：`--trajectory-weight`、`--trajectory-cls-weight`、`--trajectory-reg-weight`、`--trajectory-focal-alpha`、`--trajectory-focal-gamma`
- diffusion：`--diffusion-num-train-timesteps`、`--diffusion-train-timestep-min/max`、`--diffusion-infer-step-num`、`--diffusion-infer-timestep-span`、`--diffusion-infer-trunc-timesteps`
- data split：`--val-root-dir`、`--val-route-glob`、`--val-frame-sampling`、`--val-max-samples`、`--val-every-steps`、`--max-val-steps`
- dataset / target / time contract：`--dataset-mode`、`--target-mode`、`--spatial-target-first-distance`、`--spatial-target-interval`、`--spatial-target-max-future-frames`、`--assumed-frame-interval`、`--future-stride`、`--b2d-source-image-height`、`--b2d-source-image-width`
- scenario balancing：`--balanced-scenarios`、`--max-samples-per-scenario`
- distributed：`--distributed auto|none|ddp`，默认 `auto`，用 `torchrun` 启动且 `WORLD_SIZE>1` 时自动启用 DDP
- hard-case weighting：`--hard-left-turn-stop-loss-weight`、`--hard-left-turn-command`、`--hard-left-turn-speed-threshold`、`--hard-left-turn-y-threshold`
- sample distribution stats：`--dataset-stats-max-samples` 会落盘 `train_sample_distribution.json` / `validation_sample_distribution.json` / `eval_sample_distribution.json`
- dataloader / manifest：`--sample-manifest`、`--val-sample-manifest`、`--rebuild-sample-manifest`、`--prefetch-factor`、`--persistent-workers`
- preprocessing：`--model-image-height`、`--model-image-width`、`--no-jpeg-artifact`
- evaluation：`--eval-only` 会只加载模型并跑评估，不进入训练循环；训练 checkpoint 用 `--resume-file` 严格加载 `model`，普通权重 / NAVSIM checkpoint 可用 `--load-file` 部分加载

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

DDP 下 `--batch-size` 是每张 GPU / 每个进程的 batch size，实际 global batch size 为 `batch_size * WORLD_SIZE`。scheduler、`--warmup-steps`、`--val-every-steps`、`--save-every-steps` 都按 optimizer step 计数；由于 DDP 每个 epoch 的 optimizer step 数约为 `ceil(samples / global_batch_size)`，计算 warmup 和保存间隔时要用 global batch size。rank0 负责写 `training_config.json`、sample distribution、validation 和 checkpoint；checkpoint 保存的是未包 DDP 的普通 model state dict，后续单卡或多卡都可以 resume。当前 DDP 使用 `find_unused_parameters=True`，并在 DDP 模式下对 auxiliary outputs 加 `0.0 * output.sum()` dummy term，使未监督 heads 产生零梯度；这不改变数值 loss，只是避免 trajectory-only baseline 下的 DDP unused-branch 问题。

## Current Baseline-Basic Run

当前远端正在训练 full `baseline-basic`，目标是作为论文 baseline 和后续持续学习方法的干净起点。该实验使用 B2D Full scenario-balanced 全量 train manifest，不使用 Stage5 / Stage6 tuned checkpoint，不启用 hard-case weighting：

```text
logdir=/share/home/u19666033/ltr/dd_logs/full_baseline_basic
id=origlike_bs64_lr6e-4_ep100_fs5_spatial_imgenc0p5
train_manifest=/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl
val_manifest=/share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl
epochs=100
batch_size=64
lr=6e-4
weight_decay=1e-4
image_encoder_lr_mult=0.5
hard_left_turn_stop_loss_weight=1.0
load_file=""
```

这个 run 已尽量对齐原版 DiffusionDrive 的主要训练预算和优化器设置；当前训练入口已支持 DDP，但尚未对齐 AMP。训练结果和中途诊断记录在 `diffusiondrive_remote_training_progress.md`。

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
  --output-manifest /share/home/u19666033/ltr/dd_cache/full_stage5_train_2048ps_fs5_spatial.jsonl \
  --frame-sampling 5 \
  --balanced-scenarios \
  --max-samples-per-scenario 2048 \
  --num-workers 32 \
  --rebuild \
  --verify-load
```

验证集 manifest 单独构建：

```bash
python tools/build_diffusiondrive_manifest.py \
  --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
  --route-glob "*" \
  --output-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl \
  --frame-sampling 5 \
  --max-samples 1024 \
  --num-workers 16 \
  --rebuild \
  --verify-load
```

`--num-workers` 是 CPU 多进程数，只用于并行读取 annotation 和构造 trajectory target；不要把它和训练 DataLoader 的 `--num-workers` 混淆。预构建 manifest 时的 `root-dir / route-glob / frame-sampling / target-mode / spatial target 参数 / balanced-scenarios / max-samples-per-scenario` 必须和后续 GPU 训练保持一致。

## Eval Error Inspection

`tools/inspect_diffusiondrive_eval_errors.py` 用于定位高误差样本。它输出的是模型推理轨迹和 target 之间的 per-sample 诊断误差，不是训练里的 batch-reduced focal + regression loss。

示例：

```bash
conda run -n ltr_garage_2 python tools/inspect_diffusiondrive_eval_errors.py \
  --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
  --route-glob "*" \
  --sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl \
  --checkpoint ~/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth \
  --top-k 50 \
  --max-samples 1024 \
  --frame-sampling 5 \
  --batch-size 16 \
  --num-workers 6 \
  --device cuda:0 \
  --output-csv ~/ltr/dd_logs/full_eval_stage2/nsj_left_errors.csv
```

输出包括 `l1 / ade / fde` 的 mean / median / p90 / p95 / max，以及 top-k 样本的 `scenario / route / frame / speed / command / target_path_length / target_end / pred_end`。

## 当前限制

- B2D Full raw 图像通常是 `900x1600`。当前 dataset 会先 resize 到在线 garage sensor size `512x1024`，再执行 `crop_array -> 256x1024`。这会对齐模型输入 shape，但不消除 B2D Full raw sensor 与在线 garage sensor suite 的 FOV / pose / LiDAR 外参 gap。
- dataset helper 兼容早期 `camera/rgb_front + anno` 和 Full 原生 `rgb + measurements` 两种 route 结构；Full 原生逐帧标注来自 `measurements/*.json.gz`，不是 `records.json.gz`。
- 默认 trajectory target 已切换为 `spatial_path`：从 future ego path 中按 `2.5m, 3.5m, ..., 11.5m` 空间距离重采样，使 target 与当前 `99x10x2` anchor 的空间 checkpoint 语义一致。
- `spatial_path` 样本发现至少要求下一帧 annotation 存在；future path 不足覆盖 `11.5m` 时才沿路径末段或 command 方向外推。
- `future_stride` 只用于 `--target-mode future_ego_time` legacy 路径；默认训练不再把 target 点解释为固定时间间隔。
- `trajectory_sampling.interval_length` 目前仍保留为 DiffusionDrive config 兼容字段，不代表当前空间 checkpoint target 的真实时间间隔。
- 推理侧 `DiffusionDriveAgent` 默认启用空间 checkpoint PID，不再从 waypoint index 的 0.5s / 1.0s 时间假设估计 desired speed；可用 `DIFFUSIONDRIVE_SPATIAL_PID=0` 临时回到旧逻辑做 A/B。
- 推理侧默认使用当前 `far_command.value` 构造 `status_feature`，与训练侧当前 command 语义对齐；可用 `DIFFUSIONDRIVE_COMMAND_DELAY=1` 启用旧 garage / `sensor_agent.py` 的 `commands[-2]` 延迟逻辑做 A/B。
- 闭环 A/B 建议打开 `DIFFUSIONDRIVE_DEBUG_CONTROL=1` 和 `DIFFUSIONDRIVE_DEBUG_INTERVAL=20`，观察 command、desired speed、turn ratio、aim waypoint、control、stuck / force_move / stop sign。
- 远端闭环 A/B 前必须同时同步 `team_code/diffusiondrive_agent.py` 和 `team_code/config.py`；新版 agent 依赖 `GlobalConfig.diffusiondrive_spatial_pid*` 默认参数。
- 暂不支持 distributed / AMP / EMA。
- 暂不训练 auxiliary heads。
- `--resume-file` 会恢复 model / optimizer / scheduler / global step，并从 checkpoint 记录的下一个 epoch 继续；中途 step checkpoint 恢复时不会恢复 dataloader 在 epoch 内的位置。
- `status_feature` 已迁移到 `command_one_hot(6) + speed(1)` 的 `7` 维 schema；speed 暂不归一化。
