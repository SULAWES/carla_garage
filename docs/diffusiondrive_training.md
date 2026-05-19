# DiffusionDrive CARLA Training

本文档记录当前第一版 CARLA-native DiffusionDrive 训练入口。

## 当前入口

- 训练脚本：`team_code/train_diffusiondrive.py`
- 数据集 helper：`team_code/diffusiondrive/carla_native_dataset.py`
- 支持数据格式：原始 Bench2Drive route 目录
  - `camera/rgb_front/*.jpg`
  - `lidar/*.laz`
  - `anno/*.json.gz`

当前只训练 trajectory head 主路径。模型输出的 `trajectory_loss` 是 trajectory head 内部的未加外层权重损失；训练脚本会再乘 `DiffusionDriveConfig.trajectory_weight` 得到反传用的总 loss。当前不接入 `agent_states / agent_labels / bev_semantic_map` 辅助 loss。

训练入口会在输出目录写入：

- `training_config.json`：CLI 参数、DiffusionDrive config、数据 split、预处理和 status feature schema
- `checkpoint_epoch*_step*.pth`：周期或结束 checkpoint
- `latest.pth`：最近一次保存的 checkpoint，供 `--resume-file` 使用

可配置项已经覆盖第一阶段 full training 需要的基础参数：

- optimizer：`--optimizer adamw`、`--lr`、`--weight-decay`
- scheduler：`--scheduler none|cosine|multistep`、`--warmup-steps`、`--lr-steps`、`--lr-gamma`、`--min-lr`；其中 `--lr-steps` 是按 optimizer step 计数的逗号分隔 milestone
- loss：`--trajectory-weight`、`--trajectory-cls-weight`、`--trajectory-reg-weight`、`--trajectory-focal-alpha`、`--trajectory-focal-gamma`
- diffusion：`--diffusion-num-train-timesteps`、`--diffusion-train-timestep-min/max`、`--diffusion-infer-step-num`、`--diffusion-infer-timestep-span`、`--diffusion-infer-trunc-timesteps`
- data split：`--val-root-dir`、`--val-route-glob`、`--val-frame-sampling`、`--val-max-samples`、`--val-every-steps`、`--max-val-steps`
- preprocessing：`--model-image-height`、`--model-image-width`、`--no-jpeg-artifact`

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
- `epoch=0 step=1 loss=3031.9680`
- `trajectory_unweighted=252.6640`
- `trajectory_loss_0=126.4926`
- `trajectory_loss_1=126.1714`
- checkpoint：`/tmp/dd_train_smoke_config/smoke/checkpoint_epoch000_step0000001.pth`

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

## Full 数据集示例

远程 full 数据集只需要替换 `--root-dir` 和 `--logdir`：

```bash
conda run -n garage_2 python team_code/train_diffusiondrive.py \
  --root-dir /path/to/Bench2Drive-full-extracted \
  --logdir /path/to/logs \
  --id dd_carla_native_v1 \
  --epochs 20 \
  --batch-size 4 \
  --frame-sampling 5 \
  --num-workers 8 \
  --scheduler cosine \
  --warmup-steps 1000 \
  --save-every-steps 1000
```

如需从 NAVSIM checkpoint 初始化：

```bash
--load-file /home/HeavenlySU/sitp_workspace/diffusiondrive_navsim_88p1_PDMS
```

脚本会按 key 和 shape 部分加载 checkpoint；当前 `99x10x2` trajectory head 相关 mismatch 属于预期。

## 当前限制

- raw Bench2Drive 图像通常是 `900x1600`。当前 dataset 为了匹配已冻结 CARLA-native 主线，会先 resize 到在线 sensor size `512x1024`，再执行 `crop_array -> 256x1024`。
- 未来轨迹 target 暂时从 `anno` 的 `x/y/theta` 直接构造 `10x2` XY 轨迹。
- 暂不支持 distributed / AMP / EMA。
- 暂不训练 auxiliary heads。
- `--resume-file` 会恢复 model / optimizer / scheduler / global step，并从 checkpoint 记录的下一个 epoch 继续；中途 step checkpoint 恢复时不会恢复 dataloader 在 epoch 内的位置。
- `status_feature` 仍沿用当前定义：`command(6) + velocity(2) + acceleration(2)`；其中 lateral velocity / acceleration 暂置 0。
