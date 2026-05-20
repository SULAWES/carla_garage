# DiffusionDrive CARLA Training

本文档记录当前第一版 Bench2Drive Full / CARLA-native DiffusionDrive 训练入口。

## 当前入口

- 训练脚本：`team_code/train_diffusiondrive.py`
- 数据集 helper：`team_code/diffusiondrive/carla_native_dataset.py`
- 支持数据格式：原始 Bench2Drive / Bench2Drive Full route 目录
  - `camera/rgb_front/*.jpg`
  - `lidar/*.laz`
  - `anno/*.json.gz`

当前训练和后续主要修改以 B2D Full raw 数据为主训练分布。sensor / time / preprocessing contract 见 `b2d_full_sensor_contract.md`。

当前只训练 trajectory head 主路径。模型输出的 `trajectory_loss` 是 trajectory head 内部的未加外层权重损失；训练脚本会再乘 `DiffusionDriveConfig.trajectory_weight` 得到反传用的总 loss。当前不接入 `agent_states / agent_labels / bev_semantic_map` 辅助 loss。

训练入口会在输出目录写入：

- `training_config.json`：CLI 参数、DiffusionDrive config、数据 split、B2D Full sensor contract、时间语义、anchor shape、预处理和 status feature schema
- `checkpoint_epoch*_step*.pth`：周期或结束 checkpoint
- `latest.pth`：最近一次保存的 checkpoint，供 `--resume-file` 使用

可配置项已经覆盖第一阶段 full training 需要的基础参数：

- optimizer：`--optimizer adamw`、`--lr`、`--weight-decay`
- scheduler：`--scheduler none|cosine|multistep`、`--warmup-steps`、`--lr-steps`、`--lr-gamma`、`--min-lr`；其中 `--lr-steps` 是按 optimizer step 计数的逗号分隔 milestone
- loss：`--trajectory-weight`、`--trajectory-cls-weight`、`--trajectory-reg-weight`、`--trajectory-focal-alpha`、`--trajectory-focal-gamma`
- diffusion：`--diffusion-num-train-timesteps`、`--diffusion-train-timestep-min/max`、`--diffusion-infer-step-num`、`--diffusion-infer-timestep-span`、`--diffusion-infer-trunc-timesteps`
- data split：`--val-root-dir`、`--val-route-glob`、`--val-frame-sampling`、`--val-max-samples`、`--val-every-steps`、`--max-val-steps`
- dataset / time contract：`--dataset-mode`、`--assumed-frame-interval`、`--future-stride`、`--b2d-source-image-height`、`--b2d-source-image-width`
- preprocessing：`--model-image-height`、`--model-image-width`、`--no-jpeg-artifact`

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
- `time_semantics`: `target_interval_seconds=1.0`, `trajectory_sampling_interval_seconds=0.5`, `anchor_shape=[99,10,2]`
- `epoch=0 step=1 loss=2252.0107`
- `trajectory_unweighted=187.6676`
- `trajectory_loss_0=93.9595`
- `trajectory_loss_1=93.7080`
- checkpoint：`/tmp/dd_train_b2d_contract_smoke/smoke/checkpoint_epoch000_step0000001.pth`

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

远程 B2D Full 数据集只需要替换 `--root-dir` 和 `--logdir`。默认 `--dataset-mode b2d_full_raw`、`--assumed-frame-interval 0.1`、`--future-stride 10`，训练配置会记录这些假设：

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

- B2D Full raw 图像通常是 `900x1600`。当前 dataset 会先 resize 到在线 garage sensor size `512x1024`，再执行 `crop_array -> 256x1024`。这会对齐模型输入 shape，但不消除 B2D Full raw sensor 与在线 garage sensor suite 的 FOV / pose / LiDAR 外参 gap。
- 未来轨迹 target 优先使用 raw Bench2Drive ego vehicle `world2ego` 矩阵，将未来 ego vehicle `location` 转到当前 ego frame；缺少 bbox 矩阵时才 fallback 到经过 `preprocess_compass()` 等价处理的 `x/y/theta`。
- 轨迹时间语义仍未冻结；`training_config.json` 现在会记录 frame interval、future stride、target interval、trajectory sampling interval 和 anchor shape，供 full training 前检查。
- 暂不支持 distributed / AMP / EMA。
- 暂不训练 auxiliary heads。
- `--resume-file` 会恢复 model / optimizer / scheduler / global step，并从 checkpoint 记录的下一个 epoch 继续；中途 step checkpoint 恢复时不会恢复 dataloader 在 epoch 内的位置。
- `status_feature` 已迁移到 `command_one_hot(6) + speed(1)` 的 `7` 维 schema；speed 暂不归一化。
