# DiffusionDrive 本地环境备忘

本文档只记录当前机器上用于 DiffusionDrive 本地 smoke test 的最小事实。

## 环境

- Conda 环境：`garage_2`
- Python：通过 `conda run -n garage_2 python ...` 调用
- 已确认包：
  - `numpy 1.26.4`
  - `torch 2.5.0+cu124`
  - `timm 1.0.11`
  - `einops 0.4.1`
  - `diffusers 0.38.0`

注意：安装 `diffusers` 时，`safetensors` 被更新为 `0.8.0rc0`。

## 本地文件

- Anchor：`/home/HeavenlySU/sitp_workspace/4-0-0-1910-tracked_clusters_anchor.npy`
  - shape：`(99, 10, 2)`
- Backbone：`/home/HeavenlySU/sitp_workspace/pytorch_model.bin`
- Checkpoint：`/home/HeavenlySU/sitp_workspace/diffusiondrive_navsim_88p1_PDMS`

## 已验证

注意：本文早期 smoke 记录来自 NAVSIM checkpoint / baseline-basic 迁移阶段，其中部分 run 使用旧模型输入 `256x1024`。当前 `DiffusionDriveConfig` 和 `train_diffusiondrive.py` 默认模型输入已切到 `384x1024`，`baseline-condition-v1` 也以 `384x1024` 为主线；如果要把本地 smoke 作为当前验证结论，需要按新默认重新运行一次。

- config 自动推导：
  - `num_anchor_modes = 99`
  - `num_poses = 10`
  - `time_horizon = 5.0`
- 模型实例化通过：
  - `plan_anchor = (99, 10, 2)`
  - regression head 输出维度 `20 = 10 * 2`
  - anchor encoder 输入维度 `640 = 10 * 64`
- 随机输入 forward 通过：
  - `trajectory = (1, 10, 2)`
- checkpoint 部分加载 smoke 通过：
  - `loaded 751 / 763`
  - `missing 12`
  - `unexpected 0`
  - forward 后仍输出 `trajectory = (1, 10, 2)`
- `DiffusionDriveAgent.setup()` smoke 通过：
  - `matched 752 / 763`
  - `shape mismatch = 11`
  - `setup_ok cpu 99 10 (99, 10, 2)`
- 伪输入 `run_step()` smoke 通过：
  - 历史记录 camera feature：`(1, 3, 256, 1024)`
  - 当前默认期望 camera feature：`(1, 3, 384, 1024)`
  - LiDAR feature：`(1, 1, 256, 256)`
  - status feature：`(1, 7)`
  - route condition feature：`(1, 4)`
  - trajectory：`(1, 10, 2)`
- Bench2Drive mini 单样本 smoke 通过：
  - 数据目录：`/home/HeavenlySU/sitp_workspace/carla_garage/Bench2Drive/Bench2Drive-mini-extracted`
  - 测试 route：`AccidentTwoWays_Town12_Route1444_Weather0`
  - 测试帧：`00100`
  - 使用真实 `rgb_front`、真实 `.laz` LiDAR、真实 `anno`
  - 脚本：`tools/smoke_diffusiondrive_b2d_mini.py`
  - 命令：`conda run -n garage_2 python tools/smoke_diffusiondrive_b2d_mini.py`
  - smoke 脚本已显式暴露 `--target-mode spatial_path|future_ego_time`；默认与训练入口一致为 `spatial_path`
  - raw B2D mini 图像可能仍为旧尺寸；当前远端 B2D Full 训练图像实际观测为 `512x1024`，输入转换见 `b2d_full_sensor_contract.md`
  - target mode：`spatial_path`
  - 从 future ego path 按 `2.5m, 3.5m, ..., 11.5m` 空间距离重采样 target：`(1, 10, 2)`
  - status feature：`(1, 7)`
  - route condition feature：`(1, 4)`
  - 模型输出：`trajectory = (1, 10, 2)`
  - 最近一次结果：`loss = 17.09078598022461`，`grad_norm = 0.8400353789329529`
  - trajectory loss 为标量，backward 通过
- Bench2Drive mini 训练入口 smoke 通过：
  - 脚本：`team_code/train_diffusiondrive.py`
  - 命令见 `docs/diffusiondrive_training.md`
  - `Dataset samples: 2`
  - `data.dataset_mode=b2d_full_raw`
  - `target.mode=spatial_path`
  - `target.spatial_path`: `first=2.5m`, `interval=1.0m`, `last=11.5m`
  - `epoch=0 step=1 loss=287.6801`
  - checkpoint：`/tmp/dd_train_spatial_target_smoke/smoke/checkpoint_epoch000_step0000001.pth`
- B2D Full 原生 route 结构 smoke 通过：
  - 测试结构：`rgb/*.jpg`、`lidar/*.laz`、`measurements/*.json.gz`
  - 测试 route：`/home/HeavenlySU/sitp_workspace/Town12_Rep0_10_0_route0_11_08_23_53_07`
  - 命令：`conda run -n garage_2 python team_code/train_diffusiondrive.py --root-dir /home/HeavenlySU/sitp_workspace --route-glob Town12_Rep0_10_0_route0_11_08_23_53_07 --logdir /tmp/dd_train_full_native_smoke --id smoke --epochs 1 --batch-size 1 --max-samples 2 --max-steps 1 --frame-sampling 20 --num-workers 0 --load-file ""`
  - `Dataset samples: 2`
  - `epoch=0 step=1 loss=249.4417`
  - checkpoint：`/tmp/dd_train_full_native_smoke/smoke/checkpoint_epoch000_step0000001.pth`

手写模型 smoke 中的 12 个未加载 tensor 主要来自 LiDAR 输入通道、status 维度和旧 `20x8` / `8x3` trajectory head。真实 agent setup 使用当前 `GlobalConfig`，LiDAR 输入通道与 checkpoint 对齐，因此只剩 11 个 mismatch；这些均与当前 `99x10x2` 迁移预期一致。

Bench2Drive mini smoke 中的 `10x2` target 默认不再是 fixed-time future ego trajectory，而是从 future ego path 空间重采样得到的 checkpoint target。坐标转换优先使用 raw annotation 中 ego vehicle `world2ego` 矩阵，缺失矩阵时才 fallback 到经过 `preprocess_compass()` 等价处理的 `x/y/theta`。当前 `team_code/diffusiondrive/backbone.py` 已在本地 backbone 文件存在时优先加载本地权重，避免 smoke test 先访问 HuggingFace 再 fallback。

2026-07-15 `arc_length_stable_extrapolation_v2` 上线后重新完成：

- B2D mini fixed-noise forward/loss/backward，重复 inference `max_abs_diff=0.0`。
- B2D Full 原生 route 1-step 训练、checkpoint 保存、新 manifest 写入与重载。
- manifest header 和 `training_config.json` 均包含 `arc_length_stable_extrapolation_v2`、`0.5m` extrapolation baseline 和 `0.001m` 去重阈值。
- 旧 spatial manifest 缺少新 contract 时会明确报错，不会静默进入训练。

## 常用环境变量

```bash
export DIFFUSIONDRIVE_ANCHOR_PATH=/home/HeavenlySU/sitp_workspace/4-0-0-1910-tracked_clusters_anchor.npy
export DIFFUSIONDRIVE_BACKBONE_PATH=/home/HeavenlySU/sitp_workspace/pytorch_model.bin
export DIFFUSIONDRIVE_CHECKPOINT=/home/HeavenlySU/sitp_workspace/diffusiondrive_navsim_88p1_PDMS
```
