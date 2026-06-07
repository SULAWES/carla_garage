CUDA_VISIBLE_DEVICES=0 conda run -n ltr_garage_2 python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --logdir /share/home/u19666033/ltr/logs/dd_full_smoke \
    --id full_smoke_1gpu \
    --epochs 1 \
    --batch-size 1 \
    --max-samples 8 \
    --max-steps 2 \
    --frame-sampling 50 \
    --num-workers 0 \
    --load-file "" \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin

CUDA_VISIBLE_DEVICES=0 python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --logdir /share/home/u19666033/ltr/logs/dd_full_smoke \
    --id full_smoke_1gpu \
    --epochs 1 \
    --batch-size 1 \
    --max-samples 8 \
    --max-steps 2 \
    --frame-sampling 50 \
    --num-workers 0 \
    --load-file "" \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin

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
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file ""

### 吞吐量测试

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --logdir ~/ltr/dd_logs/full_throughput \
    --id bs4_workers6 \
    --epochs 1 \
    --batch-size 4 \
    --max-samples 512 \
    --max-steps 100 \
    --frame-sampling 5 \
    --num-workers 6 \
    --log-every 10 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file ""

### 1000step

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --logdir ~/ltr/dd_logs/full_stage0 \
    --id spatial_path_bs4_lr1e-4 \
    --epochs 3 \
    --batch-size 4 \
    --frame-sampling 5 \
    --max-samples 4096 \
    --max-steps 1000 \
    --num-workers 6 \
    --scheduler cosine \
    --warmup-steps 100 \
    --save-every-steps 500 \
    --log-every 20 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file ""

### 大batch

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --logdir ~/ltr/dd_logs/full_stage1 \
    --id spatial_path_bs16_lr1e-4 \
    --epochs 20 \
    --batch-size 16 \
    --frame-sampling 5 \
    --max-samples 4096 \
    --num-workers 6 \
    --scheduler cosine \
    --warmup-steps 500 \
    --min-lr 1e-6 \
    --save-every-steps 500 \
    --log-every 20 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file ""


### 小验证

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/HighwayCutIn \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/Accident \
    --route-glob "*" \
    --val-route-glob "*" \
    --logdir ~/ltr/dd_logs/full_val_probe \
    --id train_highway_val_accident \
    --epochs 1 \
    --batch-size 16 \
    --max-samples 1024 \
    --val-max-samples 512 \
    --max-steps 200 \
    --val-every-steps 50 \
    --max-val-steps 50 \
    --frame-sampling 5 \
    --val-frame-sampling 10 \
    --num-workers 6 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --scheduler none \
    --lr 1e-5 \
    --load-file /share/home/u19666033/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4/checkpoint_epoch019_step0005120.pth

python team_code/train_diffusiondrive.py \
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
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --resume-file ~/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4/latest.pth

for SCENE in HighwayCutIn InterurbanActorFlow NonSignalizedJunctionLeftTurn ParkingCutIn VehicleTurningRoute; do
    python team_code/train_diffusiondrive.py \
      --root-dir /share/home/u19666033/djy/carla_dataset \
      --val-root-dir /share/home/u19666033/djy/carla_dataset/$SCENE \
      --route-glob "*/*" \
      --val-route-glob "*" \
      --logdir ~/ltr/dd_logs/full_eval \
      --id ${SCENE}_eval \
      --eval-only \
      --batch-size 16 \
      --val-frame-sampling 10 \
      --val-max-samples 512 \
      --max-val-steps 50 \
      --num-workers 6 \
      --device cuda:0 \
      --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
      --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
      --resume-file ~/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4/latest.pth
  done

### 补训

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --route-glob "*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir ~/ltr/dd_logs/full_debug \
    --id nsj_left_finetune_probe \
    --epochs 5 \
    --batch-size 16 \
    --frame-sampling 5 \
    --val-frame-sampling 10 \
    --max-samples 2048 \
    --val-max-samples 512 \
    --val-every-steps 100 \
    --max-val-steps 32 \
    --num-workers 6 \
    --scheduler cosine \
    --warmup-steps 50 \
    --min-lr 1e-6 \
    --save-every-steps 500 \
    --log-every 20 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file ~/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4/latest.pth

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --logdir ~/ltr/dd_logs/full_stage2 \
    --id balanced_spatial_path_bs16_256ps \
    --epochs 20 \
    --batch-size 32 \
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
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file ~/ltr/dd_logs/full_stage1/spatial_path_bs16_lr1e-4/latest.pth

for SCENE in Accident HighwayCutIn InterurbanActorFlow NonSignalizedJunctionLeftTurn ParkingCutIn VehicleTurningRoute; do
    python team_code/train_diffusiondrive.py \
      --root-dir /share/home/u19666033/djy/carla_dataset \
      --val-root-dir /share/home/u19666033/djy/carla_dataset/$SCENE \
      --route-glob "*/*" \
      --val-route-glob "*" \
      --logdir ~/ltr/dd_logs/full_eval_stage2 \
      --id ${SCENE}_eval \
      --eval-only \
      --batch-size 16 \
      --val-frame-sampling 10 \
      --val-max-samples 512 \
      --max-val-steps 50 \
      --num-workers 6 \
      --device cuda:0 \
      --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
      --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
      --resume-file ~/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth

python tools/inspect_diffusiondrive_eval_errors.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --route-glob "*" \
    --checkpoint ~/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth \
    --top-k 50 \
    --max-samples 1024 \
    --frame-sampling 5 \
    --batch-size 16 \
    --num-workers 6 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --output-csv ~/ltr/dd_logs/full_eval_stage2/nsj_left_errors.csv

### stage3 

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --logdir ~/ltr/dd_logs/full_stage3 \
    --id hard_left_weight3_bs16_256ps \
    --epochs 5 \
    --batch-size 16 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 256 \
    --num-workers 6 \
    --scheduler cosine \
    --warmup-steps 200 \
    --min-lr 1e-6 \
    --save-every-steps 1000 \
    --log-every 50 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --resume-file ~/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth \
    --hard-left-turn-stop-loss-weight 3.0 \
    --dataset-stats-max-samples 4096

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --logdir ~/ltr/dd_logs/full_stage3 \
    --id hard_left_weight3_bs16_256ps \
    --epochs 5 \
    --batch-size 16 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 256 \
    --num-workers 6 \
    --scheduler cosine \
    --warmup-steps 200 \
    --min-lr 1e-6 \
    --save-every-steps 1000 \
    --log-every 50 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file ~/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth \
    --hard-left-turn-stop-loss-weight 5.0 \
    --dataset-stats-max-samples 4096


### stage3 long for night

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir /share/home/u19666033/ltr/dd_logs/full_stage3 \
    --id hard_left_weight5_bs32_512ps \
    --epochs 10 \
    --batch-size 64 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 512 \
    --num-workers 7 \
    --scheduler cosine \
    --warmup-steps 500 \
    --min-lr 1e-6 \
    --val-every-steps 1000 \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps 1000 \
    --log-every 50 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file /share/home/u19666033/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth \
    --hard-left-turn-stop-loss-weight 5.0 \
    --dataset-stats-max-samples 20000 \
    2>&1 | tee /share/home/u19666033/ltr/dd_logs/full_stage3/hard_left_weight5_bs32_512ps_train.log

eval 

python tools/inspect_diffusiondrive_eval_errors.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --route-glob "*" \
    --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage3/hard_left_weight5_bs32_512ps/latest.pth \
    --top-k 50 \
    --max-samples 1024 \
    --frame-sampling 5 \
    --batch-size 64 \
    --num-workers 6 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --output-csv /share/home/u19666033/ltr/dd_logs/full_stage3/hard_left_weight5_bs32_512ps/nsj_left_errors.csv


python tools/inspect_diffusiondrive_eval_errors.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/HighwayCutIn \
    --route-glob "*" \
    --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage3/hard_left_weight5_bs32_512ps/latest.pth \
    --top-k 50 \
    --max-samples 1024 \
    --frame-sampling 5 \
    --batch-size 64 \
    --num-workers 6 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --output-csv /share/home/u19666033/ltr/dd_logs/full_stage3/hard_left_weight5_bs32_512ps/weight5_bs32_512_HighwayCutIn.csv

### stage4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir /share/home/u19666033/ltr/dd_logs/full_stage4 \
    --id hard_left_weight5_bs64_1024ps_manifest \
    --epochs 15 \
    --batch-size 64 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 1024 \
    --num-workers 6 \
    --prefetch-factor 2 \
    --scheduler cosine \
    --warmup-steps 800 \
    --min-lr 1e-6 \
    --val-every-steps 3000 \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps 1500 \
    --log-every 50 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file /share/home/u19666033/ltr/dd_logs/full_stage2/balanced_spatial_path_bs16_256ps/latest.pth \
    --hard-left-turn-stop-loss-weight 5.0 \
    --dataset-stats-max-samples 0 \
    --sample-manifest /share/home/u19666033/ltr/dd_cache/full_stage4_train_1024ps_fs5_spatial.jsonl \
    --val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl \
    2>&1 | tee /share/home/u19666033/ltr/dd_logs/full_stage4/hard_left_weight5_bs64_1024ps_manifest_train.log

python tools/inspect_diffusiondrive_eval_errors.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --route-glob "*" \
    --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage4/hard_left_weight5_bs64_1024ps_manifest/latest.pth \
    --top-k 50 \
    --max-samples 1024 \
    --frame-sampling 5 \
    --batch-size 64 \
    --num-workers 6 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --output-csv /share/home/u19666033/ltr/dd_logs/full_stage4/nsj_left_weight5_bs64_1024ps.csv

python tools/inspect_diffusiondrive_eval_errors.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/InterurbanActorFlow \
    --route-glob "*" \
    --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage4/hard_left_weight5_bs64_1024ps_manifest/latest.pth \
    --top-k 50 \
    --max-samples 1024 \
    --frame-sampling 5 \
    --batch-size 64 \
    --num-workers 6 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --output-csv /share/home/u19666033/ltr/dd_logs/full_stage4/InterurbanActorFlow_left_weight5_bs64_1024ps.csv

for SCENE in Accident HighwayCutIn InterurbanActorFlow NonSignalizedJunctionLeftTurn ParkingCutIn VehicleTurningRoute; do
    python tools/inspect_diffusiondrive_eval_errors.py \
        --root-dir /share/home/u19666033/djy/carla_dataset/$SCENE \
        --route-glob "*" \
        --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage3/hard_left_weight5_bs32_512ps/latest.pth \
        --top-k 50 \
        --max-samples 1024 \
        --frame-sampling 5 \
        --batch-size 64 \
        --num-workers 6 \
        --device cuda:0 \
        --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
        --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
        --output-csv /share/home/u19666033/ltr/dd_logs/full_stage3/hard_left_weight5_bs32_512ps/weight5_bs32_512_${SCENE}.csv
done

### stage5

mkdir -p /share/home/u19666033/ltr/dd_logs/full_stage5

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir /share/home/u19666033/ltr/dd_logs/full_stage5 \
    --id stage4init_weight3_bs64_2048ps_manifest_lr1e-5 \
    --epochs 10 \
    --batch-size 64 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 2048 \
    --num-workers 6 \
    --prefetch-factor 2 \
    --scheduler cosine \
    --warmup-steps 500 \
    --min-lr 1e-6 \
    --val-every-steps 3000 \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps 1500 \
    --log-every 50 \
    --lr 1e-5 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file /share/home/u19666033/ltr/dd_logs/full_stage4/hard_left_weight5_bs64_1024ps_manifest/latest.pth \
    --hard-left-turn-stop-loss-weight 3.0 \
    --dataset-stats-max-samples 0 \
    --sample-manifest /share/home/u19666033/ltr/dd_cache/full_stage5_train_2048ps_fs5_spatial.jsonl \
    --val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl \
    2>&1 | tee /share/home/u19666033/ltr/dd_logs/full_stage5/stage4init_weight3_bs64_2048ps_manifest_lr1e-5_train.log

for SCENE in Accident HighwayCutIn InterurbanActorFlow NonSignalizedJunctionLeftTurn ParkingCutIn VehicleTurningRoute; do
    python tools/inspect_diffusiondrive_eval_errors.py \
        --root-dir /share/home/u19666033/djy/carla_dataset/$SCENE \
        --route-glob "*" \
        --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage5/stage4init_weight3_bs64_2048ps_manifest_lr1e-5/latest.pth \
        --top-k 50 \
        --max-samples 1024 \
        --frame-sampling 5 \
        --batch-size 64 \
        --num-workers 6 \
        --device cuda:0 \
        --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
        --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
        --output-csv /share/home/u19666033/ltr/dd_logs/full_stage5/stage4init_weight3_bs64_2048ps_manifest_lr1e-5/stage4init_weight3_bs64_2048ps_manifest_lr1e-5_${SCENE}.csv
done

### cpu缓存

CPU 作业里先跑 train manifest：

python tools/build_diffusiondrive_manifest.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --output-manifest /share/home/u19666033/ltr/dd_cache/full_stage5_train_2048ps_fs5_spatial.jsonl \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 2048 \
    --num-workers 16 \
    --rebuild \
    --verify-load

再跑 val manifest：

python tools/build_diffusiondrive_manifest.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --route-glob "*" \
    --output-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl \
    --frame-sampling 5 \
    --max-samples 1024 \
    --num-workers 16 \
    --rebuild \
    --verify-load

### stage6

mkdir -p /share/home/u19666033/ltr/dd_logs/full_stage6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir /share/home/u19666033/ltr/dd_logs/full_stage6 \
    --id stage5init_weight1_bs64_2048ps_lr5e-6 \
    --epochs 6 \
    --batch-size 64 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 2048 \
    --num-workers 6 \
    --prefetch-factor 2 \
    --scheduler cosine \
    --warmup-steps 300 \
    --min-lr 1e-6 \
    --val-every-steps 3000 \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps 1500 \
    --log-every 50 \
    --lr 5e-6 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file /share/home/u19666033/ltr/dd_logs/full_stage5/stage4init_weight3_bs64_2048ps_manifest_lr1e-5/latest.pth \
    --hard-left-turn-stop-loss-weight 1.0 \
    --dataset-stats-max-samples 0 \
    --sample-manifest /share/home/u19666033/ltr/dd_cache/full_stage5_train_2048ps_fs5_spatial.jsonl \
    --val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl \
    2>&1 | tee /share/home/u19666033/ltr/dd_logs/full_stage6/stage5init_weight1_bs64_2048ps_lr5e-6_train.log


for SCENE in Accident HighwayCutIn InterurbanActorFlow NonSignalizedJunctionLeftTurn ParkingCutIn VehicleTurningRoute; do
    python tools/inspect_diffusiondrive_eval_errors.py \
        --root-dir /share/home/u19666033/djy/carla_dataset/$SCENE \
        --route-glob "*" \
        --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage6/stage5init_weight1_bs64_2048ps_lr5e-6/latest.pth \
        --top-k 50 \
        --max-samples 1024 \
        --frame-sampling 5 \
        --batch-size 64 \
        --num-workers 6 \
        --device cuda:0 \
        --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
        --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
        --output-csv /share/home/u19666033/ltr/dd_logs/full_stage6/stage5init_weight1_bs64_2048ps_lr5e-6/stage5init_weight1_bs64_2048ps_lr5e-6_${SCENE}.csv
done

CUDA_VISIBLE_DEVICES=0 python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir /share/home/u19666033/ltr/dd_logs/full_stage6 \
    --id stage5init_weight2_bs64_2048ps_lr5e-6 \
    --hard-left-turn-stop-loss-weight 2.0 \
    --epochs 6 \
    --batch-size 64 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --max-samples-per-scenario 2048 \
    --num-workers 6 \
    --prefetch-factor 2 \
    --scheduler cosine \
    --warmup-steps 300 \
    --min-lr 1e-6 \
    --val-every-steps 3000 \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps 1500 \
    --log-every 50 \
    --lr 5e-6 \
    --weight-decay 1e-4 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file /share/home/u19666033/ltr/dd_logs/full_stage5/stage4init_weight3_bs64_2048ps_manifest_lr1e-5/latest.pth \
    --dataset-stats-max-samples 0 \
    --sample-manifest /share/home/u19666033/ltr/dd_cache/full_stage5_train_2048ps_fs5_spatial.jsonl \
    --val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial.jsonl \
    2>&1 | tee /share/home/u19666033/ltr/dd_logs/full_stage6/stage5init_weight2_bs64_2048ps_lr5e-6.log

for SCENE in Accident HighwayCutIn InterurbanActorFlow NonSignalizedJunctionLeftTurn ParkingCutIn VehicleTurningRoute; do
    python tools/inspect_diffusiondrive_eval_errors.py \
        --root-dir /share/home/u19666033/djy/carla_dataset/$SCENE \
        --route-glob "*" \
        --checkpoint /share/home/u19666033/ltr/dd_logs/full_stage6/stage5init_weight2_bs64_2048ps_lr5e-6/latest.pth \
        --top-k 50 \
        --max-samples 1024 \
        --frame-sampling 5 \
        --batch-size 64 \
        --num-workers 6 \
        --device cuda:0 \
        --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
        --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
        --output-csv /share/home/u19666033/ltr/dd_logs/full_stage6/stage5init_weight2_bs64_2048ps_lr5e-6/stage5init_weight2_bs64_2048ps_lr5e-6_${SCENE}.csv
done


### 全量缓存

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

python tools/build_diffusiondrive_manifest.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --output-manifest /share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl \
    --frame-sampling 5 \
    --balanced-scenarios \
    --num-workers 32 \
    --rebuild \
    --verify-load

python tools/build_diffusiondrive_manifest.py \
    --root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --route-glob "*" \
    --output-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl \
    --frame-sampling 5 \
    --max-samples 1024 \
    --balanced-scenarios \
    --num-workers 16 \
    --rebuild \
    --verify-load


### full

CUDA_VISIBLE_DEVICES=0 python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir /share/home/u19666033/ltr/dd_logs/full_baseline_basic \
    --id origlike_bs64_lr6e-4_ep100_fs5_spatial \
    --epochs 100 \
    --batch-size 64 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --num-workers 4 \
    --prefetch-factor 2 \
    --scheduler cosine \
    --warmup-steps $WARMUP_STEPS \
    --min-lr 1e-6 \
    --val-every-steps $STEPS_PER_EPOCH \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps $STEPS_PER_EPOCH \
    --log-every 50 \
    --lr 6e-4 \
    --weight-decay 1e-4 \
    --image-encoder-lr-mult 0.5 \
    --grad-clip-norm 0 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file "" \
    --hard-left-turn-stop-loss-weight 1.0 \
    --dataset-stats-max-samples 4096 \
    --sample-manifest $TRAIN_MANIFEST \
    --val-sample-manifest /share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl \
    2>&1 | tee /share/home/u19666033/ltr/dd_logs/full_baseline_basic/origlike_bs64_lr6e-4_ep100_fs5_spatial_train.log


在 carla_garage 根目录运行。下面是假设你已经缓存了这两个 manifest：

TRAIN_MANIFEST=/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl
VAL_MANIFEST=/share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl

完整命令：

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

TRAIN_MANIFEST=/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl
VAL_MANIFEST=/share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl
LOGDIR=/share/home/u19666033/ltr/dd_logs/full_baseline_basic
RUN_ID=origlike_bs64_lr6e-4_ep100_fs5_spatial_imgenc0p5

mkdir -p ${LOGDIR}

export TRAIN_MANIFEST=/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl

SAMPLES=$(python -c 'import json, os; p=os.environ["TRAIN_MANIFEST"]; print(sum(1 for l in open(p) if l.strip() and
json.loads(l).get("type") not in ("metadata",)))')


STEPS_PER_EPOCH=$(( (SAMPLES + 64 - 1) / 64 ))
WARMUP_STEPS=$(( STEPS_PER_EPOCH * 3 ))


echo samples=${SAMPLES}
echo steps_per_epoch=${STEPS_PER_EPOCH}
echo warmup_steps=${WARMUP_STEPS}

CUDA_VISIBLE_DEVICES=0 python team_code/train_diffusiondrive.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir ${LOGDIR} \
    --id ${RUN_ID} \
    --epochs 100 \
    --batch-size 64 \
    --frame-sampling 5 \
    --balanced-scenarios \
    --num-workers 16 \
    --prefetch-factor 2 \
    --scheduler cosine \
    --warmup-steps ${WARMUP_STEPS} \
    --min-lr 1e-6 \
    --val-every-steps ${STEPS_PER_EPOCH} \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps ${STEPS_PER_EPOCH} \
    --log-every 50 \
    --lr 6e-4 \
    --weight-decay 1e-4 \
    --image-encoder-lr-mult 0.5 \
    --grad-clip-norm 0 \
    --device cuda:0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file "" \
    --hard-left-turn-stop-loss-weight 1.0 \
    --dataset-stats-max-samples 4096 \
    --sample-manifest ${TRAIN_MANIFEST} \
    --val-sample-manifest ${VAL_MANIFEST} \
    2>&1 | tee ${LOGDIR}/${RUN_ID}_train.log

如果你的 train manifest 构建时用了：

--max-samples-per-scenario 4096

那训练命令里也必须补同一行：

--max-samples-per-scenario 4096 \

否则 manifest header 校验会报错。


### 4GPU

可以。下面按 4 卡 DDP、原版对齐方向 写：batch-size=64 是每卡 batch，所以 global batch 是 64 * 4 = 256。

先申请资源时建议：

srun -p L40 -J dd_baseline_basic_4gpu -N 1 -n 1 \
    --gres=gpu:l40:4 \
    --cpus-per-task=28 \
    --mem=256G \
    --pty /bin/bash

进入作业后，在 carla_garage 目录运行完整训练命令：

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

export TRAIN_MANIFEST=/share/home/u19666033/ltr/dd_cache/full_baseline_basic_train_all_fs5_spatial.jsonl
export VAL_MANIFEST=/share/home/u19666033/ltr/dd_cache/nsj_left_val_1024_fs5_spatial.jsonl
export LOGDIR=/share/home/u19666033/ltr/dd_logs/full_baseline_basic
export RUN_ID=origlike_ddp4_bs64x4_lr6e-4_ep100_fs5_spatial_imgenc0p5
export NPROC_PER_NODE=4
export PER_GPU_BATCH=64
export GLOBAL_BATCH=$(( PER_GPU_BATCH * NPROC_PER_NODE ))

mkdir -p ${LOGDIR}

SAMPLES=$(python -c 'import json, os; p=os.environ["TRAIN_MANIFEST"]; print(sum(1 for l in open(p) if l.strip() and
json.loads(l).get("type") not in ("metadata",)))')
STEPS_PER_EPOCH=$(( (SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
WARMUP_STEPS=$(( STEPS_PER_EPOCH * 3 ))

echo samples=${SAMPLES}
echo global_batch=${GLOBAL_BATCH}
echo steps_per_epoch=${STEPS_PER_EPOCH}
echo warmup_steps=${WARMUP_STEPS}

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} \
team_code/train_diffusiondrive.py \
    --distributed ddp \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --val-root-dir /share/home/u19666033/djy/carla_dataset/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir ${LOGDIR} \
    --id ${RUN_ID} \
    --epochs 100 \
    --batch-size ${PER_GPU_BATCH} \
    --frame-sampling 5 \
    --balanced-scenarios \
    --num-workers 12 \
    --prefetch-factor 2 \
    --scheduler cosine \
    --warmup-steps ${WARMUP_STEPS} \
    --min-lr 1e-6 \
    --val-every-steps ${STEPS_PER_EPOCH} \
    --val-frame-sampling 5 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps ${STEPS_PER_EPOCH} \
    --log-every 50 \
    --lr 6e-4 \
    --weight-decay 1e-4 \
    --image-encoder-lr-mult 0.5 \
    --grad-clip-norm 0 \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file "" \
    --hard-left-turn-stop-loss-weight 1.0 \
    --dataset-stats-max-samples 4096 \
    --sample-manifest ${TRAIN_MANIFEST} \
    --val-sample-manifest ${VAL_MANIFEST} \
    2>&1 | tee ${LOGDIR}/${RUN_ID}_train.log

关键点：

- --batch-size 64 是每卡 batch，不是 global batch。
- STEPS_PER_EPOCH 要按 SAMPLES / (64 * 4) 算。
- --num-workers 4 也是每个 DDP 进程 4 个 worker，总共约 16 个 worker，配 28 CPU 核比较稳。
- 如果想保持 global batch 仍为 64，而不是 256，把：

export PER_GPU_BATCH=16
但更接近原版 DDP 训练习惯的是每卡 64。


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

TRAIN_MANIFEST=/share/home/u19666033/ltr/dd_cache/full_condition_v1_train_soft_clean_fs5_spatial.jsonl

python tools/build_diffusiondrive_manifest.py \
    --root-dir /share/home/u19666033/djy/carla_dataset \
    --route-glob "*/*" \
    --output-manifest ${TRAIN_MANIFEST} \
    --frame-sampling 5 \
    --target-mode spatial_path \
    --balanced-scenarios \
    --quality-filter soft_clean \
    --num-workers 16 \
    --rebuild \
    --verify-load

如果之后想做 syb-clean 对照，只改两处：

TRAIN_MANIFEST=/share/home/u19666033/ltr/dd_cache/full_condition_v1_train_syb_clean_fs5_spatial.jsonl


并把参数改成：
--quality-filter syb_clean

### full soft clean

Manifest 缓存

cd ~/ltr/carla_garage
conda activate ltr_garage_2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

export DATA_ROOT=/share/home/u19666033/djy/carla_dataset
export CACHE_DIR=/share/home/u19666033/ltr/dd_cache
mkdir -p ${CACHE_DIR}

export TRAIN_MANIFEST=${CACHE_DIR}/full_condition_v1_train_soft_clean_fs5_skip25_spatial.jsonl
export VAL_MANIFEST=${CACHE_DIR}/nsj_left_val_fs5_spatial_skip25.jsonl

python tools/build_diffusiondrive_manifest.py \
    --root-dir ${DATA_ROOT} \
    --route-glob "*/*" \
    --output-manifest ${TRAIN_MANIFEST} \
    --frame-sampling 5 \
    --skip-first-frames 25 \
    --target-mode spatial_path \
    --balanced-scenarios \
    --quality-filter soft_clean \
    --num-workers 16 \
    --rebuild \
    --verify-load

python tools/build_diffusiondrive_manifest.py \
    --root-dir ${DATA_ROOT}/NonSignalizedJunctionLeftTurn \
    --route-glob "*" \
    --output-manifest ${VAL_MANIFEST} \
    --frame-sampling 5 \
    --skip-first-frames 25 \
    --target-mode spatial_path \
    --max-samples 1024 \
    --balanced-scenarios \
    --num-workers 16 \
    --rebuild \
    --verify-load

4 卡 Full Retrain

srun -p L40 -J dd_condv1_4gpu -N 1 -n 1 \
    --gres=gpu:l40:4 \
    --cpus-per-task=28 \
    --mem=256G \
    --pty /bin/bash

进入作业后：

cd ~/ltr/carla_garage
conda activate ltr_garage_2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_NUM_THREADS=1

export DATA_ROOT=/share/home/u19666033/djy/carla_dataset
export TRAIN_MANIFEST=/share/home/u19666033/ltr/dd_cache/full_condition_v1_train_soft_clean_fs5_skip25_spatial.jsonl
export VAL_MANIFEST=/share/home/u19666033/ltr/dd_cache/nsj_left_val_fs5_spatial_skip25.jsonl

export LOGDIR=/share/home/u19666033/ltr/dd_logs/full_baseline_condition_v1
export RUN_ID=soft_clean_skip25_imgnet_ddp4_bs64x4_lr6e-4_ep100
mkdir -p ${LOGDIR}

export NPROC_PER_NODE=4
export PER_GPU_BATCH=64
export GLOBAL_BATCH=$((NPROC_PER_NODE * PER_GPU_BATCH))
export SAMPLES=$(python -c 'import json, os; p=os.environ["TRAIN_MANIFEST"]; print(sum(1 for l in open(p) if l.strip() and
json.loads(l).get("type") != "metadata"))')
export STEPS_PER_EPOCH=$(( (SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
export WARMUP_STEPS=$(( STEPS_PER_EPOCH * 3 ))

echo samples=${SAMPLES}
echo global_batch=${GLOBAL_BATCH}
echo steps_per_epoch=${STEPS_PER_EPOCH}
echo warmup_steps=${WARMUP_STEPS}

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} \
team_code/train_diffusiondrive.py \
    --distributed ddp \
    --root-dir ${DATA_ROOT} \
    --route-glob "*/*" \
    --val-root-dir ${DATA_ROOT}/NonSignalizedJunctionLeftTurn \
    --val-route-glob "*" \
    --logdir ${LOGDIR} \
    --id ${RUN_ID} \
    --epochs 100 \
    --batch-size ${PER_GPU_BATCH} \
    --frame-sampling 5 \
    --skip-first-frames 25 \
    --balanced-scenarios \
    --num-workers 12 \
    --prefetch-factor 2 \
    --persistent-workers \
    --scheduler cosine \
    --warmup-steps ${WARMUP_STEPS} \
    --min-lr 1e-6 \
    --val-every-steps ${STEPS_PER_EPOCH} \
    --val-frame-sampling 5 \
    --val-skip-first-frames 25 \
    --val-max-samples 1024 \
    --max-val-steps 64 \
    --save-every-steps ${STEPS_PER_EPOCH} \
    --log-every 50 \
    --lr 6e-4 \
    --weight-decay 1e-4 \
    --image-encoder-lr-mult 0.5 \
    --grad-clip-norm 0 \
    --model-image-height 384 \
    --model-image-width 1024 \
    --image-normalization imagenet \
    --anchor-path /share/home/u19666033/ltr/4-0-0-1910-tracked_clusters_anchor.npy \
    --backbone-path /share/home/u19666033/ltr/pytorch_model.bin \
    --load-file "" \
    --hard-left-turn-stop-loss-weight 1.0 \
    --dataset-stats-max-samples 4096 \
    --sample-manifest ${TRAIN_MANIFEST} \
    --val-sample-manifest ${VAL_MANIFEST} \
    2>&1 | tee ${LOGDIR}/${RUN_ID}_train.log

