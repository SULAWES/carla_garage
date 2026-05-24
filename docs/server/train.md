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

吞吐量测试

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

1000step

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

大batch

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


小验证

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

补训

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

stage3 

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


stage3 long for night

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
