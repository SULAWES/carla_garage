#!/usr/bin/env python3
"""Smoke-test the CARLA DiffusionDrive model on one Bench2Drive mini frame.

This is not a training script. It only verifies that a raw Bench2Drive mini
sample can be converted into the current CARLA-native DiffusionDrive feature
contract and reach forward/loss/backward.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_CODE = REPO_ROOT / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from config import GlobalConfig  # noqa: E402
from diffusiondrive.carla_native_dataset import (  # noqa: E402
    TARGET_MODE_FUTURE_EGO_TIME,
    TARGET_MODE_SPATIAL_PATH,
    build_camera_feature,
    build_lidar_feature,
    build_route_condition_feature,
    build_status_feature,
    build_target_speed_metadata,
    build_trajectory_target,
    load_annotation,
)
from diffusiondrive.config_adapter import (  # noqa: E402
    DiffusionDriveRuntimeOverrides,
    build_diffusiondrive_config,
)
from diffusiondrive.model import V2TransfuserModel  # noqa: E402
from diffusiondrive.inference_diagnostics import (  # noqa: E402
    DIFFUSION_INFER_NOISE_MODES,
    gaussian_noise_from_seed,
    stable_diffusion_noise_seed,
)


def _first_nonzero_grad_norm(model: torch.nn.Module) -> float:
    for parameter in model.parameters():
        if parameter.grad is not None:
            return float(parameter.grad.detach().norm().cpu())
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mini-root",
        type=Path,
        default=REPO_ROOT / "Bench2Drive" / "Bench2Drive-mini-extracted",
    )
    parser.add_argument("--route", default="AccidentTwoWays_Town12_Route1444_Weather0")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--future-stride", type=int, default=10)
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_SPATIAL_PATH, TARGET_MODE_FUTURE_EGO_TIME),
        default=TARGET_MODE_SPATIAL_PATH,
    )
    parser.add_argument("--spatial-target-first-distance", type=float, default=2.5)
    parser.add_argument("--spatial-target-interval", type=float, default=1.0)
    parser.add_argument("--spatial-target-max-future-frames", type=int, default=120)
    parser.add_argument(
        "--diffusion-noise-mode",
        choices=DIFFUSION_INFER_NOISE_MODES,
        default="random",
    )
    parser.add_argument("--diffusion-noise-seed", type=int, default=0)
    parser.add_argument(
        "--anchor-path",
        default=os.environ.get(
            "DIFFUSIONDRIVE_ANCHOR_PATH",
            "/home/HeavenlySU/sitp_workspace/4-0-0-1910-tracked_clusters_anchor.npy",
        ),
    )
    parser.add_argument(
        "--backbone-path",
        default=os.environ.get(
            "DIFFUSIONDRIVE_BACKBONE_PATH",
            "/home/HeavenlySU/sitp_workspace/pytorch_model.bin",
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)

    route_dir = args.mini_root / args.route
    config = GlobalConfig()
    dd_config = build_diffusiondrive_config(
        config,
        DiffusionDriveRuntimeOverrides(
            anchor_path=args.anchor_path,
            backbone_path=args.backbone_path,
            diffusion_infer_noise_mode=args.diffusion_noise_mode,
            diffusion_infer_noise_seed=args.diffusion_noise_seed,
        ),
    )

    annotation = load_annotation(route_dir, args.frame)
    camera_feature = build_camera_feature(route_dir, args.frame, config).unsqueeze(0)
    lidar_feature = build_lidar_feature(route_dir, args.frame, config).unsqueeze(0)
    status_feature = build_status_feature(annotation).unsqueeze(0)
    route_condition_feature = build_route_condition_feature(annotation).unsqueeze(0)
    speed_target = build_target_speed_metadata(annotation)
    trajectory = build_trajectory_target(
        route_dir,
        args.frame,
        dd_config.trajectory_sampling.num_poses,
        args.future_stride,
        target_mode=args.target_mode,
        spatial_first_distance=args.spatial_target_first_distance,
        spatial_interval=args.spatial_target_interval,
        spatial_max_future_frames=args.spatial_target_max_future_frames,
    ).unsqueeze(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = V2TransfuserModel(dd_config).to(device)
    model.train()

    features = {
        "camera_feature": camera_feature.to(device),
        "lidar_feature": lidar_feature.to(device),
        "status_feature": status_feature.to(device),
        "route_condition_feature": route_condition_feature.to(device),
    }
    targets = {
        "trajectory": trajectory.to(device),
        "target_speed_twohot": torch.tensor(
            [speed_target["target_speed_twohot"]],
            dtype=torch.float32,
            device=device,
        ),
        "target_speed_label_valid": torch.tensor(
            [speed_target["target_speed_label_valid"]],
            dtype=torch.float32,
            device=device,
        ),
    }

    outputs = model(features, targets=targets)
    loss = outputs["trajectory_loss"]
    if "target_speed_logits" in outputs:
        valid = targets["target_speed_label_valid"].clamp_min(0.0)
        valid_count = valid.sum().clamp_min(1.0)
        speed_loss = -(
            targets["target_speed_twohot"] * F.log_softmax(outputs["target_speed_logits"], dim=-1)
        ).sum(dim=-1)
        loss = loss + (speed_loss * valid).sum() / valid_count
    loss.backward()

    inference_features = dict(features)
    if args.diffusion_noise_mode == "seeded":
        noise_key = stable_diffusion_noise_seed(
            args.diffusion_noise_seed,
            route_dir.name,
            args.frame,
        )
        inference_features["diffusion_noise"] = gaussian_noise_from_seed(
            (
                1,
                dd_config.num_anchor_modes,
                dd_config.trajectory_sampling.num_poses,
                2,
            ),
            noise_key,
            device=device,
        )

    model.eval()
    with torch.inference_mode():
        inference_output = model(inference_features)
        repeated_max_abs_diff = None
        if args.diffusion_noise_mode != "random":
            repeated_output = model(inference_features)
            repeated_max_abs_diff = float(
                (inference_output["trajectory"] - repeated_output["trajectory"])
                .abs()
                .max()
                .cpu()
            )
            if repeated_max_abs_diff > 1e-6:
                raise RuntimeError(
                    "Deterministic inference smoke produced different trajectories: "
                    f"max_abs_diff={repeated_max_abs_diff}."
                )

    print("mini_smoke_ok")
    print("route", args.route)
    print("frame", args.frame)
    print("target_mode", args.target_mode)
    print("diffusion_noise_mode", args.diffusion_noise_mode)
    print("diffusion_noise_seed", args.diffusion_noise_seed)
    print("device", device)
    print("camera_feature", tuple(features["camera_feature"].shape))
    print("lidar_feature", tuple(features["lidar_feature"].shape))
    print("status_feature", tuple(features["status_feature"].shape))
    print("route_condition_feature", tuple(features["route_condition_feature"].shape))
    print("target_trajectory", tuple(targets["trajectory"].shape))
    print("target_speed_valid", int(speed_target["target_speed_label_valid"]))
    print("target_speed_class", int(speed_target["target_speed_class"]))
    print("output_trajectory", tuple(outputs["trajectory"].shape))
    print("inference_trajectory", tuple(inference_output["trajectory"].shape))
    print("inference_mode_index", int(inference_output["trajectory_mode_index"][0].cpu()))
    print("inference_mode_margin", float(inference_output["trajectory_mode_margin"][0].cpu()))
    print(
        "inference_mode_entropy_normalized",
        float(inference_output["trajectory_mode_entropy_normalized"][0].cpu()),
    )
    if repeated_max_abs_diff is not None:
        print("inference_repeat_max_abs_diff", repeated_max_abs_diff)
    if "target_speed_logits" in outputs:
        print("target_speed_logits", tuple(outputs["target_speed_logits"].shape))
    print("loss", float(loss.detach().cpu()))
    print("grad_norm", _first_nonzero_grad_norm(model))
    for name, value in outputs.get("trajectory_loss_dict", {}).items():
        print(name, float(value.detach().cpu()))


if __name__ == "__main__":
    main()
