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


REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_CODE = REPO_ROOT / "team_code"
if str(TEAM_CODE) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE))

from config import GlobalConfig  # noqa: E402
from diffusiondrive.carla_native_dataset import (  # noqa: E402
    build_camera_feature,
    build_lidar_feature,
    build_status_feature,
    build_trajectory_target,
    load_annotation,
)
from diffusiondrive.config_adapter import (  # noqa: E402
    DiffusionDriveRuntimeOverrides,
    build_diffusiondrive_config,
)
from diffusiondrive.model import V2TransfuserModel  # noqa: E402


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
        DiffusionDriveRuntimeOverrides(anchor_path=args.anchor_path, backbone_path=args.backbone_path),
    )

    annotation = load_annotation(route_dir, args.frame)
    camera_feature = build_camera_feature(route_dir, args.frame, config).unsqueeze(0)
    lidar_feature = build_lidar_feature(route_dir, args.frame, config).unsqueeze(0)
    status_feature = build_status_feature(annotation).unsqueeze(0)
    trajectory = build_trajectory_target(
        route_dir,
        args.frame,
        dd_config.trajectory_sampling.num_poses,
        args.future_stride,
    ).unsqueeze(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = V2TransfuserModel(dd_config).to(device)
    model.train()

    features = {
        "camera_feature": camera_feature.to(device),
        "lidar_feature": lidar_feature.to(device),
        "status_feature": status_feature.to(device),
    }
    targets = {"trajectory": trajectory.to(device)}

    outputs = model(features, targets=targets)
    loss = outputs["trajectory_loss"]
    loss.backward()

    print("mini_smoke_ok")
    print("route", args.route)
    print("frame", args.frame)
    print("device", device)
    print("camera_feature", tuple(features["camera_feature"].shape))
    print("lidar_feature", tuple(features["lidar_feature"].shape))
    print("status_feature", tuple(features["status_feature"].shape))
    print("target_trajectory", tuple(targets["trajectory"].shape))
    print("output_trajectory", tuple(outputs["trajectory"].shape))
    print("loss", float(loss.detach().cpu()))
    print("grad_norm", _first_nonzero_grad_norm(model))
    for name, value in outputs.get("trajectory_loss_dict", {}).items():
        print(name, float(value.detach().cpu()))


if __name__ == "__main__":
    main()
