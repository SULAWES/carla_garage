"""CARLA-native DiffusionDrive dataset helpers."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import laspy
import numpy as np
import torch
from torch.utils.data import Dataset

from config import GlobalConfig
from diffusiondrive.status import build_status_feature_from_command
import transfuser_utils as t_u

TARGET_MODE_SPATIAL_PATH = "spatial_path"
TARGET_MODE_FUTURE_EGO_TIME = "future_ego_time"
TARGET_SPEED_LABEL_SCHEMA = "syb_twohot_target_speed_v1"
TARGET_SPEED_CLASSES_MPS = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)


class Bench2DriveDiffusionDataset(Dataset):
    """Raw Bench2Drive route dataset for the CARLA DiffusionDrive input contract."""

    def __init__(
        self,
        root_dirs: Iterable[str | Path],
        config: GlobalConfig,
        num_poses: int,
        future_stride: int = 10,
        frame_sampling: int = 1,
        max_samples: Optional[int] = None,
        route_glob: str = "*",
        model_image_size: tuple[int, int] = (256, 1024),
        jpeg_artifact: bool = True,
        target_mode: str = TARGET_MODE_SPATIAL_PATH,
        spatial_target_first_distance: float = 2.5,
        spatial_target_interval: float = 1.0,
        spatial_target_max_future_frames: int = 120,
        balanced_scenarios: bool = False,
        max_samples_per_scenario: Optional[int] = None,
        hard_left_turn_stop_loss_weight: float = 1.0,
        hard_left_turn_command: int = 1,
        hard_left_turn_speed_threshold: float = 0.1,
        hard_left_turn_y_threshold: float = 4.0,
        sample_manifest_path: Optional[str | Path] = None,
        rebuild_sample_manifest: bool = False,
    ) -> None:
        self.root_dirs = [Path(root) for root in root_dirs]
        self.config = config
        self.num_poses = num_poses
        self.future_stride = future_stride
        self.frame_sampling = frame_sampling
        self.max_samples = max_samples
        self.route_glob = route_glob
        self.model_image_size = model_image_size
        self.jpeg_artifact = jpeg_artifact
        self.target_mode = target_mode
        self.spatial_target_first_distance = spatial_target_first_distance
        self.spatial_target_interval = spatial_target_interval
        self.spatial_target_max_future_frames = spatial_target_max_future_frames
        self.balanced_scenarios = balanced_scenarios
        self.max_samples_per_scenario = max_samples_per_scenario
        self.hard_left_turn_stop_loss_weight = hard_left_turn_stop_loss_weight
        self.hard_left_turn_command = hard_left_turn_command
        self.hard_left_turn_speed_threshold = hard_left_turn_speed_threshold
        self.hard_left_turn_y_threshold = hard_left_turn_y_threshold
        self.sample_metadata: Optional[List[dict]] = None
        self.sample_manifest_path = Path(sample_manifest_path) if sample_manifest_path else None

        if self.sample_manifest_path and self.sample_manifest_path.is_file() and not rebuild_sample_manifest:
            self.samples, self.sample_metadata, self.scenario_sample_counts = self._load_sample_manifest(
                self.sample_manifest_path
            )
        else:
            self.samples, self.scenario_sample_counts = self._discover_samples(self.root_dirs, route_glob, max_samples)
            if self.sample_manifest_path:
                self.sample_metadata = self._write_sample_manifest(self.sample_manifest_path)

        if not self.samples:
            roots = ", ".join(str(root) for root in self.root_dirs)
            raise RuntimeError(f"No trainable Bench2Drive samples found under: {roots}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        cv2.setNumThreads(0)
        route_dir, frame, command, speed, trajectory, speed_target = self._sample_components(index)
        hard_case = is_hard_left_turn_stop_sample_from_values(
            command,
            speed,
            trajectory,
            command=self.hard_left_turn_command,
            speed_threshold=self.hard_left_turn_speed_threshold,
            y_threshold=self.hard_left_turn_y_threshold,
        )
        sample_weight = self.hard_left_turn_stop_loss_weight if hard_case else 1.0
        return {
            "features": {
                "camera_feature": build_camera_feature(
                    route_dir,
                    frame,
                    self.config,
                    model_image_size=self.model_image_size,
                    jpeg_artifact=self.jpeg_artifact,
                ),
                "lidar_feature": build_lidar_feature(route_dir, frame, self.config),
                "status_feature": build_status_feature_from_command(command, speed),
            },
            "targets": {
                "trajectory": trajectory,
                "trajectory_sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
                "hard_left_turn_stop": torch.tensor(float(hard_case), dtype=torch.float32),
                "target_speed": torch.tensor(speed_target["target_speed"], dtype=torch.float32),
                "target_speed_twohot": torch.tensor(speed_target["target_speed_twohot"], dtype=torch.float32),
                "target_speed_class": torch.tensor(float(speed_target["target_speed_class"]), dtype=torch.float32),
                "target_speed_label_valid": torch.tensor(float(speed_target["target_speed_label_valid"]), dtype=torch.float32),
                "brake": torch.tensor(float(speed_target["brake"]), dtype=torch.float32),
            },
            "route": route_dir.name,
            "frame": frame,
        }

    def get_sample_summary(self, index: int) -> dict:
        route_dir, frame, command, speed, trajectory, speed_target = self._sample_components(index)
        hard_case = is_hard_left_turn_stop_sample_from_values(
            command,
            speed,
            trajectory,
            command=self.hard_left_turn_command,
            speed_threshold=self.hard_left_turn_speed_threshold,
            y_threshold=self.hard_left_turn_y_threshold,
        )
        return {
            "route_dir": route_dir,
            "frame": frame,
            "command": command,
            "speed": speed,
            "trajectory": trajectory,
            "hard_left_turn_stop": hard_case,
            "target_speed": speed_target["target_speed"],
            "target_speed_class": speed_target["target_speed_class"],
            "target_speed_label_valid": speed_target["target_speed_label_valid"],
            "brake": speed_target["brake"],
        }

    def _sample_components(self, index: int) -> tuple[Path, int, int, float, torch.Tensor, dict]:
        route_dir, frame = self.samples[index]
        if self.sample_metadata is not None:
            metadata = self.sample_metadata[index]
            command = int(metadata["command"])
            speed = float(metadata["speed"])
            trajectory = torch.tensor(metadata["trajectory"], dtype=torch.float32)
            speed_target = target_speed_metadata_from_record(metadata, fallback_speed=speed)
            return route_dir, frame, command, speed, trajectory, speed_target

        annotation = load_annotation(route_dir, frame)
        trajectory = build_trajectory_target(
            route_dir,
            frame,
            self.num_poses,
            self.future_stride,
            target_mode=self.target_mode,
            spatial_first_distance=self.spatial_target_first_distance,
            spatial_interval=self.spatial_target_interval,
            spatial_max_future_frames=self.spatial_target_max_future_frames,
        )
        command = int(annotation.get("command_far", annotation.get("command_near", 4)))
        speed = float(annotation["speed"])
        speed_target = build_target_speed_metadata(annotation)
        return route_dir, frame, command, speed, trajectory, speed_target

    def _load_sample_manifest(self, manifest_path: Path) -> tuple[List[tuple[Path, int]], List[dict], dict[str, int]]:
        samples: List[tuple[Path, int]] = []
        metadata: List[dict] = []
        scenario_counts: dict[str, int] = {}
        header: Optional[dict] = None

        with manifest_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") == "metadata":
                    header = record
                    continue
                route_dir = Path(record["route_dir"])
                frame = int(record["frame"])
                samples.append((route_dir, frame))
                metadata.append({
                    "command": int(record["command"]),
                    "speed": float(record["speed"]),
                    "target_speed": float(record.get("target_speed", record["speed"])),
                    "brake": _annotation_bool(record.get("brake", False)),
                    "target_speed_class": int(record.get("target_speed_class", -1)),
                    "target_speed_twohot": record.get("target_speed_twohot"),
                    "target_speed_label_valid": int(record.get("target_speed_label_valid", 0)),
                    "trajectory": record["trajectory"],
                })
                scenario = route_dir.parent.name
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        self._validate_sample_manifest_header(manifest_path, header, len(samples))
        print(f"Loaded sample manifest: {manifest_path} ({len(samples)} samples)", flush=True)
        return samples, metadata, scenario_counts

    def _validate_sample_manifest_header(
        self,
        manifest_path: Path,
        header: Optional[dict],
        loaded_sample_count: int,
    ) -> None:
        if header is None:
            raise RuntimeError(f"Sample manifest is missing metadata header: {manifest_path}")
        if header.get("format") != "diffusiondrive_sample_manifest_v1":
            raise RuntimeError(
                f"Unsupported sample manifest format in {manifest_path}: {header.get('format')!r}"
            )

        expected = {
            "target_mode": self.target_mode,
            "num_poses": self.num_poses,
            "future_stride": self.future_stride,
            "spatial_target_first_distance": self.spatial_target_first_distance,
            "spatial_target_interval": self.spatial_target_interval,
            "spatial_target_max_future_frames": self.spatial_target_max_future_frames,
            "frame_sampling": self.frame_sampling,
            "max_samples": self.max_samples,
            "balanced_scenarios": self.balanced_scenarios,
            "max_samples_per_scenario": self.max_samples_per_scenario,
            "route_glob": self.route_glob,
        }
        mismatches = []
        for key, expected_value in expected.items():
            if key not in header:
                continue
            actual_value = header[key]
            if not _manifest_values_match(actual_value, expected_value):
                mismatches.append(f"{key}: manifest={actual_value!r} current={expected_value!r}")

        if "root_dir" in header:
            expected_roots = sorted(_normalize_manifest_root(root) for root in self.root_dirs)
            header_roots = header["root_dir"] if isinstance(header["root_dir"], list) else [header["root_dir"]]
            actual_roots = sorted(_normalize_manifest_root(root) for root in header_roots)
            if actual_roots != expected_roots:
                mismatches.append(f"root_dir: manifest={actual_roots!r} current={expected_roots!r}")

        if mismatches:
            details = "; ".join(mismatches)
            raise RuntimeError(
                f"Sample manifest does not match current dataset arguments: {manifest_path}. {details}. "
                "Use a matching manifest path or rebuild with --rebuild-sample-manifest."
            )

        declared_sample_count = header.get("sample_count")
        if declared_sample_count is not None and int(declared_sample_count) != loaded_sample_count:
            raise RuntimeError(
                f"Sample manifest sample_count mismatch in {manifest_path}: "
                f"header={declared_sample_count} loaded={loaded_sample_count}"
            )

    def _write_sample_manifest(self, manifest_path: Path) -> List[dict]:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_records: List[dict] = []
        header = {
            "type": "metadata",
            "format": "diffusiondrive_sample_manifest_v1",
            "root_dir": [str(root) for root in self.root_dirs],
            "route_glob": self.route_glob,
            "frame_sampling": self.frame_sampling,
            "max_samples": self.max_samples,
            "balanced_scenarios": self.balanced_scenarios,
            "max_samples_per_scenario": self.max_samples_per_scenario,
            "target_mode": self.target_mode,
            "num_poses": self.num_poses,
            "future_stride": self.future_stride,
            "spatial_target_first_distance": self.spatial_target_first_distance,
            "spatial_target_interval": self.spatial_target_interval,
            "spatial_target_max_future_frames": self.spatial_target_max_future_frames,
            "target_speed_label": target_speed_label_metadata(),
            "sample_count": len(self.samples),
        }

        with manifest_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(header, sort_keys=True) + "\n")
            for route_dir, frame in self.samples:
                annotation = load_annotation(route_dir, frame)
                trajectory = build_trajectory_target(
                    route_dir,
                    frame,
                    self.num_poses,
                    self.future_stride,
                    target_mode=self.target_mode,
                    spatial_first_distance=self.spatial_target_first_distance,
                    spatial_interval=self.spatial_target_interval,
                    spatial_max_future_frames=self.spatial_target_max_future_frames,
                )
                record = {
                    "route_dir": str(route_dir.resolve()),
                    "scenario": route_dir.parent.name,
                    "route": route_dir.name,
                    "frame": int(frame),
                    "command": int(annotation.get("command_far", annotation.get("command_near", 4))),
                    "speed": float(annotation["speed"]),
                    "trajectory": trajectory.tolist(),
                }
                record.update(build_target_speed_metadata(annotation))
                file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                metadata_records.append({
                    "command": record["command"],
                    "speed": record["speed"],
                    "target_speed": record["target_speed"],
                    "brake": record["brake"],
                    "target_speed_class": record["target_speed_class"],
                    "target_speed_twohot": record["target_speed_twohot"],
                    "target_speed_label_valid": record["target_speed_label_valid"],
                    "trajectory": record["trajectory"],
                })

        print(f"Wrote sample manifest: {manifest_path} ({len(metadata_records)} samples)", flush=True)
        return metadata_records

    def _discover_samples(
        self,
        root_dirs: Iterable[str | Path],
        route_glob: str,
        max_samples: Optional[int],
    ) -> tuple[List[tuple[Path, int]], dict[str, int]]:
        if self.balanced_scenarios:
            return self._discover_balanced_samples(root_dirs, route_glob, max_samples)

        samples: List[tuple[Path, int]] = []
        scenario_counts: dict[str, int] = {}
        for root in root_dirs:
            root_path = Path(root)
            if not root_path.exists():
                raise RuntimeError(f"Bench2Drive root does not exist: {root_path}")
            for route_dir in sorted(root_path.glob(route_glob)):
                if not _is_b2d_route(route_dir):
                    continue
                frames = sorted(int(path.stem.split(".")[0]) for path in _annotation_dir(route_dir).glob("*.json.gz"))
                for frame in frames[:: self.frame_sampling]:
                    if self._has_required_files(route_dir, frame):
                        samples.append((route_dir, frame))
                        scenario = _scenario_name(route_dir)
                        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                        if max_samples is not None and len(samples) >= max_samples:
                            return samples, scenario_counts
        return samples, scenario_counts

    def _discover_balanced_samples(
        self,
        root_dirs: Iterable[str | Path],
        route_glob: str,
        max_samples: Optional[int],
    ) -> tuple[List[tuple[Path, int]], dict[str, int]]:
        scenario_buckets: dict[str, List[tuple[Path, int]]] = {}
        for root in root_dirs:
            root_path = Path(root)
            if not root_path.exists():
                raise RuntimeError(f"Bench2Drive root does not exist: {root_path}")
            for route_dir in sorted(root_path.glob(route_glob)):
                if not _is_b2d_route(route_dir):
                    continue
                scenario = _scenario_name(route_dir)
                bucket = scenario_buckets.setdefault(scenario, [])
                if self.max_samples_per_scenario is not None and len(bucket) >= self.max_samples_per_scenario:
                    continue
                frames = sorted(int(path.stem.split(".")[0]) for path in _annotation_dir(route_dir).glob("*.json.gz"))
                for frame in frames[:: self.frame_sampling]:
                    if self._has_required_files(route_dir, frame):
                        bucket.append((route_dir, frame))
                        if self.max_samples_per_scenario is not None and len(bucket) >= self.max_samples_per_scenario:
                            break

        samples = _round_robin_scenario_samples(scenario_buckets, max_samples)
        scenario_counts: dict[str, int] = {}
        for route_dir, _frame in samples:
            scenario = _scenario_name(route_dir)
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        return samples, scenario_counts

    def _has_required_files(self, route_dir: Path, frame: int) -> bool:
        if not _has_frame_file(_image_dir(route_dir), frame, ".jpg"):
            return False
        if not _has_frame_file(route_dir / "lidar", frame, ".laz"):
            return False
        if self.target_mode == TARGET_MODE_FUTURE_EGO_TIME:
            for offset in range(self.future_stride, self.future_stride * (self.num_poses + 1), self.future_stride):
                if not _has_frame_file(_annotation_dir(route_dir), frame + offset, ".json.gz"):
                    return False
        elif self.target_mode == TARGET_MODE_SPATIAL_PATH:
            if not _has_frame_file(_annotation_dir(route_dir), frame + 1, ".json.gz"):
                return False
        else:
            raise RuntimeError(f"Unsupported DiffusionDrive target mode: {self.target_mode}")
        return True


def _is_b2d_route(path: Path) -> bool:
    return path.is_dir() and _image_dir(path).is_dir() and (path / "lidar").is_dir() and _annotation_dir(path).is_dir()


def _scenario_name(route_dir: Path) -> str:
    return route_dir.parent.name


def _round_robin_scenario_samples(
    scenario_buckets: dict[str, List[tuple[Path, int]]],
    max_samples: Optional[int],
) -> List[tuple[Path, int]]:
    samples: List[tuple[Path, int]] = []
    scenario_names = sorted(name for name, bucket in scenario_buckets.items() if bucket)
    if not scenario_names:
        return samples

    max_bucket_length = max(len(scenario_buckets[name]) for name in scenario_names)
    for index in range(max_bucket_length):
        for scenario_name in scenario_names:
            bucket = scenario_buckets[scenario_name]
            if index >= len(bucket):
                continue
            samples.append(bucket[index])
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def _manifest_values_match(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _normalize_manifest_root(root: str | Path) -> str:
    return str(Path(root).expanduser().resolve(strict=False))


def load_annotation(route_dir: Path, frame: int) -> dict:
    path = _frame_path(_annotation_dir(route_dir), frame, ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        annotation = json.load(file)
    return _normalize_measurement_annotation(annotation)


def _annotation_dir(route_dir: Path) -> Path:
    if (route_dir / "anno").is_dir():
        return route_dir / "anno"
    return route_dir / "measurements"


def _image_dir(route_dir: Path) -> Path:
    if (route_dir / "camera" / "rgb_front").is_dir():
        return route_dir / "camera" / "rgb_front"
    return route_dir / "rgb"


def _frame_path(directory: Path, frame: int, suffix: str) -> Path:
    for width in (5, 4, 0):
        stem = str(frame) if width == 0 else f"{frame:0{width}d}"
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise RuntimeError(f"Missing frame file in {directory}: frame={frame} suffix={suffix}")


def _has_frame_file(directory: Path, frame: int, suffix: str) -> bool:
    for width in (5, 4, 0):
        stem = str(frame) if width == 0 else f"{frame:0{width}d}"
        if (directory / f"{stem}{suffix}").is_file():
            return True
    return False


def _normalize_measurement_annotation(annotation: dict) -> dict:
    if "pos_global" in annotation and "x" not in annotation:
        annotation = dict(annotation)
        annotation["x"] = float(annotation["pos_global"][0])
        annotation["y"] = float(annotation["pos_global"][1])
        annotation.setdefault("command_far", int(annotation.get("command", annotation.get("next_command", 4))))
        annotation.setdefault("command_near", int(annotation.get("next_command", annotation["command_far"])))
    return annotation


def build_camera_feature(
    route_dir: Path,
    frame: int,
    config: GlobalConfig,
    model_image_size: tuple[int, int] = (256, 1024),
    jpeg_artifact: bool = True,
) -> torch.Tensor:
    image_path = _frame_path(_image_dir(route_dir), frame, ".jpg")
    camera_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if camera_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Raw B2D images are commonly 900x1600. For the CARLA-native baseline,
    # map them into the online sensor size before applying the online crop.
    camera_bgr = cv2.resize(camera_bgr, (config.camera_width, config.camera_height), interpolation=cv2.INTER_LINEAR)
    if jpeg_artifact:
        _, encoded = cv2.imencode(".jpg", camera_bgr)
        camera_bgr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    camera_rgb = cv2.cvtColor(camera_bgr, cv2.COLOR_BGR2RGB)
    camera_rgb = t_u.crop_array(config, camera_rgb)
    camera_chw = np.transpose(camera_rgb, (2, 0, 1))
    camera_tensor = torch.from_numpy(camera_chw).float().unsqueeze(0) / 255.0
    camera_tensor = torch.nn.functional.interpolate(
        camera_tensor,
        size=model_image_size,
        mode="bilinear",
        align_corners=False,
    )
    return camera_tensor.squeeze(0)


def build_lidar_feature(
    route_dir: Path,
    frame: int,
    config: GlobalConfig,
) -> torch.Tensor:
    lidar_path = _frame_path(route_dir / "lidar", frame, ".laz")
    lidar = laspy.read(str(lidar_path)).xyz
    lidar_hist = lidar_to_histogram_features(lidar, config)
    return torch.from_numpy(lidar_hist).float()


def lidar_to_histogram_features(lidar: np.ndarray, config: GlobalConfig) -> np.ndarray:
    def splat_points(point_cloud: np.ndarray) -> np.ndarray:
        xbins = np.linspace(
            config.min_x,
            config.max_x,
            (config.max_x - config.min_x) * int(config.pixels_per_meter) + 1,
        )
        ybins = np.linspace(
            config.min_y,
            config.max_y,
            (config.max_y - config.min_y) * int(config.pixels_per_meter) + 1,
        )
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
        return (hist / config.hist_max_per_pixel).T

    lidar = lidar[lidar[..., 2] < config.max_height_lidar]
    below = lidar[lidar[..., 2] <= config.lidar_split_height]
    above = lidar[lidar[..., 2] > config.lidar_split_height]
    above_features = splat_points(above)
    if config.use_ground_plane:
        below_features = splat_points(below)
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)
    return np.transpose(features, (2, 0, 1)).astype(np.float32)


def build_status_feature(annotation: dict) -> torch.Tensor:
    command = int(annotation.get("command_far", annotation.get("command_near", 4)))
    speed = float(annotation["speed"])
    return build_status_feature_from_command(command, speed)


def target_speed_label_metadata() -> dict:
    return {
        "schema": TARGET_SPEED_LABEL_SCHEMA,
        "classes_mps": list(TARGET_SPEED_CLASSES_MPS),
        "brake_class_index": 0,
        "valid_key": "target_speed_label_valid",
    }


def build_target_speed_metadata(annotation: dict) -> dict:
    has_target_speed = "target_speed" in annotation
    has_brake = "brake" in annotation or "control_brake" in annotation
    target_speed = float(annotation["target_speed"]) if has_target_speed else float(annotation.get("speed", 0.0))
    brake = _annotation_bool(annotation.get("brake", annotation.get("control_brake", False)))
    return target_speed_metadata_from_values(
        target_speed,
        brake,
        valid=has_target_speed and has_brake,
    )


def target_speed_metadata_from_record(record: dict, *, fallback_speed: float) -> dict:
    if record.get("target_speed_twohot") is not None and int(record.get("target_speed_class", -1)) >= 0:
        twohot = [float(value) for value in record["target_speed_twohot"]]
        target_speed_class = int(record["target_speed_class"])
    else:
        twohot = target_speed_twohot(
            float(record.get("target_speed", fallback_speed)),
            _annotation_bool(record.get("brake", False)),
        )
        target_speed_class = int(np.argmax(twohot))
    return {
        "target_speed": float(record.get("target_speed", fallback_speed)),
        "brake": _annotation_bool(record.get("brake", False)),
        "target_speed_twohot": twohot,
        "target_speed_class": target_speed_class,
        "target_speed_label_valid": int(record.get("target_speed_label_valid", 0)),
    }


def target_speed_metadata_from_values(target_speed: float, brake: bool, *, valid: bool) -> dict:
    twohot = target_speed_twohot(target_speed, brake)
    return {
        "target_speed": float(target_speed),
        "brake": bool(brake),
        "target_speed_twohot": twohot,
        "target_speed_class": int(np.argmax(twohot)),
        "target_speed_label_valid": int(valid),
    }


def target_speed_twohot(
    target_speed: float,
    brake: bool,
    classes_mps: tuple[float, ...] = TARGET_SPEED_CLASSES_MPS,
) -> list[float]:
    classes = np.asarray(classes_mps, dtype=np.float64)
    label = np.zeros((classes.shape[0],), dtype=np.float32)
    speed = float(target_speed)
    if brake or speed <= classes[0]:
        label[0] = 1.0
        return label.tolist()
    if speed >= classes[-1]:
        label[-1] = 1.0
        return label.tolist()

    upper_idx = int(np.argmax(classes > speed))
    lower_idx = max(upper_idx - 1, 0)
    lower_val = float(classes[lower_idx])
    upper_val = float(classes[upper_idx])
    denom = max(upper_val - lower_val, 1e-6)
    label[lower_idx] = (upper_val - speed) / denom
    label[upper_idx] = (speed - lower_val) / denom
    return label.tolist()


def _annotation_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def is_hard_left_turn_stop_sample(
    annotation: dict,
    trajectory: torch.Tensor | np.ndarray,
    *,
    command: int = 1,
    speed_threshold: float = 0.1,
    y_threshold: float = 4.0,
) -> bool:
    """Return whether a sample matches the current low-speed left-turn failure mode."""

    sample_command = int(annotation.get("command_far", annotation.get("command_near", 4)))
    speed = float(annotation.get("speed", 0.0))
    return is_hard_left_turn_stop_sample_from_values(
        sample_command,
        speed,
        trajectory,
        command=command,
        speed_threshold=speed_threshold,
        y_threshold=y_threshold,
    )


def is_hard_left_turn_stop_sample_from_values(
    sample_command: int,
    speed: float,
    trajectory: torch.Tensor | np.ndarray,
    *,
    command: int = 1,
    speed_threshold: float = 0.1,
    y_threshold: float = 4.0,
) -> bool:
    """Return whether scalar sample metadata matches the known left-turn failure mode."""

    traj = torch.as_tensor(trajectory)
    if traj.ndim < 2 or traj.shape[0] == 0 or traj.shape[-1] < 2:
        return False
    target_end_y = float(traj[-1, 1])
    return sample_command == int(command) and speed < speed_threshold and abs(target_end_y) > y_threshold


def build_trajectory_target(
    route_dir: Path,
    frame: int,
    num_poses: int,
    stride: int,
    target_mode: str = TARGET_MODE_SPATIAL_PATH,
    spatial_first_distance: float = 2.5,
    spatial_interval: float = 1.0,
    spatial_max_future_frames: int = 120,
) -> torch.Tensor:
    if target_mode == TARGET_MODE_SPATIAL_PATH:
        return build_spatial_path_target(
            route_dir,
            frame,
            num_poses,
            first_distance=spatial_first_distance,
            interval=spatial_interval,
            max_future_frames=spatial_max_future_frames,
        )
    if target_mode != TARGET_MODE_FUTURE_EGO_TIME:
        raise RuntimeError(f"Unsupported DiffusionDrive target mode: {target_mode}")
    return build_future_ego_time_target(route_dir, frame, num_poses, stride)


def build_future_ego_time_target(route_dir: Path, frame: int, num_poses: int, stride: int) -> torch.Tensor:
    origin = load_annotation(route_dir, frame)
    trajectory = []
    for offset in range(stride, stride * (num_poses + 1), stride):
        future = load_annotation(route_dir, frame + offset)
        trajectory.append(ego_relative_xy(origin, future))
    return torch.tensor(trajectory, dtype=torch.float32)


def build_spatial_path_target(
    route_dir: Path,
    frame: int,
    num_poses: int,
    first_distance: float = 2.5,
    interval: float = 1.0,
    max_future_frames: int = 120,
) -> torch.Tensor:
    origin = load_annotation(route_dir, frame)
    points = [np.zeros(2, dtype=np.float64)]
    for offset in range(1, max_future_frames + 1):
        if not _has_frame_file(_annotation_dir(route_dir), frame + offset, ".json.gz"):
            break
        future = load_annotation(route_dir, frame + offset)
        points.append(np.asarray(ego_relative_xy(origin, future), dtype=np.float64))

    polyline = _deduplicate_polyline(np.asarray(points, dtype=np.float64))
    fallback_direction = _command_direction(origin)
    target_distances = first_distance + interval * np.arange(num_poses, dtype=np.float64)
    target = _resample_polyline_by_distance(polyline, target_distances, fallback_direction)
    return torch.tensor(target, dtype=torch.float32)


def _deduplicate_polyline(points: np.ndarray, min_segment_length: float = 1e-3) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) >= min_segment_length:
            kept.append(point)
    return np.asarray(kept, dtype=np.float64)


def _resample_polyline_by_distance(
    points: np.ndarray,
    distances: np.ndarray,
    fallback_direction: np.ndarray,
) -> np.ndarray:
    if points.shape[0] == 0:
        points = np.zeros((1, 2), dtype=np.float64)

    if points.shape[0] == 1:
        direction = _normalize_direction(fallback_direction)
        return points[0][None, :] + distances[:, None] * direction[None, :]

    segments = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    valid = segment_lengths > 1e-6
    if not np.any(valid):
        direction = _normalize_direction(fallback_direction)
        return points[0][None, :] + distances[:, None] * direction[None, :]

    segments = segments[valid]
    segment_lengths = segment_lengths[valid]
    start_points = points[:-1][valid]
    end_points = points[1:][valid]
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    samples = []
    for distance in distances:
        if distance <= cumulative[-1]:
            segment_idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
            segment_idx = min(segment_idx, len(segment_lengths) - 1)
            local = (distance - cumulative[segment_idx]) / segment_lengths[segment_idx]
            samples.append(start_points[segment_idx] + local * (end_points[segment_idx] - start_points[segment_idx]))
        else:
            direction = _normalize_direction(segments[-1])
            if np.linalg.norm(direction) < 1e-6:
                direction = _normalize_direction(fallback_direction)
            samples.append(end_points[-1] + (distance - cumulative[-1]) * direction)
    return np.asarray(samples, dtype=np.float64)


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / norm


def _command_direction(origin: dict) -> np.ndarray:
    command_points = []
    for prefix in ("far", "near"):
        x_key = f"x_command_{prefix}"
        y_key = f"y_command_{prefix}"
        if x_key in origin and y_key in origin:
            command_points.append(_world_xy_to_ego_xy(origin, float(origin[x_key]), float(origin[y_key])))
    for point in command_points:
        if np.linalg.norm(point) > 1e-3:
            return point
    for key in ("target_point", "target_point_next", "aim_wp"):
        if key in origin:
            point = np.asarray(origin[key], dtype=np.float64)
            if point.shape[0] >= 2 and np.linalg.norm(point[:2]) > 1e-3:
                return point[:2]
    return np.array([1.0, 0.0], dtype=np.float64)


def _world_xy_to_ego_xy(origin: dict, x: float, y: float) -> np.ndarray:
    origin_world2ego = _ego_world2ego_matrix(origin)
    origin_location = _ego_world_location(origin)
    if origin_world2ego is not None:
        point = np.ones(4, dtype=np.float64)
        point[0] = x
        point[1] = y
        point[2] = float(origin_location[2]) if origin_location is not None else 0.0
        return (origin_world2ego @ point)[:2]

    dx = x - float(origin["x"])
    dy = y - float(origin["y"])
    theta = t_u.preprocess_compass(float(origin["theta"]))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.asarray([cos_theta * dx + sin_theta * dy, -sin_theta * dx + cos_theta * dy], dtype=np.float64)


def ego_relative_xy(origin: dict, future: dict) -> list[float]:
    """Return future ego location in the origin frame.

    Raw Bench2Drive annotations expose top-level x/y/theta fields, but their
    theta follows the IMU compass convention used online before
    preprocess_compass(). Prefer the ego vehicle world2ego matrix when present
    so the target matches the garage ego-frame waypoint convention.
    """
    origin_world2ego = _ego_world2ego_matrix(origin)
    future_location = _ego_world_location(future)
    if origin_world2ego is not None and future_location is not None:
        future_location_h = np.ones(4, dtype=np.float64)
        future_location_h[:3] = future_location
        future_ego = origin_world2ego @ future_location_h
        return [float(future_ego[0]), float(future_ego[1])]

    return _ego_relative_xy_from_pose(origin, future)


def _ego_relative_xy_from_pose(origin: dict, future: dict) -> list[float]:
    dx = float(future["x"]) - float(origin["x"])
    dy = float(future["y"]) - float(origin["y"])
    theta = t_u.preprocess_compass(float(origin["theta"]))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return [
        cos_theta * dx + sin_theta * dy,
        -sin_theta * dx + cos_theta * dy,
    ]


def _ego_vehicle_box(annotation: dict) -> Optional[dict]:
    boxes = annotation.get("bounding_boxes")
    if not boxes:
        return None
    for box in boxes:
        if box.get("class") == "ego_vehicle":
            return box
    return boxes[0]


def _ego_world2ego_matrix(annotation: dict) -> Optional[np.ndarray]:
    if "world2ego" in annotation:
        matrix = np.asarray(annotation["world2ego"], dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix
    if "ego_matrix" in annotation:
        ego_to_world = np.asarray(annotation["ego_matrix"], dtype=np.float64)
        if ego_to_world.shape == (4, 4):
            return np.linalg.inv(ego_to_world)
    ego_box = _ego_vehicle_box(annotation)
    if ego_box is None or "world2ego" not in ego_box:
        return None
    matrix = np.asarray(ego_box["world2ego"], dtype=np.float64)
    if matrix.shape != (4, 4):
        return None
    return matrix


def _ego_world_location(annotation: dict) -> Optional[np.ndarray]:
    if "ego_matrix" in annotation:
        ego_to_world = np.asarray(annotation["ego_matrix"], dtype=np.float64)
        if ego_to_world.shape == (4, 4):
            return ego_to_world[:3, 3]
    if "pos_global" in annotation:
        return np.asarray([annotation["pos_global"][0], annotation["pos_global"][1], 0.0], dtype=np.float64)
    ego_box = _ego_vehicle_box(annotation)
    if ego_box is None:
        return None
    for key in ("location", "center"):
        if key in ego_box:
            location = np.asarray(ego_box[key], dtype=np.float64)
            if location.shape == (3,):
                return location
    return None
