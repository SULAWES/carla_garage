"""DiffusionDrive port for CARLA leaderboard.

Env vars:
  - DIFFUSIONDRIVE_CHECKPOINT: path to model weights (.pth/.ckpt).
  - DIFFUSIONDRIVE_ANCHOR_PATH: path to plan anchor .npy file (required).
  - DIFFUSIONDRIVE_BACKBONE_PATH: optional timm backbone weights.
  - DIFFUSIONDRIVE_CAMERA_FOV: override online front camera FOV.
  - DIFFUSIONDRIVE_CAMERA_POS: override online front camera position, "x,y,z".
  - DIFFUSIONDRIVE_CAMERA_ROT: override online front camera rotation, "roll,pitch,yaw".
  - DIFFUSIONDRIVE_CAMERA_WIDTH / DIFFUSIONDRIVE_CAMERA_HEIGHT: override online camera resolution.
  - DIFFUSIONDRIVE_LIDAR_POS / DIFFUSIONDRIVE_LIDAR_ROT: override online LiDAR pose, "x,y,z".
  - DIFFUSIONDRIVE_CROP_IMAGE: override image crop flag (0/1).
  - DIFFUSIONDRIVE_MODEL_IMAGE_HEIGHT / DIFFUSIONDRIVE_MODEL_IMAGE_WIDTH: override model input size.
  - DIFFUSIONDRIVE_USE_ROUTE_CONDITION: feed target_point/target_point_next route token
    values (default: 1); 0 keeps the model token but zeros its values for ablation.
  - DIFFUSIONDRIVE_USE_SPEED_HEAD: use predicted target speed for longitudinal control (default: 0).
  - DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT: when speed-head control is enabled,
    use syb-style probability-weighted target speed instead of argmax (default: 1).
  - DIFFUSIONDRIVE_SPEED_HEAD_BRAKE_THRESHOLD: probability of class 0 required to force
    target speed 0 in uncertainty-weighted mode (default: 0.9).
  - DIFFUSIONDRIVE_ALLOW_PARTIAL_CHECKPOINT: allow missing/mismatched model tensors (default: 0).
  - DIFFUSIONDRIVE_ALLOW_PREPROCESS_MISMATCH: allow checkpoint preprocessing metadata mismatch (default: 0).
  - DIFFUSIONDRIVE_ZERO_LIDAR: replace LiDAR BEV with zeros for diagnostic ablation (default: 0).
  - DIFFUSIONDRIVE_DIFFUSION_NOISE_MODE: random/fixed/zero/seeded inference noise (default: random).
  - DIFFUSIONDRIVE_DIFFUSION_NOISE_SEED: fixed/seeded inference noise seed (default: 0).
  - DIFFUSIONDRIVE_COMMAND_DELAY: use the inherited one-command delay (default: 0).
  - DIFFUSIONDRIVE_SPATIAL_PID: use spatial-checkpoint speed logic (default: config value).
  - DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST: override spatial PID fast target speed.
  - DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW: override spatial PID slow target speed.
  - DIFFUSIONDRIVE_SPATIAL_PID_TURN_THRESHOLD: override turn-ratio slowdown threshold.
  - DIFFUSIONDRIVE_SPATIAL_PID_SHARP_TURN_THRESHOLD: override full-slowdown turn-ratio threshold.
  - DIFFUSIONDRIVE_LOW_SPEED_STEER: keep steering active at near-zero speed unless braking (default: 0).
  - DIFFUSIONDRIVE_STUCK_THRESHOLD: override stuck detector threshold.
  - DIFFUSIONDRIVE_CREEP_DURATION: override forced creep duration once stuck.
  - DIFFUSIONDRIVE_CREEP_THROTTLE: override forced creep throttle.
  - DIFFUSIONDRIVE_DEBUG_CONTROL: print low-frequency control diagnostics (default: 0).
  - DIFFUSIONDRIVE_DEBUG_ROUTE: print route-corridor diagnostics (default: 0).
  - DIFFUSIONDRIVE_ROUTE_DEBUG_DISTANCE_WARN: route debug distance warning threshold in meters (default: 3.0).
  - DIFFUSIONDRIVE_ROUTE_DEBUG_ANGLE_WARN_DEG: route debug angle warning threshold in degrees (default: 45.0).
  - DIFFUSIONDRIVE_DEBUG_SAFETY_BOX: print structured safety-box diagnostics (default: 0).
  - DIFFUSIONDRIVE_DEBUG_INTERVAL: control diagnostic print interval in steps (default: 20).
  - DIFFUSIONDRIVE_DEBUG_LIDAR_CONTRACT: record online half/full-scan LiDAR contract stats (default: 0).
  - DIFFUSIONDRIVE_LIDAR_DEBUG_INTERVAL: LiDAR statistics interval in steps (default: 20).
  - DIFFUSIONDRIVE_LIDAR_DUMP_INTERVAL: periodic raw/BEV NPZ dump interval; 0 disables (default: 0).
  - DIFFUSIONDRIVE_LIDAR_DUMP_MAX: maximum NPZ dumps per agent process (default: 50).
  - DIFFUSIONDRIVE_LIDAR_HISTORY_STEPS: recent diagnostic records kept for terminal/event context (default: 100).
  - DIFFUSIONDRIVE_LIDAR_DEBUG_DIR: optional output root; defaults to OUT/lidar_diagnostics.
  - DIFFUSIONDRIVE_DEBUG_TRAJECTORY: write trajectory mode/noise diagnostics (default: 0).
  - DIFFUSIONDRIVE_TRAJECTORY_DEBUG_INTERVAL: periodic trajectory record interval (default: 1).
  - DIFFUSIONDRIVE_TRAJECTORY_JUMP_WARN: aligned endpoint jump event threshold in meters (default: 5).
  - DIFFUSIONDRIVE_TRAJECTORY_DEBUG_DIR: optional output root; defaults to OUT/trajectory_diagnostics.
"""

import json
import os
import math
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import carla

from leaderboard.autoagents import autonomous_agent
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from scipy.optimize import fsolve

from config import GlobalConfig
from data import CARLA_Data
from nav_planner import RoutePlanner
import transfuser_utils as t_u

from diffusiondrive.config_adapter import DiffusionDriveRuntimeOverrides, build_diffusiondrive_config
from diffusiondrive.lidar_diagnostics import (
    LidarDiagnosticWriter,
    compute_lidar_contract_record,
    make_json_safe,
)
from diffusiondrive.model import V2TransfuserModel
from diffusiondrive.inference_diagnostics import (
    gaussian_noise_from_seed,
    stable_diffusion_noise_seed,
)
from diffusiondrive.status import build_status_feature
from birds_eye_view.run_stop_sign import RunStopSign


_UKF_INITIAL_COVARIANCE = np.diag([0.5, 0.5, 0.000001, 0.000001])
_UKF_MIN_COVARIANCE_EIGENVALUE = 1e-9
_TARGET_SPEED_CLASSES_MPS = torch.tensor(
    [0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0],
    dtype=torch.float32,
)


# Leaderboard function that selects the class used as agent.
def get_entry_point():
    return "DiffusionDriveAgent"


def strtobool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1", "True")


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {value!r}.") from exc


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}.") from exc


def env_float_list(name, default, length):
    value = os.environ.get(name)
    if value is None or value == "":
        return list(default)
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != length:
        raise RuntimeError(f"{name} must contain {length} comma-separated floats, got {value!r}.")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise RuntimeError(f"{name} must contain {length} comma-separated floats, got {value!r}.") from exc


class DiffusionDriveAgent(autonomous_agent.AutonomousAgent):
    """DiffusionDrive agent for CARLA leaderboard."""

    _CHECKPOINT_WRAPPER_PREFIXES = ("agent", "model", "module", "_transfuser_model")

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):  # pylint: disable=unused-argument
        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.MAP if os.environ.get(
            "CHALLENGE_TRACK_CODENAME") == "MAP" else autonomous_agent.Track.SENSORS

        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # CARLA config for sensors and control
        self.config = GlobalConfig()
        self._apply_runtime_sensor_overrides()
        self.apply_jpeg_artifact = strtobool(os.environ.get("DIFFUSIONDRIVE_JPEG_ARTIFACT", "1"))
        self.image_normalization = os.environ.get("DIFFUSIONDRIVE_IMAGE_NORMALIZATION", "none").lower()
        if self.image_normalization not in ("none", "imagenet"):
            raise RuntimeError(
                "DIFFUSIONDRIVE_IMAGE_NORMALIZATION must be one of: none, imagenet; "
                f"got {self.image_normalization}."
            )
        self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)
        print("Use JPEG artifact in DiffusionDrive image preprocessing:", self.apply_jpeg_artifact)
        print("DiffusionDrive image normalization:", self.image_normalization)
        self.zero_lidar = strtobool(os.environ.get("DIFFUSIONDRIVE_ZERO_LIDAR", "0"))
        self.use_route_condition = strtobool(os.environ.get("DIFFUSIONDRIVE_USE_ROUTE_CONDITION", "1"))
        self.use_speed_head_controller = strtobool(os.environ.get("DIFFUSIONDRIVE_USE_SPEED_HEAD", "0"))
        self.speed_head_uncertainty_weight = strtobool(os.environ.get(
            "DIFFUSIONDRIVE_SPEED_HEAD_UNCERTAINTY_WEIGHT",
            "1",
        ))
        self.speed_head_brake_threshold = env_float("DIFFUSIONDRIVE_SPEED_HEAD_BRAKE_THRESHOLD", 0.9)
        if not 0.0 <= self.speed_head_brake_threshold <= 1.0:
            raise RuntimeError(
                "DIFFUSIONDRIVE_SPEED_HEAD_BRAKE_THRESHOLD must be in [0, 1], "
                f"got {self.speed_head_brake_threshold}."
            )
        self.allow_partial_checkpoint = strtobool(os.environ.get("DIFFUSIONDRIVE_ALLOW_PARTIAL_CHECKPOINT", "0"))
        self.allow_preprocess_mismatch = strtobool(os.environ.get("DIFFUSIONDRIVE_ALLOW_PREPROCESS_MISMATCH", "0"))
        self.use_command_delay = strtobool(os.environ.get("DIFFUSIONDRIVE_COMMAND_DELAY", "0"))
        self.use_spatial_pid = strtobool(os.environ.get(
            "DIFFUSIONDRIVE_SPATIAL_PID",
            str(int(bool(self.config.diffusiondrive_spatial_pid))),
        ))
        self.config.diffusiondrive_spatial_pid_speed_fast = env_float(
            "DIFFUSIONDRIVE_SPATIAL_PID_SPEED_FAST",
            self.config.diffusiondrive_spatial_pid_speed_fast,
        )
        self.config.diffusiondrive_spatial_pid_speed_slow = env_float(
            "DIFFUSIONDRIVE_SPATIAL_PID_SPEED_SLOW",
            self.config.diffusiondrive_spatial_pid_speed_slow,
        )
        self.config.diffusiondrive_spatial_pid_turn_threshold = env_float(
            "DIFFUSIONDRIVE_SPATIAL_PID_TURN_THRESHOLD",
            self.config.diffusiondrive_spatial_pid_turn_threshold,
        )
        self.config.diffusiondrive_spatial_pid_sharp_turn_threshold = env_float(
            "DIFFUSIONDRIVE_SPATIAL_PID_SHARP_TURN_THRESHOLD",
            self.config.diffusiondrive_spatial_pid_sharp_turn_threshold,
        )
        self.debug_control = strtobool(os.environ.get("DIFFUSIONDRIVE_DEBUG_CONTROL", "0"))
        self.debug_route = strtobool(os.environ.get("DIFFUSIONDRIVE_DEBUG_ROUTE", "0"))
        self.debug_safety_box = strtobool(os.environ.get("DIFFUSIONDRIVE_DEBUG_SAFETY_BOX", "0"))
        self.debug_control_interval = max(1, int(os.environ.get("DIFFUSIONDRIVE_DEBUG_INTERVAL", "20")))
        self.debug_lidar_contract = strtobool(os.environ.get("DIFFUSIONDRIVE_DEBUG_LIDAR_CONTRACT", "0"))
        self.debug_trajectory = strtobool(os.environ.get("DIFFUSIONDRIVE_DEBUG_TRAJECTORY", "0"))
        self.lidar_debug_interval = max(1, env_int("DIFFUSIONDRIVE_LIDAR_DEBUG_INTERVAL", 20))
        self.lidar_dump_interval = max(0, env_int("DIFFUSIONDRIVE_LIDAR_DUMP_INTERVAL", 0))
        self.lidar_dump_max = max(0, env_int("DIFFUSIONDRIVE_LIDAR_DUMP_MAX", 50))
        self.lidar_history_steps = max(1, env_int("DIFFUSIONDRIVE_LIDAR_HISTORY_STEPS", 100))
        self.trajectory_debug_interval = max(1, env_int("DIFFUSIONDRIVE_TRAJECTORY_DEBUG_INTERVAL", 1))
        self.trajectory_jump_warn = env_float("DIFFUSIONDRIVE_TRAJECTORY_JUMP_WARN", 5.0)
        if self.trajectory_jump_warn <= 0.0:
            raise RuntimeError(
                "DIFFUSIONDRIVE_TRAJECTORY_JUMP_WARN must be positive; "
                f"got {self.trajectory_jump_warn}."
            )
        self.route_debug_distance_warn = env_float("DIFFUSIONDRIVE_ROUTE_DEBUG_DISTANCE_WARN", 3.0)
        self.route_debug_angle_warn = env_float("DIFFUSIONDRIVE_ROUTE_DEBUG_ANGLE_WARN_DEG", 45.0)
        self.low_speed_steer = strtobool(os.environ.get("DIFFUSIONDRIVE_LOW_SPEED_STEER", "0"))
        self.config.stuck_threshold = env_int("DIFFUSIONDRIVE_STUCK_THRESHOLD", self.config.stuck_threshold)
        self.config.creep_duration = env_int("DIFFUSIONDRIVE_CREEP_DURATION", self.config.creep_duration)
        self.config.creep_throttle = env_float("DIFFUSIONDRIVE_CREEP_THROTTLE", self.config.creep_throttle)
        print("DiffusionDrive command delay:", self.use_command_delay)
        print("DiffusionDrive spatial PID:", self.use_spatial_pid)
        print(
            "DiffusionDrive spatial PID params: "
            f"speed_fast={self.config.diffusiondrive_spatial_pid_speed_fast}, "
            f"speed_slow={self.config.diffusiondrive_spatial_pid_speed_slow}, "
            f"turn_threshold={self.config.diffusiondrive_spatial_pid_turn_threshold}, "
            f"sharp_turn_threshold={self.config.diffusiondrive_spatial_pid_sharp_turn_threshold}"
        )
        print("DiffusionDrive low-speed steer:", self.low_speed_steer)
        print(
            "DiffusionDrive stuck recovery params: "
            f"stuck_threshold={self.config.stuck_threshold}, "
            f"creep_duration={self.config.creep_duration}, "
            f"creep_throttle={self.config.creep_throttle}"
        )
        print("DiffusionDrive control debug:", self.debug_control)
        print("DiffusionDrive route debug:", self.debug_route)
        print("DiffusionDrive safety-box debug:", self.debug_safety_box)
        print("DiffusionDrive LiDAR contract debug:", self.debug_lidar_contract)
        print("DiffusionDrive trajectory debug:", self.debug_trajectory)
        print("DiffusionDrive zero LiDAR:", self.zero_lidar)
        print("DiffusionDrive route condition values:", self.use_route_condition)
        print("DiffusionDrive speed-head longitudinal control:", self.use_speed_head_controller)
        print(
            "DiffusionDrive speed-head inference: "
            f"uncertainty_weight={self.speed_head_uncertainty_weight}, "
            f"brake_threshold={self.speed_head_brake_threshold}"
        )
        print("DiffusionDrive allow partial checkpoint:", self.allow_partial_checkpoint)
        print("DiffusionDrive allow preprocessing mismatch:", self.allow_preprocess_mismatch)

        # DiffusionDrive model config
        dd_overrides = DiffusionDriveRuntimeOverrides.from_environment()
        self.dd_config = build_diffusiondrive_config(self.config, dd_overrides)
        self._apply_runtime_model_overrides()
        print(
            "DiffusionDrive inference noise: "
            f"mode={self.dd_config.diffusion_infer_noise_mode}, "
            f"seed={self.dd_config.diffusion_infer_noise_seed}",
            flush=True,
        )

        # Build model
        self.model = V2TransfuserModel(self.dd_config).to(self.device)
        self.model.eval()

        # Load weights
        ckpt_path = os.environ.get("DIFFUSIONDRIVE_CHECKPOINT", "")
        self.checkpoint_path = ckpt_path
        if ckpt_path:
            self._load_checkpoint(ckpt_path)
        else:
            print("DIFFUSIONDRIVE_CHECKPOINT not set, using random initialization.")

        # PID controllers for control
        self.turn_controller = t_u.PIDController(k_p=self.config.turn_kp,
                                                 k_i=self.config.turn_ki,
                                                 k_d=self.config.turn_kd,
                                                 n=self.config.turn_n)
        self.speed_controller = t_u.PIDController(k_p=self.config.speed_kp,
                                                  k_i=self.config.speed_ki,
                                                  k_d=self.config.speed_kd,
                                                  n=self.config.speed_n)

        # Filtering
        self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
        self.ukf = UKF(dim_x=4,
                       dim_z=4,
                       fx=bicycle_model_forward,
                       hx=measurement_function_hx,
                       dt=self.config.carla_frame_rate,
                       points=self.points,
                       x_mean_fn=state_mean,
                       z_mean_fn=measurement_mean,
                       residual_x=residual_state_x,
                       residual_z=residual_measurement_h)

        self.ukf.P = _UKF_INITIAL_COVARIANCE.copy()
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
        self.filter_initialized = False
        self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))
        self.ukf_reinitializations = 0

        self.stuck_detector = 0
        self.force_move = 0
        self.stop_sign_controller = int(os.environ.get("STOP_CONTROL", 0))
        print("Use stop sign controller:", self.stop_sign_controller)
        self.stop_sign_criteria = None
        self.hero_actor = None
        self.waiting_ticks_at_stop_sign = 0
        self.cleared_stop_sign = False
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # Temporal LiDAR buffer for multi-frame processing
        self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)
        self.lidar_last = None

        self.last_route_debug = {}
        self.last_safety_box_debug = {}
        self.last_speed_head_debug = {"available": False}
        self.lidar_diagnostic_history = deque(maxlen=self.lidar_history_steps)
        self.lidar_diagnostic_writer = None
        self.last_lidar_event_step = {}
        self.trajectory_diagnostic_handle = None
        self.trajectory_diagnostic_path = None
        self.last_trajectory_diagnostic = None
        self.current_diffusion_noise_key = None
        if self.debug_lidar_contract:
            output_root_value = os.environ.get("DIFFUSIONDRIVE_LIDAR_DEBUG_DIR", "").strip()
            if output_root_value:
                output_root = Path(output_root_value)
            else:
                run_output = os.environ.get("OUT", "").strip()
                output_root = Path(run_output) / "lidar_diagnostics" if run_output else Path("/tmp/diffusiondrive_lidar_diagnostics")
            route_index_value = os.environ.get("START_IDX", "unknown")
            run_tag = f"route_{route_index_value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid_{os.getpid()}"
            lidar_output_dir = output_root / run_tag
            self.lidar_diagnostic_writer = LidarDiagnosticWriter(
                lidar_output_dir,
                max_dumps=self.lidar_dump_max,
            )
            print(
                "DiffusionDrive LiDAR diagnostics: "
                f"dir={lidar_output_dir}, stats_interval={self.lidar_debug_interval}, "
                f"dump_interval={self.lidar_dump_interval}, max_dumps={self.lidar_dump_max}, "
                f"history={self.lidar_history_steps}",
                flush=True,
            )
        if self.debug_trajectory:
            output_root_value = os.environ.get("DIFFUSIONDRIVE_TRAJECTORY_DEBUG_DIR", "").strip()
            if output_root_value:
                output_root = Path(output_root_value)
            else:
                run_output = os.environ.get("OUT", "").strip()
                output_root = (
                    Path(run_output) / "trajectory_diagnostics"
                    if run_output
                    else Path("/tmp/diffusiondrive_trajectory_diagnostics")
                )
            route_index_value = os.environ.get("START_IDX", "unknown")
            run_tag = (
                f"route_{route_index_value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_pid_{os.getpid()}"
            )
            trajectory_output_dir = output_root / run_tag
            trajectory_output_dir.mkdir(parents=True, exist_ok=True)
            self.trajectory_diagnostic_path = trajectory_output_dir / "records.jsonl"
            self.trajectory_diagnostic_handle = self.trajectory_diagnostic_path.open(
                "w",
                encoding="utf-8",
                buffering=1,
            )
            metadata = {
                "route_index": route_index_value,
                "checkpoint": self.checkpoint_path,
                "noise_mode": self.dd_config.diffusion_infer_noise_mode,
                "noise_seed": self.dd_config.diffusion_infer_noise_seed,
                "record_interval": self.trajectory_debug_interval,
                "jump_warn_m": self.trajectory_jump_warn,
            }
            (trajectory_output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                "DiffusionDrive trajectory diagnostics: "
                f"path={self.trajectory_diagnostic_path}, "
                f"interval={self.trajectory_debug_interval}, "
                f"jump_warn={self.trajectory_jump_warn}",
                flush=True,
            )

    def _apply_runtime_sensor_overrides(self) -> None:
        """Apply online sensor/preprocessing overrides before data/model setup."""
        self.config.camera_fov = env_float("DIFFUSIONDRIVE_CAMERA_FOV", self.config.camera_fov)
        self.config.camera_pos = env_float_list("DIFFUSIONDRIVE_CAMERA_POS", self.config.camera_pos, 3)
        self.config.camera_rot_0 = env_float_list("DIFFUSIONDRIVE_CAMERA_ROT", self.config.camera_rot_0, 3)
        self.config.camera_width = env_int("DIFFUSIONDRIVE_CAMERA_WIDTH", self.config.camera_width)
        self.config.camera_height = env_int("DIFFUSIONDRIVE_CAMERA_HEIGHT", self.config.camera_height)
        self.config.lidar_pos = env_float_list("DIFFUSIONDRIVE_LIDAR_POS", self.config.lidar_pos, 3)
        self.config.lidar_rot = env_float_list("DIFFUSIONDRIVE_LIDAR_ROT", self.config.lidar_rot, 3)
        if self.config.camera_width <= 0 or self.config.camera_height <= 0:
            raise RuntimeError(
                "DIFFUSIONDRIVE_CAMERA_WIDTH/HEIGHT must be positive; "
                f"got {self.config.camera_width}x{self.config.camera_height}."
            )

        crop_value = os.environ.get("DIFFUSIONDRIVE_CROP_IMAGE")
        if crop_value is not None and crop_value != "":
            self.config.crop_image = strtobool(crop_value)
        self._refresh_global_config_image_anchors()

        print(
            "DiffusionDrive sensor config: "
            f"camera_size={self.config.camera_width}x{self.config.camera_height}, "
            f"camera_fov={self.config.camera_fov}, "
            f"camera_pos={self.config.camera_pos}, "
            f"camera_rot={self.config.camera_rot_0}, "
            f"crop_image={self.config.crop_image}, "
            f"cropped_size={self.config.cropped_width}x{self.config.cropped_height}, "
            f"lidar_pos={self.config.lidar_pos}, "
            f"lidar_rot={self.config.lidar_rot}",
            flush=True,
        )

    def _refresh_global_config_image_anchors(self) -> None:
        if self.config.crop_image:
            self.config.img_vert_anchors = self.config.cropped_height // 32
            self.config.img_horz_anchors = self.config.cropped_width // 32
        else:
            self.config.img_vert_anchors = self.config.camera_height // 32
            self.config.img_horz_anchors = self.config.camera_width // 32

    def _apply_runtime_model_overrides(self) -> None:
        model_height = env_int("DIFFUSIONDRIVE_MODEL_IMAGE_HEIGHT", self.dd_config.camera_height)
        model_width = env_int("DIFFUSIONDRIVE_MODEL_IMAGE_WIDTH", self.dd_config.camera_width)
        if model_height <= 0 or model_width <= 0:
            raise RuntimeError(
                "DIFFUSIONDRIVE_MODEL_IMAGE_HEIGHT/WIDTH must be positive; "
                f"got {model_height}x{model_width}."
            )
        if model_height % 32 != 0 or model_width % 32 != 0:
            raise RuntimeError(
                "DIFFUSIONDRIVE_MODEL_IMAGE_HEIGHT/WIDTH must be divisible by 32; "
                f"got {model_height}x{model_width}."
            )
        self.dd_config.camera_height = model_height
        self.dd_config.camera_width = model_width
        self.dd_config.__post_init__()
        print(
            "DiffusionDrive model image size: "
            f"{self.dd_config.camera_width}x{self.dd_config.camera_height}",
            flush=True,
        )

    def _load_checkpoint(self, ckpt_path: str) -> None:
        if torch.cuda.is_available():
            checkpoint = torch.load(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

        state_dict, source_name = self._extract_checkpoint_state_dict(checkpoint)
        filtered_state_dict, load_summary = self._align_checkpoint_state_dict(state_dict)
        self._validate_checkpoint_metadata(checkpoint, ckpt_path)
        self._validate_checkpoint_load_summary(load_summary, ckpt_path)
        self.model.load_state_dict(filtered_state_dict, strict=False)

        print(
            "Loaded DiffusionDrive checkpoint from "
            f"'{source_name}': matched {load_summary['loaded_count']}/{load_summary['model_key_count']} model tensors."
        )
        if load_summary["shape_mismatch"]:
            print(
                "Checkpoint tensors skipped due to shape mismatch "
                f"({len(load_summary['shape_mismatch'])}): {load_summary['shape_mismatch'][:10]}"
            )
        if load_summary["unexpected_keys"]:
            print(
                "Checkpoint tensors skipped because no matching model key was found "
                f"({len(load_summary['unexpected_keys'])}): {load_summary['unexpected_keys'][:10]}"
            )
        if load_summary["missing_keys"]:
            print(
                "Model tensors left uninitialized by checkpoint "
                f"({len(load_summary['missing_keys'])}): {load_summary['missing_keys'][:10]}"
            )

    def _validate_checkpoint_metadata(self, checkpoint, ckpt_path: str) -> None:
        if not isinstance(checkpoint, dict):
            return

        args = checkpoint.get("args")
        dd_config = checkpoint.get("diffusiondrive_config")
        mismatches = []

        if isinstance(args, dict):
            expected_normalization = args.get("image_normalization")
            if expected_normalization is not None and str(expected_normalization).lower() != self.image_normalization:
                mismatches.append(
                    "image_normalization: "
                    f"checkpoint={expected_normalization!r} runtime={self.image_normalization!r}"
                )

            expected_height = args.get("model_image_height")
            expected_width = args.get("model_image_width")
            if expected_height is not None and int(expected_height) != int(self.dd_config.camera_height):
                mismatches.append(
                    f"model_image_height: checkpoint={expected_height!r} runtime={self.dd_config.camera_height!r}"
                )
            if expected_width is not None and int(expected_width) != int(self.dd_config.camera_width):
                mismatches.append(
                    f"model_image_width: checkpoint={expected_width!r} runtime={self.dd_config.camera_width!r}"
                )

        if isinstance(dd_config, dict):
            expected_height = dd_config.get("camera_height")
            expected_width = dd_config.get("camera_width")
            if expected_height is not None and int(expected_height) != int(self.dd_config.camera_height):
                mismatches.append(
                    f"config.camera_height: checkpoint={expected_height!r} runtime={self.dd_config.camera_height!r}"
                )
            if expected_width is not None and int(expected_width) != int(self.dd_config.camera_width):
                mismatches.append(
                    f"config.camera_width: checkpoint={expected_width!r} runtime={self.dd_config.camera_width!r}"
                )

        if mismatches and not self.allow_preprocess_mismatch:
            details = "; ".join(mismatches)
            raise RuntimeError(
                f"DiffusionDrive checkpoint preprocessing/config metadata does not match runtime: {ckpt_path}. "
                f"{details}. Set DIFFUSIONDRIVE_ALLOW_PREPROCESS_MISMATCH=1 only for explicit diagnostics."
            )
        if mismatches:
            print(
                "WARNING: DiffusionDrive checkpoint preprocessing/config metadata mismatch allowed: "
                f"{'; '.join(mismatches)}",
                flush=True,
            )

    def _validate_checkpoint_load_summary(self, load_summary: dict, ckpt_path: str) -> None:
        if self.allow_partial_checkpoint:
            return
        if not load_summary["shape_mismatch"] and not load_summary["missing_keys"]:
            return

        details = []
        if load_summary["shape_mismatch"]:
            details.append(
                f"shape_mismatch={len(load_summary['shape_mismatch'])} "
                f"examples={load_summary['shape_mismatch'][:5]}"
            )
        if load_summary["missing_keys"]:
            details.append(
                f"missing_keys={len(load_summary['missing_keys'])} "
                f"examples={load_summary['missing_keys'][:5]}"
            )
        raise RuntimeError(
            f"DiffusionDrive checkpoint is not fully compatible with the current model: {ckpt_path}. "
            f"{'; '.join(details)}. Set DIFFUSIONDRIVE_ALLOW_PARTIAL_CHECKPOINT=1 only for warm-start ablations."
        )

    def _extract_checkpoint_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "ema_state_dict",
                "network",
                "net",
                "weights",
            ):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value, key

            if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
                return checkpoint, "root"

        raise RuntimeError("Unable to locate a tensor state_dict inside the provided checkpoint.")

    def _normalize_checkpoint_key(self, key: str) -> str:
        parts = key.split(".")
        while parts and parts[0] in self._CHECKPOINT_WRAPPER_PREFIXES:
            parts = parts[1:]
        return ".".join(parts)

    def _align_checkpoint_state_dict(self, state_dict):
        model_state_dict = self.model.state_dict()
        normalized_state_dict = {}

        for key, value in state_dict.items():
            normalized_key = self._normalize_checkpoint_key(key)
            if normalized_key not in normalized_state_dict:
                normalized_state_dict[normalized_key] = value

        filtered_state_dict = {}
        shape_mismatch = []
        unexpected_keys = []

        for key, value in normalized_state_dict.items():
            if key not in model_state_dict:
                unexpected_keys.append(key)
                continue
            if tuple(value.shape) != tuple(model_state_dict[key].shape):
                shape_mismatch.append(
                    f"{key}: ckpt{tuple(value.shape)} != model{tuple(model_state_dict[key].shape)}"
                )
                continue
            filtered_state_dict[key] = value

        missing_keys = sorted(set(model_state_dict.keys()) - set(filtered_state_dict.keys()))

        load_summary = {
            "loaded_count": len(filtered_state_dict),
            "model_key_count": len(model_state_dict),
            "missing_keys": missing_keys,
            "unexpected_keys": sorted(unexpected_keys),
            "shape_mismatch": sorted(shape_mismatch),
        }
        return filtered_state_dict, load_summary

    def _init(self):
        try:
            loc = self._global_plan[0][0].location
            wpt = self._global_plan[0][0]
            carla_map = wpt.get_map()
            lat_ref, lon_ref = carla_map.transform_to_geolocation(loc)
            self.lat_ref, self.lon_ref = lat_ref, lon_ref
        except Exception:
            # Fallback: estimate reference from global plan
            self.lat_ref, self.lon_ref = 0.0, 0.0
            try:
                loc = self._global_plan[0][0].location
                x, y = loc.x, loc.y

                def equations(x_guess):
                    earth_radius_equa = 6378137.0
                    eq1 = earth_radius_equa * math.radians(x_guess[1]) - y
                    eq2 = earth_radius_equa * math.radians(x_guess[0]) * math.cos(math.radians(x_guess[1])) - x
                    return [eq1, eq2]

                initial_guess = [0.0, 0.0]
                solution = fsolve(equations, initial_guess)
                self.lat_ref, self.lon_ref = solution[0], solution[1]
            except Exception as e:
                print(e, flush=True)
                self.lat_ref, self.lon_ref = 0.0, 0.0

        self._route_planner = RoutePlanner(self.config.route_planner_min_distance,
                                           self.config.route_planner_max_distance,
                                           self.lat_ref,
                                           self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)

        if self.stop_sign_controller:
            try:
                from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=import-outside-toplevel
                self.hero_actor = CarlaDataProvider.get_hero_actor()
                if self.hero_actor is not None:
                    self.stop_sign_criteria = RunStopSign(self.hero_actor.get_world())
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Failed to initialize stop sign controller: {exc}", flush=True)
                self.stop_sign_criteria = None
                self.hero_actor = None

        self.initialized = True

    def sensors(self):
        sensors = [{
            'type': 'sensor.camera.rgb',
            'x': self.config.camera_pos[0],
            'y': self.config.camera_pos[1],
            'z': self.config.camera_pos[2],
            'roll': self.config.camera_rot_0[0],
            'pitch': self.config.camera_rot_0[1],
            'yaw': self.config.camera_rot_0[2],
            'width': self.config.camera_width,
            'height': self.config.camera_height,
            'fov': self.config.camera_fov,
            'id': 'rgb_front'
        }, {
            'type': 'sensor.other.imu',
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'sensor_tick': self.config.carla_frame_rate,
            'id': 'imu'
        }, {
            'type': 'sensor.other.gnss',
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'sensor_tick': 0.01,
            'id': 'gps'
        }, {
            'type': 'sensor.speedometer',
            'reading_frequency': self.config.carla_fps,
            'id': 'speed'
        }]
        sensors.append({
            'type': 'sensor.lidar.ray_cast',
            'x': self.config.lidar_pos[0],
            'y': self.config.lidar_pos[1],
            'z': self.config.lidar_pos[2],
            'roll': self.config.lidar_rot[0],
            'pitch': self.config.lidar_rot[1],
            'yaw': self.config.lidar_rot[2],
            'id': 'lidar'
        })

        return sensors

    @torch.inference_mode()
    def tick(self, input_data):
        rgb = []
        camera = input_data['rgb_front'][1][:, :, :3]
        if self.apply_jpeg_artifact:
            _, compressed_image_i = cv2.imencode('.jpg', camera)
            camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

        rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
        rgb_pos = t_u.crop_array(self.config, rgb_pos)
        rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
        rgb.append(rgb_pos)

        rgb = np.concatenate(rgb, axis=1)
        rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

        gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
        speed = input_data['speed'][1]['speed']
        compass = t_u.preprocess_compass(input_data['imu'][1][-1])

        result = {
            'rgb': rgb,
            'compass': compass,
        }

        result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

        measurement = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed], dtype=np.float64)

        if not self.filter_initialized:
            self._reset_ukf(measurement, reason="initial measurement")

        try:
            self._stabilize_ukf_covariance()
            self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
            self._stabilize_ukf_covariance()
            self.ukf.update(measurement)
            self._stabilize_ukf_covariance()
            filtered_state = self.ukf.x
        except np.linalg.LinAlgError as exc:
            self._reset_ukf(measurement, reason=f"non-positive definite covariance: {exc}")
            filtered_state = self.ukf.x
        self.state_log.append(filtered_state)
        result['gps'] = filtered_state[0:2]

        waypoint_route = self._route_planner.run_step(np.append(filtered_state[0:2], gps_pos[2]))
        route_points_ego = []
        for route_point, _ in list(waypoint_route)[:20]:
            route_points_ego.append(
                t_u.inverse_conversion_2d(route_point[:2], result['gps'], result['compass'])
            )
        result['route_points_ego'] = np.asarray(route_points_ego, dtype=np.float32)
        if len(waypoint_route) > 2:
            target_point, far_command = waypoint_route[1]
            target_point_next, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, far_command = waypoint_route[1]
            target_point_next = target_point
        else:
            target_point, far_command = waypoint_route[0]
            target_point_next = target_point

        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)
        command_value = self.commands[-2] if self.use_command_delay else far_command.value
        self.last_command_debug = {
            "current": int(far_command.value),
            "delayed": int(self.commands[-2]),
            "used": int(command_value),
            "delay_enabled": bool(self.use_command_delay),
        }
        one_hot_command = t_u.command_to_one_hot(command_value)
        result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)
        result['target_point'] = ego_target_point

        ego_target_point_next = t_u.inverse_conversion_2d(target_point_next[:2], result['gps'], result['compass'])
        ego_target_point_next = torch.from_numpy(ego_target_point_next[np.newaxis]).to(self.device, dtype=torch.float32)
        result['target_point_next'] = ego_target_point_next

        result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

        return result

    def _prepare_camera_feature(self, rgb):
        """Resize camera input and apply the selected runtime normalization."""
        rgb = rgb / 255.0
        rgb = F.interpolate(
            rgb,
            size=(self.dd_config.camera_height, self.dd_config.camera_width),
            mode='bilinear',
            align_corners=False,
        )
        if self.image_normalization == "imagenet":
            if rgb.shape[1] % 3 != 0:
                raise RuntimeError(
                    "ImageNet normalization expects RGB channel groups; "
                    f"got {rgb.shape[1]} channels."
                )
            repeats = rgb.shape[1] // 3
            mean = torch.tensor([0.485, 0.456, 0.406] * repeats, device=rgb.device, dtype=rgb.dtype)
            std = torch.tensor([0.229, 0.224, 0.225] * repeats, device=rgb.device, dtype=rgb.dtype)
            rgb = (rgb - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        return rgb

    def _build_status(self, tick_data):
        return build_status_feature(tick_data['command'], tick_data['speed'], device=self.device)

    def _build_route_condition(self, tick_data):
        target_point = tick_data['target_point']
        target_point_next = tick_data.get('target_point_next', target_point)
        return torch.cat([target_point, target_point_next], dim=1).to(self.device, dtype=torch.float32)

    def _control_pid(self, waypoints, speed, route_points_ego=None, desired_speed_override=None):
        waypoints = waypoints[0].detach().cpu().numpy()
        speed = float(speed)

        endpoint_distance, turn_ratio = self._spatial_path_geometry(waypoints)
        if desired_speed_override is not None:
            desired_speed = max(float(desired_speed_override), 0.0)
            pid_mode = "speed_head"
        elif self.use_spatial_pid:
            desired_speed = self._spatial_path_desired_speed(waypoints)
            pid_mode = "spatial"
        else:
            one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
            one_second = min(max(one_second, 1), waypoints.shape[0])
            half_second = min(max(1, one_second // 2), waypoints.shape[0])
            desired_speed = np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
            pid_mode = "time_index"

        if desired_speed < 1e-4:
            desired_speed = 0.0

        brake = ((desired_speed < self.config.brake_speed) or
                 ((speed / max(desired_speed, 1e-4)) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        if desired_speed < self.config.aim_distance_threshold:
            aim_distance = self.config.aim_distance_slow
        else:
            aim_distance = self.config.aim_distance_fast

        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        raw_angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        angle = raw_angle
        angle_reset_reason = ""
        if brake:
            angle = 0.0
            angle_reset_reason = "brake"
        elif speed < 0.01 and not self.low_speed_steer:
            angle = 0.0
            angle_reset_reason = "low_speed"

        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        self.last_route_debug = self._compute_route_debug(aim, route_points_ego)

        self.last_pid_debug = {
            "mode": pid_mode,
            "desired_speed": float(desired_speed),
            "turn_ratio": float(turn_ratio),
            "endpoint_distance": float(endpoint_distance),
            "aim_index": int(aim_index),
            "aim_x": float(aim[0]),
            "aim_y": float(aim[1]),
            "raw_angle": float(raw_angle),
            "angle": float(angle),
            "angle_reset_reason": angle_reset_reason,
            "low_speed_steer": bool(self.low_speed_steer),
        }
        return steer, throttle, brake

    def _predicted_target_speed(self, outputs):
        logits = outputs.get("target_speed_logits")
        if logits is None:
            self.last_speed_head_debug = {"available": False}
            return None

        probs = torch.softmax(logits[0], dim=-1).detach().cpu()
        class_speeds = _TARGET_SPEED_CLASSES_MPS.to(dtype=probs.dtype)
        pred_class = int(torch.argmax(probs).item())
        expected_speed = float(torch.sum(probs * class_speeds).item())
        brake_prob = float(probs[0].item())
        argmax_speed = float(class_speeds[pred_class].item())

        if self.speed_head_uncertainty_weight:
            if brake_prob > self.speed_head_brake_threshold:
                desired_speed = 0.0
                selection = "prob_brake"
            else:
                desired_speed = expected_speed
                selection = "prob_expectation"
        else:
            desired_speed = argmax_speed
            selection = "argmax"

        self.last_speed_head_debug = {
            "available": True,
            "pred_class": pred_class,
            "desired_speed": desired_speed,
            "expected_speed": expected_speed,
            "argmax_speed": argmax_speed,
            "brake_prob": brake_prob,
            "max_prob": float(torch.max(probs).item()),
            "brake_threshold": float(self.speed_head_brake_threshold),
            "uncertainty_weight": bool(self.speed_head_uncertainty_weight),
            "selection": selection,
        }
        return desired_speed

    def _spatial_path_geometry(self, waypoints):
        if waypoints.shape[0] == 0:
            return 0.0, 0.0

        endpoint = waypoints[-1]
        endpoint_distance = float(np.linalg.norm(endpoint))
        turn_ratio = abs(float(endpoint[1])) / max(endpoint_distance, 1e-4)
        return endpoint_distance, turn_ratio

    def _spatial_path_desired_speed(self, waypoints):
        """Estimate a cautious target speed from spatial checkpoints."""
        endpoint_distance, turn_ratio = self._spatial_path_geometry(waypoints)
        if endpoint_distance < self.config.brake_speed:
            return 0.0

        turn_threshold = self.config.diffusiondrive_spatial_pid_turn_threshold
        sharp_turn_threshold = self.config.diffusiondrive_spatial_pid_sharp_turn_threshold
        if sharp_turn_threshold <= turn_threshold:
            turn_slowdown = float(turn_ratio >= turn_threshold)
        else:
            turn_slowdown = np.clip(
                (turn_ratio - turn_threshold) / (sharp_turn_threshold - turn_threshold),
                0.0,
                1.0,
            )

        speed_fast = self.config.diffusiondrive_spatial_pid_speed_fast
        speed_slow = self.config.diffusiondrive_spatial_pid_speed_slow
        return float(speed_fast * (1.0 - turn_slowdown) + speed_slow * turn_slowdown)

    def _compute_route_debug(self, aim, route_points_ego):
        if route_points_ego is None:
            return {"available": False}

        route_points = np.asarray(route_points_ego, dtype=np.float64)
        if route_points.ndim != 2 or route_points.shape[0] == 0 or route_points.shape[1] < 2:
            return {"available": False}
        route_points = route_points[:, :2]
        finite_mask = np.isfinite(route_points).all(axis=1)
        route_points = route_points[finite_mask]
        if route_points.shape[0] == 0:
            return {"available": False}

        aim = np.asarray(aim, dtype=np.float64)[:2]
        if not np.all(np.isfinite(aim)):
            return {"available": False}

        if route_points.shape[0] == 1:
            nearest = route_points[0]
            distance = float(np.linalg.norm(aim - nearest))
            angle_diff_deg = float("nan")
            tangent = np.array([float("nan"), float("nan")], dtype=np.float64)
            segment_index = 0
        else:
            starts = route_points[:-1]
            ends = route_points[1:]
            segments = ends - starts
            denom = np.sum(segments * segments, axis=1)
            valid = denom > 1e-8
            if not np.any(valid):
                nearest_index = int(np.argmin(np.linalg.norm(route_points - aim[None, :], axis=1)))
                nearest = route_points[nearest_index]
                distance = float(np.linalg.norm(aim - nearest))
                angle_diff_deg = float("nan")
                tangent = np.array([float("nan"), float("nan")], dtype=np.float64)
                segment_index = nearest_index
            else:
                starts_valid = starts[valid]
                segments_valid = segments[valid]
                denom_valid = denom[valid]
                projection = np.sum((aim[None, :] - starts_valid) * segments_valid, axis=1) / denom_valid
                projection = np.clip(projection, 0.0, 1.0)
                nearest_candidates = starts_valid + projection[:, None] * segments_valid
                distances = np.linalg.norm(nearest_candidates - aim[None, :], axis=1)
                valid_indices = np.flatnonzero(valid)
                best_valid_index = int(np.argmin(distances))
                nearest = nearest_candidates[best_valid_index]
                distance = float(distances[best_valid_index])
                segment_index = int(valid_indices[best_valid_index])
                tangent = segments[segment_index]
                aim_angle = math.atan2(float(aim[1]), float(aim[0]))
                route_angle = math.atan2(float(tangent[1]), float(tangent[0]))
                angle_diff_deg = abs(math.degrees(t_u.normalize_angle(aim_angle - route_angle)))

        warning = bool(
            distance > self.route_debug_distance_warn or
            (np.isfinite(angle_diff_deg) and angle_diff_deg > self.route_debug_angle_warn)
        )
        return {
            "available": True,
            "points": int(route_points.shape[0]),
            "aim_x": float(aim[0]),
            "aim_y": float(aim[1]),
            "nearest_x": float(nearest[0]),
            "nearest_y": float(nearest[1]),
            "distance": float(distance),
            "angle_diff_deg": float(angle_diff_deg),
            "segment_index": int(segment_index),
            "tangent_x": float(tangent[0]),
            "tangent_y": float(tangent[1]),
            "warning": warning,
        }

    def _stop_sign_controller_step(self, ego_speed: float) -> bool:
        """Force a full stop when approaching a route-relevant stop sign."""
        if not self.stop_sign_controller or self.stop_sign_criteria is None or self.hero_actor is None:
            return False

        self.stop_sign_criteria.tick(self.hero_actor)
        stop_sign = self.stop_sign_criteria.target_stop_sign
        if stop_sign is None:
            self.waiting_ticks_at_stop_sign = 0
            return False

        ego_location = self.hero_actor.get_location()
        stop_center = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
        distance_to_stop_sign = stop_center.distance(ego_location)

        if distance_to_stop_sign > self.config.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
            self.waiting_ticks_at_stop_sign = 0
            return False

        if ego_speed < 0.1 and distance_to_stop_sign < self.config.clearing_distance_to_stop_sign:
            self.waiting_ticks_at_stop_sign += 1
            if self.waiting_ticks_at_stop_sign > 25:
                self.cleared_stop_sign = True
        else:
            self.waiting_ticks_at_stop_sign = 0

        if self.cleared_stop_sign:
            return False

        return True

    def _filter_safety_box_points(self, lidar_points):
        safety_box = lidar_points[lidar_points[..., 2] > self.config.safety_box_z_min]
        safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]
        safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
        safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]
        safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
        safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
        return safety_box

    def _build_safety_box_debug(self, safety_box, speed, throttle_before, brake_before):
        point_count = int(len(safety_box))
        debug = {
            "point_count": point_count,
            "speed": float(speed),
            "desired_speed": float(getattr(self, "last_pid_debug", {}).get("desired_speed", float("nan"))),
            "throttle_before": float(throttle_before),
            "brake_before": bool(brake_before),
            "stuck_detector": int(self.stuck_detector),
            "force_move": int(self.force_move),
        }
        if point_count == 0:
            debug.update({
                "nearest_xy": float("nan"),
                "x_min": float("nan"),
                "x_max": float("nan"),
                "y_min": float("nan"),
                "y_max": float("nan"),
                "z_min": float("nan"),
                "z_max": float("nan"),
            })
            return debug

        xy_distance = np.linalg.norm(safety_box[:, :2], axis=1)
        debug.update({
            "nearest_xy": float(np.min(xy_distance)),
            "x_min": float(np.min(safety_box[:, 0])),
            "x_max": float(np.max(safety_box[:, 0])),
            "y_min": float(np.min(safety_box[:, 1])),
            "y_max": float(np.max(safety_box[:, 1])),
            "z_min": float(np.min(safety_box[:, 2])),
            "z_max": float(np.max(safety_box[:, 2])),
        })
        return debug

    def _maybe_print_safety_box_debug(self, event):
        if not self.debug_safety_box:
            return

        debug = getattr(self, "last_safety_box_debug", {})
        route_debug = getattr(self, "last_route_debug", {})
        print(
            "[DiffusionDriveSafetyBox] "
            f"event={event} "
            f"step={self.step} "
            f"points={debug.get('point_count', 0)} "
            f"nearest_xy={debug.get('nearest_xy', float('nan')):.3f} "
            f"x=[{debug.get('x_min', float('nan')):.3f},{debug.get('x_max', float('nan')):.3f}] "
            f"y=[{debug.get('y_min', float('nan')):.3f},{debug.get('y_max', float('nan')):.3f}] "
            f"z=[{debug.get('z_min', float('nan')):.3f},{debug.get('z_max', float('nan')):.3f}] "
            f"speed={debug.get('speed', float('nan')):.3f} "
            f"desired_speed={debug.get('desired_speed', float('nan')):.3f} "
            f"throttle_before={debug.get('throttle_before', float('nan')):.3f} "
            f"brake_before={debug.get('brake_before')} "
            f"stuck={debug.get('stuck_detector')} "
            f"force_move={debug.get('force_move')} "
            f"route_dist={route_debug.get('distance', float('nan')):.3f} "
            f"route_angle={route_debug.get('angle_diff_deg', float('nan')):.3f} "
            f"route_warning={route_debug.get('warning')}",
            flush=True,
        )

    def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
        """Align LiDAR from one coordinate frame to another."""
        pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
        rot_diff = t_u.normalize_angle(orientation_target - orientation)

        rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                    [np.sin(orientation_target), np.cos(orientation_target), 0.0],
                                    [0.0, 0.0, 1.0]])
        pos_diff = rotation_matrix.T @ pos_diff

        return t_u.algin_lidar(lidar, pos_diff, rot_diff)

    def _add_seeded_diffusion_noise(self, features):
        self.current_diffusion_noise_key = None
        if self.dd_config.diffusion_infer_noise_mode != "seeded":
            return

        self.current_diffusion_noise_key = stable_diffusion_noise_seed(
            self.dd_config.diffusion_infer_noise_seed,
            "carla_online",
            os.environ.get("START_IDX", "unknown"),
            self.step,
        )
        features["diffusion_noise"] = gaussian_noise_from_seed(
            (
                1,
                self.dd_config.num_anchor_modes,
                self.dd_config.trajectory_sampling.num_poses,
                2,
            ),
            self.current_diffusion_noise_key,
            device=self.device,
            dtype=features["camera_feature"].dtype,
        )

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            self.lidar_last = deepcopy(tick_data['lidar'])
            return control

        tick_data = self.tick(input_data)

        # Get current ego state
        ego_x = self.state_log[-1][0]
        ego_y = self.state_log[-1][1]
        ego_theta = self.state_log[-1][2]

        ego_x_last = self.state_log[-2][0]
        ego_y_last = self.state_log[-2][1]
        ego_theta_last = self.state_log[-2][2]

        # Align last half LiDAR scan to current frame
        lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last,
                                       ego_x, ego_y, ego_theta)

        # Concatenate current and last half scans to form full scan
        lidar_current = deepcopy(tick_data['lidar'])
        lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)

        self.lidar_buffer.append(lidar_full)

        # Wait until buffer is filled
        if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
            self.lidar_last = deepcopy(tick_data['lidar'])
            tmp_control = carla.VehicleControl(0.0, 0.0, 1.0)
            self.control = tmp_control
            return tmp_control

        # Prepare LiDAR indices for temporal sequence
        lidar_indices = []
        for i in range(self.config.lidar_seq_len):
            lidar_indices.append(i * self.config.data_save_freq)

        # Voxelize LiDAR and stack temporal frames
        lidar_bev = []
        for i in lidar_indices:
            lidar_point_cloud = deepcopy(self.lidar_buffer[-(i + 1)])

            # Realign historical LiDAR to current frame
            if self.config.realign_lidar and self.config.lidar_seq_len > 1:
                curr_x = self.state_log[-(i + 1)][0]
                curr_y = self.state_log[-(i + 1)][1]
                curr_theta = self.state_log[-(i + 1)][2]

                lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta,
                                                     ego_x, ego_y, ego_theta)

            lidar_histogram = self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                                    use_ground_plane=self.config.use_ground_plane)
            lidar_histogram = torch.from_numpy(lidar_histogram).unsqueeze(0).to(self.device, dtype=torch.float32)
            lidar_bev.append(lidar_histogram)

        lidar_bev_original = torch.cat(lidar_bev, dim=1)
        lidar_bev = lidar_bev_original
        if self.zero_lidar:
            lidar_bev = torch.zeros_like(lidar_bev)

        self.lidar_last = deepcopy(tick_data['lidar'])

        rgb = self._prepare_camera_feature(tick_data['rgb'])

        status = self._build_status(tick_data)

        features = {
            'camera_feature': rgb,
            'lidar_feature': lidar_bev,
            'status_feature': status,
        }
        if self.use_route_condition:
            features['route_condition_feature'] = self._build_route_condition(tick_data)
        self._add_seeded_diffusion_noise(features)
        outputs = self.model(features)

        traj = outputs['trajectory']
        if traj.shape[-1] != 2:
            raise RuntimeError(f"DiffusionDriveAgent expects 2D trajectory output, got {traj.shape}.")
        waypoints = traj

        speed = tick_data['speed'].item()
        self._maybe_record_trajectory_diagnostics(
            outputs=outputs,
            waypoints=waypoints,
            speed=speed,
            ego_state=(ego_x, ego_y, ego_theta),
        )
        desired_speed_override = self._predicted_target_speed(outputs) if self.use_speed_head_controller else None
        steer, throttle, brake = self._control_pid(
            waypoints,
            speed,
            tick_data.get('route_points_ego'),
            desired_speed_override=desired_speed_override,
        )
        stop_for_stop_sign = self._stop_sign_controller_step(speed)
        lidar_event = None

        # Restart mechanism in case the car got stuck.
        if speed < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            emergency_stop = False

            # Match sensor_agent creep protection using the latest full LiDAR scan.
            safety_box = self._filter_safety_box_points(deepcopy(self.lidar_buffer[-1]))
            self.last_safety_box_debug = self._build_safety_box_debug(
                safety_box=safety_box,
                speed=speed,
                throttle_before=throttle,
                brake_before=brake,
            )
            emergency_stop = (len(safety_box) > 0)

            if not emergency_stop:
                lidar_event = "creep"
                print('Detected agent being stuck. Step: ', self.step)
                self._maybe_print_safety_box_debug("creep")
                throttle = max(self.config.creep_throttle, throttle)
                brake = False
                self.force_move -= 1
            else:
                lidar_event = "safety_stop"
                print('Creeping stopped by safety box. Step: ', self.step)
                self._maybe_print_safety_box_debug("safety_stop")
                throttle = 0.0
                brake = True
                self.force_move = self.config.creep_duration

        if stop_for_stop_sign:
            throttle = 0.0
            brake = True

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        self._maybe_record_lidar_contract(
            lidar_current=lidar_current,
            lidar_last_aligned=lidar_last,
            lidar_full=lidar_full,
            lidar_bev_original=lidar_bev_original,
            lidar_bev_model=lidar_bev,
            waypoints=waypoints,
            route_points_ego=tick_data.get('route_points_ego'),
            speed=speed,
            event=lidar_event,
        )
        self._maybe_print_control_debug(speed, stop_for_stop_sign)
        self._maybe_print_route_debug()

        return self.control

    def _maybe_record_lidar_contract(
        self,
        *,
        lidar_current,
        lidar_last_aligned,
        lidar_full,
        lidar_bev_original,
        lidar_bev_model,
        waypoints,
        route_points_ego,
        speed,
        event,
    ):
        if not self.debug_lidar_contract or self.lidar_diagnostic_writer is None:
            return

        periodic_stats = (self.step % self.lidar_debug_interval) == 0
        periodic_dump = self.lidar_dump_interval > 0 and (self.step % self.lidar_dump_interval) == 0
        event_dump = False
        if event is not None:
            last_event_step = self.last_lidar_event_step.get(event)
            event_dump = last_event_step is None or (self.step - last_event_step) >= self.lidar_debug_interval
        if not periodic_stats and not periodic_dump and not event_dump:
            return

        diagnostic_event = event if event is not None else "periodic"
        current_bev = self.data.lidar_to_histogram_features(
            lidar_current,
            use_ground_plane=self.config.use_ground_plane,
        )
        last_bev = self.data.lidar_to_histogram_features(
            lidar_last_aligned,
            use_ground_plane=self.config.use_ground_plane,
        )
        trajectory = waypoints.detach().float().cpu().numpy()
        context = {
            "route_index": os.environ.get("START_IDX"),
            "zero_lidar": bool(self.zero_lidar),
            "speed": float(speed),
            "desired_speed": getattr(self, "last_pid_debug", {}).get("desired_speed"),
            "control": {
                "steer": float(self.control.steer),
                "throttle": float(self.control.throttle),
                "brake": float(self.control.brake),
            },
            "stuck_detector": int(self.stuck_detector),
            "force_move": int(self.force_move),
            "trajectory_end": trajectory[0, -1].tolist(),
            "command": getattr(self, "last_command_debug", {}),
            "pid": getattr(self, "last_pid_debug", {}),
            "route": getattr(self, "last_route_debug", {}),
            "safety_box": getattr(self, "last_safety_box_debug", {}),
        }
        stage_inputs = (
            ("current_half", lidar_current, current_bev),
            ("previous_half_aligned", lidar_last_aligned, last_bev),
            ("full_scan", lidar_full, lidar_bev_original),
            ("model_input", lidar_full, lidar_bev_model),
        )
        records = []
        for stage, points, bev in stage_inputs:
            record = compute_lidar_contract_record(
                points,
                bev,
                self.config,
                source="carla_online",
                stage=stage,
                step=self.step,
                event=diagnostic_event,
                context=context,
            )
            records.append(record)
            self.lidar_diagnostic_writer.record(record)
            print(
                "[DiffusionDriveLidarContract] "
                + json.dumps({
                    "step": self.step,
                    "event": diagnostic_event,
                    "stage": stage,
                    "point_count": record["point_count"],
                    "model_point_count": record["model_point_count"],
                    "above_split_ratio": record["above_split_ratio"],
                    "front_ratio": record["front_ratio"],
                    "left_ratio": record["left_ratio"],
                    "bev_nonzero_ratio": record["bev_nonzero_ratio"],
                    "bev_mean": record["bev_mean"],
                    "bev_saturation_ratio": record["bev_saturation_ratio"],
                }, separators=(",", ":"), sort_keys=True, allow_nan=False),
                flush=True,
            )

        history_entry = {
            "step": self.step,
            "event": diagnostic_event,
            "context": make_json_safe(context),
            "stages": {
                record["stage"]: {
                    "point_count": record["point_count"],
                    "model_point_count": record["model_point_count"],
                    "above_split_ratio": record["above_split_ratio"],
                    "front_ratio": record["front_ratio"],
                    "back_ratio": record["back_ratio"],
                    "left_ratio": record["left_ratio"],
                    "right_ratio": record["right_ratio"],
                    "bev_nonzero_ratio": record["bev_nonzero_ratio"],
                    "bev_mean": record["bev_mean"],
                    "bev_saturation_ratio": record["bev_saturation_ratio"],
                }
                for record in records
            },
        }
        self.lidar_diagnostic_history.append(history_entry)

        if event_dump:
            self.last_lidar_event_step[event] = self.step
        if event_dump or periodic_dump:
            dump_event = event if event_dump else "periodic"
            dump_path = self.lidar_diagnostic_writer.dump_bundle(
                step=self.step,
                event=dump_event,
                arrays={
                    "lidar_current": lidar_current,
                    "lidar_last_aligned": lidar_last_aligned,
                    "lidar_full": lidar_full,
                    "lidar_bev_original": lidar_bev_original,
                    "lidar_bev_model": lidar_bev_model,
                    "trajectory": trajectory,
                    "route_points_ego": route_points_ego,
                },
                metadata={"context": context, "contract_records": records},
                recent_history=self.lidar_diagnostic_history,
            )
            if dump_path is not None:
                print(f"[DiffusionDriveLidarDump] path={dump_path}", flush=True)

    def _maybe_record_trajectory_diagnostics(self, *, outputs, waypoints, speed, ego_state):
        if not self.debug_trajectory:
            return

        mode_index = int(outputs["trajectory_mode_index"][0].item())
        top2_index = outputs["trajectory_mode_top2_index"][0].detach().cpu().tolist()
        top2_prob = outputs["trajectory_mode_top2_prob"][0].detach().float().cpu().tolist()
        top2_endpoint = outputs["trajectory_mode_top2_endpoint"][0].detach().float().cpu().tolist()
        endpoint = waypoints[0, -1].detach().float().cpu().tolist()
        state = [float(value) for value in ego_state]

        previous = self.last_trajectory_diagnostic
        aligned_previous_endpoint = None
        aligned_endpoint_jump = None
        mode_changed = False
        if previous is not None:
            aligned = self.align_lidar(
                np.asarray(
                    [[previous["endpoint"][0], previous["endpoint"][1], 0.0]],
                    dtype=np.float64,
                ),
                previous["ego_state"][0],
                previous["ego_state"][1],
                previous["ego_state"][2],
                state[0],
                state[1],
                state[2],
            )[0, :2]
            aligned_previous_endpoint = aligned.tolist()
            aligned_endpoint_jump = float(
                np.linalg.norm(aligned - np.asarray(endpoint, dtype=np.float64))
            )
            mode_changed = mode_index != previous["mode_index"]

        current = {
            "ego_state": state,
            "endpoint": [float(value) for value in endpoint],
            "mode_index": mode_index,
        }
        self.last_trajectory_diagnostic = current

        periodic = (self.step % self.trajectory_debug_interval) == 0
        jump_event = (
            aligned_endpoint_jump is not None
            and aligned_endpoint_jump >= self.trajectory_jump_warn
        )
        if not periodic and not mode_changed and not jump_event:
            return

        record = make_json_safe(
            {
                "step": int(self.step),
                "route_index": os.environ.get("START_IDX"),
                "speed": float(speed),
                "ego_state": state,
                "noise_mode": self.dd_config.diffusion_infer_noise_mode,
                "noise_seed": int(self.dd_config.diffusion_infer_noise_seed),
                "noise_key": self.current_diffusion_noise_key,
                "mode_index": mode_index,
                "top2_index": [int(value) for value in top2_index],
                "top2_prob": [float(value) for value in top2_prob],
                "mode_margin": float(outputs["trajectory_mode_margin"][0].item()),
                "mode_entropy": float(outputs["trajectory_mode_entropy"][0].item()),
                "mode_entropy_normalized": float(
                    outputs["trajectory_mode_entropy_normalized"][0].item()
                ),
                "top2_endpoint": top2_endpoint,
                "endpoint": endpoint,
                "aligned_previous_endpoint": aligned_previous_endpoint,
                "aligned_endpoint_jump": aligned_endpoint_jump,
                "mode_changed": mode_changed,
                "jump_event": jump_event,
                "command": getattr(self, "last_command_debug", {}),
            }
        )
        line = json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False)
        if self.trajectory_diagnostic_handle is not None:
            self.trajectory_diagnostic_handle.write(line + "\n")
        print("[DiffusionDriveTrajectory] " + line, flush=True)

    def _maybe_print_control_debug(self, speed, stop_for_stop_sign):
        if not self.debug_control or (self.step % self.debug_control_interval) != 0:
            return

        command_debug = getattr(self, "last_command_debug", {})
        pid_debug = getattr(self, "last_pid_debug", {})
        route_debug = getattr(self, "last_route_debug", {})
        speed_head_debug = getattr(self, "last_speed_head_debug", {})
        print(
            "[DiffusionDriveControl] "
            f"step={self.step} "
            f"speed={float(speed):.3f} "
            f"cmd_used={command_debug.get('used')} "
            f"cmd_current={command_debug.get('current')} "
            f"cmd_delayed={command_debug.get('delayed')} "
            f"cmd_delay={command_debug.get('delay_enabled')} "
            f"pid_mode={pid_debug.get('mode')} "
            f"desired_speed={pid_debug.get('desired_speed', float('nan')):.3f} "
            f"turn_ratio={pid_debug.get('turn_ratio', float('nan')):.3f} "
            f"endpoint_dist={pid_debug.get('endpoint_distance', float('nan')):.3f} "
            f"aim_index={pid_debug.get('aim_index')} "
            f"aim=({pid_debug.get('aim_x', float('nan')):.3f},{pid_debug.get('aim_y', float('nan')):.3f}) "
            f"angle={pid_debug.get('angle', float('nan')):.3f} "
            f"raw_angle={pid_debug.get('raw_angle', float('nan')):.3f} "
            f"angle_reset={pid_debug.get('angle_reset_reason')} "
            f"low_speed_steer={pid_debug.get('low_speed_steer')} "
            f"route_dist={route_debug.get('distance', float('nan')):.3f} "
            f"route_angle={route_debug.get('angle_diff_deg', float('nan')):.3f} "
            f"route_warning={route_debug.get('warning')} "
            f"speed_head={speed_head_debug.get('available', False)} "
            f"speed_head_class={speed_head_debug.get('pred_class')} "
            f"speed_head_selection={speed_head_debug.get('selection')} "
            f"speed_head_desired={speed_head_debug.get('desired_speed', float('nan')):.3f} "
            f"speed_head_expected={speed_head_debug.get('expected_speed', float('nan')):.3f} "
            f"speed_head_brake_prob={speed_head_debug.get('brake_prob', float('nan')):.3f} "
            f"speed_head_max_prob={speed_head_debug.get('max_prob', float('nan')):.3f} "
            f"control=(steer={float(self.control.steer):.3f},"
            f"throttle={float(self.control.throttle):.3f},"
            f"brake={float(self.control.brake):.3f}) "
            f"stuck={self.stuck_detector} "
            f"force_move={self.force_move} "
            f"stop_sign={bool(stop_for_stop_sign)}"
        )

    def _maybe_print_route_debug(self):
        if not self.debug_route or (self.step % self.debug_control_interval) != 0:
            return

        route_debug = getattr(self, "last_route_debug", {})
        if not route_debug.get("available"):
            print(f"[DiffusionDriveRoute] step={self.step} available=False", flush=True)
            return

        print(
            "[DiffusionDriveRoute] "
            f"step={self.step} "
            f"points={route_debug.get('points')} "
            f"aim=({route_debug.get('aim_x', float('nan')):.3f},"
            f"{route_debug.get('aim_y', float('nan')):.3f}) "
            f"nearest=({route_debug.get('nearest_x', float('nan')):.3f},"
            f"{route_debug.get('nearest_y', float('nan')):.3f}) "
            f"distance={route_debug.get('distance', float('nan')):.3f} "
            f"angle_diff_deg={route_debug.get('angle_diff_deg', float('nan')):.3f} "
            f"segment={route_debug.get('segment_index')} "
            f"warning={route_debug.get('warning')}",
            flush=True,
        )

    def _stabilize_ukf_covariance(self):
        covariance = np.asarray(self.ukf.P, dtype=np.float64)
        if covariance.shape != (4, 4) or not np.all(np.isfinite(covariance)):
            self.ukf.P = _UKF_INITIAL_COVARIANCE.copy()
            return

        covariance = 0.5 * (covariance + covariance.T)
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if min_eigenvalue < _UKF_MIN_COVARIANCE_EIGENVALUE:
            covariance += np.eye(4) * (_UKF_MIN_COVARIANCE_EIGENVALUE - min_eigenvalue)
        self.ukf.P = covariance

    def _reset_ukf(self, measurement, reason):
        self.ukf.x = np.asarray(measurement, dtype=np.float64)
        self.ukf.P = _UKF_INITIAL_COVARIANCE.copy()
        self.filter_initialized = True
        self.ukf_reinitializations += 1
        print(
            "[DiffusionDriveUKF] reset "
            f"count={self.ukf_reinitializations} "
            f"reason={reason} "
            f"measurement=({measurement[0]:.3f},{measurement[1]:.3f},"
            f"{measurement[2]:.3f},{measurement[3]:.3f})",
            flush=True,
        )

    def destroy(self, results=None):  # pylint: disable=unused-argument
        """
        Leaderboard local evaluator calls destroy(results), while the base
        AutonomousAgent only defines destroy(self). Override it here so cleanup
        works with both evaluator variants.
        """
        trajectory_handle = getattr(self, "trajectory_diagnostic_handle", None)
        if trajectory_handle is not None:
            try:
                trajectory_handle.flush()
                trajectory_handle.close()
                print(
                    f"[DiffusionDriveTrajectory] records={self.trajectory_diagnostic_path}",
                    flush=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[DiffusionDriveTrajectory] failed to close diagnostics: {exc}", flush=True)
            finally:
                self.trajectory_diagnostic_handle = None

        writer = getattr(self, "lidar_diagnostic_writer", None)
        if writer is not None:
            try:
                writer.close(
                    context={
                        "source": "carla_online",
                        "route_index": os.environ.get("START_IDX"),
                        "last_step": getattr(self, "step", None),
                        "zero_lidar": getattr(self, "zero_lidar", None),
                        "results_type": type(results).__name__ if results is not None else None,
                    },
                    recent_history=getattr(self, "lidar_diagnostic_history", ()),
                )
                print(
                    f"[DiffusionDriveLidarContract] summary={writer.output_dir / 'summary.json'}",
                    flush=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[DiffusionDriveLidarContract] failed to finalize diagnostics: {exc}", flush=True)
        for attr in ("model", "data", "config", "dd_config", "ukf", "stop_sign_criteria", "hero_actor"):
            if hasattr(self, attr):
                delattr(self, attr)


# Filter Functions

def bicycle_model_forward(x, dt, steer, throttle, brake):
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x


def measurement_function_hx(vehicle_state):
    return vehicle_state


def state_mean(state, wm):
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def measurement_mean(state, wm):
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x


def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y


def residual_measurement_h(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y
