import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any


@dataclass
class TrajectorySampling:
    time_horizon: float = 4.0
    interval_length: float = 0.5

    @property
    def num_poses(self) -> int:
        # Example: 4.0 / 0.5 = 8; 5.0 / 0.5 = 10.
        return int(round(self.time_horizon / self.interval_length))


@dataclass
class DiffusionDriveConfig:
    """Minimal config for DiffusionDrive inference in CARLA."""

    trajectory_sampling: TrajectorySampling = field(default_factory=TrajectorySampling)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    bkb_path: str = ""
    plan_anchor_path: str = ""
    num_anchor_modes: int = 20

    latent: bool = False
    latent_rad_thresh: float = 4.0 * math.pi / 9.0

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 384
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = 8
    img_horz_anchors: int = 32
    lidar_vert_anchors: int = 8
    lidar_horz_anchors: int = 8

    block_exp: int = 4
    n_layer: int = 2
    n_head: int = 4
    n_scale: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    perspective_downsample_factor: int = 1
    transformer_decoder_join: bool = True
    detect_boxes: bool = True
    use_bev_semantic: bool = True
    use_semantic: bool = False
    use_depth: bool = False
    add_features: bool = True

    # Status feature dims: command_one_hot(6) + speed(1)
    command_dim: int = 6
    speed_dim: int = 1

    # Route condition token dims: target_point(2) + target_point_next(2)
    route_condition_enabled: bool = True
    route_condition_dim: int = 4

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    num_bounding_boxes: int = 30

    # loss weights (kept for completeness)
    trajectory_weight: float = 12.0
    speed_loss_weight: float = 1.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    trajectory_focal_alpha: float = 0.25
    trajectory_focal_gamma: float = 2.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False

    # SpeedHead-v1: two-hot target_speed/brake supervision from B2D measurements.
    speed_head_enabled: bool = True
    speed_head_num_classes: int = 8

    # Diffusion trajectory sampling. These mirror the original NAVSIM defaults
    # but are explicit here so CARLA training and inference runs are reproducible.
    diffusion_num_train_timesteps: int = 1000
    diffusion_beta_schedule: str = "scaled_linear"
    diffusion_prediction_type: str = "sample"
    diffusion_train_timestep_min: int = 0
    diffusion_train_timestep_max: int = 50
    diffusion_infer_step_num: int = 2
    diffusion_infer_timestep_span: int = 20
    diffusion_infer_trunc_timesteps: int = 8
    diffusion_infer_noise_mode: str = "random"
    diffusion_infer_noise_seed: int = 0

    # BEV mapping (not used for CARLA inference)
    bev_semantic_classes: Dict[int, Any] = field(default_factory=dict)

    bev_pixel_width: int = 256
    bev_pixel_height: int = 128
    bev_pixel_size: float = 0.25

    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # optimizer config (unused in inference but kept for compatibility)
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type: str = "AdamW"
    scheduler_type: str = "MultiStepLR"
    cfg_lr_mult: float = 0.5
    opt_paramwise_cfg: Dict[str, Any] = field(default_factory=lambda: {
        "name": {"image_encoder": {"lr_mult": 0.5}}
    })

    def __post_init__(self) -> None:
        # Derive anchor grid sizes if left at defaults
        self.img_vert_anchors = self.camera_height // 32
        self.img_horz_anchors = self.camera_width // 32
        self.lidar_vert_anchors = self.lidar_resolution_height // 32
        self.lidar_horz_anchors = self.lidar_resolution_width // 32

    @property
    def status_dim(self) -> int:
        return self.command_dim + self.speed_dim

    @property
    def condition_token_count(self) -> int:
        return 1 + int(self.route_condition_enabled)

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
