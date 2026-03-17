"""DiffusionDrive port for CARLA leaderboard.

Env vars:
  - DIFFUSIONDRIVE_CHECKPOINT: path to model weights (.pth/.ckpt).
  - DIFFUSIONDRIVE_ANCHOR_PATH: path to plan anchor .npy file (required).
  - DIFFUSIONDRIVE_BACKBONE_PATH: optional timm backbone weights.
"""

import os
import math
from collections import deque
from copy import deepcopy

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

from diffusiondrive.config import DiffusionDriveConfig
from diffusiondrive.model import V2TransfuserModel


# Leaderboard function that selects the class used as agent.
def get_entry_point():
    return "DiffusionDriveAgent"


def strtobool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1", "True")


class DiffusionDriveAgent(autonomous_agent.AutonomousAgent):
    """DiffusionDrive agent for CARLA leaderboard."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):  # pylint: disable=unused-argument
        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.MAP if os.environ.get(
            "CHALLENGE_TRACK_CODENAME") == "MAP" else autonomous_agent.Track.SENSORS

        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # CARLA config for sensors and control
        self.config = GlobalConfig()
        self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

        # DiffusionDrive model config
        self.dd_config = DiffusionDriveConfig()
        self.dd_config.lidar_min_x = self.config.min_x
        self.dd_config.lidar_max_x = self.config.max_x
        self.dd_config.lidar_min_y = self.config.min_y
        self.dd_config.lidar_max_y = self.config.max_y
        self.dd_config.lidar_resolution_height = self.config.lidar_resolution_height
        self.dd_config.lidar_resolution_width = self.config.lidar_resolution_width
        self.dd_config.lidar_seq_len = self.config.lidar_seq_len
        self.dd_config.use_ground_plane = self.config.use_ground_plane
        self.dd_config.__post_init__()

        # Anchor + backbone paths
        anchor_path = os.environ.get("DIFFUSIONDRIVE_ANCHOR_PATH", "")
        backbone_path = os.environ.get("DIFFUSIONDRIVE_BACKBONE_PATH", "")
        if not backbone_path:
            default_backbone = os.path.join(os.getcwd(), "pytorch_model.bin")
            if os.path.exists(default_backbone):
                backbone_path = default_backbone
        self.dd_config.plan_anchor_path = anchor_path
        self.dd_config.bkb_path = backbone_path

        if not self.dd_config.plan_anchor_path:
            raise RuntimeError("DIFFUSIONDRIVE_ANCHOR_PATH is required for DiffusionDrive (plan anchor .npy).")
        # Build model
        self.model = V2TransfuserModel(self.dd_config).to(self.device)
        self.model.eval()

        # Load weights
        ckpt_path = os.environ.get("DIFFUSIONDRIVE_CHECKPOINT", "")
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

        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])
        self.filter_initialized = False
        self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))

        self.prev_speed = None
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

    def _load_checkpoint(self, ckpt_path: str) -> None:
        if torch.cuda.is_available():
            checkpoint = torch.load(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip common prefixes
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("agent."):
                k = k.replace("agent.", "", 1)
            if k.startswith("model."):
                k = k.replace("model.", "", 1)
            if k.startswith("module."):
                k = k.replace("module.", "", 1)
            cleaned[k] = v

        missing_keys, unexpected_keys = self.model.load_state_dict(cleaned, strict=False)
        if missing_keys:
            print(f"Missing keys when loading DiffusionDrive weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading DiffusionDrive weights: {unexpected_keys}")

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

        if not self.filter_initialized:
            self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
            self.filter_initialized = True

        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
        self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
        filtered_state = self.ukf.x
        self.state_log.append(filtered_state)
        result['gps'] = filtered_state[0:2]

        waypoint_route = self._route_planner.run_step(np.append(filtered_state[0:2], gps_pos[2]))
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
        one_hot_command = t_u.command_to_one_hot(self.commands[-2])
        result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)
        result['target_point'] = ego_target_point

        if self.config.two_tp_input:
            ego_target_point_next = t_u.inverse_conversion_2d(target_point_next[:2], result['gps'], result['compass'])
            ego_target_point_next = torch.from_numpy(ego_target_point_next[np.newaxis]).to(self.device, dtype=torch.float32)
            result['target_point_next'] = ego_target_point_next

        result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

        return result

    def _build_status(self, tick_data):
        speed = tick_data['speed'].item()
        if self.prev_speed is None:
            accel = 0.0
        else:
            accel = (speed - self.prev_speed) / self.config.carla_frame_rate
        self.prev_speed = speed

        vel = torch.tensor([[speed, 0.0]], device=self.device, dtype=torch.float32)
        acc = torch.tensor([[accel, 0.0]], device=self.device, dtype=torch.float32)
        status = torch.cat([tick_data['command'], vel, acc], dim=1)
        return status

    def _control_pid(self, waypoints, speed):
        waypoints = waypoints[0].detach().cpu().numpy()
        speed = float(speed)

        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = max(1, one_second // 2)
        desired_speed = np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0

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
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01 or brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            self.tick(input_data)
            return control

        tick_data = self.tick(input_data)

        # Prepare LiDAR BEV
        lidar_histogram = self.data.lidar_to_histogram_features(
            tick_data['lidar'],
            use_ground_plane=self.config.use_ground_plane)
        lidar_bev = torch.from_numpy(lidar_histogram).unsqueeze(0).to(self.device, dtype=torch.float32)

        # Prepare camera (normalize to 0-1 and resize to model input)
        rgb = tick_data['rgb'] / 255.0
        rgb = F.interpolate(rgb, size=(self.dd_config.camera_height, self.dd_config.camera_width), mode='bilinear', align_corners=False)

        status = self._build_status(tick_data)

        outputs = self.model({
            'camera_feature': rgb,
            'lidar_feature': lidar_bev,
            'status_feature': status,
        })

        traj = outputs['trajectory']
        waypoints = traj[:, :, :2]

        steer, throttle, brake = self._control_pid(waypoints, tick_data['speed'].item())
        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        return self.control

    def destroy(self, results=None):  # pylint: disable=unused-argument
        """
        Leaderboard local evaluator calls destroy(results), while the base
        AutonomousAgent only defines destroy(self). Override it here so cleanup
        works with both evaluator variants.
        """
        for attr in ("model", "data", "config", "dd_config", "ukf"):
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
