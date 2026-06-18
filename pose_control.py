import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any, Dict, Callable, Tuple, List, Union
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import json
import math
import os
import random
import sys
import shutil
from dataclasses import dataclass
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from transformers import set_seed as transformers_set_seed

# Diffusers/Transformers imports
from diffusers import WanPipeline, AutoencoderKLWan, WanVideoToVideoPipeline

from src.wan import WanTransformer3DModel
from diffusers.video_processor import VideoProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# Image/Video processing
import imageio
import cv2 
import torchvision

# Custom Modules
from src.ttt3r.ttt3r import TTT3RInference, batch_warp_frames_via_3dgs

# Qwen Imports
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_VER = 2
except ModuleNotFoundError:
    flash_attn_varlen_func = None
    FLASH_VER = None

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM
)

try:
    from src.qwen.utils import process_vision_info
except ImportError:
    def process_vision_info(*args):
        return None, None

# -----------------------------------------------------------------
# 🎮 Camera Movement Library
# -----------------------------------------------------------------

# Pre-defined camera movements that can be combined
CAMERA_MOVEMENTS = {
    # Basic Movements - SUBTLE VALUES
    "DOLLY_IN": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.008, "move_right": 0.0, "move_up": 0.0,  # Reduced from 0.05
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "DOLLY_OUT": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": -0.005, "move_right": 0.0, "move_up": 0.0,  # Reduced from -0.05
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "TRUCK_RIGHT": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.004, "move_up": 0.0,  # Reduced from 0.04
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "TRUCK_LEFT": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": -0.004, "move_up": 0.0,  # Reduced from -0.04
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "PEDESTAL_UP": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": -0.005,  # Reduced from 0.04
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "PEDESTAL_DOWN": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.005,  # Reduced from -0.04
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    
    # Rotation Movements - CORRECTED for your coordinate system
    # DISCOVERED MAPPING: roll = tilt (up/down), pitch = pan (left/right), yaw = bank (roll)
    "PAN_RIGHT": {
        "roll": 0.0, "pitch": 5.0, "yaw": 0.0,  # pitch controls pan
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "PAN_LEFT": {
        "roll": 0.0, "pitch": -5.0, "yaw": 0.0,  # pitch controls pan
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "TILT_UP": {
        "roll": 5.0, "pitch": 0.0, "yaw": 0.0,  # roll controls tilt
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "TILT_DOWN": {
        "roll": -5.0, "pitch": 0.0, "yaw": 0.0,  # roll controls tilt
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "ROLL_LEFT": {
        "roll": 0.0, "pitch": 0.0, "yaw": 5.0,  # yaw controls roll/bank
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "ROLL_RIGHT": {
        "roll": 0.0, "pitch": 0.0, "yaw": -5.0,  # yaw controls roll/bank
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    
    # Continuous Rotation - OPTIMIZED
    "ORBIT_RIGHT": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.1, "angular_velocity_yaw": 0.0,
    },
    "ORBIT_LEFT": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": -0.1, "angular_velocity_yaw": 0.0,
    },
    "ORBIT_UP": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.1, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "ORBIT_DONW": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": -0.1, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    
    # Complex Movements - OPTIMIZED based on your values
    "CRANE_UP": {
        "roll": -3.0, "pitch": 0.0, "yaw": 0.0,  # Tilt down slightly while rising
        "move_forward": 0.002, "move_right": 0.0, "move_up": -0.003,  # Rise up
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "CRANE_DOWN": {
        "roll": 3.0, "pitch": 0.0, "yaw": 0.0,  # Tilt up slightly while descending
        "move_forward": 0.002, "move_right": 0.0, "move_up": 0.003,  # Move down
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "TRACKING_SHOT": {
        "roll": 0.0, "pitch": -4.0, "yaw": 0.0,  # Pan left while moving right
        "move_forward": 0.002, "move_right": 0.004, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "REVEAL_SHOT": {
        "roll": -3.0, "pitch": 0.0, "yaw": 0.0,  # Tilt down while rising
        "move_forward": 0.0, "move_right": 0.0, "move_up": -0.008,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": -0.05,
    },
    "ESTABLISHING_SHOT": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.012, "move_right": 0.0, "move_up": 0.0,  # Faster forward
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "STATIC": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    
    # Speed Variants - OPTIMIZED
    "DOLLY_IN_SLOW": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.005, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "DOLLY_IN_FAST": {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.015, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "PAN_RIGHT_SLOW": {
        "roll": 0.0, "pitch": 3.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "PAN_LEFT_SLOW": {
        "roll": 0.0, "pitch": -3.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "TILT_UP_SLOW": {
        "roll": 3.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "TILT_DOWN_SLOW": {
        "roll": -3.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    
    # Additional useful movements
    "ZOOM_IN": {  # Dolly + slight tilt up
        "roll": 1.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.006, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "HERO_SHOT": {  # Dolly in + tilt up + slight pan
        "roll": 2.0, "pitch": 2.0, "yaw": 0.0,
        "move_forward": 0.005, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "FLOAT_UP": {  # Smooth upward float with slight tilt down
        "roll": -2.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": -0.002,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "FLOAT_DOWN": {  # Smooth downward float with slight tilt up
        "roll": 2.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.002,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    },
    "CIRCLE_RIGHT": {  # Orbit with slight inward movement
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.002, "move_right": 0.002, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.1, "angular_velocity_yaw": 0.0,
    },
    "CIRCLE_LEFT": {  # Orbit left with slight inward movement
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.002, "move_right": -0.002, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": -0.1, "angular_velocity_yaw": 0.0,
    },
}


def seed_everything(seed: int):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    # 1. Python standard library
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # 4. PyTorch Determinism (Optional but recommended for strict reproducibility)
    # Note: This might slow down training/inference slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. Hugging Face Transformers (covers Qwen)
    transformers_set_seed(seed)
    
    print(f"🌱 Global seed set to: {seed}")

# -----------------------------------------------------------------
# 🎮 Camera Pose Controller
# -----------------------------------------------------------------

@dataclass
class CameraPoseConfig:
    """Configuration for camera pose control"""
    roll: float = 0.0      # degrees
    pitch: float = 0.0     # degrees
    yaw: float = 0.0       # degrees
    
    # Movement per frame (in normalized units)
    move_forward: float = 0.0   # Forward/backward (Z-axis)
    move_right: float = 0.0     # Left/right (X-axis)
    move_up: float = 0.0        # Up/down (Y-axis)
    
    # Angular velocity per frame (degrees)
    angular_velocity_roll: float = 0.0
    angular_velocity_pitch: float = 0.0
    angular_velocity_yaw: float = 0.0

class CameraPoseController:
    """
    Generates smooth camera poses based on pre-defined parameters.
    Supports both absolute and relative pose specification.
    """
    
    def __init__(self, 
                 height: int = 480, 
                 width: int = 720,
                 focal_length: float = None,
                 smoothing_method: str = "cubic",
                 smoothing_window: int = 5):
        """
        Args:
            height: Video height
            width: Video width
            focal_length: Camera focal length (if None, will be estimated)
            smoothing_method: Method for smoothing ('cubic', 'linear', 'slerp')
            smoothing_window: Window size for additional smoothing
        """
        self.height = height
        self.width = width
        self.focal_length = focal_length or (width * 1.2)  # Reasonable default
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
    def euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles (in degrees) to rotation matrix.
        Convention: ZYX (yaw-pitch-roll)
        
        Args:
            roll, pitch, yaw: Rotation angles in degrees
            
        Returns:
            3x3 rotation matrix
        """
        r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
        return r.as_matrix()
    
    def rotation_matrix_to_euler(self, rot_mat: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (in degrees).
        
        Args:
            rot_mat: 3x3 rotation matrix
            
        Returns:
            (roll, pitch, yaw) in degrees
        """
        r = R.from_matrix(rot_mat)
        yaw, pitch, roll = r.as_euler('zyx', degrees=True)
        return roll, pitch, yaw
    
    def create_pose_matrix(self, 
                          position: np.ndarray, 
                          rotation: np.ndarray) -> np.ndarray:
        """
        Create a 4x4 pose matrix from position and rotation.
        
        Args:
            position: 3D position vector [x, y, z]
            rotation: 3x3 rotation matrix
            
        Returns:
            4x4 pose matrix
        """
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        return pose
    
    def slerp_rotation(self, 
                       rot1: np.ndarray, 
                       rot2: np.ndarray, 
                       t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two rotation matrices.
        
        Args:
            rot1, rot2: 3x3 rotation matrices
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated 3x3 rotation matrix
        """
        r1 = R.from_matrix(rot1)
        r2 = R.from_matrix(rot2)
        
        # SLERP interpolation
        key_rots = R.from_matrix([rot1, rot2])
        key_times = [0, 1]
        slerp = R.from_quat(key_rots.as_quat())
        
        # Interpolate
        r_interp = R.from_quat(
            (1 - t) * r1.as_quat() + t * r2.as_quat()
        )
        r_interp = r_interp / np.linalg.norm(r_interp.as_quat())
        
        return r_interp.as_matrix()
    
    def smooth_poses_savgol(self, poses: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smoothing poses.
        
        Args:
            poses: Array of shape (N, 4, 4)
            window: Window size (must be odd)
            
        Returns:
            Smoothed poses (N, 4, 4)
        """
        from scipy.signal import savgol_filter
        
        if window % 2 == 0:
            window += 1
        if window > len(poses):
            window = len(poses) if len(poses) % 2 == 1 else len(poses) - 1
            
        # Smooth positions
        positions = poses[:, :3, 3]
        smoothed_positions = savgol_filter(positions, window, 3, axis=0)
        
        # Smooth rotations via quaternions
        rotations = [R.from_matrix(pose[:3, :3]) for pose in poses]
        quats = np.array([r.as_quat() for r in rotations])
        smoothed_quats = savgol_filter(quats, window, 3, axis=0)
        
        # Normalize quaternions
        smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
        
        # Reconstruct poses
        smoothed_poses = np.zeros_like(poses)
        for i in range(len(poses)):
            smoothed_poses[i, :3, :3] = R.from_quat(smoothed_quats[i]).as_matrix()
            smoothed_poses[i, :3, 3] = smoothed_positions[i]
            smoothed_poses[i, 3, 3] = 1.0
            
        return smoothed_poses
    
    def generate_poses_from_config(self,
                                num_frames: int,
                                pose_configs: List[CameraPoseConfig],
                                initial_pose: Optional[np.ndarray] = None) -> np.ndarray:
        if initial_pose is None:
            initial_pose = np.eye(4)
            
        frames_per_config = num_frames // len(pose_configs)
        remaining = num_frames % len(pose_configs)
        
        poses = []
        current_pose = initial_pose.copy()
        
        # Track cumulative Euler angles
        current_roll = 0.0
        current_pitch = 0.0
        current_yaw = 0.0
        
        for config_idx, config in enumerate(pose_configs):
            n_frames = frames_per_config + (1 if config_idx < remaining else 0)
            
            # Generate poses for this segment
            segment_poses, end_euler = self._generate_segment_poses(
                n_frames, config, current_pose, 
                (current_roll, current_pitch, current_yaw)  # Pass Euler angles
            )
            
            poses.append(segment_poses)
            current_pose = segment_poses[-1].copy()
            
            # Update cumulative angles
            current_roll, current_pitch, current_yaw = end_euler
            
        # Concatenate and smooth
        all_poses = np.concatenate(poses, axis=0)
        all_poses = self.smooth_poses_savgol(all_poses, self.smoothing_window)
        
        return torch.from_numpy(all_poses).unsqueeze(0).float()
    
    def _generate_segment_poses(self,
                                n_frames: int,
                                config: CameraPoseConfig,
                                start_pose: np.ndarray,
                                start_euler: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Generate poses for a single segment.
        
        Returns:
            poses: (n_frames, 4, 4)
            end_euler: (roll, pitch, yaw) at end of segment
        """
        poses = np.zeros((n_frames, 4, 4))
        
        # Extract starting position
        start_position = start_pose[:3, 3].copy()
        
        # Use provided Euler angles or extract from rotation matrix
        if start_euler is not None:
            start_roll, start_pitch, start_yaw = start_euler
            print(f"✓ Using tracked Euler: roll={start_roll:.2f}°, pitch={start_pitch:.2f}°, yaw={start_yaw:.2f}°")
        else:
            start_rotation = start_pose[:3, :3].copy()
            start_roll, start_pitch, start_yaw = self.rotation_matrix_to_euler(start_rotation)
            print(f"⚠️ Extracting Euler: roll={start_roll:.2f}°, pitch={start_pitch:.2f}°, yaw={start_yaw:.2f}°")
        
        # Target rotation (absolute)
        target_roll = start_roll + config.roll
        target_pitch = start_pitch + config.pitch
        target_yaw = start_yaw + config.yaw
        
        print(f"   Delta: roll={config.roll:.2f}°, Target: roll={target_roll:.2f}°")
        
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            
            # Interpolate rotation
            if self.smoothing_method == "cubic":
                t_smooth = self._smooth_step(t)
            else:
                t_smooth = t
                
            curr_roll = start_roll + (target_roll - start_roll) * t_smooth + config.angular_velocity_roll * i
            curr_pitch = start_pitch + (target_pitch - start_pitch) * t_smooth + config.angular_velocity_pitch * i
            curr_yaw = start_yaw + (target_yaw - start_yaw) * t_smooth + config.angular_velocity_yaw * i
            
            curr_rotation = self.euler_to_rotation_matrix(curr_roll, curr_pitch, curr_yaw)
            
            # Calculate position
            t_smooth = self._smooth_step(t)
            displacement = np.array([
                config.move_right * i,
                config.move_up * i,
                config.move_forward * i
            ])
            
            curr_position = start_position + curr_rotation @ displacement
            poses[i] = self.create_pose_matrix(curr_position, curr_rotation)
        
        # Return final Euler angles
        end_euler = (curr_roll, curr_pitch, curr_yaw)
        
        return poses, end_euler
    
    def _smooth_step(self, t: float) -> float:
        """
        Smooth interpolation function (ease-in-ease-out).
        
        Args:
            t: Input value [0, 1]
            
        Returns:
            Smoothed value [0, 1]
        """
        return t * t * (3.0 - 2.0 * t)
    
    def generate_intrinsics(self, num_frames: int) -> torch.Tensor:
        """
        Generate camera intrinsics matrix.
        
        Args:
            num_frames: Number of frames
            
        Returns:
            Tensor of shape (1, num_frames, 3, 3)
        """
        K = np.array([
            [self.focal_length, 0, self.width / 2.0],
            [0, self.focal_length, self.height / 2.0],
            [0, 0, 1]
        ])
        
        # Repeat for all frames
        intrinsics = np.tile(K[np.newaxis, :, :], (num_frames, 1, 1))
        return torch.from_numpy(intrinsics).unsqueeze(0).float()
    
    def visualize_trajectory(self, poses: torch.Tensor, save_path: str):
        """
        Visualize camera trajectory as a 3D plot.
        
        Args:
            poses: Tensor of shape (1, num_frames, 4, 4)
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        poses_np = poses[0].cpu().numpy()
        positions = poses_np[:, :3, 3]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Camera Path')
        
        # Plot start and end
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='g', s=100, marker='o', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='r', s=100, marker='s', label='End')
        
        # Plot camera orientations at intervals
        n_arrows = min(20, len(positions))
        indices = np.linspace(0, len(positions)-1, n_arrows, dtype=int)
        
        for idx in indices:
            pos = positions[idx]
            rot = poses_np[idx, :3, :3]
            # Forward direction (negative Z-axis in camera coordinates)
            forward = rot @ np.array([0, 0, -0.5])
            ax.quiver(pos[0], pos[1], pos[2], 
                     forward[0], forward[1], forward[2],
                     color='orange', alpha=0.6, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"📊 Trajectory visualization saved to: {save_path}")



def combine_movements(*movement_names: str) -> Dict[str, float]:
    """
    Combine multiple camera movements by summing their parameters.
    
    Args:
        *movement_names: Names of movements to combine (e.g., "DOLLY_IN", "PAN_RIGHT")
        
    Returns:
        Combined movement dictionary
        
    Example:
        combined = combine_movements("DOLLY_IN", "PAN_RIGHT")
    """
    combined = {
        "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        "move_forward": 0.0, "move_right": 0.0, "move_up": 0.0,
        "angular_velocity_roll": 0.0, "angular_velocity_pitch": 0.0, "angular_velocity_yaw": 0.0,
    }
    
    for name in movement_names:
        name = name.strip()
        if name not in CAMERA_MOVEMENTS:
            raise ValueError(f"Unknown movement: '{name}'. Available: {list(CAMERA_MOVEMENTS.keys())}")
        
        movement = CAMERA_MOVEMENTS[name]
        for key in combined:
            combined[key] += movement[key]
    
    return combined


def parse_movement_string(movement_str: str) -> Dict[str, float]:
    """
    Parse a movement string like "DOLLY_IN + PAN_RIGHT" into a combined movement.
    
    Args:
        movement_str: String with movement names separated by '+' or '|'
        
    Returns:
        Combined movement dictionary
    """
    # Handle multiple separators
    if '+' in movement_str:
        movement_names = movement_str.split('+')
    elif '|' in movement_str:
        movement_names = movement_str.split('|')
    else:
        movement_names = [movement_str]
    
    movement_names = [name.strip() for name in movement_names]
    return combine_movements(*movement_names)


def parse_chunk_poses(chunk_poses_config: List[Union[str, List[str], Dict]]) -> List[CameraPoseConfig]:
    """
    Parse chunk poses configuration into CameraPoseConfig objects.
    
    Args:
        chunk_poses_config: List of movements, which can be:
            - String: "DOLLY_IN" or "DOLLY_IN + PAN_RIGHT"
            - List: ["DOLLY_IN", "PAN_RIGHT"]
            - Dict: {"roll": 0.0, "pitch": 0.0, ...}
            
    Returns:
        List of CameraPoseConfig objects
    """
    parsed_configs = []
    
    for i, chunk_config in enumerate(chunk_poses_config):
        try:
            # Convert OmegaConf types to native Python types
            if hasattr(chunk_config, '__class__'):
                class_name = chunk_config.__class__.__name__
                if 'ListConfig' in class_name:
                    # Convert OmegaConf ListConfig to list
                    chunk_config = list(chunk_config)
                elif 'DictConfig' in class_name:
                    # Convert OmegaConf DictConfig to dict
                    chunk_config = dict(chunk_config)
            
            if isinstance(chunk_config, str):
                # String format: "DOLLY_IN" or "DOLLY_IN + PAN_RIGHT"
                movement_dict = parse_movement_string(chunk_config)
                parsed_configs.append(CameraPoseConfig(**movement_dict))
                print(f"  Chunk {i}: {chunk_config}")
                
            elif isinstance(chunk_config, (list, tuple)):
                # List format: ["DOLLY_IN", "PAN_RIGHT"]
                movement_names = [name.strip() for name in chunk_config]
                movement_dict = combine_movements(*movement_names)
                parsed_configs.append(CameraPoseConfig(**movement_dict))
                print(f"  Chunk {i}: {' + '.join(movement_names)}")
                
            elif isinstance(chunk_config, dict):
                # Direct dict format
                parsed_configs.append(CameraPoseConfig(**chunk_config))
                print(f"  Chunk {i}: Custom movement")
                
            else:
                raise ValueError(f"Invalid chunk config type: {type(chunk_config)}")
                
        except Exception as e:
            raise ValueError(f"Error parsing chunk {i}: {e}")
    
    return parsed_configs




# -----------------------------------------------------------------
# 📂 Path Manager
# -----------------------------------------------------------------

class PathManager:
    def __init__(self, config):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = f"{timestamp}"
        self.exp_dir = os.path.join(config.experiment.output_root, dir_name)
        
        self.dirs = {
            "root": self.exp_dir,
            "chunks": os.path.join(self.exp_dir, "chunks"),
            "final": os.path.join(self.exp_dir, "final"),
            "warped_images": os.path.join(self.exp_dir, "warped_images"), 
            "captions": os.path.join(self.exp_dir, "captions"),
            "temp": os.path.join(self.exp_dir, "temp"),
            "visualizations": os.path.join(self.exp_dir, "visualizations")  # NEW: for camera trajectory
        }
        
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        OmegaConf.save(config, os.path.join(self.exp_dir, "config.yaml"))
        print(f"📁 Experiment directory created at: {self.exp_dir}")

    def get_chunk_path(self, chunk_idx): return os.path.join(self.dirs["chunks"], f"chunk_{chunk_idx:03d}.mp4")
    def get_3dgs_paths(self, chunk_idx): 
        return (os.path.join(self.dirs["warped_images"], f"chunk_{chunk_idx:03d}_3dgs.ply"),
                os.path.join(self.dirs["warped_images"], f"vis_chunk_{chunk_idx:03d}"))
    def get_final_video_path(self): return os.path.join(self.dirs["final"], "final_long_video.mp4")
    def get_temp_caption_video_path(self): return os.path.join(self.dirs["temp"], "caption_ref.mp4")
    def get_source_gen_path(self, suffix=".mp4"): return os.path.join(self.dirs["temp"], f"source_gen{suffix}")

# -----------------------------------------------------------------
# 📝 Qwen Captioner Classes
# -----------------------------------------------------------------

VL_EN_SYS_PROMPT = """
You are an expert **3D Scene Analyst**. Your goal is to write precise, high-quality English captions for static scene reconstruction. You must accurately describe the visible environment and logically **extrapolate** the static geometry that exists beyond the current view to assist in generating novel views.

**Task Requirements:**

1.  **Static Environment Only:** Describe the scene as a completely **static, frozen environment**. Do **not** mention people, animals, or dynamic actions (e.g., walking, moving cars) if not necessary. If living beings are present in the image, ignore them or describe them as static statues only if absolutely necessary.
2.  **Scene Extrapolation (Crucial):** Logically predict the **layout** beyond the visible frame. Hallucinate the continuation of the environment to support novel view synthesis (e.g., *"The tiled pavement continues around the corner,"* or *"The row of trees likely extends further down the road"*).
3.  **Adaptive Style & Atmosphere:**
    * **If Real-World:** Describe the scene as "photorealistic" or "real-life." Focus on the clarity of the atmosphere (e.g., sunny, overcast, indoor lighting).
    * **If Stylized:** Only describe art styles (e.g., oil painting, sketch) if clearly present in the input. Otherwise, assume reality.
4.  **Visual Texture:** Describe textures and materials naturally to define 3D structure (e.g., *"mossy brick walls," "reflective glass windows," "smooth wooden floor"*).
5.  **Output Format:** Always output in **English**. Keep the response between **80-120 words**.

**Example Output:**
> A photorealistic, sun-drenched suburban street corner. A clean asphalt road with faded white markings curves gently to the right, bordered by a concrete sidewalk and well-maintained green lawns. The environment implies a residential neighborhood extending in all directions, with similar houses and fences likely lining the street beyond the frame. Tall oak trees cast static, dappled shadows across the pavement. The lighting is bright and natural, highlighting the rough texture of the road and the softness of the grass.
"""
SYSTEM_PROMPT_TYPES = {int(b'010', 2): VL_EN_SYS_PROMPT}

@dataclass
class PromptOutput(object):
    status: bool; prompt: str; seed: int; system_prompt: str; message: str

class QwenPromptExpander:
    model_dict = {"QwenVL2.5_7B": "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5_14B": "Qwen/Qwen2.5-14B-Instruct"}
    def __init__(self, model_name=None, device=0, is_vl=False, **kwargs):
        if model_name is None: model_name = 'Qwen2.5_14B' if not is_vl else 'QwenVL2.5_7B'
        self.device, self.is_vl, self.model_name = device, is_vl, model_name
        if (not os.path.exists(self.model_name)) and (self.model_name in self.model_dict):
            self.model_name = self.model_dict[self.model_name]

        if self.is_vl:
            self.processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=256*28*28, max_pixels=1280*28*28, use_fast=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16 if FLASH_VER == 2 else "auto",
                attn_implementation="flash_attention_2" if FLASH_VER == 2 else None, device_map="cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto", attn_implementation="flash_attention_2" if FLASH_VER == 2 else None, device_map="cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def __call__(self, prompt, system_prompt=None, image=None, seed=-1, style_ref=None, **kwargs):
        if system_prompt is None: system_prompt = VL_EN_SYS_PROMPT
        if seed < 0: seed = random.randint(0, sys.maxsize)
        
        if style_ref:
            if prompt:
                prompt = f"Style Reference: \"{style_ref}\". {prompt}"
            else:
                prompt = f"Style Reference: \"{style_ref}\". Describe this video frame maintaining this style."
        
        self.model = self.model.to(self.device)
        
        if self.is_vl and image:
            if not isinstance(image, list): image = [image]
            msgs = [{'role': 'system', 'content': [{"type": "text", "text": system_prompt}]}, 
                    {"role": "user", "content": [{"type": "text", "text": prompt}] + [{"video": p} for p in image]}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(msgs)
            inputs = self.processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(self.device)
            ids = self.model.generate(**inputs, max_new_tokens=512)
            out = self.processor.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, ids)], skip_special_tokens=True)[0]
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            ids = self.model.generate(**inputs, max_new_tokens=512)
            out = self.tokenizer.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, ids)], skip_special_tokens=True)[0]
            
        self.model = self.model.to("cpu")
        return PromptOutput(status=True, prompt=out, seed=seed, system_prompt=system_prompt, message="")

# -----------------------------------------------------------------
# 🛠️ Helper Functions
# -----------------------------------------------------------------

def extrapolate_camera_path(poses, intrinsics, n_new_frames, smoothing_window=10):
    """Extrapolates camera path using LOCAL velocity and angular velocity."""
    if n_new_frames <= 0: return poses, intrinsics
    
    N = poses.shape[1]
    window = max(2, min(N - 1, smoothing_window))
    print(f"📐 Extrapolating {n_new_frames} frames (Local Momentum, window={window})...")
    
    poses_np = poses.cpu().numpy().squeeze(0)
    intrinsics_np = intrinsics.cpu().numpy().squeeze(0)
    
    start_idx = N - window
    local_lin_vels = []
    local_ang_vels = []
    
    for i in range(start_idx, N - 1):
        P_curr = poses_np[i]
        P_next = poses_np[i+1]
        R_curr, t_curr = P_curr[:3, :3], P_curr[:3, 3]
        R_next, t_next = P_next[:3, :3], P_next[:3, 3]
        global_disp = t_next - t_curr
        local_vel = R_curr.T @ global_disp
        local_lin_vels.append(local_vel)
        R_diff = R_curr.T @ R_next
        rot_vec = R.from_matrix(R_diff).as_rotvec()
        local_ang_vels.append(rot_vec)
        
    avg_lin_vel = np.mean(np.stack(local_lin_vels), axis=0)
    avg_ang_vel = np.mean(np.stack(local_ang_vels), axis=0)
    R_rel_step = R.from_rotvec(avg_ang_vel)
    
    new_poses = []
    last_pose = poses_np[-1]
    curr_R = last_pose[:3, :3]
    curr_t = last_pose[:3, 3]
    
    for _ in range(n_new_frames):
        step_global = curr_R @ avg_lin_vel
        curr_t = curr_t + step_global
        curr_R_obj = R.from_matrix(curr_R)
        curr_R = (curr_R_obj * R_rel_step).as_matrix()
        new_mat = np.eye(4)
        new_mat[:3, :3] = curr_R
        new_mat[:3, 3] = curr_t
        new_poses.append(new_mat)
        
    new_poses = np.stack(new_poses, axis=0)
    full_poses_np = np.concatenate([poses_np, new_poses], axis=0)
    last_intrin = intrinsics_np[-1:]
    new_intrinsics = np.repeat(last_intrin, n_new_frames, axis=0)
    full_intrinsics_np = np.concatenate([intrinsics_np, new_intrinsics], axis=0)
    
    return (torch.from_numpy(full_poses_np).unsqueeze(0).to(poses.device, dtype=poses.dtype), 
            torch.from_numpy(full_intrinsics_np).unsqueeze(0).to(intrinsics.device, dtype=intrinsics.dtype))

def get_relative_poses(poses_c2w, origin_idx):
    B, N = poses_c2w.shape[:2]
    idx_expanded = origin_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, 4, 4)
    P_origin = torch.gather(poses_c2w, dim=1, index=idx_expanded)
    R_origin = P_origin[..., :3, :3]
    t_origin = P_origin[..., :3, 3:4]
    R_origin_inv = R_origin.transpose(-1, -2)
    t_origin_inv = -torch.matmul(R_origin_inv, t_origin)
    P_origin_inv = torch.eye(4, device=poses_c2w.device, dtype=poses_c2w.dtype).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    P_origin_inv[..., :3, :3] = R_origin_inv
    P_origin_inv[..., :3, 3:4] = t_origin_inv
    return torch.matmul(P_origin_inv, poses_c2w)

def preprocess_video_from_path(video_path: str, height: int, width: int, max_frames: Optional[int] = None) -> torch.Tensor:
    if not os.path.exists(video_path): raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        h_orig, w_orig = frame.shape[:2]
        scale = max(width / w_orig, height / h_orig)
        h_r, w_r = int(h_orig * scale), int(w_orig * scale)
        rescaled = cv2.resize(frame, (w_r, h_r), interpolation=cv2.INTER_LANCZOS4)
        y, x = (h_r - height) // 2, (w_r - width) // 2
        frames.append(cv2.cvtColor(rescaled[y:y+height, x:x+width], cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames: break
    cap.release()
    if not frames: raise ValueError("No frames read.")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(torch.float32) / 255.0

def retrieve_latents(encoder_output, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode() if sample_mode == "argmax" else encoder_output.latent_dist.sample()
    return encoder_output.latents

def get_sigmas(timesteps, schedule_timesteps, sigmas, device, n_dim=4, dtype=torch.float32):
    timesteps = timesteps.to(device)
    orig_shape = timesteps.shape
    indices = torch.argmax((timesteps.flatten().unsqueeze(-1) == schedule_timesteps.unsqueeze(0)).int(), dim=1)
    sigma = sigmas[indices].reshape(orig_shape)
    while len(sigma.shape) < n_dim: sigma = sigma.unsqueeze(-1)
    return sigma.to(dtype)

def save_video_from_tensor(video_tensor, filepath, fps=10):
    if video_tensor.dim() == 5: video_tensor = video_tensor.squeeze(0)
    video_data = (video_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu()
    torchvision.io.write_video(filename=filepath, video_array=video_data, fps=fps, video_codec='libx264')

def erode_mask(mask, kernel_size=3):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    padding = kernel_size // 2
    conv = F.conv2d(mask.float(), kernel, padding=padding)
    return (conv == kernel_size ** 2).float()

def visualize_schedule(sched_mat, save_path):
    if isinstance(sched_mat, torch.Tensor):
        data = sched_mat.detach().cpu().numpy()
    else:
        data = sched_mat
    S, B, T, C, H, W = data.shape
    data = data[:, 0, :, 0, :, :] 
    norm_data = (data / 1000.0 * 255).astype(np.uint8)
    grid_h, grid_w = S * H, T * W
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for s in range(S):
        for t in range(T):
            heatmap = norm_data[s, t]
            colored_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
            canvas[s*H:(s+1)*H, t*W:(t+1)*W] = colored_map
    cv2.imwrite(save_path, canvas)
    print(f"📊 Schedule visualization saved to: {save_path}")

# -----------------------------------------------------------------
# 🧠 Inference Logic
# -----------------------------------------------------------------

@torch.no_grad()
def flow_matching_sample_step(diffusion_model, x, curr_noise, next_noise, val_ts, val_sigmas, 
                              p_emb, neg_p_emb, guidance_scale, attention_kwargs=None):
    dtype, device = diffusion_model.dtype, x.device
    n_dim = len(x.shape)
    curr_noise_flat = curr_noise.flatten()
    next_noise_flat = next_noise.flatten()
    curr_sigma = get_sigmas(curr_noise_flat, val_ts, val_sigmas, device, n_dim, dtype).reshape(curr_noise.shape)
    next_sigma = get_sigmas(next_noise_flat, val_ts, val_sigmas, device, n_dim, dtype).reshape(next_noise.shape)
    p_t, p_h, p_w = diffusion_model.config.patch_size
    patch_timesteps = curr_noise[:, ::p_t, :, ::p_h, ::p_w]
    if patch_timesteps.shape[2] == 1: patch_timesteps = patch_timesteps.squeeze(2)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with diffusion_model.cache_context("cond"):
            noise_pred = diffusion_model(x.permute(0,2,1,3,4), patch_timesteps, encoder_hidden_states=p_emb, attention_kwargs=attention_kwargs)[0]
        with diffusion_model.cache_context("uncond"):
            noise_uncond = diffusion_model(x.permute(0,2,1,3,4), patch_timesteps, encoder_hidden_states=neg_p_emb, attention_kwargs=attention_kwargs)[0]
    pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
    pred = pred.float().permute(0, 2, 1, 3, 4)
    x_pred = x + (next_sigma - curr_sigma) * pred
    mask = (curr_noise == next_noise)
    return torch.where(mask, x, x_pred)

@torch.no_grad()
def sample_sequence(diffusion_model, scheduler, xs_warped, valid_masks, prompt_embeds, negative_prompt_embeds, 
                    cfg, generator, current_context_frames, chunk_idx=0, save_dir="./"):
    device, dtype = xs_warped.device, diffusion_model.dtype
    B, T, C, H, W = xs_warped.shape
    S = cfg.inference_params.sampling_timesteps
    scheduler.set_timesteps(S)
    val_ts = torch.cat([scheduler.timesteps, torch.zeros(1)], dim=-1).to(device)
    val_sigmas = scheduler.sigmas.to(device)
    noise = torch.randn(xs_warped.shape, generator=generator, device=device, dtype=torch.float32)
    minxs_step_idx = min(int(cfg.inference_params.minxs * S), S - 1)
    minxs_timestep = scheduler.timesteps[minxs_step_idx].item()
    max_timestep = scheduler.timesteps[0].item()
    start_timesteps = torch.full((B, T, 1, H, W), minxs_timestep, device=device)
    uniform_noise = cfg.inference_params.get("uniform_noise", False)
    if not uniform_noise:
        # non-uniform: invalid pixels get full noise (original WorldWarp behaviour)
        start_timesteps = torch.where(valid_masks.bool(), start_timesteps, torch.tensor(max_timestep, device=device))
    start_sigmas = get_sigmas(start_timesteps.flatten(), val_ts, val_sigmas, device, 5).reshape(B, T, 1, H, W)
    xs_pred = (1.0 - start_sigmas) * xs_warped + start_sigmas * noise
    num_ctx_tokens = (current_context_frames - 1) // cfg.model_params.latent_downsampling_factor[0] + 1
    context_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    context_mask[:, :num_ctx_tokens] = 1
    xs_pred = torch.where(context_mask.view(B, T, 1, 1, 1).bool(), xs_warped, xs_pred)
    base_sched = torch.from_numpy(scheduler.timesteps.numpy()).to(device)
    base_sched = base_sched.view(S, 1, 1, 1, 1, 1).expand(S, B, T, 1, H, W).clone()
    minxs_sched = base_sched.clone(); minxs_sched[:minxs_step_idx] = minxs_timestep
    valid_masks_exp = valid_masks.unsqueeze(0).expand(S, -1, -1, -1, -1, -1)
    if uniform_noise:
        # uniform: all non-context pixels use the same minxs schedule (in-distribution for base VDM)
        sched_mat = minxs_sched.expand(S, B, T, 1, H, W).clone()
    else:
        sched_mat = torch.where(valid_masks_exp, minxs_sched, base_sched)
    ctx_mask_exp = context_mask.view(1, B, T, 1, 1, 1).expand(S, B, T, 1, H, W).bool()
    sched_mat = torch.where(ctx_mask_exp, torch.tensor(0.0, device=device), sched_mat)
    sched_mat = torch.cat([sched_mat, torch.zeros_like(sched_mat[:1])], dim=0)
    vis_filename = f"schedule_vis_chunk_{chunk_idx:03d}.png"
    vis_path = os.path.join(save_dir, vis_filename)
    visualize_schedule(sched_mat, vis_path)
    pbar = tqdm(total=S, desc="Sampling")
    for m in range(S):
        from_lvl, to_lvl = sched_mat[m], sched_mat[m+1]
        xs_prev = xs_pred.clone()
        xs_pred = flow_matching_sample_step(diffusion_model, xs_pred, from_lvl, to_lvl, val_ts, val_sigmas, prompt_embeds, negative_prompt_embeds, cfg.inference_params.guidance_scale)
        xs_pred = torch.where(context_mask.view(B, T, 1, 1, 1).bool(), xs_prev, xs_pred)
        pbar.update(1)
    pbar.close()
    return xs_pred

# -----------------------------------------------------------------
# ⚙️ Configuration (Updated with Pose Control)
# -----------------------------------------------------------------
# NOTE: Config
CONFIG = OmegaConf.create({
    "experiment": {
        "output_root": "output_demo",
        "seed": 32,
    },
    "video_source": {
        "mode": "original", # "original", "t2v", "v2v"
        "prompt": "",

        "negative": "Person, people, pet, animals. Bright tones, static camera, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, unrealistic.",
        "style_strength": 0.7,
        "t2v_length": 49,
    },
    "paths": {
        "base_model_path": "ckpt/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "finetuned_checkpoint_path": "ckpt/worldwarp_latest.ckpt",

        "ttt3r_model_path": "src/ttt3r/cut3r_512_dpt_4_64.pth",
        "caption_model_path": "ckpt/Qwen/Qwen2.5-VL-7B-Instruct",
        "input_video_path": ""
    },
    "loop_params": {
        "n_chunks": 1,
        "output_fps": 30,
    },
    "prompts": {
        "positive": "", 
        "negative": "Person, people, pet, animals. Bright tones, static camera, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, unrealistic.",
    },
    "model_params": {
        "timesteps": 1000, 
        "latent_downsampling_factor": [4, 8],
    },
    "inference_params": {
        "n_frames": 81,
        "context_frames": 1,        
        "context_frames_2nd": 1,    
        "sampling_timesteps": 50,
        "guidance_scale": 5.0,
        "minxs": 0.2,
        "uniform_noise": False,
        "height": 480,
        "width": 720,
        "pose_lr": 1e-8,
        "num_gs_iterations": 500,
        "mask_thres": 0.5
    },  
    "history_guidance": {
        "name": "conditional"
    },
    # NEW: Camera pose control configuration
    "camera_pose_control": {
        "enabled": True,  # Set to True to use pre-defined poses instead of extracted poses
        "smoothing_method": "cubic",  # "cubic", "linear", "slerp"
        "smoothing_window": 7,
        # Define pose for each chunk using movement names or combinations
        "chunk_poses": [
            "DOLLY_IN"
            # "DOLLY_IN + PAN_RIGHT",      # Chunk 0: Combine movements
            # "PAN_LEFT",                   # Chunk 1: Single movement
            # "TRUCK_RIGHT + TILT_UP",     # Chunk 2: Another combination
            # ["DOLLY_IN", "ORBIT_RIGHT"], # Chunk 3: List format also works
            # "TRUCK_RIGHT",                     # Chunk 4: Hold position
        ]
    }
})

class WanVideoGenerator:
    """Modified to support camera pose control"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.experiment.seed)
        
        self.pm = PathManager(cfg)  # PathManager sets up directories in __init__
        
        # Initialize camera pose controller if enabled
        if cfg.camera_pose_control.enabled:
            self.pose_controller = CameraPoseController(
                height=cfg.inference_params.height,
                width=cfg.inference_params.width,
                smoothing_method=cfg.camera_pose_control.smoothing_method,
                smoothing_window=cfg.camera_pose_control.smoothing_window
            )
            print("🎮 Camera Pose Controller initialized")
        else:
            self.pose_controller = None
        
        # These will be initialized later
        self.caption_model = None
        self.vae = None
        self.transformer = None
        self.scheduler = None
        self.text_pipe = None
        self.video_processor = None
        self.ttt3r = None

    def _init_captioner(self, is_vl=True):
        if self.caption_model is not None:
            print("♻️ Cleaning up previous caption model...")
            del self.caption_model
            self.caption_model = None
            torch.cuda.empty_cache()

        model_type = "Vision-Language" if is_vl else "Text-Only"
        print(f"🧠 Initializing {model_type} Captioner (is_vl={is_vl})...")
        try:
            self.caption_model = QwenPromptExpander(
                model_name=self.cfg.paths.caption_model_path, 
                is_vl=is_vl, 
                device=self.device
            )
        except Exception as e:
            print(f"⚠️ Captioner failed to load: {e}")
            self.caption_model = None

    def _load_models(self):
        print("⏳ Loading Core Models...")
        self.vae = AutoencoderKLWan.from_pretrained(self.cfg.paths.base_model_path, subfolder="vae", torch_dtype=torch.float32).to(self.device).eval()
        self.transformer = WanTransformer3DModel.from_pretrained(self.cfg.paths.base_model_path, subfolder="transformer", torch_dtype=self.dtype)
        
        ckpt_path = self.cfg.paths.get("finetuned_checkpoint_path", "")
        if ckpt_path:
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            try:
                self.transformer.load_state_dict(ckpt, strict=True)
            except:
                self.transformer.load_state_dict(ckpt, strict=False)
        else:
            print("Skipping finetuned checkpoint — using base WAN 2.1 weights.")
        self.transformer = self.transformer.to(self.device).eval()
        
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=5)
        self.text_pipe = WanPipeline.from_pretrained(self.cfg.paths.base_model_path, vae=None, transformer=None, torch_dtype=self.dtype).to(self.device)
        self.video_processor = VideoProcessor(vae_scale_factor=self.cfg.model_params.latent_downsampling_factor[1])
        
        print("⏳ Loading TTT3R...")
        self.ttt3r = TTT3RInference(model_path=self.cfg.paths.ttt3r_model_path, device=self.device)

    def get_caption(self, video_path=None, prompt_text=None, style_ref=None):
        if not self.caption_model: 
            return prompt_text or self.cfg.prompts.positive
        
        if video_path: 
            print(f"📝 Captioning video: {video_path}")
            res = self.caption_model(prompt="", tar_lang="en", image=video_path, seed=self.cfg.experiment.seed, style_ref=style_ref).prompt
            print(f"Generated Caption: {res[:100]}...")
            return res
        elif prompt_text:
            print(f"📝 Expanding text prompt: {prompt_text[:50]}...")
            res = self.caption_model(prompt=prompt_text, tar_lang="en", seed=self.cfg.experiment.seed, style_ref=style_ref).prompt
            print(f"Expanded Prompt: {res[:100]}...")
            return res
        return ""

    def prepare_source_video(self):
        mode = self.cfg.video_source.mode
        if mode == "original":
            return self.cfg.paths.input_video_path

        print(f"🎬 Preparing Source Video (Mode: {mode})...")
        base_prompt = self.cfg.video_source.prompt
        enhanced_prompt = self.get_caption(prompt_text=base_prompt)
        neg_prompt = self.cfg.video_source.negative
        
        save_path = self.pm.get_source_gen_path(suffix=".mp4")
        model_id = self.cfg.paths.base_model_path 
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        
        if mode == "t2v":
            print("🌟 Running Text-to-Video Generation...")
            pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
            pipe.to(self.device)
            output = pipe(prompt=enhanced_prompt, negative_prompt=neg_prompt,
                          height=self.cfg.inference_params.height, width=self.cfg.inference_params.width,
                          num_frames=self.cfg.video_source.t2v_length, guidance_scale=5.0).frames[0]
            
        elif mode == "v2v":
            print("🎨 Running Video-to-Video Style Transfer...")
            temp_transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
            pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, transformer=temp_transformer, torch_dtype=torch.bfloat16)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
            pipe.to(self.device)
            video = load_video(self.cfg.paths.input_video_path)
            output = pipe(video=video, prompt=enhanced_prompt, negative_prompt=neg_prompt,
                          height=self.cfg.inference_params.height, width=self.cfg.inference_params.width,
                          guidance_scale=5.0, strength=self.cfg.video_source.style_strength).frames[0]
            del temp_transformer

        export_to_video(output, save_path, fps=self.cfg.loop_params.output_fps)
        print(f"✅ Source video generated: {save_path}")
        del pipe, vae, output
        torch.cuda.empty_cache()
        return save_path

    def run_inference_chunk(self, chunk_idx, input_video_path, chunk_poses, chunk_intrinsics, context_frames, is_first_chunk, style_prompt=None, combined_video_path=None, cache_all=False):
        out_path = self.pm.get_chunk_path(chunk_idx)
        ply_path, vis_dir = self.pm.get_3dgs_paths(chunk_idx)
        
        with torch.no_grad():
            loaded_path = input_video_path
            video_tensor = preprocess_video_from_path(loaded_path, self.cfg.inference_params.height, self.cfg.inference_params.width).unsqueeze(0).to(self.device)
            b, t, c, h_orig, w_orig = video_tensor.shape
        
        if is_first_chunk:
            temp_vid = self.pm.get_temp_caption_video_path()
            save_video_from_tensor(video_tensor, temp_vid)
            caption_ref = temp_vid
        else:
            caption_ref = input_video_path
            
        prompt_text = self.get_caption(video_path=caption_ref, style_ref=style_prompt)
        cap_save_path = os.path.join(self.pm.dirs["captions"], f"chunk_{chunk_idx:03d}.txt")
        with open(cap_save_path, "w") as f: f.write(prompt_text)
        print(f"📝 Caption saved: {cap_save_path}")
        
        with torch.no_grad():
            p_emb, _ = self.text_pipe.encode_prompt(prompt_text, device=self.device, max_sequence_length=512, do_classifier_free_guidance=False)
            n_emb, _ = self.text_pipe.encode_prompt(self.cfg.prompts.negative, device=self.device, max_sequence_length=512, do_classifier_free_guidance=False)
        
        source_idx = 0 if is_first_chunk else t - context_frames
        if source_idx==0:
            source_ids = torch.full((b,), source_idx, dtype=torch.long, device=self.device).expand(b, -1)
            target_ids = torch.arange(source_idx, source_idx+t, device=self.device).expand(b, -1)
        else:
            source_ids = torch.arange(source_idx, t, device=self.device).expand(b, -1)
            target_ids = torch.arange(t-context_frames, t*2-context_frames, device=self.device).expand(b, -1)
        rel_poses = get_relative_poses(chunk_poses, torch.full((b,), 0, dtype=torch.long, device=self.device).expand(b, -1))
        if chunk_idx:
            pass
        warped, valid_masks, _ = batch_warp_frames_via_3dgs(
            source_ids, target_ids, video_tensor, rel_poses, chunk_intrinsics, self.ttt3r, 
            num_gs_iterations=self.cfg.inference_params.num_gs_iterations, 
            save_ckpt=ply_path, vis_save_dir=vis_dir, visualize=False, optimize_poses=False, pose_correction_method=None, smooth_poses=False, smooth_method="savgol", smooth_window=5, conf_threshold=0.8,
            pose_lr=self.cfg.inference_params.pose_lr, mask_thres=self.cfg.inference_params.mask_thres, is_first_chunk=is_first_chunk
        )

        warped, valid_masks = warped.detach(), valid_masks.detach()
        warped = warped.clamp(0.0, 1.0)
        
        with torch.no_grad():
            valid_masks[valid_masks < self.cfg.inference_params.mask_thres] = 0.0
            valid_masks[valid_masks >= self.cfg.inference_params.mask_thres] = 1.0
            sliced_masks = valid_masks[:, ::self.cfg.model_params.latent_downsampling_factor[0]]
            valid_masks_lat = F.interpolate(erode_mask(sliced_masks.reshape(-1, 1, h_orig, w_orig), 15), 
                                            scale_factor=1/self.cfg.model_params.latent_downsampling_factor[1]).reshape(b, -1, 1, h_orig//8, w_orig//8).bool()
            frame_data_slice = slice(0, context_frames) if is_first_chunk else slice(t-context_frames, t)
            warped[:, :context_frames] = video_tensor[:, frame_data_slice]
        
        with torch.no_grad():
            warped_lat = retrieve_latents(self.vae.encode(self.video_processor.preprocess_video(warped, h_orig, w_orig).to(self.device, torch.float32)), "argmax")
            mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(self.device)
            std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(self.device)
            xs_warped = ((warped_lat - mean) * std).permute(0, 2, 1, 3, 4)
            del warped_lat, warped, valid_masks
            torch.cuda.empty_cache()
        
        xs_pred = sample_sequence(self.transformer, self.scheduler, xs_warped.to(self.dtype), valid_masks_lat, 
                                  p_emb.to(self.dtype), n_emb.to(self.dtype), self.cfg, self.generator, context_frames, 
                                  chunk_idx=chunk_idx, save_dir=self.pm.dirs["warped_images"])
        
        with torch.no_grad():
            recon = self.vae.decode(((xs_pred.permute(0, 2, 1, 3, 4).float() / std) + mean), return_dict=False)[0]
            frames = (self.video_processor.postprocess_video(recon, output_type='np')[0] * 255).astype(np.uint8)

        imageio.mimsave(out_path, frames, fps=self.cfg.loop_params.output_fps)
        print(f"💾 Saved chunk to: {out_path}")
        return out_path


    def _generate_controlled_poses(self):
        """
        Generate camera poses from pre-defined configurations.
        
        Returns:
            global_poses: Tensor of shape (1, total_frames, 4, 4)
            global_intrinsics: Tensor of shape (1, total_frames, 3, 3)
        """
        # Calculate total frames needed
        total_frames_needed = 0
        n_fps = self.cfg.inference_params.n_frames
        for i in range(self.cfg.loop_params.n_chunks):
            ctx = self.cfg.inference_params.context_frames if i==0 else self.cfg.inference_params.context_frames_2nd
            if i == 0: 
                s, e = 0, n_fps
            else:
                s = max((n_fps - ctx) * (i - 1), 0)
                e = s + n_fps - ctx + n_fps
            total_frames_needed = max(total_frames_needed, e)
        
        # Parse chunk poses configuration
        print(f"🎮 Parsing camera movements:")
        pose_configs = parse_chunk_poses(self.cfg.camera_pose_control.chunk_poses)
        
        # Generate poses
        print(f"🎬 Generating {total_frames_needed} frames with {len(pose_configs)} pose configs...")
        global_poses = self.pose_controller.generate_poses_from_config(
            num_frames=total_frames_needed,
            pose_configs=pose_configs,
            initial_pose=None
        )
        
        # Generate intrinsics
        global_intrinsics = self.pose_controller.generate_intrinsics(total_frames_needed)
        
        # Visualize trajectory
        traj_path = os.path.join(self.pm.dirs["visualizations"], "camera_trajectory.png")
        self.pose_controller.visualize_trajectory(global_poses, traj_path)
        
        return global_poses, global_intrinsics

    def run(self):
        mode = self.cfg.video_source.mode
        style_prompt = None
        
        # 1. Source Generation
        if mode in ["t2v", "v2v"]:
            self._init_captioner(is_vl=False)
            source_video_path = self.prepare_source_video()
            print("🔄 Switching to Vision-Language Captioner...")
            self._init_captioner(is_vl=True)
            style_prompt = self.cfg.video_source.prompt
        else:
            self._init_captioner(is_vl=True)
            source_video_path = self.prepare_source_video()

        # 2. Load Models
        self._load_models()

        # 3. Trajectory - MODIFIED to support controlled poses
        if self.cfg.camera_pose_control.enabled:
            print("🎮 Generating Camera Path from Pre-defined Poses...")
            global_poses, global_intrinsics = self._generate_controlled_poses()
            global_poses = global_poses.to(self.device)
            global_intrinsics = global_intrinsics.to(self.device)
        else:
            print("🎥 Analyzing Global Camera Path from Video...")
            full_vid = preprocess_video_from_path(
                source_video_path, 
                self.cfg.inference_params.height, 
                self.cfg.inference_params.width, 
                600
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad(): 
                _, global_poses, global_intrinsics = self.ttt3r.inference(full_vid)
            
            global_intrinsics[:, :, 0, -1] = self.cfg.inference_params.width / 2.0
            global_intrinsics[:, :, 1, -1] = self.cfg.inference_params.height / 2.0
            global_intrinsics = global_intrinsics.mean(dim=1, keepdim=True).repeat(
                1, global_poses.shape[1], 1, 1
            )

            # Calculate frames needed and extrapolate if necessary
            total_frames_needed = 0
            n_fps = self.cfg.inference_params.n_frames
            for i in range(self.cfg.loop_params.n_chunks):
                ctx = self.cfg.inference_params.context_frames if i==0 else self.cfg.inference_params.context_frames_2nd
                if i == 0: 
                    s, e = 0, n_fps
                else:
                    s = max((n_fps - ctx) * (i - 1), 0)
                    e = s + n_fps - ctx + n_fps
                total_frames_needed = max(total_frames_needed, e)

            if total_frames_needed > global_poses.shape[1]:
                print(f"⚠️ Need {total_frames_needed} frames, but GT has {global_poses.shape[1]}. Extrapolating...")
                global_poses, global_intrinsics = extrapolate_camera_path(
                    global_poses, 
                    global_intrinsics, 
                    total_frames_needed - global_poses.shape[1] + 10
                )

        # 4. Loop (rest remains the same)
        curr_in = source_video_path
        paths, ctx_log = [], []

        for i in range(self.cfg.loop_params.n_chunks):
            is_first = (i == 0)
            ctx = self.cfg.inference_params.context_frames if is_first else self.cfg.inference_params.context_frames_2nd
            ctx_log.append(ctx)
            n_frames_per_chunk = self.cfg.inference_params.n_frames
            start_frame = ctx * i
            end_frame = start_frame + n_frames_per_chunk

            print(f"\n📍 Chunk {i}: Global Poses [{start_frame}:{end_frame}] | Context: {ctx}")
            if end_frame > global_poses.shape[1]: 
                end_frame = global_poses.shape[1]

            chunk_poses = global_poses[:, start_frame:end_frame]
            chunk_intrinsics = global_intrinsics[:, start_frame:end_frame]
            
            out = self.run_inference_chunk(
                i, curr_in, chunk_poses, chunk_intrinsics, ctx, is_first, 
                style_prompt=style_prompt
            )
            paths.append(out)
            curr_in = out
            torch.cuda.empty_cache()

        print("🎞️ Stitching...")
        with imageio.get_writer(self.pm.get_final_video_path(), fps=self.cfg.loop_params.output_fps) as writer:
            for i, path in enumerate(paths):
                reader = imageio.get_reader(path)
                skip = 0 if i == 0 else ctx_log[i]
                for k, frame in enumerate(reader):
                    if k >= skip: 
                        writer.append_data(frame)
                reader.close()
        print(f"✅ Done: {self.pm.get_final_video_path()}")


if __name__ == "__main__":
    seed_everything(CONFIG.experiment.seed)
    WanVideoGenerator(CONFIG).run()