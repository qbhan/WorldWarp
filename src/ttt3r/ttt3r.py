import sys
import os

import torch
import imageio

import numpy as np
from copy import deepcopy
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import torchvision

from pytorch3d.ops import sample_farthest_points

from src.ttt3r.gs_utils import rgb_to_sh, downsample_with_open3d, inverse_sigmoid
from gsplat.rendering import rasterization

from src.ttt3r.dust3r.inference import inference_recurrent_lighter
from src.ttt3r.dust3r.model import ARCroco3DStereo
from src.ttt3r.dust3r.utils.camera import pose_encoding_to_camera
from src.ttt3r.dust3r.post_process import estimate_focal_knowing_depth
from simple_knn._C import distCUDA2

def preprocess_video_from_path(
    video_path: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    frame_interval: int = 1,
    max_frames: Optional[int] = None
) -> torch.Tensor:
    """Load video and return tensor in [0, 1] range."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")
    if (height is None) != (width is None):
        raise ValueError("Both height and width must be provided, or neither.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_interval == 0:
            processed_frame = frame

            if height is not None and width is not None:
                h_orig, w_orig = frame.shape[:2]
                scale = max(width / w_orig, height / h_orig)
                h_rescaled, w_rescaled = int(h_orig * scale), int(w_orig * scale)
                interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
                rescaled_frame = cv2.resize(frame, (w_rescaled, h_rescaled), interpolation=interpolation)
                
                y_start = (h_rescaled - height) // 2
                x_start = (w_rescaled - width) // 2
                processed_frame = rescaled_frame[y_start:y_start + height, x_start:x_start + width]
            
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1
        if max_frames is not None and len(frames) >= max_frames: break
            
    cap.release()

    if not frames:
        raise ValueError("No frames could be read from the video.")

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np)
    video_tensor = video_tensor.permute(0, 3, 1, 2).to(torch.float32) / 255.0

    return video_tensor

def unproject_depth_to_points(
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image: np.ndarray,
    confidence_threshold: float = 0.0,
    max_points: int = 100000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unproject depth map to 3D points in world coordinates.
    
    Args:
        depth_map: (H, W) depth values
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) camera-to-world transformation (c2w)
        image: (H, W, 3) RGB image
        confidence_threshold: Minimum depth value to consider valid
        max_points: Maximum number of points to sample
    
    Returns:
        points_3d: (N, 3) 3D points in world space
        points_xy: (N, 2) 2D coordinates (x, y)
        points_rgb: (N, 3) RGB colors
    """
    H, W = depth_map.shape
    
    # Create pixel grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
    
    # Filter valid depth points
    valid_mask = depth_map > confidence_threshold
    
    # Randomly subsample if too many points
    valid_indices = np.where(valid_mask.flatten())[0]
    if len(valid_indices) > max_points:
        selected = np.random.choice(valid_indices, max_points, replace=False)
        mask_flat = np.zeros(H * W, dtype=bool)
        mask_flat[selected] = True
        valid_mask = mask_flat.reshape(H, W)
    
    pixels_flat = pixels.reshape(-1, 3)[valid_mask.flatten()]
    depths_flat = depth_map.flatten()[valid_mask.flatten()]
    
    # Unproject to camera space
    inv_K = np.linalg.inv(intrinsic)
    points_cam = (inv_K @ pixels_flat.T).T * depths_flat[:, None]
    points_cam_homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    
    # Transform to world space (extrinsic is camera-to-world)
    points_world = (extrinsic @ points_cam_homo.T).T[:, :3]
    
    # Get 2D pixel coordinates (x, y)
    points_xy = pixels_flat[:, :2]
    
    # Get RGB colors
    colors = image[valid_mask]
    
    return points_world, points_xy, colors

def visualize_pose_corrections(
    original_poses: torch.Tensor,
    corrected_poses: torch.Tensor,
    source_ids: torch.Tensor,
    save_dir: str = "visualization"
):
    """
    Visualize camera pose corrections.
    
    Creates visualizations showing:
    1. 3D trajectory comparison (original vs corrected)
    2. Position correction magnitude over time
    3. Per-axis corrections (X, Y, Z)
    4. Statistics text file
    
    Args:
        original_poses: (T, 4, 4) original camera poses from TTT3R
        corrected_poses: (T, 4, 4) corrected poses after optimization
        source_ids: (B, num_sources) source frame indices
        save_dir: Directory to save visualizations
        
    Example:
        >>> original_poses = torch.rand(50, 4, 4)
        >>> corrected_poses = original_poses + torch.randn(50, 4, 4) * 0.01
        >>> source_ids = torch.tensor([[0, 10, 20, 30, 40]])
        >>> visualize_pose_corrections(original_poses, corrected_poses, source_ids, "results")
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    T = original_poses.shape[0]
    src_ids = source_ids[0].cpu().numpy()
    
    # Extract camera positions (translation components)
    orig_positions = original_poses[:, :3, 3].cpu().numpy()  # (T, 3)
    corr_positions = corrected_poses[:, :3, 3].cpu().numpy()  # (T, 3)
    
    print("  - Creating pose correction visualization...")
    
    # ========================================================================
    # 1. Create 3-panel figure
    # ========================================================================
    fig = plt.figure(figsize=(15, 5))
    
    # Panel 1: 3D trajectory comparison
    ax = fig.add_subplot(131, projection='3d')
    
    # Plot trajectories
    ax.plot(orig_positions[:, 0], orig_positions[:, 1], orig_positions[:, 2], 
            'b-', alpha=0.5, linewidth=2, label='Original (TTT3R)')
    ax.plot(corr_positions[:, 0], corr_positions[:, 1], corr_positions[:, 2], 
            'r-', alpha=0.7, linewidth=2, label='Corrected (Optimized)')
    
    # Mark source frames with different markers
    ax.scatter(orig_positions[src_ids, 0], 
              orig_positions[src_ids, 1], 
              orig_positions[src_ids, 2], 
              c='blue', s=100, marker='o', 
              edgecolors='black', linewidths=2, 
              label='Source (Original)', zorder=5)
    
    ax.scatter(corr_positions[src_ids, 0], 
              corr_positions[src_ids, 1], 
              corr_positions[src_ids, 2], 
              c='red', s=100, marker='s', 
              edgecolors='black', linewidths=2, 
              label='Source (Corrected)', zorder=5)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title('Camera Trajectory Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Position error magnitude over time
    ax = fig.add_subplot(132)
    
    # Compute L2 norm of position differences
    position_errors = np.linalg.norm(orig_positions - corr_positions, axis=1)
    
    # Plot error curve
    ax.plot(position_errors, 'g-', linewidth=2, label='All frames')
    
    # Highlight source frames
    ax.scatter(src_ids, position_errors[src_ids], 
              c='red', s=100, marker='o', 
              edgecolors='black', linewidths=2, 
              label='Source frames', zorder=5)
    
    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_ylabel('Position Error (L2 norm)', fontsize=10)
    ax.set_title('Position Correction Magnitude', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel 3: Per-axis corrections
    ax = fig.add_subplot(133)
    
    # Compute corrections per axis
    corrections = corr_positions - orig_positions  # (T, 3)
    
    # Plot each axis
    ax.plot(corrections[:, 0], 'r-', alpha=0.7, linewidth=1.5, label='X correction')
    ax.plot(corrections[:, 1], 'g-', alpha=0.7, linewidth=1.5, label='Y correction')
    ax.plot(corrections[:, 2], 'b-', alpha=0.7, linewidth=1.5, label='Z correction')
    
    # Add zero reference line
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Mark source frames with vertical lines
    for src_id in src_ids:
        ax.axvline(x=src_id, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_ylabel('Position Correction', fontsize=10)
    ax.set_title('Per-Axis Corrections', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Save figure
    plt.tight_layout()
    fig_path = f"{save_dir}/pose_corrections.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path}")
    
    # ========================================================================
    # 2. Create statistics text file
    # ========================================================================
    stats_path = f"{save_dir}/pose_correction_stats.txt"
    
    with open(stats_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CAMERA POSE CORRECTION STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total frames in sequence: {T}\n")
        f.write(f"Source frames used for optimization: {src_ids.tolist()}\n")
        f.write(f"Number of source frames: {len(src_ids)}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average position correction:  {position_errors.mean():.6f}\n")
        f.write(f"Median position correction:   {np.median(position_errors):.6f}\n")
        f.write(f"Maximum position correction:  {position_errors.max():.6f} (frame {position_errors.argmax()})\n")
        f.write(f"Minimum position correction:  {position_errors.min():.6f} (frame {position_errors.argmin()})\n")
        f.write(f"Std deviation:                {position_errors.std():.6f}\n\n")
        
        f.write("Per-Axis Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"X-axis - Mean: {corrections[:, 0].mean():.6f}, Std: {corrections[:, 0].std():.6f}\n")
        f.write(f"Y-axis - Mean: {corrections[:, 1].mean():.6f}, Std: {corrections[:, 1].std():.6f}\n")
        f.write(f"Z-axis - Mean: {corrections[:, 2].mean():.6f}, Std: {corrections[:, 2].std():.6f}\n\n")
        
        f.write("Source Frame Corrections:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Frame':<10} {'L2 Error':<15} {'X':<12} {'Y':<12} {'Z':<12}\n")
        f.write("-" * 70 + "\n")
        
        for src_id in src_ids:
            f.write(f"{src_id:<10} "
                   f"{position_errors[src_id]:<15.6f} "
                   f"{corrections[src_id, 0]:<12.6f} "
                   f"{corrections[src_id, 1]:<12.6f} "
                   f"{corrections[src_id, 2]:<12.6f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"    Saved: {stats_path}")
    print(f"    Average pose correction: {position_errors.mean():.6f}")
    
    # Return statistics for potential further use
    return {
        'mean_error': position_errors.mean(),
        'median_error': np.median(position_errors),
        'max_error': position_errors.max(),
        'min_error': position_errors.min(),
        'std_error': position_errors.std(),
        'source_errors': position_errors[src_ids],
    }

def visualize_warping_results(
    video_tensor: torch.Tensor,
    warped_images: torch.Tensor,
    validity_masks: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    save_dir: str = "visualization",
    depth_maps: Optional[torch.Tensor] = None,
    camera_poses: Optional[torch.Tensor] = None,
):
    """
    Visualize warping results with GT images, warped images, and validity masks.
    
    Args:
        video_tensor: (B, T, C, H, W) original video
        warped_images: (B, T_target, C, H, W) warped images
        validity_masks: (B, T_target, 1, H, W) validity masks
        source_ids: (B, num_sources) source frame indices
        target_ids: (B, T_target) target frame indices
        save_dir: Directory to save visualizations
        depth_maps: Optional (T, H, W) depth maps from TTT3R
        camera_poses: Optional (T, 4, 4) camera poses
    """
    os.makedirs(save_dir, exist_ok=True)
    
    B = video_tensor.shape[0]
    assert B == 1, "Visualization currently supports batch size 1"
    
    # Convert to numpy for visualization
    video_np = video_tensor[0].cpu().numpy()  # (T, C, H, W)
    warped_np = warped_images[0].cpu().numpy()  # (T_target, C, H, W)
    masks_np = validity_masks[0, :, 0].cpu().numpy()  # (T_target, H, W)
    
    src_ids = source_ids[0].cpu().numpy()
    tgt_ids = target_ids[0].cpu().numpy()
    
    T_target = len(tgt_ids)
    
    print(f"Creating visualizations in {save_dir}/...")
    
    # # 1. Create comparison grid for all target views
    # print("  - Creating comparison grid...")
    # fig = plt.figure(figsize=(16, 4 * T_target))
    # gs = GridSpec(T_target, 4, figure=fig, hspace=0.3, wspace=0.1)
    
    # for i, tgt_id in enumerate(tgt_ids):
    #     gt_img = video_np[tgt_id].transpose(1, 2, 0)  # (H, W, C)
    #     warped_img = warped_np[i].transpose(1, 2, 0)  # (H, W, C)
    #     mask = masks_np[i]  # (H, W)
        
    #     # GT image
    #     ax = fig.add_subplot(gs[i, 0])
    #     ax.imshow(np.clip(gt_img, 0, 1))
    #     ax.set_title(f"GT Frame {tgt_id}", fontsize=12, fontweight='bold')
    #     ax.axis('off')
        
    #     # Warped image
    #     ax = fig.add_subplot(gs[i, 1])
    #     ax.imshow(np.clip(warped_img, 0, 1))
    #     ax.set_title(f"Warped Frame {tgt_id}", fontsize=12, fontweight='bold')
    #     ax.axis('off')
        
    #     # Validity mask
    #     ax = fig.add_subplot(gs[i, 2])
    #     im = ax.imshow(mask, cmap='viridis', vmin=0, vmax=1)
    #     ax.set_title(f"Validity Mask", fontsize=12, fontweight='bold')
    #     ax.axis('off')
    #     plt.colorbar(im, ax=ax, fraction=0.046)
        
    #     # Error map (L1 difference)
    #     error = np.abs(gt_img - warped_img).mean(axis=-1)  # (H, W)
    #     ax = fig.add_subplot(gs[i, 3])
    #     im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.3)
    #     ax.set_title(f"Error Map (L1)", fontsize=12, fontweight='bold')
    #     ax.axis('off')
    #     plt.colorbar(im, ax=ax, fraction=0.046)
    
    # plt.suptitle(f"3DGS Warping Results\nSource Frames: {src_ids.tolist()}", 
    #              fontsize=16, fontweight='bold', y=0.995)
    # plt.savefig(f"{save_dir}/comparison_grid.png", dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"    Saved: {save_dir}/comparison_grid.png")
    
    # # 2. Create side-by-side comparison for each target
    # print("  - Creating individual comparisons...")
    # individual_dir = f"{save_dir}/individual"
    # os.makedirs(individual_dir, exist_ok=True)
    
    # for i, tgt_id in enumerate(tgt_ids):
    #     gt_img = video_np[tgt_id].transpose(1, 2, 0)
    #     warped_img = warped_np[i].transpose(1, 2, 0)
        
    #     # Side by side
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #     axes[0].imshow(np.clip(gt_img, 0, 1))
    #     axes[0].set_title(f"Ground Truth (Frame {tgt_id})", fontsize=14, fontweight='bold')
    #     axes[0].axis('off')
        
    #     axes[1].imshow(np.clip(warped_img, 0, 1))
    #     axes[1].set_title(f"Warped via 3DGS", fontsize=14, fontweight='bold')
    #     axes[1].axis('off')
        
    #     plt.tight_layout()
    #     plt.savefig(f"{individual_dir}/frame_{tgt_id:04d}.png", dpi=150, bbox_inches='tight')
    #     plt.close()
    # print(f"    Saved: {individual_dir}/frame_*.png ({T_target} images)")
    
    # # 3. Save source frames visualization
    # print("  - Creating source frames visualization...")
    # num_sources = len(src_ids)
    # fig, axes = plt.subplots(1, num_sources, figsize=(5 * num_sources, 5))
    # if num_sources == 1:
    #     axes = [axes]
    
    # for j, src_id in enumerate(src_ids):
    #     src_img = video_np[src_id].transpose(1, 2, 0)
    #     axes[j].imshow(np.clip(src_img, 0, 1))
    #     axes[j].set_title(f"Source Frame {src_id}", fontsize=14, fontweight='bold')
    #     axes[j].axis('off')
    
    # plt.suptitle("Source Views Used for 3DGS Training", fontsize=16, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig(f"{save_dir}/source_frames.png", dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"    Saved: {save_dir}/source_frames.png")
    
    # # 4. Create depth visualization if available
    # if depth_maps is not None:
    #     print("  - Creating depth maps visualization...")
    #     depth_np = depth_maps.cpu().numpy()
        
    #     fig, axes = plt.subplots(1, num_sources, figsize=(5 * num_sources, 5))
    #     if num_sources == 1:
    #         axes = [axes]
        
    #     for j, src_id in enumerate(src_ids):
    #         depth = depth_np[src_id]
    #         # Normalize depth for visualization
    #         depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
    #         im = axes[j].imshow(depth_norm, cmap='turbo')
    #         axes[j].set_title(f"Depth Map (Frame {src_id})", fontsize=14, fontweight='bold')
    #         axes[j].axis('off')
    #         plt.colorbar(im, ax=axes[j], fraction=0.046)
        
    #     plt.suptitle("TTT3R Estimated Depth Maps", fontsize=16, fontweight='bold')
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/depth_maps.png", dpi=150, bbox_inches='tight')
    #     plt.close()
    #     print(f"    Saved: {save_dir}/depth_maps.png")
    
    # 5. Create animated comparison video
    print("  - Creating comparison video...")
    video_frames = []
    for i, tgt_id in enumerate(tgt_ids):
        try:
            gt_img = video_np[tgt_id].transpose(1, 2, 0)
        except Exception:
            gt_img = np.zeros_like(video_np[-1].transpose(1, 2, 0))
        warped_img = warped_np[i].transpose(1, 2, 0)
        mask = masks_np[i]
        
        # Create canvas with GT, Warped, and Mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(np.clip(gt_img, 0, 1))
        axes[0].set_title(f"Ground Truth (Frame {tgt_id})", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(np.clip(warped_img, 0, 1))
        axes[1].set_title(f"3DGS Warped", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        im = axes[2].imshow(mask, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title(f"Validity", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., 1:]
        video_frames.append(frame)
        plt.close()
    
    # Save as video
    video_path = f"{save_dir}/comparison_video.mp4"
    imageio.mimsave(video_path, video_frames, fps=5, codec='libx264')
    print(f"    Saved: {video_path}")
    
    # # 6. Compute and save metrics
    # print("  - Computing metrics...")
    # metrics = compute_warping_metrics(video_tensor, warped_images, target_ids, validity_masks)
    
    # metrics_path = f"{save_dir}/metrics.txt"
    # with open(metrics_path, 'w') as f:
    #     f.write("3DGS Warping Metrics\n")
    #     f.write("=" * 50 + "\n\n")
    #     f.write(f"Source Frames: {src_ids.tolist()}\n")
    #     f.write(f"Target Frames: {tgt_ids.tolist()}\n\n")
    #     f.write(f"Average PSNR: {metrics['psnr']:.2f} dB\n")
    #     f.write(f"Average SSIM: {metrics['ssim']:.4f}\n")
    #     f.write(f"Average L1 Error: {metrics['l1']:.4f}\n")
    #     f.write(f"Average Validity: {metrics['validity']:.4f}\n\n")
    #     f.write("Per-frame metrics:\n")
    #     f.write("-" * 50 + "\n")
    #     for i, tgt_id in enumerate(tgt_ids):
    #         f.write(f"Frame {tgt_id}: PSNR={metrics['psnr_per_frame'][i]:.2f} dB, "
    #                f"SSIM={metrics['ssim_per_frame'][i]:.4f}, "
    #                f"L1={metrics['l1_per_frame'][i]:.4f}\n")
    
    # print(f"    Saved: {metrics_path}")
    
    # # Print summary
    # print("\n" + "=" * 70)
    # print("VISUALIZATION SUMMARY")
    # print("=" * 70)
    # print(f"Average PSNR: {metrics['psnr']:.2f} dB")
    # print(f"Average SSIM: {metrics['ssim']:.4f}")
    # print(f"Average L1 Error: {metrics['l1']:.4f}")
    # print(f"Average Validity: {metrics['validity']:.4f}")
    # print("=" * 70)
    
    # return metrics


class GS3DWarper:
    """
    Warp video frames to novel views using 3D Gaussian Splatting.
    Combines TTT3R depth estimation with fast 3DGS training.
    """
    
    def __init__(
        self,
        ttt3r_model,
        device: str = "cuda",
        num_gs_iterations: int = 100,
        optimize_poses: bool = True,
        pose_lr: float = 1e-4,
    ):
        """
        Args:
            model_path: Path to TTT3R model checkpoint
            device: Device to run on
            ttt3r_size: TTT3R processing resolution
            num_gs_iterations: Number of 3DGS training iterations
            optimize_poses: Whether to jointly optimize camera poses
            pose_lr: Learning rate for pose optimization
        """
        self.device = torch.device(device)
        self.num_gs_iterations = num_gs_iterations
        self.optimize_poses = optimize_poses
        self.pose_lr = pose_lr
        
        # Initialize TTT3R
        self.ttt3r = ttt3r_model
        
        print(f"Initialized GS3DWarper with {num_gs_iterations} training iterations")
        if optimize_poses:
            print(f"  Pose optimization enabled (lr={pose_lr})")
    
    def _initialize_splats_from_depth(
        self,
        video_tensor: torch.Tensor,
        depth_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        source_ids: torch.Tensor,
        conf_threshold: float = 0.1,
        max_points_per_frame: int = 50000,
        init_opacity: float = 0.9999,
        init_scale: float = 1.0,
        max_points: int = 1000000
    ) -> dict:
        """
        Initialize 3DGS splats from depth maps of source views.
        
        Args:
            video_tensor: (1, T, C, H, W)
            depth_maps: (T, H, W)
            camera_poses: (T, 4, 4) camera-to-world
            intrinsics: (T, 3, 3)
            source_ids: (B, num_sources) indices of source frames
            conf_threshold: Minimum depth value
            max_points_per_frame: Max points to sample per frame
            init_opacity: Initial opacity
            init_scale: Initial scale
            
        Returns:
            splats: Dictionary of GS parameters
        """
        
        B = source_ids.shape[0]
        assert B == 1, "Currently only supports batch size 1"
        
        src_ids = source_ids[0].cpu().numpy()  # (num_sources,)
        video_np = (video_tensor[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        
        all_points_3d = []
        all_colors = []
        
        # Collect 3D points from all source views
        for src_id in src_ids:
            depth = depth_maps[src_id].numpy()
            extrinsic = camera_poses[src_id].numpy()  # c2w
            intrinsic = intrinsics[src_id].numpy()
            image = video_np[src_id]
            
            # Unproject to 3D
            points_3d, _, colors = unproject_depth_to_points(
                depth, intrinsic, extrinsic, image,
                confidence_threshold=conf_threshold,
                max_points=max_points_per_frame
            )
            
            all_points_3d.append(points_3d)
            all_colors.append(colors / 255.0)  # Normalize to [0, 1]
        
        # Concatenate all points
        points = torch.from_numpy(np.vstack(all_points_3d)).float().to(self.device)
        rgbs = torch.from_numpy(np.vstack(all_colors)).float().to(self.device)
        if points.shape[0] > max_points:
            points, rgbs = downsample_with_open3d(points, rgbs, max_points)
            points = points.contiguous()
            rgbs = rgbs.contiguous()
        
        N = len(points)
        print(f"Initialized {N} Gaussians from {len(src_ids)} source views")
        
        # Initialize scales based on nearest neighbor distances

        # dist2_avg = (knn(points, min(4, N))[:, 1:] ** 2).mean(dim=-1)
        # dist_avg = torch.sqrt(dist2_avg)
        dist_avg = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist_avg) * init_scale).unsqueeze(-1).repeat(1, 3)
        
        # Initialize other parameters
        # quats = torch.rand((N, 4), device=self.device) # self.device
        
        # quats = F.normalize(quats, dim=-1)
        quats = torch.zeros((N, 4), device=self.device)
        quats[:, 0] = 1.0  # Identity rotation
        opacities = inverse_sigmoid(torch.full((N,), init_opacity, device=self.device))    # self.device
        
        # Convert RGB to spherical harmonics (SH)
        colors_sh = torch.zeros((N, 16, 3), device=self.device)  # degree 3 = 16 coeffs
        colors_sh[:, 0, :] = rgb_to_sh(rgbs)
        
        # Create parameter dictionary
        splats = {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(colors_sh[:, :1, :]),
            "shN": torch.nn.Parameter(colors_sh[:, 1:, :]),
        }
        
        return splats
    
    def _create_optimizers(self, splats: dict, scene_scale: float = 1.0) -> dict:
        """Create optimizers for 3DGS parameters."""
        optimizers = {
            # NOTE: change lr 
            "means": torch.optim.Adam([splats["means"]], lr=1.6e-3 * scene_scale, eps=1e-15),   # 1.6e-4
            "scales": torch.optim.Adam([splats["scales"]], lr=5e-3, eps=1e-15),
            "quats": torch.optim.Adam([splats["quats"]], lr=1e-3, eps=1e-15),
            "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2, eps=1e-15),
            "sh0": torch.optim.Adam([splats["sh0"]], lr=2.5e-3, eps=1e-15),
            "shN": torch.optim.Adam([splats["shN"]], lr=2.5e-3 / 20, eps=1e-15),
        }
        return optimizers
    
    def _create_pose_optimizer(self, pose_delta: torch.nn.Parameter) -> torch.optim.Optimizer:
        """Create optimizer for pose refinement."""
        return torch.optim.Adam([pose_delta], lr=self.pose_lr, eps=1e-15)
    
    def _train_splats(
        self,
        splats: dict,
        optimizers: dict,
        video_tensor: torch.Tensor,
        depth_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        source_ids: torch.Tensor,
        num_iterations: int = 100,
        ssim_lambda: float = 0.2,
        optimize_poses: bool = False,
        depth_lambda: float = 1e-2,  # NEW: Weight for depth loss
        enable_depth_loss: bool = True,  # NEW: Flag to enable/disable depth loss
    ) -> Tuple[dict, Optional[torch.Tensor]]:
        """
        Train 3DGS for a few iterations on source views.
        Optionally jointly optimize camera poses.
        NOW WITH DEPTH REGULARIZATION!
        
        Args:
            splats: GS parameters
            optimizers: Optimizers for each parameter
            video_tensor: (1, T, C, H, W)
            depth_maps: (T, H, W) depth maps from DUSt3R
            camera_poses: (T, 4, 4) camera-to-world
            intrinsics: (T, 3, 3)
            source_ids: (B, num_sources)
            num_iterations: Training iterations
            ssim_lambda: Weight for SSIM loss
            optimize_poses: Whether to optimize poses jointly
            depth_lambda: Weight for depth loss (default: 1e-2)
            enable_depth_loss: Whether to enable depth regularization (default: True)
            
        Returns:
            splats: Trained splats
            pose_corrections: (num_sources, 6) pose corrections [rotation_axis_angle, translation]
        """
        from fused_ssim import fused_ssim
        
        B, T, C, H, W = video_tensor.shape
        src_ids = source_ids[0]  # (num_sources,)
        num_sources = len(src_ids)
        
        # Initialize pose corrections if optimizing
        pose_delta = None
        pose_optimizer = None
        
        if optimize_poses:
            # Initialize pose delta parameters (axis-angle rotation + translation)
            # Start with zero correction
            pose_delta = torch.nn.Parameter(
                torch.zeros(num_sources, 6, device=self.device)
            )  # [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
            pose_optimizer = self._create_pose_optimizer(pose_delta)
            print(f"  Optimizing poses jointly with 3DGS")
        
        print(f"Training 3DGS for {num_iterations} iterations on {num_sources} source views...")
        
        for iteration in range(num_iterations):
            # Randomly sample a source view for this iteration
            idx = torch.randint(0, num_sources, (1,)).item()
            view_id = src_ids[idx].item()
            
            # Get camera parameters
            camtoworld = camera_poses[view_id:view_id+1].to(self.device)  # (1, 4, 4)
            
            # Apply pose correction if optimizing
            if optimize_poses:
                camtoworld = self._apply_pose_delta(camtoworld, pose_delta[idx:idx+1])
            
            K = intrinsics[view_id:view_id+1].to(self.device)  # (1, 3, 3)
            pixels = video_tensor[0, view_id:view_id+1].permute(0, 2, 3, 1).to(self.device)  # (1, H, W, 3)
            
            # Render
            render_mode = "RGB+ED" if enable_depth_loss else "RGB"
            colors, alphas, info = self._rasterize_splats(
                splats, camtoworld, K, W, H, render_mode=render_mode
            )
            if colors.shape[-1] == 4:
                colors, depths = colors[..., 0:3], colors[..., 3:4]
            else:
                colors, depths = colors, None
            
            # Compute loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), 
                pixels.permute(0, 3, 1, 2), 
                padding="valid"
            )
            loss = l1loss * (1.0 - ssim_lambda) + ssimloss * ssim_lambda

            if enable_depth_loss and depths is not None:
                depth_gt = depth_maps[view_id:view_id+1].to(self.device)  # (1, H, W)
                # Get all valid depth points (where depth_gt > 0)
                valid_mask = depth_gt > 0.0  # (1, H, W)
                
                if valid_mask.any():
                    # Get predicted depths at all pixel locations
                    depths_pred = depths.squeeze(-1)  # (1, H, W)
                    
                    # Calculate loss in disparity space (more robust for depth)
                    # This follows the implementation in simple_trainer.py
                    disp_pred = torch.where(
                        depths_pred > 0.0, 
                        1.0 / depths_pred, 
                        torch.zeros_like(depths_pred)
                    )
                    disp_gt = 1.0 / depth_gt  # (1, H, W)
                    
                    # Compute L1 loss on disparity, only on valid pixels
                    depthloss = F.l1_loss(
                        disp_pred[valid_mask], 
                        disp_gt[valid_mask]
                    )
                    
                    # Scale the depth loss and add to total loss
                    # Note: scene_scale factor from simple_trainer can be added if needed
                    loss += depthloss * depth_lambda
            else:
                depthloss = torch.tensor(0.0, device=self.device)
            
            # Backprop
            loss.backward()
            
            # Optimize
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if optimize_poses and pose_optimizer is not None:
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)
            
            if (iteration + 1) % 20 == 0 or iteration == 0:
                pose_info = ""
                if optimize_poses and pose_delta is not None:
                    avg_rot = pose_delta[:, :3].abs().mean().item()
                    avg_trans = pose_delta[:, 3:].abs().mean().item()
                    pose_info = f", pose_delta: rot={avg_rot:.4f} trans={avg_trans:.4f}"
                
                depth_loss_info = ""
                if enable_depth_loss:
                    depth_loss_info = f", depth_loss={depthloss.item():.6f}"
                
                print(f"  Iteration {iteration+1}/{num_iterations}: loss={loss.item():.4f}{depth_loss_info}{pose_info}")
        
        return splats, pose_delta
    
    def _rasterize_splats(
        self,
        splats: dict,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        sh_degree: int = 3,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        render_mode: str = "RGB",
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Rasterize Gaussians to image.
        
        Args:
            splats: GS parameters
            camtoworlds: (N, 4, 4) camera-to-world transforms
            Ks: (N, 3, 3) intrinsics
            width: Image width
            height: Image height
            sh_degree: Max SH degree to use
            
        Returns:
            colors: (N, H, W, 3) rendered colors
            alphas: (N, H, W, 1) rendered alpha
            info: Additional rendering info
        """
        means = splats["means"]
        quats = F.normalize(splats["quats"], dim=-1)
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        
        # Combine SH coefficients
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # (N, K, 3)
        
        # Rasterize
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode
        )
        
        return render_colors, render_alphas, info
    
    def _apply_pose_delta(
        self, 
        poses: torch.Tensor, 
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply pose correction to camera poses.
        
        Args:
            poses: (N, 4, 4) camera-to-world poses
            delta: (N, 6) pose corrections [rot_axis_angle, translation]
            
        Returns:
            corrected_poses: (N, 4, 4) corrected poses
        """
        N = poses.shape[0]
        
        # Extract rotation (axis-angle) and translation
        rot_axis_angle = delta[:, :3]  # (N, 3)
        translation = delta[:, 3:]  # (N, 3)
        
        # Convert axis-angle to rotation matrix
        angle = torch.norm(rot_axis_angle, dim=-1, keepdim=True)  # (N, 1)
        axis = rot_axis_angle / (angle + 1e-8)  # (N, 3)
        
        # Rodrigues' formula
        K = torch.zeros(N, 3, 3, device=poses.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        I = torch.eye(3, device=poses.device).unsqueeze(0).expand(N, -1, -1)
        R_delta = I + torch.sin(angle).unsqueeze(-1) * K + \
                  (1 - torch.cos(angle).unsqueeze(-1)) * torch.bmm(K, K)
        
        # Create delta transformation matrix
        T_delta = torch.eye(4, device=poses.device).unsqueeze(0).expand(N, -1, -1).clone()
        T_delta[:, :3, :3] = R_delta
        T_delta[:, :3, 3] = translation
        
        # Apply: corrected_pose = pose @ T_delta
        corrected_poses = torch.bmm(poses, T_delta)
        
        return corrected_poses
    
    def _propagate_pose_corrections(
        self,
        camera_poses: torch.Tensor,
        source_ids: torch.Tensor,
        pose_delta: torch.Tensor,
        method: str = "spline"
    ) -> torch.Tensor:
        """
        Propagate pose corrections from source frames to all frames.
        
        NEW APPROACH: Instead of fitting a trajectory through corrected source poses,
        we compute the correction (delta) applied to sources, then apply similar
        corrections to the original trajectory based on local motion patterns.
        
        Args:
            camera_poses: (T, 4, 4) original camera poses
            source_ids: (B, num_sources) source frame indices
            pose_delta: (num_sources, 6) learned pose corrections
            method: Propagation method
            
        Returns:
            corrected_poses: (T, 4, 4) corrected camera poses for all frames
        """
        T = camera_poses.shape[0]
        src_ids = source_ids[0].cpu().numpy()
        num_sources = len(src_ids)
        
        print(f"  Propagating pose corrections to all {T} frames using '{method}' method...")
        
        if method == "nearest":
            # Simple nearest neighbor interpolation
            corrected_poses = camera_poses.clone()
            for t in range(T):
                # Find nearest source frame
                nearest_idx = np.argmin(np.abs(src_ids - t))
                delta = pose_delta[nearest_idx:nearest_idx+1]
                corrected_poses[t] = self._apply_pose_delta(
                    camera_poses[t:t+1].to(self.device), delta
                )[0].cpu()
            
        elif method == "spline":
            # NEW: Analyze correction pattern and apply to original trajectory
            corrected_poses = self._apply_correction_pattern(
                camera_poses, src_ids, pose_delta
            )
        elif method == None:
            corrected_poses = camera_poses.clone()
        else:
            raise ValueError(f"Unknown propagation method: {method}")
        
        # Compute average correction magnitude
        pose_changes = []
        for src_id in src_ids:
            orig = camera_poses[src_id, :3, 3].numpy()
            corr = corrected_poses[src_id, :3, 3].numpy()
            pose_changes.append(np.linalg.norm(orig - corr))
        avg_change = np.mean(pose_changes)
        print(f"    Average pose translation change: {avg_change:.4f}")
        
        return corrected_poses
    
    def _apply_correction_pattern(
        self,
        camera_poses: torch.Tensor,
        source_ids: np.ndarray,
        pose_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply learned corrections to original trajectory by analyzing the correction pattern.
        
        Strategy:
        1. Compute corrections applied to source views
        2. Fit smooth functions to model how corrections vary with position/time
        3. Apply interpolated/extrapolated corrections to all frames
        
        Args:
            camera_poses: (T, 4, 4) original poses
            source_ids: (num_sources,) source frame indices
            pose_delta: (num_sources, 6) learned corrections
            
        Returns:
            corrected_poses: (T, 4, 4) corrected poses for all frames
        """
        from scipy.interpolate import interp1d
        
        T = camera_poses.shape[0]
        num_sources = len(source_ids)
        
        print(f"    Analyzing correction pattern from {num_sources} source views...")
        
        # Extract pose deltas (corrections in axis-angle + translation)
        rot_deltas = pose_delta[:, :3].cpu().numpy()  # (num_sources, 3)
        trans_deltas = pose_delta[:, 3:].cpu().numpy()  # (num_sources, 3)
        
        # Create interpolation functions for corrections
        # Use linear extrapolation beyond source range
        rot_interp_funcs = [
            interp1d(source_ids, rot_deltas[:, i], 
                    kind='linear', fill_value='extrapolate')
            for i in range(3)
        ]
        trans_interp_funcs = [
            interp1d(source_ids, trans_deltas[:, i], 
                    kind='linear', fill_value='extrapolate')
            for i in range(3)
        ]
        
        # Apply corrections to all frames
        corrected_poses = torch.zeros_like(camera_poses)
        all_indices = np.arange(T)
        
        # Interpolate corrections for all frames
        interpolated_rot = np.stack([f(all_indices) for f in rot_interp_funcs], axis=-1)  # (T, 3)
        interpolated_trans = np.stack([f(all_indices) for f in trans_interp_funcs], axis=-1)  # (T, 3)
        
        # Convert to tensor
        interpolated_delta = torch.from_numpy(
            np.concatenate([interpolated_rot, interpolated_trans], axis=-1)
        ).float().to(self.device)  # (T, 6)
        
        # Apply delta to each frame
        for t in range(T):
            pose = camera_poses[t:t+1].to(self.device)
            delta = interpolated_delta[t:t+1]
            corrected_poses[t] = self._apply_pose_delta(pose, delta)[0].cpu()
        
        # Print extrapolation info
        if all_indices[0] < source_ids[0]:
            print(f"      Extrapolated corrections for frames 0-{source_ids[0]-1}")
        if all_indices[-1] > source_ids[-1]:
            print(f"      Extrapolated corrections for frames {source_ids[-1]+1}-{T-1}")
        
        return corrected_poses
    
    def _fit_smooth_trajectory(
        self,
        camera_poses: torch.Tensor,
        source_ids: np.ndarray,
        corrected_source_poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fit a smooth trajectory through corrected source poses using spline interpolation.
        
        Args:
            camera_poses: (T, 4, 4) original poses
            source_ids: (num_sources,) source frame indices
            corrected_source_poses: (num_sources, 4, 4) corrected source poses
            
        Returns:
            smooth_poses: (T, 4, 4) smoothly interpolated poses
        """
        from scipy.interpolate import CubicSpline
        from scipy.spatial.transform import Rotation, Slerp
        
        T = camera_poses.shape[0]
        num_sources = len(source_ids)
        
        # Extract positions and rotations from corrected source poses
        corrected_positions = corrected_source_poses[:, :3, 3].cpu().numpy()  # (num_sources, 3)
        corrected_rotations = []
        for i in range(num_sources):
            R = corrected_source_poses[i, :3, :3].cpu().numpy()
            corrected_rotations.append(Rotation.from_matrix(R))
        
        # Fit cubic spline for positions
        position_spline = CubicSpline(source_ids, corrected_positions, bc_type='natural')
        
        # Fit Slerp for rotations
        rotation_slerp = Slerp(source_ids, Rotation.concatenate(corrected_rotations))
        
        # Interpolate for all frames
        all_indices = np.arange(T)
        
        # Clamp indices to valid range for Slerp
        clamped_indices = np.clip(all_indices, source_ids[0], source_ids[-1])
        
        smooth_positions = position_spline(all_indices)  # (T, 3)
        smooth_rotations = rotation_slerp(clamped_indices)  # Rotation object
        
        # For frames outside the source range, extrapolate linearly
        # Positions before first source
        if all_indices[0] < source_ids[0]:
            mask_before = all_indices < source_ids[0]
            # Use the corrected first source pose for all frames before
            smooth_positions[mask_before] = position_spline(source_ids[0])
        
        # Positions after last source
        if all_indices[-1] > source_ids[-1]:
            mask_after = all_indices > source_ids[-1]
            # Use the corrected last source pose for all frames after
            smooth_positions[mask_after] = position_spline(source_ids[-1])
        
        # Construct smooth poses
        smooth_poses = torch.zeros(T, 4, 4)
        for t in range(T):
            smooth_poses[t, :3, :3] = torch.from_numpy(
                smooth_rotations[t].as_matrix()
            ).float()
            smooth_poses[t, :3, 3] = torch.from_numpy(smooth_positions[t]).float()
            smooth_poses[t, 3, 3] = 1.0
        
        return smooth_poses
    
    def _smooth_camera_trajectory(
        self,
        camera_poses: torch.Tensor,
        method: str = "savgol",
        window_size: int = 11,
        polyorder: int = 3,
        sigma: float = 2.0,
    ) -> torch.Tensor:
        """
        Apply additional smoothing to camera trajectory to reduce jitter.
        
        Args:
            camera_poses: (T, 4, 4) camera poses
            method: Smoothing method ("savgol", "gaussian", or "slerp_refine")
            window_size: Window size for Savitzky-Golay or Gaussian filter
            polyorder: Polynomial order for Savitzky-Golay
            sigma: Standard deviation for Gaussian filter
            
        Returns:
            smoothed_poses: (T, 4, 4) smoothed camera poses
        """
        from scipy.signal import savgol_filter
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation, Slerp
        
        T = camera_poses.shape[0]
        
        if T < window_size:
            print(f"    Warning: Sequence too short for smoothing (T={T} < window={window_size}), skipping")
            return camera_poses
        
        print(f"    Applying '{method}' smoothing to camera trajectory...")
        
        # Extract positions and rotations
        positions = camera_poses[:, :3, 3].cpu().numpy()  # (T, 3)
        rotations = []
        for t in range(T):
            R = camera_poses[t, :3, :3].cpu().numpy()
            rotations.append(Rotation.from_matrix(R))
        
        if method == "savgol":
            # Savitzky-Golay filter for positions
            if window_size % 2 == 0:
                window_size += 1  # Must be odd
            smoothed_positions = savgol_filter(positions, window_size, polyorder, axis=0)
            
            # For rotations, convert to quaternions, smooth, and convert back
            quats = np.array([r.as_quat() for r in rotations])  # (T, 4)
            
            # Ensure quaternion continuity (avoid sign flips)
            for i in range(1, T):
                if np.dot(quats[i], quats[i-1]) < 0:
                    quats[i] = -quats[i]
            
            smoothed_quats = savgol_filter(quats, window_size, polyorder, axis=0)
            
            # Renormalize quaternions
            smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
            smoothed_rotations = [Rotation.from_quat(q) for q in smoothed_quats]
            
        elif method == "gaussian":
            # Gaussian filter for positions
            smoothed_positions = gaussian_filter1d(positions, sigma=sigma, axis=0)
            
            # For rotations
            quats = np.array([r.as_quat() for r in rotations])
            
            # Ensure quaternion continuity
            for i in range(1, T):
                if np.dot(quats[i], quats[i-1]) < 0:
                    quats[i] = -quats[i]
            
            smoothed_quats = gaussian_filter1d(quats, sigma=sigma, axis=0)
            smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
            smoothed_rotations = [Rotation.from_quat(q) for q in smoothed_quats]
            
        elif method == "slerp_refine":
            # Use Slerp with downsampled keyframes for smooth interpolation
            step = max(1, T // 20)  # Use ~20 keyframes
            keyframe_indices = np.arange(0, T, step)
            if keyframe_indices[-1] != T - 1:
                keyframe_indices = np.append(keyframe_indices, T - 1)
            
            # Downsample positions and rotations
            key_positions = positions[keyframe_indices]
            key_rotations = [rotations[i] for i in keyframe_indices]
            
            # Fit splines
            from scipy.interpolate import CubicSpline
            position_spline = CubicSpline(keyframe_indices, key_positions, bc_type='natural')
            rotation_slerp = Slerp(keyframe_indices, Rotation.concatenate(key_rotations))
            
            # Interpolate all frames
            all_indices = np.arange(T)
            smoothed_positions = position_spline(all_indices)
            smoothed_rotations = rotation_slerp(all_indices)
            smoothed_rotations = [smoothed_rotations[i] for i in range(T)]
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # Reconstruct smoothed poses
        smoothed_poses = torch.zeros(T, 4, 4)
        for t in range(T):
            smoothed_poses[t, :3, :3] = torch.from_numpy(
                smoothed_rotations[t].as_matrix()
            ).float()
            smoothed_poses[t, :3, 3] = torch.from_numpy(smoothed_positions[t]).float()
            smoothed_poses[t, 3, 3] = 1.0
        
        # Compute smoothing magnitude
        position_changes = np.linalg.norm(
            positions - smoothed_positions, axis=1
        ).mean()
        print(f"      Average position change: {position_changes:.6f}")
        
        return smoothed_poses
    
    def _save_checkpoint(
        self,
        save_path: str,
        splats: torch.nn.ParameterDict,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        video_shape: tuple,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        corrected_poses: Optional[torch.Tensor] = None,
    ):
        """
        Save trained 3DGS checkpoint with metadata.
        
        Args:
            save_path: Path to save checkpoint (e.g., 'checkpoint.pt' or 'model.ply')
            splats: Trained GS parameters
            camera_poses: (T, 4, 4) camera poses (original or corrected)
            intrinsics: (T, 3, 3) intrinsics
            video_shape: Original video tensor shape
            source_ids: Source frame indices used
            target_ids: Target frame indices
            corrected_poses: (T, 4, 4) corrected poses if pose optimization was used
        """
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        ext = os.path.splitext(save_path)[1].lower()
        
        if ext == '.ply':
            # Save as PLY file (standard 3DGS format)
            self._save_as_ply(save_path, splats)
        else:
            # Save as PyTorch checkpoint (default)
            self._save_as_torch(
                save_path, splats, camera_poses, intrinsics,
                video_shape, source_ids, target_ids, corrected_poses
            )
    
    def _save_as_torch(
        self,
        save_path: str,
        splats: torch.nn.ParameterDict,
        camera_poses: torch.Tensor,
        intrinsics: torch.Tensor,
        video_shape: tuple,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        corrected_poses: Optional[torch.Tensor] = None,
    ):
        """Save as PyTorch checkpoint with full metadata."""
        checkpoint = {
            'splats': {k: v.detach().cpu() for k, v in splats.items()},
            'camera_poses': camera_poses.cpu(),
            'intrinsics': intrinsics.cpu(),
            'video_shape': video_shape,
            'source_ids': source_ids.cpu(),
            'target_ids': target_ids.cpu(),
            'num_gaussians': len(splats['means']),
            'training_iterations': self.num_gs_iterations,
            'pose_optimized': self.optimize_poses,
        }
        if corrected_poses is not None:
            checkpoint['corrected_poses'] = corrected_poses.cpu()
        
        torch.save(checkpoint, save_path)
        print(f"  Saved PyTorch checkpoint: {save_path}")
        print(f"  - Number of Gaussians: {checkpoint['num_gaussians']}")
        if corrected_poses is not None:
            print(f"  - Corrected camera poses included")
    
    def _save_as_ply(self, save_path: str, splats: torch.nn.ParameterDict):
        """
        Save as PLY file compatible with standard 3DGS viewers.
        This format can be loaded by viewers like the official 3DGS viewer.
        """
        from gsplat import export_splats
        
        means = splats["means"].detach()
        scales = splats["scales"].detach()
        quats = splats["quats"].detach()
        opacities = splats["opacities"].detach()
        sh0 = splats["sh0"].detach()
        shN = splats["shN"].detach()
        
        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=save_path,
        )
        print(f"  Saved PLY file: {save_path}")
        print(f"  - Number of Gaussians: {len(means)}")
        print(f"  - Can be viewed with 3DGS viewers")
    
    @classmethod
    def load_checkpoint(cls, ckpt_path: str, device: str = "cuda") -> dict:
        """
        Load a saved 3DGS checkpoint.
        
        Args:
            ckpt_path: Path to checkpoint file
            device: Device to load to
            
        Returns:
            checkpoint: Dictionary with splats and metadata
            
        Example:
            >>> ckpt = GS3DWarper.load_checkpoint('model.pt')
            >>> splats = ckpt['splats']
            >>> print(f"Loaded {ckpt['num_gaussians']} Gaussians")
        """
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f"Loaded checkpoint from {ckpt_path}")
        print(f"  - Number of Gaussians: {checkpoint['num_gaussians']}")
        print(f"  - Training iterations: {checkpoint['training_iterations']}")
        return checkpoint
    
    def _compute_geometric_validity_mask(
        self,
        splats: dict,
        target_camtoworld: torch.Tensor,
        target_K: torch.Tensor,
        source_poses: torch.Tensor,
        source_Ks: torch.Tensor,
        source_depths: torch.Tensor,
        width: int,
        height: int,
        depth_threshold: float = 0.1,  # Relative depth error threshold
        min_sources: int = 1,  # Minimum number of source views that should see a point
    ) -> torch.Tensor:
        """
        Compute geometric validity mask based on whether rendered points
        are actually observed in source views.
        
        Args:
            splats: GS parameters
            target_camtoworld: (1, 4, 4) target camera pose
            target_K: (1, 3, 3) target intrinsics
            source_poses: (N_src, 4, 4) source camera poses
            source_Ks: (N_src, 3, 3) source intrinsics
            source_depths: (N_src, H, W) source depth maps
            width: Image width
            height: Image height
            depth_threshold: Relative depth error threshold (e.g., 0.1 = 10%)
            min_sources: Minimum number of sources that should observe the point
            
        Returns:
            validity_mask: (1, H, W, 1) binary mask where 1 = valid, 0 = invalid
        """
        N_src = source_poses.shape[0]
        device = target_camtoworld.device
        
        # Step 1: Render depth for target view
        renders, _, _ = self._rasterize_splats(
            splats, target_camtoworld, target_K, width, height, 
            render_mode="ED"  # or "RGB+ED"
        )

        # Step 2: Extract depth properly
        if renders.shape[-1] == 1:
            target_depth = renders  # (1, H, W, 1) - when using "ED" mode
        elif renders.shape[-1] == 4:
            target_depth = renders[..., 3:4]  # (1, H, W, 1) - when using "RGB+ED" mode
        
        # Step 2: For each pixel, unproject to 3D and check visibility in sources
        # Create pixel grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        pixels = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1).float()  # (H, W, 3)
        
        # Unproject target pixels to 3D camera space
        target_K_inv = torch.inverse(target_K[0])  # (3, 3)
        points_cam = (target_K_inv @ pixels.reshape(-1, 3).T).T  # (H*W, 3)
        points_cam = points_cam * target_depth.reshape(-1, 1)  # Scale by depth
        
        # Transform to world space
        target_c2w = target_camtoworld[0]  # (4, 4)
        points_world = (target_c2w[:3, :3] @ points_cam.T).T + target_c2w[:3, 3]  # (H*W, 3)
        
        # Step 3: Check visibility in each source view
        visibility_count = torch.zeros(height * width, device=device)
        
        for i in range(N_src):
            # Transform to source camera space
            src_w2c = torch.inverse(source_poses[i])  # (4, 4)
            points_src_cam = (src_w2c[:3, :3] @ points_world.T).T + src_w2c[:3, 3]  # (H*W, 3)
            
            # Project to source image
            src_K = source_Ks[i]  # (3, 3)
            points_src_img = (src_K @ points_src_cam.T).T  # (H*W, 3)
            points_src_img = points_src_img[:, :2] / (points_src_img[:, 2:3] + 1e-8)  # (H*W, 2)
            
            # Get depth in source camera
            depth_src_predicted = points_src_cam[:, 2]  # (H*W,)
            
            # Check if within image bounds
            valid_x = (points_src_img[:, 0] >= 0) & (points_src_img[:, 0] < width)
            valid_y = (points_src_img[:, 1] >= 0) & (points_src_img[:, 1] < height)
            valid_depth = depth_src_predicted > 0
            valid_bounds = valid_x & valid_y & valid_depth
            
            # Sample depth from source depth map
            src_depth_map = source_depths[i]  # (H, W)
            x_idx = points_src_img[:, 0].long().clamp(0, width - 1)
            y_idx = points_src_img[:, 1].long().clamp(0, height - 1)
            depth_src_gt = src_depth_map[y_idx, x_idx]  # (H*W,)
            
            # Check if depth matches (geometric consistency)
            depth_error = torch.abs(depth_src_predicted - depth_src_gt) / (depth_src_gt + 1e-8)
            depth_consistent = depth_error < depth_threshold
            
            # Valid if within bounds AND depth consistent
            valid = valid_bounds & depth_consistent & (depth_src_gt > 0)
            visibility_count += valid.float()
        
        # Create mask: valid if seen by at least min_sources
        validity_mask = (visibility_count >= min_sources).float()
        validity_mask = validity_mask.reshape(1, height, width, 1)
        
        return validity_mask


    def warp_with_validity_mask(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        video_tensor: torch.Tensor,
        camera_poses,
        intrinsics,
        reset_interval: int = 1000000,
        conf_threshold: float = 0.1,
        max_points_per_frame: int = 50000,
        save_ckpt: Optional[str] = None,
        pose_correction_method: str = "spline",
        smooth_poses: bool = False,
        smooth_method: str = "savgol",
        smooth_window: int = 11,
        use_geometric_validity: bool = True,
        depth_threshold: float = 0.1,
        min_source_views: int = 1,
        is_first_chunk: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Enhanced warp function with geometric validity mask.
        
        The validity mask now indicates whether a pixel in the target view
        is actually supported by observations from source views, rather than
        just the local 3DGS opacity.
        """
        B, T_video, C, H, W = video_tensor.shape
        T_target = target_ids.shape[1]

        assert B == 1, "Currently only supports batch size 1"
        
        # Step 1: Run TTT3R to get depth and poses
        print("Step 1: Running TTT3R inference...")
        with torch.no_grad():
            depth_maps, cache_poses, _ = self.ttt3r.inference(
                video_tensor, reset_interval=reset_interval
            )
            rel_cache_poses = get_relative_poses(cache_poses, torch.full((B,), T_video-1, dtype=torch.long, device=self.device).expand(B, -1))
        print(f"  Generated depth and poses for {T_video} frames")
        if not is_first_chunk:
            rel_camera_poses = get_relative_poses(camera_poses, torch.full((B,), T_video-1, dtype=torch.long, device=self.device).expand(B, -1))
            rel_camera_poses[:,:T_video,...] = rel_cache_poses
        else:
            rel_camera_poses = camera_poses.clone()
        depth_maps = depth_maps.squeeze(0)
        camera_poses = rel_camera_poses.squeeze(0)
        intrinsics = intrinsics.squeeze(0)
        video_tensor = video_tensor.cpu()
        depth_maps = depth_maps.cpu()
        camera_poses = camera_poses.cpu()
        intrinsics = intrinsics.cpu()
        
        # Store for visualization if needed
        if hasattr(self, '_store_for_viz') and self._store_for_viz:
            self._depth_maps = depth_maps
            self._camera_poses = camera_poses
        
        # Step 2: Initialize 3DGS from source views
        print("Step 2: Initializing 3DGS from source views...")
        splats = self._initialize_splats_from_depth(
            video_tensor, depth_maps, camera_poses, intrinsics,
            source_ids, conf_threshold, max_points_per_frame
        )
        
        # Convert to ParameterDict for training
        splats_param = torch.nn.ParameterDict(splats)
        
        # Step 3: Create optimizers
        optimizers = self._create_optimizers(splats_param)
        
        # Step 4: Train 3DGS (and optionally optimize poses)
        print("Step 3: Training 3DGS...")
        splats_param, pose_delta = self._train_splats(
            splats_param, optimizers, video_tensor, depth_maps,
            camera_poses, intrinsics, source_ids,
            num_iterations=self.num_gs_iterations,
            optimize_poses=self.optimize_poses
        )
        
        # Step 4.5: Propagate pose corrections to all frames if optimized
        corrected_poses = None
        if self.optimize_poses and pose_delta is not None:
            print("Step 3.5: Propagating pose corrections to all frames...")
            corrected_poses = self._propagate_pose_corrections(
                camera_poses, source_ids, pose_delta.detach(),
                method=pose_correction_method
            )
            # Use corrected poses for rendering targets
            camera_poses = corrected_poses
        
        # Step 4.6: Apply additional smoothing to reduce jitter
        if smooth_poses:
            print("Step 3.6: Applying trajectory smoothing to reduce jitter...")
            camera_poses = self._smooth_camera_trajectory(
                camera_poses,
                method=smooth_method,
                window_size=smooth_window,
            )
            # Update corrected_poses if it was set
            if corrected_poses is not None:
                corrected_poses = camera_poses
        
        # Save checkpoint if requested
        if save_ckpt is not None:
            print(f"Saving 3DGS checkpoint to {save_ckpt}...")
            self._save_checkpoint(
                save_ckpt, splats_param, camera_poses, intrinsics, 
                video_tensor.shape, source_ids, target_ids,
                corrected_poses=corrected_poses
            )
        
        # Step 5: Render target views WITH geometric validity masks
        print("Step 4: Rendering target views with geometric validity masks...")
        warped_images = []
        confidence_masks = []
        
        src_ids = source_ids[0].cpu().numpy()  # Source frame indices
        tgt_ids = target_ids[0].cpu().numpy()  # Target frame indices
        
        # Prepare source view data for visibility checking
        source_poses = camera_poses[src_ids].to(self.device)
        source_Ks = intrinsics[src_ids].to(self.device)
        source_depths = depth_maps[src_ids].to(self.device)
        
        with torch.no_grad():
            for tgt_id in tgt_ids:
                camtoworld = camera_poses[tgt_id:tgt_id+1].to(self.device)
                K = intrinsics[tgt_id:tgt_id+1].to(self.device)
                
                # Render RGB and alpha
                colors, alphas, _ = self._rasterize_splats(
                    splats_param, camtoworld, K, W, H
                )
                
                if use_geometric_validity:
                    # Compute geometric validity mask
                    validity_mask = self._compute_geometric_validity_mask(
                        splats_param, camtoworld, K,
                        source_poses, source_Ks, source_depths,
                        W, H, 
                        depth_threshold=depth_threshold,
                        min_sources=min_source_views
                    )
                    
                    # Combine with alpha (both must be valid)
                    final_mask = alphas * validity_mask
                else:
                    final_mask = alphas
                
                warped_images.append(colors)
                confidence_masks.append(final_mask)
        
        # Stack results
        warped_images = torch.stack(warped_images, dim=1)  # (1, T_target, H, W, 3)
        warped_images = warped_images.permute(0, 1, 4, 2, 3)  # (1, T_target, 3, H, W)
        
        confidence_masks = torch.stack(confidence_masks, dim=1)  # (1, T_target, H, W, 1)
        confidence_masks = confidence_masks.permute(0, 1, 4, 2, 3)  # (1, T_target, 1, H, W)
        
        print(f"  Rendered {T_target} target views with geometric validity masks")
        
        return warped_images, confidence_masks, corrected_poses

def compute_warping_metrics(
    video_tensor: torch.Tensor,
    warped_images: torch.Tensor,
    target_ids: torch.Tensor,
    validity_masks: torch.Tensor
) -> dict:
    """
    Compute metrics comparing warped images to ground truth.
    
    Returns:
        metrics: Dictionary with PSNR, SSIM, L1, etc.
    """
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    
    device = video_tensor.device
    B = video_tensor.shape[0]
    tgt_ids = target_ids[0].cpu().numpy()
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    psnr_list = []
    ssim_list = []
    l1_list = []
    
    for i, tgt_id in enumerate(tgt_ids):
        gt = video_tensor[0, tgt_id:tgt_id+1].to(device)  # (1, C, H, W)
        warped = warped_images[0, i:i+1].to(device)  # (1, C, H, W)
        
        psnr = psnr_metric(warped, gt)
        ssim = ssim_metric(warped, gt)
        l1 = F.l1_loss(warped, gt)
        
        psnr_list.append(psnr.item())
        ssim_list.append(ssim.item())
        l1_list.append(l1.item())
    
    return {
        'psnr': np.mean(psnr_list),
        'ssim': np.mean(ssim_list),
        'l1': np.mean(l1_list),
        'validity': validity_masks.mean().item() if 'validity_masks' in locals() else 1.0,
        'psnr_per_frame': psnr_list,
        'ssim_per_frame': ssim_list,
        'l1_per_frame': l1_list,
    }

def batch_warp_frames_via_3dgs(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    video_tensor: torch.Tensor,
    chunk_camera_poses,
    chunk_intrinsics,
    ttt3r_model,
    device: str = "cuda",
    num_gs_iterations: int = 100,
    save_ckpt: Optional[str] = None,
    visualize: bool = False,
    vis_save_dir: str = "visualization",
    optimize_poses: bool = True,
    pose_correction_method: str = "spline",
    smooth_poses: bool = True,
    smooth_method: str = "savgol",
    smooth_window: int = 11,
    conf_threshold: float = 0.5,
    pose_lr: float = 1e-5,
    mask_thres: float = 0.0,
    is_first_chunk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    High-level function to warp video frames using 3DGS.
    
    This is a drop-in replacement for your forward warping function,
    but uses 3DGS as an intermediate representation for better quality.
    
    Args:
        source_ids: (B, num_sources) indices of source frames
        target_ids: (B, T_target) indices of target frames  
        video_tensor: (B, T_video, C, H, W) video with values in [0, 1]
        model_path: Path to TTT3R model checkpoint
        device: Device to run on
        num_gs_iterations: Number of 3DGS training iterations
        save_ckpt: Optional path to save trained 3DGS (e.g., 'model.pt' or 'model.ply')
        visualize: Whether to create visualization outputs
        vis_save_dir: Directory to save visualizations
        optimize_poses: Whether to jointly optimize camera poses during training
        pose_correction_method: Method to propagate corrections ("spline" or "nearest")
        
    Returns:
        warped_images: (B, T_target, C, H, W) warped target images
        validity_mask: (B, T_target, 1, H, W) confidence masks
        corrected_poses: (T_video, 4, 4) corrected camera poses (None if optimize_poses=False)
        
    Example:
        >>> source_ids = torch.tensor([[0, 5, 10]])  # Use frames 0, 5, 10 as sources
        >>> target_ids = torch.tensor([[3, 7, 12]])  # Warp to frames 3, 7, 12
        >>> video = torch.rand(1, 20, 3, 512, 512)  # 20 frames
        >>> 
        >>> warped, masks, corrected_poses = batch_warp_frames_via_3dgs(
        ...     source_ids, target_ids, video,
        ...     model_path="/path/to/ttt3r/model.pth",
        ...     save_ckpt="output/trained_3dgs.ply",  # Save the trained model
        ...     visualize=True,  # Create visualizations
        ...     vis_save_dir="results/visualization",
        ...     optimize_poses=True,  # Refine camera poses
        ...     pose_correction_method="spline"  # Smooth trajectory
        ... )
        >>> print(warped.shape)  # (1, 3, 3, 512, 512)
        >>> print(corrected_poses.shape)  # (20, 4, 4)
    """
    warper = GS3DWarper(
        ttt3r_model=ttt3r_model,
        device=device,
        num_gs_iterations=num_gs_iterations,
        optimize_poses=optimize_poses,
        pose_lr=pose_lr,
    )
    
    # Store depth maps and poses for visualization
    if visualize:
        # Temporarily store these during warping
        warper._store_for_viz = True
    
    warped_images, validity_masks, corrected_poses = warper.warp_with_validity_mask(
        source_ids, target_ids, video_tensor, chunk_camera_poses, chunk_intrinsics,
        save_ckpt=save_ckpt,
        pose_correction_method=pose_correction_method,
        smooth_poses=smooth_poses,
        smooth_method=smooth_method,
        smooth_window=smooth_window,
        conf_threshold=conf_threshold,
        is_first_chunk=is_first_chunk
    )

    validity_masks[validity_masks<mask_thres] = 0.
    
    # Create visualizations if requested
    if visualize:
        depth_maps = warper._depth_maps if hasattr(warper, '_depth_maps') else None
        camera_poses = warper._camera_poses if hasattr(warper, '_camera_poses') else None
        
        visualize_warping_results(
            video_tensor=video_tensor,
            warped_images=warped_images,
            validity_masks=validity_masks,
            source_ids=source_ids,
            target_ids=target_ids,
            save_dir=vis_save_dir,
            depth_maps=depth_maps,
            camera_poses=camera_poses
        )
        
        # Visualize pose corrections if available
        if corrected_poses is not None and camera_poses is not None:
            visualize_pose_corrections(
                camera_poses, corrected_poses, source_ids, 
                save_dir=vis_save_dir
            )
    
    return warped_images, validity_masks, corrected_poses


def create_batch_warp_visualization(
    source_images: torch.Tensor,
    source_depth_maps: torch.Tensor,
    warped_images: torch.Tensor,
    target_images: torch.Tensor,
    target_depth_maps: torch.Tensor,
    validity_masks: torch.Tensor,
    depth_colormap: str = 'viridis',
    error_colormap: str = 'inferno'
) -> torch.Tensor:
    """
    Creates visualization grid for batch of warped images.
    
    Args:
        source_images: (B, C, H, W) source images
        source_depth_maps: (B, H, W) source depth maps
        warped_images: (B, N, C, H, W) warped images
        target_images: (B, N, C, H, W) target images
        target_depth_maps: (B, N, H, W) target depth maps
        validity_masks: (B, N, 1, H, W) validity masks
        
    Returns:
        Grid tensor showing [Source, Source Depth, Warped, Target, Target Depth, Mask, Error]
    """
    B, N = warped_images.shape[:2]
    
    def colorize(data_map, colormap):
        cm = plt.get_cmap(colormap)
        valid_data = data_map[data_map > 0]
        p2 = torch.quantile(valid_data, 0.02) if len(valid_data) > 0 else 0
        p98 = torch.quantile(valid_data, 0.98) if len(valid_data) > 0 else 1
        norm_data = torch.clamp((data_map - p2) / (p98 - p2 + 1e-8), 0, 1)
        colored_np = cm(norm_data.cpu().numpy())[..., :3]
        return torch.from_numpy(colored_np).permute(2, 0, 1)
    
    grid_items = []
    for b in range(B):
        for n in range(N):
            # Colorize depth maps
            source_depth_colored = colorize(source_depth_maps[b], depth_colormap)
            target_depth_colored = colorize(target_depth_maps[b, n], depth_colormap)
            
            # Compute error map
            error_map_raw = torch.abs(warped_images[b, n] - target_images[b, n]) * validity_masks[b, n]
            error_intensity = error_map_raw.mean(dim=0)
            error_colored = colorize(error_intensity, error_colormap)
            
            # Mask as RGB
            mask_rgb = validity_masks[b, n].repeat(3, 1, 1)
            
            # Add all visualizations for this batch item and target
            grid_items.extend([
                source_images[b],
                source_depth_colored,
                warped_images[b, n],
                target_images[b, n],
                target_depth_colored,
                mask_rgb,
                error_colored
            ])
    
    return torchvision.utils.make_grid(grid_items, nrow=7, padding=5)



class TTT3RInference:
    """
    Batch-enabled TTT3R inference for training integration.
    """

    def __init__(self, model_path: str, device: str = "cuda", size: int = 512, model_update_type: str = "ttt3r"):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available. Switching to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.size = size

        self._inference_recurrent_lighter = inference_recurrent_lighter
        self._pose_encoding_to_camera = pose_encoding_to_camera
        self._estimate_focal_knowing_depth = estimate_focal_knowing_depth

        print(f"Loading model from {model_path}...")
        self.model = ARCroco3DStereo.from_pretrained(model_path).to(self.device)
        self.model.config.model_update_type = model_update_type
        self.model.eval()
        print("Model loaded successfully.")

    @staticmethod
    def _preprocess_batch(frames_batch: torch.Tensor, size: int) -> Dict[str, Any]:
        """Preprocess a batch of frames (B, C, H, W) already in [0, 1] range."""
        B, C, H, W = frames_batch.shape
        scale = size / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        resized_batch = TF.resize(frames_batch, [new_h, new_w], antialias=True)
        pad_h, pad_w = size - new_h, size - new_w
        padding = [pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2)]
        padded_batch = TF.pad(resized_batch, padding, fill=0)
        normalized_batch = padded_batch * 2.0 - 1.0
        return {
            'img': normalized_batch,
            'true_shape': np.array([W, H])
        }

    def _prepare_batch_views(self, video_batch: torch.Tensor, reset_interval: int) -> List[Dict[str, Any]]:
        """
        Prepare batched views for a batch of videos.
        """
        B, T, C, H, W = video_batch.shape
        views = []
        
        for i in range(T):
            frame_batch = video_batch[:, i]
            frame_data = self._preprocess_batch(frame_batch, self.size)
            h_proc, w_proc = frame_data['img'].shape[-2:]
            true_shape_tensor = torch.from_numpy(frame_data["true_shape"]).to(self.device).unsqueeze(0).repeat(B, 1)
            
            reset_bool = (i + 1) % reset_interval == 0
            
            view = {
                "img": frame_data["img"].to(self.device),
                "true_shape": true_shape_tensor,
                "idx": i,
                "instance": [f"{b}_{i}" for b in range(B)],
                "camera_pose": torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(B, 1, 1),
                "img_mask": torch.tensor(True, device=self.device).unsqueeze(0).repeat(B), # (B,)
                "ray_map": torch.full((B, 6, h_proc, w_proc), torch.nan, device=self.device),
                "ray_mask": torch.tensor(False, device=self.device).unsqueeze(0).repeat(B), # (B,)
                
                # FIX: Must be (B,) tensor
                "update": torch.tensor(True, device=self.device).unsqueeze(0).repeat(B),
                
                # FIX: Must be (B,) tensor to avoid IndexError on line 1290
                "reset": torch.tensor(reset_bool, device=self.device).unsqueeze(0).repeat(B),
            }
            views.append(view)
            
            if reset_bool and i < T - 1:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False, device=self.device).unsqueeze(0).repeat(B) # (B,)
                views.append(overlap_view)
                
        return views

    def _matrix_cumprod_batch(self, poses: torch.Tensor) -> torch.Tensor:
        """ Batched version of matrix_cumprod. """
        B, T = poses.shape[:2]
        cumulative_poses_list = []
        current_pose = poses[:, 0]
        cumulative_poses_list.append(current_pose)
        for i in range(1, T):
            current_pose = torch.bmm(current_pose, poses[:, i])
            cumulative_poses_list.append(current_pose)
        return torch.stack(cumulative_poses_list, dim=1)

    def _process_batch_output(self, outputs: Dict[str, Any], original_h: int, original_w: int, B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batched output for a batch of videos."""
        
        # reset_mask shape is (T', B)
        reset_mask = torch.stack([view["reset"] for view in outputs["views"]], 0) 
        
        # We only need to check one item in batch since reset is same for all
        reset_mask_1d = reset_mask[:, 0] # (T',)
        
        shifted_reset_mask_1d = torch.cat([
            torch.tensor(False, device=reset_mask_1d.device).unsqueeze(0), 
            reset_mask_1d[:-1]
        ], dim=0)
        
        preds = [pred for pred, mask in zip(outputs["pred"], shifted_reset_mask_1d) if not mask]
        
        reset_mask = reset_mask[~shifted_reset_mask_1d] # (T, B)
        T_filtered = len(preds)

        # --- Get camera poses ---
        pr_poses_encoded = [p["camera_pose"] for p in preds]
        pr_poses_encoded_tensor = torch.stack(pr_poses_encoded, 0) # (T, B, ...)
        
        pr_poses_encoded_flat = pr_poses_encoded_tensor.reshape(T_filtered * B, -1)
        pr_poses_flat = self._pose_encoding_to_camera(pr_poses_encoded_flat.clone())
        pr_poses_tensor = pr_poses_flat.view(T_filtered, B, 1, 4, 4)
        
        pr_poses_tensor_sq = pr_poses_tensor.squeeze(2) # (T, B, 4, 4)
        
        # Check the 1D reset mask
        if reset_mask_1d.any():
            identity = torch.eye(4, device=pr_poses_tensor.device).view(1, 1, 4, 4)
            # `reset_mask_bcast`: (T, B, 1, 1)
            reset_mask_bcast = reset_mask.unsqueeze(-1).unsqueeze(-1)
            reset_poses = torch.where(reset_mask_bcast, pr_poses_tensor_sq, identity)
            
            reset_poses_perm = reset_poses.permute(1, 0, 2, 3) # (B, T, 4, 4)
            cumulative_bases = self._matrix_cumprod_batch(reset_poses_perm) # (B, T, 4, 4)
            
            identity_batch = torch.eye(4, device=pr_poses_tensor.device).unsqueeze(0).repeat(B, 1, 1)
            shifted_bases = torch.cat([identity_batch.unsqueeze(1), cumulative_bases[:, :-1]], dim=1)
            
            pr_poses_tensor_sq_perm = pr_poses_tensor_sq.permute(1, 0, 2, 3)
            final_poses_tensor = torch.einsum('btij,btjk->btik', shifted_bases, pr_poses_tensor_sq_perm)
        else:
            final_poses_tensor = pr_poses_tensor_sq.permute(1, 0, 2, 3)
            
        camera_poses = final_poses_tensor

        # --- Get depth maps and intrinsics (Batched) ---
        pts3ds_self_list = [p["pts3d_in_self_view"] for p in preds]
        pts3ds_self = torch.stack(pts3ds_self_list, dim=0) # (T, B, H_proc, W_proc, 3)
        _, _, H_proc, W_proc, _ = pts3ds_self.shape
        
        pp_proc_base = torch.tensor([W_proc / 2, H_proc / 2], device=pts3ds_self.device).float()
        pp_proc = pp_proc_base.view(1, 1, 2).repeat(T_filtered, B, 1)
        
        pts3ds_self_flat = pts3ds_self.reshape(T_filtered * B, H_proc, W_proc, 3)
        pp_proc_flat = pp_proc.reshape(T_filtered * B, 2)
        
        focals_proc_flat = self._estimate_focal_knowing_depth(pts3ds_self_flat, pp_proc_flat, focal_mode="weiszfeld")
        depths_low_res_flat = pts3ds_self_flat[..., 2]
        
        # --- Rescale to original resolution (Batched) ---
        scale = self.size / max(original_h, original_w)
        h_scaled, w_scaled = int(original_h * scale), int(original_w * scale)
        pad_h, pad_w = self.size - h_scaled, self.size - w_scaled
        pad_top, pad_left = pad_h // 2, pad_w // 2
        
        depth_cropped = depths_low_res_flat[:, pad_top:pad_top + h_scaled, pad_left:pad_left + w_scaled]
        
        final_depths_flat = F.interpolate(
            depth_cropped.unsqueeze(1),
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        final_depths = final_depths_flat.view(T_filtered, B, original_h, original_w).permute(1, 0, 2, 3)
        
        # --- Intrinsics (Batched) ---
        pad_left_top_tensor = torch.tensor([pad_left, pad_top], device=pp_proc_flat.device).float().view(1, 2)
        
        pp_unpadded = pp_proc_flat - pad_left_top_tensor
        pp_orig = pp_unpadded / scale
        focal_orig = focals_proc_flat / scale
        
        K_batch = torch.eye(3, device=self.device).unsqueeze(0).repeat(T_filtered * B, 1, 1)
        K_batch[:, 0, 0] = focal_orig
        K_batch[:, 1, 1] = focal_orig
        K_batch[:, 0, 2] = pp_orig[:, 0]
        K_batch[:, 1, 2] = pp_orig[:, 1]
        
        final_intrinsics = K_batch.view(T_filtered, B, 3, 3).permute(1, 0, 2, 3)
        
        return final_depths, camera_poses, final_intrinsics

    @torch.no_grad()
    def inference(self, video_batch: torch.Tensor, reset_interval: int = 1000000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run batched inference on preprocessed video tensors.
        """
        B, T, C, H, W = video_batch.shape
        views = self._prepare_batch_views(video_batch, reset_interval)
        outputs, _ = self._inference_recurrent_lighter(views, self.model, self.device)
        depths, poses, intrins = self._process_batch_output(outputs, H, W, B)
        return (depths, poses, intrins)

    


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
    

def downsample_with_fps(points, colors, target_K):
    """
    Args:
        points: (N, 3) torch tensor
        colors: (N, 3) torch tensor
        target_K: int, target number of points (N1)
    """
    # PyTorch3D expects a batch dimension (B, N, 3)
    # We unsqueeze to fake a batch size of 1
    points_batch = points.unsqueeze(0) 
    
    # Run FPS
    # sampled_points: (1, K, 3)
    # indices: (1, K)
    sampled_points, indices = sample_farthest_points(points_batch, K=target_K, random_start_point=True)
    
    # Remove batch dimension from indices
    indices = indices.squeeze(0)
    
    # ---------------------------------------------------------
    # CRITICAL STEP: Use the generated indices to gather colors
    # ---------------------------------------------------------
    sampled_colors = colors[indices]
    
    # Remove batch dimension from points
    sampled_points = sampled_points.squeeze(0)
    
    return sampled_points, sampled_colors