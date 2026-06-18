"""Headless CLI driver for WorldWarp DL3DV experiments.

Runs the "From Video" generation path (TTT3R pose extraction) for a set of
DL3DV scenes without launching the Gradio UI.

Example::

    python run_experiment.py \\
        --dl3dv_root /workspace/data/kbhan/datasets/DL3DV-10K/1K \\
        --num_scenes 10 \\
        --output_dir /workspace/data/kbhan/outputs/worldwarp/dl3dv_pose \\
        --num_chunks 3 --strength 0.6 --speed_multiplier 1.0 --ctx2 1 \\
        --seed 32 --extension_mode extrapolate \\
        --wandb_project worldwarp [--wandb_run_name NAME] [--no_wandb] [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WorldWarp headless DL3DV experiment driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    p.add_argument(
        "--dl3dv_root",
        default="/workspace/data/kbhan/datasets/DL3DV-10K/1K",
        help="Root directory containing DL3DV scene subdirectories.",
    )
    p.add_argument(
        "--num_scenes",
        type=int,
        default=10,
        help="Number of scenes to process (ignored if --scenes_file or --scene_ids given).",
    )
    p.add_argument(
        "--scenes_file",
        default=None,
        help="Path to a newline-separated file of scene hashes to process.",
    )
    p.add_argument(
        "--scene_ids",
        default=None,
        help="Comma-separated list of scene hashes to process.",
    )
    p.add_argument(
        "--frame_dir",
        default="auto",
        help="Frame subdirectory name inside each scene (auto: prefer images_4 > images_8 > images).",
    )
    p.add_argument(
        "--max_ref_frames",
        type=int,
        default=600,
        help="Maximum number of frames to include in the reference video.",
    )

    # --- Output ---
    p.add_argument(
        "--output_dir",
        default="/workspace/data/kbhan/outputs/worldwarp/dl3dv_pose",
        help="Root output directory; per-scene results go in <output_dir>/<scene_id>/.",
    )

    # --- Generation ---
    p.add_argument("--num_chunks", type=int, default=3)
    p.add_argument("--strength", type=float, default=0.6,
                   help="Denoising strength (0–1). Higher = more novel content.")
    p.add_argument("--speed_multiplier", type=float, default=1.0,
                   help="Camera speed multiplier for pose playback.")
    p.add_argument("--ctx2", type=int, default=1,
                   help="context_frames_2nd: overlap frames between chunks.")
    p.add_argument("--gs_iter", type=int, default=500,
                   help="Number of 3DGS optimisation iterations per chunk.")
    p.add_argument("--seed", type=int, default=32)
    p.add_argument(
        "--extension_mode",
        default="extrapolate",
        choices=["extrapolate", "loop", "pingpong", "slowdown"],
        help="Pose extension mode when extracted poses are shorter than needed.",
    )
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)

    # --- W&B ---
    p.add_argument("--wandb_project", default="worldwarp")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")

    # --- Misc ---
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate scene discovery + ref-video building WITHOUT loading models or generating.",
    )
    p.add_argument(
        "--no_finetune",
        action="store_true",
        help="Use base WAN 2.1 weights only, skipping the finetuned checkpoint.",
    )
    p.add_argument(
        "--uniform_noise",
        action="store_true",
        help="Apply the same noise level to all pixels (in-distribution for base VDM). "
             "When off, invalid/unwarped pixels get full noise (original WorldWarp behaviour).",
    )

    return p


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

FRAME_DIR_PRIORITY = ["images_4", "images_8", "images"]


def resolve_frame_dir(scene_root: Path, frame_dir_arg: str) -> Optional[Path]:
    """Return the absolute path of the frame directory for a scene, or None."""
    if frame_dir_arg == "auto":
        for candidate in FRAME_DIR_PRIORITY:
            d = scene_root / candidate
            if d.is_dir():
                return d
        return None
    else:
        d = scene_root / frame_dir_arg
        return d if d.is_dir() else None


def discover_scenes(dl3dv_root: Path, frame_dir_arg: str) -> List[str]:
    """Return sorted list of scene hashes that have transforms.json + a frame dir."""
    valid = []
    try:
        entries = sorted(os.listdir(str(dl3dv_root)))
    except OSError as e:
        print(f"[ERROR] Cannot list dl3dv_root '{dl3dv_root}': {e}")
        return []

    for name in entries:
        scene_root = dl3dv_root / name
        if not scene_root.is_dir():
            continue
        if not (scene_root / "transforms.json").exists():
            continue
        if resolve_frame_dir(scene_root, frame_dir_arg) is None:
            continue
        valid.append(name)

    return valid


def select_scenes(args) -> List[str]:
    """Return the list of scene hashes to process, respecting CLI flags."""
    dl3dv_root = Path(args.dl3dv_root)

    if args.scene_ids:
        return [s.strip() for s in args.scene_ids.split(",") if s.strip()]

    if args.scenes_file:
        with open(args.scenes_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    all_scenes = discover_scenes(dl3dv_root, args.frame_dir)
    return all_scenes[: args.num_scenes]


# ---------------------------------------------------------------------------
# Reference video building
# ---------------------------------------------------------------------------

def get_frame_order(scene_root: Path, frame_dir: Path) -> List[Path]:
    """Return ordered list of frame image paths for a scene.

    Uses transforms.json frame order if available; falls back to sorted filenames.
    """
    transforms_path = scene_root / "transforms.json"
    try:
        with open(transforms_path, "r") as f:
            transforms = json.load(f)
        if "frames" in transforms and transforms["frames"]:
            # frames[].file_path is relative to scene_root (e.g. "images_8/frame_00001.png")
            ordered = []
            for frame_info in transforms["frames"]:
                rel_path = frame_info.get("file_path", "")
                # The file_path may be relative to scene root or absolute
                candidate = scene_root / rel_path
                if not candidate.exists():
                    # Try just the filename in the resolved frame_dir
                    candidate = frame_dir / Path(rel_path).name
                if candidate.exists():
                    ordered.append(candidate)
            if ordered:
                return ordered
    except Exception as e:
        print(f"  [WARN] Could not parse transforms.json for frame order: {e}")

    # Fallback: sorted filenames in frame_dir
    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    files = sorted([
        frame_dir / f for f in os.listdir(str(frame_dir))
        if Path(f).suffix.lower() in IMAGE_EXTS
    ])
    return files


def build_reference_video(
    scene_root: Path,
    frame_dir: Path,
    output_path: Path,
    height: int,
    width: int,
    max_frames: int,
    fps: int = 10,
) -> Optional[Path]:
    """Build scene_ref.mp4 from scene frames.  Returns path or None on failure."""
    import cv2
    import imageio
    import numpy as np

    frames_ordered = get_frame_order(scene_root, frame_dir)
    if not frames_ordered:
        print(f"  [ERROR] No frames found in {frame_dir}")
        return None

    # Limit to max_frames
    frames_ordered = frames_ordered[:max_frames]
    print(f"  Building reference video from {len(frames_ordered)} frames -> {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine resize parameters from first frame
    first = cv2.imread(str(frames_ordered[0]))
    if first is None:
        print(f"  [ERROR] Cannot read first frame: {frames_ordered[0]}")
        return None

    h_orig, w_orig = first.shape[:2]
    scale = max(width / w_orig, height / h_orig)
    h_r, w_r = int(h_orig * scale), int(w_orig * scale)
    y_off = (h_r - height) // 2
    x_off = (w_r - width) // 2

    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame_path in frames_ordered:
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"  [WARN] Cannot read frame: {frame_path}, skipping.")
                continue
            rescaled = cv2.resize(img, (w_r, h_r), interpolation=cv2.INTER_LANCZOS4)
            cropped = rescaled[y_off:y_off + height, x_off:x_off + width]
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)

    print(f"  Reference video written: {output_path}")
    return output_path


def extract_start_frame(ref_video_path: Path, output_path: Path) -> Optional[Path]:
    """Extract first frame of ref video and save as PNG."""
    import imageio

    try:
        reader = imageio.get_reader(str(ref_video_path))
        frame = reader.get_data(0)
        reader.close()
        from PIL import Image
        Image.fromarray(frame).save(str(output_path))
        print(f"  Start frame saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"  [ERROR] Failed to extract start frame: {e}")
        return None


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_scene(
    scene_id: str,
    scene_root: Path,
    frame_dir: Path,
    output_scene_dir: Path,
    args,
    session,  # VideoGenerationSession instance (None in dry_run)
    config_negative: str,
) -> bool:
    """Process one scene.  Returns True on success."""
    print(f"\n{'='*60}")
    print(f"SCENE: {scene_id}")
    print(f"{'='*60}")

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    ref_video_path = output_scene_dir / "scene_ref.mp4"
    start_frame_path = output_scene_dir / "start_frame.png"

    # --- Build reference video ---
    result = build_reference_video(
        scene_root=scene_root,
        frame_dir=frame_dir,
        output_path=ref_video_path,
        height=args.height,
        width=args.width,
        max_frames=args.max_ref_frames,
    )
    if result is None:
        print(f"  [FAIL] Could not build reference video for {scene_id}")
        return False

    # --- Extract start frame ---
    sf = extract_start_frame(ref_video_path, start_frame_path)
    if sf is None:
        print(f"  [FAIL] Could not extract start frame for {scene_id}")
        return False

    if args.dry_run:
        print(f"  [DRY RUN] Skipping model loading and generation.")
        return True

    # --- Drive VideoGenerationSession ---
    session.reset()

    status, _ = session.set_uploaded_image(str(start_frame_path))
    print(f"  set_uploaded_image: {status}")

    extract_status, _ = session.extract_camera_from_video(str(ref_video_path), use_cache=False)
    print(f"  extract_camera_from_video: {extract_status[:80]}")
    if "Failed" in extract_status or "No video" in extract_status:
        print(f"  [FAIL] Pose extraction failed for {scene_id}")
        return False

    combined_path, gen_status, _history, _captions = session.generate_chunks_from_extracted(
        prompt="",
        negative_prompt=config_negative,
        num_chunks=args.num_chunks,
        context_frames_2nd=args.ctx2,
        strength=args.strength,
        num_gs_iterations=args.gs_iter,
        seed=args.seed,
        extension_mode=args.extension_mode,
        speed_multiplier=args.speed_multiplier,
    )
    print(f"  generate_chunks_from_extracted: {gen_status}")

    if combined_path is None or not Path(combined_path).exists():
        print(f"  [FAIL] Generation produced no output for {scene_id}")
        return False

    # --- Copy outputs ---
    generated_out = output_scene_dir / "generated.mp4"
    reference_out = output_scene_dir / "reference.mp4"
    shutil.copy2(combined_path, generated_out)
    shutil.copy2(ref_video_path, reference_out)
    print(f"  Copied generated -> {generated_out}")
    print(f"  Copied reference -> {reference_out}")

    # Copy trajectory visualization if it exists
    try:
        traj_src = Path(session.generator.pm.dirs["visualizations"]) / "extracted_camera_trajectory.png"
        if traj_src.exists():
            traj_dst = output_scene_dir / "trajectory.png"
            shutil.copy2(traj_src, traj_dst)
            print(f"  Copied trajectory -> {traj_dst}")
    except Exception as e:
        print(f"  [WARN] Could not copy trajectory visualization: {e}")

    return True


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dl3dv_root = Path(args.dl3dv_root)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("WorldWarp DL3DV Experiment")
    print("=" * 60)
    print(f"  dl3dv_root:      {dl3dv_root}")
    print(f"  output_dir:      {output_dir}")
    print(f"  num_chunks:      {args.num_chunks}")
    print(f"  strength:        {args.strength}")
    print(f"  speed_multiplier:{args.speed_multiplier}")
    print(f"  ctx2:            {args.ctx2}")
    print(f"  gs_iter:         {args.gs_iter}")
    print(f"  seed:            {args.seed}")
    print(f"  extension_mode:  {args.extension_mode}")
    print(f"  height x width:  {args.height} x {args.width}")
    print(f"  dry_run:         {args.dry_run}")
    print(f"  no_wandb:        {args.no_wandb}")

    # --- Select scenes ---
    scene_ids = select_scenes(args)
    if not scene_ids:
        print("[ERROR] No valid scenes found. Check --dl3dv_root and scene layout.")
        return 1

    print(f"\nSelected {len(scene_ids)} scene(s):")
    for sid in scene_ids:
        print(f"  {sid}")

    # --- Load models (unless dry_run) ---
    session = None
    config_negative = ""
    if not args.dry_run:
        print("\nLoading models (this may take several minutes)...")
        from gradio_demo import VideoGenerationSession
        from pose_control import CONFIG
        from omegaconf import OmegaConf
        config_negative = OmegaConf.to_container(CONFIG, resolve=True).get("prompts", {}).get(
            "negative",
            "Person, people, pet, animals. Bright tones, static camera, overexposed, static, "
            "blurred details, subtitles, style, works, paintings, images, static, overall gray, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs, many people in the "
            "background, walking backwards, unrealistic.",
        )
        cfg_overrides = {}
        if args.no_finetune:
            cfg_overrides.setdefault("paths", {})["finetuned_checkpoint_path"] = ""
            print("--no_finetune: skipping finetuned checkpoint, using base WAN 2.1 weights.")
        if args.uniform_noise:
            cfg_overrides.setdefault("inference_params", {})["uniform_noise"] = True
            print("--uniform_noise: applying uniform noise level to all pixels.")
        session = VideoGenerationSession(cfg_overrides=cfg_overrides if cfg_overrides else None)
    else:
        config_negative = (
            "Person, people, pet, animals. Bright tones, static camera, overexposed, static, "
            "blurred details, subtitles, style, works, paintings, images, static, overall gray, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs, many people in the "
            "background, walking backwards, unrealistic."
        )

    # --- W&B init ---
    from wandb_logger import WandbLogger
    logger = WandbLogger(args)
    if not args.dry_run:
        logger.init(config={
            "dl3dv_root": str(dl3dv_root),
            "num_scenes": len(scene_ids),
            "num_chunks": args.num_chunks,
            "strength": args.strength,
            "speed_multiplier": args.speed_multiplier,
            "ctx2": args.ctx2,
            "gs_iter": args.gs_iter,
            "seed": args.seed,
            "extension_mode": args.extension_mode,
            "height": args.height,
            "width": args.width,
            "model_variant": "base_wan2.1" if args.no_finetune else "worldwarp_finetuned",
            "uniform_noise": args.uniform_noise,
        })

    # --- Scene loop ---
    successes = []
    failures = []

    for scene_id in scene_ids:
        scene_root = dl3dv_root / scene_id
        frame_dir = resolve_frame_dir(scene_root, args.frame_dir)
        if frame_dir is None:
            print(f"\n[SKIP] {scene_id}: no frame directory found.")
            failures.append((scene_id, "no frame directory"))
            continue

        output_scene_dir = output_dir / scene_id

        try:
            ok = run_scene(
                scene_id=scene_id,
                scene_root=scene_root,
                frame_dir=frame_dir,
                output_scene_dir=output_scene_dir,
                args=args,
                session=session,
                config_negative=config_negative,
            )
        except Exception as e:
            print(f"\n[ERROR] Scene {scene_id} raised an exception:")
            traceback.print_exc()
            ok = False

        if ok:
            successes.append(scene_id)
            if not args.dry_run:
                logger.log_scene(
                    scene_id=scene_id,
                    generated_mp4=output_scene_dir / "generated.mp4",
                    reference_mp4=output_scene_dir / "reference.mp4",
                )
        else:
            failures.append((scene_id, "see above"))

    # --- W&B finish ---
    if not args.dry_run:
        logger.finish()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  SUCCESS ({len(successes)}):")
    for sid in successes:
        print(f"    {sid}")
    print(f"  FAIL ({len(failures)}):")
    for sid, reason in failures:
        print(f"    {sid}  ({reason})")

    if not successes and failures:
        print("\n[ERROR] All scenes failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
