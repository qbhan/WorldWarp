"""Evaluation metrics for WorldWarp headless experiments.

Ports the WorldForge FID + pose-paired CLIP metrics VERBATIM from:
  - bridges/evaluation/eval_image_metrics.py  (FID backend)
  - wan_for_worldforge/compare_pose_wan_clip.py  (per-scene scoring + pooled FID)

The math is intentionally kept identical:
  - InceptionV3 (torchvision, weights=DEFAULT, fc=Identity, AuxLogits=Identity)
    input [0,1]->[-1,1], resize 299x299 BILINEAR, 2048-d pool3 features
  - CLIP via transformers CLIPModel/CLIPProcessor, get_image_features + F.normalize p=2 dim=-1
  - FID = Frechet over Gaussian stats with scipy.linalg.sqrtm
  - Per-scene clip_novel = mean over frames[1:] (skips frame 0)
  - Pooled FID: all gen frames vs FULL real frame set across scenes

Importing this module is cheap (heavy deps imported lazily inside functions).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Image loading (ported from eval_image_metrics.py)
# ---------------------------------------------------------------------------

class ImageFolder:
    """Minimal RGB image dataset with explicit resize for a target backbone.

    Ported verbatim from WorldForge eval_image_metrics.py.
    """

    def __init__(self, paths: List[Path], resize_to: int) -> None:
        self.paths = paths
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image as PILImage
        import torch

        img = PILImage.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.resize_to, self.resize_to), PILImage.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, 3 in [0, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)  # 3, H, W


# ---------------------------------------------------------------------------
# FID via torchvision InceptionV3 (ported verbatim from eval_image_metrics.py)
# ---------------------------------------------------------------------------

class InceptionV3PoolFeatures:
    """Wrap torchvision inception_v3 and tap the post-pool3 2048-d features.

    Ported verbatim from WorldForge eval_image_metrics.py.
    """

    def __init__(self) -> None:
        import torch
        import torch.nn as nn
        from torchvision.models import Inception_V3_Weights, inception_v3

        m = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        m.eval()
        # The classifier head turns 2048-d pool3 features into 1000 logits. We
        # need the input to that head -- so wire a hook on the global avg-pool.
        m.fc = nn.Identity()
        # Disable aux logits in forward by replacing with Identity too.
        m.AuxLogits = nn.Identity()
        self.net = m
        # Per the FID convention, images are normalised to [-1, 1] before the
        # network (the canonical pytorch-fid behaviour). InceptionV3 expects
        # 299x299 RGB.
        self.input_size = 299

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def eval(self):
        self.net.eval()
        return self

    def __call__(self, x):
        import torch
        # x: (B, 3, H, W) in [0, 1] -- caller does the resize.
        x = x * 2.0 - 1.0
        return self.net(x)  # (B, 2048)


def _features(
    paths: List[Path],
    extractor: InceptionV3PoolFeatures,
    device,
    batch_size: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    extractor.to(device).eval()
    ds = ImageFolder(paths, resize_to=extractor.input_size)

    # Manual batching to avoid DataLoader collate issues with our custom Dataset
    chunks = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = torch.stack([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
            batch = batch.to(device)
            f = extractor(batch).detach().cpu().numpy()
            chunks.append(f)
    return np.concatenate(chunks, axis=0)


def _gaussian_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ported verbatim from WorldForge eval_image_metrics.py."""
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _frechet(mu_a: np.ndarray, sigma_a: np.ndarray, mu_b: np.ndarray, sigma_b: np.ndarray) -> float:
    """The Fréchet Distance formula for two multivariate Gaussians.

    Ported verbatim from WorldForge eval_image_metrics.py.
    """
    from scipy.linalg import sqrtm

    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a.dot(sigma_b), disp=False)
    # Drop a tiny imaginary part that can appear from numerical sqrtm.
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2.0 * covmean))


def compute_fid(
    gen_paths: List[Path],
    real_paths: List[Path],
    device,
    batch_size: int = 32,
) -> float:
    """Compute FID between gen and real frame sets.

    Ported verbatim from WorldForge eval_image_metrics.py.
    Uses torchvision InceptionV3 (weights=DEFAULT) 2048-d pool3 features.
    """
    extractor = InceptionV3PoolFeatures()
    feats_gen = _features(gen_paths, extractor, device, batch_size)
    feats_real = _features(real_paths, extractor, device, batch_size)
    mu_g, s_g = _gaussian_stats(feats_gen)
    mu_r, s_r = _gaussian_stats(feats_real)
    return _frechet(mu_g, s_g, mu_r, s_r)


# ---------------------------------------------------------------------------
# CLIP similarity via transformers (ported verbatim from eval_image_metrics.py)
# ---------------------------------------------------------------------------

def compute_clip_features(
    paths: List[Path],
    clip_model_name: str,
    device,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute L2-normalised CLIP image features for a list of image paths.

    Ported verbatim from WorldForge eval_image_metrics.py.
    Uses transformers CLIPModel/CLIPProcessor.
    Returns (N, D) float32 array, each row L2-normalised.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image as PILImage
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = [PILImage.open(p).convert("RGB") for p in paths[i: i + batch_size]]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            f = model.get_image_features(**inputs)
            f = F.normalize(f, p=2, dim=-1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)  # (N, D), l2-normalised


# ---------------------------------------------------------------------------
# Frame extraction (ported verbatim from compare_pose_wan_clip.py:extract_mp4)
# ---------------------------------------------------------------------------

def extract_mp4_frames(mp4_path: "str | Path", out_dir: "str | Path") -> List[Path]:
    """Extract all frames of an mp4 to out_dir/g####.png; return sorted paths.

    Ported verbatim from WorldForge compare_pose_wan_clip.py:extract_mp4.
    Clears old g*.png files in out_dir before extracting.
    """
    import cv2

    mp4_path = Path(mp4_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear old extracted frames (matches WorldForge behaviour)
    for f in out_dir.glob("g*.png"):
        f.unlink()

    cap = cv2.VideoCapture(str(mp4_path))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"g{i:04d}.png"), fr)
        i += 1
    cap.release()

    return sorted(out_dir.glob("g*.png"))


# ---------------------------------------------------------------------------
# Per-scene CLIP scoring (ported from compare_pose_wan_clip.py lines ~231-242)
# ---------------------------------------------------------------------------

def score_scene_clip(
    gen_paths: List[Path],
    real_paths: List[Path],
    clip_model_name: str,
    device,
    batch_size: int = 16,
) -> Dict[str, object]:
    """Compute pose-paired CLIP similarity for one scene.

    Replicates the per-scene scoring from WorldForge compare_pose_wan_clip.py:
      n = min(len(gen), len(real))
      gf = compute_clip_features(gen[:n], ...)
      rf = compute_clip_features(real[:n], ...)
      sim = (gf * rf).sum(1)
      clip_all = float(sim.mean())
      clip_novel = float(sim[1:].mean()) if n > 1 else nan

    Returns dict with keys: "n", "clip_all", "clip_novel".
    """
    n = min(len(gen_paths), len(real_paths))
    if n == 0:
        return {"n": 0, "clip_all": float("nan"), "clip_novel": float("nan")}

    gf = compute_clip_features(gen_paths[:n], clip_model_name, device, batch_size)
    rf = compute_clip_features(real_paths[:n], clip_model_name, device, batch_size)
    sim = (gf * rf).sum(1)

    clip_all = float(sim.mean())
    clip_novel = float(sim[1:].mean()) if n > 1 else float("nan")

    return {"n": n, "clip_all": clip_all, "clip_novel": clip_novel}


# ---------------------------------------------------------------------------
# Pooled FID (ported from compare_pose_wan_clip.py lines ~249-251)
# ---------------------------------------------------------------------------

def pooled_fid(
    all_gen_paths: List[Path],
    all_real_paths: List[Path],
    device,
    batch_size: int = 16,
) -> Optional[float]:
    """Compute pooled FID over all gen frames vs all real frames across scenes.

    Replicates WorldForge compare_pose_wan_clip.py:
      if len(all_gen) >= 2 and len(all_real) >= 2:
          fid = float(compute_fid(all_gen, all_real, device, 16))

    Returns None if either set has fewer than 2 images.
    """
    if len(all_gen_paths) < 2 or len(all_real_paths) < 2:
        return None
    print(
        f"[eval] Pooled FID: {len(all_gen_paths)} gen vs {len(all_real_paths)} real frames...",
        flush=True,
    )
    return float(compute_fid(all_gen_paths, all_real_paths, device, batch_size))
