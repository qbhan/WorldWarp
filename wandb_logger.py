"""Lightweight Weights & Biases logging helpers for WorldWarp experiment scripts.

Supports multi-scene runs: init once, log per scene, finish at end.

Usage pattern::

    logger = WandbLogger(args)
    logger.init(config={"num_scenes": 10, ...})
    for scene_id, generated_mp4, reference_mp4 in results:
        logger.log_scene(scene_id, generated_mp4, reference_mp4)
    logger.finish()

Defensive: no crash if wandb is missing, not logged in, or --no_wandb is set.
WANDB_API_KEY absence is handled gracefully (wandb will log a warning and may
operate in offline mode or skip upload).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional


def add_wandb_args(parser, default_project: str = "worldwarp") -> None:
    """Add W&B CLI arguments to an argparse parser."""
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=default_project,
        help="W&B project name (use --no_wandb to disable logging).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run / experiment name. If unset, an auto-generated name is used.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team or user). Defaults to the wandb-configured entity.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging for this run.",
    )


class WandbLogger:
    """Manages a single W&B run across multiple scenes.

    Degrades gracefully when wandb is absent, not authenticated, or --no_wandb.
    """

    def __init__(self, args) -> None:
        self._args = args
        self._run = None
        self._enabled = not getattr(args, "no_wandb", False)
        self._wandb = None

        if self._enabled:
            try:
                import wandb  # type: ignore
                self._wandb = wandb
            except ImportError:
                print("[wandb_logger] wandb is not installed; skipping (pip install wandb).")
                self._enabled = False

    def init(self, config: Optional[Mapping] = None) -> None:
        """Initialize the W&B run.  Call once before the scene loop."""
        if not self._enabled or self._wandb is None:
            return

        run_name = getattr(self._args, "wandb_run_name", None)
        project = getattr(self._args, "wandb_project", None) or "worldwarp"
        entity = getattr(self._args, "wandb_entity", None) or None

        try:
            self._run = self._wandb.init(
                project=project,
                name=run_name,
                entity=entity,
                config=dict(config) if config else None,
                reinit=True,
            )
            print(f"[wandb_logger] Run initialized: {self._run.name} (project={project})")
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb_logger] wandb.init failed: {exc}; skipping.")
            self._enabled = False
            self._run = None

    def log_scene(
        self,
        scene_id: str,
        generated_mp4: "str | os.PathLike | None",
        reference_mp4: "str | os.PathLike | None",
        fps: int = 15,
        extra: Optional[Mapping] = None,
    ) -> None:
        """Log generated and reference videos for one scene.

        Keys in W&B: ``<scene_id>/generated`` and ``<scene_id>/reference``.
        Missing or non-existent paths are silently skipped.
        """
        if not self._enabled or self._run is None:
            return

        payload: dict = {}

        if generated_mp4 and Path(str(generated_mp4)).exists():
            payload[f"{scene_id}/generated"] = self._wandb.Video(
                str(generated_mp4), fps=fps, format="mp4"
            )
        else:
            print(f"[wandb_logger] scene {scene_id}: generated mp4 not found, skipping.")

        if reference_mp4 and Path(str(reference_mp4)).exists():
            payload[f"{scene_id}/reference"] = self._wandb.Video(
                str(reference_mp4), fps=fps, format="mp4"
            )
        else:
            print(f"[wandb_logger] scene {scene_id}: reference mp4 not found, skipping.")

        if extra:
            payload.update({f"{scene_id}/{k}": v for k, v in extra.items()})

        if not payload:
            return

        try:
            self._run.log(payload)
            print(f"[wandb_logger] Logged scene '{scene_id}' to W&B run '{self._run.name}'.")
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb_logger] Failed to log scene '{scene_id}': {exc}")

    def log_metrics(self, metrics: Mapping) -> None:
        """Log arbitrary scalar (or media) metrics into the current run.

        No-op when wandb is disabled or the run is not initialised.
        """
        if not self._enabled or self._run is None:
            return
        try:
            self._run.log(dict(metrics))
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb_logger] log_metrics failed: {exc}")

    def log_scene_metrics(self, scene_id: str, metrics: Mapping) -> None:
        """Log per-scene scalar metrics namespaced as 'eval/<scene_id>/<key>'.

        No-op when wandb is disabled or the run is not initialised.
        """
        if not self._enabled or self._run is None:
            return
        payload = {f"eval/{scene_id}/{k}": v for k, v in metrics.items()}
        try:
            self._run.log(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb_logger] log_scene_metrics failed for '{scene_id}': {exc}")

    def finish(self) -> None:
        """Finish the W&B run.  Call once after the scene loop."""
        if self._run is not None:
            try:
                self._run.finish()
                print(f"[wandb_logger] Run '{self._run.name}' finished.")
            except Exception as exc:  # noqa: BLE001
                print(f"[wandb_logger] wandb.finish failed: {exc}")
            self._run = None
