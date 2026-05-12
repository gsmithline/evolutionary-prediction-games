"""Optional W&B helper for the EGTA pipeline.

Opt-in: callers pass enabled=True (typically from a --wandb CLI flag). If wandb
is not installed or enabled is False, every helper becomes a no-op.

The W&B run owns:
  * config = caller-provided dict + the on-disk metadata stored alongside the
    EGTA tensor.
  * scalar metrics logged via log_metrics().
  * .npz bundles uploaded as artifacts.
  * PNG figures uploaded as images.

The trained-model W&B runs and the EGTA analysis runs are in separate W&B runs;
to navigate from one to the other, pass the trained run_tags into the analysis
run's config (key 'training_run_tags').
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore
    _HAS_WANDB = False


class WandbSession:
    """Thin wrapper around an optional wandb.Run.

    If `enabled` is False or wandb is unavailable, all methods are no-ops.
    Otherwise the methods forward to wandb.{init, log, save, finish}.
    """

    def __init__(self, *, enabled: bool, project: str | None, name: str | None,
                 config: dict[str, Any]):
        self.run = None
        if not enabled:
            return
        if not _HAS_WANDB:
            print("[egta_wandb] wandb not installed; --wandb is a no-op")
            return
        self.run = wandb.init(project=project or "egta-analysis", name=name, config=config,
                              reinit=False)

    @property
    def active(self) -> bool:
        return self.run is not None

    def log_metrics(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.active:
            return
        if step is not None:
            self.run.log(data, step=step)
        else:
            self.run.log(data)

    def log_artifact(self, path: Path, *, name: str | None = None,
                     artifact_type: str = "egta-tensor") -> None:
        if not self.active:
            return
        artifact = wandb.Artifact(name=name or path.stem, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def log_image(self, path: Path, *, key: str) -> None:
        if not self.active:
            return
        self.run.log({key: wandb.Image(str(path))})

    def finish(self) -> None:
        if self.active:
            self.run.finish()
            self.run = None


def add_wandb_args(ap, *, default_project: str = "egta-analysis") -> None:
    """Add --wandb, --wandb-project, --wandb-name flags to an argparse parser."""
    ap.add_argument("--wandb", action="store_true",
                    help="Log this run to Weights & Biases (requires wandb installed + WANDB_API_KEY).")
    ap.add_argument("--wandb-project", type=str, default=default_project,
                    help=f"W&B project (default: {default_project!r}). Ignored unless --wandb is set.")
    ap.add_argument("--wandb-name", type=str, default=None,
                    help="W&B run name (default: derived from output path stem).")
