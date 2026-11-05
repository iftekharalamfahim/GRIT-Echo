"""Central configuration: paths, constants, and runtime settings.

Paths auto-detect between Kaggle and a local machine. Override any of them via
environment variables (``GRIT_NIFTI``, ``GRIT_SPLIT``, ``GRIT_SEQ``, ``GRIT_WORK``)
or a YAML file passed to :meth:`Settings.load`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for every pipeline stage."""

    # data locations 
    nifti_dir: Path
    split_dir: Path
    seq_dir: Path
    work_dir: Path

    # segmentation training 
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 6
    amp: bool = True  # mixed precision on CUDA; auto-disabled on CPU


    # reproducibility
    seed: int = 42

    # scientific constants
    target_size: tuple[int, int] = (256, 256)
    intensity_percentiles: tuple[float, float] = (1.0, 99.0)
    max_entropy_bits: float = 2.0  # log2(4 classes)
    views: tuple[str, str] = ("2CH", "4CH")
    instants: tuple[str, str] = ("ED", "ES")
    fg_classes: tuple[int, ...] = (1, 2, 3)

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    @property
    def ckpt_path(self) -> Path:
        return self.work_dir / "unet_resnet34_camus.pt"

    @property
    def history_path(self) -> Path:
        return self.work_dir / "history.json"

    @property
    def pred_cache_dir(self) -> Path:
        return self.work_dir / "pred_cache"

    @property
    def use_amp(self) -> bool:
        return self.amp and self.device.type == "cuda"

    @staticmethod
    def load(yaml_path: str | os.PathLike | None = None) -> "Settings":
        """Build settings from defaults + env vars + optional YAML overrides."""
        overrides: dict = {}
        if yaml_path and Path(yaml_path).exists():
            import yaml

            overrides = yaml.safe_load(Path(yaml_path).read_text()) or {}

        def env(key: str) -> str | None:
            return overrides.get(key.lower()) or os.environ.get(key)

        nifti = _first_existing(
            [
                Path(p) if (p := env("GRIT_NIFTI")) else None,
                Path("/kaggle/input/datasets/brainsect/camus-v1/database_nifti"),
                Path("/kaggle/input/camus-v1/database_nifti"),
                Path("data/camus/database_nifti"),
            ]
        )
        split = _first_existing(
            [
                Path(p) if (p := env("GRIT_SPLIT")) else None,
                Path("/kaggle/input/datasets/brainsect/camus-v1/database_split"),
                Path("/kaggle/input/camus-v1/database_split"),
                Path("data/camus/database_split"),
            ]
        )
        seq = _first_existing(
            [
                Path(p) if (p := env("GRIT_SEQ")) else None,
                Path("/kaggle/input/datasets/brainsect/camus-half-sequences"),
                Path("/kaggle/input/camus-half-sequences"),
                Path("data/camus-half-sequences"),
            ]
        )
        work = Path(env("GRIT_WORK") or "/kaggle/working" if Path("/kaggle/working").exists()
                    else "outputs")

        if nifti is None or split is None:
            raise FileNotFoundError(
                "CAMUS data not found. Set GRIT_NIFTI / GRIT_SPLIT (and GRIT_SEQ for "
                "sequence stages) or pass a YAML config. Download via make_pack.py."
            )
        return Settings(
            nifti_dir=nifti,
            split_dir=split,
            seq_dir=seq or nifti.parent / "half_sequences",
            work_dir=work,
            epochs=int(env("GRIT_EPOCHS") or 20),
        )
