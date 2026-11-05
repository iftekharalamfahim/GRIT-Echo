"""Data loading: cohort audit, transforms, dataset, and loaders.

- aspect-ratio-preserving letterbox resize (no distortion of cardiac geometry),
- per-worker deterministic seeding,
- configurable ``num_workers`` and ``pin_memory`` for Kaggle GPU throughput.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import albumentations as A
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .config import Settings

log = logging.getLogger(__name__)

IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD = (0.229, 0.224, 0.225)


# Paths and cohort audit
def frame_path(patient_dir: Path, view: str, instant: str, mask: bool = False) -> Path:
    """Tolerant NIfTI locator: accepts ``.nii`` and ``.nii.gz``; raises if absent."""
    suffix = "_gt" if mask else ""
    stem = f"{patient_dir.name}_{view}_{instant}{suffix}"
    matches = sorted(patient_dir.glob(f"{stem}.nii*"))
    if not matches:
        raise FileNotFoundError(f"Missing {'mask' if mask else 'image'}: {patient_dir.name}/{stem}")
    return matches[0]


def cohort_audit_table(cfg: Settings) -> pd.DataFrame:
    """One row per frame; headers only (nibabel defers decompression)."""
    rows = []
    for patient in sorted(p for p in cfg.nifti_dir.iterdir() if p.is_dir()):
        for view in cfg.views:
            for instant in cfg.instants:
                img = nib.load(frame_path(patient, view, instant))
                nib.load(frame_path(patient, view, instant, mask=True))
                sx, sy = img.header.get_zooms()[:2]
                rows.append({"patient": patient.name, "view": view, "instant": instant,
                             "h": img.shape[0], "w": img.shape[1],
                             "sx_mm": float(sx), "sy_mm": float(sy)})
    return pd.DataFrame(rows)


def read_subgroup(split_dir: Path, filename: str) -> list[str]:
    """Patient identifiers of one official subgroup file, in file order."""
    path = split_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Subgroup file not found: {path}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def assert_clean_splits(train: list[str], val: list[str], test: list[str]) -> None:
    """Fail fast on duplicate or leaking identifiers."""
    if len(train) != len(set(train)) or len(val) != len(set(val)) or len(test) != len(set(test)):
        raise ValueError("Duplicate identifiers inside a subgroup file.")
    if set(train) & set(test) or set(val) & set(test):
        raise ValueError("Test subgroup contaminated: overlaps train or val.")


REQUIRED_CFG_KEYS = ("ED", "ES", "NbFrame", "Sex", "Age", "ImageQuality", "EF", "FrameRate")


def parse_cfg(path: Path) -> dict:
    """Parse a CAMUS per-view metadata file; raise if required keys are absent."""
    cfg: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    missing = [k for k in REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"{path.name} missing keys: {missing}")
    return cfg


def load_metadata(cfg: Settings, patient_ids: set[str]) -> pd.DataFrame:
    """One row per (patient, view): ED/ES frame indices, quality, reference EF."""
    rows = []
    for pid in sorted(patient_ids):
        for view in cfg.views:
            c = parse_cfg(cfg.nifti_dir / pid / f"Info_{view}.cfg")
            rows.append({"patient": pid, "view": view, "ed_frame": int(c["ED"]),
                         "es_frame": int(c["ES"]), "quality": c["ImageQuality"],
                         "ref_ef": float(c["EF"]), "sex": c["Sex"], "age": int(c["Age"])})
    return pd.DataFrame(rows)


# Intensity normalization and transforms
def normalize_intensity(raw: np.ndarray, percentiles: tuple[float, float] = (1, 99)) -> np.ndarray:
    """Percentile-clip then rescale to [0, 255]; identical at train and inference."""
    lo, hi = np.percentile(raw, percentiles)
    span = max(hi - lo, 1e-6)
    return ((np.clip(raw, lo, hi) - lo) / span * 255.0).astype(np.uint8)


def pad_to_square(img: np.ndarray, **kwargs) -> np.ndarray:
    """Letterbox-pad to square with zeros so resize never distorts geometry."""
    h, w = img.shape[:2]
    if h == w:
        return img
    pad = abs(h - w)
    if h < w:
        top, bottom = pad // 2, pad - pad // 2
        return np.pad(img, ((top, bottom), (0, 0)) + ((0, 0),) * (img.ndim - 2), **kwargs)
    left, right = pad // 2, pad - pad // 2
    return np.pad(img, ((0, 0), (left, right)) + ((0, 0),) * (img.ndim - 2), **kwargs)


def build_transform(cfg: Settings, split: str) -> A.Compose:
    """Augmentation for train only; val/test/seq see pad + resize + normalize.

    Masks use nearest-neighbour everywhere so class labels stay discrete.
    """
    train_only: list[A.BasicTransform] = [
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-12, 12), scale=(0.92, 1.08), translate_percent=(-0.05, 0.05),
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.6),
        A.RandomBrightnessContrast(p=0.4),
    ]
    return A.Compose(
        (train_only if split == "train" else [])
        + [
            A.Lambda(image=pad_to_square, mask=pad_to_square, p=1.0),
            A.Resize(height=cfg.target_size[0], width=cfg.target_size[1],
                     interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
            ToTensorV2(),
        ]
    )


def seq_transform(cfg: Settings) -> A.Compose:
    """Deterministic transform used for all frame-wise inference (frame-wise inference)."""
    return A.Compose([
        A.Lambda(image=pad_to_square, p=1.0),
        A.Resize(height=cfg.target_size[0], width=cfg.target_size[1],
                 interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
        ToTensorV2(),
    ])

# Dataset and loaders
class CamusDataset(Dataset):
    """All four (view, instant) frames of each patient, in fixed order."""

    def __init__(self, cfg: Settings, ids: list[str], split: str):
        self.cfg = cfg
        self.samples = [(pid, view, instant) for pid in ids
                        for view in cfg.views for instant in cfg.instants]
        self.transform = build_transform(cfg, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pid, view, instant = self.samples[idx]
        img = np.squeeze(np.asarray(
            nib.load(frame_path(self.cfg.nifti_dir / pid, view, instant)).dataobj
        )).astype(np.float32)
        msk = np.squeeze(np.asarray(
            nib.load(frame_path(self.cfg.nifti_dir / pid, view, instant, mask=True)).dataobj
        )).astype(np.uint8)
        rgb = np.stack([normalize_intensity(img, self.cfg.intensity_percentiles)] * 3, axis=-1)
        out = self.transform(image=rgb, mask=msk)
        return out["image"], out["mask"].long()


def _worker_init(worker_id: int) -> None:
    """Deterministic augmentation streams across DataLoader workers."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(cfg: Settings, ids: list[str], split: str) -> DataLoader:
    gen = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        CamusDataset(cfg, ids, split),
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        worker_init_fn=_worker_init,
        generator=gen,
        persistent_workers=cfg.num_workers > 0,
    )
