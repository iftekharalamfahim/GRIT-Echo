"""Segmentation and boundary metrics.

All metrics operate on native-resolution masks. HD95 is computed in millimetres
using per-header voxel spacing (never assumed isotropic).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist


def largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected blob; removes stray prediction specks."""
    labs, n = ndimage.label(binary)
    if n <= 1:
        return binary
    sizes = ndimage.sum(binary, labs, index=np.arange(1, n + 1))
    return labs == int(np.argmax(sizes)) + 1


def dice_score(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    """Dice coefficient of class ``c``; NaN when the class is absent in both."""
    inter = ((pred == c) & (gt == c)).sum()
    denom = (pred == c).sum() + (gt == c).sum()
    return float(2 * inter / denom) if denom > 0 else np.nan


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Pixel coordinates of the boundary ring of a binary mask."""
    binary = mask.astype(bool)
    if not binary.any():
        return np.empty((0, 2), int)
    return np.argwhere(binary & ~ndimage.binary_erosion(binary))


def hd95_mm(pred: np.ndarray, gt: np.ndarray, c: int,
            spacing_px_mm: tuple[float, float]) -> float:
    """95th-percentile symmetric boundary distance in millimetres.

    Returns NaN when either mask lacks the class (caller decides aggregation).
    """
    pb = _boundary(pred == c)
    gb = _boundary(gt == c)
    if len(pb) == 0 or len(gb) == 0:
        return np.nan
    scale = np.asarray(spacing_px_mm, dtype=float)
    d = cdist(pb * scale, gb * scale)
    return float(max(np.percentile(d.min(axis=1), 95), np.percentile(d.min(axis=0), 95)))


def foreground_dice(logits_or_pred, targets, classes: tuple[int, ...],
                    logits: bool = True) -> float:
    """Macro Dice over the anatomical foreground classes (model-selection metric)."""
    import torch

    pred = (logits_or_pred.argmax(1) if logits else logits_or_pred).cpu().numpy()
    target = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    per_class = [dice_score(pred, target, c) for c in classes]
    valid = [d for d in per_class if not np.isnan(d)]
    return float(np.mean(valid)) if valid else np.nan
