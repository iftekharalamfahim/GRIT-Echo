"""Frame-wise inference, Shannon entropy, and information granules."""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .config import Settings
from .data import normalize_intensity, seq_transform

log = logging.getLogger(__name__)

GRANULE_MAP = {"crisp": 0, "vague": 1, "artifact-prone": 2}
GRANULE_ORDER = ["crisp", "vague", "artifact-prone"]


def load_half_sequence(cfg: Settings, pid: str, view: str) -> np.ndarray:
    """(H, W, T) float32 volume; get_fdata honors scl_slope/scl_inter."""
    matches = sorted(cfg.seq_dir.glob(f"{pid}_{view}_half_sequence.nii*"))
    if not matches:
        raise FileNotFoundError(f"Missing half-sequence for {pid} {view}")
    return np.squeeze(nib.load(matches[0]).get_fdata(dtype=np.float32))


@torch.no_grad()
def extract_entropy(cfg: Settings, model, ids: list[str]) -> pd.DataFrame:
    """Frame-wise normalized Shannon entropy over an arbitrary split.

    One row per frame: global entropy (whole image) and tissue entropy
    (predicted foreground only). This single implementation replaces the two
    duplicated loops in the original notebook.
    """
    tf = seq_transform(cfg)
    records: list[dict] = []
    model.eval()
    for pid in ids:
        for view in cfg.views:
            try:
                vol = load_half_sequence(cfg, pid, view)
            except FileNotFoundError:
                log.warning("No half-sequence for %s %s; skipping.", pid, view)
                continue
            std = [normalize_intensity(vol[..., t], cfg.intensity_percentiles)
                   for t in range(vol.shape[-1])]
            tens = torch.stack([tf(image=np.stack([s] * 3, -1))["image"] for s in std])
            tens = tens.to(cfg.device)
            probs = torch.softmax(model(tens), 1)
            ent = (-(probs * torch.log2(probs + 1e-7)).sum(1) / cfg.max_entropy_bits)
            ent = ent.cpu().numpy()
            fg = probs.argmax(1).cpu().numpy() > 0
            for t in range(vol.shape[-1]):
                tissue = ent[t][fg[t]]
                records.append({
                    "patient": pid, "view": view, "frame_idx": t,
                    "global_entropy": float(ent[t].mean()),
                    "tissue_entropy": float(tissue.mean()) if tissue.size else np.nan})
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            "No entropy rows produced. Check GRIT_SEQ / seq_dir mounting and contents.")
    return df

# Granule calibration (frozen on validation terciles)
class GranuleCalibrator:
    """Val-tercile cutpoints frozen once, applied identically to every split."""

    def __init__(self, cutpoints: tuple[float, float]):
        self.cuts = cutpoints

    @classmethod
    def fit(cls, val_entropy: pd.DataFrame) -> "GranuleCalibrator":
        cuts = tuple(np.quantile(val_entropy.tissue_entropy.dropna(), [1 / 3, 2 / 3]))
        log.info("Frozen granule cutpoints (val terciles): %.4f, %.4f", *cuts)
        return cls(cuts)

    def granule_of(self, e: float) -> str:
        if e <= self.cuts[0]:
            return "crisp"
        return "vague" if e <= self.cuts[1] else "artifact-prone"

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["granule"] = out.tissue_entropy.map(self.granule_of)
        return out
