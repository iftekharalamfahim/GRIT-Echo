"""Clinical endpoint: Simpson biplane LVEF via method of disks (ASE 2015)."""

from __future__ import annotations

import numpy as np

from .data import frame_path
from .metrics import largest_component

import nibabel as nib


def cavity_profile(mask: np.ndarray, n_disks: int = 20) -> tuple[np.ndarray, float]:
    """(disk widths in px, long-axis length in px) along the cavity principal axis."""
    blob = largest_component(mask == 1)
    ys, xs = np.nonzero(blob)
    if len(xs) < 20:
        return np.zeros(n_disks), 0.0
    pts = np.stack([xs, ys], 1).astype(float)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj, orth = pts @ vt[0], pts @ vt[1]
    edges = np.linspace(proj.min(), proj.max(), n_disks + 1)
    widths = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (proj >= a) & (proj < b)
        widths.append(float(orth[sel].max() - orth[sel].min()) if sel.any() else 0.0)
    return np.array(widths), float(proj.max() - proj.min())


def volume_ml(widths_px: np.ndarray, length_px: float, spacing_mm,
              partner_px: np.ndarray | None = None) -> float:
    """Disk summation in mL; spacing read per header, never assumed."""
    mm = float(np.mean(spacing_mm))
    h_cm = length_px * mm / 10.0 / len(widths_px)
    d1 = widths_px * mm / 10.0
    if partner_px is None:
        return float((np.pi / 4) * np.sum(d1**2) * h_cm)
    d2 = partner_px * mm / 10.0
    return float((np.pi / 4) * np.sum(d1 * d2) * h_cm)


def load_mask_with_spacing(nifti_dir, pid: str, view: str, instant: str):
    """Mask array plus its own header spacing, via the tolerant path helper."""
    img = nib.load(frame_path(nifti_dir / pid, view, instant, mask=True))
    return (np.squeeze(np.asarray(img.dataobj)), img.header.get_zooms()[:2])


def ef(edv: float, esv: float) -> float:
    """Ejection fraction in percent; NaN guards against degenerate volumes."""
    return (edv - esv) / edv * 100.0 if edv > 0 else np.nan


def patient_volumes(nifti_dir, pid: str, views: tuple[str, ...] = ("2CH", "4CH"),
                    instants: tuple[str, ...] = ("ED", "ES")) -> tuple[dict, dict]:
    """(single-plane volumes per view, biplane volumes) for one patient."""
    single, prof = {}, {}
    for view in views:
        for instant in instants:
            mask, spacing = load_mask_with_spacing(nifti_dir, pid, view, instant)
            w, l = cavity_profile(mask)
            prof[(view, instant)] = (w, l, spacing)
            single[(view, instant)] = volume_ml(w, l, spacing)
    biplane = {}
    for instant in instants:
        (w1, l1, s1), (w2, l2, _) = prof[(views[0], instant)], prof[(views[1], instant)]
        biplane[instant] = volume_ml(w1, (l1 + l2) / 2, s1, partner_px=w2)
    return single, biplane
