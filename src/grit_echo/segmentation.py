"""Model factory, training loop, and evaluation with prediction caching.

- AMP (mixed precision) + GradScaler: ~2x faster on the Kaggle T4,
- gradient clipping for the combined Dice+CE loss,
- early stopping with patience,
- test-set predictions computed **once** and cached to disk (the notebook ran
  the whole test set twice: once for metrics, once for LVEF).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy import stats
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from .config import Settings
from .data import (build_transform, frame_path, make_loader, normalize_intensity,
                   pad_to_square, read_subgroup)
from .metrics import dice_score, foreground_dice, hd95_mm

log = logging.getLogger(__name__)


def build_model(device: torch.device, pretrained: bool = True) -> smp.Unet:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=4,
    ).to(device)


class DiceCELoss(torch.nn.Module):
    """Combined soft-Dice + cross-entropy; defensive squeeze keeps (B, H, W)."""

    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce = CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        return self.dice(logits, targets) + self.ce(logits, targets.long())


# Training
def train_segmentation(cfg: Settings, force: bool = False) -> tuple[smp.Unet, list[dict]]:
    """Train (or load) the canonical U-Net; returns (model, history).

    ``force=False`` reuses the checkpoint if present (Kaggle-friendly: repeated
    runs reuse published artifacts instead of silently retraining).
    """
    if cfg.ckpt_path.exists() and not force:
        log.info("Checkpoint %s found; loading instead of retraining.", cfg.ckpt_path.name)
        model = build_model(cfg.device, pretrained=False)
        model.load_state_dict(torch.load(cfg.ckpt_path, map_location=cfg.device,
                                         weights_only=True))
        model.eval()
        history_path = cfg.history_path
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        return model, history

    model = build_model(cfg.device, pretrained=True)
    criterion = DiceCELoss()
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    train_loader = make_loader(cfg, read_subgroup(cfg.split_dir, "subgroup_training.txt"), "train")
    val_loader = make_loader(cfg, read_subgroup(cfg.split_dir, "subgroup_validation.txt"), "val")

    history: list[dict] = []
    best_dice, best_epoch, bad_epochs = -1.0, -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        t_tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device, non_blocking=True), yb.to(cfg.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                loss = criterion(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            t_tot += loss.item() * len(xb)

        model.eval()
        v_tot, dices = 0.0, []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                with torch.amp.autocast("cuda", enabled=cfg.use_amp):
                    lg = model(xb)
                v_tot += criterion(lg.float(), yb).item() * len(xb)
                dices.append(foreground_dice(lg, yb, cfg.fg_classes))
        sched.step()

        rec = {"epoch": epoch + 1,
               "train_loss": t_tot / len(train_loader.dataset),
               "val_loss": v_tot / len(val_loader.dataset),
               "val_dice": float(np.mean(dices))}
        history.append(rec)
        marker = ""
        if rec["val_dice"] > best_dice:
            best_dice, best_epoch, bad_epochs = rec["val_dice"], epoch + 1, 0
            torch.save(model.state_dict(), cfg.ckpt_path)
            marker = "  <- saved"
        else:
            bad_epochs += 1
        log.info("Epoch %02d/%d | train %.4f | val %.4f | valDice %.4f%s",
                 epoch + 1, cfg.epochs, rec["train_loss"], rec["val_loss"],
                 rec["val_dice"], marker)
        if bad_epochs >= cfg.early_stop_patience:
            log.info("Early stopping at epoch %d (no improvement for %d epochs).",
                     epoch + 1, cfg.early_stop_patience)
            break

    cfg.history_path.write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=True))
    model.eval()
    log.info("Best val Dice %.4f @ epoch %d -> %s", best_dice, best_epoch, cfg.ckpt_path.name)
    return model, history

# Inference with caching
def predict_patient(cfg: Settings, model, pid: str) -> dict[tuple[str, str], np.ndarray]:
    """Native-resolution argmax masks for all four frames of one patient."""
    tf = build_transform(cfg, "test")
    preds: dict[tuple[str, str], np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for view in cfg.views:
            for instant in cfg.instants:
                raw = np.squeeze(np.asarray(
                    nib.load(frame_path(cfg.nifti_dir / pid, view, instant)).dataobj
                )).astype(np.float32)
                rgb = np.stack([normalize_intensity(raw, cfg.intensity_percentiles)] * 3, axis=-1)
                tens = tf(image=rgb)["image"].unsqueeze(0).to(cfg.device)
                pred = model(tens).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                preds[(view, instant)] = cv2.resize(
                    pad_to_square(pred), (raw.shape[1], raw.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    return preds


def get_test_predictions(cfg: Settings, model, test_ids: list[str]) -> dict:
    """Predict the test split once and cache to disk; subsequent calls reload.

    The original notebook recomputed these for the metrics block and again for
    the LVEF block; caching halves test-time GPU cost.
    """
    cache = cfg.pred_cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    single_file = cache / "test_preds.npz"
    if single_file.exists():
        blob = np.load(single_file)
        out: dict = {}
        for key in blob.files:
            pid, view, instant = key.split("__")
            out.setdefault(pid, {})[(view, instant)] = blob[key]
        return out

    all_preds: dict = {}
    for i, pid in enumerate(test_ids, 1):
        all_preds[pid] = predict_patient(cfg, model, pid)
        if i % 10 == 0:
            log.info("test predictions %d/%d", i, len(test_ids))

    flat = {f"{pid}__{v}__{inst}": m
            for pid, d in all_preds.items() for (v, inst), m in d.items()}
    np.savez_compressed(single_file, **flat)
    return all_preds


# Evaluation blocks
def evaluate_segmentation(cfg: Settings, model, test_ids: list[str],
                          preds: dict) -> "pd.DataFrame":
    """Per-frame, per-class Dice and HD95 on the official test split."""
    import pandas as pd

    from .clinical import load_mask_with_spacing

    records = []
    for pid in test_ids:
        for (view, instant), pm in preds[pid].items():
            gm, spacing = load_mask_with_spacing(cfg.nifti_dir, pid, view, instant)
            gm = gm.astype(np.uint8)
            for c in cfg.fg_classes:
                records.append({"patient": pid, "view": view, "instant": instant,
                                "class": c, "dice": dice_score(pm, gm, c),
                                "hd95_mm": hd95_mm(pm, gm, c, spacing)})
    return pd.DataFrame(records)


def evaluate_lvef(cfg: Settings, test_ids: list[str], preds: dict,
                  meta: "pd.DataFrame") -> "pd.DataFrame":
    """LVEF from predicted masks vs expert masks vs clinical reference."""
    import pandas as pd

    from .clinical import cavity_profile, ef, load_mask_with_spacing, patient_volumes, volume_ml

    ref_ef = {(r.patient, r.view): r.ref_ef for r in meta.itertuples()}
    recs = []
    for pid in test_ids:
        prof_p = {}
        for (view, instant), pm in preds[pid].items():
            w, l = cavity_profile(pm)
            _, spacing = load_mask_with_spacing(cfg.nifti_dir, pid, view, instant)
            prof_p[(view, instant)] = (w, l, spacing)
        biplane_p = {}
        for instant in cfg.instants:
            (w1, l1, s1), (w2, l2, _) = prof_p[(cfg.views[0], instant)], \
                prof_p[(cfg.views[1], instant)]
            biplane_p[instant] = volume_ml(w1, (l1 + l2) / 2, s1, partner_px=w2)
        _, biplane_e = patient_volumes(cfg.nifti_dir, pid, cfg.views, cfg.instants)
        recs.append({"patient": pid,
                     "ef_pred": ef(biplane_p["ED"], biplane_p["ES"]),
                     "ef_expert_masks": ef(biplane_e["ED"], biplane_e["ES"]),
                     "ref_ef": ref_ef[(pid, cfg.views[0])]})
    return pd.DataFrame(recs)
