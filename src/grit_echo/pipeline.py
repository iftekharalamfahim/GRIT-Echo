"""End-to-end pipeline orchestration: audit -> train -> evaluate -> RQs."""

from __future__ import annotations

import logging

import pandas as pd

from .analysis import paired_policy_comparison, rq1_quality_effect, rq2_error_prediction
from .config import Settings
from .data import assert_clean_splits, cohort_audit_table, load_metadata, read_subgroup
from .entropy import GranuleCalibrator, extract_entropy
from .segmentation import (evaluate_lvef, evaluate_segmentation, get_test_predictions,
                           train_segmentation)

log = logging.getLogger(__name__)


def run(cfg: Settings, stages: set[str] | None = None,
        force_seg: bool = False) -> dict:
    """Run selected stages (default: all). Returns a dict of result frames."""
    stages = stages or {"audit", "seg", "eval", "rq"}
    results: dict = {}

    # Stage 0: cohort audit 
    if "audit" in stages:
        audit = cohort_audit_table(cfg)
        train = read_subgroup(cfg.split_dir, "subgroup_training.txt")
        val = read_subgroup(cfg.split_dir, "subgroup_validation.txt")
        test = read_subgroup(cfg.split_dir, "subgroup_testing.txt")
        assert_clean_splits(train, val, test)
        log.info("Cohort: %d patients, %d frames | train/val/test = %d/%d/%d",
                 audit.patient.nunique(), len(audit), len(train), len(val), len(test))
        results["audit"] = audit
        results["splits"] = {"train": train, "val": val, "test": test}

    # Stage 1: segmentation training 
    model, history = train_segmentation(cfg, force=force_seg)
    results["history"] = history

    test_ids = results.get("splits", {}).get(
        "test", read_subgroup(cfg.split_dir, "subgroup_testing.txt"))
    val_ids = results.get("splits", {}).get(
        "val", read_subgroup(cfg.split_dir, "subgroup_validation.txt"))

    # Stage 2: evaluation (predictions computed once, cached)
    meta = load_metadata(cfg, set(test_ids) | set(val_ids))
    if "eval" in stages:
        preds = get_test_predictions(cfg, model, test_ids)
        test_metrics = evaluate_segmentation(cfg, model, test_ids, preds)
        ef_test = evaluate_lvef(cfg, test_ids, preds, meta)
        results["test_metrics"] = test_metrics
        results["ef_test"] = ef_test

        d_mask = ef_test.ef_pred - ef_test.ef_expert_masks
        d_ref = ef_test.ef_pred - ef_test.ref_ef
        log.info("LVEF (test): MAE vs expert masks %.2f | vs ref %.2f EF%%",
                 d_mask.abs().mean(), d_ref.abs().mean())

    # Stage 3: entropy, granules, RQ1/RQ2 
    calibrator = None
    if "rq" in stages:
        if "test_metrics" not in results:
            raise RuntimeError("Stage 'rq' requires stage 'eval' to run first.")
        val_entropy = extract_entropy(cfg, model, val_ids)
        calibrator = GranuleCalibrator.fit(val_entropy)
        test_entropy = extract_entropy(cfg, model, test_ids).merge(
            meta[["patient", "view", "quality"]], on=["patient", "view"], how="left")

        seq_ent = (test_entropy.groupby(["patient", "view", "quality"])
                   .agg(tissue_entropy=("tissue_entropy", "mean")).reset_index())
        results["rq1"] = rq1_quality_effect(seq_ent)
        log.info("RQ1: KW H=%.2f p=%.2e eps^2=%.3f",
                 results["rq1"]["H"], results["rq1"]["p"], results["rq1"]["eps_sq"])

        seq_stats = (calibrator.assign(test_entropy)
                     .groupby(["patient", "view", "quality"])
                     .agg(artifact_frac=("granule", lambda s: (s == "artifact-prone").mean()),
                          crisp_frac=("granule", lambda s: (s == "crisp").mean()),
                          mean_tissue_entropy=("tissue_entropy", "mean"))
                     .reset_index())
        seg_err = (test_metrics.groupby(["patient", "view"])
                   .agg(mean_dice=("dice", "mean"), mean_hd95=("hd95_mm", "mean"))
                   .reset_index())
        rq2_df = seq_stats.merge(seg_err, on=["patient", "view"]).dropna(
            subset=["mean_dice", "mean_hd95"])
        results["rq2"] = rq2_error_prediction(rq2_df, ef_test)
        results["rq2_frame"] = rq2_df
        log.info("RQ2: rho(artifact_frac, Dice)=%+.3f p=%.2e",
                 results["rq2"]["rho_dice"], results["rq2"]["p_dice"])

    return results
