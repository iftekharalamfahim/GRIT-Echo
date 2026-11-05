"""RQ1 and RQ2 statistical analyses (Kruskal-Wallis, correlations, gating)."""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .entropy import GRANULE_ORDER

log = logging.getLogger(__name__)

QUALITY_ORDER = ["Good", "Medium", "Poor"]


def rq1_quality_effect(seq_ent: pd.DataFrame) -> dict:
    """RQ1: tissue entropy vs expert-rated image quality (per-sequence unit).

    (patient, view) is the independent unit; frame-level rows would
    pseudoreplicate. Includes Bonferroni-corrected post-hoc Mann-Whitney U.
    """
    groups = [seq_ent.loc[seq_ent.quality == q, "tissue_entropy"].dropna().values
              for q in QUALITY_ORDER]
    H, p_kw = stats.kruskal(*groups)
    N, K = len(seq_ent), len(QUALITY_ORDER)
    eps_sq = H / (N - 1)
    eta_sq = (H - K + 1) / (N - K)

    posthoc = {}
    for a, b in combinations(QUALITY_ORDER, 2):
        xa = seq_ent.loc[seq_ent.quality == a, "tissue_entropy"].dropna()
        xb = seq_ent.loc[seq_ent.quality == b, "tissue_entropy"].dropna()
        U, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        r_rb = abs(1 - 2 * U / (len(xa) * len(xb)))
        posthoc[f"{a}_vs_{b}"] = {"U": float(U), "p_adj": float(min(p * 3, 1.0)),
                                  "r_rb": float(r_rb)}

    return {"H": float(H), "p": float(p_kw), "eps_sq": float(eps_sq),
            "eta_sq": float(eta_sq), "n": int(N), "posthoc": posthoc}


def rq2_error_prediction(rq2: pd.DataFrame, ef_test: pd.DataFrame) -> dict:
    """RQ2: granule burden vs per-case error, plus selective risk-coverage."""
    rho_dice, p_dice = stats.spearmanr(rq2.artifact_frac, rq2.mean_dice)
    rho_hd95, p_hd95 = stats.spearmanr(rq2.artifact_frac, rq2.mean_hd95)

    pat = (rq2.groupby("patient")
           .agg(artifact_frac=("artifact_frac", "max"), crisp_frac=("crisp_frac", "min"))
           .reset_index().merge(ef_test, on="patient"))
    pat["abs_ef_err"] = (pat.ef_pred - pat.ef_expert_masks).abs()
    rho_ef, p_ef = stats.spearmanr(pat.artifact_frac, pat.abs_ef_err)

    coverage_table = []
    rq2_s = rq2.sort_values("artifact_frac", ascending=True, kind="stable")
    for coverage in (1.00, 0.90, 0.80, 0.70):
        kept = rq2_s.head(int(round(coverage * len(rq2_s))))
        coverage_table.append({
            "coverage": coverage, "n": len(kept),
            "mean_dice": float(kept.mean_dice.mean()),
            "mean_hd95": float(kept.mean_hd95.mean())})

    result = {
        "rho_dice": float(rho_dice), "p_dice": float(p_dice),
        "rho_hd95": float(rho_hd95), "p_hd95": float(p_hd95),
        "rho_ef": float(rho_ef) if not np.isnan(rho_ef) else None,
        "p_ef": float(p_ef) if not np.isnan(p_ef) else None,
        "risk_coverage": coverage_table,
    }

    flagged = rq2.artifact_frac > 0.5
    result["flagged_n"] = int(flagged.sum())
    if flagged.any() and (~flagged).any():
        gate = rq2.groupby(flagged.rename("flagged")).agg(
            n=("mean_dice", "size"), mean_dice=("mean_dice", "mean"),
            mean_hd95=("mean_hd95", "mean"))
        u_stat, p_flag = stats.mannwhitneyu(
            rq2.loc[~flagged, "mean_dice"], rq2.loc[flagged, "mean_dice"],
            alternative="greater")
        n1, n2 = int((~flagged).sum()), int(flagged.sum())
        result.update({
            "gate_retained_dice": float(gate.loc[False, "mean_dice"]),
            "gate_flagged_dice": float(gate.loc[True, "mean_dice"]),
            "gate_p": float(p_flag),
            "gate_r_rb": float(abs(1 - 2 * u_stat / (n1 * n2))),
        })
    return result


