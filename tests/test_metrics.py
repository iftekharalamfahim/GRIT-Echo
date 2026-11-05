"""Unit tests for metric and clinical-estimator correctness (no GPU, no data)."""

import numpy as np
import pytest

from grit_echo.clinical import cavity_profile, ef, volume_ml
from grit_echo.metrics import dice_score, hd95_mm, largest_component


def test_dice_perfect_and_empty():
    pred = np.zeros((10, 10), np.uint8)
    gt = np.zeros((10, 10), np.uint8)
    pred[2:6, 2:6] = 1
    gt[2:6, 2:6] = 1
    assert dice_score(pred, gt, 1) == pytest.approx(1.0)
    # class absent in both -> NaN by contract
    assert np.isnan(dice_score(pred, gt, 3))


def test_dice_half_overlap():
    pred = np.zeros((10, 10), np.uint8)
    gt = np.zeros((10, 10), np.uint8)
    pred[0:4, 0:5] = 1
    gt[0:4, 0:10] = 1
    assert dice_score(pred, gt, 1) == pytest.approx(2 * 20 / (20 + 40))


def test_largest_component_keeps_biggest_blob():
    m = np.zeros((20, 20), bool)
    m[0:2, 0:2] = True          # 4 px speck
    m[10:16, 10:16] = True      # 36 px blob
    out = largest_component(m)
    assert out[11, 11] and not out[0, 0]


def test_hd95_zero_for_identical_masks():
    m = np.zeros((50, 50), np.uint8)
    m[10:30, 10:30] = 1
    assert hd95_mm(m, m, 1, (0.5, 0.5)) == pytest.approx(0.0)


def test_hd95_scales_with_spacing():
    m = np.zeros((50, 50), np.uint8)
    m[10:30, 10:30] = 1
    g = np.zeros((50, 50), np.uint8)
    g[10:30, 20:40] = 1  # shifted 10 px
    d1 = hd95_mm(m, g, 1, (1.0, 1.0))
    d2 = hd95_mm(m, g, 1, (2.0, 2.0))
    assert d2 == pytest.approx(2 * d1, rel=1e-6)


def test_volume_ml_matches_analytic_cylinder():
    # constant width 20 px, length 40 px, spacing 0.5 mm/px
    widths = np.full(20, 20.0)
    v = volume_ml(widths, 40.0, (0.5, 0.5))
    r_cm = 20 * 0.5 / 10 / 2
    h_cm = 40 * 0.5 / 10
    assert v == pytest.approx(np.pi * r_cm**2 * h_cm, rel=1e-6)


def test_cavity_profile_returns_disks_and_length():
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 30:70] = 1  # 40x40 square
    widths, length = cavity_profile(m, n_disks=20)
    assert len(widths) == 20
    assert length == pytest.approx(40.0, rel=0.05)


def test_ef_guard_against_zero_edv():
    assert np.isnan(ef(0.0, 10.0))
    assert ef(100.0, 40.0) == pytest.approx(60.0)
