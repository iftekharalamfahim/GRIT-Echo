# GRIT-Echo: Uncertainty-Gated Segmentation for Echocardiography

[![Project Page](https://img.shields.io/badge/Project%20Page-Visit-blue)](https://iftekharalamfahim.github.io/grit-echo/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTMI.2019.2900516-blue.svg)](https://doi.org/10.1109/TMI.2019.2900516)

**Iftekhar Alam Fahim** · September 2026

includefahim@gmail.com

---

## Overview

This repository contains a reproduction and extension of granular uncertainty
gating for echocardiographic segmentation, implemented on the CAMUS dataset
([Leclerc et al., IEEE TMI 2019](https://doi.org/10.1109/TMI.2019.2900516)). A U-Net with a ResNet-34 encoder segments the
left ventricle, myocardium, and right ventricle in 2D echo frames. Frame-wise
prediction entropy is computed for every frame, and the validation split
calibrates three granules — crisp, vague, artifact-prone — at fixed tercile
cut-points. The granule statistics are then tested against the CAMUS
image-quality labels (RQ1) and against segmentation and LVEF error on the
held-out test split (RQ2).

The main findings: entropy separates the quality grades (Kruskal–Wallis
p = 1.7 × 10⁻⁵), and the fraction of artifact-prone frames correlates with
per-sequence Dice (Spearman ρ = −0.54, p = 5.4 × 10⁻⁹). Excluding flagged
sequences raises mean Dice from 0.893 to 0.919 (p = 1.0 × 10⁻⁷).

## Research Question

**RQ1:** Does frame-wise prediction entropy, grouped into the three granules (crisp, vague, artifact-prone), track the CAMUS image-quality labels (Good / Medium / Poor)?

**RQ2:** Does frame-wise uncertainty predict where the model actually fails on the held-out test set, in terms of Dice, HD95, and LVEF error, and can a gate built on it improve mean accuracy by excluding flagged sequences?

## Results

All numbers are on the official CAMUS test split (50 patients, 200 frames).

| Metric                                          | Value                           |
| ----------------------------------------------- | ------------------------------- |
| Dice, LV / myocardium / RV                      | 0.935 / 0.877 / 0.914           |
| HD95 (mm), LV / myocardium / RV                 | 3.92 / 4.17 / 4.31              |
| LVEF MAE vs expert masks, EF-points (Pearson r) | 4.38 (0.895)                    |
| LVEF MAE vs clinical reference (Pearson r)      | 7.17 (0.864)                    |
| RQ1: Kruskal–Wallis H, p, ε²                    | 21.99, 1.7 × 10⁻⁵, 0.222        |
| RQ1: Good vs Poor, p, rank-biserial r           | 1.2 × 10⁻⁴, 0.71                |
| RQ2: ρ(artifact-prone fraction, Dice), p        | −0.543, 5.4 × 10⁻⁹              |
| RQ2: ρ(artifact-prone fraction, HD95), p        | +0.393, 5.2 × 10⁻⁵              |
| Gating: Dice of retained vs flagged sequences   | 0.919 vs 0.893 (p = 1.0 × 10⁻⁷) |

Granule distribution across quality grades (fraction of sequences):

| Quality | crisp | vague | artifact-prone |
| ------- | ----- | ----- | -------------- |
| Good    | 0.362 | 0.383 | 0.255          |
| Medium  | 0.105 | 0.447 | 0.447          |
| Poor    | 0.000 | 0.267 | 0.733          |

Notes. HD95 is the 95th-percentile symmetric boundary distance in millimetres
at native spacing. LVEF follows the Simpson biplane method of disks (ASE 2015).
The offset against the clinical reference reflects implementation differences
in volume estimation; comparisons against expert masks are the primary
endpoint.  
Evaluation is deterministic: fixed seeds, `model.eval()`, argmax
decoding.

## Experimental setup

| Item               | Value                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------- |
| Host               | Kaggle notebook, NVIDIA Tesla T4                                                            |
| Environment        | Python 3.12.13, PyTorch 2.10.0+cu128, NumPy 2.0.2                                           |
| Dataset            | CAMUS, 500 patients, 2,000 annotated frames (A2C/A4C, ED/ES)                                |
| Split              | Official 400 / 50 / 50 (patient-level, leakage-checked)                                     |
| Segmentation model | U-Net, ResNet-34 encoder (ImageNet init), 4-class output                                    |
| Training           | 20 epochs max, AdamW, Dice + cross-entropy, AMP, grad clipping, early stopping (patience 6) |

Pipeline:

1. Cohort audit and integrity checks on the official split files.
2. Train the U-Net; select on validation Dice.
3. Predict the test split once, cache, and score Dice and HD95.
4. Compute Simpson biplane LVEF from predicted and expert masks.
5. Compute frame-wise entropy; freeze granule cut-points on validation.
6. Run the RQ1 and RQ2 statistics, the risk–coverage curve, and the gate.

## Repository layout

| Path                            | Contents                                                      |
| ------------------------------- | ------------------------------------------------------------- |
| `grit_echo_pipeline.ipynb`      | Main notebook (Kaggle-ready, runs top to bottom)              |
| `src/grit_echo/config.py`       | Paths and settings; Kaggle/local auto-detection               |
| `src/grit_echo/data.py`         | Cohort audit, splits, letterbox transforms, loaders           |
| `src/grit_echo/metrics.py`      | Dice, HD95, largest connected component                       |
| `src/grit_echo/clinical.py`     | Simpson biplane LVEF, method of disks                         |
| `src/grit_echo/segmentation.py` | U-Net training and cached test inference                      |
| `src/grit_echo/entropy.py`      | Frame-wise entropy and granule calibrator                     |
| `src/grit_echo/analysis.py`     | RQ1/RQ2 statistics, gating, risk–coverage                     |
| `src/grit_echo/pipeline.py`     | Stage orchestrator                                            |
| `src/grit_echo/cli.py`          | Command-line entry point (`grit-echo audit`, `grit-echo run`) |
| `tests/test_metrics.py`         | Unit tests for metrics and the EF estimator                   |

## Reproducing

On Kaggle:

1. Open `grit_echo_pipeline.ipynb` and attach `camus-v1` and
   `camus-half-sequences` dataset.
2. Enable the GPU (T4) and run all cells. Full run takes about 40 minutes,
   most of it U-Net training.

Locally:

1. `pip install -e ".[dev]"`
2. `pytest` runs the unit tests without data.
3. `grit-echo audit` verifies the cohort without a GPU.
4. `grit-echo run --stages audit,seg,eval,rq` runs the full pipeline.

Paths auto-detect Kaggle vs local and can be overridden with the `GRIT_NIFTI`,
`GRIT_SPLIT`, `GRIT_SEQ`, and `GRIT_WORK` environment variables.

## Note on the reinforcement learning stage

An earlier stage of this project implemented a PPO contour tracker on top of
the segmentation (posterior-cached Gym environment, FrameCNN policy extractor,
Stable-Baselines3). It was trained to convergence twice under opposite
hyperparameter regimes (100k–300k steps). Both runs converged to the same
passive equilibrium: terminal ES Dice stayed at 0.60, equal to the zero-action
baseline, while the critic's explained variance exceeded 0.7. The cause is
structural rather than a tuning failure. The environment's observations
already contain the full U-Net posterior, so no policy can recover boundary
information that is not already present, and the optimal action is to keep the
initial contour. RL is therefore excluded from the final pipeline. The result
serves as a control experiment consistent with RQ1/RQ2: the U-Net posterior is
locally reliable, and error concentrates where entropy is high.

## Acknowledgements

- Scientific foundation: the CAMUS benchmark, introduced in S. Leclerc,
  E. Smistad, J. Pedrosa, A. Østvik, F. Cervenansky, F. Espinosa, T. Espeland,
  E. A. R. Berg, M.-M. Grenne, J. D'hooge, L. Lovstakken, and O. Bernard,
  "Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D
  Echocardiography," IEEE Transactions on Medical Imaging, vol. 38, no. 9,
  pp. 2198–2210, 2019. [DOI: 10.1109/TMI.2019.2900516](https://doi.org/10.1109/TMI.2019.2900516)
  in this work come from that study.
- Dataset: CAMUS was produced at the University Hospital of Nantes (CREATIS)
  and released openly.
- Clinical endpoint: R. M. Lang et al., "Recommendations for Cardiac Chamber
  Quantification by Echocardiography in Adults: An Update from the American
  Society of Echocardiography and the European Association of
  Cardiovascular Imaging," JASE, vol. 28, no. 1, pp. 1–39, 2015.
- Compute: Kaggle (NVIDIA Tesla T4).
- Libraries: PyTorch, segmentation-models-pytorch, albumentations, nibabel,
  SciPy, pandas, and Stable-Baselines3 (RL exploration stage).

## License

[MIT Licsence](https://opensource.org/license/MIT)
