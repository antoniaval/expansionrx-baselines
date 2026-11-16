# ExpansionRx baselines

Reproducible molecular-property prediction on the OpenADMET–ExpansionRx dataset, starting with a CPU baseline and a protected final evaluation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniaval/expansionrx-baselines/blob/main/notebooks/01_fingerprint_baseline.ipynb)

**Current scope:** Morgan fingerprints → nine independent random forests, training-median predictors, and a nine-task Chemprop directed message-passing neural network. Pretrained-model comparisons are planned. These are development baselines, not new methods or claimed competition placements.

## First verified result

The default 200-tree baseline achieved **0.671 validation macro RAE**, compared with **1.010** for the training-median predictor, on one scaffold holdout. These are preliminary internal-validation scores, not leaderboard results. [See the saved metrics, label counts, configuration and plot](reports/first_baseline/README.md).

## Run in Colab

1. Click **Open in Colab** above and optionally save a copy to your Drive.
2. Select a **CPU runtime** for this baseline; a GPU provides no benefit to this implementation.
3. Run the cells in order. The notebook can mount Drive to retain results across runtime resets.
4. Inspect the per-endpoint table, plot, and aggregate validation error.

The notebook installs pinned packages and calls the same Python implementation used locally. The default fits 200 trees per endpoint. Allow several minutes to tens of minutes, depending on the allocated CPU; this is an estimate, not a Colab timing measurement. A 20-tree run can check setup but is not the main experiment. No paid service or API key is needed.

## Run locally

Python 3.11–3.13:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
expansionrx-baseline --output runs/rf_seed42 --trees 200 --jobs 2
pytest
```

Repeat the same command to resume: completed endpoints are skipped. An interrupted endpoint is retrained. Different settings, package versions or source code require a new output folder. Run separate seeds in separate folders.

## Multitask Chemprop on a Colab GPU

[![Open Chemprop in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniaval/expansionrx-baselines/blob/main/notebooks/02_multitask_chemprop.ipynb)

Choose a **T4 GPU** runtime, or **CPU** when GPUs are unavailable, then run notebook 02 in order. The notebook automatically detects CUDA and otherwise trains on CPU; model size and epoch count stay the same. It installs an isolated Python 3.11 environment (independent of Colab’s host Python), mounts Drive if requested, reuses or creates the matching fingerprint baseline, trains Chemprop and produces a comparison table and plot. Expect a several-minute first installation; GPU training time has not yet been measured by this project.

Default model: Chemprop 2.2.1, one shared directed bond-message-passing network (hidden size 300, depth 3), mean aggregation, a nine-output regression head, dropout 0.1, 50 epochs and batch size 64. No external pretraining or added molecular descriptors. Dependencies are pinned in the `chemprop` extra; the notebook isolates them from Colab's preinstalled packages.

```bash
pip install -e '.[chemprop]'
expansionrx-chemprop --accelerator gpu --output runs/chemprop_seed42_split42 \
  --baseline-dir runs/rf_seed42 --split-seed 42 --seed 42 --epochs 50
```

Use `--accelerator cpu --epochs 2 --hidden-size 32` with a new output folder for a small integration check. This is not the full model experiment.

**Fair comparison:** the dataset revision, scaffold split and evaluation formulas are shared with the forest. Model seed and split seed are separate. Missing targets are masked, not filled with fabricated labels. Target standardization is fitted only on training observations; predictions are unscaled before evaluation. Non-LogD predictions are clipped to their nonnegative transformed domain, matching the challenge's raw-value clipping.

**Checkpoint selection:** Chemprop selects the lowest masked standardized validation MSE, not the lowest macro RAE. Its default loss aggregates observed entries, so densely measured tasks contribute more observations; the reported macro RAE still weights endpoints equally. Since the holdout selects the epoch, its score is a development estimate. The original published test remains untouched.

**Resume:** checkpoints and CSV logs are saved every epoch. Rerunning the identical command restores `last.ckpt`, including optimizer/scheduler state. Completed training reloads `best.ckpt`. Keep the same path, code and package versions. Exact bitwise reproduction of uninterrupted training after a disconnect is not promised. Changed settings require a new folder; the fixed learning-rate schedule means changing the planned epoch budget is a new experiment.

**Outputs:** `metrics.csv`, `predictions.csv`, `target_scaler.json`, `checkpoints/best.ckpt`, `checkpoints/last.ckpt`, training logs, and (when `--baseline-dir` is supplied) `comparison.csv`/`comparison.png`. Negative `delta_rae_chemprop_minus_rf` favours Chemprop. Keep the scaler alongside the checkpoint for inference. Load only trusted checkpoints.

## Evaluation protocol

- Download **only the original challenge training CSV**, at an immutable Hugging Face revision; verify its SHA-256.
- Use RDKit Bemis–Murcko scaffolds as groups with `GroupShuffleSplit(seed=42, test_size=0.2)`. Twenty percent refers to scaffold groups, not necessarily rows. All acyclic molecules share one group. This is internal scaffold validation, **not a temporal split**.
- Retain stereochemistry in Morgan fingerprints (radius 2, 2,048 bits); group scaffolds without stereochemistry to prevent stereoisomer leakage. Check canonical-molecule and scaffold overlap explicitly.
- Train a separate model per endpoint using only its available labels. Never impute target labels. No molecular identifiers are used as features.
- Keep LogD unchanged; apply `log10(1 + max(y, 0))` to the other endpoints, following the organizer's scoring transformation. Train and score in those coordinates.
- Report MAE and RAE, where `RAE = MAE / mean(abs(y_validation - mean(y_validation)))`. Average RAE equally over all nine endpoints. The denominator uses evaluation labels by definition; it is not fitted preprocessing.
- The median baseline is fitted on training labels only. RAE of one corresponds to the validation-mean reference in the metric, **not** necessarily to this training-median baseline.
- Save label counts per endpoint. Sparse endpoints can have unstable scores. One split and one seed are preliminary evidence; repeated runs and uncertainty analysis come later.

Our point-estimate RAE uses the same formula and transform as the organizer. It does not replicate the organizer's bootstrap aggregation, and is measured on a different split. **Do not compare these validation numbers directly with the competition leaderboard.**

The published challenge test labels are not downloaded. Freeze model choices before adding the later chronological-test evaluation. Because the challenge has ended and labels are public, any future comparison is retrospective, not an official blind placement.

## Outputs

Each run saves:

- `identity.json`: configuration, package versions, source checksum and dataset revision/checksum.
- `audit.json` and `split.csv`: missing-label counts and exact split membership.
- `endpoint_*/model.joblib`, `predictions.csv`, `result.json`: fitted models and completed-endpoint checkpoints.
- `metrics.csv`, `summary.json`, `validation.png`: results ready for analysis.

Models and predictions are saved locally/Drive and excluded from Git. Load only trusted joblib files. Training works from local VM storage; outputs can be written to Drive. Resume across machines requires the same pinned package versions.

## Research direction

**How much ADMET prediction performance can public data and a modest compute budget deliver?**

1. Establish fingerprint baselines and validate the data pipeline (this release).
2. Run the implemented multitask Chemprop comparison; then add an appropriately licensed pretrained molecular model.
3. Select one extension from evidence in training-only error analysis.
4. Freeze the protocol; evaluate the original time-based test split and report accuracy, limitations and measured compute.

No benefit from bounded-measurement modelling or any other extension is assumed in advance.

## Data and attribution

Data: Expansion Therapeutics and the OpenADMET consortium, [original training dataset](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data), licensed **CC BY 4.0**. The dataset is downloaded separately and is not relicensed by this repository.

- Pinned revision: `f40a23ea75a56560e9a1ce6ddc1586dd6a1f25d9`
- Training CSV SHA-256: `c5214ad8c8a4d7d4d09082fcce24fe16c97a980a1e0873b91b8c8e474b79f6e4`
- [Organizer's evaluation implementation](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge/blob/main/utils.py)
- [Official challenge tutorial](https://github.com/OpenADMET/ExpansionRx-Challenge-Tutorial)
- [Challenge results and temporal-split description](https://openadmet.ghost.io/the-openadmet-expansionrx-blind-challenge-has-come-to-an-end/)

Project code is MIT licensed. Research benchmark software; no clinical or general-purpose ADMET validation is claimed.

If notebook 02 fails during setup, its setup output now includes the full underlying installation or import error. Reopen the current notebook from GitHub to load this fix. The new `.chemprop-env-py311` environment leaves old environments and saved results untouched.
