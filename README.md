# ExpansionRx baselines

Reproducible molecular-property prediction on the OpenADMET–ExpansionRx dataset, starting with a CPU baseline and a protected final evaluation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoniaval/expansionrx-baselines/blob/main/notebooks/01_fingerprint_baseline.ipynb)

**Current scope:** Morgan fingerprints → nine independent random forests, compared with training-median predictors. This is an initial baseline, not a new method or a claimed competition result. Multitask and pretrained-model comparisons are planned, not implemented.

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
2. Compare multitask Chemprop and an appropriately licensed pretrained molecular model.
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
