"""CPU baseline. Never downloads or evaluates the published challenge test set."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

REVISION = "f40a23ea75a56560e9a1ce6ddc1586dd6a1f25d9"
DATA_SHA256 = "c5214ad8c8a4d7d4d09082fcce24fe16c97a980a1e0873b91b8c8e474b79f6e4"
DATA_URL = f"https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data/resolve/{REVISION}/expansion_data_train.csv"
ENDPOINTS = ["LogD", "KSOL", "HLM CLint", "MLM CLint", "Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux", "MPPB", "MBPB", "MGMB"]


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def download_train(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "expansion_data_train.csv"
    if not path.exists():
        response = requests.get(DATA_URL, timeout=120)
        response.raise_for_status()
        payload = response.content
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded data checksum mismatch; refusing to continue.")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Cached data checksum mismatch. Remove the cached CSV and retry.")
    return path


def molecular_identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    # Empty Murcko scaffolds are kept in one group, conservatively preventing
    # acyclic molecules from being distributed across train and validation.
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return mol, canonical, scaffold or "ACYCLIC"


def transform(values, endpoint):
    values = np.asarray(values, dtype=float)
    return values if endpoint == "LogD" else np.log10(np.clip(values, 0, None) + 1)


def inverse_transform(values, endpoint):
    return values if endpoint == "LogD" else np.maximum(np.power(10.0, values) - 1, 0)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if not np.isfinite(y).all() or not np.isfinite(pred).all() or len(y) == 0:
        raise ValueError("Metrics require non-empty finite observations and predictions")
    mae = float(np.mean(np.abs(y - pred)))
    denominator = float(np.mean(np.abs(y - np.mean(y))))
    return {"mae": mae, "rae": mae / denominator if denominator > 0 else None}


def split_groups(groups, seed=42, fraction=0.2):
    train, valid = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(np.zeros(len(groups)), groups=groups))
    assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[valid])
    return train, valid


def run(args):
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = {"seed": args.seed, "trees": args.trees, "min_samples_leaf": 2,
              "max_features": "sqrt", "fingerprint": "Morgan bits, radius=2, size=2048, chirality=True",
              "validation_group_fraction": 0.2, "data_revision": REVISION,
              "data_sha256": DATA_SHA256, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "rdkit", "joblib"]}
    identity = {"config": config, "versions": versions}
    if (out / "identity.json").exists() and json.loads((out / "identity.json").read_text()) != identity:
        raise ValueError("Output belongs to a different experiment. Choose a new --output directory.")
    save_json(out / "identity.json", identity)
    df = pd.read_csv(download_train(Path(args.cache)))
    if not set(["Molecule Name", "SMILES", *ENDPOINTS]).issubset(df.columns):
        raise ValueError("Unexpected dataset schema")
    if df["Molecule Name"].duplicated().any():
        raise ValueError("Duplicate molecule identifiers require an explicit curation decision")
    molecules, canonical, scaffolds = zip(*(molecular_identity(s) for s in df.SMILES))
    train, valid = split_groups(scaffolds, args.seed)
    assert not set(np.array(canonical)[train]) & set(np.array(canonical)[valid])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    X = np.zeros((len(df), 2048), dtype=np.uint8)
    for i, mol in enumerate(molecules):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), X[i])
    split = df[["Molecule Name", "SMILES"]].copy()
    split["canonical_smiles"] = canonical
    split["scaffold"] = scaffolds
    split["partition"] = "train"
    split.loc[valid, "partition"] = "validation"
    split.to_csv(out / "split.csv", index=False)
    audit = {"rows": len(df), "training_rows": len(train), "validation_rows": len(valid),
             "scaffold_groups": len(set(scaffolds)), "canonical_duplicate_rows": int(pd.Series(canonical).duplicated().sum()),
             "missing_labels": {ep: int(df[ep].isna().sum()) for ep in ENDPOINTS},
             "scaffold_overlap": 0, "canonical_overlap": 0}
    save_json(out / "audit.json", audit)
    print(f"Data: {len(df)} molecules; {len(train)} train / {len(valid)} validation. Published test set untouched.", flush=True)
    for idx, ep in enumerate(ENDPOINTS):
        folder = out / f"endpoint_{idx:02d}"
        folder.mkdir(exist_ok=True)
        if (folder / "result.json").exists():
            print(f"Resume: {ep} already complete", flush=True)
            continue
        y = pd.to_numeric(df[ep], errors="raise").to_numpy(dtype=float)
        tr = train[np.isfinite(y[train])]
        va = valid[np.isfinite(y[valid])]
        if len(tr) < 10 or len(va) < 2:
            raise ValueError(f"Too few labels for {ep}: train={len(tr)}, validation={len(va)}")
        ytr, yva = transform(y[tr], ep), transform(y[va], ep)
        model = RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=2,
                                      max_features="sqrt", random_state=args.seed, n_jobs=args.jobs)
        tick = time.time()
        model.fit(X[tr], ytr)
        predictions = model.predict(X[va])
        dummy = np.full(len(va), np.median(ytr))
        result = {"endpoint": ep, "train_labels": len(tr), "validation_labels": len(va),
                  "fit_predict_seconds": time.time() - tick,
                  "random_forest": metrics(yva, predictions), "training_median": metrics(yva, dummy)}
        pd.DataFrame({"molecule_id": df.loc[va, "Molecule Name"].to_numpy(), "observed_raw": y[va],
                      "predicted_raw": inverse_transform(predictions, ep), "observed_transformed": yva,
                      "predicted_transformed": predictions, "median_transformed": dummy}).to_csv(folder / "predictions.csv", index=False)
        joblib.dump(model, folder / "model.joblib", compress=3)
        # Completion marker is written last, so interrupted endpoints are retrained.
        save_json(folder / "result.json", result)
        print(f"{ep}: RAE={result['random_forest']['rae']:.3f}; median={result['training_median']['rae']:.3f}", flush=True)
    rows = []
    for idx, ep in enumerate(ENDPOINTS):
        r = json.loads((out / f"endpoint_{idx:02d}" / "result.json").read_text())
        for name in ["random_forest", "training_median"]:
            rows.append({"endpoint": ep, "model": name, "n_train": r["train_labels"], "n_valid": r["validation_labels"], **r[name]})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "metrics.csv", index=False)
    macro = {}
    for model, frame in scores.groupby("model"):
        macro[model] = float(frame.rae.mean()) if frame.rae.notna().all() else None
    save_json(out / "summary.json", {"evaluation": "internal scaffold holdout; not an official leaderboard score",
              "macro_rae": macro, "session_seconds": time.time() - started,
              "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores.pivot(index="endpoint", columns="model", values="rae").plot.barh(figsize=(9, 5))
    plt.xlabel("Relative absolute error (lower is better)")
    plt.title("ExpansionRx: internal scaffold validation")
    plt.tight_layout()
    plt.savefig(out / "validation.png", dpi=160)
    plt.close()
    print("Validation macro RAE:", macro, "\nSaved to", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/rf_seed42")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.trees < 1 or args.jobs < 1:
        parser.error("--trees and --jobs must be positive")
    run(args)


if __name__ == "__main__":
    main()
