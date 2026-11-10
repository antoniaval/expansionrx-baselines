"""Train a nine-task Chemprop D-MPNN on original training data only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from . import baseline
from .baseline import ENDPOINTS, download_train, molecular_identity, split_groups, transform, inverse_transform, metrics, save_json
from .comparison import compare_runs


def prepare_data(cache, split_seed):
    df = pd.read_csv(download_train(Path(cache)))
    if df["Molecule Name"].duplicated().any():
        raise ValueError("Duplicate molecule identifiers require explicit curation.")
    identities = [molecular_identity(s) for s in df.SMILES]
    canonical = [x[1] for x in identities]
    scaffolds = [x[2] for x in identities]
    tr, va = split_groups(scaffolds, split_seed)
    if set(np.array(canonical)[tr]) & set(np.array(canonical)[va]):
        raise ValueError("Canonical molecule overlap detected.")
    raw = df[ENDPOINTS].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if np.isinf(raw).any():
        raise ValueError("Infinite labels are invalid.")
    y = np.column_stack([transform(raw[:, j], ep) for j, ep in enumerate(ENDPOINTS)])
    if (np.isfinite(y[tr]).sum(axis=0) < 10).any() or (np.isfinite(y[va]).sum(axis=0) < 2).any():
        raise ValueError("Insufficient train/validation labels in at least one task.")
    split = df[["Molecule Name", "SMILES"]].copy()
    split["canonical_smiles"] = canonical
    split["scaffold"] = scaffolds
    split["partition"] = "train"
    split.loc[va, "partition"] = "validation"
    return df, raw, y, tr, va, split


def standardize_targets(y, train_indices):
    """StandardScaler ignores NaNs when fitting and preserves them when transforming."""
    scaler = StandardScaler().fit(y[train_indices])
    z = scaler.transform(y)
    if not np.array_equal(np.isnan(y), np.isnan(z)):
        raise ValueError("Missing-target mask changed during scaling.")
    return z, scaler


def prediction_to_scoring_space(z, mean, scale):
    p = np.asarray(z) * np.asarray(scale) + np.asarray(mean)
    # These endpoints are log10(1 + nonnegative quantity). Match the scorer's
    # clipping in raw units; LogD can remain negative.
    p[:, 1:] = np.maximum(p[:, 1:], 0)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite model predictions.")
    return p


def make_model(hidden_size=300, depth=3, dropout=0.1, epochs=50):
    from chemprop import models, nn
    mp = nn.BondMessagePassing(d_h=hidden_size, depth=depth, dropout=dropout)
    ffn = nn.RegressionFFN(n_tasks=len(ENDPOINTS), input_dim=mp.output_dim,
                           hidden_dim=hidden_size, n_layers=1, dropout=dropout)
    return models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False,
                       warmup_epochs=min(2, epochs - 1), init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)


def run(args):
    # Required by deterministic CUDA matrix multiplication; set before CUDA initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from chemprop import data, models
    started = time.time()
    torch.set_num_threads(args.threads)
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. In Colab choose Runtime > Change runtime type > T4 GPU, then rerun setup.")
    pl.seed_everything(args.seed, workers=True)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ["seed", "split_seed", "epochs", "batch_size", "hidden_size", "depth", "dropout"]}
    config.update({"endpoints": ENDPOINTS, "data_revision": baseline.REVISION, "data_sha256": baseline.DATA_SHA256,
                   "precision": "32-true", "selection_metric": "masked standardized validation MSE (val_loss)",
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "baseline_source_sha256": hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest()})
    versions = {p: importlib.metadata.version(p) for p in ["chemprop", "torch", "lightning", "torchmetrics", "numpy", "pandas", "scikit-learn", "rdkit"]}
    identity = {"config": config, "versions": versions}
    identity_path = out / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Run settings, source or packages changed. Use a new --output folder; do not mix experiments.")
    save_json(identity_path, identity)
    df, raw, y, tr, va, split = prepare_data(args.cache, args.split_seed)
    split.to_csv(out / "split.csv", index=False)
    z, scaler = standardize_targets(y, tr)
    save_json(out / "target_scaler.json", {"endpoints": ENDPOINTS, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                           "fit_partition": "train only"})
    save_json(out / "audit.json", {"rows": len(df), "training_rows": len(tr), "validation_rows": len(va),
             "training_labels": dict(zip(ENDPOINTS, np.isfinite(y[tr]).sum(axis=0).tolist())),
             "validation_labels": dict(zip(ENDPOINTS, np.isfinite(y[va]).sum(axis=0).tolist())),
             "canonical_overlap": 0, "scaffold_overlap": 0})
    train_ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(df.SMILES.iloc[i], z[i]) for i in tr])
    val_ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(df.SMILES.iloc[i], z[i]) for i in va])
    train_ds.cache = True
    val_ds.cache = True
    train_loader = data.build_dataloader(train_ds, batch_size=args.batch_size, num_workers=0, shuffle=True, seed=args.seed)
    val_loader = data.build_dataloader(val_ds, batch_size=args.batch_size, num_workers=0, shuffle=False)
    checkpoints = out / "checkpoints"
    callback = ModelCheckpoint(dirpath=checkpoints, filename="best", monitor="val_loss", mode="min",
                               save_top_k=1, save_last=True, enable_version_counter=False)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, max_epochs=args.epochs,
                         precision="32-true", deterministic=True, callbacks=[callback],
                         logger=CSVLogger(str(out), name="training_logs", version=0),
                         num_sanity_val_steps=0, log_every_n_steps=10, enable_progress_bar=args.progress)
    last = checkpoints / "last.ckpt"
    best = checkpoints / "best.ckpt"
    done = out / "training_complete.json"
    if not done.exists():
        model = make_model(args.hidden_size, args.depth, args.dropout, args.epochs)
        print(f"Training {len(tr)} molecules across nine tasks; validating {len(va)}. Resume={last.exists()}", flush=True)
        trainer.fit(model, train_loader, val_loader, ckpt_path=str(last) if last.exists() else None)
        if trainer.current_epoch < args.epochs:
            raise RuntimeError("Training stopped before the planned epoch budget. Rerun the same command to resume.")
        if not best.exists():
            raise RuntimeError("Training did not produce a best checkpoint.")
        save_json(done, {"epochs": args.epochs, "best_checkpoint": "checkpoints/best.ckpt",
                         "best_standardized_validation_mse": float(callback.best_model_score),
                         "training_session_seconds": time.time() - started})
    else:
        print("Training already complete; evaluating saved best checkpoint.", flush=True)
    # Checkpoints are produced by this run; never load untrusted checkpoint files.
    model = models.MPNN.load_from_checkpoint(best, map_location="cpu", weights_only=False)
    batches = trainer.predict(model, dataloaders=val_loader)
    zpred = torch.cat(batches).detach().cpu().numpy()
    if zpred.shape != (len(va), len(ENDPOINTS)):
        raise ValueError(f"Unexpected prediction shape: {zpred.shape}")
    pred = prediction_to_scoring_space(zpred, scaler.mean_, scaler.scale_)
    rows = []
    records = []
    for j, ep in enumerate(ENDPOINTS):
        observed = np.isfinite(y[va, j])
        obs_y, obs_p = y[va, j][observed], pred[:, j][observed]
        row = {"endpoint": ep, "model": "chemprop_multitask", "n_train": int(np.isfinite(y[tr, j]).sum()),
               "n_valid": int(observed.sum()), **metrics(obs_y, obs_p)}
        rows.append(row)
        for k, idx in enumerate(va):
            records.append({"molecule_id": df["Molecule Name"].iloc[idx], "endpoint": ep,
                            "observed_raw": raw[idx, j], "observed_transformed": y[idx, j],
                            "predicted_transformed": pred[k, j], "predicted_raw": inverse_transform(pred[k, j], ep)})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
    summary = {"evaluation": "internal scaffold holdout; best epoch selected on this holdout; not an unbiased final test",
               "macro_rae": float(scores.rae.mean()) if scores.rae.notna().all() else None,
               "session_seconds": time.time() - started, "python": platform.python_version(),
               "device": torch.cuda.get_device_name(0) if args.accelerator == "gpu" else "CPU",
               "max_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if args.accelerator == "gpu" else 0.0}
    save_json(out / "summary.json", summary)
    if args.baseline_dir:
        table = compare_runs(out, args.baseline_dir)
        print(table[["endpoint", "rae_chemprop", "rae_rf", "delta_rae_chemprop_minus_rf"]].to_string(index=False), flush=True)
    print("Chemprop validation macro RAE:", summary["macro_rae"], "\nSaved to", out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/chemprop_seed42_split42")
    p.add_argument("--cache", default="data")
    p.add_argument("--baseline-dir")
    p.add_argument("--seed", type=int, default=42, help="Model initialization and data-loader seed")
    p.add_argument("--split-seed", type=int, default=42, help="Keep fixed to compare with the original baseline")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size, args.depth, args.threads) < 1 or not 0 <= args.dropout < 1:
        p.error("Sizes/epochs/threads must be positive and dropout must be in [0, 1).")
    if args.epochs < 2:
        p.error("Use at least 2 epochs to allow the warmup/cooldown learning-rate schedule.")
    run(args)


if __name__ == "__main__":
    main()
