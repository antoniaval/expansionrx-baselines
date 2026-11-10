"""Compare predictions only when data, split membership and evaluation counts agree."""
from pathlib import Path
import numpy as np
import pandas as pd


def compare_runs(chemprop_dir, baseline_dir):
    chemprop_dir, baseline_dir = Path(chemprop_dir), Path(baseline_dir)
    cols = ["Molecule Name", "SMILES", "canonical_smiles", "scaffold", "partition"]
    splits = [pd.read_csv(d / "split.csv", keep_default_na=False)[cols].sort_values("Molecule Name").reset_index(drop=True)
              for d in [chemprop_dir, baseline_dir]]
    if not splits[0].equals(splits[1]):
        raise ValueError("Cannot compare: split membership or input molecules differ.")
    cp = pd.read_csv(chemprop_dir / "metrics.csv")
    rf = pd.read_csv(baseline_dir / "metrics.csv")
    rf_only = rf[rf.model == "random_forest"]
    cp_only = cp[cp.model == "chemprop_multitask"]
    merged = cp_only.merge(rf_only, on="endpoint", suffixes=("_chemprop", "_rf"), validate="one_to_one")
    if len(merged) != 9 or not np.array_equal(merged.n_train_chemprop, merged.n_train_rf) or not np.array_equal(merged.n_valid_chemprop, merged.n_valid_rf):
        raise ValueError("Cannot compare: endpoint coverage or label counts differ.")
    if not np.isfinite(merged[["rae_chemprop", "rae_rf"]].to_numpy()).all():
        raise ValueError("Cannot compare undefined scores.")
    # Negative delta means Chemprop is better.
    merged["delta_rae_chemprop_minus_rf"] = merged.rae_chemprop - merged.rae_rf
    merged.to_csv(chemprop_dir / "comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combined = pd.concat([rf, cp_only], ignore_index=True)
    ax = combined.pivot(index="endpoint", columns="model", values="rae").plot.barh(figsize=(11, 6))
    ax.set_xlabel("Relative absolute error (lower is better)")
    ax.set_title("Same scaffold holdout: Chemprop vs fingerprint baseline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    plt.tight_layout()
    plt.savefig(chemprop_dir / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close()
    return merged
