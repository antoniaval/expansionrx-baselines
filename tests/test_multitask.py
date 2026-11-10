import numpy as np
import pandas as pd
import pytest
from expansionrx.multitask import standardize_targets, prediction_to_scoring_space
from expansionrx.comparison import compare_runs
from expansionrx.baseline import ENDPOINTS


def test_scaling_uses_only_training_labels_and_preserves_missingness():
    y = np.array([[1., np.nan], [3., 10.], [1000., 2000.]])
    z, scaler = standardize_targets(y, [0, 1])
    np.testing.assert_allclose(scaler.mean_, [2., 10.])
    np.testing.assert_allclose(scaler.scale_, [1., 1.])
    np.testing.assert_array_equal(np.isnan(z), np.isnan(y))
    np.testing.assert_allclose(scaler.inverse_transform(z), y)


def test_predictions_are_scored_after_unscaling_with_physical_bounds():
    z = np.full((2, 9), -2.)
    pred = prediction_to_scoring_space(z, np.ones(9), np.ones(9) * 2)
    np.testing.assert_allclose(pred[:, 0], [-3., -3.])
    np.testing.assert_allclose(pred[:, 1:], 0)


def write_comparison_fixture(root, model, rae):
    root.mkdir()
    pd.DataFrame({"Molecule Name": ["a", "b"], "SMILES": ["CC", "CCC"],
                  "canonical_smiles": ["CC", "CCC"], "scaffold": ["ACYCLIC", "ACYCLIC"],
                  "partition": ["train", "validation"]}).to_csv(root / "split.csv", index=False)
    pd.DataFrame([{"endpoint": ep, "model": model, "n_train": 10, "n_valid": 2, "mae": 0.5, "rae": rae}
                  for ep in ENDPOINTS]).to_csv(root / "metrics.csv", index=False)


def test_comparison_rejects_changed_split_and_counts(tmp_path):
    cp, rf = tmp_path / "cp", tmp_path / "rf"
    write_comparison_fixture(cp, "chemprop_multitask", 0.7)
    write_comparison_fixture(rf, "random_forest", 0.8)
    table = compare_runs(cp, rf)
    np.testing.assert_allclose(table.delta_rae_chemprop_minus_rf, -0.1)
    scores = pd.read_csv(rf / "metrics.csv")
    scores.loc[0, "n_valid"] = 3
    scores.to_csv(rf / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="label counts"):
        compare_runs(cp, rf)
    split = pd.read_csv(rf / "split.csv")
    split.loc[0, "partition"] = "validation"
    split.to_csv(rf / "split.csv", index=False)
    with pytest.raises(ValueError, match="split membership"):
        compare_runs(cp, rf)
