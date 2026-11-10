import numpy as np
from expansionrx.baseline import molecular_identity, transform, inverse_transform, metrics, split_groups


def test_scaffolds_and_stereoisomers_stay_together():
    smiles = ["c1ccccc1O", "c1ccccc1N", "C[C@H](O)c1ccccc1", "C[C@@H](O)c1ccccc1", "C1CCCCC1", "C1CCCC1", "CCO", "CCN", "c1ccncc1", "C1CCC1"]
    groups = [molecular_identity(s)[2] for s in smiles]
    tr, va = split_groups(groups)
    assert not {groups[i] for i in tr} & {groups[i] for i in va}
    assert groups[0] == groups[1] == groups[2] == groups[3]
    assert groups[6] == groups[7]


def test_challenge_transform_and_round_trip():
    np.testing.assert_allclose(transform([-2, 0, 9, 99], "KSOL"), [0, 0, 1, 2])
    np.testing.assert_allclose(transform([-2, 0, 9], "LogD"), [-2, 0, 9])
    np.testing.assert_allclose(inverse_transform(transform([0, 9, 99], "KSOL"), "KSOL"), [0, 9, 99])
    assert np.isnan(transform([np.nan], "KSOL")[0])


def test_rae_denominator_and_constant_endpoint():
    assert metrics([0, 2], [1, 1]) == {"mae": 1.0, "rae": 1.0}
    assert metrics([0, 2], [0, 2])["rae"] == 0
    assert metrics([1, 1], [0, 0])["rae"] is None
