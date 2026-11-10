"""Tests for backtest_overfitting.cv (fixed seeds throughout)."""

from __future__ import annotations

import numpy as np
import pytest

from backtest_overfitting.cv import purged_kfold_split, walk_forward_split


def _random_events(
    n: int, max_duration: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted start indices with random durations, so labels overlap heavily."""
    rng = np.random.default_rng(seed)
    t0 = np.sort(rng.integers(0, n, size=n))
    t1 = t0 + rng.integers(0, max_duration + 1, size=n)
    return t0, t1


# --- purged_kfold_split -----------------------------------------------------


def test_purged_split_never_leaks_overlapping_events() -> None:
    t0, t1 = _random_events(n=300, max_duration=15, seed=0)
    folds = list(purged_kfold_split(t0, t1, n_splits=5, embargo_pct=0.02))
    assert len(folds) == 5

    for train_idx, test_idx in folds:
        assert train_idx.size > 0, (
            "an empty training set would make the leak check vacuous"
        )
        assert np.intersect1d(train_idx, test_idx).size == 0
        tr0, tr1 = t0[train_idx][:, None], t1[train_idx][:, None]
        te0, te1 = t0[test_idx][None, :], t1[test_idx][None, :]
        overlap = (tr0 <= te1) & (tr1 >= te0)
        assert not overlap.any(), "a training event overlaps a test event"


def test_purged_split_test_folds_partition_all_samples() -> None:
    t0, t1 = _random_events(n=120, max_duration=5, seed=1)
    test_union = np.concatenate(
        [test for _, test in purged_kfold_split(t0, t1, n_splits=4)]
    )
    np.testing.assert_array_equal(np.sort(test_union), np.arange(120))


def test_purging_removes_overlapping_training_samples_exactly() -> None:
    n = 100
    t0 = np.arange(n)
    t1 = t0 + 5  # each event spans 5 steps
    folds = list(purged_kfold_split(t0, t1, n_splits=5, embargo_pct=0.0))
    train_idx, test_idx = folds[3]  # test fold is [60, 80), span [60, 84]
    np.testing.assert_array_equal(test_idx, np.arange(60, 80))
    expected_train = np.concatenate([np.arange(0, 55), np.arange(85, 100)])
    np.testing.assert_array_equal(train_idx, expected_train)


def test_embargo_removes_samples_after_test_fold() -> None:
    n = 100
    t0 = np.arange(n)
    t1 = t0.copy()  # point events: purging alone removes nothing outside the test fold
    folds = list(
        purged_kfold_split(t0, t1, n_splits=5, embargo_pct=0.05)
    )  # embargo = 5
    train_idx, test_idx = folds[3]
    np.testing.assert_array_equal(test_idx, np.arange(60, 80))
    np.testing.assert_array_equal(
        train_idx, np.concatenate([np.arange(0, 60), np.arange(85, 100)])
    )


def test_no_embargo_keeps_immediately_following_sample() -> None:
    n = 100
    t0 = np.arange(n)
    t1 = t0.copy()
    folds = list(purged_kfold_split(t0, t1, n_splits=5, embargo_pct=0.0))
    train_idx, _ = folds[3]
    assert 80 in train_idx


def test_embargo_window_never_in_training_for_random_events() -> None:
    t0, t1 = _random_events(n=200, max_duration=3, seed=4)
    embargo = 10  # 0.05 * 200
    for train_idx, test_idx in purged_kfold_split(t0, t1, n_splits=4, embargo_pct=0.05):
        assert train_idx.size > 0, (
            "an empty training set would make the embargo check vacuous"
        )
        last = int(test_idx[-1])
        window = np.arange(last + 1, min(last + 1 + embargo, 200))
        assert np.intersect1d(train_idx, window).size == 0


@pytest.mark.parametrize(
    "t0, t1, kwargs",
    [
        (np.array([3, 1, 2]), np.array([3, 2, 4]), {}),  # unsorted t0
        (np.array([0, 1, 2]), np.array([0, 0, 1]), {}),  # t1 < t0 for event 1
        (np.array([0, 1, 2]), np.array([0, 1]), {}),  # length mismatch
        (
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            {"n_splits": 4},
        ),  # more folds than samples
        (
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            {"n_splits": 1},
        ),  # need at least 2 folds
        (
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            {"embargo_pct": 1.0},
        ),  # embargo must be < 1
    ],
)
def test_purged_split_validates_eagerly(
    t0: np.ndarray, t1: np.ndarray, kwargs: dict
) -> None:
    # Validation runs at call time, before any fold is requested.
    with pytest.raises(ValueError):
        purged_kfold_split(t0, t1, **kwargs)


# --- walk_forward_split -----------------------------------------------------


def test_walk_forward_windows_are_ordered_and_disjoint() -> None:
    windows = list(walk_forward_split(n_samples=50, train_size=10, test_size=5))
    assert len(windows) > 1
    for train_idx, test_idx in windows:
        assert len(train_idx) == 10
        assert len(test_idx) == 5
        assert np.intersect1d(train_idx, test_idx).size == 0
        assert test_idx.min() == train_idx.max() + 1  # test starts right after train
        assert train_idx.min() >= 0 and test_idx.max() < 50


def test_walk_forward_step_and_window_count() -> None:
    windows = list(walk_forward_split(n_samples=20, train_size=5, test_size=3, step=3))
    starts = [int(train[0]) for train, _ in windows]
    assert starts == [0, 3, 6, 9, 12]
    _, last_test = windows[-1]
    assert last_test.tolist() == [17, 18, 19]


def test_walk_forward_default_step_is_test_size() -> None:
    default = list(walk_forward_split(30, 6, 4))
    explicit = list(walk_forward_split(30, 6, 4, step=4))
    assert len(default) == len(explicit)
    for (tr_a, te_a), (tr_b, te_b) in zip(default, explicit):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_walk_forward_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        walk_forward_split(10, train_size=8, test_size=5)
    with pytest.raises(ValueError):
        walk_forward_split(10, train_size=0, test_size=2)
    with pytest.raises(ValueError):
        walk_forward_split(10, train_size=4, test_size=2, step=0)


def test_walk_forward_exact_fit_yields_one_window() -> None:
    windows = list(walk_forward_split(n_samples=5, train_size=4, test_size=1))
    assert len(windows) == 1
    train_idx, test_idx = windows[0]
    assert train_idx.tolist() == [0, 1, 2, 3]
    assert test_idx.tolist() == [4]


def test_walk_forward_invalid_first_window_raises_eagerly() -> None:
    # Validation runs at call time, so an oversized first window never yields.
    with pytest.raises(ValueError, match="exceeds n_samples"):
        walk_forward_split(5, train_size=4, test_size=2)


# --- input validation and embargo size --------------------------------------


@pytest.mark.parametrize(
    "t0, t1",
    [
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, np.nan, 3.0])),  # NaN end time
        (np.array([0.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0])),  # NaN start time
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, np.inf])),  # infinite end time
        (np.array([-np.inf, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),  # infinite start
    ],
)
def test_purged_split_rejects_non_finite_times(t0: np.ndarray, t1: np.ndarray) -> None:
    # NaN fails every comparison, so without this check purging would be silently off.
    with pytest.raises(ValueError, match="finite"):
        purged_kfold_split(t0, t1, n_splits=2, embargo_pct=0.0)


def test_purged_split_rejects_unsorted_unsigned_t0() -> None:
    # np.diff wraps around for unsigned dtypes, so the sort check must not rely on it.
    t0 = np.array([5, 3, 7, 9], dtype=np.uint32)
    with pytest.raises(ValueError, match="sorted"):
        purged_kfold_split(t0, t0 + 1, n_splits=2)


def test_purged_split_accepts_sorted_unsigned_t0() -> None:
    t0 = np.array([3, 5, 7, 9], dtype=np.uint32)
    folds = list(purged_kfold_split(t0, t0 + 1, n_splits=2, embargo_pct=0.0))
    assert len(folds) == 2


def test_purged_split_rejects_non_integer_n_splits() -> None:
    # A float such as 2.5 must not be truncated silently to 2 folds.
    t0 = np.arange(10.0)
    with pytest.raises(TypeError):
        purged_kfold_split(t0, t0 + 1, n_splits=2.5, embargo_pct=0.0)


def test_embargo_rounds_tiny_positive_product_up_to_one_sample() -> None:
    # embargo_pct * n = 1e-10 is positive, so ceil() embargoes one sample.
    t0 = np.arange(10.0)
    folds = list(purged_kfold_split(t0, t0.copy(), n_splits=2, embargo_pct=1e-11))
    train_idx, test_idx = folds[0]
    np.testing.assert_array_equal(test_idx, np.arange(5))
    np.testing.assert_array_equal(train_idx, np.arange(6, 10))


def test_embargo_ignores_float_noise_in_the_product() -> None:
    # 0.07 * 100 is 7.000000000000001 in floating point; the embargo must be 7, not 8.
    n = 100
    t0 = np.arange(n)
    folds = list(purged_kfold_split(t0, t0.copy(), n_splits=5, embargo_pct=0.07))
    train_idx, test_idx = folds[3]
    np.testing.assert_array_equal(test_idx, np.arange(60, 80))
    np.testing.assert_array_equal(
        train_idx, np.concatenate([np.arange(0, 60), np.arange(87, 100)])
    )
