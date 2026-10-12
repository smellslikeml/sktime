#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for the BORF transformer and BORF classifier."""

__author__ = ["smellslikeml"]

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold

from sktime.classification.dictionary_based import BORFClassifier
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.borf import BORF
from sktime.transformations.borf._borf import _pd_multiindex_to_borf_array
from sktime.utils._testing.panel import _make_panel

# small configurations, so tests run fast where the soft dependency is present
SMALL_PARAMS = {"max_window_size": 4, "max_word_length": 2, "max_dilation": 1}


def _make_known_panel(lengths, n_columns=1):
    """pd-multiindex panel with known values, one series per given length.

    The value at instance i, time t, channel c is ``100 * i + 10 * t + c``,
    so that instance, channel and time ordering can be checked exactly.
    """
    frames = []
    for i, n_timepoints in enumerate(lengths):
        index = pd.MultiIndex.from_product(
            [[f"inst_{i}"], range(n_timepoints)], names=["instances", "timepoints"]
        )
        data = {
            f"var_{c}": [100.0 * i + 10.0 * t + c for t in range(n_timepoints)]
            for c in range(n_columns)
        }
        frames.append(pd.DataFrame(data, index=index))
    return pd.concat(frames)


def _make_unequal_panel(n_columns=1, random_state=42):
    """pd-multiindex panel of variable-length random series."""
    rng = np.random.default_rng(random_state)
    frames = []
    for i, n_timepoints in enumerate([8, 12, 10, 6]):
        index = pd.MultiIndex.from_product(
            [[f"inst_{i}"], range(n_timepoints)], names=["instances", "timepoints"]
        )
        data = {f"var_{c}": rng.normal(size=n_timepoints) for c in range(n_columns)}
        frames.append(pd.DataFrame(data, index=index))
    return pd.concat(frames)


def _make_separable_panel(n_per_class=10, n_timepoints=20, random_state=42):
    """Two-class pd-multiindex panel: flat noise versus an increasing ramp.

    The two classes differ in their local slope, which SAX words encode,
    so BORF features are expected to separate the classes.

    Returns the panel and the class labels as a pd.Series indexed by
    instance, so that panels can be split by instance in cross-validation.
    """
    rng = np.random.default_rng(random_state)
    frames = []
    labels = {}
    for label in (0, 1):
        for i in range(n_per_class):
            instance = f"inst_{label}_{i}"
            index = pd.MultiIndex.from_product(
                [[instance], range(n_timepoints)],
                names=["instances", "timepoints"],
            )
            noise = rng.normal(0.0, 0.05, size=n_timepoints)
            if label == 0:
                values = noise
            else:
                values = np.linspace(0.0, 5.0, n_timepoints) + noise
            frames.append(pd.DataFrame({"var_0": values}, index=index))
            labels[instance] = label
    return pd.concat(frames), pd.Series(labels, name="class")


# ---------------------------------------------------------------------------
# data-format bridge and import guard, run without the soft dependency
# ---------------------------------------------------------------------------


def test_convert_equal_length_panel():
    """The bridge maps instance to row, channel to column, time to slice."""
    X = _make_known_panel(lengths=[4, 4], n_columns=2)
    X_borf = _pd_multiindex_to_borf_array(X)

    assert X_borf.shape == (2, 2, 4)
    assert X_borf.dtype == np.float64
    assert not np.isnan(X_borf).any()
    for i in range(2):
        for c in range(2):
            for t in range(4):
                assert X_borf[i, c, t] == 100.0 * i + 10.0 * t + c


def test_convert_unequal_length_panel():
    """Unequal-length series are NaN-padded to the longest series."""
    X = _make_known_panel(lengths=[3, 5], n_columns=1)
    X_borf = _pd_multiindex_to_borf_array(X)

    assert X_borf.shape == (2, 1, 5)
    # observed values are preserved in the leading positions
    for i, length in enumerate([3, 5]):
        for t in range(length):
            assert X_borf[i, 0, t] == 100.0 * i + 10.0 * t
        # padding beyond the series length is NaN, which fast_borf skips
        assert np.isnan(X_borf[i, 0, length:]).all()


def test_module_import_does_not_require_fast_borf(monkeypatch):
    """Module import and construction work without the soft dependency."""
    import sktime.transformations.borf._borf as borf_module

    # a None entry blocks import of the soft dependency
    monkeypatch.setitem(sys.modules, "fast_borf", None)

    reloaded = importlib.reload(borf_module)
    assert hasattr(reloaded, "BORF")

    # the constructor must not import the soft dependency either
    instance = reloaded.BORF()
    assert instance.min_window_size == 4


# ---------------------------------------------------------------------------
# BORF transformer, run only with the soft dependency present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_for_class(BORF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_basic_functionality():
    """Test fit_transform on an equal-length univariate panel."""
    X = _make_panel(
        n_instances=7, n_columns=1, n_timepoints=15, random_state=42,
        return_mtype="pd-multiindex",
    )

    transformer = BORF(**SMALL_PARAMS)
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == 7
    assert Xt.shape[1] > 0
    # word counts are finite, non-negative integers
    assert np.isfinite(Xt.to_numpy()).all()
    assert (Xt.to_numpy() >= 0).all()


@pytest.mark.skipif(
    not run_test_for_class(BORF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multivariate_panel():
    """Test that each signal of a multivariate panel obtains columns."""
    X = _make_panel(
        n_instances=5, n_columns=2, n_timepoints=15, random_state=42,
        return_mtype="pd-multiindex",
    )

    Xt = BORF(**SMALL_PARAMS).fit_transform(X)

    assert Xt.shape[0] == 5
    assert any("_s0_" in col for col in Xt.columns)
    assert any("_s1_" in col for col in Xt.columns)


@pytest.mark.skipif(
    not run_test_for_class(BORF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_unequal_length_panel():
    """Test that variable-length series are accepted, per the capability tag."""
    X = _make_unequal_panel()

    Xt = BORF(**SMALL_PARAMS).fit_transform(X)

    assert Xt.shape[0] == 4
    assert Xt.shape[1] > 0
    assert np.isfinite(Xt.to_numpy()).all()


@pytest.mark.skipif(
    not run_test_for_class(BORF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_transform_columns_stable():
    """Test that transform returns the columns learned in fit."""
    X = _make_panel(
        n_instances=7, n_columns=1, n_timepoints=15, random_state=42,
        return_mtype="pd-multiindex",
    )
    X_test = _make_panel(
        n_instances=4, n_columns=1, n_timepoints=15, random_state=43,
        return_mtype="pd-multiindex",
    )

    transformer = BORF(**SMALL_PARAMS).fit(X)
    Xt = transformer.transform(X)
    Xt_test = transformer.transform(X_test)

    assert list(Xt.columns) == list(Xt_test.columns)
    assert Xt.shape[0] == 7
    assert Xt_test.shape[0] == 4


@pytest.mark.skipif(
    not run_test_for_class(BORF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_channel_mismatch_raises():
    """Test that transforming with a different number of channels errors."""
    X = _make_panel(
        n_instances=5, n_columns=2, n_timepoints=15, random_state=42,
        return_mtype="pd-multiindex",
    )
    X_univariate = _make_panel(
        n_instances=5, n_columns=1, n_timepoints=15, random_state=42,
        return_mtype="pd-multiindex",
    )

    transformer = BORF(**SMALL_PARAMS).fit(X)
    with pytest.raises(ValueError, match="signals"):
        transformer.transform(X_univariate)


@pytest.mark.skipif(
    not run_test_for_class(BORF),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_features_classify_above_chance():
    """Test that BORF features separate classes better than chance."""
    X, y = _make_separable_panel()
    # sklearn splitters see rows, not instances, so split on the instance index
    instances = y.index.to_numpy()

    scores = []
    for train, test in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0
    ).split(instances.reshape(-1, 1), y.to_numpy()):
        transformer = BORF(**SMALL_PARAMS)
        Xt_train = transformer.fit_transform(X.loc[instances[train]], y.iloc[train])
        Xt_test = transformer.transform(X.loc[instances[test]])
        classifier = RidgeClassifierCV().fit(Xt_train, y.iloc[train])
        scores.append(classifier.score(Xt_test, y.iloc[test]))

    assert np.mean(scores) > 0.75


# ---------------------------------------------------------------------------
# BORFClassifier, run only with the soft dependency present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not run_test_for_class(BORFClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_classifier_scores_above_chance():
    """Test that the BORF pipeline classifier beats chance out-of-sample."""
    X, y = _make_separable_panel()
    # sklearn splitters see rows, not instances, so split on the instance index
    instances = y.index.to_numpy()

    scores = []
    for train, test in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0
    ).split(instances.reshape(-1, 1), y.to_numpy()):
        classifier = BORFClassifier(**SMALL_PARAMS)
        classifier.fit(X.loc[instances[train]], y.iloc[train])
        predictions = classifier.predict(X.loc[instances[test]])
        scores.append(np.mean(predictions == y.iloc[test].to_numpy()))

    assert np.mean(scores) > 0.75
