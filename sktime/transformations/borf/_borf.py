# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""BORF transformer, bag-of-receptive-fields features for time series panels."""

__author__ = ["smellslikeml"]
__all__ = ["BORF"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class BORF(BaseTransformer):
    """Bag-Of-Receptive-Fields (BORF) features for time series panels.

    This transformer is a soft-dependency wrapper around the ``fast-borf``
    package [2]_ which implements the BORF feature extraction of
    Spinnato [1]_. Each (possibly irregular or unequal-length) series in the
    panel is turned into a sparse bag of SAX "receptive-field" words:
    for every configuration of window size, word length, alphabet size,
    dilation and stride, the number of occurrences of each SAX word is
    counted, per signal of the series.

    The output is one row of word counts per series in the panel, i.e.,
    a tabular "primitives" representation, one column per
    (configuration, signal, word) combination.

    BORF is a dictionary-based feature transform in the WEASEL/BOSS family,
    and, unlike those, natively handles irregular and unequal-length series:
    series of different lengths are padded with ``NaN`` and missing
    observations are skipped when words are counted.

    Parameters
    ----------
    min_window_size : int, default=4
        Smallest window size to consider. Window sizes are powers of two
        in the range ``[min_window_size, max_window_size]``.
    max_window_size : int or None, default=None
        Largest window size to consider. If None, defaults to the length of
        the longest series in the panel.
    max_word_length : int, default=8
        Largest word length to consider. Word lengths are the powers of two
        from 2 up to ``max_word_length``.
    alphabet_sizes : sequence of int, default=(3,)
        Alphabet sizes, i.e., number of symbols per letter, to use.
    min_dilation : int, default=1
        Smallest dilation to consider. Dilations are powers of two in the
        range ``[min_dilation, max_dilation]``.
    max_dilation : int or None, default=None
        Largest dilation to consider. If None, defaults to the base-2
        logarithm of the length of the longest series in the panel.
    complexity : {"quadratic", "linear", "linear_logarithmic"},
        default="quadratic"
        How the stride of each configuration is chosen. "quadratic" uses
        stride 1, "linear" uses the word length as stride,
        "linear_logarithmic" uses a length-dependent stride.
    min_window_to_signal_std_ratio : float, default=0.0
        Windows whose standard deviation is below this fraction of the
        signal's standard deviation are treated as flat, i.e., z-score 0.
    time_channel : bool, default=False
        If True, the last column of the panel is interpreted as holding the
        timestamps of each series, which must be strictly increasing within
        each series; the remaining columns are the signals. If False,
        observations are treated as evenly spaced. For unequal-length
        panels, timestamps must be missing where all signals are missing.
    n_jobs : int, default=1
        Number of numba threads used by ``fast-borf``. -1 uses numba's
        default, all logical cores.

    Attributes
    ----------
    _borf : fast_borf.BORF
        The fitted underlying ``fast_borf.BORF`` estimator. Its fitted
        attributes, e.g., ``configs_``, ``feature_index_`` and
        ``vocabularies_``, document the learned configurations and the
        words that obtained a column.

    See Also
    --------
    sktime.classification.dictionary_based.BORFClassifier

    References
    ----------
    .. [1] Francesco Spinnato.
        "A Time-Aware Bag-of-Receptive-Fields for Interpretable Irregular
        Time Series Classification", arXiv:2609.39268, 2026.
        https://arxiv.org/abs/2609.39268
    .. [2] fast-borf Python package, MIT license, (c) Francesco Spinnato.
        https://github.com/fspinna/borf

    Examples
    --------
    >>> from sktime.transformations.borf import BORF
    >>> from sktime.datasets import load_unit_test
    >>> X, y = load_unit_test(return_X_y=True)
    >>> transformer = BORF(
    ...     max_window_size=8, max_word_length=4
    ... )  # doctest: +SKIP
    >>> Xt = transformer.fit_transform(X, y)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["smellslikeml"],
        "maintainers": ["smellslikeml"],
        "python_dependencies": ["fast-borf"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        #
        # behavioural tags: internal type
        # ----------------------------------
        # pd-multiindex natively supports multivariate and unequal-length
        # panels, and is the container the conversion below is written for
        "X_inner_mtype": "pd-multiindex",
        "y_inner_mtype": "None",
        "capability:multivariate": True,
        "requires_y": False,
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:missing_values": False,
        "capability:categorical_in_X": False,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:specific": ["sktime.transformations.borf.tests.test_borf"],
        # see #5658, numba threading is not safe under joblib multiprocessing
        "tests:skip_by_name": ["test_multiprocessing_idempotent"],
    }

    def __init__(
        self,
        min_window_size=4,
        max_window_size=None,
        max_word_length=8,
        alphabet_sizes=(3,),
        min_dilation=1,
        max_dilation=None,
        complexity="quadratic",
        min_window_to_signal_std_ratio=0.0,
        time_channel=False,
        n_jobs=1,
    ):
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.max_word_length = max_word_length
        self.alphabet_sizes = alphabet_sizes
        self.min_dilation = min_dilation
        self.max_dilation = max_dilation
        self.complexity = complexity
        self.min_window_to_signal_std_ratio = min_window_to_signal_std_ratio
        self.time_channel = time_channel
        self.n_jobs = n_jobs

        super().__init__()

    def _get_borf_params(self):
        """Collect hyper-parameters to forward to fast_borf.BORF."""
        return {
            "min_window_size": self.min_window_size,
            "max_window_size": self.max_window_size,
            "max_word_length": self.max_word_length,
            "alphabet_sizes": self.alphabet_sizes,
            "min_dilation": self.min_dilation,
            "max_dilation": self.max_dilation,
            "complexity": self.complexity,
            "min_window_to_signal_std_ratio": (
                self.min_window_to_signal_std_ratio
            ),
            "time_channel": self.time_channel,
            "n_jobs": self.n_jobs,
        }

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Panel data, of mtype pd-multiindex
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        self: reference to self
        """
        # lazy imports to avoid hard dependency
        from fast_borf import BORF as _FastBORF

        X_borf = _pd_multiindex_to_borf_array(X)
        self._borf = _FastBORF(**self._get_borf_params()).fit(X_borf, y)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : Panel data, of mtype pd-multiindex
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pd.DataFrame
            One row of word counts per series in X, one column per
            (configuration, signal, word) learned in fit.
        """
        X_borf = _pd_multiindex_to_borf_array(X)
        Xt = self._borf.transform(X_borf)

        feature_names = _borf_feature_names(self._borf.feature_index_)
        instances = X.index.get_level_values(0).unique()
        Xt = pd.DataFrame(Xt.toarray(), index=instances, columns=feature_names)

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        # small configurations so that the estimator check suite runs fast
        params1 = {
            "max_window_size": 4,
            "max_word_length": 2,
            "max_dilation": 1,
        }
        params2 = {
            "max_window_size": 8,
            "max_word_length": 4,
            "max_dilation": 2,
            "alphabet_sizes": (2,),
        }
        return [params1, params2]


def _pd_multiindex_to_borf_array(X):
    """Convert a pd-multiindex panel to the array format of fast_borf.BORF.

    Parameters
    ----------
    X : pd.DataFrame with MultiIndex (instance, time) index
        panel to convert, columns are the signals (channels) of each series

    Returns
    -------
    np.ndarray of shape (n_instances, n_channels, n_timepoints)
        float64 array as expected by ``fast_borf.BORF``, axes enumerating
        instances, signals and time points, respectively. Series of unequal
        length are padded with ``NaN`` to the length of the longest series,
        which ``fast_borf`` treats as missing observations.
    """
    grouped = X.groupby(level=0, sort=False)
    # one (n_channels, n_timepoints) array per instance,
    # rows ordered by the time index, channel order = column order
    signals = [
        group.sort_index(level=-1).to_numpy(dtype="float64").T
        for _, group in grouped
    ]

    lengths = {signal.shape[1] for signal in signals}
    if len(lengths) == 1:
        return np.stack(signals)

    # unequal-length panel: pad to the longest series with NaN,
    # which fast_borf skips when counting words
    max_length = max(lengths)
    X_borf = np.full(
        (len(signals), signals[0].shape[0], max_length), np.nan, dtype="float64"
    )
    for i, signal in enumerate(signals):
        X_borf[i, :, : signal.shape[1]] = signal

    return X_borf


def _borf_feature_names(feature_index):
    """Name the output columns from the fitted feature_index_ of fast_borf.

    Parameters
    ----------
    feature_index : np.ndarray of shape (n_features, 3)
        rows of (configuration index, signal index, word), as attached by
        ``fast_borf.BORF`` in its ``feature_index_`` fitted attribute

    Returns
    -------
    list of str
        one column name per row of feature_index
    """
    return [
        f"c{config}_s{signal}_w{word}" for config, signal, word in feature_index
    ]
