"""BORF Classifier.

Pipeline classifier using the BORF transformer and a
``sklearn.linear_model.RidgeClassifierCV`` estimator.
"""

__author__ = ["smellslikeml"]
__all__ = ["BORFClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.transformations.borf import BORF


class BORFClassifier(BaseClassifier):
    """Bag-Of-Receptive-Fields (BORF) classifier.

    This classifier transforms the input panel using the BORF transformer,
    a soft-dependency wrapper around the ``fast-borf`` package [2]_ which
    implements the bag-of-receptive-fields features of Spinnato [1]_, and
    builds a ridge regression classifier on the resulting word counts,
    following the classification head of [1]_.

    Since BORF natively handles irregular and unequal-length series,
    so does this classifier.

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
        How the stride of each configuration is chosen, see
        :class:`sktime.transformations.borf.BORF`.
    min_window_to_signal_std_ratio : float, default=0.0
        Windows whose standard deviation is below this fraction of the
        signal's standard deviation are treated as flat, i.e., z-score 0.
    time_channel : bool, default=False
        If True, the last column of the panel is interpreted as holding the
        timestamps of each series, see
        :class:`sktime.transformations.borf.BORF`.
    n_jobs : int, default=1
        Number of numba threads used by ``fast-borf``. -1 uses numba's
        default, all logical cores.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data.
        If None, a ``sklearn.linear_model.RidgeClassifierCV`` is used,
        the classification head of [1]_. Note that the default head does
        not support ``predict_proba``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.
    n_classes_ : int
        Number of classes.

    See Also
    --------
    sktime.transformations.borf.BORF

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
    >>> from sktime.classification.dictionary_based import BORFClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = BORFClassifier(
    ...     max_window_size=8, max_word_length=4
    ... )  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    BORFClassifier(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["smellslikeml"],
        "maintainers": ["smellslikeml"],
        "python_dependencies": ["fast-borf"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:predict_proba": False,
        # pd-multiindex natively supports unequal-length panels,
        # and is the container the BORF transformer converts from
        "X_inner_mtype": ["pd-multiindex", "numpy3D"],
        "classifier_type": "dictionary",
        # CI and test flags
        # -----------------
        "tests:vm": True,
        "tests:libs": ["sktime.transformations.borf"],
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
        estimator=None,
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
        self.estimator = estimator

        super().__init__()

    def _get_borf_params(self):
        """Collect hyper-parameters to forward to the BORF transformer."""
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

    def _fit(self, X, y):
        """Fit a pipeline on cases (X, y), where y is the target variable.

        Parameters
        ----------
        X : pd.DataFrame with MultiIndex (instance, time) index
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self : reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        self._transformer = BORF(**self._get_borf_params())
        self._estimator = _clone_estimator(
            RidgeClassifierCV() if self.estimator is None else self.estimator
        )

        X_t = self._transformer.fit_transform(X, y)
        self._Xt_columns = X_t.columns

        if X_t.shape[1] == 0:
            raise ValueError(
                "BORF has produced no features for this data, all "
                "configurations were too large for the series given. "
                "Consider decreasing min_window_size or max_window_size."
            )

        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : pd.DataFrame with MultiIndex (instance, time) index
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        X_t = self._transformer.transform(X)
        X_t = X_t.reindex(self._Xt_columns, axis=1, fill_value=0)
        return self._estimator.predict(X_t)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

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
