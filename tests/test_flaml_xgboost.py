"""
Tests for sample.flaml_xgboost.

FlamlXGBoostEstimator itself needs `flaml`, only present in the `sample-ml` environment -- module-
level skip guard, same pattern as tests/test_esm_finetune.py, rather than mocking the search away
(a mocked FLAML fit would prove nothing about the real integration). FrozenXGBoostEstimator needs
only xgboost, present everywhere, so its own tests run unconditionally.
"""
from typing import Any, Dict

import numpy as np
import pytest

from sample.flaml_xgboost import FrozenXGBoostEstimator

try:
    import flaml  # noqa: F401

    from sample.flaml_xgboost import FlamlXGBoostEstimator
    _FLAML_AVAILABLE = True
except ImportError:
    _FLAML_AVAILABLE = False

requires_flaml = pytest.mark.skipif(not _FLAML_AVAILABLE, reason="flaml is not installed here")


def _separable_data(n: int = 80, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = np.column_stack([y * 4 + rng.normal(0, 1, n), rng.normal(0, 1, n)]).astype(np.float32)
    return X, y


class TestFrozenXGBoostEstimator:
    def _fitted_xgb_classifier(self) -> Any:
        from xgboost import XGBClassifier

        X, y = _separable_data()
        clf = XGBClassifier(n_estimators=17, max_depth=3, random_state=0)
        clf.fit(X, y)
        return clf, X, y

    def test_fit_predict_proba_round_trip(self) -> None:
        fitted, X, y = self._fitted_xgb_classifier()
        from sklearn.base import clone

        frozen = FrozenXGBoostEstimator(clone(fitted))
        frozen.fit(X, y)
        proba = frozen.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_reuses_the_given_configuration_not_a_default_one(self) -> None:
        # A clone of an estimator built with an unusual n_estimators must still carry it -- this
        # is the entire point of the class: refit the REAL model's own found configuration, not
        # some other default.
        from xgboost import XGBClassifier
        from sklearn.base import clone

        odd = XGBClassifier(n_estimators=3, max_depth=1, random_state=0)
        frozen = FrozenXGBoostEstimator(clone(odd))
        assert frozen._estimator.get_params()["n_estimators"] == 3
        assert frozen._estimator.get_params()["max_depth"] == 1

    def test_scale_pos_weight_is_applied_as_sample_weight_not_left_uncorrected(self) -> None:
        # Same mechanism as FlamlXGBoostEstimator's own imbalance correction -- verified here by
        # checking a heavily-imbalanced fit actually shifts predictions toward the minority class
        # relative to an uncorrected fit, rather than by inspecting internals.
        rng = np.random.default_rng(0)
        n = 200
        y = np.zeros(n, dtype=int)
        y[:10] = 1  # 10 positives, 190 negatives -- badly imbalanced
        X = np.column_stack([y * 4 + rng.normal(0, 0.3, n), rng.normal(0, 1, n)]).astype(np.float32)

        from xgboost import XGBClassifier
        from sklearn.base import clone

        base = XGBClassifier(n_estimators=50, max_depth=3, random_state=0)

        uncorrected = FrozenXGBoostEstimator(clone(base), scale_pos_weight=None)
        uncorrected.fit(X, y)
        corrected = FrozenXGBoostEstimator(clone(base), scale_pos_weight=19.0)  # 190/10
        corrected.fit(X, y)

        # The corrected model must predict a HIGHER mean positive-class probability on the
        # positive rows than the uncorrected one -- the whole point of the weighting.
        pos_rows = y == 1
        assert (corrected.predict_proba(X)[pos_rows, 1].mean()
               > uncorrected.predict_proba(X)[pos_rows, 1].mean())

    def test_regression_uses_no_sample_weight(self) -> None:
        from xgboost import XGBRegressor
        from sklearn.base import clone

        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (30, 2)).astype(np.float32)
        y = X[:, 0] * 2.0
        frozen = FrozenXGBoostEstimator(clone(XGBRegressor(n_estimators=10, random_state=0)),
                                        is_regression=True, scale_pos_weight=5.0)
        frozen.fit(X, y)  # must not raise despite scale_pos_weight being set
        preds = frozen.predict(X)
        assert preds.shape == (30,)


@requires_flaml
class TestYRandomizationFactory:
    def test_raises_before_fit(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            FlamlXGBoostEstimator().y_randomization_factory()

    def test_builds_a_frozen_estimator_bound_to_the_found_configuration(self) -> None:
        X, y = _separable_data(n=60)
        real = FlamlXGBoostEstimator(time_budget=5)
        real.fit(X, y)

        factory = real.y_randomization_factory()
        frozen = factory(is_regression=False, scale_pos_weight=1.0)
        assert isinstance(frozen, FrozenXGBoostEstimator)
        # Same hyperparameters as the real found model -- a clone, not a fresh/default estimator.
        assert (frozen._estimator.get_params()["n_estimators"]
               == real._estimator.get_params()["n_estimators"])

    def test_each_call_returns_an_independent_unfitted_clone(self) -> None:
        X, y = _separable_data(n=60)
        real = FlamlXGBoostEstimator(time_budget=5)
        real.fit(X, y)
        factory = real.y_randomization_factory()

        first = factory(is_regression=False)
        second = factory(is_regression=False)
        assert first._estimator is not second._estimator  # independent clones
        first.fit(X, y)
        # The second instance's underlying estimator must be untouched by fitting the first.
        with pytest.raises(Exception):
            second.predict_proba(X)  # unfitted -- xgboost raises on predict before fit


@requires_flaml
class TestFitAndValidateSkipsFlamlSearchForCalibrationCvAndYRandomization:
    """
    The actual point of this whole module: sample.ml_models.fit_and_validate must call
    model.y_randomization_factory() when the real model exposes it, rather than re-invoking the
    expensive model_factory (a fresh FLAML search) for threshold calibration, cross-validation, or
    y-randomization.

    `model_factory` MUST be passed as the class itself or a `functools.partial` around it, never a
    plain lambda/closure: fit_and_validate's cheap-refit detection checks
    `hasattr(getattr(model_factory, "func", model_factory), "y_randomization_factory")`, which
    reads a `functools.partial`'s wrapped class correctly but cannot see through an opaque
    closure. TestFallsBackToTheSlowPathForAnOpaqueFactory below pins that boundary down
    explicitly, since it is easy to defeat by accident (a test in this suite did, before this
    fix).
    """

    def test_calibration_cv_and_y_randomization_all_skip_the_search(
            self, monkeypatch: Any) -> None:
        from sample import ml_models as M

        search_calls: Dict[str, int] = {"count": 0}
        real_fit = FlamlXGBoostEstimator.fit

        def counting_fit(self: Any, X: Any, y: Any) -> Any:
            search_calls["count"] += 1
            return real_fit(self, X, y)

        monkeypatch.setattr(FlamlXGBoostEstimator, "fit", counting_fit)

        rng = np.random.default_rng(0)
        n = 120
        y = rng.integers(0, 2, n)
        X = np.column_stack([y * 4 + rng.normal(0, 1, n),
                             rng.normal(0, 1, n)]).astype(np.float32)
        groups = np.arange(n)
        entry_ids = np.array([f"P{i}" for i in range(n)])

        import functools

        record = M.fit_and_validate(
            X, y, groups, entry_ids,
            model_factory=functools.partial(FlamlXGBoostEstimator, time_budget=5),
            n_calibration_folds=1, n_y_randomizations=3, n_cv_folds=3)

        # Exactly ONE real FLAML search, total, for the whole call: fit_and_validate fits the
        # model once and reuses that found configuration -- via cheap FrozenXGBoostEstimator
        # clones -- for every calibration fold, every cross-validation fold, and every
        # y-randomization repeat. None of those three diagnostics may trigger
        # FlamlXGBoostEstimator.fit() again.
        assert search_calls["count"] == 1
        assert "cross_validation" in record and "mean_roc_auc" in record["cross_validation"]
        assert "y_randomization" in record and "mean_roc_auc" in record["y_randomization"]
        assert "threshold" in record


@requires_flaml
class TestFallsBackToTheSlowPathForAnOpaqueFactory:
    """
    A `model_factory` fit_and_validate cannot see the class of (a bare lambda/closure, as opposed
    to the class itself or a `functools.partial` around it) is correctly treated as NOT
    cheap-refit-capable -- the same as any architecture that genuinely has no found-configuration
    hook -- rather than crashing or silently miscounting. This is a real, easy-to-hit boundary:
    an earlier version of this test suite used exactly this lambda pattern to pin
    the-then-two-searches behavior, without realizing the lambda was itself the reason the
    cheap-refit path never engaged.
    """

    def test_a_lambda_wrapped_factory_does_not_engage_the_cheap_refit_path(
            self, monkeypatch: Any) -> None:
        from sample import ml_models as M

        search_calls: Dict[str, int] = {"count": 0}
        real_fit = FlamlXGBoostEstimator.fit

        def counting_fit(self: Any, X: Any, y: Any) -> Any:
            search_calls["count"] += 1
            return real_fit(self, X, y)

        monkeypatch.setattr(FlamlXGBoostEstimator, "fit", counting_fit)

        rng = np.random.default_rng(0)
        n = 120
        y = rng.integers(0, 2, n)
        X = np.column_stack([y * 4 + rng.normal(0, 1, n),
                             rng.normal(0, 1, n)]).astype(np.float32)
        groups = np.arange(n)
        entry_ids = np.array([f"P{i}" for i in range(n)])

        record = M.fit_and_validate(
            X, y, groups, entry_ids,
            model_factory=lambda **kw: FlamlXGBoostEstimator(time_budget=5, **kw),
            n_calibration_folds=1, n_y_randomizations=1, n_cv_folds=1)

        # The old, slower behavior: a real search for the single-split calibration fit, and a
        # SECOND real search for the final model -- because a lambda has no `y_randomization_
        # factory` attribute to detect, fit_and_validate cannot know it is safe to reuse either
        # search's found configuration for the other purpose.
        assert search_calls["count"] == 2
        assert "threshold" in record
