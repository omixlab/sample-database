"""
Tests for :mod:`sample.esm_finetune` -- the full-network ESM-2 fine-tuning estimator.

Not mocked: the real `facebook/esm2_t6_8M_UR50D` checkpoint (already cached locally by every other
part of this project that uses the same model -- ~30MB, 8M parameters) is loaded and genuinely
fit/forward-passed. `tests/conftest.py`'s `no_network` fixture forces `HF_HUB_OFFLINE=1` for every
test in the suite so `from_pretrained`'s Hub update-check never attempts the real network call
that fixture's socket block would otherwise reject as an opaque low-level error -- a cache hit
alone is not enough to guarantee that on its own. Kept deliberately small (few rows, few epochs)
since this is a genuine forward/backward pass through a transformer, not a mocked stand-in --
these tests are the closest this project gets to fitting the real architecture end to end.

The default `sample` test environment (`make test`'s `ENV_DOCS`) carries a stray `torchvision`
0.26.0 -- a transitive dependency of `docling-ibm-models`/`vllm`, unrelated to anything here --
that is incompatible with this project's own pinned `torch==2.5.1+cu121` (chosen for this
machine's CUDA 12.2 driver ceiling; `vllm` itself appears to want a much newer torch, a separate,
pre-existing conflict out of this module's scope to resolve). `transformers`' `EsmModel` import
chain pulls in `torchvision` for an unrelated vision utility and crashes on that mismatch with a
`RuntimeError` from `torch.library`, not an ImportError -- confirmed this only happens under
`sample`; the dedicated `sample-ml` environment (what actually trains these models) has no
torchvision installed at all and imports cleanly. Skipped, not failed, under `sample` so `make
test` stays green without a false failure; run explicitly under `sample-ml` to exercise these for
real: `conda run -n sample-ml python -m pytest tests/test_esm_finetune.py`.
"""
import os
import pickle
from typing import Any, List

import numpy as np
import pytest

try:
    import transformers.models.esm.modeling_esm  # noqa: F401
except (ImportError, RuntimeError):
    # RuntimeError, not just ImportError -- torchvision's own top-level import crashes with
    # "operator torchvision::nms does not exist" (a torch.library registration mismatch), which
    # propagates raw through a plain import (unlike AutoModel's lazy-attribute path, which happens
    # to catch and rewrap it as ModuleNotFoundError -- pytest.importorskip only catches ImportError
    # and would let this one through as a collection error instead of a skip).
    pytest.skip(
        "EsmModel is not importable in this environment (see module docstring: a stray "
        "torchvision incompatible with this project's torch pin) -- run under sample-ml.",
        allow_module_level=True)

from sample.esm_finetune import ESM2FineTuneEstimator, _extract_sequences


@pytest.fixture(autouse=True)
def few_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caps every test's fit to 2 epochs -- these tests check shape/behavior contracts, not
    convergence quality (the pilot run itself, not this suite, is what validates real performance
    on real data)."""
    import sample.esm_finetune as mod
    monkeypatch.setattr(mod, "MAX_EPOCHS", 2)


def _toy_sequences(n: int, seed: int = 0) -> List[str]:
    """Short, valid-looking peptide sequences -- long enough to tokenize meaningfully, short
    enough that a real forward pass over `n` of them stays fast."""
    rng = np.random.default_rng(seed)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return ["".join(rng.choice(list(alphabet), size=12)) for _ in range(n)]


def _toy_classification_data(n: int = 24, seed: int = 0) -> Any:
    """Sequences whose label is composition-separable (positives systematically richer in K),
    unlike a purely random label -- a real fine-tune has something genuine to learn within a
    2-epoch test budget."""
    rng = np.random.default_rng(seed)
    sequences: List[str] = []
    labels: List[int] = []
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for i in range(n):
        label = i % 2
        residues = list(rng.choice(list(alphabet), size=12))
        if label == 1:
            residues[0] = residues[1] = "K"
        sequences.append("".join(residues))
        labels.append(label)
    X = np.array(sequences, dtype=object).reshape(-1, 1)
    y = np.array(labels, dtype=int)
    return X, y


class TestExtractSequences:
    def test_reads_the_sequence_column_from_a_dataframe(self) -> None:
        import pandas as pd
        df = pd.DataFrame({"sequence": ["AAAA", "CCCC"]})
        assert _extract_sequences(df) == ["AAAA", "CCCC"]

    def test_reads_the_first_column_from_a_2d_object_array(self) -> None:
        X = np.array([["AAAA"], ["CCCC"]], dtype=object)
        assert _extract_sequences(X) == ["AAAA", "CCCC"]


class TestESM2FineTuneEstimatorClassification:
    def test_constructed_like_make_model_with_is_regression_and_overrides(self) -> None:
        # Must accept exactly _make_model's calling convention (factory(is_regression=..., **
        # overrides)) since fit_and_validate/_y_randomization call model_factory this same way.
        est = ESM2FineTuneEstimator(is_regression=False, scale_pos_weight=3.0)
        assert est.is_regression is False
        assert est.scale_pos_weight == 3.0

    def test_fit_predict_proba_shape_and_range(self) -> None:
        X, y = _toy_classification_data()
        est = ESM2FineTuneEstimator(is_regression=False, scale_pos_weight=1.0)
        est.fit(X, y)
        probs = est.predict_proba(X)
        assert probs.shape == (len(y), 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_returns_hard_labels(self) -> None:
        X, y = _toy_classification_data()
        est = ESM2FineTuneEstimator(is_regression=False)
        est.fit(X, y)
        pred = est.predict(X)
        assert set(np.unique(pred)) <= {0, 1}
        assert len(pred) == len(y)

    def test_deterministic_under_eval_mode(self) -> None:
        # Two predict_proba calls on the same input without an intervening fit() must agree --
        # catches dropout not actually being switched to eval (model.eval()) at inference time.
        X, y = _toy_classification_data()
        est = ESM2FineTuneEstimator(is_regression=False)
        est.fit(X, y)
        first = est.predict_proba(X)
        second = est.predict_proba(X)
        assert np.allclose(first, second)

    def test_pos_weight_shifts_the_loss_but_not_the_call_signature(self) -> None:
        # A cheap contract check, not a convergence check (2 epochs is not enough to reliably
        # move the decision boundary): scale_pos_weight must not raise or change predict_proba's
        # shape, mirroring _make_model's own RandomForest-fallback contract for the same param.
        X, y = _toy_classification_data()
        est = ESM2FineTuneEstimator(is_regression=False, scale_pos_weight=5.0)
        est.fit(X, y)
        probs = est.predict_proba(X)
        assert probs.shape == (len(y), 2)

    def test_pickle_round_trip_gives_identical_predictions(self, tmp_path: Any) -> None:
        X, y = _toy_classification_data()
        est = ESM2FineTuneEstimator(is_regression=False)
        est.fit(X, y)
        before = est.predict_proba(X)

        path = os.path.join(str(tmp_path), "model.pkl")
        with open(path, "wb") as fh:
            pickle.dump(est, fh)
        with open(path, "rb") as fh:
            reloaded = pickle.load(fh)

        after = reloaded.predict_proba(X)
        assert np.allclose(before, after)

    def test_pickle_does_not_carry_the_live_module_or_tokenizer(self, tmp_path: Any) -> None:
        # The state dict pattern, not a raw nn.Module/tokenizer pickle -- __getstate__ must expose
        # only plain, picklable state.
        X, y = _toy_classification_data()
        est = ESM2FineTuneEstimator(is_regression=False)
        est.fit(X, y)
        state = est.__getstate__()
        assert "backbone_state" in state and "head_state" in state
        assert all(not hasattr(v, "forward") for v in state.values() if v is not None)

    def test_predict_before_fit_raises_rather_than_silently_scoring(self) -> None:
        est = ESM2FineTuneEstimator(is_regression=False)
        with pytest.raises(RuntimeError):
            est.predict_proba(np.array([["AAAA"]], dtype=object))

    def test_single_residue_and_long_sequence_do_not_crash(self) -> None:
        X, y = _toy_classification_data(n=6)
        est = ESM2FineTuneEstimator(is_regression=False)
        est.fit(X, y)
        edge_cases = np.array([["A"], ["A" * 200]], dtype=object)
        probs = est.predict_proba(edge_cases)
        assert probs.shape == (2, 2)
        assert np.all(np.isfinite(probs))


class TestESM2FineTuneEstimatorRegression:
    def test_fit_predict_returns_continuous_values(self) -> None:
        rng = np.random.default_rng(0)
        sequences = _toy_sequences(16)
        y = rng.normal(0, 1, 16).astype(np.float32)
        X = np.array(sequences, dtype=object).reshape(-1, 1)
        est = ESM2FineTuneEstimator(is_regression=True)
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (16,)
        assert np.all(np.isfinite(preds))
