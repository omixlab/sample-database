"""
Benchmark a trained sAMPle model against an external reference tool on the *same* held-out peptides.

A held-out ROC-AUC only means something relative to what else could have produced it. When an
established tool already answers the same question -- HemoPI2 for hemolytic activity, say -- the
honest comparison is not two numbers from two papers measured on two different test sets, but both
tools scored on the identical peptides, with the identical metric code. That is what this module
does: it takes a model's own holdout split (the `test_entry_ids` that steps 45-46 record beside every
kept model), runs the reference tool over those exact sequences, and reports the two metric sets
side by side plus their difference.

The comparison is deliberately conservative about what it claims:

* Both sides are scored on the model's holdout, which the *reference* tool has never been tuned on
  but the sAMPle model has (its holdout was withheld from training, but the feature pipeline and the
  task definition are this project's own). A win here is evidence the sAMPle model is competitive on
  this project's data, not a claim of universal superiority -- and the report says so.
* The reference tool is run through the same isolated-environment `ExternalPredictor` framework the
  annotator uses, so "the tool is not installed" is reported as *unavailable*, never silently
  substituted with a neutral guess that would flatter or damage either side.

Only classification tasks with a registered reference tool can be benchmarked; a task no external
tool covers has nothing to compare against and is simply not in :data:`REFERENCE_TOOLS`.
"""
import logging
import os
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger("sample.benchmark")

#: Benchmarking against a model's full holdout (potentially tens of thousands of peptides) is
#: impractical for the slowest wrapped reference tool: AMPlify pays a full subprocess spin-up
#: (TensorFlow/Keras import, five model loads) per call, and even chunked (see
#: REFERENCE_CHUNK_SIZE) that dominates runtime at real holdout sizes. Capped instead to a
#: reproducible random sample -- large enough for a stable ROC-AUC estimate, small enough that
#: even AMPlify's per-chunk cost stays bounded to a practical runtime. holdout_fasta samples the
#: SAME entry ids (same seed, same source list, same algorithm, see _sample_entry_ids) so a
#: peptide handed to an unwrapped tool by hand is one of the SAME peptides this comparison
#: actually used, not a differently-sized or differently-composed set.
BENCHMARK_SAMPLE_SIZE = 1000
BENCHMARK_SAMPLE_SEED = 0

#: Reference tool calls are chunked rather than either one whole-holdout batch (unreliable at
#: scale -- confirmed on a real 12,712-peptide holdout: AMPlify's own script crashes when even a
#: single submitted sequence falls outside its accepted input footprint, see
#: sample.peptide_predictors._is_amplify_valid) or one sequence per call (robust, but each call
#: pays a full subprocess/model-load cost that dominates runtime long before the actual inference
#: does). A chunk that comes back with nothing usable falls back to scoring itself one sequence at
#: a time, isolating whichever one(s) are the actual problem without paying that cost up front for
#: every chunk.
REFERENCE_CHUNK_SIZE = 10


def _sample_entry_ids(entry_ids: Sequence[str], size: int = BENCHMARK_SAMPLE_SIZE,
                      seed: int = BENCHMARK_SAMPLE_SEED) -> List[str]:
    """
    A reproducible random sample of ``entry_ids``, capped at ``size``.

    :param entry_ids: The full candidate list to sample from (a model's ``test_entry_ids``).
    :param size: Maximum sample size; returns every id, unchanged, when there are already this
        many or fewer -- a small task's holdout is never padded or upsampled.
    :param seed: Fixed so the SAME peptides are picked every time this runs for the same model --
        :func:`benchmark_holdout` and :func:`holdout_fasta` both need to agree on which subset "the
        benchmark" means for a given model without one having to call the other.
    :returns: The sampled ids, in their ORIGINAL relative order (not shuffled) -- order carries no
        meaning for a comparison of unordered metrics, and a stable order keeps the output (and a
        diff against a previous run) easier to read.
    """
    import numpy as np

    ids = list(entry_ids)
    if len(ids) <= size:
        return ids
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(ids), size=size, replace=False))
    return [ids[i] for i in chosen]


class ReferenceTool(NamedTuple):
    """
    How to run one external tool as a benchmark reference for a task.

    :param key: The task keyword this tool is a reference for, matched against a model's namespace
        (e.g. ``"hemolytic"`` matches ``.../hemolytic/...``).
    :param predictor: The :class:`sample.peptide_predictors.ExternalPredictor` ``name`` to run.
    :param score_key: Column in the predictor's per-sequence output holding a continuous score
        (higher = more likely positive), used for ROC-AUC / average precision.
    :param label_key: Column holding the tool's own 0/1 call, used for threshold metrics when the
        continuous score is absent. May be ``None``.
    :param display_name: Human-readable tool name for the report.
    :param best_effort: When true, the tool is looked up in :data:`BENCHMARK_ONLY_PREDICTORS`
        rather than the live pipeline's :data:`sample.peptide_predictors.EXTERNAL_PREDICTORS`, and
        a whole-batch failure triggers a slower per-sequence retry rather than losing the tool's
        entire side of the comparison. For a tool retired from the live pipeline for reliability
        reasons (AMPlify -- see :data:`BENCHMARK_ONLY_PREDICTORS`) but still worth comparing
        against on a benchmark-sized holdout, where that exposure is small.
    """
    key: str
    predictor: str
    score_key: str
    label_key: Optional[str]
    display_name: str
    best_effort: bool = False


#: Registry of tasks a reference tool exists for. Keyed by the namespace keyword a benchmarkable
#: model carries. A key may have more than one tool (antibacterial has both Macrel and AMPlify,
#: hemolytic has both HemoPI2 and ToxinPred2); every match is compared. Extend this as tools are
#: wired in, one entry per (task, tool) once that tool is confirmed to run.
REFERENCE_TOOLS: Tuple[ReferenceTool, ...] = (
    ReferenceTool(key="hemolytic", predictor="hemopi2", score_key="hemopi2_score",
                  label_key="hemopi2_is_hemolytic", display_name="HemoPI 2.0"),
    ReferenceTool(key="antibacterial", predictor="macrel", score_key="macrel_amp_probability",
                  label_key="macrel_is_amp", display_name="Macrel"),
    # No "anticancer" entry, by instruction: this project's benchmarking is scoped to antibacterial
    # peptide prediction tools only, antibacterial kept as the one case study while the FLAML-
    # searched training path is proved out. An anticancer classifier can still be trained (steps
    # 42/44 do not know or care about this file), but reference_tools_for() returns [] for one and
    # benchmark_all()'s model discovery -- which only benchmarks a model with a registered tool --
    # skips it, so it never reaches the comparison report. Re-add a ReferenceTool(key="anticancer",
    # predictor="anticp", ...) entry to bring AntiCP 2.0 back into scope later.
    # Best-effort only: AMPlify was retired from the live pipeline (sample.peptide_predictors'
    # EXTERNAL_PREDICTORS) after a real, reproducible upstream Keras bug -- a malformed batch
    # crashes AMPlify.py's own ensemble() and the whole batch's scores are lost, measured at a
    # 7-12%+ failure rate across production-sized (tens of thousands of peptides) runs. A
    # benchmark holdout is two to three orders of magnitude smaller, so the exposure is small
    # enough to be worth it for the comparison, as long as a crash on some peptides only drops
    # those peptides from AMPlify's side rather than aborting the whole benchmark -- see
    # reference_scores' best_effort handling.
    ReferenceTool(key="antibacterial", predictor="amplify", score_key="amplify_score",
                  label_key="amplify_is_amp", display_name="AMPlify", best_effort=True),
    # Best-effort only, and a deliberate SCOPE CAVEAT: ToxinPred2 predicts general peptide/protein
    # toxicity, not hemolysis specifically -- there is no hemolysis-specific tool besides HemoPI2
    # itself, so this is the closest actively-maintained, easily-installable adjacent comparator,
    # not a second hemolysis-specific reference. Read its "reference" metrics here as "how does the
    # sAMPle hemolytic model compare to a general toxicity classifier on these peptides", not as a
    # second opinion on hemolysis itself. best_effort=True because, like AMPlify, it lives in
    # _benchmark_only_predictors() rather than the live pipeline's EXTERNAL_PREDICTORS -- it is NOT
    # hemolysis-specific enough to enrich every peptide with in the live annotator, only to compare
    # against on a benchmark holdout. See ToxinPred2Predictor's docstring for why it runs Model 1
    # rather than the paper's hybrid Model 2 (a confirmed, reproducible upstream crash in the
    # latter).
    ReferenceTool(key="hemolytic", predictor="toxinpred2", score_key="toxinpred2_score",
                  label_key="toxinpred2_is_toxic", display_name="ToxinPred 2.0", best_effort=True),
    # Best-effort only, same class of reason as AMPlify above: AMPidentifierPredictor lives in
    # _benchmark_only_predictors() rather than EXTERNAL_PREDICTORS because it was built purely as a
    # comparator for this benchmark, never as part of the live annotator's design -- there was no
    # retirement here, just no reason to add a fourth per-peptide AMP predictor to every enrichment
    # run when three (macrel, AMPlify, and the live pipeline's own set) already cover the ground.
    # best_effort=True is required regardless of crash rate: _find_predictor only consults
    # _benchmark_only_predictors() for tools marked best_effort, so "ampidentifier" would never be
    # found without it.
    ReferenceTool(key="antibacterial", predictor="ampidentifier", score_key="ampidentifier_score",
                  label_key="ampidentifier_is_amp", display_name="AMPidentifier", best_effort=True),
    # Best-effort only, same class of reason as AMPidentifier above: AmPEPpyPredictor lives in
    # _benchmark_only_predictors() rather than EXTERNAL_PREDICTORS -- a fifth antibacterial
    # comparator built purely for this benchmark, not on PyPI/bioconda (see
    # environments/sample-ampeppy.yml), never part of the live annotator's design. best_effort=True
    # is required regardless of crash rate: _find_predictor only consults
    # _benchmark_only_predictors() for tools marked best_effort, so "ampeppy" would never be found
    # without it.
    ReferenceTool(key="antibacterial", predictor="ampeppy", score_key="ampeppy_amp_probability",
                  label_key="ampeppy_is_amp", display_name="amPEPpy", best_effort=True),
)

#: The conventional probability threshold for turning a score into a 0/1 call. Reported alongside
#: the metrics, not hidden, so the choice is visible and changeable.
DEFAULT_THRESHOLD: float = 0.5

def _benchmark_only_predictors() -> Tuple[Any, ...]:
    """
    Predictors available for a ``best_effort`` :class:`ReferenceTool` but deliberately absent from
    :data:`sample.peptide_predictors.EXTERNAL_PREDICTORS` -- the list the live pipeline's own
    annotator enriches every peptide with.

    AMPlify lives here, not there: it is undeleted dead code in ``peptide_predictors.py`` (kept as
    a working reference, per that module's own comment) after being retired from live-pipeline use
    for its crash rate.

    ToxinPred2 lives here for a different reason: it predicts general toxicity, not hemolysis
    specifically, so it does not belong among the annotator's per-peptide predictors the way a
    hemolysis-specific tool would -- it is only useful as an adjacent comparator on a benchmark
    holdout (see its ``ReferenceTool`` entry's comment in :data:`REFERENCE_TOOLS`).

    AMPidentifier lives here for a third reason again: it was never part of the live annotator's
    design at all (unlike AMPlify, nothing retired it -- it simply exists to be a fourth
    antibacterial comparator for the benchmark), so there is no live-pipeline home for it to have
    been removed from.

    amPEPpy is the same class of reason as AMPidentifier: a fifth antibacterial comparator built
    purely for this benchmark, never part of the live annotator's design. It is also not
    pip/bioconda-installable at all (see ``environments/sample-ampeppy.yml``), which alone would
    make it a poor fit for the live pipeline's per-peptide enrichment step even if there were a
    reason to add it there.

    Importing any of these here does not reintroduce it into ``enrich_with_predictors()`` or any
    Makefile step -- only into a benchmark run's own, much smaller comparison. Imported lazily,
    matching this module's existing deferred import of ``EXTERNAL_PREDICTORS`` itself, rather than
    at module load.
    """
    from sample.peptide_predictors import (AMPidentifierPredictor, AmPEPpyPredictor, AMPlifyPredictor,
                                           ToxinPred2Predictor)
    return (AMPlifyPredictor(), ToxinPred2Predictor(), AMPidentifierPredictor(), AmPEPpyPredictor())


def reference_tools_for(model_id: str) -> List[ReferenceTool]:
    """
    Finds every reference tool that benchmarks a given model.

    :param model_id: A registry model id (its namespace path plus feature block).
    :returns: The matching :class:`ReferenceTool` entries, in :data:`REFERENCE_TOOLS` order;
        empty when no external tool covers the model's task. A task can have more than one (e.g.
        antibacterial has both Macrel and AMPlify).
    """
    segments = {s.lower() for s in model_id.split("/")}
    # Every registered tool is a binary classifier (AMP/not, anticancer/not, hemolytic/not) with
    # nothing to compare against a REGRESSION target (predicting a continuous MIC/IC50) -- matching
    # on the task keyword alone would pair, say, an antibacterial MIC regressor with Macrel's
    # yes/no call, which binary_metrics cannot even score. "classification" must be in the
    # namespace too.
    if "classification" not in segments:
        return []
    return [tool for tool in REFERENCE_TOOLS if tool.key.lower() in segments]


def reference_tool_for(model_id: str) -> Optional[ReferenceTool]:
    """
    :param model_id: A registry model id (its namespace path plus feature block).
    :returns: The FIRST matching :class:`ReferenceTool` (see :func:`reference_tools_for` for every
        match), or ``None``. Kept for callers that only ever expected a single reference tool.
    """
    tools = reference_tools_for(model_id)
    return tools[0] if tools else None


def binary_metrics(y_true: Sequence[float], y_score: Sequence[float],
                   threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Computes the standard binary-classification metric set.

    :param y_true: True 0/1 labels.
    :param y_score: Continuous scores (probability-like), higher = more positive.
    :param threshold: Score at or above which a prediction is called positive.
    :returns: ``roc_auc``, ``average_precision``, ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``mcc``, plus ``n``, ``n_positives`` and the ``threshold`` used. ROC-AUC / average precision
        are ``None`` when the holdout has only one class (undefined, not zero).
    """
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                  matthews_corrcoef, precision_score, recall_score, roc_auc_score)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)

    both_classes = len(np.unique(y_true_arr)) > 1
    return {
        "n": int(len(y_true_arr)),
        "n_positives": int(y_true_arr.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)) if both_classes else None,
        "average_precision": (float(average_precision_score(y_true_arr, y_score_arr))
                              if both_classes else None),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred)) if both_classes else 0.0,
    }


def _find_predictor(name: str, best_effort: bool) -> Any:
    """
    :param name: An :class:`~sample.peptide_predictors.ExternalPredictor` ``name``.
    :param best_effort: Look in :func:`_benchmark_only_predictors` first (falling back to the live
        pipeline's list), rather than only the live pipeline's list.
    :returns: The predictor instance, or ``None`` if no source has it registered.
    """
    from sample.peptide_predictors import EXTERNAL_PREDICTORS

    if best_effort:
        found = next((p for p in _benchmark_only_predictors() if p.name == name), None)
        if found is not None:
            return found
    return next((p for p in EXTERNAL_PREDICTORS if p.name == name), None)


def _rows_to_scores(rows: Sequence[Dict[str, Any]], tool: ReferenceTool
                    ) -> Tuple[List[float], List[float]]:
    """
    :param rows: One dict per sequence, as returned by an
        :class:`~sample.peptide_predictors.ExternalPredictor`.
    :param tool: Which columns to read.
    :returns: ``(scores, labels)``, ``nan`` at any position the tool did not score.
    """
    import numpy as np

    scores: List[float] = []
    labels: List[float] = []
    for row in rows:
        score = row.get(tool.score_key)
        scores.append(float(score) if score not in (None, "") else float("nan"))
        if tool.label_key is not None:
            label = row.get(tool.label_key)
            labels.append(float(label) if label not in (None, "") else float("nan"))
    if labels and all(np.isnan(s) for s in scores):
        # A tool that reports a call but no continuous score can still be compared on threshold
        # metrics: fall back to its 0/1 label as the "score" so ROC-AUC degrades gracefully rather
        # than erroring.
        scores = list(labels)
    return scores, labels


def reference_scores(tool: ReferenceTool, sequences: Sequence[str]) -> Dict[str, Any]:
    """
    Runs a reference tool over sequences in small chunks, returning its scores and availability.

    Scored :data:`REFERENCE_CHUNK_SIZE` sequences per call rather than either extreme: one whole
    call for the entire (benchmark-sample-sized) sequence list is unreliable at scale (confirmed
    on a real holdout: AMPlify's own script crashes rather than skipping when even one submitted
    sequence falls outside its accepted input footprint), while one sequence per call is robust
    but pays a full subprocess/model-load cost per sequence, dwarfing the actual inference and
    making a real-sized comparison impractically slow. A chunk is the unit of "did this call work
    at all": if it comes back with nothing usable, THAT chunk (and only that chunk) is retried one
    sequence at a time, isolating whichever entry(ies) are the actual problem -- so the cost of a
    bad sequence is one wasted chunk call plus up to :data:`REFERENCE_CHUNK_SIZE` single-sequence
    calls, not the whole comparison.

    :param tool: The reference tool to run.
    :param sequences: One-letter peptide sequences.
    :returns: ``{"available": bool, "scores": [...], "labels": [...]}``, aligned 1:1 with
        `sequences` -- ``nan`` at any position the tool did not score, rather than a shorter list,
        so a caller can filter by position without losing alignment. When the tool is not
        installed, ``available`` is ``False`` and the score/label lists are empty -- the caller
        must not treat that as a set of negative predictions.
    """
    import numpy as np

    predictor = _find_predictor(tool.predictor, tool.best_effort)
    if predictor is None or not predictor.available():
        logger.warning("Reference tool %s (%s) is not available; skipping its side of the "
                       "benchmark.", tool.display_name, tool.predictor)
        return {"available": False, "scores": [], "labels": []}

    sequences = list(sequences)
    scores: List[float] = [float("nan")] * len(sequences)
    labels: List[float] = [float("nan")] * len(sequences)
    n_single_retries = 0

    for start in range(0, len(sequences), REFERENCE_CHUNK_SIZE):
        chunk = sequences[start:start + REFERENCE_CHUNK_SIZE]
        try:
            rows = predictor.predict(chunk)
        except Exception:
            rows = []
        chunk_scores, chunk_labels = _rows_to_scores(rows, tool)

        if not chunk_scores or all(np.isnan(s) for s in chunk_scores):
            # The whole chunk came back with nothing usable -- fall back to one sequence at a
            # time, JUST for this chunk, to isolate the failure(s) rather than losing all of it.
            n_single_retries += len(chunk)
            chunk_scores, chunk_labels = [], []
            for seq in chunk:
                try:
                    one_row = predictor.predict([seq])
                except Exception:
                    one_row = []
                s, l = _rows_to_scores(one_row, tool)
                chunk_scores.append(s[0] if s else float("nan"))
                chunk_labels.append(l[0] if l else float("nan"))

        # chunk_labels is EMPTY (not one-per-row) whenever tool.label_key is None -- _rows_to_scores
        # only ever appends to it when there is a label column to read -- so this must not zip the
        # two together (zip silently truncates to the shorter one, dropping every score).
        for i, s in enumerate(chunk_scores):
            scores[start + i] = s
            labels[start + i] = chunk_labels[i] if i < len(chunk_labels) else float("nan")

    n_failed = sum(1 for s in scores if np.isnan(s))
    if n_failed:
        logger.warning("%s: %d/%d sequences could not be scored; excluded from this side of the "
                       "comparison.", tool.display_name, n_failed, len(scores))
    if n_single_retries:
        logger.info("%s: %d sequence(s) needed a per-sequence retry after their chunk of %d came "
                   "back empty.", tool.display_name, n_single_retries, REFERENCE_CHUNK_SIZE)

    return {"available": True, "scores": scores, "labels": labels}


def compare(y_true: Sequence[float], model_scores: Sequence[float],
            ref_scores: Optional[Sequence[float]], reference_available: bool,
            threshold: float = DEFAULT_THRESHOLD,
            model_threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Builds the side-by-side metric comparison on one shared holdout.

    :param y_true: True 0/1 labels for the holdout.
    :param model_scores: The sAMPle model's scores on the holdout.
    :param ref_scores: The reference tool's scores on the SAME holdout, or ``None``/empty when the
        tool was unavailable.
    :param reference_available: Whether the reference tool actually ran.
    :param threshold: Score threshold for the reference tool's 0/1 call -- the conventional 0.5,
        the standard way its own reported score is normally read.
    :param model_threshold: Score threshold for the sAMPle model's 0/1 call, when it differs from
        `threshold` -- its own F1-calibrated cutoff (:func:`sample.ml_models._best_f1_threshold`),
        computed at training time without ever touching this comparison holdout. `threshold` when
        omitted (a model trained before calibration existed), so every existing caller that passes
        only `threshold` keeps scoring both sides at the identical cutoff.
    :returns: ``{"n", "model": {...}, "reference": {...}|None, "delta": {...}|None,
        "model_ci95": {...}, "reference_ci95": {...}|None}``. ``delta`` is model-minus-reference
        per metric (positive = the sAMPle model is ahead), present only when the reference ran.
        The ``_ci95`` dicts are bootstrap 95% confidence intervals per metric (see
        :func:`sample.ml_models._bootstrap_ci`), so a difference between the two point estimates
        can be read as real or as noise on this holdout's size.
    """
    from sample.ml_models import _bootstrap_ci

    import numpy as np

    resolved_model_threshold = threshold if model_threshold is None else model_threshold

    y_arr = np.asarray(y_true, dtype=float)
    model_metrics = binary_metrics(y_true, model_scores, resolved_model_threshold)
    model_ci95 = _bootstrap_ci(y_arr, np.asarray(model_scores, dtype=float), is_regression=False,
                               threshold=resolved_model_threshold)
    if not reference_available or not ref_scores:
        return {"n": model_metrics["n"], "model": model_metrics, "model_ci95": model_ci95,
                "reference": None, "reference_ci95": None, "delta": None}

    ref_metrics = binary_metrics(y_true, ref_scores, threshold)
    ref_ci95 = _bootstrap_ci(y_arr, np.asarray(ref_scores, dtype=float), is_regression=False,
                             threshold=threshold)
    delta: Dict[str, Optional[float]] = {}
    for metric in ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1", "mcc"):
        a, b = model_metrics.get(metric), ref_metrics.get(metric)
        delta[metric] = (float(a) - float(b)) if (a is not None and b is not None) else None
    return {"n": model_metrics["n"], "model": model_metrics, "model_ci95": model_ci95,
            "reference": ref_metrics, "reference_ci95": ref_ci95, "delta": delta}


def _model_calibrated_threshold(model_id: str, model_dir: Optional[str]) -> Optional[float]:
    """
    Reads a model's own F1-calibrated decision threshold back out of its stored
    ``<block>.dataset.json`` (:func:`sample.ml_models._best_f1_threshold`, computed at training
    time from an internal split of the training data only -- never any comparison holdout).

    :param model_id: The sAMPle model to look up.
    :param model_dir: Model namespace root; the registry default when omitted.
    :returns: The calibrated threshold, or ``None`` when the model has no stored dataset record,
        is a regressor, or predates calibration -- callers fall back to :data:`DEFAULT_THRESHOLD`.
    """
    from sample.model_registry import get_model_dataset

    dataset = get_model_dataset(model_id, model_dir)
    if not dataset:
        return None
    threshold = dataset.get("threshold")
    return float(threshold) if threshold is not None else None


def _model_vs_chance(model_id: str, model_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Reads the model's own y-randomization null distribution back out of its stored
    ``<block>.dataset.json`` (written by :func:`sample.ml_models.train_all`), so a benchmark
    report can answer "does this beat random chance?" beside "does this beat tool X?" without
    recomputing y-randomization here -- it was already run once, on this exact holdout, at
    training time.

    :param model_id: The sAMPle model to look up.
    :param model_dir: Model namespace root; the registry default when omitted.
    :returns: ``{"holdout_roc_auc", "y_randomization_mean_roc_auc", "null_comparison"}``, or
        ``None`` when the model has no stored dataset record.
    """
    from sample.model_registry import get_model_dataset

    dataset = get_model_dataset(model_id, model_dir)
    if not dataset:
        return None
    return {
        "holdout_roc_auc": dataset.get("holdout", {}).get("roc_auc"),
        "y_randomization_mean_roc_auc": dataset.get("y_randomization", {}).get("mean_roc_auc"),
        "null_comparison": dataset.get("null_comparison", {}).get("roc_auc"),
    }


def benchmark_model(model_id: str, sequences: Sequence[str], y_true: Sequence[float],
                    model_dir: Optional[str] = None, tool: Optional[ReferenceTool] = None,
                    threshold: float = DEFAULT_THRESHOLD,
                    model_scores: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """
    Scores one model against every registered reference tool for its task, on the same labelled
    peptides, and compares them -- plus whether the model beats random chance (its own stored
    y-randomization null, see :func:`_model_vs_chance`).

    :param model_id: The sAMPle model to benchmark.
    :param sequences: The evaluation peptides (ideally the model's own holdout -- see
        :func:`benchmark_holdout`). Always needed for the reference tool's own scoring, even when
        `model_scores` makes the sAMPle side of it unnecessary.
    :param y_true: True 0/1 labels for those peptides.
    :param model_dir: Model namespace root; the registry default when omitted.
    :param tool: A single reference tool to compare against, overriding auto-discovery; when
        omitted, EVERY tool :func:`reference_tools_for` finds for this task is compared.
    :param threshold: Score threshold for the REFERENCE tool's 0/1 call -- the conventional 0.5.
        The sAMPle model's own side is scored at its own calibrated threshold when its stored
        dataset record has one (:func:`_model_calibrated_threshold`), falling back to this same
        `threshold` otherwise (a model trained before calibration existed).
    :param model_scores: The sAMPle model's own scores for `sequences`, already computed. When
        omitted, scored here via :func:`sample.inference.predict` (which re-derives every feature,
        including an ESM-2 embedding, from the raw sequence -- the right thing for a genuinely novel
        query, wasted work for a model's own already-embedded holdout rows). Pass this when the
        caller already has the scores from cheaper means -- see :func:`benchmark_holdout`, which
        computes them from the dataset's own cached feature columns via
        :func:`_score_model_from_dataset`.
    :returns: The comparison report. Top-level ``reference_tool``/``reference_available``/
        ``model``/``reference``/``delta``/``model_ci95``/``reference_ci95`` mirror the FIRST
        matching tool (backward-compatible with a single-tool task); ``comparisons`` is the full
        list, one entry per matching tool, each carrying its own ``reference_tool`` name plus the
        same keys; ``model_vs_chance`` is :func:`_model_vs_chance`'s result.
    :raises ValueError: When no reference tool covers this model's task.
    """
    tools = [tool] if tool is not None else reference_tools_for(model_id)
    if not tools:
        raise ValueError(f"No reference tool is registered for {model_id}; nothing to benchmark "
                         f"against. See REFERENCE_TOOLS.")

    if model_scores is None:
        from sample.inference import predict

        model_rows = predict(list(sequences), model_id, model_dir)
        # A classifier reports `probability`; fall back to the 0/1 `predicted_label` for the rare
        # model with no probability output, so the benchmark still runs (threshold metrics only).
        model_scores = [r.get("probability") if r.get("probability") is not None
                        else r.get("predicted_label", 0.0) for r in model_rows]

    model_threshold = _model_calibrated_threshold(model_id, model_dir)

    comparisons: List[Dict[str, Any]] = []
    for one_tool in tools:
        ref = reference_scores(one_tool, sequences)
        # Align by position, then drop wherever the reference has no usable score (a best-effort
        # tool's per-sequence failures, or any tool simply not covering a sequence) -- keeps
        # binary_metrics/compare themselves ignorant of NaN, isolating that handling to one place.
        if ref["available"] and ref["scores"]:
            import numpy as np
            keep = ~np.isnan(np.asarray(ref["scores"], dtype=float))
            y_kept = [v for v, k in zip(y_true, keep) if k]
            model_kept = [v for v, k in zip(model_scores, keep) if k]
            ref_kept = [v for v, k in zip(ref["scores"], keep) if k]
        else:
            y_kept, model_kept, ref_kept = list(y_true), list(model_scores), []
        one_report = compare(y_kept, model_kept, ref_kept, ref["available"], threshold,
                             model_threshold)
        one_report["reference_tool"] = one_tool.display_name
        one_report["reference_available"] = ref["available"]
        # Explicit in the report, not just implied by model["threshold"]/reference["threshold"]:
        # the two sides are not necessarily scored at the same cutoff (see `compare`'s
        # `model_threshold` parameter), and that is easy to miss reading the numbers alone.
        one_report["model_threshold"] = model_threshold if model_threshold is not None else threshold
        one_report["reference_threshold"] = threshold
        comparisons.append(one_report)

    report = dict(comparisons[0])
    report["model_id"] = model_id
    report["comparisons"] = comparisons
    report["model_vs_chance"] = _model_vs_chance(model_id, model_dir)
    return report


def _holdout_eval_set(model_id: str, dataset_csv: str, activities_csv: str,
                      model_dir: Optional[str] = None
                      ) -> Tuple[List[str], List[float], List[str], "pd.DataFrame"]:
    """
    Reconstructs a model's held-out peptides and their true labels.

    The holdout membership (`test_entry_ids`) is read from the model's ``dataset.json``; the labels
    are recomputed with the very same :func:`sample.ml_models.build_tasks` the training used, so the
    benchmark's ``y_true`` is exactly the label the model was evaluated against -- never a second,
    subtly different re-derivation.

    :param model_id: The model whose holdout to reconstruct.
    :param dataset_csv: The assembled dataset (``final_dataset.csv``) the model was trained on.
    :param activities_csv: The step-14 activity table that supplies the class labels.
    :param model_dir: Model namespace root; the registry default when omitted.
    :returns: ``(sequences, y_true, entry_ids, df)`` for a :func:`_sample_entry_ids`-capped subset
        of the holdout rows that have a defined label (so, up to :data:`BENCHMARK_SAMPLE_SIZE`, not
        necessarily the model's entire holdout -- see :func:`benchmark_holdout`). ``df`` is the full
        assembled dataset this was read from -- returned so a caller that also needs the model's
        own already-computed feature columns (:func:`_score_model_from_dataset`) does not have to
        read ``dataset_csv`` a second time.
    :raises ValueError: When the model has no stored dataset, or its task cannot be matched.
    """
    import numpy as np
    import pandas as pd

    from sample.ml_models import add_derived_feature_columns, build_tasks
    from sample.model_registry import get_model_dataset

    dataset = get_model_dataset(model_id, model_dir)
    if not dataset or not dataset.get("test_entry_ids"):
        raise ValueError(f"No stored holdout for {model_id}; run training (steps 45-46) first.")
    test_ids = set(_sample_entry_ids([str(e) for e in dataset["test_entry_ids"]],
                                     size=BENCHMARK_SAMPLE_SIZE))

    df = pd.read_csv(dataset_csv, low_memory=False)
    if "entry_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "entry_id"})
        df["entry_id"] = df["entry_id"].astype(str)
    # The dpc_*/ctd_* columns a FLAML-trained model's feature block needs are deliberately NOT
    # stored in dataset_csv (547 extra columns would add several hundred MB to an already-774MB
    # release file for numbers that are a pure function of `sequence`, which the file already
    # carries -- see add_derived_feature_columns' own docstring). Without this, every column-
    # availability check below fails for such a model, and _score_model_from_dataset's caller
    # falls back to sample.inference.predict -- which re-embeds and re-derives EVERY feature for
    # EVERY sampled peptide one at a time through the general-purpose scoring path, measured
    # directly at ~20 minutes for a 1,000-peptide sample, against ~20 SECONDS to derive these
    # columns once here for the whole dataset and let the fast cached-feature path (this
    # function's whole reason to exist, per its own docstring) actually run.
    df = add_derived_feature_columns(df)

    namespace = model_id.rsplit("/", 1)[0]
    spec = next((t for t in build_tasks(df, activities_csv) if t.namespace == namespace), None)
    if spec is None:
        raise ValueError(f"Could not match {model_id} to a training task; the dataset or activity "
                         f"table may differ from the one it was trained on.")

    entry_ids = df["entry_id"].astype(str).to_numpy()
    sequences_all = df["sequence"].astype(str).to_numpy()
    y_all = spec.y.to_numpy()

    sequences: List[str] = []
    y_true: List[float] = []
    kept_ids: List[str] = []
    for entry_id, sequence, label in zip(entry_ids, sequences_all, y_all):
        if entry_id in test_ids and not np.isnan(label):
            sequences.append(sequence)
            y_true.append(float(label))
            kept_ids.append(entry_id)
    return sequences, y_true, kept_ids, df


def _score_model_from_dataset(df: "pd.DataFrame", entry_ids: Sequence[str], model_id: str,
                              model_dir: Optional[str] = None) -> List[float]:
    """
    Scores a model on feature columns the dataset already carries, instead of routing through
    :func:`sample.inference.predict` -- which rebuilds every feature from the raw sequence,
    including a fresh ESM-2 forward pass, on the assumption that the sequence is a genuinely novel
    query the pipeline has never seen. That assumption is correct for the live ``/predict``
    endpoint and wrong here: a model's own holdout rows were already run through the enrichment
    step that wrote ``dataset_csv``'s ``esm2_dim_*`` columns, so re-embedding them again on every
    benchmark run reloads the transformer and re-does tens of thousands of forward passes for
    features that are already sitting in the file :func:`_holdout_eval_set` just read.

    Mirrors :func:`sample.inference.predict`'s scoring step exactly (same numeric coercion as
    :func:`sample.ml_models.train_task`'s own feature extraction, no scaler -- this project's
    estimator, XGBoost, does not need one) so the score this returns is identical to what
    ``predict()`` would have computed from the raw sequence, just without recomputing it.

    :param df: The assembled dataset (:func:`_holdout_eval_set`'s already-loaded frame).
    :param entry_ids: Which rows to score, in the exact order the caller's ``y_true`` is aligned to.
    :param model_id: The sAMPle model to score.
    :param model_dir: Model namespace root; the registry default when omitted.
    :returns: One score per entry id -- probability of the positive class for a classifier, the
        predicted value for a regressor.
    :raises ValueError: When the model's feature columns are not all present in ``df`` (an older
        dataset built before this model's feature block existed).
    """
    import numpy as np
    import pandas as pd

    from sample.model_registry import load_model

    estimator, columns, entry = load_model(model_id, model_dir)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{model_id} needs {len(missing)} feature column(s) not present in the "
                         f"dataset (first missing: {missing[:5]}); cannot score from cached "
                         f"features.")

    id_col = df["entry_id"].astype(str) if "entry_id" in df.columns else df.index.astype(str)
    indexed = df.set_axis(id_col, axis=0)
    rows = indexed.loc[[str(e) for e in entry_ids], list(columns)]
    # A sequence-input model (e.g. a fine-tuned transformer) takes the raw string; coercing it
    # through pd.to_numeric would silently zero it out and score on garbage with no exception at
    # all -- worse than the same bug in train_task, which at least only affects training.
    from sample.ml_models import _is_sequence_input
    if _is_sequence_input(columns):
        X = rows
    else:
        X = rows.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    is_regression = str(entry.get("learning_task", "")).lower() == "regression"
    if is_regression or not hasattr(estimator, "predict_proba"):
        return [float(v) for v in np.asarray(estimator.predict(X)).ravel()]
    return [float(p) for p in np.asarray(estimator.predict_proba(X))[:, 1]]


def benchmark_holdout(model_id: str,
                      dataset_csv: str = "data/sample_final_release/final_dataset.csv",
                      activities_csv: str = "data/sample_raw_release/activities.csv",
                      model_dir: Optional[str] = None,
                      threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Benchmarks a model against its reference tool on a sample of the model's own recorded holdout.

    This is the turnkey entry point: it reconstructs a :data:`BENCHMARK_SAMPLE_SIZE`-capped sample
    of the held-out peptides and labels (:func:`_holdout_eval_set`) and runs :func:`benchmark_model`
    on them, so the two tools are compared on identical data with identical metrics. Capped rather
    than run against the full holdout because the slowest wrapped reference tool (AMPlify) pays a
    real per-call cost that a holdout of tens of thousands of peptides multiplies past practicality
    -- see :data:`BENCHMARK_SAMPLE_SIZE`/:data:`REFERENCE_CHUNK_SIZE`; :func:`holdout_fasta` samples
    the identical subset, so a peptide handed to some other tool by hand is one this comparison
    actually used. The sAMPle side of that scoring is computed from the dataset's own cached feature
    columns (:func:`_score_model_from_dataset`) rather than re-embedding every sampled sequence
    through :func:`sample.inference.predict` -- those columns were already written by the enrichment
    step that produced ``dataset_csv``, so re-deriving them (an ESM-2 forward pass, for a model that
    used that block) on every benchmark run repeats work already done. Falls back to `predict()`-
    based scoring if a needed column is genuinely absent (e.g. a dataset that predates this model's
    feature block).

    :param model_id: The model to benchmark. Must have a registered reference tool.
    :param dataset_csv: The assembled dataset the model was trained on.
    :param activities_csv: The step-14 activity table.
    :param model_dir: Model namespace root; the registry default when omitted.
    :param threshold: Score threshold for the 0/1 call.
    :returns: The comparison report, additionally carrying ``n_holdout`` -- the number of peptides
        actually scored, i.e. the sample size, not necessarily the model's full holdout size.
    """
    sequences, y_true, entry_ids, df = _holdout_eval_set(
        model_id, dataset_csv, activities_csv, model_dir)
    try:
        model_scores = _score_model_from_dataset(df, entry_ids, model_id, model_dir)
    except ValueError as e:
        logger.warning(f"Could not score {model_id} from cached features ({e}); falling back to "
                       f"sample.inference.predict.")
        model_scores = None
    report = benchmark_model(model_id, sequences, y_true, model_dir=model_dir, threshold=threshold,
                             model_scores=model_scores)
    report["n_holdout"] = len(sequences)
    return report


def holdout_fasta(model_id: str,
                  dataset_csv: str = "data/sample_final_release/final_dataset.csv",
                  model_dir: Optional[str] = None) -> str:
    """
    Renders a model's benchmark sample as FASTA -- the exact peptides :func:`benchmark_holdout`
    scores it and its reference tool against (a :func:`_sample_entry_ids`-capped subset of the
    model's full holdout, not necessarily all of it -- see :data:`BENCHMARK_SAMPLE_SIZE`).

    Macrel, AntiCP2 and HemoPI2 are wrapped and benchmarked automatically; a tool with no
    :class:`~sample.peptide_predictors.ExternalPredictor` (a web upload form, a CLI this project
    has not packaged) can still be compared on the identical peptides by hand, given this file --
    the point being the *same* sampled set :func:`benchmark_holdout` actually used (same seed, same
    source list, same algorithm), not a fresh, differently-composed one that would make the two
    comparisons answer slightly different questions.

    Deliberately reads ``test_entry_ids`` straight from the model's stored ``dataset.json`` rather
    than going through :func:`_holdout_eval_set` (which re-derives a matching task from
    :func:`sample.ml_models.build_tasks` in order to recompute labels). A pure sequence export needs
    no label, and re-deriving one adds a failure mode this function has no reason to inherit: if a
    scoping rule in ``build_tasks`` changes after a model was trained, the model's on-disk namespace
    no longer matches anything the current code would build, and label re-derivation raises even
    though the peptides themselves -- and their ids -- never changed. ``test_entry_ids`` is the
    trained model's own permanent record of which peptides it was held out on; it does not need
    ``build_tasks`` to still agree.

    :param model_id: The model whose holdout to export.
    :param dataset_csv: The assembled dataset the model was trained on -- only its ``entry_id`` and
        ``sequence`` columns are read.
    :param model_dir: Model namespace root; the registry default when omitted.
    :returns: FASTA text, one record per sampled peptide, header = entry id.
    :raises ValueError: When the model has no stored holdout.
    """
    import pandas as pd

    from sample.model_registry import get_model_dataset

    dataset = get_model_dataset(model_id, model_dir)
    if not dataset or not dataset.get("test_entry_ids"):
        raise ValueError(f"No stored holdout for {model_id}; run training (steps 45-46) first.")
    sampled_ids = _sample_entry_ids([str(e) for e in dataset["test_entry_ids"]],
                                    size=BENCHMARK_SAMPLE_SIZE)

    df = pd.read_csv(dataset_csv, usecols=lambda c: c in ("entry_id", "sequence"), low_memory=False)
    if "entry_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "entry_id"})
    seq_by_id = dict(zip(df["entry_id"].astype(str), df["sequence"].astype(str)))

    lines: List[str] = []
    missing = 0
    for entry_id in sampled_ids:
        sequence = seq_by_id.get(entry_id)
        if sequence is None:
            missing += 1
            continue
        lines.append(f">{entry_id}\n{sequence}\n")
    if missing:
        logger.warning(f"{missing} of {len(sampled_ids)} sampled holdout entry ids for "
                       f"{model_id} were not found in {dataset_csv}; the FASTA omits them.")
    return "".join(lines)


def benchmark_all(model_dir: Optional[str] = None,
                  dataset_csv: str = "data/sample_final_release/final_dataset.csv",
                  activities_csv: str = "data/sample_raw_release/activities.csv",
                  report_path: Optional[str] = None,
                  threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Benchmarks every trained model that has a registered reference tool, on each one's own
    recorded holdout, and writes one combined report.

    The turnkey, whole-release entry point: :func:`benchmark_holdout` runs one model at a time by
    design (a diagnostic CLI command reaching for a specific model id); this walks
    :func:`sample.model_registry.list_models` and calls it for every model
    :func:`reference_tools_for` covers -- today, every antibacterial classifier (vs Macrel/AMPlify/
    AMPidentifier/amPEPpy) plus every hemolytic classifier (vs HemoPI2/ToxinPred2). Anticancer is
    deliberately NOT in that set -- see :data:`REFERENCE_TOOLS`'s own comment.

    :param model_dir: Model namespace root; the registry default when omitted.
    :param dataset_csv: The assembled dataset the models were trained on.
    :param activities_csv: The step-14 activity table.
    :param report_path: Where to write the combined report; not written when omitted.
    :param threshold: Score threshold for the 0/1 call.
    :returns: ``{model_id: comparison_report}``, one entry per benchmarked model. A model whose
        holdout could not be reconstructed or whose benchmark run raised is recorded under
        ``{"error": str(exception)}`` instead of skipped silently.
    """
    from sample.model_registry import list_models

    candidates = [m["model_id"] for m in list_models(model_dir) if reference_tools_for(m["model_id"])]
    logger.info(f"Benchmarking {len(candidates)} model(s) with a registered reference tool.")

    from tqdm import tqdm

    results: Dict[str, Any] = {}
    pbar = tqdm(candidates, desc="Benchmarking models", unit="model")
    for model_id in pbar:
        pbar.set_description(model_id)
        try:
            results[model_id] = benchmark_holdout(
                model_id, dataset_csv=dataset_csv, activities_csv=activities_csv,
                model_dir=model_dir, threshold=threshold)
        except Exception as e:
            logger.warning(f"Could not benchmark {model_id}: {e}")
            results[model_id] = {"error": str(e)}

    if report_path:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"Benchmark report for {len(results)} model(s) written to {report_path}.")
    return results
