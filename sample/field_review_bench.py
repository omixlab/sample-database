"""
Measures which local model reviews which field best, instead of assuming it.

`field_review`'s specs each name a model, and every one of them currently names the pipeline's
own model. That is a deliberate starting position, not a finding: this project has a rule about
exactly this mistake, earned twice --

    Verify with `make curation-model-check MODEL=<tag>` before changing this; two swaps made on
    reasoning alone cost ~680 documents of degraded output on 2026-10-13/14.

-- and "a smaller model should be fine for a yes/no question" is the same kind of reasoning that
produced those two swaps. So this module exists to turn the model column into a measurement.

**Where the labels come from.** Not from hand annotation, which would cap the eval at a few dozen
cases. Two populations the corpus already labels for free:

* **Known-bad**: records a deterministic rule rejects. `sequence.validate_extracted_sequence`
  rejecting `C10H22N6O4` is ground truth, not an opinion. The rule is suppressed for the duration
  of the call so the model has to reach the verdict itself.
* **Known-good**: records in the trustworthy tier -- a clean sequence with a typed, dose-bearing,
  targeted, non-suspect measurement. 1,026 peptides and 4,917 measurements qualified at the last
  review, and they include the canonical AMPs (LL-37, magainin 2, omiganan) with correct
  sequences.

A model that cannot reproduce the rules on the first population, or that rejects the second, is
not usable on the ambiguous residue in between -- which is the only population the field prompts
actually run on. That makes this a NECESSARY condition, not a sufficient one, and the summary
says so rather than implying the winner is validated for production.

**Cost.** Every case is one model call. Keep `limit` small while an extraction is running: the
calls queue against the same Ollama slots, and this pipeline has already measured what
oversubscription does to document budgets.
"""
import json
import logging
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sample import config
from sample.field_review import (FIELD_SPECS, VERDICT_OK, VERDICT_UNSURE, VERDICT_WRONG,
                                 Verdict, review_field)

logger = logging.getLogger("sample.field_review_bench")

#: Models worth measuring, newest measurement first in the comment.
#:
#: Only tags `ollama list` actually reports on this host; a missing tag is skipped with a warning
#: rather than pulled, because a pull during a run competes for the same disk and VRAM the
#: extraction is using.
CANDIDATE_MODELS: Tuple[str, ...] = (
    "gemma4:latest",   # the pipeline model; measured 12/12 on the triage judgement
    "gemma3:4b",       # 3.3GB; measured BAD at extraction (returns reagents as peptides) but
                       # never measured at classification, which is a different task
    "qwen3:4b",        # 2.5GB; unmeasured here
    "llama3.2:3b",     # 2.0GB; unmeasured here
    "qwen3:14b",       # 9.3GB; does NOT fit beside the KV cache on this host (13%/87% CPU/GPU)
    "deepseek-v2:16b", # 8.9GB; unmeasured, same fit concern
)


@dataclass
class Case:
    """
    One labelled review case.

    :param field: Field under review.
    :param record: Peptide record.
    :param activity: Owning activity for activity-scoped fields.
    :param text: Document text.
    :param expected: The verdict a correct reviewer returns.
    :param label_source: How the label was obtained, for the report.
    """
    field: str
    record: Dict[str, Any]
    activity: Optional[Dict[str, Any]]
    text: str
    expected: str
    label_source: str


def _document_text(path: str) -> str:
    """:returns: The source text for an extraction JSON, or ``""`` when it is not on disk."""
    stem = os.path.basename(path)[:-5]
    for source in ("pubmed", "scopus"):
        candidate = os.path.join("data", "literature_data_text", source, f"{stem}.txt")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()
            except OSError:
                return ""
    return ""


def _is_trustworthy(record: Dict[str, Any], activity: Dict[str, Any]) -> bool:
    """
    :returns: Whether this record/activity pair is in the trustworthy tier.

    The same criteria the corpus reviews use, so "known-good" here means what it means there.
    """
    from sample.concentration import activity_unit_is_coherent
    from sample.sequence import validate_extracted_sequence
    sequence = (record.get("peptide_sequence") or "").strip()
    if not sequence or not validate_extracted_sequence(sequence, record.get("peptide_name") or "")[0]:
        return False
    unit = (activity.get("activity_unit") or "").strip()
    kind = (activity.get("activity_type") or "").strip()
    if not unit or "�" in unit or not activity_unit_is_coherent(kind, unit)[0]:
        return False
    if kind.lower() in ("", "unknown", "n/a", "na", "none", "nd"):
        return False
    target = ((activity.get("target_organism") or "") or
              (activity.get("target_cell_line") or "")).strip()
    if target.lower() in ("", "n/a", "na", "none", "nd", "unknown", "not specified", "-"):
        return False
    if activity.get("value_scale_suspect"):
        return False
    return activity.get("value_lower") is not None


def build_cases(field: str, limit: int = 40, base_dir: Optional[str] = None,
                seed: int = 0) -> List[Case]:
    """
    Labelled cases for one field, balanced between known-bad and known-good.

    :param field: A key of :data:`sample.field_review.FIELD_SPECS`.
    :param limit: Maximum cases, split as evenly as the corpus allows.
    :param base_dir: Extraction output; ``data/literature_data_json`` by default.
    :param seed: Shuffle seed, so a re-measurement sees the same cases.
    :returns: Cases with `expected` set.

    Documents are located through `config.iter_extraction_files`, never a raw walk: the output
    tree accumulates manual backups beside the live sources, and a walk over it once shipped a
    database built from superseded extractions.
    """
    spec = FIELD_SPECS[field]
    root = base_dir or str(config.DATA_DIR / "literature_data_json")
    paths = config.iter_extraction_files(root)
    random.Random(seed).shuffle(paths)

    bad: List[Case] = []
    good: List[Case] = []
    half = max(1, limit // 2)

    for path in paths:
        if len(bad) >= half and len(good) >= half:
            break
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        text = ""
        for record in payload.get("peptides") or []:
            if not isinstance(record, dict):
                continue
            activities = [a for a in (record.get("biological_activities") or [])
                          if isinstance(a, dict)]
            targets: List[Optional[Dict[str, Any]]] = (
                activities if spec.scope == "activity" else [None])
            for activity in targets:
                source = activity if spec.scope == "activity" else record
                value = (source or {}).get(field)
                # An empty field is a degenerate case: there is nothing for a reviewer to judge,
                # the answer carries no information about the model, and `deterministic_verdict`
                # settles it in production anyway. Without this filter the known-bad population
                # for `peptide_sequence` is dominated by `rule:empty` -- 26,716 such records were
                # measured on this corpus -- and the measurement says nothing.
                if not str(value or "").strip():
                    continue
                from sample.field_review import deterministic_verdict
                ruled = deterministic_verdict(field, value, record, activity)
                if ruled is not None and ruled.verdict == VERDICT_WRONG and len(bad) < half:
                    text = text or _document_text(path)
                    bad.append(Case(field, record, activity, text, VERDICT_WRONG,
                                    f"rule:{ruled.failure_mode}"))
                elif (ruled is None and spec.scope == "activity" and activity is not None
                      and _is_trustworthy(record, activity) and len(good) < half):
                    text = text or _document_text(path)
                    good.append(Case(field, record, activity, text, VERDICT_OK,
                                     "trustworthy_tier"))
                elif (ruled is None and spec.scope == "peptide" and activities
                      and _is_trustworthy(record, activities[0]) and len(good) < half):
                    text = text or _document_text(path)
                    good.append(Case(field, record, None, text, VERDICT_OK,
                                     "trustworthy_tier"))
    cases = bad[:half] + good[:half]
    logger.info(f"bench {field}: {len(bad[:half])} known-bad + {len(good[:half])} known-good "
                f"= {len(cases)} case(s)")
    return cases


def _suppress_rules(field: str):
    """
    Context manager neutralising :func:`deterministic_verdict` for one field.

    Without this the bench measures the rules, not the model: `review_field` consults the rule
    first by design, so every known-bad case would be answered before the model saw it.
    """
    import contextlib

    from sample import field_review

    @contextlib.contextmanager
    def _ctx():
        original = field_review.deterministic_verdict

        def passthrough(f, value, record, activity=None):
            if f == field:
                return None
            return original(f, value, record, activity)

        field_review.deterministic_verdict = passthrough
        try:
            yield
        finally:
            field_review.deterministic_verdict = original

    return _ctx()


def measure_field(field: str, models: Sequence[str] = (), limit: int = 20,
                  timeout: Optional[float] = 120.0, base_dir: Optional[str] = None,
                  variants: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    """
    Scores each model on one field's labelled cases.

    :param field: Field to measure.
    :param models: Tags to try; :data:`CANDIDATE_MODELS` filtered to what is installed when empty.
    :param limit: Cases per model.
    :param timeout: Seconds per call.
    :param base_dir: Extraction output to draw cases from.
    :param variants: Prompt variants to compare. Each (model, variant) pair is scored separately
        and keyed ``"model@variant"`` when more than one variant is asked for, so a prompt change
        is measured on the same cases rather than argued about.
    :returns: ``{model: {caught, false_alarms, unsure, decided, median_seconds, n}}``.

    **`caught` and `false_alarms` are the numbers that matter**, and they trade off: a model
    answering "wrong" to everything scores perfect recall and is useless. For a review layer the
    safety property is `false_alarms` -- a layer that contradicts good data costs more than one
    that misses bad data, because the bad data is also caught by the rules downstream.

    `decided` is the share answered either way rather than `unsure`, and it is a COST, not an
    accuracy: an `unsure` is a case the layer declined, which is the behaviour the prompt asks
    for when the evidence does not settle the question. An earlier version of this report called
    that figure "accuracy" and ranked on it, which penalised exactly the conservatism the prompts
    request -- gemma4:latest measured 67% caught with 0% false alarms and was reported as "33%
    accurate" because it declined the known-good cases rather than affirming them.
    """
    cases = build_cases(field, limit=limit, base_dir=base_dir)
    if not cases:
        logger.warning(f"bench {field}: no cases found")
        return {}
    installed = available_models()
    chosen = [m for m in (models or CANDIDATE_MODELS) if m in installed]
    for missing in [m for m in (models or CANDIDATE_MODELS) if m not in installed]:
        logger.warning(f"bench: {missing} is not installed; skipped (not pulled on purpose)")

    from sample.field_review import PROMPT_VARIANT
    wanted_variants = tuple(variants) if variants else (PROMPT_VARIANT,)

    results: Dict[str, Dict[str, Any]] = {}
    for model in chosen:
      for variant in wanted_variants:
        label = model if len(wanted_variants) == 1 else f"{model}@{variant}"
        hits = bad_caught = bad_total = false_alarms = good_total = unsure = 0
        seconds: List[float] = []
        with _suppress_rules(field):
            for case in cases:
                started = time.time()
                verdict = review_field(field, case.record, case.activity, case.text,
                                       model=model, timeout=timeout, variant=variant)
                seconds.append(time.time() - started)
                if verdict.verdict == VERDICT_UNSURE:
                    unsure += 1
                if case.expected == VERDICT_WRONG:
                    bad_total += 1
                    if verdict.verdict == VERDICT_WRONG:
                        bad_caught += 1
                        hits += 1
                else:
                    good_total += 1
                    if verdict.verdict == VERDICT_WRONG:
                        false_alarms += 1
                    elif verdict.verdict == VERDICT_OK:
                        hits += 1
        results[label] = {
            "n": len(cases),
            "caught": bad_caught / bad_total if bad_total else None,
            "false_alarms": false_alarms / good_total if good_total else None,
            "unsure": unsure / len(cases) if cases else 0.0,
            "decided": 1.0 - (unsure / len(cases)) if cases else 0.0,
            "median_seconds": statistics.median(seconds) if seconds else None,
        }
        r = results[label]
        logger.info(f"bench {field} {label}: caught={r['caught']} "
                    f"false_alarms={r['false_alarms']} decided={r['decided']:.0%} "
                    f"median={r['median_seconds']:.1f}s")
    return results


def available_models() -> List[str]:
    """
    :returns: Model tags the local Ollama server reports, or ``[]`` when it cannot be reached.

    Read from the server rather than assumed, so a spec naming an uninstalled model is skipped
    with a warning instead of failing mid-measurement.
    """
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as response:
            payload = json.load(response)
    except Exception as e:
        logger.warning(f"bench: cannot list models: {type(e).__name__}: {e}")
        return []
    return [m.get("name", "") for m in payload.get("models") or []]


def format_results(field: str, results: Dict[str, Dict[str, Any]]) -> str:
    """
    :param field: Field measured.
    :param results: Output of :func:`measure_field`.
    :returns: A table, ranked by accuracy.
    """
    if not results:
        return f"{field}: no results"
    lines = [f"{field}  (n={next(iter(results.values()))['n']})",
             f"  {'model':17s}{'caught':>8s}{'false':>7s}{'decided':>9s}{'median':>8s}"]
    def pct(v): return "-" if v is None else f"{v:.0%}"
    # Ranked by recall on known-bad, then by fewest false alarms. Not by the decided share: an
    # `unsure` is a declined case, which is what the prompt asks for on evidence that does not
    # settle the question, and ranking on it rewards guessing.
    ordered = sorted(results.items(),
                     key=lambda kv: (-(kv[1]["caught"] or 0), kv[1]["false_alarms"] or 0))
    for model, r in ordered:
        lines.append(f"  {model:17s}{pct(r['caught']):>8s}{pct(r['false_alarms']):>7s}"
                     f"{r['decided']:>8.0%}{r['median_seconds'] or 0:>7.1f}s")
    lines.append("  caught  = known-bad correctly called wrong (recall; higher is better)")
    lines.append("  false   = known-good wrongly called wrong (the safety property; 0 is the bar)")
    lines.append("  decided = answered either way rather than 'unsure' -- a COST, not accuracy;")
    lines.append("            declining on unsettled evidence is what the prompt asks for.")
    lines.append("  A NECESSARY condition only: agreeing with the rules does not establish "
                 "accuracy on the ambiguous cases the prompts actually run on.")
    return "\n".join(lines)
