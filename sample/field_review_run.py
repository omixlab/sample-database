"""
Step 12: the final polish. Reviews the fields a rule cannot settle, with the model.

Runs after step 11, and the order is the whole economy of it. Step 11 applies every repair a rule
can make -- 1,334 units canonicalized, 931 range bounds filled, 16,927 incoherent activities
dropped, all without a model call. What reaches this step is the residue no rule can decide, and
measured over the refined output that residue is bounded:

    target absent on a typed measurement                7,959
    activity_type is a placeholder the paper could fill  1,458
    unit is mM on a potency metric (a lost micro sign)     686
    sequence is three-letter notation that passed all rules 460
    unit carries a U+FFFD no rule can name                  56
    ------------------------------------------------------------
    total                                                10,619   ~14.7h at 2 slots

Running it on every field of every activity would be 87,000 calls instead, which is why
:func:`select_targets` exists: a field is reviewed when a cheap signal says it is both ambiguous
AND consequential. The largest group is the one that matters most -- 45.6% of activities carry no
target, and that is the single biggest loss of usable data in the corpus.

**Corrections are recorded by default and applied only on request.** A model's proposed value is
evidence, not an instruction, and `--apply` additionally refuses any correction the field's own
deterministic check would reject -- so this step can never introduce a defect that step 11 would
have to clean up.
"""
import json
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from sample import config
from sample.field_review import (FIELD_SPECS, PROMPT_VARIANT, REVIEW_MODEL, VERDICT_WRONG,
                                 Verdict, deterministic_verdict, review_field,
                                 warn_if_review_model_differs_from_extraction)
from sample.polish import _write_atomically

logger = logging.getLogger("sample.field_review_run")

#: Bumped when the selection policy or the prompts change, so a later run can tell which rules
#: produced a stored verdict.
REVIEW_VERSION: int = 1

#: Metadata key holding this step's record of what it judged.
REVIEW_KEY: str = "field_review"

#: Values that mean "nobody filled this in".
_PLACEHOLDER = frozenset({"", "unknown", "n/a", "na", "none", "nd", "not specified",
                          "unspecified", "-"})


def _blank(value: Any) -> bool:
    return str(value or "").strip().lower() in _PLACEHOLDER


#: Which field a scale finding puts in doubt. A fault a rescale would fix belongs to the unit; a
#: stock solution or a misread number belongs to the value, because changing the unit on either
#: would produce a well-formed wrong answer and erase the evidence that anything was amiss.
_SCALE_FIELD_BY_FAULT = {
    "scale_prefix_lost": ("activity_unit", "scale_prefix_suspected"),
    "implausible_magnitude": ("activity_unit", "implausible_magnitude"),
    "not_a_potency_value": ("value_raw", "stock_solution_suspected"),
    "value_contradicts_context": ("value_raw", "contradicts_context_sentence"),
}


def _scale_target(record: Dict[str, Any],
                  activity: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    :param record: The peptide record, for the sibling measurements the diagnosis uses.
    :param activity: One activity.
    :returns: ``(field, reason)`` when :mod:`sample.scale` found something it could not prove,
        else ``None``. Confident findings are excluded: step 11 has already applied those, and
        re-reviewing a corrected record would spend a model call to confirm a rule.
    """
    from sample import scale as scale_module

    finding = scale_module.diagnose(record, activity)
    if finding.status != scale_module.AMBIGUOUS:
        return None
    return _SCALE_FIELD_BY_FAULT.get(finding.fault)


def select_targets(record: Dict[str, Any]) -> List[Tuple[str, Optional[Dict[str, Any]], str]]:
    """
    The fields of one record worth a model call, with why.

    :param record: A peptide record.
    :returns: ``[(field, activity_or_None, reason), ...]``.

    Selection, not review: every candidate here is a field where a rule returned no verdict AND
    a cheap signal says the answer is consequential. Reviewing everything else would be 87,000
    calls to confirm data no signal doubts.
    """
    out: List[Tuple[str, Optional[Dict[str, Any]], str]] = []

    sequence = (record.get("peptide_sequence") or "").strip()
    if sequence:
        settled = deterministic_verdict("peptide_sequence", sequence, record, None)
        if settled is not None and settled.failure_mode == "sequence_of_another_peptide":
            # Selected BECAUSE a rule settles it, which is the opposite of the branch below and
            # the reason this is its own case. The name catalogue says this record pairs one
            # peptide's name with another's sequence; there is no model call to make and nothing
            # to correct -- which of the two fields is wrong is not recoverable from the record --
            # so the point of selecting it is to get the finding recorded against the document.
            # Gating it behind the hyphen test, as the three-letter branch is, meant the rule
            # fired 865 times when measured directly and zero times in the actual run.
            out.append(("peptide_sequence", None, "name_identifies_another_peptide"))
        elif settled is None and "-" in sequence:
            # Three-letter notation passes the structural rules even when a residue triplet is
            # OCR-damaged (`CVS` for `CYS`, `CY'S`, `AL~`), so this is the one sequence population
            # the rules cannot screen. 20 such strings were found by hand on this corpus.
            out.append(("peptide_sequence", None, "three_letter_notation"))

    for activity in record.get("biological_activities") or []:
        if not isinstance(activity, dict):
            continue
        kind = activity.get("activity_type")
        unit = str(activity.get("activity_unit") or "")

        if activity.get("value_scale_suspect"):
            # mM on a potency metric. The conversion already applied is arithmetically right;
            # what is in doubt is the SOURCE unit, and only the document settles it.
            out.append(("activity_unit", activity, "value_scale_suspect"))
        elif "�" in unit and deterministic_verdict(
                "activity_unit", unit, record, activity) is None:
            out.append(("activity_unit", activity, "unnameable_mojibake"))
        else:
            # Last in the chain, so a measurement already queued for a unit review is not queued
            # twice for the same doubt about the same unit.
            scaled = _scale_target(record, activity)
            if scaled is not None:
                out.append((scaled[0], activity, scaled[1]))

        if _blank(kind):
            out.append(("activity_type", activity, "placeholder_type"))

        if (not _blank(kind) and _blank(activity.get("target_organism"))
                and _blank(activity.get("target_cell_line"))):
            # The biggest group, and the biggest prize: a typed, valued measurement with no
            # target is excluded from every usable tier purely for want of this field.
            out.append(("target_organism", activity, "absent_target"))
    return out


def _document_text(path: str) -> str:
    """:returns: Source text for an extraction JSON, or ``""`` when it is not on disk."""
    stem = os.path.basename(path)[:-5]
    for source in config.EXTRACTION_SOURCE_DIRS:
        candidate = os.path.join("data", "literature_data_text", source, f"{stem}.txt")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()
            except OSError:
                return ""
    return ""


def _acceptable_correction(field: str, proposed: str) -> str:
    """
    What a correction must look like before it may be written.

    :param field: Field being corrected.
    :param proposed: The model's replacement value.
    :returns: ``""`` when the proposal is acceptable, else a short slug saying why it is not.

    Separate from :func:`deterministic_verdict`, which judges values that are ALREADY in the
    corpus and must stay permissive about them -- 10,368 activities carry an off-vocabulary unit
    and flagging every one as wrong would change what step 12 even looks at. This governs only
    what may be written, where the bar is higher: a correction is a free choice, so it has no
    excuse to land outside the release vocabulary.

    Without this, the re-check through `deterministic_verdict` passes anything that is merely
    unrecognised rather than known-bad -- `mcg/ml`, `uM/L`, `ug/mL (final)` -- and step 12 would
    spend 5,240 calls partly undoing the 2,768 unit repairs step 11 had just made.
    """
    text = (proposed or "").strip()
    if not text:
        return "empty"
    if field == "activity_unit":
        from sample.preprocessing import canonicalize_unit
        from sample.scale import basis_of, canonical_unit, display_unit
        from sample.validation_rules import VALID_UNITS

        vocabulary = {unit.strip().lower() for unit in VALID_UNITS if unit}
        for candidate in (canonicalize_unit(text) or text, display_unit(canonical_unit(text))):
            if candidate and candidate.strip().lower() in vocabulary:
                return ""
        if not basis_of(canonical_unit(text)):
            return "not_a_concentration_unit"
        return "outside_the_release_vocabulary"
    if field == "value_raw":
        from sample.nlp import parse_activity_value_fast

        _operator, lower, _upper = parse_activity_value_fast(text)
        if lower is None:
            return "no_number_in_the_proposal"
    return ""


def _apply_correction(field: str, verdict: Verdict, record: Dict[str, Any],
                      activity: Optional[Dict[str, Any]]) -> bool:
    """
    Writes a proposed correction, when it survives the field's own rules.

    :param field: Field being corrected.
    :param verdict: The reviewer's answer.
    :param record: Peptide record.
    :param activity: Owning activity, for activity-scoped fields.
    :returns: Whether the value was changed.

    The deterministic re-check is the point. A model's `corrected` value is a proposal, and this
    step must not be able to introduce a defect that step 11 would then have to clean up -- so a
    correction that the field's own rule rejects is discarded and counted, not written.
    """
    if verdict.verdict != VERDICT_WRONG or not verdict.corrected.strip():
        return False
    target = activity if FIELD_SPECS[field].scope == "activity" else record
    if target is None:
        return False
    proposed = verdict.corrected.strip()
    unacceptable = _acceptable_correction(field, proposed)
    if unacceptable:
        logger.debug(f"field_review: refusing correction {field}={proposed!r} ({unacceptable})")
        return False
    ruled = deterministic_verdict(field, proposed, record, activity)
    if ruled is not None and ruled.verdict == VERDICT_WRONG:
        logger.debug(f"field_review: refusing correction {field}={proposed!r} "
                     f"({ruled.failure_mode})")
        return False
    if field == "activity_unit":
        # Store the spelling the rest of the corpus uses, not whatever the model typed. A model
        # that answers `µg/ml` is right about the unit and wrong about the spelling, and writing
        # it verbatim adds a 911th off-vocabulary variant.
        from sample.preprocessing import canonicalize_unit
        from sample.scale import canonical_unit, display_unit
        from sample.validation_rules import VALID_UNITS

        vocabulary = {unit.strip().lower() for unit in VALID_UNITS if unit}
        first = canonicalize_unit(proposed) or proposed
        proposed = (first if first.strip().lower() in vocabulary
                    else display_unit(canonical_unit(proposed)))
    target[field] = proposed
    return True


def review_extractions(base_dir: Optional[str] = None, dry_run: bool = False,
                       apply: bool = False, max_calls: int = 0, limit: int = 0,
                       model: Optional[str] = None, timeout: Optional[float] = 120.0,
                       fields: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """
    Reviews the ambiguous fields across the extraction corpus.

    :param base_dir: Extraction output; ``data/literature_data_json`` by default.
    :param dry_run: Select and report targets without calling the model or writing.
    :param apply: Write proposed corrections that survive the field's own rules.
    :param max_calls: Stop after this many model calls; 0 means no cap. The whole pass is ~10,619
        calls and ~14.7 hours, so a cap is how it is run in pieces.
    :param limit: Stop after this many documents; 0 means all.
    :param model: Override :data:`sample.field_review.REVIEW_MODEL`.
    :param timeout: Seconds per model call.
    :param fields: Restrict to these fields.
    :returns: A summary of targets, verdicts, corrections and refusals.

    Documents are located through `config.iter_extraction_files`, never a raw walk: the output
    tree accumulates manual backups beside the live sources, and a walk over it once shipped a
    database built from superseded extractions.
    """
    warn_if_review_model_differs_from_extraction()
    root = base_dir or str(config.DATA_DIR / "literature_data_json")
    paths = config.iter_extraction_files(root)
    if limit:
        paths = paths[:limit]

    summary: Dict[str, Any] = {
        "documents": 0, "documents_changed": 0, "unreadable": 0,
        "targets": 0, "calls": 0, "ok": 0, "wrong": 0, "unsure": 0,
        "corrections_applied": 0, "corrections_refused": 0,
        "skipped_changed_underneath": 0,
        "by_reason": {}, "by_field": {},
        "model": model or REVIEW_MODEL, "variant": PROMPT_VARIANT,
    }
    started = time.time()

    for path in paths:
        summary["documents"] += 1
        try:
            mtime = os.path.getmtime(path)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as e:
            summary["unreadable"] += 1
            logger.warning(f"field_review: cannot read {path}: {type(e).__name__}: {e}")
            continue
        if not isinstance(payload, dict):
            summary["unreadable"] += 1
            continue

        text = ""
        verdicts: List[Dict[str, Any]] = []
        changed = False
        for record in payload.get("peptides") or []:
            if not isinstance(record, dict):
                continue
            for field, activity, reason in select_targets(record):
                if fields and field not in fields:
                    continue
                summary["targets"] += 1
                summary["by_reason"][reason] = summary["by_reason"].get(reason, 0) + 1
                summary["by_field"][field] = summary["by_field"].get(field, 0) + 1
                if dry_run:
                    continue
                if max_calls and summary["calls"] >= max_calls:
                    continue
                text = text or _document_text(path)
                verdict = review_field(field, record, activity, text,
                                       model=model, timeout=timeout)
                summary["calls"] += 1
                summary[verdict.verdict] = summary.get(verdict.verdict, 0) + 1
                entry = {"field": field, "reason": reason, "verdict": verdict.verdict,
                         "why": verdict.reason, "failure_mode": verdict.failure_mode,
                         "peptide": record.get("peptide_name", "")}
                if verdict.corrected:
                    entry["corrected"] = verdict.corrected
                if apply:
                    if _apply_correction(field, verdict, record, activity):
                        summary["corrections_applied"] += 1
                        entry["applied"] = True
                        changed = True
                    elif verdict.verdict == VERDICT_WRONG and verdict.corrected.strip():
                        summary["corrections_refused"] += 1
                        entry["applied"] = False
                verdicts.append(entry)

        if dry_run or not verdicts:
            continue
        metadata = payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = payload["metadata"] = {}
        metadata[REVIEW_KEY] = {"version": REVIEW_VERSION, "model": summary["model"],
                                "variant": summary["variant"], "verdicts": verdicts}
        if not _write_atomically(path, payload, expect_mtime=mtime):
            summary["skipped_changed_underneath"] += 1
            continue
        if changed or verdicts:
            summary["documents_changed"] += 1

    elapsed = time.time() - started
    logger.info(
        f"field_review: {summary['documents']:,} document(s), {summary['targets']:,} target(s), "
        f"{summary['calls']:,} call(s) in {elapsed / 60:.1f} min "
        f"(model {summary['model']}, prompt {summary['variant']})")
    for reason, count in sorted(summary["by_reason"].items(), key=lambda kv: -kv[1]):
        logger.info(f"field_review:   {reason:34s} {count:>8,}")
    if summary["calls"]:
        logger.info(f"field_review:   verdicts: ok={summary['ok']:,} wrong={summary['wrong']:,} "
                    f"unsure={summary['unsure']:,}")
        logger.info(f"field_review:   corrections applied={summary['corrections_applied']:,} "
                    f"refused_by_rules={summary['corrections_refused']:,}")
    if dry_run:
        logger.info("field_review: dry run -- no model calls and nothing written")
    return summary
