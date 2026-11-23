"""
Re-applies the current activity-field rules to extraction records already written.

Step 9 writes records under whatever rules were in force at the time, and those rules keep
tightening. A document is only re-extracted when the model pass runs again, which on this corpus
is a multi-day job -- so without this stage a rule fixed today reaches the corpus at the speed of
re-extraction. Measured on the output on disk when this was written:

    6,173 of 53,107 activities (11.6%)  a concentration type in a mass/time/energy/count unit
    1,098 of 52,713 units      (2.1%)   U+FFFD where PDF extraction lost a character
    2,312 value_raw                     a range whose upper bound was parsed and discarded
       58 activity_type                 prose describing an assay instead of naming one

**Targeted repairs, not a re-run of `preprocessing.standardize_and_filter_activities`.** That
function is not idempotent on its own output: it assigns `value_raw_unit` from the unit it is
currently looking at, so a second pass over an already-converted record overwrites the
document-printed unit (`mM`) with the canonical one (`uM`) and loses the fact that `value_raw`
was measured in millimolar. A consumer reading `value_raw` beside `value_raw_unit` would then be
wrong by 1000x -- the exact failure that field exists to prevent. Each repair below is instead
guarded to be a no-op once applied, so this stage can run on every pipeline invocation.

Nothing here calls a model, reads a PDF or touches a sequence. Sequence-level correction comes
from step 9 re-extracting the document and from step 12's resolution revalidation; conflating
them here would hide how much of the corpus is still stale.
"""
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from sample import config
from sample.concentration import activity_unit_is_coherent
from sample.nlp import parse_activity_value_fast
from sample.preprocessing import (canonicalize_unit, _MAX_ACTIVITY_TYPE_CHARS, _WORD_RANGE)
from sample.scale import CONFIDENT, diagnose, display_unit

logger = logging.getLogger("sample.polish")

#: Written into each repaired file so a later reader can tell which rules produced it, and so a
#: future rule change can target only the files that predate it. Bump when a repair is added.
POLISH_VERSION: int = 2

#: Metadata key holding this stage's record of what it changed.
POLISH_KEY: str = "polish"

#: Repairs this stage performs, in the order applied. Order matters for exactly one pair: the
#: unit is canonicalized BEFORE coherence is judged, so a mojibake `�M` is read as the
#: micromolar it is rather than rejected as an unrecognized unit.
REPAIRS: Tuple[str, ...] = ("unit_repaired", "type_moved", "range_bound_added",
                            "scale_repaired", "activity_dropped_incoherent_unit")

#: Key written on a repaired activity recording the unit it carried before a scale repair, which
#: is both the provenance and the idempotence guard: a second pass sees the corrected unit, finds
#: nothing to answer for, and leaves it alone.
SCALE_REPAIR_KEY: str = "scale_repair"


def _repair_unit(activity: Dict[str, Any]) -> bool:
    """
    Canonicalizes `activity_unit` and `value_raw_unit` in place.

    :param activity: One activity dict, mutated.
    :returns: Whether anything changed.

    Idempotent because :func:`sample.preprocessing.canonicalize_unit` is: a unit already in
    canonical spelling maps to itself. Only the SPELLING is touched -- no magnitude conversion
    happens here, so `value_lower` stays consistent with `activity_unit`.

    **Two canonicalizers, in that order.** `canonicalize_unit` handles the spellings the pipeline
    produces; it does not handle the ones PDF extraction produces, and measured over the corpus it
    repaired none of `u M` (251), `AM` (113), `WM` (113), `lgml` (75), `mg/liter` (70), `Ag/mL`
    (58), `ug mL-1` (33), `microgram/mL` (25) or `$\mu$ M` (20) -- 1,212 activities in all, every
    one of them a unit the field does use, written in a way the release vocabulary does not
    contain. :func:`sample.scale.canonical_unit` recognises those, so it runs second on whatever
    the first left off-vocabulary. It is applied only when it yields a unit that IS in the release
    vocabulary, so an unrecognised string is left exactly as written rather than coerced.
    """
    from sample.validation_rules import VALID_UNITS
    from sample.scale import basis_of, canonical_unit as scale_canonical_unit, display_unit

    vocabulary = {unit.strip().lower() for unit in VALID_UNITS if unit}
    changed = False
    for field in ("activity_unit", "value_raw_unit"):
        current = activity.get(field)
        if not isinstance(current, str) or not current:
            continue
        repaired = canonicalize_unit(current)
        if repaired and repaired.strip().lower() not in vocabulary:
            key = scale_canonical_unit(repaired)
            if basis_of(key):
                spelled = display_unit(key)
                if spelled.strip().lower() in vocabulary:
                    repaired = spelled
        if repaired != current:
            activity[field] = repaired
            changed = True
    return changed


def _move_prose_type(activity: Dict[str, Any]) -> bool:
    """
    Moves an over-long `activity_type` into `activity_type_detail`.

    :param activity: One activity dict, mutated.
    :returns: Whether anything changed.

    Idempotent: after the move the type is empty, so the length test cannot fire again. The prose
    is kept rather than discarded because the value, unit and target beside it may all be real.
    """
    kind = activity.get("activity_type")
    if not isinstance(kind, str) or len(kind) <= _MAX_ACTIVITY_TYPE_CHARS:
        return False
    activity["activity_type_detail"] = kind
    activity["activity_type"] = ""
    return True


def _add_range_bound(activity: Dict[str, Any]) -> bool:
    """
    Fills `value_upper` from a range in `value_raw`, in the unit `value_lower` is already in.

    :param activity: One activity dict, mutated.
    :returns: Whether anything changed.

    The scale is derived from the record rather than from the unit tables: `value_lower` has
    already been through whatever magnitude conversion its unit required, so the ratio between it
    and the lower bound parsed out of `value_raw` IS that conversion. A record storing
    ``value_raw "0.3 to 0.5"`` with ``value_lower 300.0`` yields a factor of 1000 and an upper
    bound of 500.0, matching its `activity_unit` of uM -- without this stage needing to know that
    the document printed millimolar.

    Skipped when the factor cannot be established (a parsed lower bound of zero), when the bounds
    run backwards, or when a bound is already present. Idempotent on all three counts.
    """
    if activity.get("value_upper") not in (None, ""):
        return False
    raw = activity.get("value_raw")
    if not isinstance(raw, str) or not raw.strip():
        return False
    lower = activity.get("value_lower")
    if not isinstance(lower, (int, float)):
        return False

    _, parsed_lower, parsed_upper = parse_activity_value_fast(_WORD_RANGE.sub("-", raw))
    if parsed_lower in (None, 0) or parsed_upper is None:
        return False
    if parsed_upper < parsed_lower:
        return False
    scale = float(lower) / float(parsed_lower)
    upper = float(parsed_upper) * scale
    if upper < float(lower):
        return False
    activity["value_upper"] = upper
    return True


def _repair_scale(record: Dict[str, Any], activity: Dict[str, Any]) -> bool:
    """
    Corrects a unit whose prefix was lost in extraction, when the JSON proves which one.

    :param record: The record the activity belongs to, for its sibling measurements.
    :param activity: One activity dict, mutated.
    :returns: Whether anything changed.

    Only :data:`sample.scale.CONFIDENT` findings, which means one of two things: the activity's
    own `context_sentence` prints this very value in a different unit, or the unit is one a
    potency cannot be stated in at all (`g/mL` is three orders past solubility and appears zero
    times in 182,766 curated measurements). Everything merely improbable -- `mg/mL`, which the
    curated databases use 38 times and this extraction 1,733 -- is left for step 12 and a model,
    because a paper may report one assay in mg/mL and the rest in ug/mL, and sibling agreement is
    suggestive rather than proof.

    **Only the unit string changes; no number is touched.** `value_lower` is expressed in
    `activity_unit`, so restoring a dropped prefix is a change of unit at a fixed number -- 640
    g/mL was always the figure 640, printed beside a micro sign that PDF extraction dropped.
    Rescaling the number as well would move the quantity twice.

    Measured over the corpus: 268 activities, of which 227 are `g/mL` to `ug/mL`.
    """
    if activity.get(SCALE_REPAIR_KEY):
        return False
    finding = diagnose(record, activity)
    if finding.status != CONFIDENT or not finding.proposed_unit or not finding.rescalable:
        return False
    before = activity.get("activity_unit")
    # The finding carries a lower-case lookup key; the corpus stores `uM` and `ug/mL`. Writing
    # the key verbatim introduced a spelling nothing else used, and `_repair_unit` then rewrote
    # it on the next pass -- so the repair was not idempotent, it just looked that way until the
    # two stages ran in the same order twice.
    corrected = canonicalize_unit(display_unit(finding.proposed_unit)) or display_unit(
        finding.proposed_unit)
    activity["activity_unit"] = corrected
    activity[SCALE_REPAIR_KEY] = {"from": before, "to": corrected,
                                  "factor": finding.factor, "why": finding.why}
    return True


def polish_activities(activities: List[Any]) -> Tuple[List[Any], Dict[str, int]]:
    """
    Applies every repair to one record's activities.

    :param activities: The record's ``biological_activities``.
    :returns: ``(kept, counts)`` where `counts` is keyed by :data:`REPAIRS`.
    """
    counts: Dict[str, int] = {name: 0 for name in REPAIRS}
    kept: List[Any] = []
    # The scale repair reads the record's other measurements of the same assay, and this list IS
    # that record as far as `scale.sibling_units` is concerned -- so it is wrapped rather than
    # threaded through a new parameter, which keeps this function's signature.
    as_record = {"biological_activities": activities or []}
    for activity in activities or []:
        if not isinstance(activity, dict):
            kept.append(activity)
            continue
        if _repair_unit(activity):
            counts["unit_repaired"] += 1
        if _move_prose_type(activity):
            counts["type_moved"] += 1
        # After the unit is canonicalized and before coherence is judged: a repaired unit is what
        # the scale test should see, and a unit this corrects may be what makes the pair coherent.
        if _repair_scale(as_record, activity):
            counts["scale_repaired"] += 1
        # Judged after the unit is canonicalized, so a repaired `�M` is recognized as
        # micromolar instead of being rejected as an unknown unit.
        coherent, _reason = activity_unit_is_coherent(activity.get("activity_type"),
                                                      activity.get("activity_unit"))
        if not coherent:
            counts["activity_dropped_incoherent_unit"] += 1
            continue
        if _add_range_bound(activity):
            counts["range_bound_added"] += 1
        kept.append(activity)
    return kept, counts


def polish_record(payload: Dict[str, Any]) -> Dict[str, int]:
    """
    Polishes one extraction document in place.

    :param payload: A parsed ``{"metadata": {...}, "peptides": [...]}`` document, mutated.
    :returns: Counts keyed by :data:`REPAIRS`, all zero when nothing needed repair.
    """
    totals: Dict[str, int] = {name: 0 for name in REPAIRS}
    for record in payload.get("peptides") or []:
        if not isinstance(record, dict):
            continue
        kept, counts = polish_activities(record.get("biological_activities") or [])
        record["biological_activities"] = kept
        for name, value in counts.items():
            totals[name] += value
    return totals


def _write_atomically(path: str, payload: Dict[str, Any],
                      expect_mtime: Optional[float] = None) -> bool:
    """
    Replaces `path` with `payload` via a temporary file in the same directory.

    :param path: Destination document.
    :param payload: Document to serialize.
    :param expect_mtime: The mtime observed when `path` was read. When given and the file has
        changed since, nothing is written and this returns ``False``.
    :returns: Whether the file was replaced.

    A plain ``open(path, "w")`` truncates before it writes, so an interruption mid-write leaves a
    truncated JSON that every later stage has to treat as a lost document. This stage rewrites
    thousands of files that already hold the only copy of their extraction, so the window matters.

    `expect_mtime` guards against the realistic concurrency case rather than a theoretical one:
    step 9 writes into the same directory for days at a time, and an operator may well run this
    stage by hand while it does. Without the check, this sequence loses a document's work --
    polish reads it, step 9 re-extracts and rewrites it, polish's replace then reinstates the
    older content, and because that older content still carries the refined-pass marker, step 9's
    resume will never revisit it. The window is narrowed to the few milliseconds between the
    final stat and `os.replace` instead of the seconds a document takes to repair.
    """
    if expect_mtime is not None:
        try:
            if os.path.getmtime(path) != expect_mtime:
                return False
        except OSError:
            return False
    directory = os.path.dirname(os.path.abspath(path)) or "."
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".polish-", suffix=".tmp", delete=False)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
        return True
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def polish_extractions(base_dir: Optional[str] = None, dry_run: bool = False,
                       limit: int = 0) -> Dict[str, int]:
    """
    Re-applies the current activity-field rules to every extraction document under `base_dir`.

    :param base_dir: Directory of extraction output; ``data/literature_data_json`` by default.
    :param dry_run: Report what would change without writing anything.
    :param limit: Stop after this many documents; 0 means all.
    :returns: A summary keyed by :data:`REPAIRS` plus ``documents``, ``documents_changed``,
        ``unreadable`` and ``activities_seen``.

    Documents are located through :func:`sample.config.iter_extraction_files`, never a raw walk:
    the output tree accumulates manual backups beside the live sources (`pubmed.bak`,
    `pubmed_old`), and a walk over it once shipped a database built from superseded extractions
    with nothing to indicate it.
    """
    root = base_dir or str(config.DATA_DIR / "literature_data_json")
    paths = config.iter_extraction_files(root)
    if limit:
        paths = paths[:limit]

    summary: Dict[str, int] = {name: 0 for name in REPAIRS}
    summary.update({"documents": 0, "documents_changed": 0, "unreadable": 0,
                    "activities_seen": 0, "skipped_changed_underneath": 0})
    logger.info(f"polish: {len(paths)} extraction document(s) under {root} "
                f"(dry_run={dry_run})")

    for path in paths:
        summary["documents"] += 1
        try:
            mtime = os.path.getmtime(path)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as e:
            # Counted and named rather than passed over: a document that cannot be read here is
            # one this stage silently declines to correct, which is exactly the kind of partial
            # result that has to be visible.
            summary["unreadable"] += 1
            logger.warning(f"polish: cannot read {path}: {type(e).__name__}: {e}")
            continue
        if not isinstance(payload, dict):
            summary["unreadable"] += 1
            logger.warning(f"polish: {path} is not an extraction document")
            continue

        summary["activities_seen"] += sum(
            len(r.get("biological_activities") or [])
            for r in (payload.get("peptides") or []) if isinstance(r, dict))

        counts = polish_record(payload)
        if not any(counts.values()):
            continue

        if dry_run:
            summary["documents_changed"] += 1
            for name, value in counts.items():
                summary[name] += value
            continue

        metadata = payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = payload["metadata"] = {}
        record = metadata.setdefault(POLISH_KEY, {})
        if not isinstance(record, dict):
            record = metadata[POLISH_KEY] = {}
        record["version"] = POLISH_VERSION
        for name, value in counts.items():
            if value:
                record[name] = record.get(name, 0) + value
        if not _write_atomically(path, payload, expect_mtime=mtime):
            # Another writer got there first -- almost always step 9 re-extracting the document,
            # whose fresh output is better than anything this stage would have produced from the
            # version it read. Counted so a run during an active extraction is legible rather
            # than looking like a smaller corpus.
            summary["skipped_changed_underneath"] += 1
            logger.debug(f"polish: {path} changed while being polished; left alone")
            continue
        summary["documents_changed"] += 1
        for name, value in counts.items():
            summary[name] += value

    logger.info(
        f"polish: {summary['documents']:,} document(s) read, "
        f"{summary['documents_changed']:,} changed, {summary['unreadable']:,} unreadable, "
        f"{summary['skipped_changed_underneath']:,} changed underneath; "
        f"{summary['activities_seen']:,} activities seen")
    for name in REPAIRS:
        logger.info(f"polish:   {name} = {summary[name]:,}")
    if dry_run:
        logger.info("polish: dry run -- nothing was written")
    return summary
