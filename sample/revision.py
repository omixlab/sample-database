"""
Step 18: produce the revised extraction set that every later step consumes.

The pipeline's earlier stages each remove one class of error and hand the rest forward. This stage
is where that stops: it applies the strict contract in sample/strict_schema.py, gives the LLM one
chance to repair what the contract rejected, and writes the result to `*_revised`. Nothing
downstream reads the pre-revision directories, so the contract holds for the whole database rather
than for whichever fields a given consumer happens to check.

The order -- deterministic gate, then LLM, then the SAME deterministic gate again -- is the point.
A model proposal is a candidate value, never an accepted one: it is published only if the rules
that rejected the original independently accept the replacement. A model can therefore recover data
this stage would otherwise lose, but it cannot put anything into the database on its own authority.

Provenance is recorded honestly. Records carry `revision` describing what this automated stage did;
nothing here is human curation and nothing is labelled as such.
"""
import concurrent.futures
import glob
import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sample.strict_schema import (
    enforce_activity, enforce_peptide, VALID_ACTIVITY_TYPES, VALID_UNITS,
)

logger = logging.getLogger("sample.revision")

# Violations the LLM has a realistic chance of repairing from the sentence the value came from.
# Everything else is deliberately excluded: a rejected sequence cannot be "corrected" without
# inventing residues, and a measurement with no target cannot have one supplied without inventing
# the experiment. Asking a model to fill those is asking it to fabricate.
_LLM_REPAIRABLE_FIELDS: Set[str] = {"target_organism", "activity_type", "activity_unit", "value_lower"}

_REVISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "target_organism": {"type": "string"},
                    "activity_type": {"type": "string"},
                    "activity_unit": {"type": "string"},
                    "value_lower": {"type": "string"},
                },
                "required": ["id"],
            },
        }
    },
    "required": ["records"],
}

_REVISION_SYSTEM = (
    "You correct malformed fields in data extracted from scientific papers.\n"
    "For each numbered record you are given the sentence it came from and which fields are "
    "invalid. Return a corrected value for those fields ONLY, read from the sentence.\n"
    "Rules:\n"
    "- target_organism: the full scientific binomial of the organism the peptide was tested "
    'AGAINST (e.g. "Escherichia coli"). Not the organism the peptide came from. If the sentence '
    'does not name one, return "".\n'
    "- activity_type: the assay, as one of: " + ", ".join(sorted(VALID_ACTIVITY_TYPES)) + ".\n"
    "- activity_unit: the unit as written in the sentence, one of: "
    + ", ".join(sorted(VALID_UNITS)) + ".\n"
    "- value_lower: the measured number only, as a plain decimal string.\n"
    'Return "" for any field the sentence does not state. Never guess from prior knowledge -- an '
    "empty answer is always better than an invented one. Answer only with the JSON schema given."
)

#: Records per repair call. Small batches keep the model's per-record accuracy up.
_MAX_LLM_BATCH: int = 12
#: Characters of the source sentence shown to the model as evidence for a repair.
_EVIDENCE_CHARS: int = 400


def _repairable(violations: List[Any]) -> Set[str]:
    """
    :param violations: Violations returned by :func:`strict_schema.enforce_activity`.
    :returns: The names of the failing fields the LLM has a realistic chance of repairing.
    """
    return {v.field for v in violations if v.field in _LLM_REPAIRABLE_FIELDS}


def _revise_activities_llm(pending: List[Tuple[Dict[str, Any], Set[str]]],
                           doc_id: str = "",
                           model: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Asks the model for corrected values for `pending`, a list of (activity, failing_fields).

    Returns a dict {index: {field: proposed_value}}. Fails open to {} -- when the model is
    unavailable the deterministic verdict simply stands, which drops the record. That is the safe
    direction here: this layer only ever adds data back.
    """
    if not pending:
        return {}
    try:
        from sample.llm import llm_call, clean_json, _VALIDATION_BREAKER, _LLM_CALL_TIMEOUT
        from sample.config import DEFAULT_EXTRACT_MODEL
    except Exception as e:
        logger.debug(f"[{doc_id}] LLM revision unavailable: {e}")
        return {}

    if _VALIDATION_BREAKER.is_open():
        return {}

    lines = []
    for i, (act, fields) in enumerate(pending):
        evidence = " ".join(str(act.get("context_sentence") or "").split())[:_EVIDENCE_CHARS]
        current = ", ".join(f"{f}={act.get(f)!r}" for f in sorted(fields))
        lines.append(f'{i}. invalid: {current}\n   sentence: "{evidence}"')

    try:
        raw = llm_call(
            [{"role": "system", "content": _REVISION_SYSTEM},
             {"role": "user", "content": "\n".join(lines) +
              '\n\nReturn {"records":[{"id":<int>,...corrected fields...}]}'}],
            temperature=0.0, model=model or DEFAULT_EXTRACT_MODEL,
            max_tokens=512, timeout=_LLM_CALL_TIMEOUT, response_schema=_REVISION_SCHEMA,
        )
        _VALIDATION_BREAKER.record_success()
    except Exception as e:
        logger.debug(f"[{doc_id}] LLM revision call failed: {e}")
        try:
            _VALIDATION_BREAKER.record_failure(e)
        except Exception:
            pass
        return {}

    try:
        parsed = clean_json(raw)
    except Exception:
        return {}
    records = parsed.get("records") if isinstance(parsed, dict) else None
    if not isinstance(records, list):
        return {}

    proposals = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        try:
            idx = int(rec.get("id"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(pending)):
            continue
        fields = pending[idx][1]
        proposal = {f: rec[f] for f in fields if isinstance(rec.get(f), (str, int, float))
                    and str(rec.get(f)).strip()}
        if proposal:
            proposals[idx] = proposal
    return proposals


def revise_document(doc: Any, doc_id: str = "", use_llm: bool = True,
                    model: Optional[str] = None) -> Tuple[Any, Counter]:
    """
    Applies the contract to one document, with an LLM repair attempt for what it rejected.

    Returns (revised_document, stats).
    """
    stats = Counter()
    if isinstance(doc, list):
        peptides, envelope = doc, None
    elif isinstance(doc, dict):
        peptides, envelope = doc.get("peptides") or [], doc
    else:
        return doc, stats

    # Pass 1: deterministic contract. Collect what failed on a field the LLM might repair.
    #
    # Two kinds of failure are collected, not one. An activity the contract REJECTED outright is an
    # obvious candidate for repair. An activity that SURVIVED with a field cleared is just as much
    # one: "an MIC of 4 uM was recorded against Pseudomonas aeruginosa" whose organism arrived as
    # something NCBI does not recognize passes the contract with a blank target, and the correct
    # organism is sitting in the sentence. Only chasing outright rejections left that data on the
    # floor while reporting the record as clean.
    kept, pending, pending_owner = [], [], []
    for p_idx, pep in enumerate(peptides):
        stats["peptides_in"] += 1
        if not isinstance(pep, dict):
            stats["peptides_dropped"] += 1
            continue

        surviving, retry = [], []
        for a_idx, act in enumerate(pep.get("biological_activities") or []):
            stats["activities_in"] += 1
            clean, violations = enforce_activity(act, path=f"peptides[{p_idx}].act[{a_idx}]")
            fields = _repairable(violations)
            if clean is not None:
                surviving.append(clean)
                for v in violations:
                    stats[f"field_cleared:{v.field}"] += 1
                if use_llm and fields:
                    # Replace this entry in place if the model can restore the cleared field.
                    retry.append((act, fields, len(surviving) - 1))
                continue
            stats["activities_dropped"] += 1
            for v in violations:
                stats[f"drop_reason:{v.field}"] += 1
            if use_llm and fields:
                retry.append((act, fields, None))

        candidate = dict(pep)
        candidate["biological_activities"] = surviving
        kept.append(candidate)
        for act, fields, slot in retry:
            pending.append((act, fields))
            pending_owner.append((len(kept) - 1, slot))

    # Pass 2: one LLM call per batch for everything worth repairing.
    if pending:
        for start in range(0, len(pending), _MAX_LLM_BATCH):
            batch = pending[start:start + _MAX_LLM_BATCH]
            proposals = _revise_activities_llm(batch, doc_id=doc_id, model=model)
            for local_idx, proposal in proposals.items():
                act, _ = batch[local_idx]
                repaired = dict(act)
                repaired.update(proposal)
                # Pass 3: the SAME contract judges the model's proposal. Nothing reaches the
                # database because a model suggested it.
                clean, _ = enforce_activity(repaired, path="llm-revised")
                if clean is None:
                    continue
                owner, slot = pending_owner[start + local_idx]
                acts = kept[owner]["biological_activities"]
                if slot is None:
                    acts.append(clean)
                    stats["activities_recovered_by_llm"] += 1
                else:
                    acts[slot] = clean
                    stats["fields_restored_by_llm"] += 1

    # Pass 4: peptide-level contract, now that activity lists are final.
    final = []
    for pep in kept:
        clean, violations = enforce_peptide(pep, path="peptide")
        if clean is None:
            stats["peptides_dropped"] += 1
            for v in violations:
                stats[f"drop_reason:{v.field}"] += 1
            continue
        clean["revision"] = {
            # NOT A STEP REFERENCE -- do not renumber. This is a provenance value PERSISTED into
            # every peptide record, and the records already on disk carry
            # "step_13_automated_revision". Shifting it with a pipeline renumber makes new records
            # disagree with every old one for no benefit; it has never matched a Makefile target
            # name. Two renumbers have now bumped it by accident.
            "stage": "step_13_automated_revision",
            "contract": "sample.strict_schema",
            "llm_assisted": bool(use_llm),
        }
        final.append(clean)
        stats["peptides_out"] += 1
        stats["activities_out"] += len(clean.get("biological_activities") or [])

    if envelope is None:
        return final, stats
    out = dict(envelope)
    out["peptides"] = final
    return out, stats


def _revise_file(job: Tuple[str, str, bool, Optional[str]]) -> Counter:
    """
    Revises one document file. The unit of work for the thread pool.

    :param job: ``(source_path, destination_path, use_llm, model)``.
    :returns: Per-file statistics, ready to be summed into a run total.
    """
    src, dst, use_llm, model = job
    try:
        with open(src, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        logger.debug(f"Unreadable {src}: {e}")
        return Counter({"files_unreadable": 1})

    revised, stats = revise_document(doc, doc_id=os.path.basename(src),
                                     use_llm=use_llm, model=model)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(revised, f, indent=2, ensure_ascii=False)
        os.replace(tmp, dst)
        stats["files_written"] += 1
    except Exception as e:
        logger.error(f"Could not write {dst}: {e}")
        stats["files_failed"] += 1
    return stats


def revise_directory(input_dir: str, output_dir: str, use_llm: bool = True,
                     model: Optional[str] = None, workers: int = 8,
                     limit: Optional[int] = None) -> Counter:
    """Applies the contract to every JSON under `input_dir`, mirroring the tree into `output_dir`."""
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True))
    if limit:
        files = files[:limit]
    if not files:
        logger.warning(f"No JSON files found under {input_dir}")
        return Counter()

    jobs = [(src, os.path.join(output_dir, os.path.relpath(src, input_dir)), use_llm, model)
            for src in files]
    totals = Counter()
    from tqdm import tqdm
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for stats in tqdm(pool.map(_revise_file, jobs), total=len(jobs),
                          desc=f"Revising {os.path.basename(input_dir)}"):
            totals.update(stats)
    return totals


def run_revision(sources: Any = "all", workers: int = 8, use_llm: bool = True,
                 model: Optional[str] = None, limit: Optional[int] = None) -> Counter:
    """
    Step 18 entry point. Writes `<dir>_revised` beside each input directory.

    Returns the aggregate stats so the caller can report what the contract removed.
    """
    from sample.config import DATA_DIR

    def _best_input(base: str) -> str:
        """
        Prefers step 14's LLM-validated output over the raw extraction.

        Step 14 runs a per-field LLM audit and writes `<base>_validated`, but nothing read those
        files -- it used them only to compute its own report and then discarded them, so every
        correction it made was thrown away and this stage started again from the unvalidated
        extraction. Chaining them means the revision builds on that work instead of repeating it.
        Falls back to the raw directory when step 14 has not been run.
        """
        validated = str(DATA_DIR / f"{base}_validated")
        if os.path.isdir(validated) and glob.glob(os.path.join(validated, "**", "*.json"),
                                                  recursive=True):
            logger.info(f"Using step-10 validated input: {validated}")
            return validated
        return str(DATA_DIR / base)

    wanted = set(sources) if isinstance(sources, (list, tuple, set)) else {sources}
    targets = []
    if wanted & {"all", "pubmed", "scopus", None}:
        targets.append((_best_input("literature_data_json"),
                        str(DATA_DIR / "literature_data_json_revised")))
    if wanted & {"all", "patents", None}:
        targets.append((_best_input("patent_data_json"),
                        str(DATA_DIR / "patent_data_json_revised")))

    totals = Counter()
    for src, dst in targets:
        if not os.path.isdir(src):
            logger.warning(f"Skipping {src}: not a directory")
            continue
        logger.info(f"Revising {src} -> {dst}")
        totals.update(revise_directory(src, dst, use_llm=use_llm, model=model,
                                       workers=workers, limit=limit))

    pep_in = totals.get("peptides_in", 0)
    act_in = totals.get("activities_in", 0)
    logger.info(
        f"Revision complete. Peptides {pep_in} -> {totals.get('peptides_out', 0)}; "
        f"activities {act_in} -> {totals.get('activities_out', 0)} "
        f"(recovered by LLM: {totals.get('activities_recovered_by_llm', 0)}); "
        f"{totals.get('files_written', 0)} files written."
    )
    reasons = {k.split(":", 1)[1]: v for k, v in totals.items() if k.startswith("drop_reason:")}
    for field, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:10]:
        logger.info(f"  dropped on {field}: {n}")
    return totals
