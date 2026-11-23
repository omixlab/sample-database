"""
Compare a candidate LangExtract model against what is already on disk, on the same documents.

**Why this exists.** On 2026-10-13 ``gemma3:4b`` was made the refined pass's model on the reasoning
that it is the same family as ``gemma4``, so the worker's tuned prompt would still apply. It was
not validated first. It returned derivatization reagents (``o-phthalaldehyde``, ``fluorescamine``,
``4-halo-7-nitro-benzo-2,1,3-oxadiazole``), categories (``20 essential amino acids``) and names
fused to identifiers (``Magainin10442437``) as peptides, and ~456 documents were written before
anyone looked. A model's size, family and benchmark scores say nothing about whether it will
respect this project's extraction contract; only running it on real documents does.

**What it measures.** The candidate runs through the real backend -- same worker, same prompt, same
chunking -- over documents that already have LLM-enabled output on disk, so the baseline costs
nothing to produce. Reported per model: records per document, how many carry a sequence (the only
ones ``merge-extractions`` keeps), how many carry no activity at all, a junk-name rate, and the
name-level overlap with the baseline, which is what distinguishes "finds different things" from
"finds nonsense".

The candidate's cache is disabled, so the numbers are real calls rather than a replay.

Usage::

    python -m sample.model_check --model qwen3:14b --limit 30
"""
import argparse
import collections
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sample.model_check")

#: The 20 standard amino acids, for deciding whether a "name" is really a sequence.
_AA = set("ACDEFGHIKLMNPQRSTVWY")

#: Heuristics for a name that cannot be a peptide. Each is a measured failure class from the
#: gemma3:4b incident or from the audit of gemma4's own output, not a guess.
_JUNK: Tuple[Tuple[str, Any], ...] = (
    ("phrase", lambda s: len(s.split()) >= 4),
    ("category", lambda s: re.search(
        r"\b(peptides|analogues|analogs|derivatives|compounds|fractions|variants|proteins)\b",
        s, re.I)),
    ("reagent", lambda s: re.search(
        r"(chlorid|oxadiazol|aldehyde|sulphon|sulfon|phthal|naphthalen|benzo|isoindol|fluoresc)",
        s, re.I)),
    ("fused-digits", lambda s: re.search(r"[A-Za-z]\d{5,}", s)),
    ("leading-digit", lambda s: re.match(r"^\d", s)),
    ("over-40-chars", lambda s: len(s) > 40 and not _is_sequence(s)),
    ("too-short", lambda s: len(s.strip()) <= 2),
    ("markup", lambda s: re.search(r"\$|\\[a-zA-Z]{2,}", s)),
)


def _is_sequence(value: str, minimum: int = 10) -> bool:
    """
    :param value: A candidate string.
    :param minimum: Shortest residue run to accept.
    :returns: True when the string is a bare run of standard amino acids. Used both to spot a
        sequence sitting in the name field (recoverable) and to stop a long sequence being counted
        as an over-long junk name.

    Membership in the 20-letter alphabet is NOT sufficient, and assuming it was is a mistake this
    project has already made once: ``sample.synonyms`` fabricated sequences like ``THEPEPTIDES``
    and ``CASEICIN`` from prose that way. Ordinary English passes the alphabet test -- strip the
    space from "magainin peptides" and ``MAGAININPEPTIDES`` is composed entirely of valid residue
    letters. Two further conditions do the real work:

    * **One token.** A sequence is written without spaces, so anything containing whitespace is
      prose however its letters spell out.
    * **Written uppercase.** Sequences are conventionally capitalised and peptide names usually are
      not, which separates ``GNPVPILIPCHR`` from ``magainin``. This under-counts a sequence written
      in lower case, which is the right direction to be wrong in: it reports less recoverable data
      than exists rather than inventing some.
    """
    text = (value or "").strip()
    if not text or re.search(r"\s", text):
        return False
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < minimum or letters != letters.upper():
        return False
    return all(c in _AA for c in letters.upper())


def _profile(records: List[Dict[str, Any]], documents: int, seconds: float) -> Dict[str, Any]:
    """
    :param records: Peptide records from one model.
    :param documents: How many documents produced them.
    :param seconds: Wall time spent.
    :returns: The comparable summary for one model.
    """
    names = [(r.get("peptide_name") or "").strip() for r in records]
    sequenced = sum(1 for r in records if (r.get("peptide_sequence") or "").strip())
    no_activity = sum(1 for r in records if not (r.get("biological_activities") or []))
    name_is_sequence = sum(1 for r, n in zip(records, names)
                           if not (r.get("peptide_sequence") or "").strip() and _is_sequence(n))
    classes = collections.Counter()
    junk = 0
    for n in names:
        hit = False
        for label, test in _JUNK:
            if n and test(n):
                classes[label] += 1
                hit = True
        junk += bool(hit)
    total = max(1, len(records))
    return {
        "documents": documents,
        "records": len(records),
        "per_document": len(records) / max(1, documents),
        "with_sequence_pct": sequenced / total * 100,
        "name_is_sequence_pct": name_is_sequence / total * 100,
        "no_activity_pct": no_activity / total * 100,
        "junk_name_pct": junk / total * 100,
        "junk_classes": dict(classes.most_common()),
        "seconds_per_document": seconds / max(1, documents),
        "names": {n.lower() for n in names if n},
    }


def _baseline_records(path: str) -> Optional[List[Dict[str, Any]]]:
    """
    :param path: An extraction output JSON.
    :returns: Its LLM-attributed records, or None when the document was not run with the backend
        enabled (a screening-pass document has no LLM output to compare against).
    """
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if str((document.get("metadata") or {}).get("langextract_backend")) != "enabled":
        return None
    return [p for p in (document.get("peptides") or [])
            if str(p.get("discovery_path")) == "langextract_match"]


def run(model: str, limit: int = 30, seed: int = 20261014,
        data_dir: str = "data", baseline_before: Optional[float] = None) -> Dict[str, Any]:
    """
    Runs `model` over documents that already carry LLM output, and compares.

    :param model: The candidate Ollama tag.
    :param limit: How many documents to test.
    :param seed: Sampling seed, so a rerun uses the same documents.
    :param data_dir: Project data directory.
    :param baseline_before: Only use documents last written BEFORE this epoch time as the
        baseline. Needed because the on-disk output is not uniformly trustworthy: on 2026-10-14 an
        Ollama eviction war returned 503 for nearly every request and ~284 documents were written
        with ZERO LLM records, followed by ~400 more from a model that averaged 0.81 records per
        document. Drawing a baseline from those would flatter any candidate -- the comparison would
        be against a broken run, not against the pipeline working. Pass a cutoff from before the
        damage to compare against healthy output.
    :returns: ``{"candidate": ..., "baseline": ..., "overlap": ...}``.
    """
    from sample import langextract_records as backend

    outputs: List[Tuple[str, str]] = []
    for source in ("pubmed", "scopus"):
        directory = os.path.join(data_dir, "literature_data_json", source)
        text_root = os.path.join(data_dir, "literature_data_text", source)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.endswith(".json"):
                text = os.path.join(text_root, name[:-5] + ".txt")
                if os.path.exists(text):
                    outputs.append((os.path.join(directory, name), text))

    random.Random(seed).shuffle(outputs)
    chosen: List[Tuple[List[Dict[str, Any]], str]] = []
    for json_path, text_path in outputs:
        if len(chosen) >= limit:
            break
        if baseline_before is not None and os.path.getmtime(json_path) >= baseline_before:
            continue
        base = _baseline_records(json_path)
        if base is not None:
            chosen.append((base, text_path))
    if not chosen:
        raise SystemExit("No documents with LLM-enabled output found to compare against.")

    # The candidate must make real calls; a cache hit would measure the cache.
    os.environ["LANGEXTRACT_CACHE"] = ""
    backend.LANGEXTRACT_CACHE_DIR = ""

    baseline_records: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []
    started = time.time()
    for index, (base, text_path) in enumerate(chosen, start=1):
        baseline_records.extend(base)
        try:
            candidate_records.extend(
                backend.extract_records_from_text(text_path, model_id=model) or [])
        except Exception as e:  # noqa: BLE001 -- one bad document must not sink the comparison
            logger.warning("%s failed on %s: %s", model, os.path.basename(text_path), e)
        logger.info("%d/%d documents", index, len(chosen))
    elapsed = time.time() - started

    candidate = _profile(candidate_records, len(chosen), elapsed)
    baseline = _profile(baseline_records, len(chosen), float("nan"))
    shared = candidate["names"] & baseline["names"]
    union = candidate["names"] | baseline["names"]
    overlap = {
        "shared_names": len(shared),
        "candidate_only": len(candidate["names"] - baseline["names"]),
        "baseline_only": len(baseline["names"] - candidate["names"]),
        "jaccard_pct": (len(shared) / len(union) * 100) if union else 0.0,
    }
    for profile in (candidate, baseline):
        profile.pop("names", None)
    return {"model": model, "candidate": candidate, "baseline": baseline, "overlap": overlap}


def _print(result: Dict[str, Any]) -> None:
    """:param result: The output of :func:`run`, printed as an aligned comparison."""
    candidate, baseline = result["candidate"], result["baseline"]
    print(f"\n{'metric':26} {result['model']:>16} {'on-disk baseline':>18}")
    print("-" * 64)
    for key, label, suffix in (
            ("records", "records", ""),
            ("per_document", "records / document", ""),
            ("with_sequence_pct", "with a sequence", "%"),
            ("name_is_sequence_pct", "name IS a sequence", "%"),
            ("no_activity_pct", "no activity at all", "%"),
            ("junk_name_pct", "junk-looking names", "%"),
            ("seconds_per_document", "seconds / document", "s"),
    ):
        a, b = candidate.get(key), baseline.get(key)
        fa = f"{a:.2f}{suffix}" if isinstance(a, float) else f"{a}{suffix}"
        fb = "-" if isinstance(b, float) and b != b else (
            f"{b:.2f}{suffix}" if isinstance(b, float) else f"{b}{suffix}")
        print(f"{label:26} {fa:>16} {fb:>18}")
    print(f"\ncandidate junk classes: {candidate['junk_classes'] or 'none'}")
    print(f"baseline  junk classes: {baseline['junk_classes'] or 'none'}")
    o = result["overlap"]
    print(f"\nname overlap with baseline: {o['jaccard_pct']:.1f}% Jaccard "
          f"({o['shared_names']} shared, {o['candidate_only']} candidate-only, "
          f"{o['baseline_only']} baseline-only)")
    print("\nRead it this way: a LOW overlap with a LOW junk rate means the candidate finds "
          "different real things.\nA low overlap with a HIGH junk rate is the gemma3:4b failure, "
          "and is the signal to stop.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", required=True, help="Candidate Ollama tag, e.g. qwen3:14b")
    parser.add_argument("--limit", type=int, default=30, help="Documents to test (default 30)")
    parser.add_argument("--seed", type=int, default=20261014, help="Sampling seed")
    parser.add_argument("--baseline-before", default=None,
                        help="Only baseline against documents written before this time "
                             "(ISO 8601, e.g. 2026-10-14T12:00). Use it to exclude a window where "
                             "the pipeline was known to be writing degraded output.")
    parser.add_argument("--json", default=None, help="Also write the result to this path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cutoff = None
    if args.baseline_before:
        import datetime
        cutoff = datetime.datetime.fromisoformat(args.baseline_before).timestamp()
        print(f"baseline restricted to documents written before {args.baseline_before}")
    result = run(args.model, limit=args.limit, seed=args.seed, baseline_before=cutoff)
    _print(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
