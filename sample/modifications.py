"""
Post-translational and synthetic chemical modifications, harvested per peptide.

A peptide's sequence is not the whole molecule. Two thirds of the antimicrobial peptides in
DBAASP carry a chemical modification that the one-letter sequence cannot express, and the most
common of them changes the molecule's charge:

* **C-terminal amidation** (11,016 records, 44 % of DBAASP). The terminal ``-COOH`` becomes
  ``-CONH2``, removing one negative charge at physiological pH. A net charge computed from the
  sequence alone is wrong by +1 for every one of them, and net charge is the single most-cited
  property of an antimicrobial peptide.
* **N-terminal acylation** -- acetylation (1,336) and fatty-acid conjugation from caproyl (C6) to
  stearoyl (C18), which is how a lipopeptide is built.
* **Unusual residues** at stated positions: D-amino acids, ornithine, dehydro forms and the rest
  of the non-canonical alphabet, each recorded with the position it occupies.

None of this was harvested. The cyclisation and disulfide bracing already stored in
:class:`~sample.models.PeptideTopology` are the other half of the same picture; this module covers
what happens at the ends and at individual residues.

**Why the charge is not corrected here.** Recording the modification and silently rewriting the
descriptor are different claims. The descriptors are computed from the sequence by
:mod:`sample.enrichment`, and a peptide can be amidated in one source and not in another; folding
that into the number would hide the disagreement. The modification is published beside the
descriptor so a reader can see both, and :func:`charge_correction` gives the size of the effect
for anyone who wants to apply it.
"""
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from sample.config import DATA_DIR

logger = logging.getLogger("sample.modifications")

#: Terminal modification codes that remove the terminal carboxylate's negative charge. Amidation
#: is the overwhelming majority; the others are the same chemistry on a different scale.
_AMIDATING = frozenset({"AMD", "NH2", "AMIDE"})

#: Terminal codes that cap the N-terminal amine, removing its positive charge. Acetylation and
#: every fatty-acid conjugation do this -- the acyl group replaces a proton on the alpha-amino
#: nitrogen.
_N_ACYLATING = frozenset({
    "ACT", "FOR", "MYR", "PAL", "LIN",
    "C6", "C8", "C10", "C12", "C14", "C16", "C18",
})

#: Charge in elementary units that each class of terminal modification removes.
AMIDATION_CHARGE_DELTA: float = 1.0
N_ACYLATION_CHARGE_DELTA: float = -1.0


def classify_terminus(code: Any) -> str:
    """
    Names what a terminal modification does chemically.

    :param code: The source's short code, e.g. ``AMD``, ``ACT``, ``C16``.
    :returns: ``"amidation"``, ``"acylation"`` or ``"other"``.
    """
    token = str(code or "").strip().upper()
    if token in _AMIDATING:
        return "amidation"
    if token in _N_ACYLATING:
        return "acylation"
    return "other"


def charge_correction(modifications: List[Dict[str, Any]]) -> float:
    """
    How far a sequence-only net charge is from the modified molecule's.

    :param modifications: Rows as produced by :func:`harvest_modifications`.
    :returns: The signed correction to add to a charge computed from the bare sequence.
        C-terminal amidation contributes ``+1`` (a carboxylate that is no longer there to be
        deprotonated); N-terminal acylation contributes ``-1`` (an amine that can no longer be
        protonated). Zero when nothing charged was modified.
    """
    delta = 0.0
    for modification in modifications:
        kind = modification.get("kind")
        if kind == "amidation":
            delta += AMIDATION_CHARGE_DELTA
        elif kind == "acylation":
            delta += N_ACYLATION_CHARGE_DELTA
    return delta


def _terminus_rows(record: Dict[str, Any], sequence: str,
                   accession: str) -> Iterator[Dict[str, Any]]:
    """
    :param record: One DBAASP peptide record.
    :param sequence: Its cleaned sequence.
    :param accession: Its DBAASP id.
    :returns: One row per populated terminus.
    """
    for field, terminus in (("nTerminus", "N"), ("cTerminus", "C")):
        value = record.get(field)
        if not isinstance(value, dict):
            continue
        code = str(value.get("name") or "").strip()
        if not code:
            continue
        yield {
            "sequence": sequence,
            "position": None,
            "terminus": terminus,
            "code": code,
            "description": str(value.get("description") or "").strip(),
            "kind": classify_terminus(code),
            "source_db": "DBAASP",
            "source_accession": accession,
        }


def _residue_rows(record: Dict[str, Any], sequence: str,
                  accession: str) -> Iterator[Dict[str, Any]]:
    """
    :param record: One DBAASP peptide record.
    :param sequence: Its cleaned sequence.
    :param accession: Its DBAASP id.
    :returns: One row per non-canonical residue, carrying the position it occupies.
    """
    for entry in record.get("unusualAminoAcids") or []:
        if not isinstance(entry, dict):
            continue
        modification = entry.get("modificationType") or {}
        code = str(modification.get("name") or "").strip()
        if not code:
            continue
        try:
            position = int(entry.get("position"))
        except (TypeError, ValueError):
            position = None
        yield {
            "sequence": sequence,
            "position": position,
            "terminus": None,
            "code": code,
            "description": str(modification.get("description") or "").strip(),
            "kind": "unusual_residue",
            "source_db": "DBAASP",
            "source_accession": accession,
        }


def harvest_modifications(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Reads every modification DBAASP records.

    DBAASP is the only source here that states them. DRAMP, ADP and CAMP describe modifications in
    free-text notes when they mention them at all, which is not something to parse into a
    structured claim -- a wrong position is worse than a missing one.

    :param directory: DBAASP JSON directory; the packaged location when omitted.
    :returns: One row per modification, keyed by sequence so the loader can join on it.
    """
    import glob

    directory = directory or os.path.join(str(DATA_DIR), "antimicrobial_databases", "dbaasp")
    if not os.path.isdir(directory):
        logger.warning(f"DBAASP directory not found at {directory}; no modifications harvested.")
        return []

    from sample.sequence import sequence_to_1_letter

    rows: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    unreadable = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception:
            unreadable += 1
            continue
        if not isinstance(record, dict):
            continue
        sequence = sequence_to_1_letter(str(record.get("sequence") or "")).upper().strip()
        if not sequence:
            continue
        accession = str(record.get("dbaaspId") or "").strip()
        rows.extend(_terminus_rows(record, sequence, accession))
        rows.extend(_residue_rows(record, sequence, accession))

    if unreadable:
        logger.warning(f"{unreadable} of {len(files)} DBAASP files could not be read.")

    by_kind: Dict[str, int] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    logger.info(f"Modifications: {len(rows)} across "
                f"{len({r['sequence'] for r in rows})} sequences {by_kind}.")
    return rows


def build_modifications(output_path: Optional[str] = None,
                        directory: Optional[str] = None) -> Dict[str, Any]:
    """
    Harvests modifications and writes them for the loader.

    :param output_path: Where to write; ``data/modifications.json`` when omitted.
    :param directory: DBAASP JSON directory.
    :returns: A summary of what was written.
    """
    rows = harvest_modifications(directory)
    output_path = output_path or os.path.join(str(DATA_DIR), "modifications.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"modifications": rows}, handle, ensure_ascii=False, indent=1)

    summary = {
        "path": output_path,
        "modifications": len(rows),
        "sequences": len({r["sequence"] for r in rows}),
        "amidated": sum(1 for r in rows if r["kind"] == "amidation"),
        "acylated": sum(1 for r in rows if r["kind"] == "acylation"),
        "unusual_residues": sum(1 for r in rows if r["kind"] == "unusual_residue"),
    }
    logger.info(f"Wrote {summary['modifications']} modifications for "
                f"{summary['sequences']} sequences to {output_path} "
                f"({summary['amidated']} amidated, {summary['acylated']} acylated, "
                f"{summary['unusual_residues']} unusual residues).")
    return summary
