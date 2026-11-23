"""
Validates the published ``SMILES`` column, and replaces it with sequence-derived chemistry
wherever one exists.

Step 27 stopped reading the stored SMILES column entirely -- every descriptor and structure in
this pipeline is now built from the sequence (see :mod:`sample.enrichment`,
:mod:`sample.folding`). That fixed every *internal* consumer, but the column itself, as published
in ``final_dataset.csv`` and ``peptides.SMILES``, was never touched: it still carries whatever the
source databases or an earlier LLM pass wrote into it, including sentinels (``Not available``),
residue letters left in place of a structure (``C(G)``), truncated generations (``...``), and
model special tokens.

Two passes fix this, applied in order of trust:

1. **Prefer the sequence-derived structure.** ``data/predicted_structures.json``
   (:mod:`sample.folding`) already carries a SMILES read straight off a real 3D structure --
   either ColabFold's prediction or an RDKit-embedded conformer -- for every peptide that has one.
   That SMILES is round-trippable and connected by construction; the stored one is neither
   guaranteed. Wherever a derived record exists, it replaces the stored value outright.
2. **Reject what is left.** For a peptide with no derived structure yet, the stored value is kept
   only if :func:`is_plausible_smiles` cannot rule it out as one of the four garbage shapes above.
   A rejected value is set to empty, not replaced with a guess -- an empty cell says "unknown";
   a wrong guess says something false with the same confidence as a right one.
"""
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from sample.config import DATA_DIR

logger = logging.getLogger("sample.smiles_validation")

#: Strings occupying the SMILES column that are not attempts at a structure.
_SMILES_SENTINELS = frozenset({
    "not", "not available", "none", "null", "nan", "unknown", "n/a", "na", "-", "",
    "no entry found", "not found", "not applicable",
})

#: Characters that can appear in a SMILES string. Deliberately permissive -- this is a cheap
#: pre-filter to keep obvious non-molecules away from the parser, not a SMILES validator.
_SMILES_ALLOWED = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$:/\\.%*]+$")

#: Letters that are never a bare SMILES atom. The organic subset outside brackets is B, C, N, O,
#: P, S, F, I (plus Cl/Br as pairs and lowercase aromatics), so any of these appearing in an
#: all-letter string means the string is not SMILES notation at all.
_NEVER_SMILES_ATOMS = set("ADEGJKLMQRTUVWXYZ")

#: One-letter residue codes appearing where an atom should be. LLM-generated SMILES in this corpus
#: contain fragments such as ``C(G)`` and ``C(GAG)`` -- the amino acid letter was copied through
#: instead of being expanded into a structure.
#:
#: Restricted to residue letters that are NEVER valid as a bare SMILES atom. C, N, O, S, P, F, I
#: and B are all both residue codes and real atoms, so matching them would reject ``(C)`` -- a
#: methyl branch present in nearly every peptide SMILES, and the largest false positive available
#: here.
_RESIDUE_LEAK = re.compile(r"\((?:[GADEKLMQRTVWY]{1,4})\)")


def is_plausible_smiles(value: Any) -> bool:
    """
    Cheap pre-filter for strings that cannot be a molecule.

    The SMILES column is not validated anywhere upstream, and it holds several kinds of thing that
    are not structures: the sentinels every source uses for missing data (``Not available``,
    ``unknown``); LLM output that leaked residue letters into the string (``C(G)``, ``C(GAG)``);
    truncated generations ending in a literal ``...``; and, in at least one row, model special
    tokens (``<channel|>[{``).

    :param value: The SMILES cell.
    :returns: Whether it is worth keeping. False negatives are acceptable -- a genuine SMILES
        rejected here is nulled rather than published, which loses one structure; a sentinel
        accepted would publish a lie about the molecule.
    """
    text = str(value or "").strip()
    if text.lower() in _SMILES_SENTINELS:
        return False
    # A single-atom SMILES parses -- "C" is methane, "N" is ammonia -- but a peptide whose
    # structure is one atom is a degenerate record, not a molecule worth publishing.
    if len(text) < 2:
        return False
    if "..." in text or "<" in text or "|" in text:
        return False
    if not _SMILES_ALLOWED.match(text):
        return False
    if text.isalpha() and set(text.upper()) & _NEVER_SMILES_ATOMS:
        return False
    if _RESIDUE_LEAK.search(text):
        return False
    return True


def load_derived_smiles(predicted_structures_json: Optional[str] = None) -> Dict[str, str]:
    """
    Reads the sequence-derived SMILES already produced by step 27's structure pass.

    :param predicted_structures_json: Path to the record; the packaged location when omitted.
    :returns: ``entry_id -> smiles``. Empty when the file does not exist -- a checkout that has
        not run the fold yet gets pure validation (pass 2) instead of a replacement (pass 1).
    """
    predicted_structures_json = predicted_structures_json or os.path.join(
        str(DATA_DIR), "predicted_structures.json")
    if not os.path.exists(predicted_structures_json):
        return {}
    with open(predicted_structures_json, encoding="utf-8") as handle:
        records = json.load(handle) or {}
    return {entry_id: rec["smiles"] for entry_id, rec in records.items() if rec.get("smiles")}


def load_curated_smiles(base_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    DRAMP's curated SMILES, keyed by DRAMP id, for use as a trust tier below a real structure
    and above whatever the column happens to hold.

    :param base_dir: Root of the raw database dumps; ``data/antimicrobial_databases`` when
        omitted.
    :returns: ``{DRAMP_ID: {"smiles", "modified", "sequence"}}``, empty for a pre-5.0 dump.
    """
    from sample.database_activities import load_dramp_smiles

    root = base_dir or str(DATA_DIR / "antimicrobial_databases")
    try:
        return load_dramp_smiles(root)
    except Exception as e:                            # noqa: BLE001 -- additive; never fatal
        logger.warning(f"curated DRAMP SMILES unavailable ({type(e).__name__}: {e}); "
                       f"falling back to the sequence-derived and stored values only.")
        return {}


def validate_smiles_column(df: "Any", entry_id_column: str = "entry_id",
                           smiles_column: str = "SMILES",
                           predicted_structures_json: Optional[str] = None,
                           curated: Optional[Dict[str, Dict[str, Any]]] = None,
                           curated_id_column: str = "DRAMP") -> Dict[str, int]:
    """
    Replaces or nulls the SMILES column of a dataset in place.

    :param df: The dataset; modified in place, column by column, so the caller's index and other
        columns are untouched.
    :param entry_id_column: Column carrying the peptide id.
    :param smiles_column: Column to validate.
    :param predicted_structures_json: Source for the sequence-derived replacement.
    :param curated: Curated chemistry from a source database, as
        :func:`load_curated_smiles` returns it. ``None`` loads it; pass ``{}`` to switch the
        tier off.
    :param curated_id_column: Column holding the id that ``curated`` is keyed by. A dataset
        without this column simply gets no curated values -- no error, because the column is
        absent from any dataset built before DRAMP ids were carried through.
    :returns: Counts per outcome: ``replaced_from_structure``, ``replaced_from_curated``,
        ``kept``, ``rejected``, ``already_empty``, and ``curated_preferred_over_structure``.

    **Three tiers, and the order is not simply "most computed wins".**

    1. A **sequence-derived** structure (:mod:`sample.folding`), read off a real 3D conformer,
       is round-trippable and connected by construction.
    2. A **curated** structure from DRAMP 5.0 (Nucleic Acids Research, doi:10.1093/nar/gkag989),
       which annotates 22,298 of its 32,296 entries.
    3. The **stored** value, kept only if :func:`is_plausible_smiles` cannot rule it out.

    Tier 1 beats tier 2 for an unmodified peptide. It does **not** for a modified one, and that
    exception is the point of adding tier 2 rather than appending it at the bottom. A derived
    structure is built from the sequence, so it describes a *linear, unmodified* molecule: for a
    cyclic, stapled or terminally-modified peptide it is wrong, and wrong in the worst way
    available here -- it parses, round-trips, and looks exactly as trustworthy as a right answer.
    DRAMP's curated string encodes the modification, and the paper is explicit that it covers
    precisely these cases (unmodified, cyclized/disulfide, terminally modified; more complex
    chemistry is left to a later release).

    So where DRAMP records a modification AND supplies a structure, the curated value wins. Both
    branches are counted, so the log says how often each fired instead of leaving a trust
    decision implicit.
    """
    derived = load_derived_smiles(predicted_structures_json)
    if curated is None:
        curated = load_curated_smiles()

    # A dataset built before DRAMP ids were carried through has no such column; that is a
    # missing join key, not a fault, and it costs only the curated tier.
    curated_ids = (list(df[curated_id_column]) if curated and curated_id_column in df
                   else [None] * len(df[entry_id_column]))

    replaced = from_curated = kept = rejected = already_empty = preferred = 0
    new_values = []
    for entry_id, current, curated_id in zip(df[entry_id_column], df[smiles_column], curated_ids):
        derived_smiles = derived.get(str(entry_id))
        record = curated.get(str(curated_id).strip()) if curated_id is not None else None
        curated_smiles = (record or {}).get("smiles")

        if curated_smiles and (record or {}).get("modified"):
            # The modified case: a structure computed from the sequence cannot carry the
            # modification, so it loses here even though it is the more rigorously derived of
            # the two.
            new_values.append(curated_smiles)
            from_curated += 1
            if derived_smiles:
                preferred += 1
            continue

        if derived_smiles:
            new_values.append(derived_smiles)
            replaced += 1
            continue

        if curated_smiles:
            new_values.append(curated_smiles)
            from_curated += 1
            continue

        current_text = "" if current is None else str(current)
        # pandas reads a blank CSV cell as float NaN; comparing text avoids importing pandas here
        # just to call isna().
        if not current_text or current_text.lower() == "nan":
            new_values.append(current)
            already_empty += 1
            continue

        if is_plausible_smiles(current_text):
            new_values.append(current)
            kept += 1
        else:
            new_values.append("")
            rejected += 1

    df[smiles_column] = new_values
    summary = {"replaced_from_structure": replaced, "replaced_from_curated": from_curated,
               "kept": kept, "rejected": rejected, "already_empty": already_empty,
               "curated_preferred_over_structure": preferred}
    logger.info(f"SMILES column: {replaced} replaced with sequence-derived structures, "
               f"{from_curated} with curated DRAMP structures ({preferred} of them in "
               f"preference to a derived structure, because DRAMP records the entry as "
               f"modified), {kept} kept as plausible, {rejected} rejected as unparseable, "
               f"{already_empty} already empty.")
    return summary


def validate_smiles_file(dataset_csv: str, output_csv: Optional[str] = None,
                         entry_id_column: str = "entry_id", smiles_column: str = "SMILES",
                         predicted_structures_json: Optional[str] = None,
                         curated: Optional[Dict[str, Dict[str, Any]]] = None,
                         curated_id_column: str = "DRAMP") -> Dict[str, int]:
    """
    Runs :func:`validate_smiles_column` over a CSV file.

    :param dataset_csv: Input dataset, e.g. ``final_dataset.csv``.
    :param output_csv: Where to write the result; overwrites ``dataset_csv`` when omitted.
    :param entry_id_column: Column carrying the peptide id.
    :param smiles_column: Column to validate.
    :param predicted_structures_json: Source for the sequence-derived replacement.
    :param curated: Curated chemistry, as :func:`load_curated_smiles` returns it; loaded when
        omitted, switched off with ``{}``.
    :param curated_id_column: Column holding the id ``curated`` is keyed by.
    :returns: The same summary as :func:`validate_smiles_column`.
    """
    import pandas as pd

    df = pd.read_csv(dataset_csv, low_memory=False)
    if smiles_column not in df.columns:
        logger.error(f"No {smiles_column!r} column in {dataset_csv}; nothing to validate.")
        # Every key the full summary carries, so a caller reading one of them does not get a
        # KeyError on the no-op path.
        return {"replaced_from_structure": 0, "replaced_from_curated": 0, "kept": 0,
                "rejected": 0, "already_empty": 0, "curated_preferred_over_structure": 0}

    summary = validate_smiles_column(df, entry_id_column, smiles_column,
                                     predicted_structures_json, curated=curated,
                                     curated_id_column=curated_id_column)
    df.to_csv(output_csv or dataset_csv, index=False)
    return summary
