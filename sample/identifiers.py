"""
Cross-database identity: UniProt and PDB accessions for a peptide.

``activities.csv`` carries a ``cross_references`` column -- ``uniprot:P81941 | pdb:1MAG`` --
resolved during step 19 (:func:`sample.merge.build_integrated_activity_table`) from the source
databases' own cross-reference fields and step 12's recovered external-database hits. It never
reached ``peptides.csv`` or the release database: step 20 onward reads
``merged_from_database.csv``, a different lineage that does not carry the column at all, so a
UniProt or PDB accession that was successfully resolved was simply discarded before the site could
show it.

**Why this reads ``activities.csv`` rather than re-deriving from the source databases.** Unlike
:mod:`sample.provenance` and :mod:`sample.modifications`, which read DBAASP directly because
``activities.csv`` does not carry the fact they need, ``cross_references`` is already resolved and
validated there -- :func:`sample.external_links.normalize_cross_references` is the same parser
:mod:`sample.merge` calls on the way in, so re-parsing it here is one join, not a second harvest.

PubChem and ChEMBL tokens also appear in this column, but are not surfaced through this module:
the peptide page's "Assay Archives" panel already links them, sourced independently through step
37 (:mod:`sample.bioactivity`), and duplicating the same accession from two provenance paths would
read as two separate confirmations of one fact.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

from sample.config import DATA_DIR

logger = logging.getLogger("sample.identifiers")

#: Namespaces surfaced through this module. UniProt and PDB describe the molecule's identity;
#: PubChem and ChEMBL describe assay archives and belong to the bioactivity panel instead.
IDENTITY_NAMESPACES = ("uniprot", "pdb")


def harvest_cross_references(activities_csv: Optional[str] = None) -> Dict[str, str]:
    """
    Reads UniProt/PDB accessions per sequence from the activity table.

    :param activities_csv: Path to the wide activity table; the packaged release location when
        omitted.
    :returns: Sequence -> raw ``cross_references`` cell, restricted to rows carrying at least one
        UniProt or PDB token. The raw cell is kept (not pre-parsed) so the loader can run the same
        :func:`~sample.external_links.cross_reference_links` the page uses, rather than
        maintaining a second representation that could drift from it.
    """
    import pandas as pd

    from sample.external_links import normalize_cross_references

    activities_csv = activities_csv or os.path.join(
        str(DATA_DIR), "sample_raw_release", "activities.csv")
    if not os.path.exists(activities_csv):
        logger.warning(f"No activity table at {activities_csv}; no cross-references harvested.")
        return {}

    frame = pd.read_csv(activities_csv, low_memory=False, usecols=["sequence", "cross_references"])
    frame = frame[frame["cross_references"].notna() & (frame["cross_references"] != "")]
    frame = frame.drop_duplicates(subset=["sequence"])

    out: Dict[str, str] = {}
    for sequence, blob in zip(frame["sequence"], frame["cross_references"]):
        parsed = normalize_cross_references(blob)
        if any(ns in parsed for ns in IDENTITY_NAMESPACES):
            out[str(sequence).upper().strip()] = str(blob)

    logger.info(f"Cross-references: {len(out)} sequences carry a UniProt or PDB accession.")
    return out


def build_cross_references(output_path: Optional[str] = None,
                           activities_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Harvests cross-references and writes them for the loader.

    :param output_path: Where to write; ``data/cross_references.json`` when omitted.
    :param activities_csv: Source activity table.
    :returns: A summary of what was written.
    """
    refs = harvest_cross_references(activities_csv)
    output_path = output_path or os.path.join(str(DATA_DIR), "cross_references.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"cross_references": refs}, handle, ensure_ascii=False, indent=1)

    summary = {"path": output_path, "sequences": len(refs)}
    logger.info(f"Wrote {summary['sequences']} cross-references to {output_path}.")
    return summary
