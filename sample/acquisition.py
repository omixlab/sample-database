"""
Incremental acquisition: find the peptides a fresh pipeline run added, and place them.

``sample.incremental.update_families`` already solves the hard half of "add new peptides without
disturbing the existing corpus" -- it inherits or mints ``SAMPLE_FAM_*`` clusters for a batch of
``{id: sequence}`` without renumbering anything. What it deliberately does not do is decide *which*
sequences are new; that is acquisition (fresh papers, refreshed external databases, re-extraction),
and by the time it reaches this module it is nothing more than steps the batch pipeline already
runs, each already able to skip what it has done before:

* Steps 1-6 (:mod:`sample.acquisition_wrappers` command wiring in ``sample/cli.py``: ``pubmed``,
  ``scopus``, ``download-pdfs``, ``google-patents``, ``databases``, ``import-anticancer``) fetch
  new papers and refresh external databases -- each already resumable/idempotent on its own.
* Steps 7-19 (``merge``, ``extract``, ``external-enrich``, ``revise``, ``extract-repair``,
  ``merge-extractions``) turn those into peptide records. ``merge_database_and_extractions``
  (step 19) already deduplicates by sequence and assigns the next free ``SAMPLE*`` id to anything
  genuinely new, so its output consensus CSV is a strict superset of the previous run's: existing
  ids keep their id, new sequences get a new one.

So the one piece of real logic this module adds is small: read that consensus CSV, and diff each
of its ``entry_id`` values against what the *live release database* already has. Anything present in
the CSV but absent from the database is a candidate this run added since the database was built --
hand exactly those to :func:`sample.incremental.update_families`.

**What this module deliberately does not do**: load the new rows into the live SQLite database,
generate their descriptors/3D structures, or rebuild bibliography/embeddings (steps 20-42, 44).
Those are full-dataset operations with their own discipline (``sample.assembly``, ``load-sql``) and
are not made incremental here -- running them against a release database while it is also serving
traffic is a deliberate, reviewed action, not something to auto-chain onto a network-bound fetch.
"""
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from sample import families

logger = logging.getLogger("sample.acquisition")

#: Where step 19 (``merge-extractions``) leaves the running consensus: database peptides plus
#: everything extracted from literature and patents, deduplicated by sequence.
DEFAULT_CONSENSUS_CSV = os.path.join(families.DATA_DIR, "sample_raw_release",
                                     "merged_from_database.csv")


def find_new_peptides(consensus_csv: str = DEFAULT_CONSENSUS_CSV,
                      db_path: Optional[str] = None) -> Dict[str, str]:
    """
    Diffs the pipeline's running consensus against the live release database.

    :param consensus_csv: Step 19's output -- ``entry_id``/``sequence`` (plus metadata) for the
        whole corpus as the pipeline currently sees it.
    :param db_path: The release database to diff against; the project default when omitted.
    :returns: ``{entry_id: sequence}`` for rows in the consensus CSV whose ``entry_id`` the release
        database does not yet have. Empty (never an error) when the consensus CSV does not exist
        yet -- there is nothing to diff before step 19 has run once.
    """
    if not os.path.exists(consensus_csv):
        logger.info(f"No consensus CSV at {consensus_csv} yet; nothing to diff.")
        return {}

    from sample import db as release_db

    known_ids = set()
    connection = release_db.get_connection(db_path, shared=False)
    try:
        rows = connection.execute("SELECT entry_id FROM peptides").fetchall()
    finally:
        connection.close()
    known_ids = {str(row["entry_id"]) for row in rows}

    df = pd.read_csv(consensus_csv, dtype=str)
    if "entry_id" not in df.columns or "sequence" not in df.columns:
        logger.warning(f"{consensus_csv} has no entry_id/sequence columns; nothing to diff.")
        return {}

    new_peptides: Dict[str, str] = {}
    for _, row in df.iterrows():
        # pandas reads an empty cell as NaN (a float) even under dtype=str; str(nan) is the
        # truthy string "nan", so this must be checked before falling back to "" (the same class
        # of bug fixed in sample.cycpeptmpdb's loader).
        raw_id, raw_seq = row.get("entry_id"), row.get("sequence")
        entry_id = "" if pd.isna(raw_id) else str(raw_id).strip()
        sequence = "" if pd.isna(raw_seq) else str(raw_seq).strip().upper()
        if not entry_id or not sequence or entry_id in known_ids:
            continue
        new_peptides[entry_id] = sequence
    return new_peptides


def run_incremental_update(consensus_csv: str = DEFAULT_CONSENSUS_CSV,
                           families_json: Optional[str] = None,
                           db_path: Optional[str] = None,
                           dry_run: bool = False) -> Dict[str, Any]:
    """
    The acquisition driver's final step: diff, then hand the result to
    :func:`sample.incremental.update_families`.

    Everything upstream of this call (papers, external databases, extraction, merge) is orchestrated
    by the existing pipeline steps -- see ``make incremental-update``, which chains them and calls
    this as its last action. This function itself does no downloading; it is safe to call directly
    against whatever the consensus CSV currently holds.

    :param consensus_csv: Step 19's output.
    :param families_json: The family map to read and (unless ``dry_run``) update.
    :param db_path: The release database new-ness is measured against.
    :param dry_run: Report the assignment without writing ``families.json``.
    :returns: A summary combining the diff count with
        :func:`sample.incremental.update_families`'s own summary (``inherited``, ``new_clusters``,
        ``skipped``, ...). ``{"new_peptides": 0}`` when the diff finds nothing to place.
    """
    from sample.incremental import update_families

    new_peptides = find_new_peptides(consensus_csv, db_path=db_path)
    if not new_peptides:
        logger.info("No new peptides since the last release build; nothing to place.")
        return {"new_peptides": 0}

    logger.info(f"{len(new_peptides)} new peptide(s) since the last release build; placing them "
                f"into the existing family map{' (dry run)' if dry_run else ''}.")
    summary = update_families(new_peptides, families_json=families_json, db_path=db_path,
                              apply=not dry_run)
    summary["new_peptides"] = len(new_peptides)
    return summary
