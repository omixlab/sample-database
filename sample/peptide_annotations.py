"""
Pre-computed, comprehensive per-peptide annotation.

Assembles, for every release peptide and stores in SQLite so the peptide page shows it without
recomputing:

* **External predictors** -- macrel (AMP + hemolysis), HemoPI2 (hemolysis), AntiCP2 (anticancer),
  AlgPred2 (allergenicity), ToxinPred3 (toxicity, no working adapter -- see below). AMPlify (AMP)
  was retired 2026-08-26 after a sustained upstream crash rate in production (see
  :data:`sample.peptide_predictors.EXTERNAL_PREDICTORS`); its columns simply stop appearing.
  Run in one batch over the whole corpus
  (:func:`sample.peptide_predictors.enrich_with_predictors`), because each tool amortises its model
  load across many sequences; a peptide simply carries whichever predictors' environments were built.
* **Pfam domains and the SAMPLE family / cluster** -- read from the release database (already assigned
  by the families step), so the annotation gathers them rather than recomputing.
* **Near-identical peptides** -- other release peptides within a couple of substitutions
  (:func:`sample.similarity.search`).
* **Family HMM hits** -- the query scanned against SAMPLE's own profile HMMs with ``hmmscan``
  (:func:`sample.api.common.run_hmm_search`).

**Circular peptides are handled explicitly.** Every one of the sequence-based predictors, the
near-identity search and the HMM scan assume a *linear* one-letter sequence; a head-to-tail
macrocycle has no such linear reading, and scoring it as if it did is a guess. So a peptide known to
be cyclic (its :mod:`sample.cyclization` topology, or membership of the cyclic
:mod:`sample.cycpeptmpdb` source) carries ``is_circular = True`` and a ``circular_caveat`` on the
annotation, and its sequence-based results are labelled approximate rather than presented as if the
molecule were linear.
"""
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.peptide_annotations")

#: Predictor column -> a friendly label, for the ones enrich_with_predictors may populate.
PREDICTOR_FIELDS: Dict[str, str] = {
    "macrel_amp_probability": "macrel AMP probability",
    "macrel_is_amp": "macrel AMP call",
    "macrel_hemolytic_probability": "macrel hemolysis probability",
    "hemopi2_score": "HemoPI2 hemolysis score",
    "hemopi2_is_hemolytic": "HemoPI2 hemolysis call",
    "amplify_score": "AMPlify AMP score",
    "amplify_is_amp": "AMPlify AMP call",
    "anticp_score": "AntiCP2 anticancer score",
    "anticp_is_anticancer": "AntiCP2 anticancer call",
    "algpred2_score": "AlgPred2 allergenicity score",
    "algpred2_is_allergen": "AlgPred2 allergenicity call",
    # ToxinPred3 has no working adapter: its shipped model pickle is incompatible with every
    # scikit-learn version tried (1.1-1.6, see KNOWN_ISSUES.md) -- an upstream packaging defect, not
    # something this project's env pin can fix. Left named here so the field appears the moment an
    # adapter starts populating it, rather than needing this map edited too.
    "toxinpred3_score": "ToxinPred3 toxicity score",
    "toxinpred3_is_toxic": "ToxinPred3 toxicity call",
}

CIRCULAR_CAVEAT = (
    "This peptide is circular (head-to-tail). The sequence-based predictions and hits below are "
    "computed on its linear sequence and are approximate for a cyclic backbone.")


def default_db_path() -> str:
    """:returns: Path to the annotation database (``SAMPLE_PEPTIDE_ANNOTATIONS_DB`` overrides)."""
    override = os.environ.get("SAMPLE_PEPTIDE_ANNOTATIONS_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "peptide_annotations.sqlite")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """:returns: A WAL-mode connection with row access by name."""
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Optional[str] = None) -> str:
    """Creates the annotation table if absent. :returns: the database path."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS peptide_annotations ("
            "entry_id TEXT PRIMARY KEY, is_circular INTEGER NOT NULL DEFAULT 0, "
            "annotation TEXT NOT NULL, computed_at REAL NOT NULL)")
        conn.commit()
    finally:
        conn.close()
    return db_path or default_db_path()


def _predictor_values(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param row: A peptide's record/columns (post-predictor-enrichment).
    :returns: Only the predictor fields that are present and not null, labelled -- so the annotation
        shows what actually ran and omits (rather than zero-fills) what did not.
    """
    import math

    out: Dict[str, Any] = {}
    for field, label in PREDICTOR_FIELDS.items():
        value = row.get(field)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        out[field] = {"label": label, "value": number}
    return out


def _near_identity_hits(sequence: str, self_entry_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """:returns: Up to ``limit`` release peptides within a couple of substitutions, self excluded."""
    from sample import similarity

    try:
        hits = similarity.search(sequence, max_mismatches=2, limit=limit + 1)
    except Exception as exc:  # noqa: BLE001 -- a search failure must not sink the whole annotation
        logger.debug("near-identity search failed for %s: %s", self_entry_id, exc)
        return []
    return [h for h in hits if h.get("entry_id") != self_entry_id][:limit]


def _hmm_hits(sequence: str) -> List[Dict[str, Any]]:
    """:returns: Family-HMM (`hmmscan`) hits for the sequence, or ``[]`` when no HMM database exists."""
    from sample.api.common import family_hmm_database_path, run_hmm_search

    db = family_hmm_database_path()
    if not db:
        return []
    try:
        return run_hmm_search(sequence, db)
    except Exception as exc:  # noqa: BLE001
        logger.debug("HMM scan failed: %s", exc)
        return []


def annotate_one(entry_id: str, sequence: str, is_circular: bool,
                 predictor_row: Optional[Dict[str, Any]] = None,
                 family: Optional[Dict[str, Any]] = None,
                 pfam_domains: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Assembles one peptide's annotation.

    :param entry_id: The peptide's release id.
    :param sequence: Its one-letter sequence.
    :param is_circular: Whether the backbone is a head-to-tail cycle -- gates the caveat.
    :param predictor_row: The peptide's columns after predictor enrichment (predictor fields read
        from it); ``None`` when no predictors have run.
    :param family: The peptide's ``{family_id, family_name, source}`` (SAMPLE cluster or Pfam family).
    :param pfam_domains: Pfam accessions assigned to the peptide (its ``all_domains``).
    :returns: The annotation dict -- predictors, Pfam domains, family/cluster, near-identity hits,
        HMM hits, ``is_circular`` and, when circular, a ``circular_caveat``.
    """
    annotation: Dict[str, Any] = {
        "entry_id": entry_id,
        "is_circular": bool(is_circular),
        "predictors": _predictor_values(predictor_row or {}),
        "pfam_domains": list(pfam_domains or []),
        "family": family or None,
        "near_identity_hits": _near_identity_hits(sequence, entry_id),
        "hmm_hits": _hmm_hits(sequence),
    }
    if is_circular:
        annotation["circular_caveat"] = CIRCULAR_CAVEAT
    return annotation


def save(entry_id: str, annotation: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Upserts one peptide's annotation."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO peptide_annotations (entry_id, is_circular, annotation, "
            "computed_at) VALUES (?,?,?,?)",
            (entry_id, int(bool(annotation.get("is_circular"))), json.dumps(annotation),
             time.time()))
        conn.commit()
    finally:
        conn.close()


def get(entry_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """:returns: The stored annotation for a peptide, or ``None`` when none has been computed."""
    if not os.path.exists(db_path or default_db_path()):
        return None
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT annotation FROM peptide_annotations WHERE entry_id = ?", (entry_id,)).fetchone()
    finally:
        conn.close()
    return json.loads(row["annotation"]) if row else None


def processed_ids(db_path: Optional[str] = None) -> set:
    """:returns: entry_ids already annotated (for a resumable batch)."""
    if not os.path.exists(db_path or default_db_path()):
        return set()
    conn = get_connection(db_path)
    try:
        return {r["entry_id"] for r in conn.execute(
            "SELECT entry_id FROM peptide_annotations")}
    finally:
        conn.close()


def build_all(db_path_release: Optional[str] = None, store_path: Optional[str] = None,
              limit: Optional[int] = None, resume: bool = True,
              run_predictors: bool = True) -> Dict[str, Any]:
    """
    Pre-computes and stores the annotation for every release peptide.

    Predictors run **once, in batch** over the whole corpus (amortising each tool's model load);
    a peptide whose predictor environments were not built simply carries fewer predictor fields.
    The per-peptide part (family/Pfam gather, near-identity search, HMM scan) then runs for each and
    is stored, skipping peptides already annotated so the batch is resumable and slice-able
    (``limit``).

    :param db_path_release: The release database to read peptides from; the default when omitted.
    :param store_path: The annotation database to write; the default when omitted.
    :param limit: Annotate at most this many *new* peptides this run.
    :param resume: Skip peptides already annotated (the default).
    :param run_predictors: Run the external predictors in batch first; ``False`` annotates from
        whatever predictor columns are already on the records (or none).
    :returns: A summary (``annotated`` this run, ``total_stored``).
    """
    import pandas as pd

    from sample import db as release_db

    init_db(store_path)
    already = processed_ids(store_path) if resume else set()

    connection = release_db.get_connection(db_path_release, shared=False)
    try:
        rows = connection.execute(
            "SELECT entry_id, sequence, SMILES FROM peptides WHERE COALESCE(is_decoy, 0) = 0"
        ).fetchall()
    finally:
        connection.close()

    todo = [dict(r) for r in rows if r["entry_id"] not in already and str(r["sequence"] or "").strip()]
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"annotated": 0, "total_stored": len(already)}

    frame = pd.DataFrame(todo)
    if run_predictors:
        try:
            from sample.peptide_predictors import enrich_with_predictors
            frame = enrich_with_predictors(frame, sequence_column="sequence")
        except Exception as exc:  # noqa: BLE001 -- predictors are optional; annotate the rest
            logger.warning("Predictor enrichment unavailable; annotating without it: %s", exc)

    annotated = 0
    for record in frame.to_dict("records"):
        entry_id = str(record["entry_id"])
        peptide = release_db.get_peptide(entry_id, db_path=db_path_release) or {}
        topology = peptide.get("topology") or {}
        is_circular = bool(topology.get("is_cyclic"))
        family = peptide.get("family")
        pfam_domains = (family or {}).get("all_domains") if isinstance(family, dict) else None
        annotation = annotate_one(entry_id, str(record["sequence"]), is_circular,
                                  predictor_row=record, family=family, pfam_domains=pfam_domains)
        save(entry_id, annotation, store_path)
        annotated += 1

    logger.info("Annotated %d peptide(s) this run; store now holds %d.",
                annotated, len(already) + annotated)
    return {"annotated": annotated, "total_stored": len(already) + annotated}
