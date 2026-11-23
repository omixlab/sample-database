"""
Structural validation of a ColabFold prediction against the closest real, experimentally-solved
peptide in the PDB, when one exists.

pLDDT and Ramachandran geometry (:mod:`sample.structure_analysis`) both say whether a *prediction*
is internally self-consistent -- they cannot say whether it is *right*. The one check that can is
comparison against a real structure of a close sequence relative, when the PDB happens to hold one.
This module:

1. Searches RCSB's public sequence-search API for PDB entries close to the peptide's own sequence
   (:func:`search_similar_pdb`).
2. Downloads the closest hit above an identity floor and resolves which auth chain it corresponds to
   (:func:`_fetch_reference_chain`).
3. Aligns the ColabFold model onto that chain, guided by a pairwise sequence alignment so residue
   correspondence survives indels between the predicted and reference sequences, then reports
   CA-RMSD over the aligned, structurally-equivalent positions (:func:`superpose`).

This is deliberately a CA-RMSD superposition, not a TM-score: TM-align/US-align are not part of any
environment this project already carries, and RMSD from a sequence-alignment-guided correspondence
is honest, cheap, and needs nothing beyond BioPython. It is a coarser signal than TM-score for very
divergent sequences, which matters less here since a match is only reported at all above an identity
floor where the correspondence is unambiguous.

Most peptides will have no close PDB match at all -- the PDB is small and biased toward peptides
that were solved because they were already interesting, and most of this corpus (~53k sequences) was
never a crystallography target. A peptide simply carrying ``has_match: False`` after a real search is
therefore the expected, common case, not a failure to explain away.

Like :mod:`sample.genomic_context` and :mod:`sample.peptide_annotations`, results are cached in
SQLite (one row per peptide) so the corpus-wide sweep is resumable and the webserver reads a
precomputed answer rather than hitting RCSB on every page view.
"""
import logging
import os
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.pdb_homology")

#: Below this fraction identity, a "hit" is not a validation signal -- it is two unrelated peptides
#: that happen to share some residues, and reporting it as a structural check would be misleading.
DEFAULT_MIN_IDENTITY: float = 0.7

#: RCSB's sequence search will not return meaningful hits below this length -- MMseqs2 (which powers
#: it) needs enough sequence to seed an alignment. Below this, skip the search entirely rather than
#: send a query that will only ever come back empty.
MIN_SEARCHABLE_LENGTH: int = 8

_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
_ENTRY_PDB_URL = "https://files.rcsb.org/download/{entry_id}.pdb"


# --- Thin, individually-mockable network calls -------------------------------------------------
# Kept as separate one-line wrappers (rather than inlined requests.* calls) so tests can monkeypatch
# exactly the network boundary and exercise the real parsing/superposition logic against canned
# responses, without a live call to RCSB.

def _post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    import requests
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_json(url: str, timeout: float = 30.0) -> Dict[str, Any]:
    import requests
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _get_text(url: str, timeout: float = 60.0) -> str:
    import requests
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


# --- Search & fetch ------------------------------------------------------------------------------

def search_similar_pdb(sequence: str, min_identity: float = DEFAULT_MIN_IDENTITY,
                       max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches RCSB for PDB polymer entities close to ``sequence``.

    :param sequence: One-letter peptide sequence.
    :param min_identity: Fraction (0-1) identity floor passed to RCSB as its own cutoff.
    :param max_results: Cap on hits returned.
    :returns: ``[{"pdb_id", "entity_id", "identity", "evalue"}]``, best match first. Empty on no
        hits, a too-short query, or any network/parse failure -- this is an optional enrichment, not
        something that can fail a peptide's page.
    """
    sequence = (sequence or "").strip().upper()
    if len(sequence) < MIN_SEARCHABLE_LENGTH:
        return []
    payload = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1.0,
                "identity_cutoff": min_identity,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "scoring_strategy": "sequence",
            "paginate": {"start": 0, "rows": max_results},
        },
    }
    try:
        data = _post_json(_SEARCH_URL, payload)
    except Exception as e:
        logger.debug(f"RCSB sequence search failed for a {len(sequence)}-residue query: {e}")
        return []

    hits: List[Dict[str, Any]] = []
    for result in (data.get("result_set") or [])[:max_results]:
        identifier = str(result.get("identifier") or "")
        if "_" not in identifier:
            continue
        pdb_id, entity_id = identifier.split("_", 1)
        identity = evalue = None
        for service in result.get("services") or []:
            for node in service.get("nodes") or []:
                context = node.get("match_context") or [{}]
                if context:
                    identity = context[0].get("sequence_identity")
                    evalue = context[0].get("evalue")
        hits.append({"pdb_id": pdb_id.upper(), "entity_id": entity_id,
                     "identity": float(identity) if identity is not None else None,
                     "evalue": float(evalue) if evalue is not None else None})
    return hits


def _fetch_reference_chain(pdb_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Resolves an RCSB polymer-entity hit to an actual chain letter and downloads its structure.

    :param pdb_id: Four-character PDB accession.
    :param entity_id: The polymer entity number within that entry (RCSB's ``entry_entity``).
    :returns: ``{"chain", "pdb_text", "title"}``, or ``None`` when the entity/entry cannot be
        resolved (withdrawn entry, transient RCSB error, unexpected response shape).
    """
    try:
        entity = _get_json(_ENTITY_URL.format(entry_id=pdb_id, entity_id=entity_id))
        chains = ((entity.get("rcsb_polymer_entity_container_identifiers") or {})
                  .get("auth_asym_ids") or [])
        if not chains:
            return None
        title = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
        pdb_text = _get_text(_ENTRY_PDB_URL.format(entry_id=pdb_id))
        return {"chain": chains[0], "pdb_text": pdb_text, "title": title}
    except Exception as e:
        logger.debug(f"Could not resolve/download PDB entity {pdb_id}_{entity_id}: {e}")
        return None


# --- Superposition ---------------------------------------------------------------------------

def _first_chain_id(structure: Any) -> Optional[str]:
    """:returns: The id of the first chain in a parsed structure's first model, or ``None``."""
    model = next(iter(structure))
    chain = next(iter(model), None)
    return chain.id if chain is not None else None


def _chain_ca_atoms(structure: Any, chain_id: str) -> List[Any]:
    """:returns: ``[(one_letter_aa, CA atom)]`` for one chain of a parsed BioPython structure."""
    from Bio.PDB.Polypeptide import three_to_index, index_to_one

    model = next(iter(structure))
    if chain_id not in model:
        return []
    out = []
    for residue in model[chain_id]:
        if "CA" not in residue:
            continue
        try:
            aa = index_to_one(three_to_index(residue.get_resname()))
        except (KeyError, ValueError):
            continue
        out.append((aa, residue["CA"]))
    return out


def superpose(predicted_pdb_path: str, reference_pdb_text: str,
             reference_chain: str) -> Optional[Dict[str, Any]]:
    """
    Superimposes a ColabFold prediction onto one chain of a reference PDB structure.

    Residue correspondence comes from a pairwise sequence alignment, not position-by-position
    pairing, because the two sequences are rarely identical length (the reference is often a longer
    construct, or differs by a few residues from the exact query) -- pairing raw indices would
    silently misalign the moment there is a single indel.

    :param predicted_pdb_path: The ColabFold-predicted structure.
    :param reference_pdb_text: The reference PDB entry's file content (the whole entry, which may
        hold several chains).
    :param reference_chain: The auth_asym_id of the chain to align against.
    :returns: ``{"rmsd", "aligned_residues", "identity"}`` over the aligned CA pairs; ``None`` when
        either structure has too few residues to align meaningfully (fewer than 3 CA pairs -- below
        that, Superimposer's least-squares fit is not meaningful and RMSD is not a real signal).
    """
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.PDB import PDBParser, Superimposer

    predicted = PDBParser(QUIET=True).get_structure("predicted", predicted_pdb_path)
    predicted_chain = _first_chain_id(predicted)
    predicted_residues = _chain_ca_atoms(predicted, predicted_chain) if predicted_chain else []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
        fh.write(reference_pdb_text)
        ref_path = fh.name
    try:
        reference = PDBParser(QUIET=True).get_structure("reference", ref_path)
        reference_residues = _chain_ca_atoms(reference, reference_chain)
    finally:
        os.unlink(ref_path)

    if len(predicted_residues) < 3 or len(reference_residues) < 3:
        return None

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    predicted_seq = "".join(aa for aa, _ in predicted_residues)
    reference_seq = "".join(aa for aa, _ in reference_residues)
    alignment = aligner.align(predicted_seq, reference_seq)[0]

    predicted_atoms, reference_atoms, matches = [], [], 0
    for (p_start, p_end), (r_start, r_end) in zip(*alignment.aligned):
        for offset in range(p_end - p_start):
            p_aa, p_atom = predicted_residues[p_start + offset]
            r_aa, r_atom = reference_residues[r_start + offset]
            predicted_atoms.append(p_atom)
            reference_atoms.append(r_atom)
            if p_aa == r_aa:
                matches += 1

    if len(predicted_atoms) < 3:
        return None

    superimposer = Superimposer()
    superimposer.set_atoms(reference_atoms, predicted_atoms)
    return {"rmsd": round(float(superimposer.rms), 2), "aligned_residues": len(predicted_atoms),
            "identity": round(matches / len(predicted_atoms), 3)}


# --- Orchestration ---------------------------------------------------------------------------

def homology_for_peptide(entry_id: str, sequence: str, predicted_pdb_path: str,
                         min_identity: float = DEFAULT_MIN_IDENTITY) -> Dict[str, Any]:
    """
    Full pipeline for one peptide: search, fetch the best hit, superpose.

    :param entry_id: The peptide's release id (used only for logging).
    :param sequence: Its one-letter sequence.
    :param predicted_pdb_path: Its ColabFold PDB.
    :param min_identity: Identity floor for a search hit to be worth fetching at all.
    :returns: ``{"has_match": False}`` when the search found nothing above the identity floor, the
        fetch/superposition failed, or there was no predicted structure to compare; otherwise
        ``{"has_match": True, "pdb_id", "chain", "title", "search_identity", "evalue", "rmsd",
        "aligned_residues", "structural_identity"}``.
    """
    if not predicted_pdb_path or not os.path.exists(predicted_pdb_path):
        return {"has_match": False}

    hits = search_similar_pdb(sequence, min_identity=min_identity)
    for hit in hits:
        reference = _fetch_reference_chain(hit["pdb_id"], hit["entity_id"])
        if reference is None:
            continue
        result = superpose(predicted_pdb_path, reference["pdb_text"], reference["chain"])
        if result is None:
            continue
        return {"has_match": True, "pdb_id": hit["pdb_id"], "chain": reference["chain"],
                "title": reference["title"], "search_identity": hit["identity"],
                "evalue": hit["evalue"], "rmsd": result["rmsd"],
                "aligned_residues": result["aligned_residues"],
                "structural_identity": result["identity"]}
    return {"has_match": False}


# --- Persistent store, mirroring sample.peptide_annotations -----------------------------------

def default_db_path() -> str:
    """:returns: Path to the PDB-homology database (``SAMPLE_PDB_HOMOLOGY_DB`` overrides)."""
    override = os.environ.get("SAMPLE_PDB_HOMOLOGY_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "pdb_homology.sqlite")


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
    """Creates the homology table if absent. :returns: the database path."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pdb_homology ("
            "entry_id TEXT PRIMARY KEY, has_match INTEGER NOT NULL, result TEXT NOT NULL, "
            "computed_at REAL NOT NULL)")
        conn.commit()
    finally:
        conn.close()
    return db_path or default_db_path()


def save(entry_id: str, result: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Upserts one peptide's homology result."""
    import json

    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO pdb_homology (entry_id, has_match, result, computed_at) "
            "VALUES (?,?,?,?)",
            (entry_id, int(bool(result.get("has_match"))), json.dumps(result), time.time()))
        conn.commit()
    finally:
        conn.close()


def get(entry_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """:returns: The stored homology result for a peptide, or ``None`` when never computed."""
    import json

    if not os.path.exists(db_path or default_db_path()):
        return None
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT result FROM pdb_homology WHERE entry_id = ?", (entry_id,)).fetchone()
    finally:
        conn.close()
    return json.loads(row["result"]) if row else None


def processed_ids(db_path: Optional[str] = None) -> set:
    """:returns: entry_ids already searched (for a resumable batch)."""
    if not os.path.exists(db_path or default_db_path()):
        return set()
    conn = get_connection(db_path)
    try:
        return {r["entry_id"] for r in conn.execute("SELECT entry_id FROM pdb_homology")}
    finally:
        conn.close()


def build_all(structures_dir: Optional[str] = None, db_path_release: Optional[str] = None,
             store_path: Optional[str] = None, limit: Optional[int] = None,
             resume: bool = True, min_identity: float = DEFAULT_MIN_IDENTITY,
             sleep_seconds: float = 0.5) -> Dict[str, Any]:
    """
    Runs the PDB-homology search for every folded release peptide not yet processed.

    Only peptides with a ColabFold PDB on disk are candidates -- there is nothing to superpose
    otherwise. A short sleep between peptides keeps this polite to RCSB's public API; there is no
    API key or rate-limit contract to lean on instead.

    :param structures_dir: Where ColabFold PDBs live; the release default when omitted.
    :param db_path_release: The release database to read peptide ids/sequences from.
    :param store_path: The homology database to write.
    :param limit: Process at most this many *new* peptides this run.
    :param resume: Skip peptides already processed (the default).
    :param min_identity: Passed through to :func:`homology_for_peptide`.
    :param sleep_seconds: Delay between peptides.
    :returns: ``{"processed", "matched", "total_stored"}``.
    """
    from sample import db as release_db
    from sample.config import DATA_DIR

    structures_dir = structures_dir or os.path.join(DATA_DIR, "sample_enriched_release", "structures")
    init_db(store_path)
    already = processed_ids(store_path) if resume else set()

    if not os.path.isdir(structures_dir):
        return {"processed": 0, "matched": 0, "total_stored": len(already)}

    folded_ids = {os.path.splitext(f)[0] for f in os.listdir(structures_dir) if f.endswith(".pdb")}
    todo = sorted(folded_ids - already)
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"processed": 0, "matched": 0, "total_stored": len(already)}

    processed = matched = 0
    for entry_id in todo:
        peptide = release_db.get_peptide(entry_id, db_path=db_path_release) or {}
        sequence = str(peptide.get("sequence") or "").strip()
        if not sequence:
            continue
        pdb_path = os.path.join(structures_dir, f"{entry_id}.pdb")
        result = homology_for_peptide(entry_id, sequence, pdb_path, min_identity=min_identity)
        save(entry_id, result, store_path)
        processed += 1
        if result.get("has_match"):
            matched += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)

    logger.info(f"pdb-homology-sweep: searched {processed} peptide(s) this run "
                f"({matched} with a PDB match); store now holds {len(already) + processed}.")
    return {"processed": processed, "matched": matched, "total_stored": len(already) + processed}
