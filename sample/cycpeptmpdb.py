"""
CycPeptMPDB as a source database: cyclic peptides with membrane-permeability annotations.

CycPeptMPDB (cycpeptmpdb.com) is a curated collection of ~8,500 **cyclic** peptides, each with a
SMILES, its monomer sequence, and a standardised membrane-permeability value (``log Papp``, plus the
underlying PAMPA / Caco2 assays). It was first pulled in to *train* a permeability model, but that
was the wrong use: its macrocyclic, N-methylated, non-natural-residue chemistry is a different world
from this database's linear antimicrobial/anticancer peptides, so a model trained on it predicts
nothing trustworthy for them. Its *data*, though, is worth indexing on its own terms — a large,
well-provenanced set of cyclic peptides and the permeability numbers this project otherwise has none
of.

So this imports CycPeptMPDB as a **source**, normalising each entry to the shape the rest of the
pipeline speaks, with two things made explicit:

* **They are circular.** Every CycPeptMPDB peptide is a head-to-tail macrocycle, so each carries
  ``topology = "head_to_tail"`` — the same label :mod:`sample.cyclization` assigns a genuinely
  circular backbone — rather than being silently treated as linear (which would misstate its mass,
  its structure and its identity, see that module). Circular peptides from the *other* sources are
  already captured by the topology harvest (`build-topology`); this adds a source that is entirely
  circular.
* **Permeability is an annotation, not a bioactivity.** ``log Papp`` is a physicochemical property,
  not an antimicrobial/anticancer activity measured against a target, so it is kept as a permeability
  annotation on the peptide rather than forced into the activity-vs-decoy task table where it would
  invent a meaningless "permeability class" model.

Because these peptides are cyclic with non-standard monomers, they have no plain one-letter sequence
for the sequence-keyed consensus/activity tables; they are therefore indexed as their own normalised
table (``data/cycpeptmpdb/cycpeptmpdb_peptides.csv``), keyed by structure (SMILES), and linked to the
release peptides by structure where they coincide rather than by a sequence string they do not have.
"""
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.cycpeptmpdb")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Where the raw download and the normalised table live.
CYCPEPTMPDB_DIR = os.path.join(DATA_DIR, "cycpeptmpdb")

#: The full peptide table (SMILES + permeability), verified reachable 2026-08-13.
CYCPEPTMPDB_URL = "http://cycpeptmpdb.com/static//download/peptides/CycPeptMPDB_Peptide_All.csv"

#: The topology every CycPeptMPDB peptide has -- a head-to-tail macrocycle, the same label
#: :mod:`sample.cyclization` uses for a closed backbone.
TOPOLOGY = "head_to_tail"

#: Source-database name, as it appears in provenance.
DATABASE = "CycPeptMPDB"

#: The columns of the normalised table.
COLUMNS = ["cycpeptmpdb_id", "name", "source_reference", "database", "topology", "is_cyclic",
           "smiles", "sequence", "monomer_length", "permeability_logpapp", "pampa_logpapp",
           "caco2_logpapp"]


def download(dest_path: Optional[str] = None) -> str:
    """
    Downloads the CycPeptMPDB peptide table.

    :param dest_path: Where to write it; ``data/cycpeptmpdb/CycPeptMPDB_Peptide_All.csv`` when
        omitted.
    :returns: The path written.
    :raises RuntimeError: When the download fails, naming the URL.
    """
    import urllib.request

    dest_path = dest_path or os.path.join(CYCPEPTMPDB_DIR, "CycPeptMPDB_Peptide_All.csv")
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    try:
        urllib.request.urlretrieve(CYCPEPTMPDB_URL, dest_path)
    except Exception as exc:  # noqa: BLE001 -- re-raised with the URL for context
        raise RuntimeError(f"Failed to download CycPeptMPDB from {CYCPEPTMPDB_URL}: {exc}") from exc
    logger.info("Downloaded CycPeptMPDB to %s.", dest_path)
    return dest_path


def _first_column(columns: Dict[str, str], *candidates: str) -> Optional[str]:
    """:returns: The first of ``candidates`` (case-insensitive) present in ``columns``, or None."""
    for candidate in candidates:
        actual = columns.get(candidate.lower())
        if actual:
            return actual
    return None


def load(csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Reads the raw CycPeptMPDB CSV into normalised records.

    :param csv_path: The raw CSV; the packaged download location when omitted.
    :returns: One dict per peptide, in the shape of :data:`COLUMNS`. Rows without a SMILES are
        dropped (the structure is the identity here). Permeability fields are floats or ``None``.
    :raises FileNotFoundError: When the CSV is absent.
    """
    import pandas as pd

    csv_path = csv_path or os.path.join(CYCPEPTMPDB_DIR, "CycPeptMPDB_Peptide_All.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CycPeptMPDB not found at {csv_path}. Download it with "
            f"`sample-cli import-cycpeptmpdb --download`.")

    frame = pd.read_csv(csv_path, low_memory=False)
    columns = {c.lower(): c for c in frame.columns}
    id_col = _first_column(columns, "ID", "CycPeptMPDB_ID")
    smiles_col = _first_column(columns, "SMILES")
    if not smiles_col:
        raise ValueError(f"{csv_path} has no SMILES column.")
    seq_col = _first_column(columns, "Sequence")
    name_col = _first_column(columns, "Original_Name_in_Source_Literature", "Name")
    source_col = _first_column(columns, "Source")
    len_col = _first_column(columns, "Monomer_Length")
    perm_col = _first_column(columns, "Permeability")
    pampa_col = _first_column(columns, "PAMPA")
    caco2_col = _first_column(columns, "Caco2")

    def _num(row: Any, col: Optional[str]) -> Optional[float]:
        if not col:
            return None
        value = pd.to_numeric(row.get(col), errors="coerce")
        return None if pd.isna(value) else float(value)

    def _str(row: Any, col: Optional[str]) -> str:
        # pandas reads an empty cell as NaN (a float), and str(NaN) is the truthy "nan" -- guard it,
        # or an empty SMILES would survive as "nan" and a blank name would print as "nan".
        if not col:
            return ""
        value = row.get(col)
        return "" if value is None or pd.isna(value) else str(value).strip()

    records: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        smiles = _str(row, smiles_col)
        if not smiles:
            continue
        records.append({
            "cycpeptmpdb_id": _str(row, id_col),
            "name": _str(row, name_col),
            "source_reference": _str(row, source_col),
            "database": DATABASE,
            "topology": TOPOLOGY,
            "is_cyclic": True,
            "smiles": smiles,
            "sequence": _str(row, seq_col),
            "monomer_length": _num(row, len_col),
            "permeability_logpapp": _num(row, perm_col),
            "pampa_logpapp": _num(row, pampa_col),
            "caco2_logpapp": _num(row, caco2_col),
        })
    logger.info("CycPeptMPDB: %d cyclic peptides loaded from %s.", len(records), csv_path)
    return records


def build_table(csv_path: Optional[str] = None, output_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Writes the normalised CycPeptMPDB source table.

    :param csv_path: The raw CSV; the packaged download location when omitted.
    :param output_csv: Where to write the normalised table;
        ``data/cycpeptmpdb/cycpeptmpdb_peptides.csv`` when omitted.
    :returns: A summary (``path``, ``rows``, ``with_permeability``, ``with_caco2``).
    """
    import pandas as pd

    output_csv = output_csv or os.path.join(CYCPEPTMPDB_DIR, "cycpeptmpdb_peptides.csv")
    records = load(csv_path)
    table = pd.DataFrame(records, columns=COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)

    with_perm = int(table["permeability_logpapp"].notna().sum()) if len(table) else 0
    with_caco2 = int(table["caco2_logpapp"].notna().sum()) if len(table) else 0
    logger.info("CycPeptMPDB source table: %d cyclic peptides (%d with permeability, %d with Caco2) "
                "-> %s.", len(table), with_perm, with_caco2, output_csv)
    return {"path": output_csv, "rows": len(table), "with_permeability": with_perm,
            "with_caco2": with_caco2}


def inchikey(smiles: str = "", sequence: str = "") -> Optional[str]:
    """
    Computes a molecule's InChIKey -- the canonical structure identifier used to decide when two
    entries are the *same molecule* regardless of how each drew it.

    :param smiles: A SMILES (preferred; it is the exact structure).
    :param sequence: A one-letter sequence, used only when no SMILES is available (the linear
        molecule RDKit builds from it).
    :returns: The InChIKey, or ``None`` when nothing parses.
    """
    from rdkit import Chem

    mol = None
    if smiles:
        mol = Chem.MolFromSmiles(str(smiles))
    if mol is None and sequence:
        cleaned = "".join(ch for ch in str(sequence).upper() if ch.isalpha())
        if cleaned:
            mol = Chem.MolFromSequence(cleaned)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001 -- InChI generation can fail on exotic structures
        return None
    return key or None


def link_to_release(csv_path: Optional[str] = None, db_path: Optional[str] = None,
                    output_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Links CycPeptMPDB entries to release peptides that are the **same molecule**.

    These cyclic peptides have no one-letter sequence to join on, so the link is by *structure*: an
    entry links to a release peptide when their InChIKeys match. Most will not match -- the release
    is mostly linear peptides -- but a genuinely cyclic release peptide (a cyclotide, a known
    macrocyclic drug) that also appears in CycPeptMPDB is linked, carrying CycPeptMPDB's permeability
    over to it. The match is exact structural identity, never a name or a fuzzy guess.

    :param csv_path: The normalised CycPeptMPDB table; the packaged location when omitted.
    :param db_path: The release database; the project default when omitted.
    :param output_csv: Where to write the links; ``data/cycpeptmpdb/cycpeptmpdb_release_links.csv``
        when omitted.
    :returns: A summary (``path``, ``links``, ``cycpeptmpdb_matched``, ``release_matched``).
    """
    import pandas as pd

    from sample import db as release_db

    csv_path = csv_path or os.path.join(CYCPEPTMPDB_DIR, "cycpeptmpdb_peptides.csv")
    output_csv = output_csv or os.path.join(CYCPEPTMPDB_DIR, "cycpeptmpdb_release_links.csv")

    source = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(load(csv_path))

    # Release peptides, indexed by InChIKey (from their SMILES, else their sequence).
    connection = release_db.get_connection(db_path, shared=False)
    try:
        rows = connection.execute(
            "SELECT entry_id, sequence, SMILES FROM peptides WHERE COALESCE(is_decoy, 0) = 0"
        ).fetchall()
    finally:
        connection.close()
    release_by_key: Dict[str, List[str]] = {}
    for row in rows:
        key = inchikey(smiles=str(row["SMILES"] or ""), sequence=str(row["sequence"] or ""))
        if key:
            release_by_key.setdefault(key, []).append(str(row["entry_id"]))

    links: List[Dict[str, Any]] = []
    matched_source: set = set()
    for record in source.to_dict("records"):
        key = inchikey(smiles=str(record.get("smiles") or ""))
        if not key:
            continue
        for entry_id in release_by_key.get(key, []):
            links.append({
                "cycpeptmpdb_id": record.get("cycpeptmpdb_id", ""),
                "name": record.get("name", ""),
                "entry_id": entry_id,
                "inchikey": key,
                "permeability_logpapp": record.get("permeability_logpapp"),
            })
            matched_source.add(str(record.get("cycpeptmpdb_id", "")))

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame(links, columns=["cycpeptmpdb_id", "name", "entry_id", "inchikey",
                                 "permeability_logpapp"]).to_csv(output_csv, index=False)
    release_matched = len({link["entry_id"] for link in links})
    logger.info("CycPeptMPDB -> release: %d structural link(s) (%d CycPeptMPDB entries to %d release "
                "peptides) -> %s.", len(links), len(matched_source), release_matched, output_csv)
    return {"path": output_csv, "links": len(links), "cycpeptmpdb_matched": len(matched_source),
            "release_matched": release_matched}
