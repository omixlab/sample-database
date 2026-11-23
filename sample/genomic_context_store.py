"""
Persistent store for RefSeq genomic-context sweeps.

The sweep in :mod:`sample.genomic_context` searches one genome at a time and deletes it before the
next, so peak disk use is one genome. This store makes the *results* just as bounded and, crucially,
**resumable**: each genome's surviving hits are written to SQLite and the genome is recorded as
processed in the same transaction, so a sweep over thousands of genomes that is interrupted (or run
in nightly slices) picks up exactly where it stopped rather than re-downloading everything.

Two tables, keyed by the query peptide's id:

* ``genomic_hits`` — one row per surviving tBLASTn hit (the columns of :class:`sample.genomic_context.Hit`).
* ``genomic_processed`` — one row per (peptide, genome) already searched, with its hit count. This
  is what makes the sweep resumable and what the "genomes searched / with hits" counts are read from,
  so a genome that produced zero hits is still remembered and never re-downloaded.

The webserver reads this store directly (see the peptide page), rebuilding the Krona hierarchy from
the stored hits, so the sunburst reflects whatever the sweep has processed so far.
"""
import os
import sqlite3
import time
from typing import List, Optional, Set

#: Where the sweep results live. ``SAMPLE_GENOMIC_CONTEXT_DB`` overrides it (e.g. to a shared volume).
def default_db_path() -> str:
    """:returns: Path to the genomic-context database (``SAMPLE_GENOMIC_CONTEXT_DB`` overrides)."""
    override = os.environ.get("SAMPLE_GENOMIC_CONTEXT_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "genomic_context.sqlite")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Opens a connection with WAL journaling and row access by name.

    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: An open connection.
    """
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Optional[str] = None) -> str:
    """
    Creates the tables and indexes if they do not exist.

    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The database path.
    """
    conn = get_connection(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS genomic_hits (
                peptide_id TEXT NOT NULL,
                genome_accession TEXT NOT NULL,
                contig TEXT,
                identity REAL,
                coverage REAL,
                evalue REAL,
                bitscore REAL,
                sstart INTEGER,
                send INTEGER,
                taxid TEXT,
                organism TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_genomic_hits_peptide ON genomic_hits(peptide_id);

            CREATE TABLE IF NOT EXISTS genomic_processed (
                peptide_id TEXT NOT NULL,
                genome_accession TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                processed_at REAL NOT NULL,
                PRIMARY KEY (peptide_id, genome_accession)
            );
            """)
        conn.commit()
    finally:
        conn.close()
    return db_path or default_db_path()


def processed_accessions(peptide_id: str, db_path: Optional[str] = None) -> Set[str]:
    """
    :param peptide_id: The query peptide's id.
    :param db_path: Database file.
    :returns: The genome accessions already searched for this peptide — the set a resumed sweep
        skips. Empty (not an error) when the database has not been created yet.
    """
    if not os.path.exists(db_path or default_db_path()):
        return set()
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT genome_accession FROM genomic_processed WHERE peptide_id = ?",
            (peptide_id,)).fetchall()
    finally:
        conn.close()
    return {r["genome_accession"] for r in rows}


def record_genome(peptide_id: str, genome_accession: str, hits: List["object"],
                  db_path: Optional[str] = None) -> None:
    """
    Writes one genome's hits and marks it processed, in a single transaction.

    Writing the hits and the processed-marker together is what keeps the sweep consistent under
    interruption: a genome is either fully recorded (hits + marker) or not at all, so a resumed sweep
    never double-counts a genome nor skips one whose hits did not land.

    :param peptide_id: The query peptide's id.
    :param genome_accession: The genome's assembly accession.
    :param hits: The surviving :class:`sample.genomic_context.Hit` objects for this genome.
    :param db_path: Database file.
    """
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO genomic_hits (peptide_id, genome_accession, contig, identity, coverage, "
            "evalue, bitscore, sstart, send, taxid, organism) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(peptide_id, genome_accession, h.contig, h.identity, h.coverage, h.evalue,
              h.bitscore, h.sstart, h.send, h.taxid, h.organism) for h in hits])
        conn.execute(
            "INSERT OR REPLACE INTO genomic_processed (peptide_id, genome_accession, hit_count, "
            "processed_at) VALUES (?,?,?,?)",
            (peptide_id, genome_accession, len(hits), time.time()))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_hits(peptide_id: str, db_path: Optional[str] = None) -> List["object"]:
    """
    :param peptide_id: The query peptide's id.
    :param db_path: Database file.
    :returns: Every stored :class:`sample.genomic_context.Hit` for this peptide (empty when none or
        the database does not exist yet).
    """
    from sample.genomic_context import Hit

    if not os.path.exists(db_path or default_db_path()):
        return []
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT genome_accession, contig, identity, coverage, evalue, bitscore, sstart, send, "
            "taxid, organism FROM genomic_hits WHERE peptide_id = ?", (peptide_id,)).fetchall()
    finally:
        conn.close()
    return [Hit(peptide_id=peptide_id, contig=r["contig"], identity=r["identity"],
                coverage=r["coverage"], evalue=r["evalue"], bitscore=r["bitscore"],
                sstart=r["sstart"], send=r["send"], genome_accession=r["genome_accession"],
                taxid=r["taxid"] or "", organism=r["organism"] or "") for r in rows]


def summary(peptide_id: str, db_path: Optional[str] = None) -> dict:
    """
    :param peptide_id: The query peptide's id.
    :param db_path: Database file.
    :returns: ``{"genomes_searched", "genomes_with_hits", "total_hits"}`` for this peptide, read
        from the processed table (so a genome with zero hits still counts as searched).
    """
    if not os.path.exists(db_path or default_db_path()):
        return {"genomes_searched": 0, "genomes_with_hits": 0, "total_hits": 0}
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS searched, "
            "COALESCE(SUM(CASE WHEN hit_count > 0 THEN 1 ELSE 0 END), 0) AS with_hits, "
            "COALESCE(SUM(hit_count), 0) AS total FROM genomic_processed WHERE peptide_id = ?",
            (peptide_id,)).fetchone()
    finally:
        conn.close()
    return {"genomes_searched": int(row["searched"]), "genomes_with_hits": int(row["with_hits"]),
            "total_hits": int(row["total"])}


def clear_peptide(peptide_id: str, db_path: Optional[str] = None) -> None:
    """
    Removes all stored hits and processed markers for a peptide, for a clean re-sweep.

    :param peptide_id: The query peptide's id.
    :param db_path: Database file.
    """
    if not os.path.exists(db_path or default_db_path()):
        return
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM genomic_hits WHERE peptide_id = ?", (peptide_id,))
        conn.execute("DELETE FROM genomic_processed WHERE peptide_id = ?", (peptide_id,))
        conn.commit()
    finally:
        conn.close()
