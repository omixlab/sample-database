"""
Near-identity peptide search, backed by PepMatch.

The search page offered BLAST, which is the wrong tool for this dataset's shape. Half of these
sequences are under 20 residues, and BLAST's model -- seed on a shared word, extend, score with
statistics derived for long sequences -- degrades badly there: an E-value over a 15-residue query
is not meaningful, and the default seeding misses genuine one-substitution analogues outright.
That matters because "find me everything within a couple of substitutions of this peptide" is the
central question for a database that is largely analogue series.

PepMatch answers it directly by indexing k-mers and searching for an exact mismatch count, which
is both the right model at this length and directly interpretable: *differs at two positions*
needs no statistical caveat. BLAST is kept for the cases it is good at -- longer sequences and
remote similarity -- so the two are offered as separate modes rather than one replacing the other.

The same k-mer seeding constraint as :mod:`sample.families` applies: guaranteeing a hit at ``e``
mismatches requires the query to be at least ``k * (e + 1)`` residues, so ``k`` is chosen from the
query's own length rather than left at a default that silently under-reports for short queries.
"""
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.similarity")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INDEX_DIR = os.path.join(DATA_DIR, "similarity")

#: Default mismatch budget. Two substitutions is the range in which peptides are still routinely
#: described as analogues of one another.
DEFAULT_MAX_MISMATCHES: int = 2

#: Hard cap, so a request cannot ask for a scan that returns most of the database.
MAX_MISMATCHES_ALLOWED: int = 5


def index_path(k: int) -> str:
    """
    :param k: k-mer size.
    :returns: Path of the PepMatch index for that k.
    """
    return os.path.join(INDEX_DIR, f"peptides_{k}mers.pepidx")


def peptides_fasta() -> str:
    """
    :returns: Path to the FASTA of real peptides, written from the release database on demand.
    """
    return os.path.join(INDEX_DIR, "peptides.fasta")


def export_peptides(db_path: Optional[str] = None, force: bool = False) -> str:
    """
    Writes the real peptides to FASTA for indexing.

    :param db_path: Release database; the default when omitted.
    :param force: Rewrite even if the file exists.
    :returns: Path to the FASTA.
    :raises FileNotFoundError: When the release database has not been built.
    """
    import re

    from sample import db as release_db

    path = peptides_fasta()
    if os.path.exists(path) and not force:
        return path

    os.makedirs(INDEX_DIR, exist_ok=True)
    connection = release_db.get_connection(db_path, shared=False)
    try:
        rows = connection.execute(
            "SELECT entry_id, sequence FROM peptides WHERE COALESCE(is_decoy, 0) = 0").fetchall()
    finally:
        connection.close()

    # Decoys are excluded above, for the reason given in sample.db: they are shuffled sequences,
    # and returning one as "similar to your query" would present a fabrication as a result.
    valid = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZ]+$")
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            sequence = str(row["sequence"] or "").strip().upper()
            if sequence and valid.match(sequence):
                handle.write(f">{row['entry_id']}\n{sequence}\n")
                written += 1

    logger.info(f"Wrote {written} peptides to {path} for similarity search.")
    return path


def ensure_index(k: int, db_path: Optional[str] = None) -> str:
    """
    Builds the PepMatch index for one k-mer size if it is missing.

    :param k: k-mer size.
    :param db_path: Release database.
    :returns: Directory holding the index.
    """
    from pepmatch import Preprocessor

    fasta = export_peptides(db_path)
    if not os.path.exists(index_path(k)):
        logger.info(f"Building similarity index (k={k})...")
        Preprocessor(fasta, proteome_name="peptides",
                     preprocessed_files_path=INDEX_DIR).preprocess(k=k)
    return INDEX_DIR


def choose_k(query_length: int, max_mismatches: int) -> int:
    """
    Picks the largest k-mer size that still guarantees recall for this query.

    :param query_length: Length of the query peptide.
    :param max_mismatches: Mismatches to tolerate.
    :returns: The k-mer size. Larger k is faster and more specific, so this takes the largest that
        satisfies ``k * (max_mismatches + 1) <= query_length``, with a floor of 2.
    """
    return max(2, min(5, query_length // (max_mismatches + 1)))


def search(sequence: str, max_mismatches: int = DEFAULT_MAX_MISMATCHES,
           limit: int = 100, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Finds released peptides within ``max_mismatches`` substitutions of a query.

    :param sequence: The query peptide, one-letter code.
    :param max_mismatches: Substitutions tolerated, capped at :data:`MAX_MISMATCHES_ALLOWED`.
    :param limit: Maximum hits returned.
    :param db_path: Release database.
    :returns: Hits sorted by increasing mismatch count, each with the matched peptide's id,
        sequence, mismatch count and percent identity.
    """
    import re

    from pepmatch import Matcher

    cleaned = "".join(
        line for line in str(sequence or "").strip().splitlines()
        if not line.startswith(">")).upper().replace(" ", "")
    cleaned = re.sub(r"[^A-Z]", "", cleaned)
    if len(cleaned) < 4:
        return []

    budget = max(0, min(int(max_mismatches), MAX_MISMATCHES_ALLOWED))
    k = choose_k(len(cleaned), budget)
    ensure_index(k, db_path)

    frame = Matcher(query=[cleaned], proteome_file=peptides_fasta(),
                    max_mismatches=budget, k=k,
                    preprocessed_files_path=INDEX_DIR,
                    sequence_version=False).match()
    if frame is None or len(frame) == 0:
        return []

    hits: List[Dict[str, Any]] = []
    for row in frame.iter_rows(named=True):
        subject = row.get("Protein ID")
        if not subject:
            continue
        mismatches = int(row.get("Mismatches") or 0)
        hits.append({
            "entry_id": str(subject).rsplit(".", 1)[0] if "." in str(subject) else str(subject),
            "matched_sequence": row.get("Matched Sequence"),
            "mismatches": mismatches,
            "identity": round(100.0 * (len(cleaned) - mismatches) / len(cleaned), 1),
            "query_length": len(cleaned),
        })

    hits.sort(key=lambda h: (h["mismatches"], h["entry_id"]))
    return hits[:limit]
