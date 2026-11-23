"""
Shared paths, templates and helpers for the web layer.

Everything here was previously module-level state in ``sample/webserver.py``. It is separated from
the routes so the HTML pages and the JSON API can share one implementation of BLAST, structure
rendering and the dataset allow-list, rather than the API growing a second copy that drifts.
"""
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from starlette.templating import Jinja2Templates

# Re-exported so the embedding search mode is reachable the same way as BLAST and HMMER search --
# both defined directly in this module because they wrap an external binary via subprocess; this
# one is pure Python (Chroma + the cached in-process ESM-2 model), so the implementation itself
# lives in sample.embeddings rather than being duplicated here.
from sample.embeddings import embedding_index_path, run_embedding_search  # noqa: F401

logger = logging.getLogger("sample.api")

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR: str = os.path.join(BASE_DIR, "sample", "templates")
STATIC_DIR: str = os.path.join(BASE_DIR, "sample", "static")

#: Jinja environment for the HTML pages. The templates were written for Flask but use no Flask
#: helpers -- no `url_for`, no `get_flashed_messages` -- so they render unchanged here. The only
#: difference is that Starlette requires `request` in the context of every render.
templates: Jinja2Templates = Jinja2Templates(directory=TEMPLATE_DIR)

#: Release artifacts exposed for download, keyed by the symbolic name used in the URL. An
#: allow-list, so a crafted key cannot reach an arbitrary path.
DATASET_PATHS: Dict[str, str] = {
    "consensus_csv": os.path.join(DATA_DIR, "sample_raw_release", "merged_from_database.csv"),
    "consensus_parquet": os.path.join(DATA_DIR, "sample_raw_release",
                                      "merged_from_database.parquet"),
    "enriched_properties_csv": os.path.join(DATA_DIR, "sample_enriched_release", "enriched.csv"),
    "decoys_negative_set": os.path.join(DATA_DIR, "sample_decoy_databases", "decoys.csv"),
    "final_training_dataset": os.path.join(DATA_DIR, "sample_final_release", "final_dataset.csv"),
    "serialized_records_archive": os.path.join(DATA_DIR, "sample_final_release", "sample_records_json_pdb_sdf.zip"),
}

#: Display names for the same artifacts, in the order the Datasets page lists them. Derived from
#: DATASET_PATHS rather than duplicated -- the Flask version kept two dicts that had to agree.
DATASET_LABELS: Dict[str, str] = {
    "consensus_csv": "Consensus CSV",
    "consensus_parquet": "Consensus Parquet",
    "enriched_properties_csv": "Enriched Properties CSV",
    "decoys_negative_set": "Decoys Negative Set",
    "final_training_dataset": "Final Training Dataset",
    "serialized_records_archive": "Serialized Records (JSON, PDB, SDF) ZIP Archive",
}

#: The source databases described on the Databases page.
SOURCE_DATABASES: List[Dict[str, str]] = [
    {"name": "APD (Antimicrobial Peptide Database)",
     "desc": "Focuses on natural antimicrobial peptides with defined activities. Hosted by "
             "University of Nebraska Medical Center.",
     "url": "https://aps.unmc.edu/"},
    {"name": "CAMP (Collection of Anti-Microbial Peptides)",
     "desc": "A comprehensive database of sequence, structural, and activity info for AMPs. "
             "Exposes extensive alignment options.",
     "url": "https://camp.bicnirrh.res.in/"},
    {"name": "DRAMP (Data Repository of Antimicrobial Peptides)",
     "desc": "Focuses on general, patent, and clinical trial peptides with biological activity "
             "parameters.",
     "url": "http://dramp.cpu-bioinfor.org/"},
    {"name": "dbAMP",
     "desc": "Exposes functional activity, categories, species groupings, and chemical metadata "
             "descriptors.",
     "url": "https://ycclab.cuhk.edu.cn/dbAMP/"},
    {"name": "DBAASP",
     "desc": "Database of Bioactive Peptides. Includes rich thermodynamic, toxicity, and "
             "structural metadata parameters.",
     "url": "https://dbaasp.org/"},
]

_VALID_FASTA_SEQUENCE = re.compile(r"^[A-Z*-]+$")


def extraction_dir(kind: str = "literature") -> str:
    """
    :param kind: ``"literature"`` or ``"patent"``.
    :returns: The directory the Extractions viewer should read -- the revised set (step 17) when it
        exists, otherwise the raw extraction, so a partially-run pipeline still shows something.
    """
    base = "literature_data_json" if kind == "literature" else "patent_data_json"
    revised = os.path.join(DATA_DIR, base + "_revised")
    return revised if os.path.isdir(revised) else os.path.join(DATA_DIR, base)


#: Cluster excluded from HMM search because its own profile does not reliably score even its own
#: member sequences -- an over-chained near-identity clustering artifact, not a real family. See
#: KNOWN_ISSUES.md. Reporting a match against it would look like a confident family assignment
#: that means nothing.
_UNRELIABLE_FAMILY_HMMS = frozenset({"SAMPLE_FAM_000001"})


def family_hmm_database_path() -> str:
    """
    :returns: Path to the combined, `hmmpress`-ed family HMM database
        (:func:`sample.family_hmms.combined_hmm_database`), or ``""`` when it has not been built.
    """
    path = os.path.join(DATA_DIR, "families", "hmms", "all_families.hmm")
    return path if os.path.exists(path + ".h3p") else ""


def run_hmm_search(query_seq: str, hmm_db_path: str, max_hits: int = 50) -> List[Dict[str, Any]]:
    """
    Scans a query sequence against SAMPLE's derived family HMM profiles with `hmmscan`.

    Pfam family assignment already runs the other direction (`hmmsearch` at each Pfam family's
    curated gathering threshold, in `sample.families`) and is not duplicated here. This searches
    only the 519 profiles built from this project's own clusters -- the ones with no existing way
    to test a new sequence against them at all.

    :param query_seq: Query sequence, plain or FASTA; headers and line breaks are stripped.
    :param hmm_db_path: Combined, pressed HMM database, from :func:`family_hmm_database_path`.
    :param max_hits: Maximum family hits to report.
    :returns: One dict per hit, ordered by ascending E-value, or ``[]`` when the search fails or
        finds nothing. Hits against :data:`_UNRELIABLE_FAMILY_HMMS` are dropped rather than
        reported.
    """
    lines = [line.strip() for line in str(query_seq).strip().splitlines()]
    cleaned = "".join(l for l in lines if l and not l.startswith(">")).upper().replace(" ", "")
    if not cleaned:
        return []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as handle:
        handle.write(f">query\n{cleaned}\n")
        query_path = handle.name
    tblout_path = query_path + ".tblout"

    try:
        subprocess.run(
            ["hmmscan", "--tblout", tblout_path, "--noali", hmm_db_path, query_path],
            capture_output=True, text=True, check=True, timeout=120)

        hits: List[Dict[str, Any]] = []
        with open(tblout_path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                # Whitespace-separated, but the free-text description column at the end may
                # itself contain spaces -- split with a cap so it survives intact as one field.
                parts = line.split(None, 18)
                if len(parts) < 18:
                    continue
                family_id = parts[0]
                if family_id in _UNRELIABLE_FAMILY_HMMS:
                    continue
                hits.append({
                    "family_id": family_id,
                    "evalue": float(parts[4]), "score": float(parts[5]), "bias": float(parts[6]),
                })
        hits.sort(key=lambda h: h["evalue"])
        return hits[:max_hits]
    except Exception as e:
        logger.error(f"HMM search failed: {e}")
        return []
    finally:
        for path in (query_path, tblout_path):
            if os.path.exists(path):
                os.remove(path)


def init_blast_database() -> str:
    """
    Ensures the consensus sequences are compiled into a BLAST+ protein database.

    :returns: Path to the formatted database, or ``""`` when it could not be built.
    """
    blast_dir = os.path.join(DATA_DIR, "blast")
    os.makedirs(blast_dir, exist_ok=True)

    fasta_path = os.path.join(blast_dir, "peptides.fasta")
    db_path = os.path.join(blast_dir, "peptides_db")

    consensus_csv = os.path.join(DATA_DIR, "sample_raw_release", "merged_from_database.csv")
    if not os.path.exists(consensus_csv):
        consensus_csv = os.path.join(DATA_DIR, "antimicrobial_databases", "merged_source.csv")
    if not os.path.exists(consensus_csv):
        # Nothing to build from. An index built by an earlier run is still better than no search
        # at all, so it is served rather than discarded.
        if os.path.exists(db_path + ".pin"):
            return db_path
        logger.warning("Consensus dataset not found; BLAST search is unavailable.")
        return ""

    # Rebuilt when the index is missing OR older than the consensus dataset it indexes. Existence
    # alone used to be the whole test, which is how a June index of 26,358 sequences came to be
    # served against a 53,614-peptide release: every peptide added since was unreachable by BLAST,
    # and nothing anywhere reported it, because a stale index answers queries perfectly well.
    if (os.path.exists(db_path + ".pin")
            and os.path.getmtime(db_path + ".pin") >= os.path.getmtime(consensus_csv)):
        return db_path

    logger.info("BLAST index is missing or older than the consensus dataset; rebuilding it.")

    # Either the index is absent or it is stale, so any existing FASTA is superseded too.
    if os.path.exists(fasta_path):
        os.remove(fasta_path)

    logger.info(f"Generating BLAST FASTA at {fasta_path}...")
    try:
        import pandas as pd
        from sample.sequence import sequence_to_1_letter

        frame = pd.read_csv(consensus_csv, low_memory=False)
        with open(fasta_path, "w", encoding="utf-8") as fh:
            for _, row in frame.iterrows():
                entry_id = row.get("entry_id")
                sequence = sequence_to_1_letter(str(row.get("sequence")))
                if pd.isna(entry_id) or not sequence.strip():
                    continue
                cleaned = sequence.upper().strip()
                if _VALID_FASTA_SEQUENCE.match(cleaned):
                    fh.write(f">{entry_id}\n{cleaned}\n")
    except Exception as e:
        logger.error(f"Failed to generate BLAST FASTA: {e}")
        return ""

    try:
        subprocess.run(["makeblastdb", "-in", fasta_path, "-dbtype", "prot", "-out", db_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("BLAST database index created.")
    except Exception as e:
        logger.error(f"makeblastdb failed: {e}")
        return ""
    return db_path


def run_blast_search(query_seq: str, db_path: str, max_hits: int = 50) -> List[Dict[str, Any]]:
    """
    Runs a blastp query against the peptide database.

    :param query_seq: Query sequence, plain or FASTA; headers and line breaks are stripped.
    :param db_path: Formatted BLAST database, from :func:`init_blast_database`.
    :param max_hits: Maximum subject sequences to report.
    :returns: One dict per hit, or ``[]`` when the search fails.
    """
    lines = [line.strip() for line in str(query_seq).strip().splitlines()]
    cleaned = "".join(l for l in lines if l and not l.startswith(">")).upper().replace(" ", "")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as handle:
        handle.write(f">query\n{cleaned}\n")
        query_path = handle.name

    try:
        result = subprocess.run(
            ["blastp", "-query", query_path, "-db", db_path,
             "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send "
                        "evalue bitscore",
             "-max_target_seqs", str(max_hits)],
            capture_output=True, text=True, check=True, timeout=300)

        hits: List[Dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            hits.append({
                "subject_id": parts[1], "identity": float(parts[2]),
                "align_len": int(parts[3]), "mismatches": int(parts[4]),
                "gaps": int(parts[5]), "q_start": int(parts[6]), "q_end": int(parts[7]),
                "s_start": int(parts[8]), "s_end": int(parts[9]),
                "evalue": float(parts[10]), "bitscore": float(parts[11]),
            })
        return hits
    except Exception as e:
        logger.error(f"BLAST query failed: {e}")
        return []
    finally:
        if os.path.exists(query_path):
            os.remove(query_path)
