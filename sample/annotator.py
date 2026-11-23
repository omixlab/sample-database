"""
The annotator: runs a battery of predictions and lookups over one arbitrary submitted sequence.

Distinct from every other search mode this project exposes (BLAST, HMMER, embedding), which each
answer one question quickly. The annotator answers several at once and is not fast enough to be
one of them -- measured on this host: a Pfam scan against the full Pfam-A database (`hmmsearch`,
~20,000 profiles) alone takes ~18s for a single sequence, and HemoPI2 adds another ~3s of
subprocess-spawn overhead. `sample.tasks` runs it as a Celery job for that reason, the same
reasoning BLAST/HMMER/embedding search were each measured *out* of needing (see their own
docstrings) -- this is the one search mode in this project that genuinely needs to be async.

**What actually runs, as of this session -- see `KNOWN_ISSUES.md` for the rest.** Of the six tools
TODO.md originally named for this feature (anticp2, amplify, macrel, pfam, hemopi, toxinpred),
only HemoPI2 is installed and working; macrel's adapter bugs are fixed but it has no environment to
run in yet, toxinpred3 is installed but cannot load its own shipped model, and amplify/anticp2 have
no usable distribution found. Each contributes nothing rather than a fabricated value when
unavailable -- `HemoPI2Predictor.available()` already handles this the same way it does for every
other consumer of the external-predictor framework.
"""
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from sample.config import DATA_DIR

logger = logging.getLogger("sample.annotator")

#: Full Pfam-A HMM database, the same file `sample.families` scans the whole corpus against at
#: ingestion time. Not `hmmpress`-ed for `hmmscan` the way the 519 SAMPLE family profiles are --
#: at 2.2 GB, pressing it roughly doubles the on-disk size for a scan that only needs to run once
#: per annotation request, already comfortably inside the job's async budget without it.
PFAM_HMM_PATH = os.path.join(str(DATA_DIR), "pfam", "Pfam-A.hmm")


def pfam_database_available() -> bool:
    """:returns: Whether :data:`PFAM_HMM_PATH` exists, so a request can fail fast and clearly."""
    return os.path.exists(PFAM_HMM_PATH)


def _pfam_scan(sequence: str, hmm_path: Optional[str] = None,
              max_hits: int = 20) -> List[Dict[str, Any]]:
    """
    Scans one sequence against the full Pfam-A database.

    Runs `hmmsearch` with Pfam-A as the query profiles and the single submitted sequence as the
    target -- the same orientation `sample.families.build_families` already uses for the whole
    corpus, so this is the ingestion-time Pfam assignment answered for a sequence that was never
    part of the release.

    :param sequence: One-letter sequence, already cleaned.
    :param hmm_path: Pfam-A HMM database; :data:`PFAM_HMM_PATH` when omitted.
    :param max_hits: Maximum families to report.
    :returns: One dict per hit (``family``, ``accession``, ``evalue``), best (lowest) E-value
        first, or ``[]`` when the database is missing, the scan fails, or nothing scores.
    """
    hmm_path = hmm_path or PFAM_HMM_PATH
    if not os.path.exists(hmm_path):
        logger.warning(f"No Pfam-A database at {hmm_path}; skipping the Pfam scan.")
        return []

    with tempfile.TemporaryDirectory() as tmp:
        query_path = os.path.join(tmp, "query.fasta")
        domtbl_path = os.path.join(tmp, "hits.domtbl")
        with open(query_path, "w", encoding="utf-8") as handle:
            handle.write(f">query\n{sequence}\n")
        try:
            subprocess.run(
                ["hmmsearch", "--cut_ga", "--domtblout", domtbl_path, hmm_path, query_path],
                capture_output=True, text=True, check=True, timeout=120)
        except Exception as e:
            logger.error(f"Pfam scan failed: {e}")
            return []

        hits: List[Dict[str, Any]] = []
        try:
            with open(domtbl_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("#") or not line.strip():
                        continue
                    # hmmsearch --domtblout column order: target(peptide), accession, tlen,
                    # query(family), accession, qlen, full E-value, full score, ... -- same layout
                    # sample.families.parse_pfam_hits already reads for the corpus-wide scan.
                    fields = line.split()
                    if len(fields) < 7:
                        continue
                    family, accession = fields[3], fields[4]
                    try:
                        evalue = float(fields[6])
                    except ValueError:
                        continue
                    hits.append({
                        "family": family,
                        "accession": accession.split(".")[0],
                        "evalue": evalue,
                    })
        except OSError as e:
            logger.error(f"Could not read Pfam scan output: {e}")
            return []

    hits.sort(key=lambda h: h["evalue"])
    return hits[:max_hits]


def _sample_hits(sequence: str, max_mismatches: int = 2, max_hits: int = 20) -> List[Dict[str, Any]]:
    """
    Finds near-identical peptides already in the release, and whether each carries a literature or
    patent reference.

    :param sequence: One-letter sequence, already cleaned.
    :param max_mismatches: Substitutions tolerated; passed through to
        :func:`sample.similarity.search`.
    :param max_hits: Maximum peptides to report.
    :returns: One dict per hit (``entry_id``, ``identity``, ``mismatches``, ``has_literature``,
        ``has_patent``), closest match first, or ``[]`` when nothing is near enough or the lookup
        fails.
    """
    from sample import db, similarity

    try:
        hits = similarity.search(sequence, max_mismatches=max_mismatches, limit=max_hits)
    except Exception as e:
        logger.error(f"SAMPLE hit lookup failed: {e}")
        return []

    annotated: List[Dict[str, Any]] = []
    for hit in hits:
        peptide = db.get_peptide(hit["entry_id"])
        annotated.append({
            "entry_id": hit["entry_id"],
            "identity": hit["identity"],
            "mismatches": hit["mismatches"],
            "has_literature": bool(peptide and peptide.get("LITERATURE")),
            "has_patent": bool(peptide and peptide.get("PATENT")),
        })
    return annotated


def run_annotation(sequence: str) -> Dict[str, Any]:
    """
    Runs the full annotator battery over one sequence.

    :param sequence: Query sequence, plain or FASTA; headers and line breaks are stripped.
    :returns: ``{"sequence", "hemopi2", "pfam_hits", "sample_hits", "predictors_available"}``.
        ``hemopi2`` is ``None`` when the tool is unavailable, distinguishing "not predicted" from a
        real negative call, the same distinction the rest of the external-predictor framework
        keeps. ``predictors_available`` reports every named tool's status (see the module
        docstring for which are real right now), so a caller can show what did not run and why,
        rather than a silent gap.
    """
    from sample.peptide_predictors import EXTERNAL_PREDICTORS, predictor_availability

    lines = [line.strip() for line in str(sequence).strip().splitlines()]
    cleaned = "".join(l for l in lines if l and not l.startswith(">")).upper().replace(" ", "")

    result: Dict[str, Any] = {
        "sequence": cleaned,
        "hemopi2": None,
        "pfam_hits": [],
        "sample_hits": [],
        "predictors_available": predictor_availability(),
    }
    if not cleaned:
        return result

    hemopi = next((p for p in EXTERNAL_PREDICTORS if p.name == "hemopi2"), None)
    if hemopi is not None and hemopi.available():
        prediction = hemopi.predict([cleaned])[0]
        result["hemopi2"] = prediction or None

    result["pfam_hits"] = _pfam_scan(cleaned)
    result["sample_hits"] = _sample_hits(cleaned)
    return result
