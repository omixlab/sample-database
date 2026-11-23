"""
Profile HMMs for SAMPLE's own derived peptide families.

`sample/families.py` anchors on Pfam first and clusters only what Pfam does not explain by
near-identity (PepMatch). That leaves 519 clusters with 5 or more members — real, recurring
sequence families this project found, that Pfam has no model for. Nothing lets a new sequence be
tested against one of them, because there is no profile to search: a Pfam family is searchable
because Pfam ships an HMM; a SAMPLE cluster is just a list of ids.

This builds exactly that, one profile per qualifying cluster: align the cluster's members with
MAFFT, build a profile HMM from the alignment with HMMER's `hmmbuild`. The result is a `.hmm` file
per cluster, checkable with `hmmscan`/`hmmsearch` the same way a Pfam family already is (see
**C4** — wiring this into a search mode is separate work, tracked in `TODO.md`).

**Why 5 members, not fewer.** A profile built from 2-3 sequences is mostly memorizing them, not
generalizing a family; MAFFT and `hmmbuild` both still run on a smaller set and produce *a* file,
but the resulting model would not usefully classify a new sequence. Matches the threshold
`sample.families` already reports clusters against (2,245 clusters have >=2 members, 519 have
>=5), so this covers the subset that threshold was chosen to describe as real families.

**Why MAFFT rather than a pairwise-only aligner.** Family sizes here range from 5 to 2,559
members (one dominant cluster). MAFFT's `--auto` mode picks its own strategy by input size --
progressive alignment for the large ones, more exhaustive options for the small ones -- so one
invocation handles the whole size range without hand-tuning per cluster.
"""
import glob
import json
import logging
import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sample.config import DATA_DIR

logger = logging.getLogger("sample.family_hmms")

#: Clusters smaller than this are not built -- see the module docstring for why.
MIN_FAMILY_SIZE = 5

#: Where per-cluster FASTA, alignment and HMM files live.
HMM_DIR = os.path.join(str(DATA_DIR), "families", "hmms")


def clustered_families(families_json: Optional[str] = None,
                       min_size: int = MIN_FAMILY_SIZE) -> Dict[str, List[str]]:
    """
    Reads SAMPLE-derived clusters (not Pfam-curated families) with at least `min_size` members.

    :param families_json: Path to `sample.families`' output; the packaged location when omitted.
    :param min_size: Minimum member count to include a cluster.
    :returns: ``cluster_id -> [entry_id, ...]``, largest first.
    """
    families_json = families_json or os.path.join(str(DATA_DIR), "families", "families.json")
    if not os.path.exists(families_json):
        logger.warning(f"No families at {families_json}; nothing to build HMMs from.")
        return {}

    with open(families_json, encoding="utf-8") as handle:
        assignments = json.load(handle) or {}

    members: Dict[str, List[str]] = {}
    for entry_id, record in assignments.items():
        # Pfam-curated families already have a real HMM (Pfam-A itself); building a second,
        # weaker one from this project's own members would compete with the authoritative one
        # rather than fill a gap.
        if record.get("source") != "cluster":
            continue
        members.setdefault(record["family_id"], []).append(entry_id)

    qualifying = {fid: ids for fid, ids in members.items() if len(ids) >= min_size}
    logger.info(f"{len(qualifying)} of {len(members)} clusters have >= {min_size} members "
               f"({sum(len(v) for v in qualifying.values())} peptides total).")
    return dict(sorted(qualifying.items(), key=lambda kv: -len(kv[1])))


def _sequences_for(entry_ids: Iterable[str], dataset_csv: str) -> Dict[str, str]:
    """
    :param entry_ids: Ids to look up.
    :param dataset_csv: Dataset carrying `entry_id` and `sequence` columns.
    :returns: ``entry_id -> sequence`` for whichever of `entry_ids` were found.
    """
    import pandas as pd

    wanted = set(entry_ids)
    out: Dict[str, str] = {}
    for chunk in pd.read_csv(dataset_csv, usecols=["entry_id", "sequence"], chunksize=50_000,
                             low_memory=False):
        hit = chunk[chunk["entry_id"].isin(wanted)]
        out.update(zip(hit["entry_id"], hit["sequence"]))
        if len(out) == len(wanted):
            break
    return out


def build_one_hmm(family_id: str, sequences: Dict[str, str], out_dir: str,
                  mafft_timeout: int = 600) -> Optional[str]:
    """
    Aligns one cluster's sequences and builds its profile HMM.

    :param family_id: Cluster id, used as the HMM's name and the output filename stem.
    :param sequences: ``entry_id -> sequence`` for this cluster's members.
    :param out_dir: Directory for the FASTA, alignment and `.hmm` files.
    :param mafft_timeout: Seconds to allow MAFFT before giving up on one cluster -- so one
        pathological alignment cannot stall the whole batch.
    :returns: Path to the written `.hmm` file, or ``None`` if alignment or building failed.
    """
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, f"{family_id}.fasta")
    aln_path = os.path.join(out_dir, f"{family_id}.aln")
    hmm_path = os.path.join(out_dir, f"{family_id}.hmm")

    with open(fasta_path, "w", encoding="utf-8") as handle:
        for entry_id, sequence in sequences.items():
            handle.write(f">{entry_id}\n{sequence}\n")

    try:
        with open(aln_path, "w", encoding="utf-8") as out_handle:
            # --amino, not left to MAFFT's own auto-detection: a peptide cluster this short can
            # look like nucleotide data to the heuristic (A/C/G/T are valid amino acid codes too
            # -- Ala/Cys/Gly/Thr), and MAFFT then aligns with a nucleotide scoring matrix, which
            # fails outright when the sequences contain any other residue. Every input here is a
            # peptide sequence by construction, so there is nothing to actually detect.
            result = subprocess.run(
                ["mafft", "--amino", "--auto", "--thread", "1", fasta_path],
                stdout=out_handle, stderr=subprocess.PIPE, text=True, timeout=mafft_timeout)
        if result.returncode != 0:
            logger.warning(f"{family_id}: MAFFT failed ({result.returncode}): "
                           f"{result.stderr.strip()[:200]}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"{family_id}: MAFFT exceeded {mafft_timeout}s on "
                       f"{len(sequences)} sequences; skipped.")
        return None

    try:
        result = subprocess.run(
            ["hmmbuild", "--amino", "-n", family_id, hmm_path, aln_path],
            capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"{family_id}: hmmbuild failed ({result.returncode}): "
                           f"{result.stderr.strip()[:200]}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"{family_id}: hmmbuild timed out; skipped.")
        return None

    return hmm_path


def build_family_hmms(families_json: Optional[str] = None,
                      dataset_csv: Optional[str] = None,
                      out_dir: Optional[str] = None,
                      min_size: int = MIN_FAMILY_SIZE,
                      limit: Optional[int] = None,
                      resume: bool = True) -> Dict[str, Any]:
    """
    Builds a profile HMM for every qualifying SAMPLE-derived family cluster.

    :param families_json: `sample.families` output.
    :param dataset_csv: Dataset carrying sequences; `final_dataset.csv` when omitted.
    :param out_dir: Where to write FASTA/alignment/HMM files; :data:`HMM_DIR` when omitted.
    :param min_size: Minimum cluster size to build.
    :param limit: Build at most this many new HMMs this run (resumable, so a bounded run is safe).
    :param resume: Skip a cluster whose `.hmm` file already exists.
    :returns: A summary of what was built.
    """
    out_dir = out_dir or HMM_DIR
    dataset_csv = dataset_csv or os.path.join(
        str(DATA_DIR), "sample_final_release", "final_dataset.csv")

    clusters = clustered_families(families_json, min_size)
    if not clusters:
        return {"built": 0, "skipped_existing": 0, "failed": 0, "total_clusters": 0}

    all_ids = {eid for ids in clusters.values() for eid in ids}
    logger.info(f"Looking up sequences for {len(all_ids)} peptides across {len(clusters)} "
               f"clusters...")
    sequences = _sequences_for(all_ids, dataset_csv)

    built = skipped_existing = failed = 0
    for family_id, entry_ids in clusters.items():
        hmm_path = os.path.join(out_dir, f"{family_id}.hmm")
        if resume and os.path.exists(hmm_path):
            skipped_existing += 1
            continue
        if limit is not None and built >= limit:
            break

        members = {eid: sequences[eid] for eid in entry_ids if eid in sequences}
        if len(members) < min_size:
            # Some ids in the cluster had no row in `dataset_csv` (e.g. a decoy-only or stale
            # assignment); if what's left drops below the threshold, this cluster no longer
            # qualifies and building from it would misrepresent its size.
            logger.warning(f"{family_id}: only {len(members)}/{len(entry_ids)} members had a "
                           f"sequence; skipped.")
            failed += 1
            continue

        result = build_one_hmm(family_id, members, out_dir)
        if result is None:
            failed += 1
        else:
            built += 1

    summary = {"built": built, "skipped_existing": skipped_existing, "failed": failed,
              "total_clusters": len(clusters)}
    logger.info(f"Family HMMs: {built} built, {skipped_existing} already existed, "
               f"{failed} failed, of {len(clusters)} qualifying clusters.")
    return summary


def combined_hmm_database(out_dir: Optional[str] = None,
                          output_path: Optional[str] = None) -> Optional[str]:
    """
    Concatenates every built family HMM into one file and presses it for `hmmscan`.

    Kept as a separate step from :func:`build_family_hmms` rather than folded into it: pressing
    has to happen after *all* the HMMs it will contain exist, and re-running it after each
    individual build would redo the same work every time in a resumed, multi-session run.

    :param out_dir: Directory containing the per-cluster `.hmm` files; :data:`HMM_DIR` when
        omitted.
    :param output_path: Combined file path; ``<out_dir>/all_families.hmm`` when omitted.
    :returns: The combined, pressed file's path, or ``None`` if there was nothing to combine or
        `hmmpress` failed.
    """
    out_dir = out_dir or HMM_DIR
    output_path = output_path or os.path.join(out_dir, "all_families.hmm")

    hmm_files = sorted(f for f in glob.glob(os.path.join(out_dir, "*.hmm"))
                       if os.path.basename(f) != os.path.basename(output_path))
    if not hmm_files:
        logger.warning(f"No per-cluster HMMs in {out_dir}; nothing to combine.")
        return None

    with open(output_path, "w", encoding="utf-8") as out_handle:
        for path in hmm_files:
            with open(path, encoding="utf-8") as in_handle:
                out_handle.write(in_handle.read())

    for ext in ("h3f", "h3i", "h3m", "h3p"):
        stale = f"{output_path}.{ext}"
        if os.path.exists(stale):
            os.remove(stale)

    result = subprocess.run(["hmmpress", output_path], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.warning(f"hmmpress failed on {output_path}: {result.stderr.strip()[:200]}")
        return None

    logger.info(f"Combined {len(hmm_files)} family HMMs into {output_path} and pressed it "
               f"for hmmscan.")
    return output_path
