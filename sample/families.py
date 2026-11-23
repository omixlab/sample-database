import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("sample.families")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FAMILY_DIR = os.path.join(DATA_DIR, "families")
PFAM_DIR = os.path.join(DATA_DIR, "pfam")

#: Fraction of positions that must be identical for two peptides to share a family.
MIN_IDENTITY: float = 0.8

#: How close in length two peptides must be. The match always covers the query in full, so this is
#: what stops a 5-residue peptide joining the family of every 700-residue protein that happens to
#: contain it -- containment is not homology, and without this the graph collapses (measured: one
#: component of 23,647 of 23,811 peptides).
MIN_LENGTH_RATIO: float = 0.8

#: Upper bound on mismatches in any single search, for tractability on long sequences.
MAX_MISMATCH_CAP: int = 6

#: Length buckets, as ``(min_length, max_length)``. The mismatch budget and k-mer size for each are
#: derived from :data:`MIN_IDENTITY` by :func:`bucket_parameters`, rather than fixed here, so the
#: two constraints that matter -- an identity floor and guaranteed k-mer recall -- stay in step.
LENGTH_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (5, 9),
    (10, 14),
    (15, 24),
    (25, 49),
    (50, 10_000),
)


def bucket_parameters(min_length: int, identity: float = MIN_IDENTITY) -> Tuple[int, int]:
    """
    Derives the search parameters for a length bucket.

    Two constraints have to hold together. The mismatch budget must keep identity at or above
    ``identity`` for the *shortest* sequence in the bucket -- a flat budget of 2 is 40 % divergence
    on a 5-mer but 2 % on a 100-mer. And the k-mer size must satisfy ``k * (e + 1) <= L``, since
    with ``e`` mismatches placed as badly as possible only that guarantees one intact k-mer to seed
    the search.

    :param min_length: Shortest sequence the bucket must handle.
    :param identity: Required fraction of identical positions.
    :returns: ``(max_mismatches, k)``.
    """
    # Rounded, not truncated: at 80 % identity a 5-mer tolerates exactly one mismatch (4/5), but
    # 1.0 - 0.8 is 0.19999999999999996 in binary floating point, so truncation yields zero and the
    # whole bucket silently matches nothing but exact duplicates.
    mismatches = max(0, min(MAX_MISMATCH_CAP, int(round(min_length * (1.0 - identity) + 1e-9))))
    k = max(2, min_length // (mismatches + 1))
    return mismatches, k

#: Clusters larger than this are re-examined rather than trusted. Single-linkage clustering chains:
#: A links B, B links C, and C need not resemble A at all. A component running to thousands of
#: members is that artefact, not a family.
MAX_TRUSTED_CLUSTER: int = 500


def read_fasta(path: str) -> Dict[str, str]:
    """
    :param path: A FASTA file.
    :returns: Mapping from identifier to sequence.
    """
    sequences: Dict[str, str] = {}
    name: Optional[str] = None
    chunks: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    sequences[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name:
        sequences[name] = "".join(chunks)
    return sequences


def write_fasta(sequences: Dict[str, str], path: str) -> str:
    """
    :param sequences: Mapping from identifier to sequence.
    :param path: Destination file.
    :returns: The path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name}\n{sequence}\n")
    return path


#: Residues ColabFold-independent tools accept. B, X and Z are ambiguity codes that hmmsearch and
#: PepMatch both handle.
_FASTA_RESIDUES = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZ]+$")


def export_peptides_fasta(dataset_csv: Optional[str] = None, output_path: Optional[str] = None,
                          min_length: int = 5) -> str:
    """
    Writes the peptide FASTA that ``hmmsearch`` and PepMatch consume.

    Derived from ``final_dataset.csv`` -- step 29's output -- and **not** from ``sample.db``. That
    distinction is a real ordering constraint rather than a preference: the SQLite database is
    built by step 44, so sourcing the FASTA from it would make family assignment depend on a step
    that runs after it. The first version of this module did exactly that, working only because
    the FASTA happened to exist from a manual run; on a clean checkout step 35 would have had no
    input at all.

    :param dataset_csv: Assembled dataset; ``data/sample_final_release/final_dataset.csv`` when
        omitted.
    :param output_path: FASTA to write; ``data/pfam/peptides.fasta`` when omitted.
    :param min_length: Shortest sequence to include. Below this a profile search returns noise.
    :returns: Path to the FASTA.
    :raises FileNotFoundError: When the assembled dataset does not exist. Run step 31 first.
    """
    import pandas as pd

    dataset_csv = dataset_csv or os.path.join(
        DATA_DIR, "sample_final_release", "final_dataset.csv")
    output_path = output_path or os.path.join(PFAM_DIR, "peptides.fasta")

    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(
            f"Assembled dataset not found at {dataset_csv}. Run step 31 "
            f"(`sample-cli assembly`) before assigning families.")

    frame = pd.read_csv(dataset_csv, low_memory=False)
    columns = {c.lower(): c for c in frame.columns}
    decoy_column = columns.get("is_decoy")

    sequences: Dict[str, str] = {}
    skipped = 0
    for row in frame.itertuples(index=False):
        entry_id = str(getattr(row, columns.get("entry_id", "entry_id"), "") or "").strip()
        sequence = str(getattr(row, columns.get("sequence", "sequence"), "") or "").strip().upper()
        # Decoys are shuffled sequences with no provenance; a family assignment for one would be
        # a statement about a fabrication.
        if decoy_column and str(getattr(row, decoy_column, 0)) in ("1", "1.0", "True", "true"):
            continue
        if not entry_id or not sequence or len(sequence) < min_length:
            skipped += 1
            continue
        if not _FASTA_RESIDUES.match(sequence):
            skipped += 1
            continue
        sequences[entry_id] = sequence

    write_fasta(sequences, output_path)
    logger.info(f"Wrote {len(sequences)} peptides to {output_path} "
                f"({skipped} skipped: too short or non-standard residues).")
    return output_path


def parse_pfam_hits(domtbl_path: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads the ``--domtblout`` table produced by ``hmmsearch`` against Pfam-A.

    :param domtbl_path: The domain table; ``data/pfam/pfam_hits.domtbl`` when omitted.
    :returns: Peptide id -> its Pfam domains, best (lowest full-sequence E-value) first.
    """
    domtbl_path = domtbl_path or os.path.join(PFAM_DIR, "pfam_hits.domtbl")
    hits: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.exists(domtbl_path):
        logger.warning(f"No Pfam hits at {domtbl_path}; run the hmmsearch step first.")
        return hits

    with open(domtbl_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 14:
                continue
            # hmmsearch --domtblout column order: target(peptide), accession, tlen,
            # query(family), accession, qlen, full E-value, full score, ...
            peptide, family, accession = fields[0], fields[3], fields[4]
            try:
                evalue = float(fields[6])
            except ValueError:
                evalue = float("inf")
            hits.setdefault(peptide, []).append({
                "family": family,
                # Pfam accessions carry a version suffix (PF00711.25); the base accession is what
                # InterPro and every other resource resolves.
                "accession": accession.split(".")[0],
                "evalue": evalue,
            })

    for peptide in hits:
        hits[peptide].sort(key=lambda h: h["evalue"])
    logger.info(f"Pfam: {len(hits)} peptides matched "
                f"{len({h['accession'] for v in hits.values() for h in v})} families.")
    return hits


def similarity_edges(sequences: Dict[str, str], identity: float = MIN_IDENTITY,
                     length_ratio: float = MIN_LENGTH_RATIO,
                     work_dir: Optional[str] = None) -> List[Tuple[str, str, int]]:
    """
    Finds near-identical peptide pairs with PepMatch.

    One pass per length bucket, each with the mismatch budget and k-mer size that
    :func:`bucket_parameters` derives for it. Every pass searches against the full set, so a short
    peptide can still be linked to a longer one -- but only if their lengths are within
    ``length_ratio``, because PepMatch reports containment and containment is not homology.

    :param sequences: The peptides to compare.
    :param identity: Required fraction of identical positions.
    :param length_ratio: Required ratio of the shorter length to the longer.
    :param work_dir: Where index files are written; ``data/families`` when omitted.
    :returns: ``(peptide_a, peptide_b, mismatches)`` triples, deduplicated and self-matches
        removed.
    """
    from pepmatch import Matcher, Preprocessor

    work_dir = work_dir or FAMILY_DIR
    os.makedirs(work_dir, exist_ok=True)
    target_fasta = write_fasta(sequences, os.path.join(work_dir, "_cluster_target.fasta"))

    # PepMatch's Preprocessor caches a k-mer index next to the FASTA it was built from and reuses
    # it on a later call without checking whether the FASTA's content has since changed. work_dir
    # defaults to FAMILY_DIR, a long-lived, shared directory -- so a stale "_cluster_target_*mers"
    # index from a previous, differently-sized candidate set silently wins here, and the resulting
    # search runs against sequences that no longer exist, finding no real matches while reporting
    # no error. Measured directly: a fresh run against 38,064 candidates reused an 8-day-old index
    # built from a smaller set and came back with every peptide as its own singleton cluster.
    # Removing any pre-existing target index before rebuilding forces PepMatch to index the FASTA
    # just written above, every time.
    for existing in os.listdir(work_dir):
        if existing.startswith("_cluster_target_") and existing.endswith("mers.pepidx"):
            os.remove(os.path.join(work_dir, existing))

    by_sequence: Dict[str, List[str]] = {}
    for name, sequence in sequences.items():
        by_sequence.setdefault(sequence, []).append(name)

    edges: Dict[Tuple[str, str], int] = {}
    rejected_length = rejected_identity = 0

    for low, high in LENGTH_BUCKETS:
        queries = sorted({s for s in sequences.values() if low <= len(s) <= high})
        if not queries:
            continue
        max_mismatches, k = bucket_parameters(low, identity)

        Preprocessor(target_fasta, proteome_name=f"cluster_k{k}",
                     preprocessed_files_path=work_dir).preprocess(k=k)
        frame = Matcher(query=queries, proteome_file=target_fasta,
                        max_mismatches=max_mismatches, k=k,
                        preprocessed_files_path=work_dir,
                        sequence_version=False).match()

        if frame is None or len(frame) == 0:
            continue

        for row in frame.iter_rows(named=True):
            query_sequence = row.get("Query Sequence")
            subject = row.get("Protein ID")
            if not query_sequence or not subject:
                continue
            # PepMatch can return the identifier with a UniProt-style version suffix
            # (``SAMPLE0000000707.1``). Left unstripped, every subject lookup below misses and
            # each row is skipped in silence -- which reads as "no similar peptides exist".
            subject = str(subject).rsplit(".", 1)[0] if str(subject) not in sequences else subject
            subject_sequence = sequences.get(subject)
            if not subject_sequence:
                continue

            mismatches = int(row.get("Mismatches") or 0)
            query_length, subject_length = len(query_sequence), len(subject_sequence)

            # The match covers the query in full, so the query is the shorter of the two whenever
            # this is a containment hit. Requiring comparable lengths is what keeps a 5-mer out of
            # every long protein's family.
            shorter, longer = min(query_length, subject_length), max(query_length, subject_length)
            if longer == 0 or shorter / longer < length_ratio:
                rejected_length += 1
                continue
            if query_length and (query_length - mismatches) / query_length < identity:
                rejected_identity += 1
                continue

            for query_name in by_sequence.get(query_sequence, []):
                if query_name == subject:
                    continue
                pair = (query_name, subject) if query_name < subject else (subject, query_name)
                if pair not in edges or mismatches < edges[pair]:
                    edges[pair] = mismatches

        logger.info(f"Bucket {low}-{high} (k={k}, max_mismatches={max_mismatches}): "
                    f"{len(queries)} queries, {len(edges)} cumulative edges.")

    logger.info(f"Rejected {rejected_length} matches on length ratio and {rejected_identity} on "
                f"identity; kept {len(edges)} edges.")
    return [(a, b, m) for (a, b), m in edges.items()]


#: cd-hit's word size must be small enough that two sequences at the target identity still share a
#: word; the tool rejects the combination outright otherwise. These are the bands from its own
#: user guide, and 0.4 is the documented floor for cd-hit proper (below it the guide directs you to
#: psi-cd-hit, which needs BLAST and a great deal more time).
_CDHIT_WORD_SIZE: Tuple[Tuple[float, int], ...] = ((0.7, 5), (0.6, 4), (0.5, 3), (0.4, 2))

#: Shortest sequence cd-hit will keep. Its own default is 10, which would silently DISCARD every
#: peptide of 10 residues or fewer -- a large fraction of this dataset, and precisely the fraction
#: where an unclustered near-duplicate does the most damage to a split.
_CDHIT_MIN_LENGTH: int = 4


def cdhit_available() -> bool:
    """
    :returns: Whether the ``cd-hit`` executable is on ``PATH``.
    """
    import shutil

    return shutil.which("cd-hit") is not None


def cluster_cdhit(sequences: Dict[str, str], identity: float,
                  length_ratio: float = MIN_LENGTH_RATIO,
                  work_dir: Optional[str] = None) -> Dict[str, int]:
    """
    Groups peptides with cd-hit, the standard tool for homology reduction at a stated identity.

    Preferred over :func:`similarity_edges`' PepMatch route whenever the requested identity is
    below what that route can actually deliver. PepMatch searches a fixed mismatch budget, and
    :data:`MAX_MISMATCH_CAP` bounds that budget at 6 for tractability -- so a request for 40 %
    identity silently becomes 60 % for a 15-mer, 76 % for a 25-mer and 88 % for anything 50
    residues or longer. That is not a small discrepancy in the direction that flatters a model:
    under-clustering leaves homologues split across train and test, and the reported score then
    includes memorised near-duplicates. cd-hit has no such ceiling and clusters the full set in
    seconds.

    :param sequences: The peptides to cluster.
    :param identity: Required fraction of identical positions. Values below 0.4 are raised to
        0.4 with a warning -- cd-hit proper does not go lower.
    :param length_ratio: Required ratio of the shorter length to the longer, passed as cd-hit's
        ``-s``. Same purpose as on the PepMatch path: containment is not homology.
    :param work_dir: Working directory for the FASTA and cd-hit's own output files;
        :data:`FAMILY_DIR` when omitted.
    :returns: Peptide id -> cluster number, every input id present. Singletons for anything cd-hit
        dropped (a sequence below :data:`_CDHIT_MIN_LENGTH`), which is the honest outcome rather
        than silently omitting those peptides from the grouping the caller is about to split on.
    :raises RuntimeError: When cd-hit is not installed or exits non-zero.
    """
    import subprocess

    if not cdhit_available():
        raise RuntimeError("cd-hit is not on PATH.")
    if identity < 0.4:
        logger.warning(f"cd-hit cannot cluster below 40% identity; {identity:.0%} raised to 40%.")
        identity = 0.4

    word_size = next(k for threshold, k in _CDHIT_WORD_SIZE if identity >= threshold)
    work_dir = work_dir or FAMILY_DIR
    os.makedirs(work_dir, exist_ok=True)
    fasta = write_fasta(sequences, os.path.join(work_dir, "_cdhit_input.fasta"))
    out_path = os.path.join(work_dir, "_cdhit_output")

    command = [
        "cd-hit", "-i", fasta, "-o", out_path,
        "-c", f"{identity:.4f}", "-n", str(word_size),
        "-l", str(_CDHIT_MIN_LENGTH),
        "-s", f"{length_ratio:.4f}",
        "-d", "0",      # keep the full identifier; the default truncates at 20 characters
        "-M", "0",      # no memory ceiling
        "-T", "0",      # all cores
        "-g", "1",      # compare against every cluster and take the best, not the first hit
    ]
    logger.info(f"Clustering {len(sequences)} peptides with cd-hit at {identity:.0%} identity "
               f"(word size {word_size})...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cd-hit failed ({result.returncode}): "
                           f"{(result.stderr or result.stdout or '').strip()[:500]}")

    assignment: Dict[str, int] = {}
    cluster_index = 0
    with open(f"{out_path}.clstr", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">Cluster"):
                cluster_index += 1
                continue
            # Member lines read: `0       23aa, >SAMPLE0000000707... *`
            start = line.find(">")
            end = line.find("...", start)
            if start == -1 or end == -1:
                continue
            assignment[line[start + 1:end]] = cluster_index

    # Anything cd-hit dropped for being too short still has to receive a group: the caller splits
    # on this mapping, and a missing key would raise there rather than here.
    dropped = 0
    for name in sequences:
        if name not in assignment:
            cluster_index += 1
            assignment[name] = cluster_index
            dropped += 1

    n_clusters = len(set(assignment.values()))
    logger.info(f"cd-hit: {len(sequences)} peptides into {n_clusters} clusters at "
               f">={identity:.0%} identity"
               + (f" ({dropped} below {_CDHIT_MIN_LENGTH} residues kept as singletons)."
                  if dropped else "."))
    return assignment


def cluster(sequences: Dict[str, str], identity: float = MIN_IDENTITY,
            length_ratio: float = MIN_LENGTH_RATIO,
            work_dir: Optional[str] = None) -> Dict[str, int]:
    """
    Groups peptides into families by connected components of the similarity graph.

    :param sequences: The peptides to cluster.
    :param identity: Required fraction of identical positions.
    :param length_ratio: Required ratio of the shorter length to the longer.
    :param work_dir: Working directory for index files.
    :returns: Peptide id -> cluster number. Peptides with no neighbour become singleton clusters,
        which is the honest outcome for a sequence nothing else resembles.
    """
    import networkx as nx

    edges = similarity_edges(sequences, identity=identity, length_ratio=length_ratio,
                             work_dir=work_dir)
    graph = nx.Graph()
    graph.add_nodes_from(sequences)
    graph.add_edges_from((a, b) for a, b, _ in edges)

    assignment: Dict[str, int] = {}
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for index, component in enumerate(components, start=1):
        for name in component:
            assignment[name] = index

    sizes = [len(c) for c in components]
    multi = [s for s in sizes if s > 1]
    logger.info(f"Clustered {len(sequences)} peptides into {len(components)} components "
                f"({len(multi)} with more than one member; largest {sizes[0] if sizes else 0}).")
    if sizes and sizes[0] > MAX_TRUSTED_CLUSTER:
        logger.warning(
            f"Largest cluster has {sizes[0]} members, above the {MAX_TRUSTED_CLUSTER} "
            f"single-linkage chaining threshold -- treat it as a superfamily, not a family.")
    return assignment


def build_families(peptides_fasta: Optional[str] = None, domtbl_path: Optional[str] = None,
                   output_path: Optional[str] = None,
                   identity: float = MIN_IDENTITY) -> Dict[str, Any]:
    """
    Assigns every peptide a family: its Pfam family where one exists, otherwise a cluster.

    :param peptides_fasta: Peptide FASTA; ``data/pfam/peptides.fasta`` when omitted.
    :param domtbl_path: ``hmmsearch`` domain table; ``data/pfam/pfam_hits.domtbl`` when omitted.
    :param output_path: Where to write the assignment; ``data/families/families.json`` when
        omitted.
    :param identity: Required fraction of identical positions when clustering the remainder.
    :returns: A summary of the assignment.
    """
    peptides_fasta = peptides_fasta or os.path.join(PFAM_DIR, "peptides.fasta")
    output_path = output_path or os.path.join(FAMILY_DIR, "families.json")

    # Regenerated from the assembled dataset when absent, so this step has an input on a clean
    # checkout instead of depending on a file a later step happens to have produced.
    if not os.path.exists(peptides_fasta):
        logger.info(f"No peptide FASTA at {peptides_fasta}; deriving it from the assembled "
                    f"dataset.")
        export_peptides_fasta(output_path=peptides_fasta)

    sequences = read_fasta(peptides_fasta)
    pfam = parse_pfam_hits(domtbl_path)

    unmatched = {name: seq for name, seq in sequences.items() if name not in pfam}
    logger.info(f"{len(pfam)} peptides anchored on Pfam; clustering the remaining "
                f"{len(unmatched)}.")

    clusters = cluster(unmatched, identity=identity) if unmatched else {}

    assignment: Dict[str, Dict[str, Any]] = {}
    for name in sequences:
        if name in pfam:
            best = pfam[name][0]
            assignment[name] = {
                "family_id": best["accession"],
                "family_name": best["family"],
                "source": "pfam",
                "evalue": best["evalue"],
                "all_domains": [d["accession"] for d in pfam[name]],
            }
        else:
            number = clusters.get(name)
            assignment[name] = {
                "family_id": f"SAMPLE_FAM_{number:06d}" if number else None,
                "family_name": None,
                "source": "cluster" if number else "unassigned",
                "evalue": None,
                "all_domains": [],
            }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(assignment, handle, indent=1)

    pfam_families = {a["family_id"] for a in assignment.values() if a["source"] == "pfam"}
    cluster_families = {a["family_id"] for a in assignment.values() if a["source"] == "cluster"}
    summary = {
        "path": output_path,
        "peptides": len(sequences),
        "pfam_anchored": sum(1 for a in assignment.values() if a["source"] == "pfam"),
        "clustered": sum(1 for a in assignment.values() if a["source"] == "cluster"),
        "pfam_families": len(pfam_families),
        "novel_families": len(cluster_families),
    }
    logger.info(f"Families: {summary['pfam_anchored']} Pfam-anchored across "
                f"{summary['pfam_families']} families; {summary['clustered']} clustered into "
                f"{summary['novel_families']} novel families.")
    return summary
