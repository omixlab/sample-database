"""
Incremental family assignment for newly ingested peptides.

The batch pipeline (``sample.families``) assigns families in one shot over the whole corpus: Pfam
anchoring, then near-identity clustering of the remainder. Re-running that from scratch every time a
handful of new sequences arrive is wasteful and, worse, *unstable* -- connected-component numbering
is derived from the whole set, so a full rebuild can renumber ``SAMPLE_FAM_*`` ids that other
tables, exports and any external citation already point at.

This module places new sequences into the *existing* family map without disturbing it. For each new
peptide it asks the same question the batch clusterer asks -- "is this within
:data:`sample.families.MIN_IDENTITY` of something already in the corpus, at a comparable length?" --
and:

* **inherits** the family of the peptide(s) it matches when it does resemble the existing corpus
  (including Pfam-anchored members: near-identity to a curated-family peptide is good evidence of
  the same family), or
* **mints a new** ``SAMPLE_FAM_*`` id, continuing the existing numbering, when it resembles nothing
  already known. Several new peptides that resemble each other but nothing existing are grouped into
  **one** new family together, exactly as connected-component clustering would place them.

The matching is deliberately the same recall-safe, length-bucketed PepMatch search
:mod:`sample.families` uses, sharing its constants so the incremental and batch answers agree. What
this path does *not* do is run ``hmmsearch`` against Pfam for a genuinely novel sequence -- that
needs the Pfam database and the ``hmmsearch`` binary, and is a batch concern. A new sequence that
would have anchored on Pfam only by profile (not by near-identity to an existing member) is placed
in a fresh ``SAMPLE_FAM_*`` cluster here and re-anchored the next time the full
:func:`sample.families.build_families` is run. That trade-off -- stable ids and no external
dependency now, in exchange for periodic re-anchoring -- is the whole point of an incremental step.
"""
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sample import families

logger = logging.getLogger("sample.incremental")

#: Sequences shorter than this are not clustered, matching :func:`sample.families.export_peptides_fasta`.
MIN_LENGTH: int = 5

#: Residues PepMatch accepts (B/X/Z are ambiguity codes it and hmmsearch both tolerate).
_FASTA_RESIDUES = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZ]+$")

#: Pattern of a minted family id, so :func:`next_family_number` can read the current maximum back.
_SAMPLE_FAM = re.compile(r"^SAMPLE_FAM_(\d+)$")


def next_family_number(existing_families: Iterable[Dict[str, Any]]) -> int:
    """
    Finds the next free ``SAMPLE_FAM_*`` cluster number.

    Pfam-anchored families carry a Pfam accession, not a ``SAMPLE_FAM_*`` id, and are ignored here;
    only the clustered families contribute to the numbering.

    :param existing_families: The value dicts of the current family map (each may carry a
        ``family_id``).
    :returns: One past the highest existing ``SAMPLE_FAM_*`` number, or 1 when none exist.
    """
    highest = 0
    for record in existing_families:
        match = _SAMPLE_FAM.match(str((record or {}).get("family_id") or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def format_family_id(number: int) -> str:
    """
    :param number: A cluster number.
    :returns: The zero-padded ``SAMPLE_FAM_*`` id, matching :func:`sample.families.build_families`.
    """
    return f"SAMPLE_FAM_{number:06d}"


def _clean_sequences(sequences: Dict[str, str]) -> Dict[str, str]:
    """
    Uppercases and filters a sequence map to what PepMatch can index.

    :param sequences: Raw ``id -> sequence`` map.
    :returns: The subset long enough and using only standard/ambiguity residues.
    """
    cleaned: Dict[str, str] = {}
    for name, sequence in sequences.items():
        value = str(sequence or "").strip().upper()
        if len(value) >= MIN_LENGTH and _FASTA_RESIDUES.match(value):
            cleaned[str(name)] = value
    return cleaned


def _new_sequence_edges(new: Dict[str, str], existing: Dict[str, str],
                        identity: float, length_ratio: float,
                        work_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Finds near-identity edges that *touch a new sequence*.

    Every new peptide is searched against the union of the existing corpus and the other new
    peptides, so an edge is emitted whenever a new sequence is within ``identity`` (at a length
    within ``length_ratio``) of any peptide, new or existing. Existing-to-existing pairs are never
    queried -- their assignment is already settled and must not be recomputed here.

    The search mirrors :func:`sample.families.similarity_edges` exactly (same length buckets, same
    :func:`sample.families.bucket_parameters`), so the incremental answer and a full rebuild agree
    on which peptides are neighbours.

    :param new: New ``id -> sequence`` peptides (already cleaned).
    :param existing: Existing ``id -> sequence`` peptides (already cleaned).
    :param identity: Required fraction of identical positions.
    :param length_ratio: Required ratio of the shorter length to the longer.
    :param work_dir: Directory for PepMatch index files; a temp dir when omitted.
    :returns: ``(a, b)`` id pairs, at least one of which is a new peptide, deduplicated.
    """
    from pepmatch import Matcher, Preprocessor

    if not new:
        return []

    own_dir = work_dir is None
    if own_dir:
        import tempfile
        work_dir = tempfile.mkdtemp(prefix="sample_incr_")
    os.makedirs(work_dir, exist_ok=True)

    # Target set is the union: a new peptide may match an existing one or another new one. New ids
    # win a collision so a re-run that re-ingests an existing sequence under a new id still matches.
    union: Dict[str, str] = {**existing, **new}
    target_fasta = families.write_fasta(union, os.path.join(work_dir, "_incr_target.fasta"))

    new_ids = set(new)
    by_sequence: Dict[str, List[str]] = {}
    for name, sequence in union.items():
        by_sequence.setdefault(sequence, []).append(name)

    edges: Dict[Tuple[str, str], int] = {}
    try:
        for low, high in families.LENGTH_BUCKETS:
            # Query only the new sequences in this bucket; the whole union is the target.
            queries = sorted({s for name, s in new.items() if low <= len(s) <= high})
            if not queries:
                continue
            max_mismatches, k = families.bucket_parameters(low, identity)

            Preprocessor(target_fasta, proteome_name=f"incr_k{k}",
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
                subject = str(subject).rsplit(".", 1)[0] if str(subject) not in union else subject
                subject_sequence = union.get(subject)
                if not subject_sequence:
                    continue

                mismatches = int(row.get("Mismatches") or 0)
                shorter = min(len(query_sequence), len(subject_sequence))
                longer = max(len(query_sequence), len(subject_sequence))
                if longer == 0 or shorter / longer < length_ratio:
                    continue
                if (len(query_sequence) - mismatches) / len(query_sequence) < identity:
                    continue

                for query_name in by_sequence.get(query_sequence, []):
                    # An edge is only interesting if it involves a new peptide; two existing
                    # peptides sharing this sequence value are already clustered together.
                    if query_name == subject:
                        continue
                    if query_name not in new_ids and subject not in new_ids:
                        continue
                    pair = (query_name, subject) if query_name < subject else (subject, query_name)
                    if pair not in edges or mismatches < edges[pair]:
                        edges[pair] = mismatches
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    return list(edges)


def _pick_inherited_family(members: Iterable[str],
                           existing_families: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chooses which existing family a new peptide inherits when its component spans several.

    Single-linkage components can bridge more than one existing family (A links B, B links C). The
    inherited family is the one most existing members of the component already carry; ties break on
    the lowest family id so the choice is deterministic.

    :param members: Ids in the connected component.
    :param existing_families: The current family map.
    :returns: The chosen family's record (``family_id``/``family_name``/``source``), or ``None``
        when no member of the component has a family yet.
    """
    counts: Dict[str, int] = {}
    representative: Dict[str, Dict[str, Any]] = {}
    for name in members:
        record = existing_families.get(name)
        family_id = (record or {}).get("family_id")
        if not family_id:
            continue
        counts[family_id] = counts.get(family_id, 0) + 1
        representative.setdefault(family_id, record)
    if not counts:
        return None
    winner = min(counts, key=lambda fid: (-counts[fid], fid))
    return representative[winner]


def assign_families(new_sequences: Dict[str, str], existing_sequences: Dict[str, str],
                    existing_families: Dict[str, Dict[str, Any]],
                    identity: float = families.MIN_IDENTITY,
                    length_ratio: float = families.MIN_LENGTH_RATIO,
                    work_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Assigns each new peptide an existing family, or a freshly minted one.

    :param new_sequences: New ``id -> sequence`` peptides to place.
    :param existing_sequences: The corpus already assigned, ``id -> sequence``.
    :param existing_families: The current family map, ``id -> {family_id, family_name, source, ...}``.
    :param identity: Required fraction of identical positions.
    :param length_ratio: Required ratio of the shorter length to the longer.
    :param work_dir: PepMatch working directory; a temp dir when omitted.
    :returns: ``id -> record`` for the **new** peptides only. Each record carries ``family_id``,
        ``family_name``, ``source`` (``"inherited"`` or ``"new-cluster"``), ``matched_entry_id``
        (the nearest existing peptide when inherited, else ``None``) and ``all_domains`` (empty;
        Pfam anchoring is a batch concern -- see the module docstring).
    """
    import networkx as nx

    new_clean = _clean_sequences(new_sequences)
    existing_clean = _clean_sequences(existing_sequences)

    # Peptides too short or non-standard to cluster are still placed, each in its own new family, so
    # nothing is silently dropped from the update.
    unclusterable = [name for name in new_sequences if name not in new_clean]

    edges = _new_sequence_edges(new_clean, existing_clean, identity, length_ratio, work_dir)

    graph = nx.Graph()
    graph.add_nodes_from(new_clean)
    graph.add_edges_from(edges)

    next_number = next_family_number(existing_families.values())
    assignment: Dict[str, Dict[str, Any]] = {}

    for component in nx.connected_components(graph):
        new_members = [name for name in component if name in new_clean]
        if not new_members:
            continue  # a component of existing peptides only; nothing to assign
        inherited = _pick_inherited_family(component, existing_families)
        if inherited is not None:
            matched = _nearest_existing(new_members, component, existing_families, inherited)
            for name in new_members:
                assignment[name] = {
                    "family_id": inherited.get("family_id"),
                    "family_name": inherited.get("family_name"),
                    "source": "inherited",
                    "matched_entry_id": matched,
                    "all_domains": list(inherited.get("all_domains") or []),
                }
        else:
            family_id = format_family_id(next_number)
            next_number += 1
            for name in new_members:
                assignment[name] = {
                    "family_id": family_id,
                    "family_name": None,
                    "source": "new-cluster",
                    "matched_entry_id": None,
                    "all_domains": [],
                }

    # Each unclusterable new sequence becomes its own singleton family.
    for name in unclusterable:
        family_id = format_family_id(next_number)
        next_number += 1
        assignment[name] = {
            "family_id": family_id,
            "family_name": None,
            "source": "new-cluster",
            "matched_entry_id": None,
            "all_domains": [],
        }

    inherited_n = sum(1 for a in assignment.values() if a["source"] == "inherited")
    new_family_ids = {a["family_id"] for a in assignment.values() if a["source"] == "new-cluster"}
    logger.info("Assigned %d new peptides: %d inherited an existing family, %d placed in %d new "
                "clusters.", len(assignment), inherited_n,
                len(assignment) - inherited_n, len(new_family_ids))
    return assignment


def _nearest_existing(new_members: List[str], component: Iterable[str],
                      existing_families: Dict[str, Dict[str, Any]],
                      inherited: Dict[str, Any]) -> Optional[str]:
    """
    Names an existing peptide of the inherited family, for provenance on the new record.

    :param new_members: New ids in the component (unused beyond documenting intent).
    :param component: All ids in the connected component.
    :param existing_families: The current family map.
    :param inherited: The chosen family record.
    :returns: The id of an existing peptide in the component carrying the inherited family, or
        ``None``.
    """
    family_id = inherited.get("family_id")
    for name in sorted(component):
        record = existing_families.get(name)
        if record and record.get("family_id") == family_id:
            return name
    return None


def load_existing(families_json: Optional[str] = None,
                  db_path: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Reads the current corpus and its family map from disk.

    :param families_json: The family assignment file; ``data/families/families.json`` when omitted.
    :param db_path: The release database; the project default when omitted.
    :returns: ``(existing_sequences, existing_families)``. ``existing_sequences`` maps ``entry_id ->
        sequence`` for every non-decoy peptide; ``existing_families`` is the parsed family map (empty
        when the file does not exist yet, i.e. the very first update).
    """
    from sample import db as release_db

    families_json = families_json or os.path.join(
        families.FAMILY_DIR, "families.json")

    existing_families: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(families_json):
        with open(families_json, encoding="utf-8") as handle:
            existing_families = json.load(handle)

    existing_sequences: Dict[str, str] = {}
    connection = release_db.get_connection(db_path, shared=False)
    try:
        rows = connection.execute(
            "SELECT entry_id, sequence FROM peptides WHERE COALESCE(is_decoy, 0) = 0").fetchall()
    finally:
        connection.close()
    for row in rows:
        sequence = str(row["sequence"] or "").strip().upper()
        if sequence:
            existing_sequences[str(row["entry_id"])] = sequence

    return existing_sequences, existing_families


def apply_assignments(assignment: Dict[str, Dict[str, Any]],
                      families_json: Optional[str] = None) -> str:
    """
    Merges new family records into ``families.json`` and rewrites it.

    New records are written in the same shape as :func:`sample.families.build_families`' output --
    plus ``matched_entry_id`` for provenance -- so a subsequent ``load-sql`` step loads them into
    ``peptide_families`` unchanged. Existing entries are never modified.

    :param assignment: The new ``id -> record`` map from :func:`assign_families`.
    :param families_json: The family assignment file; ``data/families/families.json`` when omitted.
    :returns: The path written.
    """
    families_json = families_json or os.path.join(
        families.FAMILY_DIR, "families.json")

    merged: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(families_json):
        with open(families_json, encoding="utf-8") as handle:
            merged = json.load(handle)

    for name, record in assignment.items():
        merged[name] = {
            "family_id": record.get("family_id"),
            "family_name": record.get("family_name"),
            "source": record.get("source"),
            "evalue": None,
            "all_domains": list(record.get("all_domains") or []),
            "matched_entry_id": record.get("matched_entry_id"),
        }

    os.makedirs(os.path.dirname(os.path.abspath(families_json)), exist_ok=True)
    with open(families_json, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=1)
    logger.info("Merged %d new family assignments into %s (now %d peptides).",
                len(assignment), families_json, len(merged))
    return families_json


def update_families(new_sequences: Dict[str, str], families_json: Optional[str] = None,
                    db_path: Optional[str] = None,
                    identity: float = families.MIN_IDENTITY,
                    length_ratio: float = families.MIN_LENGTH_RATIO,
                    apply: bool = True,
                    work_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Places a batch of new peptides into the family map, optionally writing the result.

    This is the offline core of the incremental update: it neither downloads nor extracts anything,
    it only assigns families to sequences already in hand. The acquisition of *which* sequences are
    new -- new papers, refreshed external databases -- is orchestrated separately (the pipeline's
    download/merge/extract steps) and is deliberately kept out of this function so the assignment is
    testable and side-effect-free on its own.

    :param new_sequences: New ``id -> sequence`` peptides to place.
    :param families_json: The family assignment file to read and (when ``apply``) update.
    :param db_path: The release database to read the existing corpus from.
    :param identity: Required fraction of identical positions.
    :param length_ratio: Required ratio of the shorter length to the longer.
    :param apply: When true, merge the result into ``families.json``; when false, only report it.
    :param work_dir: PepMatch working directory.
    :returns: A summary with per-peptide assignments and counts (``inherited``, ``new_clusters``).
    """
    existing_sequences, existing_families = load_existing(families_json, db_path)
    # A new id that already exists in the corpus is an update of that peptide, not a new one; drop it
    # from the "new" set so it keeps its established family rather than being re-clustered.
    genuinely_new = {name: seq for name, seq in new_sequences.items()
                     if name not in existing_families}
    skipped = len(new_sequences) - len(genuinely_new)

    assignment = assign_families(genuinely_new, existing_sequences, existing_families,
                                 identity=identity, length_ratio=length_ratio, work_dir=work_dir)

    if apply and assignment:
        apply_assignments(assignment, families_json)

    inherited = sum(1 for a in assignment.values() if a["source"] == "inherited")
    new_clusters = {a["family_id"] for a in assignment.values() if a["source"] == "new-cluster"}
    return {
        "assignments": assignment,
        "assigned": len(assignment),
        "inherited": inherited,
        "new_clusters": len(new_clusters),
        "skipped_existing": skipped,
        "applied": bool(apply and assignment),
    }
