"""
Embedding search: find peptides whose ESM-2 representation is closest to a query sequence's.

Distinct from BLAST (exact-alignment identity) and HMMER (profile membership in one of the 519
SAMPLE-derived clusters): this finds peptides the protein language model places nearby in
representation space even with no strong pairwise alignment, which is where sequence-identity
search structurally cannot help -- two peptides can be functionally similar (same fold class, same
class of activity) while sharing too little raw sequence to align.

**Model and cost, measured, not assumed.** `facebook/esm2_t6_8M_UR50D` -- ESM-2's smallest
checkpoint, 320-dimensional output, already loaded elsewhere in this pipeline for the analytics
UMAP chart (`sample.analysis`) -- benchmarked at 8 ms/sequence on this host's CPU alone (`torch`'s
own CUDA check is unreliable across this host's environments: `False` when checked directly against
this host's driver, but the actual `sample-chem` build of `torch` did find a GPU when the index was
built for real, finishing in under two minutes) over 500 real peptides averaging 37 residues. The
released corpus is 37,042 real peptides (`peptides.is_decoy = 0`; 74,084 is that count *plus* an
equal number of decoys, which are never indexed -- see below), so even the CPU-only estimate is
under 5 minutes for a full build -- nowhere near the multi-hour/day GPU jobs this project's rule on
long-running actions is about (ColabFold, RefSeq) -- so building the index does not need a
go-ahead the way those do.

**Chroma, not a hand-rolled index.** At this corpus size (37,042 x 320 floats, ~45 MB) an exact
brute-force search would also be fast enough; Chroma is used anyway because it is the tool TODO.md
named for this feature and it persists the index to disk between requests, which a from-scratch
in-memory array would need its own loading/caching layer to do.

**Synchronous, like BLAST and HMMER before it, not Celery.** Embedding one query sequence is a
single forward pass through an 8M-parameter model (part of the 8 ms/sequence figure above) plus one
Chroma k-NN query over the 37,042 indexed vectors -- both well under a second, measured below (a
warm-process query -- the model already loaded, the normal state of a running web server --
measured at ~20 ms end to end). Queueing a job for that would add latency and infrastructure for no
benefit; TODO.md's original "-> Celery job" framing predates this benchmark.
"""
import logging
import os
from typing import Any, Dict, List, Optional

from sample.config import DATA_DIR

logger = logging.getLogger("sample.embeddings")

#: Where the persistent Chroma index lives.
CHROMA_DIR = os.path.join(str(DATA_DIR), "embeddings", "chroma")

#: One collection for the whole released corpus. Decoys are never indexed -- see
#: :func:`build_embedding_index` -- so a hit here is never a fabricated training sequence.
COLLECTION_NAME = "sample_peptides_esm2"

#: Rows added to Chroma per call, and how often build_embedding_index logs progress. Keeps memory
#: bounded and gives a resumability-free build something to report every few seconds rather than
#: going silent for the full multi-minute run.
_BATCH_SIZE = 250


def embedding_index_path() -> str:
    """
    :returns: :data:`CHROMA_DIR` if a Chroma index has been built there, else ``""``.
    """
    return CHROMA_DIR if os.path.isdir(CHROMA_DIR) else ""


def build_embedding_index(dataset_csv: Optional[str] = None, chroma_dir: Optional[str] = None,
                          batch_size: int = _BATCH_SIZE, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Computes ESM-2 embeddings for every real (non-decoy) peptide and indexes them in Chroma.

    Rebuilds the collection from scratch each run rather than resuming a partial one -- at under 5
    minutes for the full corpus (see the module docstring), the complexity a resumable build would
    need is not worth it, unlike :func:`sample.family_hmms.build_family_hmms`, which runs far
    longer per unit of work.

    :param dataset_csv: Source of ``entry_id``/``sequence``/``is_decoy``; the packaged consensus
        dataset when omitted.
    :param chroma_dir: Where to persist the index; :data:`CHROMA_DIR` when omitted.
    :param batch_size: Peptides embedded and added to Chroma per batch.
    :param limit: Index at most this many peptides. For development/testing; omit for a real
        build.
    :returns: A summary: how many peptides were indexed and how many were skipped (embedding
        failed for that batch).
    """
    import chromadb
    import pandas as pd

    from sample.esm_enrichment import compute_esm2_embeddings

    # final_dataset.csv, not the earlier merged_from_database.csv or enriched.csv lineages --
    # those two do not carry is_decoy at all, and indexing without that column risks a fabricated
    # training sequence coming back as though it were a real peptide (see the is_decoy filter
    # below).
    dataset_csv = dataset_csv or os.path.join(
        str(DATA_DIR), "sample_final_release", "final_dataset.csv")
    chroma_dir = chroma_dir or CHROMA_DIR

    if not os.path.exists(dataset_csv):
        logger.warning(f"No dataset at {dataset_csv}; nothing to index.")
        return {"path": chroma_dir, "indexed": 0, "skipped": 0}

    if batch_size < 1:
        # `range(0, total, 0)` raises ValueError: range() arg 3 must not be zero -- reachable from
        # the CLI's `--batch-size`, which has no lower bound of its own, so this turns a Python
        # internals error into a message that names the actual mistake.
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")

    df = pd.read_csv(dataset_csv, usecols=["entry_id", "sequence", "is_decoy"], low_memory=False)
    # Decoys are fabricated training negatives; indexing them would let a similarity search
    # return one as though it were a real peptide, the same invariant sample.db enforces on every
    # other read path.
    df = df[df["is_decoy"].fillna(0).astype(int) != 1]
    df = df.dropna(subset=["entry_id", "sequence"])
    if limit:
        df = df.head(limit)

    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # No prior collection to replace; not an error.
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    entry_ids = df["entry_id"].astype(str).tolist()
    sequences = df["sequence"].astype(str).tolist()
    total = len(sequences)
    indexed = 0
    skipped = 0

    for start in range(0, total, batch_size):
        batch_ids = entry_ids[start:start + batch_size]
        batch_seqs = sequences[start:start + batch_size]
        embeddings = compute_esm2_embeddings(batch_seqs, batch_size=batch_size)
        if embeddings is None:
            skipped += len(batch_ids)
            continue
        collection.add(ids=batch_ids, embeddings=embeddings.tolist())
        indexed += len(batch_ids)
        if (start // batch_size) % 20 == 0:
            logger.info(f"Embedding index: {indexed + skipped}/{total} peptides processed...")

    summary = {"path": chroma_dir, "collection": COLLECTION_NAME, "indexed": indexed,
              "skipped": skipped}
    logger.info(f"Embedding index built: {indexed} peptides indexed, {skipped} skipped "
               f"(ESM-2 unavailable for their batch) -> {chroma_dir}")
    return summary


def run_embedding_search(query_seq: str, chroma_dir: str, max_hits: int = 50) -> List[Dict[str, Any]]:
    """
    Finds the indexed peptides whose ESM-2 embedding is closest to the query's.

    :param query_seq: Query sequence, plain or FASTA; headers and line breaks are stripped.
    :param chroma_dir: Persistent Chroma index, from :func:`embedding_index_path`.
    :param max_hits: Maximum peptides to report.
    :returns: One dict per hit (``entry_id``, ``distance``, ``similarity``), ordered by ascending
        distance (= descending similarity), or ``[]`` when the query is empty, the index is
        missing, or the search fails.
    """
    lines = [line.strip() for line in str(query_seq).strip().splitlines()]
    cleaned = "".join(l for l in lines if l and not l.startswith(">")).upper().replace(" ", "")
    if not cleaned:
        return []

    try:
        import chromadb

        from sample.esm_enrichment import compute_esm2_embeddings

        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(COLLECTION_NAME)

        embeddings = compute_esm2_embeddings([cleaned], batch_size=1)
        if embeddings is None:
            return []

        result = collection.query(query_embeddings=embeddings.tolist(), n_results=max_hits)
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [{"entry_id": entry_id, "distance": float(distance),
                 "similarity": 1.0 - float(distance)}
                for entry_id, distance in zip(ids, distances)]
    except Exception as e:
        logger.error(f"Embedding search failed: {e}")
        return []
