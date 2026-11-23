# Extraction & Enrichment Optimization Guide

This document is the authoritative reference for how `sample-cli extract` and the enrichment
stages (`sample-cli external-enrich`, `sample-cli enrich`) work after the 2026-07-31 refactor,
why they're built this way, and what it takes to get from here to a high-quality, low-noise
antimicrobial + anticancer peptide (AMP/ACP) dataset. It replaces the old `PROJECT.md`, which
described a `sample-cli consolidate` command and milestone plan that no longer matches the code.

**A note on how this document came to be, because it's relevant to trusting it**: the first pass
of this refactor (§2-3 below) was verified only with small `--limit` smoke tests (a handful of
documents). Running it for real afterward surfaced three further, more serious bugs that no
small smoke test would have caught — a process-pool memory design that looked correct in theory
but failed under real concurrent load, a pre-existing catastrophic-backtracking regex that could
hang a worker indefinitely on ordinary-looking real documents, and a client-per-call resource
leak in the LLM fallback path. All three are described in §2.6-2.8 with the actual measurements
that found them, and the extraction pool architecture in §1.1/§3.1 reflects the corrected
design, not the first attempt. The lesson generalizes: for a pipeline like this, "it works on 10
documents" and "it works on 20,000 documents" are different claims, and only the second one
matters.

---

## 1. Architecture

### 1.1 Extraction (`sample-cli extract`)

```
data/*_data_text/<source>/<id>.txt          (pre-chunked document text, produced upstream)
        │
        ▼
sample/pipeline_runner.py :: process_all_extractions
        │  resolves the source's document list (CSV of internal_ids), builds a task list
        │  of (internal_id, text_path, output_path), clamps --workers to CPU core count
        ▼
        run_parallel_pipeline
        │  ThreadPoolExecutor (not a process pool -- see §2.6 for why), bounded submission
        │  window (2× workers in flight at once, so memory ∝ worker count, not corpus size)
        ▼
        per thread: sample/extractor.py :: extract_peptides_from_text(text, doc_id)
        │
        │  1. Candidate sequence discovery
        │     - sample/nlp.py: Aho-Corasick automatons (pyahocorasick) over curated
        │       1-letter/3-letter known sequences and peptide names, plus regex patterns
        │       for IUPAC 1-letter/3-letter runs and alphanumeric peptide labels (LL-37, AMP-5)
        │  2. Sentence-level activity discovery
        │     - regex for MIC/IC50/EC50/... values + units, near a candidate sequence/name
        │     - Aho-Corasick lookups for organism names (from data/taxonomy.db) and
        │       cell-line names (data/cellosaurus_names.txt) to classify bacteria vs. cancer
        │  3. LLM verification (sample/llm.py :: verify_activities_llm)
        │     - all of a peptide's activities go in ONE call, checked against the text they
        │       were extracted from; fails OPEN, so a backend outage cannot empty the database
        │     - replaced a cross-encoder NLI gate (sample/bert_nlp.py, since removed) that was
        │       measured to reject 100% of activities while carrying no signal -- see 2.5b
        │  4. LLM fallback (sample/llm.py :: recover_peptide_llm)
        │     - only for candidates carrying a recovered SEQUENCE; name-only candidates are
        │       triaged separately (triage_candidates_llm) rather than each costing a call
        ▼
data/*_data_json/<source>/<id>.json   {"metadata": {...}, "peptides": [...]}
```

Every downstream stage — `sample-cli validate`, `sample-cli external-enrich`,
`sample-cli merge-extractions`, and the webserver's Extractions viewer — globs this directory
for one JSON file per document. That contract is now what `pipeline_runner.py` actually
produces (see §2.1 for what was broken before).

### 1.2 Enrichment (`sample-cli external-enrich`, `sample-cli enrich`)

```
data/*_data_json/<source>/<id>.json
        │
        ▼ sample-cli external-enrich (sample/external_db.py)
        │  ThreadPoolExecutor (I/O-bound): per peptide, RF-filter + DNA/protein check to drop
        │  false positives, then look up PubChem/ChEMBL for a SMILES structure, validated by
        │  Tanimoto similarity against the extracted sequence before accepting a cross-reference
        ▼
data/*_data_json/<source>/<id>.json   (same files, updated in place with SMILES/cross-refs)
        │
        ▼ sample-cli merge / merge-extractions / serialize  →  consensus CSV
        ▼ sample-cli enrich (sample/enrichment.py)
        │  1. enrich_with_descriptors: RDKit MolWt/LogP/HBD/HBA + net charge (Henderson-Hasselbalch)
        │  2. enrich_with_embeddings: 28-dim AA-composition vector + ProtBERT + ESM-2 mean-pooled
        │     embeddings (batched, GPU if available)
        │  3. enrich_with_2D_and_3D_structures: RDKit 2D coords + 3D distance-geometry embedding
        │     + MMFF94 force-field optimization, one SDF file per peptide (now parallelized,
        │     see §3.4)
        ▼
data/sample_enriched_release/enriched.csv, .../structures/*.sdf
```

---

## 2. What was broken, and what was fixed

### 2.1 The extraction → downstream contract was disconnected (P0)
Before this refactor, `pipeline_runner.py` wrote a single aggregate
`<source>_extractions.parquet` file. Nothing else in the codebase read it back — `validate`,
`external-enrich`, `merge-extractions`, and the webserver all glob for per-document JSON files.
Running `extract` produced output the rest of the pipeline literally could not see. This is now
fixed: extraction writes one `{"metadata": {...}, "peptides": [...]}` JSON file per document,
matching what every consumer already expects. This was fixed before any performance work,
because a faster pipeline that produces unusable output isn't actually progress.

### 2.2 `extract` was single-threaded despite an ignored `--workers` flag
`run_sequential_pipeline` was a plain `for _, row in df.iterrows()` loop. `--workers` (default
50) was accepted by the CLI parser but never passed through. Rewritten as
`run_parallel_pipeline`, backed by a thread pool (see §3.1 for the final design and §2.6 for
why a process pool was tried first and abandoned).

### 2.3 `screen` / `llm-screen` / `extract` were the same code path
All three CLI subcommands routed through identical logic — the standalone screening stage no
longer exists as separate code (screening/relevance filtering now happens implicitly via
candidate discovery finding nothing). `screen` and `llm-screen` were undocumented and provided
no distinct behavior, so they were removed. Only `extract` remains.

### 2.4 Dead CLI flags removed
`--prompt`, `--schema`, `--verify-prompt`, `--screen-text-prompt`, `--model`,
`--screen-text-model`, `--timeout`, `--use-bert`, `--bert-batch-size`, `--nli-threshold`,
`--v2`/`--legacy`, `--skip-screening` were all defined but never consumed by the actual
pipeline — a leftover from the old LLM-first architecture in `sample/llm.py` (see §2.5). They
were removed rather than silently ignored, since a flag that does nothing is worse than no flag.

### 2.5 ~1500 lines of dead extraction code removed from `sample/llm.py`
The file shrank from 2162 to ~660 lines. The deleted code was the old monolithic,
LLM-first extraction engine (`process_all_llm_extractions` and its exclusive helpers) from
before the pivot to the classical-NLP-first engine in `sample/extractor.py` — it was imported
by `cli.py` but never called, and internally still referenced a module
(`sample.classical_extractor`) that had already been renamed to `sample/extractor.py`, so it
would have crashed if anything had still invoked it. `sample/bert_nlp.py` similarly dropped
~150 lines of functions (`get_sentence_embedder`, `filter_high_density_paragraphs`,
`extract_peptide_synonyms`) with no callers anywhere in the codebase.

### 2.5b The NLI gate was removed outright
The cross-encoder gate described above is gone. It was measured on 25 real documents: it rejected
**0 of 84** extracted activities as entailed -- a 100% rejection rate -- and on hand-built pairs
TRUE claims scored 0.015-0.053 entailment while FALSE claims scored 0.034-0.354. The ranges
overlap, with the highest score belonging to a false claim, so no threshold could have separated
them. Sentence-pair entailment is simply not the question being asked: "does this text state that
this peptide had this value against this target?" is a reading-comprehension question, and an
instruction-following model answers it directly.

`sample/bert_nlp.py` has since been deleted. Nothing imported it, and its `SAMPLE_ENABLE_NLI`
escape hatch appeared only inside the file itself, so the flag had no caller left to re-enable.
Removing it also dropped the only use of `onnxruntime` in the project, and orphaned
`data/onnx_models/` (~553MB).

### 2.6 The process pool didn't actually save memory -- it made it worse
The first version of `run_parallel_pipeline` used a `ProcessPoolExecutor`, pre-warming the
Aho-Corasick automatons and dictionary/vocab sets in the parent process before forking, on the
theory that Linux's copy-on-write fork semantics would let worker processes share that memory
rather than each duplicating it. That's a real technique, and it's real for genuinely read-only
C-level data. Measured on a real run, it didn't hold: a single warmed-up worker process was
**~5.1GB RSS** in isolation (dominated by the NLI model, the ~2M-row taxonomy automaton behind
`get_organism_kp()`, and a spaCy vocab), and even with pre-warming, forked worker processes
still independently grew to **1.5-3.5GB each** rather than sharing the parent's pages. Running
with 7 worker processes exhausted 15GB of RAM and pushed the system into heavy swapping within
under a minute. The reason: CPython's reference counting touches (and thus copy-on-write-dirties)
the memory page behind *any* Python object a thread or process reads, not just ones it writes —
and extraction touches these lookup structures on every single document. COW sharing degrades
toward full duplication under continuous read access; it does not hold up for a workload that
constantly queries the "shared" structures, only for ones that build them once and barely touch
them again. **The fix was architectural, not tuning**: switch to a `ThreadPoolExecutor`. Threads
in the same process share memory outright — there is exactly one automaton, one vocab, one NLI
model, full stop, no copy-on-write gamble involved. The candidate-discovery and NLI-inference
hot paths run in C/C++ extensions (pyahocorasick, ONNX Runtime) that release the GIL during their
actual work, so this still parallelizes real CPU work; see §3.1 for how the resulting thread
count was chosen (empirically, not by assuming "more is better" — more threads made this
particular workload *slower*, not faster, once GIL contention set in).

### 2.7 Catastrophic regex backtracking in the two core sequence-matching patterns
`sample/nlp.py`'s `SEQ_EXACT` and `SEQ_3LETTER` (pre-existing patterns, not introduced by this
refactor) build a repeated-and-optional sub-pattern for each residue:
`(?:[DLdl]-?)?[ACDEFGHIKLMNPQRSTVWY]` — an optional D/L stereochemistry prefix followed by an
amino-acid letter. The bug: D and L are *themselves* valid amino-acid letters (Asp and Leu), so
for a run of ambiguous characters the regex engine has exponentially many ways to decide, for
each character, whether it was "the optional prefix" or "the actual residue." Applied via
`.finditer()` across a whole document (tens of thousands of characters, and real papers
routinely contain long runs of repeated/uniform letters from OCR artifacts, table formatting, or
gene-name runs), this is a textbook catastrophic-backtracking landmine. Measured directly: a
**30-character** run of a repeated ambiguous letter took **0.67 seconds**; a **45-character**
run didn't finish in **120 seconds**. This is what "not working" actually was, on top of the
memory problem in §2.6 — worker threads/processes weren't hung in the sense of a deadlock, they
were doing real (exponential-time) work on ordinary-looking real documents. **Fix**: wrap the
ambiguous optional group in an atomic group, `(?>(?:[DLdl]-?)?)` (Python 3.11+ supports atomic
groups natively). Atomic groups commit to their first successful match and never reconsider it
during backtracking, which eliminates the exponential blowup at the source without changing
which strings match — verified against a battery of legitimate sequences (all matched
identically) and against the same 45-character pathological input (now **<0.001s**, and a
5000-character version of the same input still resolves in under a millisecond).

### 2.8 A fresh LLM client on every fallback call
`sample/llm.py::recover_peptide_llm` (the LLM fallback for candidates the NLI model couldn't
verify) created a brand-new `OpenAI(...)` client -- with its own `httpx` connection pool -- on
*every single call*. With no local vLLM/Qwen endpoint running (the common case when extraction
is run without also standing up that endpoint), every call fails after a 1-second connect
timeout, but the repeated client/connection-pool allocation across many concurrent worker
threads compounded into real memory growth and, since every unverified candidate in every
document pays that ~1s timeout serially, severe slowdown. **Fix**: a single shared client
(created once, reused across every call and every thread) plus a 30-second circuit breaker --
after one failed call, subsequent calls return immediately without attempting a connection until
the cooldown expires, rather than re-discovering "the endpoint is down" on every candidate.
Measured effect: a batch that reached 29/200 documents in 8 seconds after this fix had taken
roughly a minute to reach the same point beforehand, on otherwise-identical runs.

### 2.9 Redundant full-document NLI re-chunking per activity
`verify_relation_entailment_batch` (§1.1 step 3) split `context_text` into overlapping windows
to search for chunks containing a candidate activity's value -- but it rebuilt that identical
chunk list from scratch inside the loop over each activity, even though the chunking doesn't
depend on the activity at all. Worse, the caller (`extractor.py`) was passing the *entire raw
document* as `context_text` rather than the ~4000-character context window it already computes
per peptide (`extract_context_window`) -- so a peptide with several activities in a 34,000-character
document re-sliced that whole document into ~23 chunks per activity, instead of ~3 chunks (once,
reused) from its own local context. Both are now fixed: chunking happens once per call, and the
call site passes the peptide's own context window. Combined with §2.7's regex fix, this brought
the single slowest document found during testing (47 candidate peptides, previously unable to
finish in over 90 seconds) down to **23.9 seconds** -- still the slowest document observed, but
now a bounded, finite cost rather than an effectively-unbounded one, and it's a genuine
computation cost (real NLI inference over an unusually candidate-dense document) rather than a
bug at that point.

---

## 3. What was optimized, and why

Hardware this was tuned for: 24 CPU cores, ~15GB RAM, 1× RTX 3060 (12GB VRAM). The guidance
below is written generally, but the specific defaults (worker clamp, batch sizes) assume a
similar single-GPU workstation, not a multi-GPU server.

### 3.1 Extraction parallelism: thread pool, default 4 workers
`--workers` drives a `ThreadPoolExecutor`, defaulting to `min(4, os.cpu_count())` — not the core
count. This wasn't the first design (see §2.6 for the process-pool attempt and why it was
abandoned) and 4 wasn't a guess either — it's what empirical testing on this 24-core/15GB
machine actually showed: this workload's per-worker footprint is dominated by several
gigabytes of shared, mostly-Python-level state (an ONNX NLI model, a ~2M-entry taxonomy
automaton, a spaCy vocab, GIL-bound sentence/regex processing), not raw CPU-bound work that
scales cleanly with core count. Going from 4 to 24 workers didn't increase throughput — it
caused severe GIL contention (individual documents that took ~1s standalone took over a minute
under 24-way contention) while linearly increasing the number of partially-processed documents
held in memory simultaneously. 4 threads reached steady throughput almost immediately with a
stable, bounded memory footprint (~11-12GB on this box, most of it the shared singletons —
see §2.6); if your hardware has less RAM than that, expect to need `--workers 1`. Submission is
windowed (2× workers in flight) regardless of worker count, so memory stays bounded by worker
count rather than corpus size. Raise `--workers` only after verifying on your own hardware that
it actually improves wall-clock throughput rather than just adding contention.

### 3.2 One model server, shared by every worker
*Historical note: this section described the shared in-process NLI model, which no longer exists
(see 2.5b). The reasoning still applies to what replaced it.*

Verification now goes to a local Ollama endpoint, so there is exactly one model instance regardless
of `--workers` — the same "one shared instance" property the in-process model had, but the memory
lives in the model server rather than in the pipeline. That moves the binding constraint: the
useful thread count is now set by how many requests Ollama can actually run concurrently
(`OLLAMA_NUM_PARALLEL`) and by whether the model's weights plus N KV caches fit in VRAM, not by CPU
cores. `sample/pipeline_runner.py` warns when it cannot confirm `OLLAMA_NUM_PARALLEL`.

An earlier version of this paragraph reasoned that gemma4 (9.6GB on disk) leaves "roughly 2GB for
KV cache, which supports about 1-2 concurrent slots", and that estimate is what kept the slot count
at its default for months. **It is wrong, and measurement contradicts it.** Only the text weights
are resident on the GPU -- the vision and audio projector are not -- so gemma4 occupies ~3.4GB of
VRAM, not 9.6GB. Running it at `OLLAMA_NUM_PARALLEL=4` with `-c 16384` (four slots x 4096) measured
**4,549MiB** of VRAM for `llama-server`, 5,249MiB including the desktop, against a 12,288MiB card:
7GB spare, and `ollama ps` reporting `100% GPU` with no CPU offload. Four slots were never the
constraint. Do not re-derive a slot budget from a model's file size.

**Measured 2026-10-13, and this analysis was right in a way nobody acted on.** A refined-extraction
run was left at Ollama's default `OLLAMA_NUM_PARALLEL=1`: `llama-server` ran with `-np 1`, every
log line read `slot id 0`, and over a 10-minute window the single slot was busy **97.7%** of wall
time serving **0.32 calls/s** while eight client requests (4 extraction workers x
`LANGEXTRACT_MAX_WORKERS=2`) were in flight. Seven of every eight were simply queued. Throughput
was 79 documents/hour, which is 20 days for the 46,189-document corpus. The GPU was not the limit
(93% busy, 147/170W, no throttling) and 7.4GB of VRAM sat unused.

Two changes follow from it, and they compound:

* **Give the server slots.** `OLLAMA_NUM_PARALLEL=4` in the systemd drop-in. Batched decoding also
  amortises the per-step weight read, which is what generation is bound by here (58.7 tok/s single
  stream against a ~360GB/s card).
* **Stop paying for weights this path never uses.** The refined pass is the one stage where model
  size dominates wall clock, and gemma4 is an 8B **vision+audio** model used for pure text
  extraction: 9.8GB resident for 4.2GB of GPU work, which pushed 11GiB into swap on a 15GB
  machine. `LANGEXTRACT_MODEL=gemma3:4b` (~3.3GB) is the same family, so the worker's tuned prompt
  and chat template still apply, and it frees both the RAM and the VRAM the slots need. Not
  `qwen3:4b`: `sample/llm.py` lists `qwen3` and `gemma4` as reasoning families that must have
  thinking disabled or they return empty, and the LangExtract worker cannot pass `think` -- it runs
  in an environment that cannot import `sample.*`.

With a ~3.3GB model the KV-cache arithmetic above lifts: four slots at 4096 context fit
comfortably, which is why the two changes are a pair rather than alternatives.

CUDA is usable on this machine (the earlier note here, that the driver was older than the installed
CUDA-built PyTorch, no longer holds); the `llama-server` measurements above are GPU-resident.

**If extraction volume grows enough that NLI/LLM-fallback throughput becomes the bottleneck**,
see §5 for the producer/consumer GPU-batching design that would help — not implemented here
because it wasn't yet the bottleneck relative to the correctness bugs in §2.6-2.8.

### 3.2b Chunked execution: the throughput decay nobody could explain

**Measured 2026-10-16.** With the slot count fixed, the context window pinned and a single runner
confirmed (1 `llama-server` start, 0 evictions over an 8-minute window, 563 × 200 against 3 × 500),
a long refined-extraction run still slowed down — from ~1 doc/s to **~0.04 docs/s after about 2,000
documents**. No error was raised, no failure logged, and a manual `ollama` restart restored full
speed immediately.

That shape — gradual, silent, reversible by restarting one process — is state accumulating inside
the serving process, not a bug in the client. Everything on the client side is bounded and
observably drains: the thread pool, the submission window, the per-document langextract
subprocesses. `llama-server` is the only component that spans an entire multi-day run while holding
a KV cache per slot and a request queue that Ollama exposes no endpoint for.

**The response is to bound the cause rather than identify it**, because it is not observable from
here and three prior attempts to tune this pipeline by reasoning about the server instead of
measuring it cost ~680 documents of degraded output. Extraction now runs in chunks of
`SAMPLE_EXTRACT_CHUNK_SIZE` (100) documents; between chunks the loaded model is unloaded
(`keep_alive: 0`), which terminates the runner, and the next chunk's first request starts a fresh
one. Runner-side state can therefore accumulate for at most one chunk.

Cost: one model load per chunk — a few seconds for a 3.3GB model against a chunk that takes minutes,
well under a percent. The chunk boundary is a real barrier: a fresh `ThreadPoolExecutor` per chunk,
joined before the unload, because unloading the model while a worker still has a request outstanding
would hand that document a hard failure for no reason of its own.

**What is NOT claimed: the decay is bounded, not explained.** The per-chunk rate lines now in the
log are what distinguishes the remaining possibilities. If the rate still decays *within* a single
100-document chunk, the cause is client-side after all (candidates: `lru_cache` growth, the taxonomy
automaton's working set, or RSS pressure swapping the model's anonymous pages — Ollama does not mmap
weights under CUDA on this host). If each chunk starts fast and ends slow, repeatably, the runner is
accumulating and a smaller `CHUNK_SIZE` is the lever. See KNOWN_ISSUES.md.

### 3.2c The per-document budget, and the timeout that never fired

Each document gets `SAMPLE_DOCUMENT_BUDGET_SECONDS` (300) of wall clock. Implementing it exposed
that **`llm_call`'s existing timeout did not bound anything**: it ran inside
`with ThreadPoolExecutor(...)`, and that context manager's `__exit__` calls `shutdown(wait=True)` —
so `future.result(timeout=...)` raised `TimeoutError` and the block then sat there waiting for the
very call that had just timed out. Callers were billed the full duration of a call they had already
given up on. Fixed by shutting the executor down with `wait=False`.

The budget is not a timeout in the usual sense, and the distinction is in the data it produces:

* It **cannot abort a document.** CPython cannot interrupt a thread, and abandoning the future would
  leave the worker running and still holding an Ollama slot — the exact backlog a timeout exists to
  prevent.
* It **refuses new expensive work.** Each LLM call is clamped to the remainder and refused below a
  5-second floor; a call issued with 2 seconds left cannot return in time, but Ollama finishes the
  generation anyway after the client gives up, so it costs the *next* document a slot. The
  langextract subprocess — the one thing here that genuinely can be killed — is hard-capped.
* It is **clamped in `llm_call`, not at each call site.** Six LLM layers each honouring a 90-second
  timeout is a 540-second document, so a nominal 300-second cap enforced per-call would bound
  nothing.
* The budget is **thread-local** (`sample/deadlines.py`), since one document's work spans six
  modules and about a dozen call sites; a deadline parameter would be correct only until a caller
  forgot one, and a forgotten one fails silently by reverting to unbounded.
* Budget refusals are **exempt from the circuit breakers**. They open after three *consecutive*
  failures, globally, for 30 seconds — and one over-budget document refuses several calls in a row,
  which would disable a verification layer for every document in flight beside it.

A document that hits the cap is **written, with fewer records, flagged `metadata.budget_exceeded`**
beside `processing_seconds`. Not dropped: a missing output file is indistinguishable from "not
reached yet" and would be re-attempted forever by `--resume`, while a flagged one can be counted,
audited and re-run deliberately.

### 3.3 Enrichment: pooled HTTP sessions, retries, rate limiting, persistent cache
`sample/external_db.py`'s `fetch_pubchem`/`fetch_chembl` used bare `requests.get()` calls: a new
TCP+TLS handshake per lookup, no retry on transient 429/5xx, and caching via
`functools.lru_cache` that's lost the moment the process exits — every re-run of
`external-enrich` re-fetched every peptide from scratch. Now:
- A shared `requests.Session()` with an `HTTPAdapter`/`Retry` (3 retries, exponential backoff,
  retrying on 429/500/502/503/504) gives connection pooling and resilience to transient failures.
- A simple per-host rate limiter caps PubChem to ~5 req/s (their published PUG-REST guidance)
  and ChEMBL to ~10 req/s, so a `--workers 10`+ run doesn't get throttled or banned.
- An on-disk SQLite cache (`data/external_db_cache.sqlite`) sits under the existing
  `functools.lru_cache`, so repeated or resumed enrichment runs skip network calls entirely for
  peptides already resolved — consistent with the project's existing use of SQLite elsewhere
  (`taxonomy.db`, `sample.db`).

### 3.4 Enrichment: parallel RDKit 3D structure generation
`enrich_with_2D_and_3D_structures` (embedding + MMFF94 optimization per peptide) was a serial
per-row loop — pure CPU-bound native code, embarrassingly parallel across peptides. It's now a
`ProcessPoolExecutor` mapping over `(entry_id, smiles, sequence)` tuples (cheap to pickle); each
worker builds its own RDKit `Mol` and writes its own SDF file independently.

### 3.5 Enrichment: embedding batch size, precision, and redundant work
`enrich_with_embeddings` computed `raw_sequences` via `df.iterrows()` twice — once for ProtBERT,
once for ESM-2. It's computed once now and reused. Batch size was hardcoded to 16 regardless of
device; both ProtBERT and ESM-2-8M are small enough that a 12GB GPU comfortably handles larger
batches, so batch size is now 64 on CUDA (16 on CPU, unchanged), with `torch.float16` on CUDA to
roughly halve activation memory and increase throughput, and a fallback that halves the batch
and retries on `CUDA OutOfMemoryError` instead of aborting the whole enrichment run.

---

## 4. Getting to a high-quality AMP/ACP dataset

Speed doesn't matter if the output is noisy. The extraction design already leans toward
precision over raw recall in a few specific ways worth understanding when tuning it:

**Sequence validity gate (`sample/nlp.py::is_valid_sequence`).** Every candidate sequence is
checked against: an acronym blocklist (~150 lab-jargon terms like `MIC`, `PBS`, `EDTA`), a
false-positive word list (~150 English words that happen to match the amino-acid-letter
alphabet, like `FACT`, `CAT`, `PEPTIDE`), OS dictionary + SciSpaCy PubMed vocabulary + UMLS
lexicon membership checks (skipped only if the token has explicit stereochemistry/terminal
modification markers, since those are strong peptide signals), suffix blocks for antibiotic
classes (`-MYCIN`, `-CILLIN`, etc.), and low-complexity/homopolymer rejection. This is the first
and cheapest line of defense against false positives — most noise gets filtered before any
model inference happens.

**Sequence hallucination defense (`sample/rf_filter.py`).** A Random Forest classifier trained
on 27-dim amino-acid-composition features (length + per-letter frequency) discriminates real
peptide sequences from English words and surnames that happen to consist entirely of valid
amino-acid letters (the classifier's negative class is specifically "hard negatives" — dictionary
words and surnames that already pass the regex/format checks). Applied both to LLM-extracted
sequences (`external_db.py::enrich_single_json`) and available as a standalone gate
(`train-rf-filter`/`filter_sequence`). Retrain it (`sample-cli train-rf-filter`) whenever the
consensus positive set changes meaningfully — it's trained against
`data/sample_raw_release/merged_from_database.json`.

**DNA/protein disambiguation (`external_db.py::predict_sequence_type`).** A log-likelihood
classifier over amino-acid vs. nucleotide frequency distributions catches the specific failure
mode where a gene/nucleotide sequence gets misidentified as a peptide sequence (also checked
directly in `sample/llm.py::validate_extraction_logic` for LLM output).

**Activity grounding, not just sequence validity.** A candidate sequence isn't enough — the NLI
verification step (§1.1, step 3) is what prevents "peptide X was mentioned in a sentence near a
number" from becoming "peptide X has activity value N". This is the most important lever for
precision: `verify_relation_entailment_batch`'s `threshold=0.85` in `extractor.py` is
deliberately strict. Lowering it will recover more activities (higher recall) at the cost of more
spurious (sequence, activity) pairings; if recall turns out to be the bottleneck rather than
precision, that threshold — not more LLM fallback — is the first knob to try.

**Metric-type coverage (`sample/nlp.py::detect_activity_type`).** The sentence-level activity
loop originally classified `activity_type` with a two-way check (`"MIC" in sent`, else `"IC50"`,
else `"unknown"`) even though `nlp.py` already defined a much broader metric taxonomy
(`METRIC_PATTERN_EXACT`: MIC/MIC50/90/100, IC50/90, EC50/90, MBC/50/90, MFC, CC50, LC50, LD50,
ED50, ZOI) for an unrelated screening step. `detect_activity_type` now reuses that same taxonomy
for activity classification, plus a descriptive-phrase fallback (`METRIC_PATTERN_IGNORE`-style
phrases) for sentences that say "growth inhibition" or "zone of inhibition" without ever writing
the bare acronym. This is purely a recall improvement for the activity-type field — a `MIC90` or
`EC50` measurement that used to get recorded as generic `"MIC"`/`"IC50"`/`"unknown"` now keeps
its actual reported metric, and `target_group` classification (below) now runs on that more
accurate type instead of a crude bacteria/cancer-only heuristic that didn't use
`infer_activity_group` at all despite it already existing for exactly this purpose.

**Antimicrobial vs. anticancer classification (`sample/nlp.py::infer_activity_group`).** Cell-line
names (via the Cellosaurus keyword list) are always classified `anticancer`; everything else
falls through a keyword taxonomy (`get_organism_group_kp`) covering antibacterial/antifungal/
antiviral/antiparasitic/anticancer, with a final fallback on metric-type keywords (`MIC`/`MBC` →
antibacterial, `IC50`/cytotoxicity → anticancer). This is deterministic and fast, but it's a
keyword taxonomy, not a classifier — it will misclassify organisms/cell lines not in its lists.
**Current limitation worth flagging**: the cell-line and organism/pathogen lists
(`data/cellosaurus_names.txt`, `data/organism_names.txt` + `data/taxonomy.db`) directly bound
recall for activity-group classification; a peptide tested against an unlisted cell line or a
pathogen not in the taxonomy DB will fall through to the generic keyword heuristics or `"other"`.
Expanding those reference lists (particularly cancer cell lines, which are a long tail) is
higher-leverage for anticancer-peptide recall specifically than any pipeline performance change.

**Deduplication and merging** happens downstream in `sample/merge.py` (sequence-level dedup
using `sample.sequence.sequence_to_1_letter` as the join key), not in the extraction stage
itself — extraction intentionally does not deduplicate across documents, so the same peptide
reported in multiple papers accumulates multiple `LITERATURE`/`Reference` provenance entries
rather than being silently collapsed.

---

## 5. Not implemented in this pass (forward-looking recommendations)

Being explicit about scope so these don't get assumed-done:

- **RAM is the binding constraint on this pipeline, not CPU.** Expect the extraction process's
  steady-state footprint to sit around 11-12GB regardless of `--workers` (it's dominated by
  shared singletons, not per-worker duplication -- see §2.6). On a machine with less RAM than
  that, extraction may need `--workers 1` to stay stable, and everything else running on the box
  competes with it. If this pipeline needs to run on smaller machines routinely, the taxonomy
  automaton (`get_organism_kp()`, ~2M rows from `data/taxonomy.db`) is the largest single
  reducible cost -- most of that row count is almost certainly not relevant to AMP/ACP literature,
  and filtering it down before building the automaton (rather than after) would directly cut the
  dominant memory cost.
- **GPU batching microservice for LLM fallback / NLI.** If corpus volume grows to the point
  where NLI inference or serialized LLM fallback calls become the throughput bottleneck (they
  were not, in this pass -- correctness bugs were), the right next architecture is a
  producer/consumer queue: worker threads do candidate discovery and push (sequence, context)
  pairs to a dedicated inference loop that batches across documents rather than per-document.
- **Async LLM fallback calls.** `recover_peptide_llm` still makes a synchronous network call per
  unverified candidate (mitigated, not eliminated, by the §2.8 circuit breaker). Since `aiohttp`
  is already a project dependency, decoupling network concurrency (want: high, e.g. 50+
  concurrent requests) from the extraction thread count via `asyncio`/`aiohttp` would let
  LLM-fallback-heavy corpora scale network concurrency independently of CPU/GIL constraints.
- **Dependency pinning.** `onnxruntime-gpu`, `optimum`, `pyahocorasick`, and `vllm` are installed
  in the environment and load-bearing for the extraction pipeline, but aren't declared in
  `environment.yml`/`pyproject.toml`. A fresh environment built from those files today would be
  missing them (NLI would silently degrade to its no-op fallback rather than erroring loudly).
  Worth fixing before this pipeline is run anywhere but this machine.
- **Full-corpus benchmarking, still.** Even the expanded validation in this pass (§2.6-2.9) only
  covered ~200 documents, not the full ~22,000-document pubmed source or the other sources. The
  bugs in §2.6-2.9 were found specifically *because* testing moved past small `--limit` smoke
  tests to a few hundred real documents -- there is no reason to assume 200 documents exhausts
  the space of documents that could be slow or unusual. Treat a full corpus run as still
  partially unvalidated territory, and watch memory/throughput the first time it's attempted at
  full scale.
