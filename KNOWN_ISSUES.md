# Known issues and limitations

Things that are not actionable right now, either because the data does not exist, or because the
next step needs a decision only the project owner can make. See `TODO.md` for what *is*
actionable, and `DECISIONS.md` for the reasoning behind decisions already made.

## There is no extraction "slowdown" -- 0.27 docs/s is the baseline

Settled 2026-10-16, after **two wrong hypotheses of mine** (runner-side state accumulation, then
host memory thrashing). Both were falsified by the same experiment: the collapse happens at
chunk 4 of every run, in the same place, whatever the configuration.

| run | chunk 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `-np 4`, f16 KV, 8.5 GB llama-server, swap thrashing | 89s | 102s | 97s | **322s** | 470s | 353s |
| `-np 2`, q8_0 KV, 0.9 GB llama-server, no thrashing | 98s | 111s | 90s | **414s** | | |

The memory fix worked on its own terms -- `llama-server` RSS fell 8,500 MB to 910 MB, free RAM
rose 185 MB to 5,328 MB, swap-in fell from 488-946 KiB/s to 29-43 with swap-out at zero -- and it
changed the throughput profile not at all.

**What the collapse actually is: the model starting to run.** Model calls per chunk, current run:

| chunk | secs | `/api/generate` | `/api/chat` | calls/doc |
|---|---|---|---|---|
| 1 | 98 | 9 | 5 | 0.14 |
| 2 | 111 | 11 | 31 | 0.42 |
| 3 | 90 | 2 | 42 | 0.44 |
| 4 | **414** | **44** | **63** | **1.07** |

`/api/generate` is langextract. In chunks 1-3 it barely ran -- those documents are served from the
payload cache or skipped by `LANGEXTRACT_GATE` -- and in chunk 4 it ran on ~44% of them. Per-call
latency is healthy in both regimes (~4-7 s/call at 2 slots); only the call VOLUME changes.

So the fast phase was never a baseline. It is the cached prefix of the corpus, and
**0.27 docs/s is the real cost of refined extraction** -- measured over 26 chunks spanning three
runs, mean 366 s per 100 documents, which puts the 22,050-document corpus at **~22 hours**.

This is the same error this file already warns about ("do not quote a docs/hour figure from a
window shorter than 30 minutes"), committed again in a new form: a long window is not enough if
it sits entirely inside the cached prefix. A throughput figure for this step is only meaningful
from chunks where `/api/generate` is actually being called.

**What was genuinely fixed along the way**, and is worth keeping: the host no longer swaps, and
the HTTP 500 rate that came with the memory pressure is gone from the client's point of view
(bounded retry in `sample.langextract_records`). Chunking, however, is not earning its cost --
it bounds a decay that does not exist. `CHUNK_SIZE=0` turns it off; the per-chunk rate lines are
its only remaining value, as a time series.

## Superseded: the host-thrashing explanation

Kept for the measurements, which stand even though the conclusion drawn from them did not.
Measured during a live run before the Ollama memory settings were applied:

```
Mem:  15Gi total, 13Gi used, 185Mi free        Swap: 29Gi total, 5.8Gi used
vmstat si: 488-946 KiB/s continuous swap-IN
RSS: llama-server 8.5-9.3 GB   sample-cli 1.9 GB
```

`llama-server` holds **8.5-9.3 GB of host RSS for a 3.3 GB model that is 100% on the GPU**
(`ollama ps` reports `size_vram` = the whole model). The excess is Ollama 0.35's prompt cache:
the journal shows `saving prompt with length 1716, total state size = 149.548 MiB`, context
checkpoints of 75-116 MiB each, `3 of 32` per slot, and 65 × `making room for prompt cache
entry, removing oldest entry (220-440 MiB)` in twelve minutes.

One cause, four symptoms:

- **The 3-4x slowdown.** Chunks 1-3 run at ~1.0 docs/s, then every later chunk at 0.21-0.31
  docs/s, reproducibly, in two independent runs. Once the extraction process's pages are evicted
  it page-faults on every document, and 5.8 GB stays in swap.
- **The 16% HTTP 500 rate on `/api/generate`** (7 of 43 calls; zero of 172 on `/api/chat`). A
  prompt-cache allocation of 150-460 MB cannot be satisfied with 185 MB free, and the request
  500s after 6-22 s.
- **Why the chunk reset does not help.** The reset genuinely frees the prompt cache -- RSS
  oscillates 8.5-9.3 GB across chunk boundaries -- but it cannot un-swap `sample-cli`'s pages,
  so the damage is persistent. This is why speed does not recover after a reset.
- **Why restarting `ollama` by hand "fixed" it before.** It did not; restarting the *pipeline*
  freed the resident set.

**Ollama 0.35.0 has no prompt-cache size setting** (checked `ollama serve --help`). The levers
that remain are indirect, and none has been measured yet:

- `OLLAMA_NUM_PARALLEL=2` instead of 4 -- the cache is per slot, so this should roughly halve it.
  Costs some batching throughput, but the GPU is already the bound at 27-30 tok/s.
- `OLLAMA_KV_CACHE_TYPE=q8_0` -- halves the byte size of the KV state and therefore of every
  checkpoint.
- Target: keep llama-server + sample-cli + desktop under ~12 GB so nothing swaps at all.

Try them one at a time and compare the per-chunk rate lines, which now give a clean time series.
Do not infer success from the first three chunks -- those are fast under every configuration
tried so far.

## The model was copying its own prompt into its output (fixed: gate 4)

Found 2026-10-16 by the per-activity logging, and the most damaging defect found in this
pipeline so far, because a fabricated measurement is indistinguishable from a real one by
inspection: a real assay type, a round value, a real cell line.

The LangExtract worker's few-shot example is the sentence

```
"Peptide JG436 inhibited MCF-7 cells with an IC50 of 12.5 uM."   # _langextract_worker.py:74
```

Measured over one run's activity log, the tuple **`IC50 / 12.5 uM / MCF-7` was emitted for 49
different documents**. The next most repeated measurement in the same corpus appeared in 3
documents, which is what genuine coincidence on a common value looks like. Checking those 49
documents' text:

| | documents |
|---|---|
| contain neither "12.5" nor "MCF" | **38** |
| contain "12.5" only | 8 |
| contain "MCF" only | 2 |
| contain both | 1 |

The example's peptide name, `JG436`, also appears in the output as an extracted peptide.

**Fixed by gate 4** (`document_quality.activity_supported_by_text`): a measurement whose number
is nowhere in the document is dropped, counted in `metadata.dropped_activity_not_in_text`, and
recorded in `metadata.dropped_activities` so the call can be audited. Applied over the same log,
it drops **543 of 4,363 measured activities (12.4%)**.

Scoping decisions, both deliberate:

- **The target is not checked.** Organism normalisation rewrites it on purpose -- a paper saying
  "E. coli" yields `Escherichia coli` -- so the extracted target legitimately differs from the
  document's wording. Value-only catches 40 of the 49 fabrications; adding the target catches 46
  at a false-positive cost not worth paying.
- **`table_match` records are exempt.** Their values come from the PDF via docling, so a number
  genuinely in a table need not appear in the pdfplumber `.txt`.
- **The activity is dropped, not the peptide.** A peptide can be real and correctly named even
  when a measurement attributed to it was invented.

**Still open: the prompt itself.** Gate 4 catches the symptom on both new and already-cached
payloads, which is why it repairs the existing corpus without re-running the model. It does not
stop the model emitting the fabrication. Changing the few-shot example would, but it changes
`_langextract_worker.py`, whose digest is part of the payload cache key -- so it invalidates
every cached payload and implies a full re-extraction. That is a multi-day job and a separate
decision. If it is done, the example should use values that could not be mistaken for data and
that are cheap to grep for.

## The extracted sequence field was never validated

Found 2026-10-16 by the per-activity logging, and fixed in
`sample.sequence.validate_extracted_sequence`. The identity gate used to ask only whether the
sequence field was non-empty, so anything in it counted as an identified peptide.

Measured over 1,421 unique peptide records from one run, **88.4% would now be withheld**:

| reason | share |
|---|---|
| the model copied the NAME into the sequence field (`Cholesterol`/`CHOLESTEROL`, `Ricin`/`RICIN`) | 49.3% |
| under 5 residues (`ALA`, `MYR`, `G`, `N = 17`) | 18.6% |
| contains digits (`C2H5`, `RC-160`, `7.515.022.530.07,515...`) | 13.0% |
| whitespace-separated prose | 2.7% |
| an English word or a nucleotide run | 2.4% |
| non-residue characters | 2.1% |
| OCR-noised nucleotide read | 0.2% |

**These are withheld, not deleted** -- they go to `data/literature_data_json_withheld/` with a
`withheld_reason` on each record, so the judgement can be revisited without re-running the model.

**That gap is now closed too** (`sequence.name_likelihood`). A compound name spelled entirely in
residue letters -- `ANTIMYCINA`, `SANDOSTATIN`, `ENKEPHALIN`, `LINOLEOYL`, `METHENAMINEMANDELATE`
-- passes every structural test, because every letter in it IS a residue, and the dictionary gate
in `sample.nlp` only catches the ones that are ordinary English words. Letter-PAIR statistics
separate them where composition does not: English and chemical coinages are built from
TH/ER/IN/ON/TI, peptides from LL/KK/GG/AA/KL/GI.

Modelled on the existing `nlp.dna_log_likelihood_ratio`, which does the same for nucleotide runs
with a unigram model; bigrams are needed here because the confusable strings draw on the same
alphabet with a similar composition. Fitted from this project's own catalogue (40,000 sequences)
against the 17,300 words in `/usr/share/dict/american-english` spelled only in residue letters,
and committed as `data/peptide_name_bigrams.json` (6.7 kB) so the gate is on in a fresh clone.
Regenerate with `sample-cli train-sequence-discriminator`.

Held-out separation: peptide median **+1.99**, name median **-1.27**. The threshold is **-0.5**,
not 0, and the asymmetry is deliberate -- losing a real sequence is worse than keeping a name
that other gates will catch. At -0.5, 0.1% of real sequences are rejected and ~5% of names
survive. All 17 offenders observed in the run score between -0.61 and -2.05; the hardest real
case, `ACYCRIPACIA` (the human neutrophil peptide 1 core, which `sample.rf_filter` rejects),
scores -0.05 and survives with margin.

It is applied **last**, after every structural check, because it is the only statistical test in
the validator and so the only one that can be wrong about a real sequence.

**Current totals** over 6,946 unique peptide records from the run's log: **86.6% withheld**,
13.4% published.

| reason | share |
|---|---|
| the model copied the NAME into the sequence field | 42.5% |
| contains digits (formula, catalogue code, table label) | 24.4% |
| under 5 residues | 8.3% |
| non-residue characters | 5.2% |
| an English word or a nucleotide run | 2.6% |
| whitespace-separated prose | 2.2% |
| a compound name in residue letters | 1.2% |
| OCR-noised nucleotide read | 0.1% |

**`sample.rf_filter` is deliberately NOT used as the gate.** Measured on this corpus it rejects
6 of 11 junk strings but also rejects `ACYCRIPACIA`, the core of human neutrophil peptide 1. A
classifier that deletes genuine published sequences to catch half the junk is the wrong trade
for a gate.

## Three documents fail the model pass with HTTP 500, cause unknown

`ae6525c4-c9e3-5c67-c67c-ee70afa25265ff40a7ad`, `bafd1082-8eec-0524-d3f0-302da5368c1c03eead28`
and `714ef825-999b-616d-d2a5-ae8d85ea49d7805026de` have returned 500 in every configuration tried
(300s timeout, 600s timeout, and the current pinned 4096-context runner). The Ollama log around the
failure shows a normal slot lifecycle -- `n_ctx_slot = 4096`, `n_keep = 4`, `truncated = 0`, clean
release -- and no error message.

Each 500 silently costs that document its entire LLM pass. At 3 in 260 documents (~1%) that
projects to ~250 documents across the corpus extracted without the model. Now isolatable: the
eviction noise that used to mask it is gone, so running one of these three through
`sample/_langextract_worker.py` alone would surface the real error. Worth doing between runs, not
during one -- it competes for the same slot.

## The released database is built from DRAMP 4.0, not 5.0

DRAMP 5.0 was published 15 October 2026 (*Nucleic Acids Research*, doi:10.1093/nar/gkag989):
32,296 entries against 4.0's 30,260, and 22,298 curated SMILES annotations that 4.0 did not have.

The code to pick it up is in place (`sample-cli databases --targets dramp --refresh`, which
archives the existing tree and writes a manifest), and the parsing already handles the new column.
**The data has not been fetched.** Two reasons it needs a deliberate decision rather than a
default:

1. The fetch replaces the dump the current release database was built from. The refresh archives
   it to `dramp_4_0_<date>/` rather than overwriting, so this is recoverable — but the database
   and its sources would be out of step until the downstream chain is re-run.
2. That re-run is not small: merge → activity load → enrichment → the final dataset. The gain is
   2,036 new entries (+6.7%) and curated chemistry for modified peptides; it is not urgent, and it
   should not land in the middle of an extraction run.

Check what is actually on disk at any time with `make sources-status`.

**A trap to avoid when this is done.** DRAMP 5.0 introduces a FASTS format whose sequence lines
replace non-standard residues with `X`. `sample.merge` drops every sequence containing `X`
(`if 'X' in sequence: continue`), so reading the FASTS or FASTA files instead of the `.txt`
exports would silently discard exactly the modified peptides this release is about. The `.txt`
exports keep the original notation and are what the pipeline reads.

**An upstream labelling discrepancy, unresolved.** The paper's Table 1 counts the "specific"
category as 412 entries in 5.0 (377 in 4.0) and labels it the stapled dataset, while the
`specific_amps.txt` download holds 6,286 rows and `Subclass/stapled_amps.txt` is a separate file.
The two are not the same population, so release detection deliberately keys on `general` and
`patent` only. Worth a note to the DRAMP authors.

## Blocked on external data

- **Anticancer peptide databases — three of four imported; only ApInAPDB is still blocked.**
  Re-checked twice as hosts came back: `webs.iiitd.edu.in` answers again (2026-08-13) and
  DCTPep/CancerPPD published usable bulk downloads (2026-08-17). `sample.anticancer` /
  `sample-cli import-anticancer` normalises **AntiCP2, DCTPep and CancerPPD** into
  `data/anticancer_databases/anticancer_peptides.csv` — **7,533 unique sequences, 5,730
  experimentally-validated anticancer positives** plus AntiCP2's 1,803 labelled hard negatives.
  That is the data the broad anticancer model was starved of (see the cell-line note below).
  DCTPep and CancerPPD were unreachable when first attempted and were rechecked rather than
  assumed (2026-08-17): DCTPep now publishes a bulk `.xlsx`, and CancerPPD six id/sequence lists
  split by chirality and natural/non-natural. CancerPPD's *measurements* live only on its per-entry
  detail pages, so `scrape_cancerppd_details` fetches them — rate-limited, resumable, and never run
  implicitly (~3,100 requests against an academic server).
  **ApInAPDB remains not importable**: `bioinf.modares.ac.ir` does not respond at all (the
  connection times out, rechecked 2026-08-17), so there is no format to inspect and this project's
  rule against shipping a parser guessed at an uninspected format applies.
- **Cell-line-specific cancer models: not enough data yet.** `target_cell_line` *is* populated now
  (~330 rows across 106 distinct spellings), and `sample.cell_lines` classifies each via Cellosaurus
  into a cancer type or normal tissue. But even pooled by cancer type, the best category
  (leukemia/lymphoma) reaches ~28 positives against `MIN_POSITIVES_PER_TASK` = 60, so no
  cell-line/cancer-type model clears the quality floor. The code is ready; the unblock is the
  anticancer-database imports above, not implementation. (Supersedes the earlier "Cellosaurus links
  / target_cell_line 0% populated" entry, which is stale.)
- **Synergy prediction against anticancer drugs.** DBAASP's synergy partners are all antibiotics;
  no peptide/anticancer-drug combination data exists in any source this project reads.

## Needs a compute run, not new code

- **Synergy models must be retrained** to use the antibiotic Morgan fingerprint
  (`sample.antibiotics`) instead of the old one-hot column — `sample-cli train-synergy`, which is a manual/diagnostic command and is NOT one of the numbered `make all` steps.
  The change is in `synergy.featurise`; the shipped models still carry the old feature scheme.
- **The rigorous `train_all` (step 45) has not been run on the real dataset.** It is now wired into
  the CLI/Makefile and verified end-to-end on synthetic data, but a full release run over ~74k rows
  is multi-hour and needs a go-ahead, same as any compute-heavy step.
- **The reference-tool benchmark (`sample-cli benchmark`) has produced no numbers yet.** The harness
  (`sample.benchmark`) is complete and unit-tested with the tool mocked, but a real comparison needs
  both the trained models (the `train_all` run above) and the reference tool installed in
  `sample-predictors` (HemoPI2 is the one confirmed-working tool). `REFERENCE_TOOLS` covers only
  hemolytic→HemoPI2 today; macrel and toxinpred3 join once their environments are built.

## Latent NameErrors: a class of bug worth re-scanning for

Step 18 crashed on 2026-08-18 with `NameError: name 'unreadable' is not defined` in
`load_dramp`, killing the whole integrated activity table. The cause was two lines copy-pasted
from the DBAASP loader (they even said "DBAASP:" inside the DRAMP function) referencing names
that do not exist in that scope. It survived unnoticed because it sits *after* the loop's
`limit` early-return: every bounded/smoke run returned before reaching it, and only a full
production run got there. DRAMP's 36,366 records and 17,775 measured activities were being lost
entirely as a result.

An AST scan of the package for the same class of bug found one more real instance --
`sample/cli.py` called `np.save()` in `enrich-esm2` without `numpy` imported anywhere in the
file, so that command died *after* computing all its embeddings. Both are fixed.

Worth repeating that scan after any large merge or refactor: nothing in the test suite catches
this, because the failing paths are the ones only a full, unbounded run reaches. A linter in CI
(`pyflakes` is not currently installed in any environment) would catch the whole class cheaply.

## Data quality: one degenerate family cluster

`SAMPLE_FAM_000001` (`data/families/hmms/SAMPLE_FAM_000001.hmm`) has 2,559 members — 27% of every
peptide this project clustered, in one cluster — and its profile HMM does not reliably score even
its own member sequences above `hmmsearch`'s default threshold, unlike every other cluster tested.
Very likely the same clustering failure already on record in `DECISIONS.md` (a flat mismatch
budget chaining unrelated short peptides into one component) rather than a real shared family.
Already excluded from HMMER search results (`_UNRELIABLE_FAMILY_HMMS` in `sample/api/common.py`),
so a confident-looking but meaningless match is not reachable through that path. Re-clustering to
split it apart is a separate `sample/families.py` change, not attempted this session.

## 105 microbial-assay activities name an animal or a plant as their target

After `apply-verdicts` (step 12's second half), 22,376 activities carry an NCBI taxonomy id. 105 of
them pair a microbial assay — MIC, MBC, MFC, ZOI — with a target whose NCBI domain is animal or
plant: `Homo sapiens` 38, `Crambe glabrata` 20, `Ovis aries` 6, `Mus musculus` 6. A minimum
inhibitory concentration is measured against a microbe, so each of these is wrong.

These are **extraction** errors, not normalization errors, and `apply-verdicts` deliberately does
not touch them. Its domain guard refuses a *proposal* — a value it would write where there was
none — and it refused 211 on exactly these grounds, almost all of them the peptide's source
organism (frogs, bees, silkworms) offered as its target. But where the extractor itself chose the
organism, this stage only canonicalizes the spelling and attaches the id; silently deleting the
extractor's choice would hide an extraction defect rather than report it. The cost is that the id
makes the bad row look more curated than it is.

Fixing it belongs in extraction or in `validation_rules`, and `validation_rules.TARGET_GROUP_DOMAINS`
cannot currently express it: every `ActivityRule` lists the `other` catch-all among its groups and
`other` carries no domain restriction, so the permitted-domain union is unrestricted for every
assay. The catch-all is right for hemolysis and in vivo dosing, which do have animal targets — it
just cannot also serve as a guard. Either those rules need a separate "target must be a microbe"
flag, or the microbial assays need their `other` group removed.

Related, already fixed: `nlp.classify_organism_domain("Human immunodeficiency virus Type 1")`
returns `animal`. It falls back to the first token when the full name is absent from
`organism_names`, and that token is "human", which resolves to *Homo sapiens*.
`verdict_apply.domain_permits_target` works around it locally by matching `virus`/`phage` in the
name first; the shared function is untouched because validation also uses it.

## Annotator: which of the named tools are actually usable

TODO.md's P2 "annotator" item names anticp2, amplify, macrel, pfam, hemopi, toxinpred, and
allergenicity (algpred2, added once permeability/toxicity vetting raised the same question for it).
All seven were checked against the real, installed, running tool — not assumed — with a final pass
on 2026-08-14 that built every environment and ran every tool on a real peptide.

- **hemopi2, macrel, AMPlify, AntiCP2, AlgPred2 — all five working, verified end to end** through
  `sample.peptide_predictors.enrich_with_predictors` on a real sequence (2026-08-14):
  `macrel_amp_probability=0.95`, `amplify_score=0.99999297`, `anticp_score=1.0`,
  `hemopi2_score=0.72`, `algpred2_score=0.434` for a magainin-like AMP. Real bugs found and fixed
  along the way:
  - **macrel: `_run` hardcoded a bare `"macrel"` instead of `self._command`.** `available()`'s
    per-environment probing resolved the tool correctly, but the actual invocation ignored that and
    depended on `macrel` being on the *caller's* `PATH` regardless — defeating the reason each
    predictor gets its own isolated environment. Fixed; regression-tested (a `_command` that is
    deliberately not `["macrel"]` must be what `subprocess.run` receives).
  - **AMPlify: its own script's shebang picks the wrong interpreter outside its environment.**
    bioconda's `AMPlify` console script execs `AMPlify.py`, whose shebang
    (`#!/usr/bin/env python3`) resolves through the *caller's* `PATH`, not its own environment; run
    under any other active environment it fails with `ModuleNotFoundError: No module named 'keras'`.
    `AMPlifyPredictor._run` now invokes `<env>/bin/python <env>/share/amplify/src/AMPlify.py`
    explicitly, sidestepping the fragile shebang. `available()`'s probing is unchanged.
  - **AMPlify env: bioconda's `amplify` needs python=3.6, not 3.8** — `conda install amplify` on
    python=3.8 fails outright (`LibMambaUnsatisfiableError: ... requires python 3.6.*`). Also: `pip
    install git+https://github.com/bcgsc/AMPlify.git` fails outright (no setup.py/pyproject.toml);
    the working install is bioconda's own `amplify` package.
  - **AntiCP2 env: needs python=3.6, not the "≤3.7" the metadata suggests.** `anticp2==1.1` (the
    real PyPI package; `pip install git+.../anticp2.git` fails the same way AMPlify's does — no
    setup.py) declares `Requires-Python >=3.3,<=3.7`, a PEP 440 specifier *without* a wildcard — a
    strict comparison, so even `python=3.7.12` fails it (`3.7.12 <= 3.7` is False). `python=3.6`
    satisfies the bound unambiguously.
  - `environments/sample-amplify.yml` and `sample-anticp2.yml` are corrected to the working install
    method and Python pin (`sample-macrel.yml` needed no change).
- **toxinpred3 — genuinely blocked on an upstream packaging defect, exhaustively verified.** Its
  shipped `toxinpred3.0_model.pkl` (a pickled `sklearn.tree._tree.Tree`) is incompatible with *every*
  scikit-learn version tried against the real environment (2026-08-14, each verified via the env's
  own `pip`/binary directly, not through a shell alias): 1.1.3, 1.2.2, 1.3.1 (its own env's original
  pin), 1.4.2, 1.5.2, 1.6.1. Most give `ValueError: node array from the pickle has an incompatible
  dtype`; 1.1.3 gets past that and dies on an unrelated `numpy.exceptions.AxisError` deeper in the
  tool's own code. The package's PyPI metadata doesn't even declare scikit-learn as a dependency
  (`Requires: argparse, joblib, numpy, pandas`) — whatever version its authors pickled with is not
  discoverable from the package itself. Not fixable from this project's side; needs an upstream fix
  or a from-scratch retrain, neither of which is this project's to do. No `ToxinPredPredictor`
  adapter is written, since there is nothing for it to call successfully.
- **pfam** — not actually a gap: Pfam-A assignment already exists (`sample/families.py`, curated
  `hmmsearch` at gathering threshold, exposed via `GET /families`) for peptides in the release, and
  `sample.peptide_annotations` (below) now also runs a request-time family-HMM scan
  (`hmmscan`/`run_hmm_search`) per peptide.
- **SAMPLE/literature/patent hits** — covered by `sample.peptide_annotations`' near-identity search
  (`sample.similarity.search`), stored per peptide alongside the predictors and family/Pfam data.

## Pre-computed per-peptide annotation (`sample.peptide_annotations`)

Built 2026-08-14: `sample-cli build-annotations` computes, for every release peptide, the four
working external predictors above (batched — each tool's model loads once for the whole corpus, not
once per peptide), the peptide's Pfam domains and SAMPLE family/cluster (read from the release
database), near-identical release peptides (`sample.similarity.search`), and family-HMM hits
(`hmmscan` against SAMPLE's own profile HMMs). Stored in `data/peptide_annotations.sqlite`, resumable
(`--limit` for slices), and read by the peptide page (absent — panel skipped — until a peptide has
been annotated). **Circular peptides are handled explicitly**: a peptide whose topology is a
head-to-tail cycle (from `sample.cyclization`, or CycPeptMPDB membership) carries `is_circular=True`
and a `circular_caveat` — every one of the sequence-based predictors, the near-identity search and
the HMM scan assume a linear sequence, so a cyclic peptide's results are computed (not skipped) but
explicitly flagged as approximate rather than presented as if the molecule were linear.

**Run in full on 2026-11-24: all 55,879 non-decoy peptides are annotated** (`data/peptide_annotations.sqlite`,
101 MB), so the panel renders for every peptide and the ADMET hemolysis gap resolves to real
HemoPI2/macrel numbers rather than a pointer. Coverage: predictors 100%, SAMPLE family 94.1%,
near-identity hits 61.9%, family-HMM hits 34.1%, Pfam domains 26.0%, circular 5.7%. 54,874 peptides
carry all nine predictor fields; the shortfall is AntiCP2 declining 1,005 peptides, HemoPI2 213 and
macrel 1, all of them 1-4 residues long and below those tools' feature-extraction minimum — a
refusal reproduced live against the same sequences, not a transient failure, and omitted rather than
zero-filled.

The run took 36 minutes rather than the ~4.2 hours the batch implies, because `--predictors-from`
took the predictor values from `data/sample_final_release/final_dataset.csv`, which already held all
four predictors for the whole corpus. That is sound because the predictors are pure functions of the
sequence and the reuse is guarded by sequence identity: all 55,879 ids matched, **0 rows were dropped
on a mismatch**, and a row whose sequence disagreed with the database would have been dropped and
counted rather than attached. It also makes the peptide page's numbers identical to the published
dataset's by construction instead of by coincidence.

## Environment tooling note: `conda run -n <env>` is unreliable in this session's shell

Discovered 2026-08-14 while verifying the predictor environments: this shell pre-sets `PATH` with
`envs/sample/bin` first for every command, and `conda run -n <env>` (even with the full `-p
<prefix>` form) does not override that for the *subprocess* it launches — `conda run -n
sample-anticp2 python --version` reported `envs/sample`'s python, not `sample-anticp2`'s, and (more
seriously) `conda run -n sample-toxinpred3 pip install scikit-learn==X` silently installed into the
`sample` environment instead. **One real casualty**: `sample`'s scikit-learn was overwritten from the
required `1.9.0` (the pin `environments/sample-web.yml`/`sample-ml.yml` share, needed to unpickle
trained models) down to `1.4.2`, discovered and restored the same session — verified a real trained
model unpickles again afterward. The production code was never at risk: `_predictor_command`
resolves each tool by its **direct absolute path** (`envs/<env>/bin/<executable>`), never through
`conda run`, specifically so this class of bug cannot reach it (see the per-tool env note above) —
this was purely an artifact of *interactive* verification commands in this shell. The reliable
pattern for any future manual environment work here: use each environment's binaries by their full
path (`/home/.../envs/<env>/bin/pip`, not `conda run -n <env> pip`), and re-verify with `pip show`
against that direct path rather than trusting `conda run`'s reported version.

Net: one of six named tools is live (hemopi2). The other three that have adapters — macrel, amplify,
anticp — now each have a dedicated env file (`sample-macrel`/`sample-amplify`/`sample-anticp2`) and
are wired to be probed in that env; what remains for each is a *build* to confirm the install and
entry point (the AMPlify/AntiCP2 repos install from GitHub, not PyPI). toxinpred3 has its env file
but no adapter yet. The annotator's job infrastructure (`sample.annotator` +
`sample.annotation_store`, the Celery `annotate_peptide` task, the one-week result cache, and the
`/api/v1/annotate` routes) **is built** and runs hemopi2 + a request-time Pfam scan +
SAMPLE/literature/patent hits; it is the *additional* external predictors above that are pending
their environment builds, not the infrastructure or the wiring.

## Prediction types with no tool identified at all

TODO.md's P2 backlog originally listed four prediction types with no implementation in any form:
membrane permeability/disruption, allergenicity, immunogenicity, and IEDB epitope hits (B-cell,
T-cell, MHC). Two are now resolved; two remain, and for a reason more specific than "no tool found."

**Membrane permeability — resolved by indexing, not modelling (2026-08-13/14).** `PharmPapp`
(`ifyoungnet/PharmPapp`) turned out to be a KNIME-only workflow system with no Python API, so a
model was trained from scratch on CycPeptMPDB (the public peptide-permeability dataset) instead —
and that model's holdout R²≈0.24 for this database's *linear* peptides confirmed CycPeptMPDB's
chemical space (cyclic, N-methylated macrocycles; passive diffusion) doesn't transfer to
membrane-disrupting linear AMPs. Rather than ship a misleading predictor, CycPeptMPDB is indexed as
a source instead (`sample.cycpeptmpdb`, structurally linked to release peptides by InChIKey where
they coincide); `sample.admet` keeps permeability as an honest, explicit gap for linear peptides.
See `DECISIONS.md`'s 2026-08-14 entries for the fuller account.

**Allergenicity — resolved (2026-08-14).** `algpred2` (Raghava lab, pip-installable) is built,
fixed (`scikit-learn==1.2.2` + `numpy<2` for its 2016-vintage pickle — the same failure shape as
toxinpred3, but fixable), verified against a real peptide, and wired into
`sample.peptide_predictors`/`sample.peptide_annotations`. See this file's annotator section above.

**Immunogenicity and IEDB epitope hits — candidates identified, but each has a real reason it
isn't a drop-in install (2026-08-14).** `mhcflurry` (openvax, pip-installable, peer-reviewed,
actively maintained) is a genuinely trustworthy pan-allele MHC-I binding predictor -- but every
prediction requires an explicit HLA allele (`mhcflurry predict --alleles HLA-A0201 --peptides
...`), and this database's peptides carry no target-allele/patient context. Predicting against an
invented allele panel would be exactly the unfounded-guess mistake `DECISIONS.md`'s rules exist to
prevent; wiring this needs a scoped, justified reference-allele-panel decision first (e.g. from a
population-coverage study), not another install attempt. IEDB's own allele-independent "Class I
Immunogenicity" tool (Calis et al. 2013) would sidestep that problem, but its download is gated
behind a license-acceptance page with no fetchable install docs -- unlike algpred2's plain PyPI
listing, its actual installability is unconfirmed; the next concrete step is downloading and
inspecting it, not guessing at its interface or reimplementing the published formula from the paper
without validating the reimplementation against the paper's own benchmark numbers first. An `iedb`
PyPI package exists at version 0.0.2 with minimal apparent adoption -- exactly the shape of package
that turned out, for `amplify`, to be an unrelated project sharing a name -- and was not adopted on
that basis alone.

## Technical debt: silent exception handlers -- resolved 2026-08-12

An AST scan (2026-08-11, re-run 2026-08-12) for a bare or `except Exception` handler wrapping a
loop body — the shape that hid a 60% activity-table loss with no error anywhere, see
`DECISIONS.md` — found 70 such sites codebase-wide. All are now either counted-and-logged or
reviewed and confirmed already safe by design; see `DECISIONS.md`'s 2026-08-12 entry for the full
per-site breakdown and the two genuinely fail-safe exceptions (`validation_extras.py`'s control and
hard-negative checks, which already turn a crash into a conservative counted failure rather than
losing it).

## Environment quirk, not a code bug

This host's `sample` conda environment intermittently fails to import `sqlite3` with an ICU/
libstdc++ symbol-version mismatch. See `DECISIONS.md` for the full description and the partial
workaround already in place (`sample/__init__.py` preloads `sqlite3` first). Running the full test
suite rather than an isolated single file avoids it reliably; this is not something to "fix" by
chasing an individual test failure as if it were a regression.

## A killed LibreOffice leaves a lock file that silently blocks every later recalculation

Recalculating `data/manual_curation/curation_review.xlsx` (the step the xlsx workflow requires,
because openpyxl writes formulas with no cached values) launches LibreOffice headless. If that run
is interrupted -- a timeout, a `kill` -- it leaves `.~lock.curation_review.xlsx#` beside the file
and, often, a `soffice.bin` still holding it. Every subsequent attempt then **waits on the lock
instead of failing**, so it reports only "LibreOffice timed out after Ns; formulas were NOT
recalculated" and a longer timeout never helps.

The tell is in the timing: a blocked run burns wall-clock with almost no CPU (`real 7m0s` against
`user 0m1.9s`). The fix is to clear the lock and any orphaned process, then re-run:

```bash
pkill -x soffice.bin; pkill -x oosplash          # -x, NOT pkill -f: a -f pattern naming
rm -f 'data/manual_curation/.~lock.curation_review.xlsx#'   # the file matches your own shell
```

Worth stating because it cost an hour here and produced a wrong diagnosis on the way: the stall was
first attributed to the workbook's data-validation style, and an inline-vs-range experiment appeared
to confirm it, because the two candidate files were scratch copies with no lock while the real file
had one. Measured on clean paths the two styles are indistinguishable (4.4s inline, 4.9s
range-based, 2,991 rows x 5 validated columns), and the inline form the code uses is fine. A
scratch-copy comparison is not a controlled one when the variable you are not measuring is a lock
on one of the paths.

## The shipped benchmark numbers and the manuscript's Table 2 disagree

Found while answering a reviewer on Section 3.10 (2026-09-15). `paper/README.md`'s Table 2 and
`data/sample_final_release/benchmark_report.json` describe the same sampled holdout — identical
sample sizes (999/999/999/1000) and the identical 187 positives — with different numbers **on both
sides of every comparison**:

| | Table 2 | benchmark_report.json |
|---|---|---|
| sAMPle ROC-AUC / recall / precision | 0.747 / 0.425 / 0.730 | 0.939 / 0.668 / 0.850 |
| sAMPle threshold | 0.505 | 0.5 |
| Macrel ROC-AUC / recall / precision | 0.869 / 0.780 / 0.769 | 0.836 / 0.717 / 0.444 |

The table has sAMPle behind all four comparators; the artifact has it ahead on ROC-AUC and far
ahead on precision. `paper/dataset_stats.json` (written by `generate_dataset_report.py`) agrees
with the artifact, and **Figure 16 was rendered from it** — so the shipped PDF prints a figure and
a table that contradict each other. Grepping every JSON and log in the repo for `0.747` or `0.425`
finds nothing: the table's numbers have no run on disk. Neither set has been adopted; Section 3.10
now carries a correction notice instead, and the benchmark needs a re-run (`sample-cli benchmark`)
to settle it. Note the artifact's own 0.939 is not automatically the right answer either — it was
computed against a negative class made entirely of decoys the model trained on the generator of,
which is exactly what the negative-class change below addresses.

## The decoy set must be regenerated before any model metric is reported again

`sample.decoys` now generates a third, adversarial source (`uniprot_amp_like`: non-AMP by UniProt
annotation, AMP-like in net charge and hydrophobic fraction) and `method="combined"` is a three-way
mix rather than 50/50 shuffled/UniProt — see `DECOY_METHOD_MIX`. Separately,
`sample.ml_models.USE_MEASURED_INACTIVE_NEGATIVES` adds real, measured-inactive peptides to every
classification task's negative class.

Both change what every classifier is trained against, so **no metric computed under the old
decoy-only negative class is comparable to one computed after**. The release's own
`decoys.csv` predates the change. Regenerating means step 28 (`sample-cli decoys`) → step 29
(assembly) → step 45 (`train_all`) → step 47 (`benchmark`), which is a multi-hour run and by
project rule is not started without an explicit go-ahead.

Also worth knowing, with an earlier version of this entry corrected: `negative_type` is absent
from the shipped `final_dataset.csv` (531 columns, only `is_decoy` and `source`), but NOT because
assembly drops it. `pd.concat` unions columns, so the column survives assembly on its own --
verified directly. The shipped dataset simply predates the column. `sample-cli assembly` now sets
it explicitly on both sides (`real_positive` for positives, `unspecified` for a decoy file that
predates it) so the column has one defined meaning per row instead of an incidental NaN that reads
the same as an unrecorded decoy source. A fresh assembly carries it; no re-plumbing is needed.

## The last `train_all` rollout mostly failed on an I/O error, not on modelling

`data/sample_final_release/flaml_rollout_progress.json` records the state of the last pass: the
general antibacterial model and one target-specific model written and gate-passed, the anticancer
model gate-failed degenerately (threshold 0.755, precision 0.0, recall 0.0, balanced accuracy
exactly 0.5 — it predicts a single class), one XGBoost quantile crash, and **16 further tasks dead
with `[Errno 5] Input/output error`**. That is a disk/environment failure, not a result. Section
3.8 cannot be filled from this rollout; re-running it is a prerequisite, and the I/O error should
be understood before the re-run rather than after.

Related, and visible in the same artifacts: several shipped `esm2` models pair a high ROC-AUC with
a recall at or near zero (`botrytis_cinerea/any` 0.971 / 0.000; `pseudomonas_aeruginosa/mbc`
0.997 / 0.036). A model that ranks well and calls nothing positive is mis-thresholded, not
mis-trained — the F1 calibration is selecting a precision-favouring cutoff under heavy imbalance.
A recall-weighted criterion (F-beta, beta > 1) or serving the threshold as a user-selectable
operating point is the fix; changing the negatives is not.

## `data/zenodo_dump/scripts/` ships 28 scripts numbered for a pipeline that no longer exists

Untracked, dated 2026-07-01, referenced by nothing in the tree, and **not generated by step 52** --
which nonetheless tars the whole of `data/zenodo_dump/` into the published artifact. So a Zenodo
release carries 28 one-line `sample-cli` wrappers whose filenames describe a 28-step pipeline while
the Makefile has 52 contiguous steps.

Measured against the current CLI and Makefile: 26 of the 28 still invoke a live subcommand, 2 call
nothing that exists any more, and **18 carry a step number that now means a different step** --
`step_14_merge_database_and_extracted_peptides.sh` is step 19 today, `step_22_load_sql_database.sh`
is 44, `step_27_run_webserver.sh` is 51. Four more (`enrich`, `eda`, `validate-ml`,
`validate-manual-curation`) wrap commands that are no longer numbered steps at all.

Renumbering them is not the fix: the numbers and the contents disagree about *what the steps are*,
not just where they fall, so renaming to 1--52 would produce filenames that actively misdescribe
their own bodies. The two real options are to **delete the directory** (recommended -- it is
orphaned, unreferenced and misleading) or to have step 52 **generate** the wrappers from the
Makefile's own target list so they cannot drift again. Left in place pending that decision, and
flagged here because the consistency sweep of 2026-10-07 could not resolve it without one.

## The peptide--antibiotic synergy (FICI) models are retired, not served

Decided 2026-11-23, by the project owner, on the grounds that they are "not reliable enough for
the current usage." Removed from the manuscript and from the web server the same day.

They were never actually reachable: `model_registry.list_models` has always skipped the
`synergy/` namespace, so no dropdown, no `/api/v1/models` entry, no route. But the manuscript
claimed in three separate places that they *were* served ("made directly accessible through the
web portal and REST API"; "across all 29 validated bioactivity classes and 18 synergy targets";
"18 ESKAPE pathogen synergy models"), and `sample/worker.py`'s docstring claimed it executed
"bioactivity and synergy prediction jobs" while calling a sequence-only `inference.predict`.

**What the numbers actually were**, from `data/sample_final_release/synergy_report.json`
(11 Sep 2026 -- FICI labels are unaffected by the three-band potency change, so these stand):

| scope | n | ROC-AUC | PR-AUC | 95% CI (ROC) | gate |
|---|---|---|---|---|---|
| *A. baumannii* | 396 | 0.859 | 0.885 | [0.775, 0.930] | **pass** |
| *K. pneumoniae* | 373 | 0.834 | 0.927 | [0.728, 0.915] | **pass** |
| *S. epidermidis* | 84 | 0.841 | 0.942 | [0.572, 1.000] | fail -- CI lower 0.572 < 0.65 |
| *M. luteus* | 87 | 0.795 | 0.444 | [0.508, 1.000] | fail -- y-randomized 0.636 = leakage |
| *P. aeruginosa* | 1,151 | 0.642 | 0.665 | [0.572, 0.729] | fail |
| all pathogens | 4,853 | 0.637 | 0.652 | [0.604, 0.667] | fail |
| *E. coli* | 1,029 | 0.618 | 0.648 | [0.544, 0.695] | fail |
| *C. albicans* | 129 | 0.549 | 0.583 | [0.280, 0.769] | fail -- n.s. vs y-randomized |
| *S. aureus* | 814 | **0.482** | 0.446 | [0.389, 0.585] | fail -- below chance |

**2 of 18 passed.** All nine regression scopes failed as well: every held-out R2 was <= 0.073 and
most were negative (*S. epidermidis* reached **-9.99**, pooled -0.034), against a gate of 0.3.
The manuscript's old Table 2 showed six of the nine classification scopes and omitted the three
worst, including the below-chance one; it bolded *S. epidermidis* as a headline result on n=84.

The three best-sounding figures were also not what they appeared. *S. epidermidis* and
*M. luteus* have CIs reaching 1.000 on n<90. And the pooled `p = 0.00115` quoted beside ROC-AUC
and PR-AUC was the *balanced-accuracy* t-test; the actual ROC-AUC and PR-AUC p-values were
4.0e-04 and 5.7e-05.

**Where things now live.** The 18 artifacts moved from
`data/sample_final_release/models/synergy/` to
`data/sample_final_release/models_retired/synergy/`, joining the two retired ESM-2 pickles
already there. `synergy._model_root` and `train-synergy --model-dir` both default to the retired
tree, so re-training writes outside the served root. The guard in `model_registry` stays as a
backstop against a `--model-dir` pointed back at the served root.

**Kept deliberately:** the 4,853 audited checkerboard measurements (2,394 synergistic, 2,459
not), from 5,507 raw assays less 654 discarded for >25% FICI discrepancy. The data is
experimental and its reliability was never in question -- only the models trained on it. The
manuscript keeps this as a dataset contribution under "Peptide--Antibiotic Combination Assay
Curation" and states plainly that no predictor ships.

**Still open:** `data/zenodo_dump/` still carries its own copy of the 18 synergy pickles under
`sample_final_release/models/synergy/`. Pulling them needs a tarball rebuild (7.37 GB), so it is
left pending a decision on whether an archived-but-unadvertised copy is acceptable.

## 13 peptides carry no RDKit-derived descriptors

Surfaced 2026-11-23 by the anomaly detector's new fill accounting, which was added precisely so
this class of gap stops being invisible. The step-45 report now carries
`feature_values_filled`, and the 2026-11-23 16:33 run recorded **39** values filled with 0.0 in
the physicochemical block. They localize exactly:

| descriptor | values filled |
|---|---|
| `logp` | 13 |
| `hydrogen_bond_donors` | 13 |
| `hydrogen_bond_acceptors` | 13 |

Three columns, 13 each: the same **13 peptides**, all three of whose values come from RDKit via a
sequence-derived SMILES graph. So this is one failure mode (RDKit could not build a molecule for
those 13 sequences), not three.

Why it matters more than 39-in-1.4-million suggests: those rows are fed to the Isolation Forest
as `logp = 0.0`, which is not a neutral value. The release median is about $-12$, so a filled
zero sits roughly 12 units above the centre of the distribution and reads as an outlier on that
axis. A peptide whose RDKit parse failed is therefore nudged toward being flagged *for the parse
failure*, not for its chemistry. At 13 rows this cannot move the headline 6,224, but it is the
wrong mechanism and it would scale with any future parse regression.

Not fixed here because the fix is a decision, not code: either impute from the sequence (the
other ten descriptors are computed without RDKit and are present for all 110,945 records), or
exclude the 13 rows from the fit and record them as unscored rather than as inliers or
anomalies. Both are defensible; neither should be chosen silently. Identify them with
`is_anomaly` plus a null check on `logp` in `final_dataset.csv`.
