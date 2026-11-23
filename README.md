# sAMPle: A Web Server and Data Engineering Pipeline for Antimicrobial Peptides

> ### This is a source-only repository
>
> It carries the code, the `Makefile`, the environment specifications, the tests and the
> documentation — 6 MB in total. It deliberately does **not** carry:
>
> | not here | why | how to get it |
> |---|---|---|
> | `data/` (~127 GB) | downloaded corpora, extraction JSON, trained models, release artifacts | `make all`, or the Zenodo archive; layout in [`data/README.md`](data/README.md) |
> | `paper/figures/`, the rendered PDF | regenerated from the release data | `python paper/generate_dataset_report.py`, then `paper/build_pdf.py` |
> | `docs/_build/` | Sphinx output | `make docs` |
> | EDA figures under `sample/static/img/` | written by the exploratory-analysis step; the analytics page shows gaps until then | `make all START_STEP=50 END_STEP=50` |
> | the Expression Wizard | split out on 2026-11-23 into its own tool, with its own web and CLI interfaces and its own repository | see *Provenance* below |
>
> Nothing here is generated. If a command needs an input that is missing, it says so rather than
> inventing one — that is the same principle the extraction pipeline is built on.
>
> **Start here:** [Installation & Setup](#installation--setup) → `make envs` → `make all`.

## Overview

**sAMPle** is an end-to-end data-engineering pipeline and interactive web server for antimicrobial
(and anticancer) peptide data. It systematically queries, downloads, scrapes, extracts, validates,
enriches, models, and serves peptide records — bridging the gap between the primary literature /
patents and structured, provenance-tracked database entries.

The pipeline parses publications and patent PDFs with a classical-NLP-first, LLM-verified extraction
engine; computes physicochemical descriptors, protein-language-model embeddings, and 3D structures;
groups peptides into families; trains and rigorously validates per-task machine-learning models; and
serves everything through a **FastAPI** dashboard with background prediction jobs. It runs end to
end from a single `make all`, with each of its 54 steps executed in the conda environment that
carries exactly its dependencies.

---

## Key Features

1. **Multi-source acquisition** — PubMed, Scopus, and Google Patents metadata and full-text PDFs.
2. **Reference-database scrapers** — ADP, DBAASP, DRAMP, DBAMP, and CAMP, including the ~194k
   measured activities DBAASP alone contributes. DRAMP is tracked to **5.0** (32,296 entries;
   *Nucleic Acids Research*, [doi:10.1093/nar/gkag989](https://doi.org/10.1093/nar/gkag989)),
   whose curated SMILES annotations are now read — see "Which DRAMP release is on disk" below,
   because DRAMP's download URL does not name the release it serves. Plus an
   **anticancer-peptide import**
   (`import-anticancer`) covering AntiCP2, DCTPep and CancerPPD — 7,533 unique sequences, 5,730
   anticancer positives, carrying their measured potencies (IC50/EC50 with operator, unit, cell
   line and assay), 3,433 PMIDs / 2,115 DOIs, and cross-references to DRAMP/DBAASP/UniProt/PubChem/
   PDB. Positives flow through the merge into both the feature set and the `anticancer` labels the
   models train on, and into the extraction pipeline's known-sequence search. CancerPPD's
   measurements live only on per-entry pages, so a rate-limited, resumable scraper fetches them.
3. **LLM structured extraction** — Aho-Corasick/regex candidate discovery over primary text, with a
   local LLM (Ollama) or Gemini verifying biological activities and recovering sequences. A strict
   output contract (deterministic rules → LLM → the *same* rules again) decides what is published;
   the model can recover data but never writes to the database on its own authority.
4. **Validation & provenance** — error-injection schema tests, extraction metrics against database
   baselines, and a long-form `activities.csv` (one row per measurement) tagging every value's
   `provenance`, `source_database`, `sequence_novelty`, and `activity_novelty`. Novelty is a
   graded judgment, not set membership — see [What counts as a new
   sequence](#what-counts-as-a-new-sequence).
5. **Molecular enrichment** — RDKit/BioPython descriptors, ESM-2 embeddings (indexed in Chroma for
   embedding search), and 2D/3D structures (ColabFold, resumable, pLDDT in the B-factor).
6. **Families & search** — Pfam anchoring (`hmmsearch` at gathering threshold) then PepMatch
   near-identity clustering of the remainder; profile HMMs per cluster; and four search modes on the
   dashboard: BLAST, HMMER, PepMatch near-identity, and ESM-2 embedding k-NN.
7. **Rigorous ML** — a per-task, per-feature-block model matrix (XGBoost with a RandomForest
   fallback) with clustering-based train/test splits to prevent analogue leakage, a normality test
   gating Welch's t-test vs Kruskal-Wallis, y-randomization, bootstrap CIs, and separate
   classification/regression quality gates. Every model exposes a **dataset** page with its full
   validation report and train/test peptide split, and a k-NN **applicability-domain** warning at
   prediction time.
8. **Structure & design tools** — a synthetic-gene **Expression
   Wizard** — four hosts (*E. coli*, *S. cerevisiae*, *P. pastoris*, human), an optional secretion
   signal, a His tag, a carrier (Trx/MBP/GST) and a choice of **SUMO or TEV** cleavable linker;
   codon optimization with CAI banding *and* MPEPE's learned yield model; restriction-site
   avoidance; whole-construct mRNA folding read over the 5' region (with optional vector 5'-UTR
   context); N-end-rule half-life of the released peptide; SUMOylation consensus-motif flagging;
   ColabFold construct folding; a 2D construct map (SVG/PNG/PDF) and GenBank/FASTA export for
   SnapGene and Benchling — and DSSP secondary structure, Ramachandran, and disorder diagrams on
   each peptide page.
9. **Analysis & prediction tools** — an **ADMET-lite** profile (`admet`: sequence-derived
   physicochemistry + rule flags; hemolysis links to the annotator's live prediction, permeability
   stays an honest gap answered by indexing **CycPeptMPDB**'s ~8,500 cyclic peptides as a source
   rather than by extrapolating a model to the wrong chemistry), a **pre-computed per-peptide
   annotation** (`build-annotations`: four verified external predictors — macrel, AntiCP2, HemoPI2,
   AlgPred2 (AMPlify was retired 2026-08-26 after a sustained upstream crash rate in production) —
   plus Pfam/family, near-identity and family-HMM hits, with circular peptides
   flagged rather than silently scored as linear), a **reference-tool benchmark** (`benchmark`),
   **RefSeq genomic context** (`genomic-context` / `genomic-context-sweep`: per-peptide tBLASTn
   presence/absence with a Krona-style taxonomy sunburst, resumable one genome at a time), and an
   **incremental update** path (`incremental-update` / `update-families`) that places newly
   ingested peptides into the existing family map without renumbering.
10. **Interactive dashboard** — background prediction jobs (Redis + Celery), a database explorer with
    per-peptide detail, analytics, an annotator, and external-database cross-links. The analyses
    that cost minutes each (ColabFold folding, MPEPE yield) are cached in Redis for seven days
    (`sample.result_cache`) — a cache, never a record of truth, so an unreachable server just means
    the work is redone rather than anything failing.

---

## Installation & Setup

### 1. Prerequisites

Most binaries are supplied by the conda environments below. You only need a base system with:

* **conda / mamba** — to create the environments.
* **Poppler** (`sudo apt install poppler-utils`) — PDF rendering during extraction.

BLAST+, HMMER, MAFFT, DSSP (`mkdssp`), and ViennaRNA (`RNAfold`) are installed *into* the
environments that use them, not system-wide. ColabFold (`colabfold_batch`, GPU) is optional and only
needed to (re)generate 3D structures.

### 2. Conda Environments

Each stage runs in the environment carrying its dependencies and no more. All are defined in
[`environments/`](environments/), and the Makefile maps every step to the right one automatically.

| Environment | Steps | Carries |
|---|---|---|
| `sample-acquisition` | 1–6 | scraping, HTTP, PDF text, anticancer-database import |
| `sample-extraction` | 7–10, 12–19, 33–42 | NLP, LLM, validation, merges, families, harvests |
| `sample-chem` | 20–32, 44 | RDKit, descriptors, predictors, embeddings, structures, decoys, assembly, SQLite load |
| `sample-ml` | 11, 43, 45–50 | scikit-learn, xgboost, torch/transformers, plots |
| `sample-web` | 51 | FastAPI, uvicorn, Celery, BLAST, DSSP |
| `sample-predictors` | 22–25 (sub-call) | third-party activity predictors |
| `sample-fold` | 27 (sub-call) | ColabFold structure prediction |
| `sample-macrel` / `sample-anticp2` / `sample-algpred2` / `sample-toxinpred3` | annotator (sub-call) | one external predictor each, isolated so their conflicting scikit-learn/TensorFlow pins never meet (`sample-amplify` also exists, for the retired AMPlify class kept as a manual reference) |
| `sample` | 50 (PDF build), all | the union of the above; also builds the paper PDF in step 52, and is required by the docs build and the single-env mode |

```bash
make envs        # creates the pipeline environments and verifies the pins
make env-full    # additionally creates `sample`, needed for `make docs`
```

The per-tool sub-call environments are built **on demand**, not by `make envs`, since each one
exists only to hold a single tool whose pins fight the others:

```bash
make env-fold                                          # ColabFold structure prediction
make env-macrel env-anticp2 env-algpred2               # the four working annotator predictors
```

The groupings are the **transitive import closures** of the modules each step actually enters,
computed from the source rather than guessed. `environment.yml` in the repository root is a symlink
to `environments/sample.yml`.

Prefer one environment instead? Everything still works:

```bash
make env-full
make all SINGLE_ENV=sample
```

**The split is a correctness requirement, not tidiness.** Pickles cross environment boundaries —
step 13 trains `rf_filter_model.pkl` in `sample-ml` and step 8 loads it in `sample-extraction`;
steps 47–48 train the models and the webserver loads them in `sample-web` — and a pickled estimator carries
no version negotiation. `scikit-learn`, `xgboost`, and `numpy` are therefore pinned identically
across those environments, and `make check-envs` asserts the installed versions actually agree.
Separately, `hemopi2` and `toxinpred3` pin scikit-learn 1.3.1, which silently downgraded the main
environment; steps 24–27 (the external predictors) shell out to `sample-predictors` instead. See
[`environments/README.md`](environments/README.md).

### 3. Install the sAMPle package

Install the project in editable mode to register the `sample-cli` entry point:

```bash
pip install -e .
```

### 4. Create directories

```bash
make dirs
```

### 5. Running the pipeline with the Makefile

`make all` runs the full sequence (steps 1–54). Each step verifies that its output artifact exists
before the run is allowed to continue.

```bash
make all                          # resume: run what is not finished yet
make all FROM_SCRATCH=1           # start a new run; re-run every step
make all START_STEP=17            # skip earlier stages, start AT step 19
make all END_STEP=26              # stop after step 28 (inclusive)
make all START_STEP=17 END_STEP=26   # an inclusive range, steps 19-28
make step_37_build_families       # a single step (honours START_STEP/END_STEP guards)
```

#### Resuming an interrupted run

The pipeline is a multi-day job, so **resuming is the default**: a `make` that was interrupted
continues where it stopped instead of rebuilding what it already finished. Two mechanisms do this,
and both are needed:

| level | where | what it skips |
|---|---|---|
| step | `run_step` in the Makefile, via `data/pipeline_runs.sqlite` | a step already recorded `SUCCESS` under the current run |
| item | the step's own command (`--resume-mode`, `--resume`, `--no-resume`) | documents, clusters or namespaces that step already processed |

Step-level resume is what makes an interrupted run continue at step 32 rather than step 1.
Item-level resume is what continues a single long step part-way through — which matters most for
extraction. Measured on a live step 9 re-walk: an already-refined document costs ~3s to re-process
(1,195 docs/h, because the payload cache returns its model calls for free) against ~38s for a new
one (94 docs/h), so re-walking 11,578 finished documents to reach the unfinished remainder is
**~9.7 hours** of work that produces nothing — on top of the 4.4 days the remaining 9,963 documents
genuinely need.

A step that was still `RUNNING` when the run died is recorded `INTERRUPTED` and **re-run**: it has
no trustworthy artifact. Likewise a `FAILURE`. Only `SUCCESS` earns a skip.

```bash
make pipeline-status              # the progress table for the last run
make pipeline-status RUN_ID=20261021T160222Z-83c323   # or a specific run
make pipeline-status JSON=1       # the same, machine-readable
make pipeline-runs                # recent runs, newest first
make pipeline-finish              # close the open run without resuming it
make pipeline-selftest            # verify the resume machinery; runs no pipeline work
```

`make pipeline-status` prints one row per step of the run:

```
run 20261021T160222Z-83c323

  step  status        elapsed  artifact  name
  ----  ------------ -------- ---------  ----------------------------------
     1  SUCCESS           12s         1  Download PubMed Papers Metadata
     8  SUCCESS        4h18m    22,050  Screening extraction (NLP + flashte
     9  INTERRUPTED     31h02m   22,050  Refined extraction (adds a local L

  2 of 3 recorded step(s) succeeded
```

`FROM_SCRATCH=1` starts a **new run id** in which nothing is complete, and turns item-level resume
off as well. It does **not** delete anything — it re-runs steps that overwrite their own outputs,
and is not a substitute for `make clean`.

`make all` closes the run on success, so the next `make all` starts a fresh one rather than
reporting everything as already done.

**Progress is stored per run.** `data/pipeline_runs.sqlite` holds a `runs` table (one row per run:
id, status, start/end, `from_scratch`, `resumed_count`, git commit, host) and a `step_progress`
table (one row per step per run: status, duration, exit code, attempts, artifact path and entry
count, plus item counts where the command reports them). It is machine-local bookkeeping and is
gitignored; `SAMPLE_RUN_REGISTRY_DB` relocates it.

> Step 9 resumes with `--resume-mode refined`, **not** `--resume`. Step 8 writes an output JSON for
> every document, so resuming step 9 on file existence would skip the entire corpus and add no
> model records at all. `refined` reads `metadata.langextract_backend` instead, which is `enabled`
> only once the model pass has run for that document.
>
> A document that gate 1 marked `skipped` also counts as done — gate 1 is a pure function of the
> text, so the refined pass cannot do better on a captcha page or a blank scan. And a document
> whose `.txt` is **newer** than its output is re-queued whatever the output says, so re-acquiring
> a document after a failed download is never silently wasted.

### 6. Environment variables

Create a `.env` in the repository root:

```env
# NCBI/PubMed
EMAIL="your.email@example.com"

# Scopus
X-ELS-APIKEY="your_scopus_api_key"
X-ELS-INST="your_scopus_institutional_token"

# LLM APIs
GEMINI_KEY="your_google_gemini_api_key"

# PDF download routing
ROUTER="https://sci-hub.box"
```

---

## Command-Line Interface (`sample-cli`)

Everything is driven through one entry point:

```bash
sample-cli <command> [options]
sample-cli <command> --help        # authoritative, per-command flag reference
```

The commands, grouped by pipeline phase (the source of truth for flags is always `--help`):

| Phase | Commands |
|---|---|
| **Acquisition (1–6)** | `pubmed`, `scopus`, `download-pdfs`, `google-patents`, `databases`, `import-anticancer` |
| **Extraction & merge (7–19)** | `merge`, `extract`, `extract-from-pdf`, `revalidate-resolutions`, `train-rf-filter`, `external-enrich`, `validate`, `validate-error-injection`, `select-curation-set`, `revise`, `extract-repair`, `merge-extractions` |
| **Serialize, enrich, assemble (20–32)** | `serialize`, `enrich-base`, `enrich-macrel` / `enrich-anticp2` / `enrich-hemopi2` / `enrich-algpred2` (AMPlify retired 2026-08-26), `enrich-frequency-embeddings`, `fold` + `enrich-structures-fallback` (or the monolithic `enrich`, `enrich-esm2`, `enrich-descriptors`), `decoys`, `assembly`, `update-dataset`, `validate-smiles`, `build-embedding-index` |
| **Harvest & annotate (33–42)** | `build-names`, `build-references`, `build-families`, `build-family-hmms`, `link-bioactivity`, `build-modifications`, `build-source-organisms`, `build-cross-references`, `build-gene-ontology`, `build-go-obo` |
| **Load & model (43–49)** | `load-sql`, `anomaly-detection`, `train-antibacterial-ml-model`, `train-organism-specific-ml-models` (both FLAML-searched XGBoost, 600s budget, holdout + cross-validation + y-randomization), `benchmark-all`, `eda`, `compare-chemical-spaces` |
| **Structure & design** | `fold`, `express`, `admet` (sequence-only ADMET-lite profile) |
| **Sources & annotation (outside `make all`)** | `import-cycpeptmpdb` (index ~8,500 cyclic peptides, with structural linking to release entries), `build-annotations` (pre-compute the five external predictors + Pfam/family/near-identity/HMM hits per peptide) |
| **Analysis & maintenance (outside `make all`)** | `update-families` (place new peptides into the existing family map), `incremental-update` (diff a fresh acquisition run against the live release, then place what is new), `train-ml` (the full per-task/per-block matrix, fixed-hyperparameter XGBoost by default or FLAML-searched with the default omitted -- see `--fixed-hyperparameters`), `train-synergy` (peptide-antibiotic synergy, out of scope for the numbered pipeline), `validate-ml`, `benchmark` (a model vs a reference tool on one holdout; `benchmark-all` runs every model with a registered reference tool and is step 49), `genomic-context` / `genomic-context-sweep` (per-peptide tBLASTn presence/absence; the sweep is resumable, one genome at a time), `build-topology` (cyclic/disulfide topology), `build-pdb-homology` (RCSB sequence search + CA-RMSD superposition of each ColabFold prediction onto the closest real solved structure, when one exists; resumable, one HTTP round-trip per peptide), `enrich-macrel` / `enrich-anticp2` / `enrich-hemopi2` / `enrich-algpred2` (`make tool-enrich-<name>`: re-run ONE external predictor over the already-enriched dataset, filling in only the peptides missing its result -- not descriptors, not any other predictor, not a full steps-21-27 enrichment re-run; these are also steps 24-27 in the numbered pipeline, `ALL=1` forces a full recompute; AMPlify had the same command until it was retired 2026-08-26), `extract-tables` |
| **Serve & admin (51)** | `webserver`, `admin {create,passwd,list,remove,enable,disable}` |

> `validate-manual-curation` remains as a deprecated alias for `revise`. The command it used to run
> stamped records as expert-curated without anyone having seen them, and its output was read by no
> later step. Nothing in this pipeline claims human curation; `revise` records automated-revision
> provenance instead.

### The strict output contract (step 19, `revise`)

`revise` is the boundary that decides what the database contains. Order matters — deterministic
rules, then the LLM, then the **same** deterministic rules again: a model proposal is only a
candidate, published only if the rules that rejected the original independently accept the
replacement. Enforcement is field-level, not record-level — a peptide with a good sequence, a good
MIC, and one malformed organism loses the organism, not the record. Nothing after step 19 reads the
pre-revision directories, so the contract holds for the whole database. See
[`sample/strict_schema.py`](sample/strict_schema.py) and [`sample/revision.py`](sample/revision.py).

---

## Pipeline Execution Sequence

`make all` runs these in order; each reads the previous stage's output. Step 19 is the contract
boundary; steps 15–16 deliberately run before it (13 audits the raw extraction, 14 measures whether
the contract catches the faults it claims to).

Steps 23–29 used to be one monolithic step (`enrich`, "descriptors, embeddings, 3D structures").
Split so a single slow or failing piece -- almost always one of the external predictors -- costs
only its own step's time to redo, not the whole enrichment's. Each reads and writes
`data/sample_enriched_release/enriched.csv` in place, touching only its own columns. AMPlify had
its own step here (originally 20) until it was retired 2026-08-26 after crashing on a sustained
7-12%+ of batches across multiple real production runs with a genuine upstream bug -- see the
comment on `sample.peptide_predictors.EXTERNAL_PREDICTORS`.

| # | Command | Produces | Env |
|---|---|---|---|
| 1 | `pubmed` | Download PubMed Papers Metadata | acquisition |
| 2 | `scopus` | Download Scopus Papers Metadata | acquisition |
| 3 | `download-pdfs` | Download & Extract Papers Data | acquisition |
| 4 | `google-patents` | Download Google Patents Data | acquisition |
| 5 | `databases` | Download Antimicrobial Peptide Metadata | acquisition |
| 6 | `import-anticancer` | Import Anticancer Peptide Databases | acquisition |
| 7 | `merge` | Merge Antimicrobial Peptide Data | extraction |
| 8 | `extract` | Screening extraction (NLP + flashtext + tables; no gemma4) | extraction |
| 9 | `extract` | Refined extraction (adds a local LLM pass; interruptible) | extraction |
| 10 | `revalidate-resolutions` | Revalidate Applied Sequence Resolutions | extraction |
| 11 | `polish-extractions` | Polish Extraction Records (re-apply current activity rules) | extraction |
| 12 | `review-fields --run` | Final polish: per-field model review of what rules cannot settle | extraction |
| 13 | `train-rf-filter` | Train RF Filter to Discriminate Peptides vs English Words | ml |
| 14 | `external-enrich` | Enrich Extracted Peptides with External DBs | extraction |
| 15 | `validate` | Validate Peptide Data Extraction with LLM | extraction |
| 16 | `validate-error-injection` | Validate Extractions using Error Injection | extraction |
| 17 | `select-curation-set` | Select Manual-Curation Paper Set | extraction |
| 18 | `evaluate_with_gemini.py` | Review Curation Set with Gemini | extraction |
| 19 | `revise` | Revise Extractions against the Strict Output Contract | extraction |
| 20 | `extract-repair` | Extract Repair Extractions | extraction |
| 21 | `merge-extractions` | Merge Database and Extracted Peptides | extraction |
| 22 | `serialize` | Serialize Raw Datasets | chem |
| 23 | `enrich-base` | Compute Descriptors and Stability Features | chem |
| 24 | `enrich-macrel` | Predict AMP/Hemolytic Activity with Macrel | chem |
| 25 | `enrich-anticp2` | Predict Anticancer Activity with AntiCP2 | chem |
| 26 | `enrich-hemopi2` | Predict Hemolytic Activity with HemoPI2 | chem |
| 27 | `enrich-algpred2` | Predict Allergenicity with AlgPred2 | chem |
| 28 | `enrich-frequency-embeddings` | Compute Amino-Acid Frequency Embeddings | chem |
| 29 | `fold` | Predict/Embed 3D Structures | chem |
| 30 | `decoys` | Generate Decoy Datasets | chem |
| 31 | `assembly` | Assemble Final Master Dataset | chem |
| 32 | `update-dataset` | Update Final Dataset | chem |
| 33 | `validate-smiles` | Validate Published SMILES Column | chem |
| 34 | `build-embedding-index` | Build ESM-2 Embedding Index for Embedding Search | chem |
| 35 | `build-names` | Harvest Peptide Names and Synonyms | extraction |
| 36 | `build-references` | Build Bibliography from Sources and CrossRef | extraction |
| 37 | `build-families` | Assign Peptide Families (Pfam anchors + PepMatch clusters) | extraction |
| 38 | `build-family-hmms` | Build Profile HMMs for SAMPLE-Derived Family Clusters | extraction |
| 39 | `link-bioactivity` | Link Peptides to ChEMBL/PubChem BioAssay/BindingDB | extraction |
| 40 | `build-modifications` | Harvest Chemical Modifications | extraction |
| 41 | `build-source-organisms` | Harvest Peptide Source Organisms | extraction |
| 42 | `build-cross-references` | Harvest Peptide Cross-References (UniProt/PDB) | extraction |
| 43 | `build-gene-ontology` | Harvest Gene Ontology Terms | extraction |
| 44 | `build-go-obo` | Slice go-basic.obo to SAMPLE's Harvested GO Terms | extraction |
| 45 | `anomaly-detection` | Anomaly Detection | ml |
| 46 | `load-sql` | Load SQLite Database | chem |
| 47 | `train-antibacterial-ml-model` | Train Antibacterial ML Model | ml |
| 48 | `train-organism-specific-ml-models` | Train Organism-Specific Antimicrobial ML Models | ml |
| 49 | `benchmark-all` | Benchmark Models Against External Antimicrobial Peptide Prediction Tools | ml |
| 50 | `eda` | Exploratory Data Analysis | ml |
| 51 | `compare-chemical-spaces` | Compare Chemical Spaces | ml |
| 52 | `generate_dataset_report.py` | Generate Paper Figures and Tables | ml |
| 53 | `webserver` | Run Webserver | web |
| 54 | *(generate zenodo dump)* | Generate Zenodo Dump | - |

The pipeline is scoped to antimicrobial (antibacterial) peptide prediction, by instruction -- the
one case study while the FLAML-searched training path is proved out. `train-ml` (the full
per-task/per-block matrix) and `train-synergy` (peptide-antibiotic synergy) still exist as
manual/diagnostic commands, just not wired into the numbered pipeline above.

Extraction JSONs are read only from recognized source subdirectories (`pubmed`, `scopus`,
`google_patents`) via `config.iter_extraction_files`, never a bare `*/*.json` glob — which is how a
merge could otherwise be silently built from superseded files in a `pubmed.bak` sibling while the
live directories were empty.

### Extraction runs as three steps (8, 9, 10)

Extraction is split because the two passes have very different costs and failure modes, and the
screening pass is a usable result on its own.

| step | what runs | cost |
|---|---|---|
| 8 `screening` | NLP, flashtext, the catalogued-sequence automaton, docling table grids. `LANGEXTRACT_BACKEND=0`, so no gemma4. | ~0.27 docs/s with 4 workers |
| 9 `refined` | the same documents again, with the LangExtract pass enabled (`gemma3:4b` by default) | minutes per document |
| 10 `revalidate` | re-checks every sequence that synonym resolution applied in 8 and 9, withdrawing those that do not qualify | seconds |
| 11 `polish` | re-applies the current activity-field rules to records 9 already wrote: repairs units, moves prose out of `activity_type`, fills range bounds, drops type/unit mismatches | seconds |
| 12 `review` | the final polish: judges the ~10,619 fields no rule can settle, with the model | ~14.7h |

**The table pass dominates, not the model.** Measured on 24 documents with four workers: with the
docling table pass on, screening took 521s and found 57 peptides; with it off, 92s and **zero**.
So the table grids are 82% of the step's time *and* the source of most of its yield --
`table_match` is the largest discovery path in the database. Disabling it is not an optimisation.

**Step 9 deliberately does not pass `--resume`.** `--resume` skips any document whose output file
already exists, and step 8 has just written one for every document, so resuming step 9 would skip
the entire corpus and add no LLM records at all. To continue an interrupted refined pass, just
re-run it: the payload cache returns the calls already made for free.

**Step 10 is non-destructive and idempotent.** A withdrawal clears the *sequence* while keeping the
`sequence_resolution` entry annotated with `withdrawn_reason`, so what was removed and why stays
legible; a record whose resolutions still hold is not rewritten. Run `DRY_RUN=1` first.

### Per-field model review — step 12, the final polish

Each field of an extracted record fails in its own way, so each gets its own prompt, JSON schema,
evidence budget and model. `sample/field_review.py` holds one spec per field; the failure modes in
each were measured on this corpus, which is what the prompts are built from.

**It runs after steps 9–11, and that order is the economy of it.** Step 9 extracts, step 10
revalidates the sequence resolutions it applied, step 11 makes every repair a *rule* can make —
1,334 units canonicalized, 931 range bounds filled, 16,927 incoherent activities dropped, no model
call at all. What reaches step 12 is the residue nothing cheaper can decide:

| what is reviewed | count | why a rule cannot settle it |
|---|---|---|
| target absent on a typed measurement | 24,638 | only the document names the organism |
| `activity_type` is a placeholder | 3,634 | the methods section names the assay |
| rare unit at an implausible magnitude | 1,848 | the paper may genuinely work in mg/mL |
| unit is `mM` on a potency metric | 1,715 | a lost micro sign and a real millimolar read identically |
| ordinary unit outside the potency band | 1,098 | either the unit or the number is wrong |
| sequence is three-letter notation | 859 | `CVS` for `CYS` passes every structural rule |
| value contradicts its own context sentence | 272 | the unit is right and the number was misread |
| stock solution suspected | 113 | no change of unit repairs it |
| unit carries a U+FFFD no rule can name | 96 | the lost character was not a micro sign |
| **total** | **34,273** | measured over 38,548 refined documents |

Those counts are from 2026-11-04, against a corpus that has grown to 38,548 refined documents
since the 10,619 figure was first measured; the shape of the work has not changed but its volume
has roughly tripled, most of it the absent-target group.

A sixth population is selected here but costs no model call: ~887 records whose extracted name
identifies a peptide this project catalogues under a different sequence. The name catalogue settles
those by rule, so step 12 records the finding rather than asking the model. Which of the two fields
is wrong is not recoverable from the record, so they are flagged, not corrected — see
[What counts as a new sequence](#what-counts-as-a-new-sequence).

Reviewing every field of every activity would be ~87,000 calls. The largest group is also the most
valuable: 45.6% of activities carry no target and are excluded from every usable tier for want of
that one field.

```bash
make step_12_review_extracted_fields DRY_RUN=1        # target counts, no model calls
make step_12_review_extracted_fields MAX_CALLS=500    # run the pass in pieces
make step_12_review_extracted_fields APPLY=1          # write corrections, not just record them

sample-cli review-fields --list                       # every spec: model, budgets, modes
sample-cli review-fields --show-prompt activity_unit  # the exact prompt and schema it sends
sample-cli review-fields --measure --field activity_unit --variant v1 --variant v3 --limit 40
```

**Corrections are recorded by default and written only with `APPLY=1`** — and then only when they
survive the field's own deterministic check, so this step can never introduce a defect step 11
would have to clean up. Each document records what was judged under `metadata.field_review`.

**The prompt is `v1`, because it measured best — the "optimized" rewrite was worse.** `v2` added
reason-before-verdict, a positive criterion for `ok`, and a clause naming the evidence. At n=10 it
looked better; at n=40 on the same cases it caught 10% of known-bad `activity_unit` defects against
v1's 75% (2 of 20 against 15). The positive criterion is the likely cause: telling the model the
text "does not have to restate the value verbatim" bought leniency, and `min` on an MIC does appear
in a paper about minutes. `v3` isolates the one change that might have helped (the evidence clause,
alone) and ships only if it measures better.

**The model is `gemma4:latest`, pinned** in `sample/field_review.py` rather than inherited from
`DEFAULT_EXTRACT_MODEL`. The two settings answer different questions: extraction's model is chosen
for recall on generation, a reviewer for precision on a verdict, and this project has recorded that
those rankings disagree — `gemma3:4b` wins on extraction recall but returned reagent names as
peptides and 65% of its records without a sequence. gemma4 is also the only model here with a
measured judgement score (12/12 on the triage verdict). If the two ever diverge, the run warns:
Ollama evicts one model to load another, measured at 31 llama-server restarts in 20 minutes.

| field | scope | evidence | needles | rule-settled modes |
|---|---|---|---|---|
| `peptide_name` | peptide | 1400 | name, sequence | 0 of 7 |
| `peptide_sequence` | peptide | 1600 | sequence, name | 5 of 7 |
| `activity_type` | activity | 1800 | name, value, organism | 1 of 4 |
| `activity_unit` | activity | 1200 | value, name, organism | 3 of 5 |
| `value_raw` | activity | 2000 | value, name, organism | 1 of 4 |
| `target_organism` | activity | 1600 | organism, value, name | 0 of 5 |
| `target_cell_line` | activity | 1400 | cell line, value, name | 0 of 3 |

Three measured constraints shape the design, and each one changes the answer:

**Most field defects need no model.** A rule settles 1,423 incoherent units, 1,163 mojibake units,
882 unbounded ranges and 68 prose types — 3,536 defects that a per-record LLM pass would have
spent calls rediscovering at ~38s of model time per document. `deterministic_verdict` runs first
for every field and short-circuits when it decides; the prompts are written for what is left.

**Context snaps to a rung, so the per-field knob is evidence, not `num_ctx`.** `llm_call` pins the
window and escalates once, because a continuously varying size makes Ollama rebuild its runner —
measured at 7 restarts in 10 minutes. Each spec therefore budgets *characters of evidence* so the
finished prompt lands on the bottom rung, and a test asserts it does.

**Needles are per field, because a field's own value is often the worst way to find it.** Measured
directly: reviewing `activity_unit="min"` on an MIC used `min` as the search needle, selected
windows around "minute", "determined" and "amino", and the model correctly answered *"unsure — the
document text provided does not contain any information about the activity unit"*. A unit is
written beside its **value**, so `value_raw` is the needle that finds it. Switching the needles
took that field from 0% to 67% recall on known-bad cases.

#### Model choice is measured, not asserted

Every spec currently names the pipeline model. That is a starting position, not a finding — and
deliberately so, for two reasons. Loading a second model evicts the first (measured at 31
llama-server restarts and 23 evictions in 20 minutes), so a per-field model has to earn its
eviction; and this project's rule is that two model swaps made on reasoning alone cost ~680
documents of degraded output.

`--measure` produces the justification. Labels come free from the corpus: known-bad cases are
records a deterministic rule rejects, known-good cases come from the trustworthy tier. The rule is
suppressed during the call, so the model has to reach the verdict itself.

```
activity_unit  (n=6)
  model             caught  false  decided  median
  gemma4:latest        67%     0%      33%   13.3s
```

`caught` is recall on known-bad, `false` is known-good wrongly rejected — the safety property, and
0 is the bar. `decided` is a **cost**, not an accuracy: declining on unsettled evidence is what
the prompt asks for. This is a *necessary* condition only — agreeing with the rules does not
establish accuracy on the ambiguous cases the prompts actually run on.

> Run `--measure` with a small `--limit` while an extraction is in progress, and measure one model
> at a time: testing a second model evicts the one the extraction is using.

### Acting on the verdicts — step 12's second half (`apply-verdicts`)

Recording a verdict is not correcting anything. Run without `APPLY=1`, step 12 writes verdicts and
nothing else, and this corpus reached **34,048 verdicts and zero corrections** after 37 hours of
local inference. `sample-cli apply-verdicts` is the pass that acts on them. It costs no model
calls: the proposals are already on disk, and this stage only decides which of them an authority
confirms.

Two layers, and most of the volume is in the one that needs no verdict at all.

**The mechanical layer** corrects what the record plus a local authority decides, across the whole
corpus rather than the 11.5% of documents the review reached:

| correction | authority | count |
|---|---|---|
| organism canonicalized, tax id attached | NCBI taxonomy | 18,353 |
| abbreviated genus expanded | the document's own spelled-out genera | 4,511 |
| assay name moved out of the unit column | the 38-term activity vocabulary | 1,518 |
| unit cleared — it was not a unit | measured table debris | 1,443 |
| cell-line spelling canonicalized | local Cellosaurus name list | 1,189 |
| placeholder cleared (`n/a`, `unsure`) | — | 1,167 |
| case variant folded | the unit and activity vocabularies | 641 |
| three-letter sequence decoded in place | the amino-acid alphabet | 349 |

**The authorized layer** takes step 12's proposals and writes only what the field's own authority
confirms — 6,366 of them, with **9,992 refused**. The refusals are the point: the review's answers
are frequently right in substance and unusable as text.

| refused because | count |
|---|---|
| the verdict's peptide name matches several peptides in the document | 5,809 |
| the answer names more than one organism | 1,063 |
| a name/sequence conflict is not a sequence error | 1,016 |
| the current unit is already in the vocabulary (step 11 settled it) | 928 |
| unresolved in NCBI taxonomy | 328 |
| the organism's domain is impossible for a microbial assay | 253 |
| not in the activity vocabulary | 94 |

Why each of those has to be a refusal rather than a write:

- **A conjunction is not a target.** The review answers `target_organism` with
  `"Acinetobacter baumannii or Pseudomonas aeruginosa"`. Every such string has a prefix that
  resolves perfectly well, so a lookup returns a confident answer and files the measurement
  against whichever organism was written first. OCR makes this worse by fusing the conjunction to
  the next genus — `"S. epidermidis andE. faecium"` has no word boundary between the `d` and the
  `E`, so it resolved cleanly to *Staphylococcus epidermidis* alone in a full corpus pass before
  the guard was widened.
- **An animal is not an MIC's target.** 253 proposals name a frog, a bee or a silkworm —
  *Rana pretiosa*, *Apis mellifera*, *Bombyx mori*, *Phyllomedusa sauvagii* — as the target of a
  minimum inhibitory concentration. Those are the peptide's **source** organisms. NCBI confirms
  each is a real species, so no name-based check would question it; only the domain does.
- **A name/sequence conflict says nothing about which is wrong.** 1,016 verdicts read "the name X
  identifies sequence Y, which this sequence does not match". That is true and it does not make
  the sequence wrong. Rewriting it would replace what the paper reported with what the catalogue
  holds, and those records are exactly the ones where the two sources disagree.
- **A verdict names its peptide and nothing else.** Step 12 records `peptide` but no peptide or
  activity index, so a name shared by two peptides in one document makes every verdict against it
  unattributable. 10.6% of verdicts name the empty string and 428 documents hold more than one
  unnamed peptide; applying those spreads one proposal across peptides the review never saw.

Measured across the corpus — 41,061 documents, **30,955 corrections**:

| | before | after |
|---|---|---|
| activities carrying a taxonomy id | **0** | **22,864** |
| abbreviated genus | 5,672 | 394 |
| organism placeholders (`n/a`) | 1,139 | **0** |
| distinct organism strings | 4,202 | 3,582 |
| unit case variants | 380 | **0** |
| activity-type case variants | 215 | **0** |
| units in vocabulary | 39,323 | 39,955 |
| types in vocabulary | 41,390 | 42,349 |
| sequences not in one-letter code | 929 | 396 |
| distinct sequence strings | 15,016 | 14,850 |

The 2,371 activities whose unit became empty are not a loss: each held an assay name or table
debris where a quantity belonged, so the number now has no unit and is excluded from every
quantitative view. That is the honest state — nothing in the record said what was measured.

> A logic fix in this module needs a **restore and re-run**, not a re-run. Every value it writes
> is well-formed, so a later pass with a better rule has nothing to object to and will not undo
> it. Copy the tree before applying; `APPLY_VERSION` records which logic produced a tree.

### Polishing already-extracted records (step 11)

Step 9 writes records under whatever rules were in force at the time, and those rules keep
tightening. A document is only re-extracted when the model pass runs again — a multi-day job on
this corpus — so without step 11 a rule fixed today would reach the corpus at the speed of
re-extraction. Measured on the output on disk when the step was added:

| defect | count | share |
|---|---|---|
| concentration type in a mass/time/energy/count unit | 6,173 | 11.6% of activities |
| unit carrying U+FFFD from PDF decoding | 1,098 | 2.1% of units |
| `value_raw` range whose upper bound was never stored | 2,312 | — |
| `activity_type` holding prose instead of a type | 58 | — |

The first is the one that matters most: `brevinin-1RTb MIC = 2548.2 Da` is a molecular weight read
out of a mass-spec table into the MIC column, and once such a row is in a database nothing
distinguishes it from a real MIC.

```bash
make step_11_polish_extraction_records DRY_RUN=1   # report only
make step_11_polish_extraction_records             # repair in place
make step_11_polish_extraction_records LIMIT=1000  # a sample of a large corpus
```

**It is idempotent**, verified on real files: a second pass changes nothing and leaves them
byte-identical. That is why it runs as part of `make all` rather than being a one-off backfill —
it is cheap (no model, no PDF, no network) and it guarantees that whatever is on disk conforms to
the current rules before step 14 becomes the first stage to read it. Each repaired document
records what was changed under `metadata.polish`.

It does **not** re-run `preprocessing.standardize_and_filter_activities`, which is not idempotent
on its own output: that function assigns `value_raw_unit` from the unit it is currently looking
at, so a second pass over an already-converted record overwrites the document-printed unit (`mM`)
with the canonical one (`uM`) and loses the fact that `value_raw` was measured in millimolar — a
1000× misreading for anything pairing those two fields. Each repair is instead individually
guarded to be a no-op once applied.

Safe to run while step 9 is extracting: documents are rewritten through a temporary file and
`os.replace`, and a file that changed between being read and being written is left alone and
counted as `skipped_changed_underneath`. Reinstating the version this step read would otherwise
discard a fresh re-extraction, and because that older content still carries the refined-pass
marker, step 9's resume would never revisit it.

Sequence-level correction is deliberately not here — that comes from step 9 re-extracting the
document and from step 10's resolution revalidation.

### Name-only peptides and synonym resolution

Most extracted peptides carry a name and an activity measurement but no sequence (72% of 20,604 in
a recent pubmed run), and `merge-extractions` keys on `peptide_sequence` with no name fallback --
`if not seq: continue` -- so those records are discarded before they reach the database.
`sample.synonyms` recovers the ones whose sequence can be established from evidence that can be
re-checked, in tiers, strongest first:

| tier | evidence | applied? |
|---|---|---|
| `internal_explicit` | the document writes `Name (SEQUENCE)` or `Name: SEQUENCE` | yes (0.95) |
| `accession` | the document cites a UniProt/PDB/GenBank id beside the name, and a catalogue entry carries the same one | yes (0.90) |
| `catalogue_corroborated` | one catalogue sequence, and the source organism or activity agrees | yes (0.80) |
| `catalogue_disambiguated` | several candidates, evidence leaves exactly one | yes (0.75) |
| `internal_alias` | an alias hop | recorded only (0.70) |
| `catalogue_unique_bare` | one catalogue match, nothing corroborating | recorded only (0.50) |

It **never joins on a name alone**: 9.4% of catalogue names map to more than one distinct sequence.
Recovery is ~6–7% of name-only records; rejected attempts are recorded too, so the rest can be
revisited without redoing the work. Two things it refuses outright: document-local labels
("Peptide 1", "4a", "C2") whose meaning is confined to one paper, and strings that are not peptide
names at all -- organisms ("E. coli"), table headers ("Range", "Control") and measurements
("0.62 l g ml - 1"), all of which the table reader does capture as names.

### Caches, OCR and the GPU

* **docling conversion cache** (`data/docling_cache`) -- keyed on PDF content, the OCR decision, the
  language set and a digest of the worker script. This is what makes step 9 cheap: it re-reads the
  same PDFs step 8 converted.
* **LangExtract payload cache** (`data/langextract_cache`) -- keyed on text content, model and
  chunk parameters, so re-running step 9 only pays for documents it has not already done.
* **OCR is a rescue, not a fallback.** 36.7% of sampled patents have a text layer of *exactly zero*
  characters, so OCR runs **before** the "not a document" gate and its output replaces the
  document's text -- otherwise those documents are skipped as empty before any PDF pass can run.
  The OCR language is detected per document (tesseract OSD narrows the script, the local model
  names the language); asking for `eng` on a Cyrillic page returns confident Latin look-alikes, and
  long runs of capitals are exactly what a sequence detector keys on.
* **TableFormer runs on the GPU** when `sample-docling` has a CUDA torch build, falling back to CPU
  silently otherwise. Screening disables the LLM pass, so the GPU is otherwise idle during step 8.

### Which DRAMP release is on disk

```bash
make sources-status          # downloads nothing; prints the detected release and the evidence
```

**DRAMP's download URL does not name the release it serves.** Every file comes from a path still
called `download_data/DRAMP3.0_new/`, and DRAMP updates those files in place. Verified 2026-10-16:
that path answers 200 with a 28-column DRAMP 5.0 header, while `DRAMP5.0/` and `DRAMP4.0_new/` both
404.

This had already gone wrong here, silently. The URLs say 3.0; the files fetched through them in
March 2026 are **4.0** (general 11,612 and patent 17,886 match 4.0's published table exactly); the
same URLs now serve **5.0**. Nothing recorded any of it, so "which release is this database built
from?" — a fair question to ask of a published resource — had no answer short of counting rows by
hand. Two things fix that:

* `detect_dramp_release` identifies the release **from the data** — row counts against the
  published tables, plus the `SMILES` column 5.0 introduced — and reports `None` rather than
  guessing when the two signals disagree.
* every download writes `data/antimicrobial_databases/dramp/MANIFEST.json`: per-file size, SHA-256,
  row and column counts, and the detected release.

A refresh is **opt-in**, because skipping existing files is what makes an interrupted step 5
resumable — and is also why this host sat on 4.0 for seven months:

```bash
make step_05_download_antimicrobial_peptide_metadata REFRESH_SOURCES=true
sample-cli databases --targets dramp --refresh        # the same, directly
```

Refreshing **archives** the existing tree to `dramp_<release>_<date>/` instead of overwriting it:
the current database was built from those files, so discarding them would make it irreproducible.

#### What DRAMP 5.0 adds, and what the pipeline now does with it

| | 4.0 | 5.0 |
|---|---|---|
| total entries | 30,260 | 32,296 (2,036 new) |
| general / patent | 11,612 / 17,886 | 12,784 / 18,715 |
| SMILES annotations | — | 22,298 |
| predicted structures | 1,570 | 16,016 |

The schema change is **append-only** — a `SMILES` column added last on the general, patent and
specific exports (not stability) — so nothing in the parsing path had to move.

That SMILES is read as a **trust tier in step 33**, and it sits *above* a sequence-derived
structure for modified peptides rather than below it. A derived structure is computed from the
sequence, so it describes a linear, unmodified molecule: for a cyclic, stapled or
terminally-modified peptide it is wrong in the worst way available — it parses, round-trips and
looks exactly as trustworthy as a right answer. DRAMP's curated string encodes the modification,
and the paper scopes it to precisely those cases (unmodified, cyclized/disulfide, terminally
modified; more complex chemistry is left to a later release). Both branches are counted in the step
log, so the trust decision is visible rather than implicit.

DRAMP's separate `*_smiles.*` downloads are deliberately **not** fetched: the same strings are
already the last column of the exports, and extra `.txt` files inside `Datasets/` would be read as
rows by `_dramp_rows`, which de-duplicates by `DRAMP_ID` on first-seen — so a two-column SMILES
stub could shadow a peptide's full record.

### Extraction knobs

| variable | default | effect |
|---|---|---|
| `LANGEXTRACT_BACKEND` | `1` | `0` disables the LLM pass (what step 8 sets) |
| `LANGEXTRACT_GATE` | `unit` | which documents reach gemma4. `unit` skips 8.5% of calls for 2.4% of its unique peptides; `assay+unit` skips 55.8% but loses **69.2%**; `none` gates nothing |
| `LANGEXTRACT_REQUEST_TIMEOUT` | `600` | per-request timeout to Ollama. The library default of 120s silently discarded whole documents on one slow chunk. Capped by whatever the document's budget has left |
| `SAMPLE_EXTRACT_CHUNK_SIZE` | `100` | documents processed between Ollama runtime resets; `0` disables chunking. See "Why extraction is chunked" below |
| `SAMPLE_DOCUMENT_BUDGET_SECONDS` | `300` | wall-clock seconds one document may spend; `0` disables the cap. A document that hits it is still written, flagged `metadata.budget_exceeded` |
| `SAMPLE_OLLAMA_RESET_CMD` | unset | a command to run at a chunk boundary instead of unloading the model (e.g. `sudo systemctl restart ollama`) |
| `DOCLING_OCR` | `auto` | `always` / `never` |
| `DOCLING_CACHE`, `LANGEXTRACT_CACHE` | set | empty string disables that cache |
| `SAMPLE_DOCLING_THREADS` | 16 | threads per conversion; the slot count is derived so `threads x slots <= cores` |
| `SAMPLE_SYNONYM_MIN_CONFIDENCE` | `0.75` | the bar a resolution must clear to be applied |
| `SAMPLE_NAME_LIKELIHOOD_THRESHOLD` | `-0.5` | below this bigram log-odds score, a sequence field is treated as a compound name rather than a sequence |
| `SAMPLE_LLM_THINK` | `0` | `1` lets a reasoning family (gemma4, qwen3, deepseek-r1, phi4) emit a chain of thought. Off by default — see the warning below |
| `SAMPLE_THINK_TOKEN_FACTOR` | `6` | multiplies each call's token budget when thinking is on, so the trace cannot starve the answer |
| `LANGEXTRACT_THINK` | follows `SAMPLE_LLM_THINK` | the same switch for the langextract path only |

> **Thinking mode has a measured failure mode.** Reasoning models write the trace to
> `message.thinking` and the answer to `message.content`, and `num_predict` caps the two
> *together*. On one verification prompt, thinking enabled returned `content=""` in 7.1 s with
> `done_reason="length"` — the trace consumed the whole budget — against the correct
> `{"supported": false}` in 0.3 s with `think=False`. That is exactly what the "gemma4 returns an
> empty completion" symptom was, and disabling it is what took gemma4 from unusable to 12/12 on
> the triage benchmark. `SAMPLE_THINK_TOKEN_FACTOR` exists to make thinking survivable by giving
> the answer room; it has **not** been measured against a real extraction run. Verify with
> `make curation-model-check MODEL=<tag>` before a long run.

**Thinking also moves the context rung**, so it is one switch rather than four. A trace shares the
window with the prompt and the answer, so turning `SAMPLE_LLM_THINK` on steps `DEFAULT_NUM_CTX`
4096 → 8192 and the fallback 8192 → 16384; turning it off steps them back.
`langextract_records.NUM_CTX` derives from `DEFAULT_NUM_CTX` rather than repeating it, so the two
cannot drift — a mismatch makes Ollama rebuild the runner between the langextract backend's calls
and everything else's.

**Whatever the rung is, the server default must match it**, or Ollama rebuilds the runner on every
request — the exact fault the pinning exists to prevent. With thinking off (the default) that is
4096:

```
# /etc/systemd/system/ollama.service.d/override.conf
OLLAMA_CONTEXT_LENGTH=4096     # 8192 if SAMPLE_LLM_THINK=1
OLLAMA_NUM_PARALLEL=2
OLLAMA_KV_CACHE_TYPE=q8_0
```

### Why extraction is chunked

Steps 8 and 9 process documents in groups of 100 and reset the Ollama runtime between groups.

This is a response to a measured decay, not a precaution. Throughput on the current corpus held
near 1 doc/s and fell to roughly **0.04 docs/s after about 2,000 documents** -- a 25x collapse that
raised no error, logged no failure, and recovered completely when `ollama` was restarted by hand.
Everything on the client side is bounded and observably drains; the `llama-server` runner is the
only component that spans an entire multi-day run while holding a KV cache per slot and an internal
request queue nothing in this project can inspect. Chunking replaces an unobservable cause with a
bounded one: runner-side state can accumulate for at most one chunk before it is discarded.

The reset is an **unload** (`keep_alive: 0`), which makes Ollama terminate the `llama-server`
subprocess; the next request starts a fresh one. That needs no privileges, which matters for a
pipeline run by an unprivileged user — a reset requiring a sudo password would simply not happen
overnight. `SAMPLE_OLLAMA_RESET_CMD` is there for anyone who wants a service restart instead.

**The reset is global to the Ollama server**, not scoped to the step that triggers it. Do not run
two extraction steps, or `make curation-model-check`, against the same Ollama at the same time: one
of them will unload the model out from under the other. Set `CHUNK_SIZE=0` on the second if you
must overlap them.

Each document also gets a **300-second budget**. It is not a timeout in the usual sense, and the
difference shows up in the output: it cannot abort a document, because no thread can be interrupted
in CPython and abandoning the worker would leave it running and still holding an Ollama slot --
which is the backlog a timeout is meant to prevent. What it does is stop a document *starting* new
expensive work once its time is gone. Every LLM call is clamped to the remainder and refused below a
5-second floor, and the langextract subprocess (the one thing here that genuinely can be killed) is
hard-capped. A document that hits the cap therefore **completes with fewer records and says so** --
`metadata.budget_exceeded`, beside `processing_seconds` -- rather than vanishing. A missing output
file is indistinguishable from "not reached yet" and would be re-attempted forever by `--resume`;
a flagged one can be counted, audited, and re-run deliberately.

One operation is deliberately exempt: the **docling conversion**, which keeps its own 900s timeout.
On a scanned document OCR runs at ~12 s/page, so a long one can spend the whole budget before any
model is called — but clamping it instead would kill the conversion and leave the document with no
text at all, and text-and-tables without an LLM pass beats nothing. So on a corpus of large scanned
documents (step 8, where the table pass is most of the work) expect `budget_exceeded` flags, and use
`DOC_TIMEOUT=0` there.

```bash
# the defaults, explicit
make step_09_refined_extract_peptide_data CHUNK_SIZE=100 DOC_TIMEOUT=300

# the old single-pass behaviour
make step_09_refined_extract_peptide_data CHUNK_SIZE=0 DOC_TIMEOUT=0
```

---

---

## Web Dashboard & Platform

### Running the full stack (recommended)

Predictions run as **background jobs**, so the dashboard needs a Redis broker and a Celery worker.
`docker-compose.yml` brings up all three (the `web` service runs the app under **uvicorn**, ASGI):

```bash
docker compose up -d --build                 # redis + web (uvicorn) + worker
docker compose --profile monitoring up -d    # ...and Flower on :5555
```

The dashboard is then on `http://localhost:8000`. The host `data/` directory is mounted rather than
copied, because it holds the trained models and release datasets the pipeline produces outside the
containers. `web` and `worker` deliberately run the **same image**, so the two agree on their
scikit-learn and xgboost versions exactly.

### Running without Docker

```bash
redis-server &                    # or point SAMPLE_BROKER_URL at an existing one
make run-worker                   # the Celery worker
sample-cli webserver              # the dashboard (uvicorn)
```

Without a reachable broker the dashboard still serves every other page; prediction submissions are
refused with a clear 503 rather than accepting a job that would never run. For development,
`SAMPLE_CELERY_EAGER=1` runs predictions inline in the request.

| Variable | Default | Purpose |
|---|---|---|
| `SAMPLE_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `SAMPLE_RESULT_BACKEND` | `redis://localhost:6379/1` | Celery task state |
| `SAMPLE_PREDICTIONS_DB` | `data/predictions.db` | Prediction job/result store |
| `SAMPLE_MODEL_DIR` | `data/sample_final_release/models` | Trained-model namespace root |
| `SAMPLE_CELERY_EAGER` | unset | Run tasks inline, no broker needed |

### How prediction jobs work

1. Choose a model and paste sequences (plain or FASTA, up to 1000) on `/models`.
2. `POST /predict` validates, records the job, queues it, and returns a **UUID** with HTTP 202.
3. The browser goes to `/predict/<job_id>`, which shows a spinner and polls
   `/predict/status/<job_id>` with a 1s→5s backoff until a terminal state.
4. Results are written to **`data/predictions.db`**, separate from the read-only release
   `data/sample.db` (rebuilt by step 46 and shipped to Zenodo) so a pipeline re-run can never
   destroy a user's results and a busy queue never holds write locks on the served dataset. An
   invalid sequence is recorded as a failed **row**, not a failed job — one typo among fifty does not
   discard the other forty-nine. Each prediction row also carries an **applicability-domain** flag
   (k-NN distance to the training set); out-of-domain sequences are still scored, but flagged.

`sample.result_cache` (Redis, seven-day TTL, keyed on the exact input) exists for analyses that
sit outside this queue because they are one-off and deterministic rather than batched. Its two
callers were the Expression Wizard's fold and yield endpoints, which left this repository with
the wizard; nothing imports it today. The cache is deliberately **not** a record of truth — an
eviction, a restart, or no Redis at all simply means the work is redone — and a failed run is never
cached, since that reflects the machine rather than the input.

The endpoints also serve scripted callers:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"model_id": "general/general/any/classification/descriptors",
          "peptides": ["GIGKFLHSAKKFGKAFVGEIMNS", "ILPWKWPWWPWRR"]}'
curl localhost:8000/predict/<job_id>/json
```

### Dashboard features

* **ML Models** — browse the models the FLAML rollout (steps 45–46) actually published: a model is
  offered only when its namespace cleared the quality gate *and* the run recorded it as written, so
  a rollout that died part-way leaves no half-trained, unthresholded model selectable. Each model
  links to a **dataset** page showing
  its full validation report (normality test, cross-validation, bootstrap CI, y-randomization,
  quality gate) and its exact train/test peptide split. Submit sequences and follow the job by id.
* **Explore / Datasets** — browse, search, filter, and inspect peptide records; per-peptide detail
  with 2D structure, DSSP secondary structure, a Ramachandran plot with a favored/allowed/outlier
  call per residue, disorder diagrams, families, modifications, cross-references, and citations.
  Four further panels appear when their data exists: an **ADMET-lite** profile (with hemolysis
  resolved from the annotator once
  `build-annotations` has scored the peptide, and permeability left as an explicit gap), the
  pre-computed **annotation** (five external predictors, Pfam domains, near-identity and family-HMM
  hits, with a caveat shown for circular peptides), **PDB structural validation** (once
  `build-pdb-homology` has reached the peptide and RCSB holds a close enough match: the ColabFold
  prediction superposed onto the closest real, experimentally-solved structure, reported as
  CA-RMSD — the one check that says whether a prediction agrees with reality, not only with
  itself), and the **genomic context** sunburst once a
  `genomic-context-sweep` has recorded hits for that peptide.
* **Search** — four modes: **BLAST** (`blastp`), **HMMER** (profile search), **PepMatch**
  near-identity, and **ESM-2 embedding** k-NN.
* **Analytics** and **External DBs** — dataset analytics and cross-links to the source databases.

---

## Documentation & Tests

```bash
make docs        # Sphinx HTML into docs/_build/html/index.html
make test        # the full pytest suite
```

Design decisions and their rationale live in [`DECISIONS.md`](DECISIONS.md); the remaining backlog
is in [`TODO.md`](TODO.md); blocked items and known limitations in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

---

## Provenance

### Split out of this repository

* **Expression Wizard** (2026-11-23) — peptide-to-synthetic-gene construct design: codon
  optimization, His/carrier/SUMO fusion assembly, CAI, 5' mRNA structure, restriction-site
  removal, GenBank export, plus optional ColabFold and MPEPE analyses. It was
  `sample/expression.py`, `sample/templates/expression.html`, four `/api/v1/express*` routes and a
  `sample-cli express` command. Nothing in the numbered pipeline ever depended on it, which is
  what made the split clean. It now has its own web and CLI interfaces, its own Docker Compose
  setup and its own repository; the `sample-expression` and `sample-mpepe` environments went with
  it. `sample.result_cache` stayed behind and is now unused — see its module docstring.

### Data provenance

The released tables distinguish what this project derived from what it collected:

* **Third Party** — read out of ADP, CAMP, DBAMP, DRAMP, or DBAASP, carrying the source database's
  own record id and the publication behind it.
* **SAMPLE** — extracted from primary literature or patents by this pipeline.

`activities.csv` additionally marks whether each SAMPLE measurement is `New in SAMPLE` or
`Corroborates` an existing third-party finding, and whether each sequence is new to the field —
computed by comparison rather than asserted, so the claim that SAMPLE adds new data is checkable.

## Correcting unit scale, and the faults that look like it

Measured on 2026-11-03 against the curated databases, over 4,446 findings this project extracted
that a source database also reports for the same peptide, assay and organism:

| unit | overlapping findings | >10x off | ~1000x off | median ratio |
|---|---:|---:|---:|---:|
| `uM` | 2,322 | 149 | 2 | 1.66x |
| `ug/mL` | 1,416 | 83 | 6 | 1.00x |
| **`mg/mL`** | **670** | **642** | **485** | **1000.00x** |
| `nM` | 31 | 13 | 4 | 1.15x |

`uM` and `ug/mL` agree as well as two labs measuring the same MIC ever do. `mg/mL` does not agree
at all: its median ratio is exactly 1000, which is a lost micro prefix rather than a measurement
difference. **Excluding that one unit, agreement is 67.0% within 2x, 86.0% within 4x and 93.4%
within 10x, median 1.43x.** Including it, `mg/mL` is 56% of every gross disagreement in the corpus.

The prior agrees: across 182,766 curated potency measurements `mg/mL` appears **38 times**, 0.02%.
This extraction has **1,733**, forty-five times the entire curated corpus.

### Why this is not "rewrite mg/mL as ug/mL"

Pulling one record apart shows three faults wearing the same clothes:

```
MIC          value_raw "2"    mg/mL   "cathelicidin-RC1 solution (2 mg/ml, dissolved in
                                        sterile deionized water) was incubated at ..."
MIC          value_raw "0"    uM      "The MICs of cathelicidin-RC1 and meropenem against
                                        E. coli ATCC25922 are 4.69 ..."
Bactericidal value_raw "75"   mg/mL   "Among the 48 microorganism strains, 34 strains could
                                        be killed by cathelicidin-..."
```

The first is a **stock solution** captured as a potency — 2 mg/mL is what the peptide was
dissolved at, and no rescaling makes it an MIC. The second is a **parse failure**: the unit is
right, the sentence says 4.69, the field says 0. The third is not a concentration at all.
Rescaling all three turns one wrong answer into a differently wrong answer and erases the evidence
that anything was amiss. So [`sample/scale.py`](sample/scale.py) diagnoses rather than rewrites,
and separates what it can prove from what it cannot.

### The evidence is the JSON, not the PDF

Extraction already saves `context_sentence`, the sentence each value came from — present on
**53.2%** of activities, **32.8%** with a readable value and unit, median length 66 characters.
For a scale question that sentence is usually decisive, and reading it costs nothing: no PDF is
reopened and no passage has to be relocated. The record's **sibling measurements** are the second
source — a paper reports a table of MICs in one unit, so when one row disagrees the others are the
evidence.

### What gets corrected without a model, and what does not

Step 11 applies only findings the JSON *proves*, which is two cases: the activity's own context
sentence prints that very value in a different unit, or the unit is one a peptide potency cannot be
stated in at all. `g/mL` is three orders past solubility and appears **zero** times in 182,766
curated measurements, so when it shows up the document's own spelling is the damage, not evidence
of intent. **Only the unit string changes; no number is touched** — `value_lower` is expressed in
`activity_unit`, so restoring a dropped prefix is a change of unit at a fixed number.

Everything merely *improbable* is escalated. `mg/mL` is rare, not impossible: a paper may report
one assay in it, its context sentence often confirms it (`"IC50 was reported to be 6.62 mg/ml"`),
and 12 of the first 14 such records had no context sentence at all. Sibling agreement is
suggestive, not proof.

| | activities | handled by |
|---|---:|---|
| corrected by rule | 268 | step 11, no model call |
| rare unit at an implausible magnitude | 1,848 | step 12, `activity_unit` |
| ordinary unit outside the potency band | 1,098 | step 12, `activity_unit` |
| contradicts its own context sentence | 272 | step 12, `value_raw` |
| stock solution suspected | 113 | step 12, `value_raw` |

Of the 268 rule corrections, 227 are `g/mL` to `ug/mL`.

### Two corrections that needed no model at all

Measured over 51,572 activities carrying a unit, 13,258 were outside the release vocabulary. Two
deterministic repairs account for 2,240 of them.

**Notation damage, 1,212 activities.** `preprocessing.canonicalize_unit` handles the spellings the
pipeline produces, and repaired *none* of the ones PDF extraction produces:

| written | repaired | count |
|---|---|---:|
| `u M` | `uM` | 251 |
| `u g/mL` | `ug/mL` | 142 |
| `AM` | `uM` | 113 |
| `WM` | `uM` | 113 |
| `lgml` | `ug/mL` | 75 |
| `mg/liter` | `mg/L` | 70 |
| `Ag/mL` | `ug/mL` | 58 |
| `lm` | `uM` | 40 |
| `ug mL−1` | `ug/mL` | 33 |
| `microgram/mL` | `ug/mL` | 25 |
| `$\mu$ M` | `uM` | 20 |

Every one is a unit the field does use, written in a way the vocabulary does not contain — a lost
micro sign, a split prefix, a reciprocal exponent, or LaTeX that survived extraction. Step 11 now
applies `scale.canonical_unit` to whatever the first canonicalizer leaves off-vocabulary, **but
only when the result lands in the vocabulary**, so `kcal/mol`, `fold`, `Range` and `540 nm` are
left exactly as written rather than coerced into the nearest concentration.

**A unit that is not a unit, 1,028 activities.** `metric_as_unit` was declared deterministic and
counted at 100, and nothing implemented it: `MIC` appears in the unit column 622 times, `MBC` 111,
`MRSA` 89, `MFC` 28, `VRE` 25. Every one was escalating to a model that cannot recover the real
unit either, because it is not in the record. These are now settled by rule and **flagged, never
guessed**.

Recovering the unit from the context sentence was tried and rejected. Only 219 activities had an
exact value match, and the matches were wrong: `50 nmol of OVA-I peptide` is a dose in moles, not
a concentration, and `0.22 mmol/kg` is per body weight. Worse, the regex that found them matched
the `nm` inside `nmol`, which would have let a sentence "correct" a sound concentration into the
wrong unit — so an amount is now refused unless it carries an explicit per-litre denominator.

### The same context serves every field

`record_context` builds one evidence block — recorded sentence, sibling measurements, and the
deterministic diagnosis when there is one — and every field spec receives it: `activity_unit`,
`value_raw`, `activity_type`, `target_organism`, `target_cell_line`, `peptide_name` and
`peptide_sequence`. It is labelled in the prompt as what the *extractor* recorded rather than as
the document's own words, because it is extraction output and can itself be the thing that went
wrong — one measured record carries a stock-solution sentence under an MIC. Where it disagrees
with the document text, the document wins.

Two measured traps are encoded as tests rather than prose:

- **No-op proposals.** `mg/L` and `ug/mL` are the same quantity spelled differently. Before that
  guard, every one of 398 "confident" findings was exactly this no-op.
- **Band-chasing instead of prefix restoration.** Both `ug/mL` and `ng/mL` put 640 inside the
  plausible band, and choosing by band proposed `g/mL → ng/mL` on nine records — a 10⁹ correction
  for a 10⁶ error. The micro prefix is what the glyph loss dropped, so the micro prefix goes back.


## What counts as a new sequence

`sequence_novelty` used to be set membership: canonicalize the sequence, ask whether the string
appears in the merged source databases, and call it `SAMPLE (new)` if not. That is right about what
it asks and wrong about what the column is read as meaning. Over a 32,408-document snapshot the test
called 975 sequences new; hand-classifying every one of them left roughly 230 defensible, and that
classification is what the graded test below was built from:

| What it actually was | Count |
|---|---:|
| A fragment of a catalogued sequence | 145 |
| A catalogued sequence plus flanking text | 121 |
| Three-letter notation, never decoded (`ARGLEUTYRLEU…`) | 103 |
| Within two substitutions of a catalogued sequence | 60 |
| An identifiable MHC epitope (SIINFEKL in 76 documents) | 46 |
| Carrying an ambiguity code (`IIIUKIKTDO`, labelled melittin) | 36 |
| Within two edits once insertions are allowed | 5 |
| A name or prose spelled in residue letters (`ISGRIFFITHSIN`) | 4 |

The decisive measurement is narrower and worse. For the 40 of those sequences whose extracted
*name* is a peptide this project already catalogues, the sequence disagreed with the catalogued one
**40 times out of 40**. Where novelty was checkable at all, it was always an extraction error.

[`sample/novelty.py`](sample/novelty.py) replaces the binary test with a graded one. Scopus
extraction is still running, so these counts drift upward between runs; measured against 32,473
refined documents, the old test claims 969 novel sequences carrying a trustworthy measurement and
the graded test claims **460**, a 53% reduction. A further 388 are reported as defects —
uncountable either way — rather than silently claimed.

| Verdict | Meaning |
|---|---|
| `known_exact` | The catalogue holds this sequence |
| `known_encoding` | Three-letter notation of a catalogued sequence |
| `known_fragment` | A fragment of a catalogued sequence |
| `known_extension` | A catalogued sequence with flanking text |
| `analogue` | Within `MAX_ANALOGUE_EDITS` (2) edits, parent and distance reported |
| `name_conflict` | The name identifies a catalogued peptide the sequence does not match |
| `not_comparable` | An ambiguity code, or too short for absence to mean anything |
| `novel` | Survives every test above |

Three principles decide the design, each forced by a row of that table:

- **A sequence that cannot be compared is not novel.** B/Z/X/U/O are legitimate IUPAC codes and
  `validate_extracted_sequence` accepts all five on purpose, but each masks a position, so identity
  — which novelty requires — is undecidable. Below 8 residues absence carries no information
  either. Both get `not_comparable` rather than a guess in either direction.
- **Notation is not identity.** `sequence_to_1_letter` handles `Gly-Ile-Gly`; PDF extraction loses
  the hyphens, and `GLYILEGLY` reached novelty undecoded. That was the largest single source of
  false novelty.
- **The sequence is judged first, then the name is asked to corroborate it.** The reverse order was
  tried and measured: it reported 107 name conflicts in the trustworthy tier against the correct
  29, because a name test applied before the containment tests cannot tell a truncation from a
  different peptide — `DMS-H4` carrying `GLWSTIKNVGKEAAIAAGKAALGAL` where the catalogue holds
  `GLWSTIKNVGKEAAIAAGKAALGALG` is the same peptide, one residue short.

### Ambiguous names

A name is evidence about a sequence only when it identifies something. The threshold is
`names.NAME_SEQUENCE_THRESHOLD`, already measured for the name harvest: a name attached to 20 or
more distinct sequences is a class label or an organism that some source filed in a name column —
dbAMP supplied `Mouse-ear cress` as a name for 300 peptides. Names are classified `unique`,
`series` (an analogue series, a handful of sequences), `ambiguous`, `unknown` or `absent`, and only
the first two adjudicate. The rest are recorded as `silent`, which is deliberately distinguishable
from `disagrees`: "the name had no standing to say" is not "the name contradicts this".

Over the refined corpus, 8,850 records carry a name that adjudicates. 7,682 corroborate their
sequence; **865 (10%) contradict it**, and those split two ways:

- **439** have a sequence matching nothing catalogued, so the novelty claim collapses to
  `name_conflict` and the row is published `Unverifiable`.
- **426** have a sequence the catalogue *does* hold, under a different peptide's name. The novelty
  verdict stands — the catalogue holds the sequence — but `name_agreement` marks the record a
  defect, because every activity on it is attributed to the wrong peptide.

The second case is `sequence_of_another_peptide`, the one failure mode on this field that nothing
could previously detect: the sequence is structurally perfect so no rule objects, and a model shown
only the document cannot know the name belongs elsewhere. Step 12 now settles it by rule.

Three columns carry the detail into `activities.csv` so the refinement is auditable rather than
asserted: `novelty_basis` (the verdict), `novelty_parent` (what it matched against) and
`name_agreement`. `sequence_novelty` gains one value, `Unverifiable`; the original three are
unchanged, so anything already reading the column keeps working.

**Novelty is not scope.** An MHC epitope absent from every AMP database is a genuinely new
*sequence* — SIINFEKL is simply not an antimicrobial peptide. 46 of the 975 were identifiable
epitopes, and judging that is a domain question, not an identity one. Folding it in here would
repeat the mistake the name ordering already made, so it is left to a separate scope column that
does not exist yet.

