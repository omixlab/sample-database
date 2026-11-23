# Conda environments

Each stage of the pipeline runs in the environment carrying its dependencies and no more. Every
file here defines exactly one environment and is the single source of truth for it.

| File | Environment | Steps | Carries |
|---|---|---|---|
| `sample-acquisition.yml` | `sample-acquisition` | 1–5 | scraping, HTTP, PDF text |
| `sample-extraction.yml` | `sample-extraction` | 6–14 (except 8), 22–31 | NLP, LLM, validation, merges, names, bibliography, families, family HMMs, modifications, source organisms, cross-references, Gene Ontology, GO `.obo` slice |
| `sample-chem.yml` | `sample-chem` | 15–21, 32 | RDKit, embeddings, decoys, assembly, SMILES validation, embedding-search index, SQLite load |
| `sample-ml.yml` | `sample-ml` | 8, 33–38 | scikit-learn, xgboost, plots |
| `sample-web.yml` | `sample-web` | 39 | FastAPI, Celery, BLAST |
| `sample-predictors.yml` | `sample-predictors` | 16 (sub-call) | third-party activity predictors |
| `sample-docling.yml` | `sample-docling` | 7–8 (sub-call) | docling/TableFormer table extraction, Tesseract OCR |
| `sample.yml` | `sample` | all | the union of every file above |

```bash
make envs        # creates all six pipeline environments and verifies the pins
make env-full    # additionally creates `sample`, needed for the docs build
```

`make envs` installs the package into each environment with `pip install --no-deps -e .`, so
`sample-cli` exists everywhere. `--no-deps` matters: letting pip re-resolve from `pyproject.toml`
would pull the union back into every environment and undo the split.

`../environment.yml` is a symlink to `sample.yml`, so `conda env create -f environment.yml` from
the repository root still works.

## How the groupings were derived

Not by judgement about what "feels" related. Each step's entry modules were resolved to their
**transitive import closure** over `sample/`, and steps sharing a closure were grouped. That is
why steps 7 and 19 sit with the extraction stage despite looking like merge operations: both
enter `sample/nlp.py`, which pulls in spacy, nltk, torch and the sentence-transformers stack.

It is also why the extraction environment is as large as it is and **cannot usefully be split
further**. Doing so means refactoring those imports first, not writing more environment files.

The acquisition environment is the clearest win: steps 1–5 touch no torch, no RDKit, no
scikit-learn and no spacy, so it creates in a fraction of the time — and those are the steps most
often re-run alone, after a failed download batch or a database refresh.

## Two things a static analysis of the source would get wrong

Both are load-bearing, and both are invisible to an import scan.

**1. The webserver needs scikit-learn and xgboost, though it imports neither.** They are required
at *unpickle* time: `sample/model_registry.py` loads estimators trained by steps 46-47, and the
pickle names `sklearn.ensemble._forest` internally. An environment built from imports alone
starts the server fine and cannot serve a single prediction.

**2. `python-dotenv` is needed in every environment.** `sample/cli.py` and `sample/config.py`
import it at module level, so it is required by anything running `sample-cli` — not only the
steps that read a `.env` file.

## Pinned versions, and why `make check-envs` exists

Three environments share pickles across their boundaries:

```
step 12 trains rf_filter_model.pkl in sample-ml  ->  step 8 loads it in sample-extraction
steps 46-47 train the models in sample-ml        ->  the webserver loads them in sample-web
```

A pickled estimator carries no version negotiation. Unpickling a scikit-learn 1.9 RandomForest
under 1.3 either raises `InconsistentVersionWarning` and hands back a subtly different object, or
fails outright. The first case is the dangerous one: the symptom is wrong predictions, not an
error, and nothing in the output indicates it.

So `scikit-learn`, `xgboost` and `numpy` are pinned identically in `sample-extraction`,
`sample-ml`, `sample-web` and `sample`, and `make check-envs` asserts that the *installed*
versions actually agree. The pins are verified, not merely intended.

## Why `sample-predictors` is separate

`hemopi2` and `toxinpred3` ship pickled estimators built against scikit-learn 1.3.x. Installing
them into the main environment downgraded scikit-learn there from 1.9.0 to 1.3.1 — silently, and
in a way that would have changed the behaviour of every model trained by steps 46-47 while every test
still passed.

Steps 23-26 cross the boundary by shelling out: `sample/peptide_predictors.py` runs each predictor
through `conda run -n sample-predictors …` and merges the columns back. The Makefile exports
`SAMPLE_PREDICTOR_ENV` so relocating the environment needs no code change.

Each adapter checks the executable actually exists in the target environment's `bin/` before
reporting the predictor available. A missing or half-installed predictor contributes no columns,
rather than a column of fabricated zeros that would read downstream as a real negative result.

## Adding a predictor

Of the six tools the annotator (`sample/annotator.py`) was designed around, only `hemopi2` is both
installed and working. Status as of the 2026-08-12 check (see `KNOWN_ISSUES.md` and `DECISIONS.md`
for the full detail on each):

- **`macrel`** — its adapter's two real bugs are fixed and tested, but the package itself is not
  installed anywhere: it pulls its own pinned scikit-learn and numpy, so it needs its own
  environment rather than a new line in `sample-predictors.yml` (installing it there once, to
  verify the fix, was reverted the same session for exactly that reason). Add a `sample-macrel.yml`
  here, point the adapter at it with its own `SAMPLE_*_ENV` variable, and the availability report
  picks it up.
- **`toxinpred3`** — installed, but its shipped model pickle is incompatible with the scikit-learn
  version it is paired with and cannot currently run at all.
- **`AMPlify` and `AntiCP`** — neither has an installable package under an obvious name. (The PyPI
  package literally named `amplify` is an unrelated quantum-computing SDK -- installing it to
  check would be a mistake, not a fix; see `DECISIONS.md`.)

## Running against a different layout

Every environment name is a Makefile variable:

```bash
make step_47_train_antibacterial_ml_model ENV_ML=my-other-env
make all SINGLE_ENV=sample        # collapse the split entirely
```

The docs build always needs the full environment: Sphinx autodoc imports every module in
`sample/` to read its signatures, so a narrower environment yields a build full of
import-failure stubs rather than an error.

## The Ollama server is configuration too, and it is not in any environment file

Every LLM-using stage talks to one local Ollama server, so its settings bound the whole pipeline
and no conda environment describes them. They live in a systemd drop-in:

```bash
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'CONF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_KEEP_ALIVE=24h"
CONF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**`OLLAMA_NUM_PARALLEL` is the one that matters, and its default is a trap.** Recent Ollama
defaults it to **1**: the server then serves one request at a time and every additional client
request queues, silently. Measured on this host at the default, during the refined extraction
pass: the single slot was busy **97.7%** of a ten-minute window at 0.32 calls/s while **eight**
client requests were in flight (4 extraction workers x `LANGEXTRACT_MAX_WORKERS=2`), so seven of
every eight were waiting. Throughput was 79 documents/hour — 20 days for the corpus — while the
GPU showed 93% busy with 7.4GB of its 12GB unused. Nothing errors; throughput is just flat.

After setting it to 4, all four slots are in use and occupancy across them runs ~78%.

Verify it took effect, because a `daemon-reload` without a restart silently changes nothing:

```bash
systemctl show ollama -p Environment            # must list OLLAMA_NUM_PARALLEL=4
pgrep -a llama-server | grep -o '\-np [0-9]*'   # must print: -np 4
ollama ps                                       # PROCESSOR must stay 100% GPU
```

That third check is the one to watch when changing models or raising the slot count. Four slots
multiply the KV cache, and if the total no longer fits in VRAM Ollama offloads layers to CPU
instead of failing — which costs far more than the parallelism gains. If `ollama ps` shows a
CPU split, lower `OLLAMA_NUM_PARALLEL` rather than accepting it.

`sample/pipeline_runner.py` warns at startup when it cannot confirm this setting. It cannot read
it — Ollama exposes no endpoint reporting its slot count — so the warning names the checks above
rather than asserting a fault.
