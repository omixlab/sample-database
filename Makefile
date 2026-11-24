# Define starting and ending steps to run
START_STEP ?= 1
END_STEP ?= 54
SOURCES ?= all

# RESUMING. Every step continues from where it stopped; nothing already finished is redone.
#
# Two independent mechanisms, because one cannot cover both cases:
#
# 1. STEP level, in `run_step` below. Each step records itself in data/pipeline_runs.sqlite
#    (sample/run_registry.py) under a run id. A step already recorded SUCCESS under the current
#    run is announced and skipped, so a `make` interrupted at step 30 continues at step 30
#    instead of rebuilding the 29 steps behind it. A step that was RUNNING when the run died is
#    marked INTERRUPTED and re-run, because it has no trustworthy artifact.
#
# 2. ITEM level, inside the steps whose commands support it (8, 9, 36 and the stores that take
#    --no-resume). This is what continues a single long step part-way through, and matters most
#    for extraction: re-processing an already-refined document costs ~3s (1,195 docs/h, the
#    payload cache returning its model calls) against ~38s for a new one (94 docs/h), so
#    re-walking 11,578 finished documents to reach the remainder is ~9.7 hours of nothing.
#
# FROM_SCRATCH=1 turns both off and starts a NEW run in which nothing is complete:
#
#     make                     # resume: the open run, skipping what it already finished
#     make FROM_SCRATCH=1      # a new run id, every step re-run from the beginning
#     make pipeline-status     # the progress table for the last run
#     make pipeline-runs       # recent runs, newest first
#
# FROM_SCRATCH does NOT delete anything. It re-runs steps that overwrite their own outputs; it
# is not a `clean`.
FROM_SCRATCH ?= 0

# The registry runs under the SYSTEM python3, not `conda run`. It is standard-library only for
# exactly this reason: `run_step` calls it three times per step, and conda's interpreter and
# solver startup is 1-2s a call -- a minute of overhead per invocation across 54 steps, paid even
# by `make help`. See sample/run_registry.py's module docstring.
REGISTRY ?= python3 -m sample.run_registry

# Item-level resume flags, derived so that resuming is the default and FROM_SCRATCH=1 the
# exception. RESUME_REFINED is a different mode from RESUME_EXISTS on purpose -- see step 9.
# Three spellings because the commands disagree, not because the intent differs. RESUME_FLAG is
# the plain opt-in `--resume` (steps 45-46, which default to retraining every namespace);
# RESUME_EXISTS/RESUME_REFINED are extraction's modes; NO_RESUME is the opt-OUT carried by the
# commands that already resume by default (step 36 and the annotation/homology/sweep stores), for
# which FROM_SCRATCH has to actively ask for a rebuild.
ifeq ($(FROM_SCRATCH),1)
RESUME_FLAG    :=
RESUME_EXISTS  :=
RESUME_REFINED :=
NO_RESUME      := --no-resume
else
RESUME_FLAG    := --resume
RESUME_EXISTS  := --resume-mode exists
RESUME_REFINED := --resume-mode refined
NO_RESUME      :=
endif

# RESUME was a boolean that applied to step 8 alone and defaulted to OFF. Both of those are now
# wrong: resuming is the default everywhere, and step 9 needed a mode rather than a flag. Fail
# loudly rather than ignore it, so an operator who still passes RESUME=true is told where it went
# instead of quietly getting the opposite of a resume.
ifdef RESUME
$(error RESUME has been replaced. Steps resume by default now; use FROM_SCRATCH=1 to force a full rebuild, or pass --resume-mode {off,exists,refined} to `sample-cli extract` directly.)
endif
#
# Extraction is SPLIT into two numbered steps rather than one target with a mode flag, because the
# two passes have different costs, different failure modes and different outputs, and because the
# screening pass is a usable result on its own:
#
#   step 8  screening  NLP + flashtext + catalogued sequences + docling tables, no gemma4
#   step 9  refined    the same documents again, with gemma4 enabled
#
# MEASURED, and not what it first appears. On 24 documents with four workers:
#
#   gemma4 off, table pass ON    521.4s   0.05 docs/s   57 peptides
#   gemma4 off, table pass OFF    92.0s   0.26 docs/s    0 peptides
#
# The docling table pass is 82% of screening time (429 of 521s) AND found all 57 peptides, which
# matches `table_match` being the largest discovery path in the database (5,133 records). So gemma4
# is not the bottleneck and screening is not quick: ~0.05 docs/s is ~257 h (~10.7 days) for the
# 46,189-document corpus. Running both steps therefore costs MORE total wall clock than one
# combined pass would. What it buys is a complete, usable extraction after step 8 instead of only
# at the very end, and a step 9 that can be interrupted at any point leaving the corpus strictly
# better than step 8 and never worse.
LIMIT ?=
# Steps 8 and 9: how many documents are processed between Ollama runtime resets, and how long one
# document may take. Both are passed through to `sample-cli extract`; both have defaults in
# sample/pipeline_runner.py (100 documents, 300s) and are overridable here for a one-off run.
#
# Why chunking exists: refined extraction decays rather than failing. Measured on this corpus it
# held near 1 doc/s and fell to ~0.04 docs/s after ~2,000 documents, logging no error, and
# recovered completely when ollama was restarted by hand. Resetting the runtime every CHUNK_SIZE
# documents bounds how long that state can build up, at a cost of one model load per chunk.
#
# CHUNK_SIZE=0 disables chunking; DOC_TIMEOUT=0 disables the per-document cap.
CHUNK_SIZE ?=
DOC_TIMEOUT ?=
# Step 14 injection trials, spread evenly across the corruption types. Each type needs enough
# trials for its detection rate to mean something rather than be noise.
INJECTION_SAMPLES ?= 10000
# Step 15: how many documents land in the manual-curation review set, per corpus. Not equal and
# not a shared budget: the literature corpus can support a large sample, while only ~44 patents
# carry extracted peptides at all (26 of them referenced by a released peptide), so a 200-patent
# request is mostly negatives by necessity. Half of each corpus is documents the extractor found
# NO peptide in -- the only half that can show what it missed entirely.
CURATION_PAPERS ?= 800
CURATION_PATENTS ?= 200
# Step 16: the Gemini model that AUDITS that set, and an optional cap for a cheap trial run.
# NOTE there is no `gemini-3.0-pro`; the Gemini 3 Pro family exposes gemini-3.1-pro-preview
# (gemini-3-pro-image* are image models). Checked against the API's own model list.
CURATION_MODEL ?= gemini-3.1-pro-preview
CURATION_LIMIT ?=

# ==========================================================================================
# CONDA ENVIRONMENTS
# ==========================================================================================
# Each stage of the pipeline runs in the environment that carries its dependencies and no more.
# The groupings are the transitive import closures of the modules each step actually enters, not
# a tidy-looking guess -- see environments/README.md for the derivation and the constraints.
#
# Override any one of these to relocate a stage:
#     make step_47_train_antibacterial_ml_model ENV_ML=my-other-env
#
# Or collapse everything into one environment, which is what the full environments/sample.yml is
# for (and what the docs build requires):
#     make all SINGLE_ENV=sample
#
# NOTE: comments go ABOVE each assignment, never trailing on the same line. Make includes the
# whitespace between a value and a trailing `#` in the value itself, and SAMPLE_PREDICTOR_ENV is
# exported into the environment where sample/peptide_predictors.py reads it -- a padded value
# there produces `conda run -n "sample-predictors   "`, which does not resolve.

# Steps 1-6: downloads, scraping, PDF text, anticancer database import.
ENV_ACQUISITION ?= sample-acquisition
# Steps 7-19 (except 9) and 31-42: NLP, LLM, validation, merges, names, bibliography, families,
# family HMMs, modifications, source organisms, cross-references, Gene Ontology, GO .obo slice.
ENV_EXTRACTION ?= sample-extraction
# Steps 20-32 and 44: serialize, descriptors, external predictors, embeddings, structures, decoys,
# assembly, SMILES validation, the embedding-search index, SQLite load.
ENV_CHEM ?= sample-chem
# Steps 11, 43, 45-50: anomaly detection, FLAML-searched antimicrobial training (general +
# organism-specific), benchmarking, analysis, the paper's figures and tables.
ENV_ML ?= sample-ml
# Step 51: dashboard and Celery worker.
ENV_WEB ?= sample-web
# Steps 21-24 sub-call, entered via `conda run` from sample/peptide_predictors.py.
ENV_PREDICTORS ?= sample-predictors
# Step 27 sub-call, entered via `conda run` from sample/folding.py.
ENV_FOLD ?= sample-fold
# The docs build needs the union of every dependency; see environments/sample.yml.
ENV_DOCS ?= sample

# SINGLE_ENV collapses the split. Assigned with `override` so it wins over the ?= defaults above
# even when one of them is also set in the environment.
ifdef SINGLE_ENV
override ENV_ACQUISITION := $(SINGLE_ENV)
override ENV_EXTRACTION  := $(SINGLE_ENV)
override ENV_CHEM        := $(SINGLE_ENV)
override ENV_ML          := $(SINGLE_ENV)
override ENV_WEB         := $(SINGLE_ENV)
override ENV_DOCS        := $(SINGLE_ENV)
endif

# ------------------------------------------------------------------------------------------
# Fall back to ENV_FALLBACK for any environment that has not been created.
#
# The split is an optimization, not a prerequisite: a machine with only the full `sample`
# environment must still be able to run every step. Without this, splitting the Makefile broke
# working checkouts with conda's `EnvironmentLocationNotFound`, which names the missing
# environment but not the remedy.
#
# Resolved once at parse time so the cost is a single `conda env list`, not one per step.
# ------------------------------------------------------------------------------------------
ENV_FALLBACK ?= sample
_EXISTING_ENVS := $(notdir $(shell conda env list 2>/dev/null | grep -v '^\#' | awk 'NF {print $$1}'))
env_or_fallback = $(if $(filter $(1),$(_EXISTING_ENVS)),$(1),$(ENV_FALLBACK)$(warning \
    conda environment '$(1)' not found; using '$(ENV_FALLBACK)'. Run `make envs` to create it.))

# The names BEFORE fallback. The env-* targets must create and populate the environment they are
# named for, not the one a missing environment falls back to -- otherwise `make env-chem` builds
# sample-chem and then pip-installs the package into `sample`.
NAME_ACQUISITION := $(ENV_ACQUISITION)
NAME_EXTRACTION  := $(ENV_EXTRACTION)
NAME_CHEM        := $(ENV_CHEM)
NAME_ML          := $(ENV_ML)
NAME_WEB         := $(ENV_WEB)
NAME_DOCS        := $(ENV_DOCS)
NAME_FOLD        := $(ENV_FOLD)

ENV_ACQUISITION := $(call env_or_fallback,$(ENV_ACQUISITION))
ENV_EXTRACTION  := $(call env_or_fallback,$(ENV_EXTRACTION))
ENV_CHEM        := $(call env_or_fallback,$(ENV_CHEM))
ENV_ML          := $(call env_or_fallback,$(ENV_ML))
ENV_WEB         := $(call env_or_fallback,$(ENV_WEB))
ENV_DOCS        := $(call env_or_fallback,$(ENV_DOCS))

# sample/peptide_predictors.py reads this to locate the predictor environment it shells out to.
# NOT put through the fallback: these predictors exist only in their own environment, and
# silently running them from `sample` would load their model pickles under the wrong
# scikit-learn. peptide_predictors reports them unavailable instead, which is the honest answer.
export SAMPLE_PREDICTOR_ENV = $(ENV_PREDICTORS)

# Read by sample/folding.py to locate colabfold_batch. Not put through the fallback for the same
# reason as the predictors: ColabFold exists only in its own environment.
export SAMPLE_FOLD_ENV = $(ENV_FOLD)

# Solver used to CREATE environments. mamba resolves these dependency sets in a fraction of the
# time conda's solver takes -- which matters here because there are six of them and the extraction
# environment alone pulls torch, spacy and the transformers stack. Falls back to conda when mamba
# is not installed, so the Makefile works either way. Override with `make envs CONDA_CREATE=conda`.
CONDA_CREATE ?= $(if $(shell command -v mamba 2>/dev/null),mamba,conda)

# Environments are ENTERED with conda regardless: `conda run` is present in every installation,
# and it is the same command sample/peptide_predictors.py uses internally to reach the predictor
# environment. Creation speed is worth a second tool; execution consistency is not.
RUN = conda run --no-capture-output -n

# Every environment needs the package itself installed to expose the `sample-cli` entry point.
ALL_ENVS = $(ENV_ACQUISITION) $(ENV_EXTRACTION) $(ENV_CHEM) $(ENV_ML) $(ENV_WEB)

.PHONY: all dirs install clean test test-cov run-webserver run-worker \
	pipeline-status pipeline-runs pipeline-finish pipeline-selftest _selftest_step \
        \
        tool-update-families tool-admet tool-benchmark \
        tool-enrich-macrel tool-enrich-anticp2 tool-enrich-hemopi2 \
        tool-enrich-algpred2 \
        incremental-update \
        sources-status curation-model-check \
        envs env-acquisition env-extraction env-chem env-ml env-web env-predictors env-fold env-full \
        env-langextract \
        env-macrel env-amplify env-anticp2 env-ampidentifier env-ampeppy env-toxinpred3 env-toxinpred2 env-algpred2 \
        check-envs compose-up compose-down compose-logs docs docs-pdf docs-clean \
         \
        step_01_download_pubmed_papers_metadata step_02_download_scopus_papers_metadata \
        step_03_download_and_extract_papers step_04_download_google_patents_data \
        step_05_download_antimicrobial_peptide_metadata \
        step_06_import_anticancer_databases \
        step_07_merge_antimicrobial_peptide_data \
        step_08_screening_extract_peptide_data \
        step_09_refined_extract_peptide_data \
        step_10_revalidate_sequence_resolutions \
        step_11_polish_extraction_records \
        step_12_review_extracted_fields \
        step_13_train_rf_filter \
        step_14_enrich_extracted_peptides_with step_15_validate_peptide_data_extraction \
        step_16_validate_extractions_using_error step_17_select_curation_papers \
        step_18_review_curation_with_gemini \
        step_19_validate_extractions_using_manual \
        step_20_extract_repair_extractions step_21_merge_database_and_extracted \
        step_22_serialize_raw_datasets \
        step_23_enrich_base step_24_enrich_macrel \
        step_25_enrich_anticp2 step_26_enrich_hemopi2 step_27_enrich_algpred2 \
        step_28_enrich_frequency_embeddings step_29_enrich_structures \
        step_30_generate_decoy_datasets step_31_assemble_final_master_dataset \
        step_32_update_final_dataset step_33_validate_smiles step_34_build_embedding_index \
        step_35_build_peptide_names step_36_build_bibliography step_37_build_families \
        step_38_build_family_hmms step_39_link_bioactivity step_40_build_modifications \
        step_41_build_source_organisms step_42_build_cross_references step_43_build_gene_ontology \
        step_44_build_go_obo step_45_anomaly_detection step_46_load_sqlite_database \
        step_47_train_antibacterial_ml_model step_48_train_organism_specific_ml_models \
        step_49_benchmark step_50_exploratory_data_analysis \
        step_51_compare_chemical_spaces step_52_generate_paper_report \
        step_53_run_webserver step_54_generate_zenodo_dump

all: dirs step_01_download_pubmed_papers_metadata step_02_download_scopus_papers_metadata \
        step_03_download_and_extract_papers step_04_download_google_patents_data \
        step_05_download_antimicrobial_peptide_metadata \
        step_06_import_anticancer_databases \
        step_07_merge_antimicrobial_peptide_data \
        step_08_screening_extract_peptide_data \
        step_09_refined_extract_peptide_data \
        step_10_revalidate_sequence_resolutions \
        step_11_polish_extraction_records \
        step_12_review_extracted_fields \
        step_13_train_rf_filter \
        step_14_enrich_extracted_peptides_with step_15_validate_peptide_data_extraction \
        step_16_validate_extractions_using_error step_17_select_curation_papers \
        step_19_validate_extractions_using_manual \
        step_20_extract_repair_extractions step_21_merge_database_and_extracted \
        step_22_serialize_raw_datasets \
        step_23_enrich_base step_24_enrich_macrel \
        step_25_enrich_anticp2 step_26_enrich_hemopi2 step_27_enrich_algpred2 \
        step_28_enrich_frequency_embeddings step_29_enrich_structures \
        step_30_generate_decoy_datasets step_31_assemble_final_master_dataset \
        step_32_update_final_dataset step_33_validate_smiles step_34_build_embedding_index \
        step_35_build_peptide_names step_36_build_bibliography step_37_build_families \
        step_38_build_family_hmms step_39_link_bioactivity step_40_build_modifications \
        step_41_build_source_organisms step_42_build_cross_references step_43_build_gene_ontology \
        step_44_build_go_obo step_45_anomaly_detection step_46_load_sqlite_database \
        step_47_train_antibacterial_ml_model step_48_train_organism_specific_ml_models \
        step_49_benchmark step_50_exploratory_data_analysis \
        step_51_compare_chemical_spaces step_52_generate_paper_report \
        step_53_run_webserver step_54_generate_zenodo_dump
	@run_id=$$($(REGISTRY) resolve --from-scratch 0); \
	$(REGISTRY) finish --run-id "$$run_id" --status SUCCESS; \
	echo ""; \
	$(REGISTRY) show --run-id "$$run_id"

# Progress of the last run, or of a named one: make pipeline-status RUN_ID=20261021T160222Z-e4a277
pipeline-status:
	@$(REGISTRY) show $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(JSON),--json,)

# Recent runs, newest first, with how many steps each completed.
pipeline-runs:
	@$(REGISTRY) list --limit $(if $(LIMIT),$(LIMIT),20) $(if $(JSON),--json,)

# Exercises the resume machinery end to end -- the real run_step macro, twice -- against a
# throwaway registry, with `true` as the step command so no pipeline work runs. The first pass
# must report "Running", the second "already completed under run ...". Not a numbered step; a
# diagnostic, and the only way to check this without starting a multi-day stage.
pipeline-selftest:
	@tmp=$$(mktemp -d); \
	echo "--- pass 1: expect Running ---"; \
	SAMPLE_RUN_REGISTRY_DB=$$tmp/runs.sqlite $(MAKE) --no-print-directory _selftest_step; \
	echo "--- pass 2: expect 'already completed' ---"; \
	SAMPLE_RUN_REGISTRY_DB=$$tmp/runs.sqlite $(MAKE) --no-print-directory _selftest_step; \
	echo "--- pass 3: FROM_SCRATCH=1, expect Running again under a NEW run ---"; \
	SAMPLE_RUN_REGISTRY_DB=$$tmp/runs.sqlite $(MAKE) --no-print-directory _selftest_step FROM_SCRATCH=1; \
	echo "--- pass 4: a failing command must record FAILURE and exit nonzero ---"; \
	SAMPLE_RUN_REGISTRY_DB=$$tmp/runs.sqlite $(MAKE) --no-print-directory _selftest_step \
	        FROM_SCRATCH=1 SELFTEST_CMD=false && echo "UNEXPECTED: make returned 0" || true; \
	echo "--- pass 5: a missing artifact must record FAILURE ---"; \
	SAMPLE_RUN_REGISTRY_DB=$$tmp/runs.sqlite $(MAKE) --no-print-directory _selftest_step \
	        FROM_SCRATCH=1 SELFTEST_ARTIFACT=no/such/file && echo "UNEXPECTED: make returned 0" || true; \
	echo "--- runs recorded ---"; \
	SAMPLE_RUN_REGISTRY_DB=$$tmp/runs.sqlite $(REGISTRY) list; \
	rm -rf $$tmp

# SELFTEST_CMD/SELFTEST_ARTIFACT exist so pipeline-selftest can drive the failure branches too.
SELFTEST_CMD ?= true
SELFTEST_ARTIFACT ?= Makefile

_selftest_step:
	$(call run_step,1,Registry self-test,$(SELFTEST_CMD),-f,$(SELFTEST_ARTIFACT))

# Close the open run so the next `make` starts a new one instead of adopting this. `all` does
# this itself on success; call it by hand after abandoning a run you do not intend to resume.
pipeline-finish:
	@run_id=$$($(REGISTRY) resolve --from-scratch 0); \
	$(REGISTRY) finish --run-id "$$run_id" --status $(if $(STATUS),$(STATUS),SUCCESS); \
	echo "run $$run_id closed as $(if $(STATUS),$(STATUS),SUCCESS)"

dirs:
	@mkdir -p data/literature_data/pubmed data/literature_data/scopus \
	          data/literature_data_json/pubmed data/literature_data_json/scopus \
	          data/literature_data_condensed/pubmed data/literature_data_condensed/scopus \
	          data/literature_data_json_repaired \
	          data/literature_data_json_revised data/patent_data_json_revised/google_patents \
	          data/repaired_logs data/patent_data/google_patents \
	          data/patent_data_condensed/google_patents \
	          data/patent_data_json/google_patents data/patent_data_json_repaired/google_patents data/antimicrobial_databases \
	          data/sample_raw_release data/sample_decoy_databases \
	          data/sample_enriched_release data/sample_final_release \
	          sample/static/img/eda data/zenodo_dump

install:
	pip install -e .

# Creates every pipeline environment and installs the package into each, so `sample-cli` exists
# there. --no-deps on the install because the yml file already resolved the dependency set;
# letting pip re-resolve from pyproject.toml would pull the union back into every environment and
# undo the split.
envs: env-acquisition env-extraction env-chem env-ml env-web env-predictors env-fold
	@echo "=== All environments ready. Verifying cross-environment pins ==="
	@$(MAKE) check-envs

# Brings one environment to the state its file describes, whether or not it exists yet.
#
# `env create` refuses an existing prefix with `CondaValueError: prefix already exists`, which
# made `make envs` a one-shot command: re-running it after editing a yml -- or after any step
# failed partway through -- just errored. `env update --prune` converges an existing environment
# to the file instead, adding what is new and removing what the file no longer lists, so the two
# paths end in the same place and the target is safe to re-run.
#
# Args: 1=environment name, 2=yml file
define sync_env
	@set -e; \
	if conda env list | awk 'NF && $$1 !~ /^#/ {print $$1}' | grep -qx "$(1)"; then \
	    echo "=== Updating existing environment $(1) ==="; \
	    $(CONDA_CREATE) env update --prune -n "$(1)" -f "$(2)"; \
	else \
	    echo "=== Creating environment $(1) ==="; \
	    $(CONDA_CREATE) env create -f "$(2)"; \
	fi
endef

env-acquisition:
	$(call sync_env,$(NAME_ACQUISITION),environments/sample-acquisition.yml)
	conda run -n $(NAME_ACQUISITION) pip install --no-deps -e .

env-extraction:
	$(call sync_env,$(NAME_EXTRACTION),environments/sample-extraction.yml)
	conda run -n $(NAME_EXTRACTION) pip install --no-deps -e .
	conda run -n $(NAME_EXTRACTION) python -m sample.install_models

env-chem:
	$(call sync_env,$(NAME_CHEM),environments/sample-chem.yml)
	conda run -n $(NAME_CHEM) pip install --no-deps -e .

env-ml:
	$(call sync_env,$(NAME_ML),environments/sample-ml.yml)
	conda run -n $(NAME_ML) pip install --no-deps -e .

env-web:
	$(call sync_env,$(NAME_WEB),environments/sample-web.yml)
	conda run -n $(NAME_WEB) pip install --no-deps -e .

# The predictors environment is standalone: it runs vendor CLIs, never sample-cli, so the package
# is deliberately not installed into it.
env-predictors:
	$(call sync_env,$(ENV_PREDICTORS),environments/sample-predictors.yml)

# LangExtract driven against the LOCAL Ollama server -- the default structured-extraction backend,
# with the docling table backend as the fallback. A tool environment holding only langextract: it
# pulls its own pydantic/httpx tree, which would move sample-extraction's numpy<2 pin (spacy needs
# numpy 1.x), so sample/langextract_records.py shells into this one rather than importing it.
#
# No API key is involved. The backend always passes a local model_url and an Ollama model tag; with
# this environment absent, extraction falls back to docling automatically and logs that it did.
# The Ollama server must already hold the model: `ollama pull $(DEFAULT_EXTRACT_MODEL)`.
#
# NOT built with `conda env create`. conda-forge's repodata.json is ~100MB and this host's link is
# slow enough that the index download times out before resolution even begins -- measured, twice.
# langextract needs only python 3.11 plus one pip package, so the environment is a venv at the same
# path the code resolves (.../envs/sample-langextract/bin/python) and conda is never involved.
#
# google-genai is installed FIRST, on purpose. Installing langextract alone makes pip resolve that
# dependency during backtracking, and a transient read timeout there makes pip report
# "Could not find a version that satisfies the requirement google-genai (from versions: none)" --
# which reads like the package does not exist, when it is simply a network failure mid-resolution.
# Installing it up front turns a confusing resolver error into an ordinary retryable download.
env-langextract:
	@set -e; \
	ENVDIR="$(HOME)/miniconda3/envs/sample-langextract"; \
	if [ ! -x "$$ENVDIR/bin/python" ]; then \
	    echo "=== Creating venv $$ENVDIR ==="; \
	    python -m venv "$$ENVDIR"; \
	fi; \
	echo "=== Installing langextract (google-genai first; slow link tolerated) ==="; \
	"$$ENVDIR/bin/pip" install --timeout 300 --retries 10 "google-genai>=1.39.0"; \
	"$$ENVDIR/bin/pip" install --timeout 300 --retries 10 "langextract==1.7.0"; \
	"$$ENVDIR/bin/python" -c "import langextract; print('langextract OK')"

# ColabFold. A tool environment holding only colabfold_batch: its JAX/AlphaFold2 stack cannot be
# resolved against this project's dependencies, so sample/folding.py shells into it rather than
# importing it, exactly as peptide_predictors reaches sample-predictors.
env-fold:
	$(call sync_env,$(NAME_FOLD),environments/sample-fold.yml)

# Macrel for the annotator, in its own environment (it pins its own scikit-learn/numpy and cannot
# share sample-predictors without a pin fight). sample/peptide_predictors.py shells out to it.
env-macrel:
	$(call sync_env,sample-macrel,environments/sample-macrel.yml)

# Per-tool annotator predictor environments. Each holds one tool whose scikit-learn/numpy (or
# TensorFlow) pin would fight the others in a shared env; sample/peptide_predictors.py locates each
# by environment name and shells out. Built on demand, not by `make envs`.
env-amplify:
	$(call sync_env,sample-amplify,environments/sample-amplify.yml)

env-anticp2:
	$(call sync_env,sample-anticp2,environments/sample-anticp2.yml)

# AMPidentifier -- benchmark-only fourth antibacterial comparator (sample/benchmark.py), alongside
# Macrel and AMPlify. Not part of the live pipeline (see AMPidentifierPredictor's own docstring),
# so this is built on demand like env-amplify, not by `make envs`.
env-ampidentifier:
	$(call sync_env,sample-ampidentifier,environments/sample-ampidentifier.yml)

# amPEPpy -- benchmark-only fifth antibacterial comparator (sample/benchmark.py), alongside Macrel,
# AMPlify and AMPidentifier. Not part of the live pipeline (see AmPEPpyPredictor's own docstring).
# Unlike a plain sync_env, amPEPpy is not on PyPI/bioconda (the yml's pip section installs it
# straight from its git repo) and its own pretrained model is not part of that pip install at all
# (setup.py ships no package_data) -- so this also clones the tool's repo into data/ampeppy/ purely
# to reach pretrained_models/amPEP.model, the same convention uses for its own weights.
env-ampeppy:
	$(call sync_env,sample-ampeppy,environments/sample-ampeppy.yml)
	@if [ ! -f data/ampeppy/pretrained_models/amPEP.model ]; then \
	    echo "=== sample-ampeppy: fetching amPEPpy's own repo for its bundled pretrained model ==="; \
	    git clone --depth 1 https://github.com/tlawrence3/amPEPpy.git data/ampeppy; \
	else \
	    echo "data/ampeppy already present, skipping clone"; \
	fi

env-toxinpred3:
	$(call sync_env,sample-toxinpred3,environments/sample-toxinpred3.yml)

# ToxinPred2, benchmark-only (sample.benchmark's "hemolytic" reference comparator, not part of the
# live annotator -- see sample/peptide_predictors.py's ToxinPred2Predictor docstring). Unlike
# toxinpred3, its shipped model is an ONNX export with no scikit-learn pickle to fight, but it
# still gets its own environment matching every other per-tool predictor env.
env-toxinpred2:
	$(call sync_env,sample-toxinpred2,environments/sample-toxinpred2.yml)

env-algpred2:
	$(call sync_env,sample-algpred2,environments/sample-algpred2.yml)

# The full environment. Needed for `make docs` (Sphinx autodoc imports every module) and for
# `make all SINGLE_ENV=sample`.
env-full:
	$(call sync_env,$(NAME_DOCS),environments/sample.yml)
	conda run -n $(NAME_DOCS) pip install --no-deps -e .

# Asserts the constraint the split cannot express on its own: pickles cross environment
# boundaries, and a pickled estimator carries no version negotiation.
#
#   step 11 trains rf_filter_model.pkl in ENV_ML  -> step 8 unpickles it in ENV_EXTRACTION
#   steps 45-46 train the models in ENV_ML        -> the webserver unpickles them in ENV_WEB
#
# A mismatch does not necessarily fail loudly at load time: scikit-learn may warn and hand back a
# subtly different estimator, so the symptom is wrong predictions rather than an error. Checked
# here so it surfaces at setup time instead.
check-envs:
	@set -e; \
	fail=0; \
	ref="$(ENV_ML)"; \
	for env in $(ENV_EXTRACTION) $(ENV_WEB); do \
	    for pkg in scikit-learn numpy; do \
	        a=$$(conda run -n $$ref python -c "import importlib.metadata as m; print(m.version('$$pkg'))" 2>/dev/null || echo MISSING); \
	        b=$$(conda run -n $$env python -c "import importlib.metadata as m; print(m.version('$$pkg'))" 2>/dev/null || echo MISSING); \
	        if [ "$$a" = "$$b" ] && [ "$$a" != "MISSING" ]; then \
	            printf '  OK    %-14s %-20s %s\n' "$$pkg" "$$env" "$$a"; \
	        else \
	            printf '  FAIL  %-14s %s(%s) != %s(%s)\n' "$$pkg" "$$ref" "$$a" "$$env" "$$b"; \
	            fail=1; \
	        fi; \
	    done; \
	done; \
	xm=$$(conda run -n $(ENV_ML) python -c "import importlib.metadata as m; print(m.version('xgboost'))" 2>/dev/null || echo MISSING); \
	xw=$$(conda run -n $(ENV_WEB) python -c "import importlib.metadata as m; print(m.version('xgboost'))" 2>/dev/null || echo MISSING); \
	if [ "$$xm" = "$$xw" ] && [ "$$xm" != "MISSING" ]; then \
	    printf '  OK    %-14s %-20s %s\n' "xgboost" "$(ENV_WEB)" "$$xm"; \
	else \
	    printf '  FAIL  %-14s %s(%s) != %s(%s)\n' "xgboost" "$(ENV_ML)" "$$xm" "$(ENV_WEB)" "$$xw"; \
	    fail=1; \
	fi; \
	for pkg in torch transformers; do \
	    tm=$$(conda run -n $(ENV_ML) python -c "import importlib.metadata as m; print(m.version('$$pkg'))" 2>/dev/null || echo MISSING); \
	    tw=$$(conda run -n $(ENV_WEB) python -c "import importlib.metadata as m; print(m.version('$$pkg'))" 2>/dev/null || echo MISSING); \
	    if [ "$$tm" = "$$tw" ] && [ "$$tm" != "MISSING" ]; then \
	        printf '  OK    %-14s %-20s %s\n' "$$pkg" "$(ENV_WEB)" "$$tm"; \
	    else \
	        printf '  FAIL  %-14s %s(%s) != %s(%s)\n' "$$pkg" "$(ENV_ML)" "$$tm" "$(ENV_WEB)" "$$tw"; \
	        fail=1; \
	    fi; \
	done; \
	if [ $$fail -ne 0 ]; then \
	    echo "Pickles written in $(ENV_ML) may not load correctly elsewhere. Reconcile the pins"; \
	    echo "in environments/*.yml, then re-run: conda env update --prune -f environments/<file>"; \
	    exit 1; \
	fi

# The test suite runs in the full environment: it imports every module, so it needs the union of
# dependencies for the same reason the docs build does.
test:
	$(RUN) $(ENV_DOCS) python -m pytest tests/

test-cov:
	$(RUN) $(ENV_DOCS) python -m pytest tests/ --cov=sample --cov-report=term-missing \
	    --cov-report=html:htmlcov
	@echo "HTML coverage report: htmlcov/index.html"

run-webserver:
	@echo "Starting sAMPle ML prediction worker and webserver..."
	@$(RUN) $(ENV_WEB) python -m sample.worker & WORKER_PID=$$!; \
	trap "kill $$WORKER_PID 2>/dev/null || true" EXIT INT TERM; \
	$(RUN) $(ENV_WEB) sample-cli webserver

serialize-records:
	$(RUN) $(ENV_WEB) python sample/serialize_archive.py

# --- On-demand maintenance / analysis tools (not part of the linear 1-41 build) ---------------
# Each is a thin wrapper over a CLI subcommand; see `sample-cli <cmd> --help` for options.

# Incrementally place NEW peptides into the existing family map without renumbering existing
# families. NEW=<fasta|csv of new sequences>, e.g. `make tool-update-families NEW=new.fasta`.
tool-update-families:
	$(RUN) $(ENV_WEB) sample-cli update-families --new "$(NEW)"

# Incremental acquisition: fetch new papers and refresh external databases, extract/merge them,
# then place whatever is genuinely new into the existing family map without a full rebuild. Chains
# the SAME steps the linear 1-51 build already uses (1-9, 11, 16-18) -- each already
# resumable/idempotent on its own -- then runs the one piece of logic this needs that they don't
# already provide: diffing step 19's consensus CSV against the live release database
# (`sample-cli incremental-update`) and handing exactly the new rows to `sample.incremental`.
# Steps 11 (RF filter training), 13 and 14 (LLM/error-injection validation *reports*) are
# independent quality checks step 17 does not read from, so they are not in this chain -- run them
# separately (`make step_13_...` etc.) if you want them refreshed too.
#
# This fetches papers and hits external databases: network-bound, like every acquisition step, so
# it is never started implicitly -- run it yourself. DRY_RUN=1 reports the new-peptide placement
# without writing families.json. Loading the new rows into the live database, descriptors,
# bibliography etc. (steps 19-34, 42) is a separate, deliberate follow-on step, not chained here.
incremental-update: step_01_download_pubmed_papers_metadata step_02_download_scopus_papers_metadata \
    step_03_download_and_extract_papers step_04_download_google_patents_data \
    step_05_download_antimicrobial_peptide_metadata step_06_import_anticancer_databases \
    step_07_merge_antimicrobial_peptide_data step_08_screening_extract_peptide_data \
    step_09_refined_extract_peptide_data \
    step_14_enrich_extracted_peptides_with step_19_validate_extractions_using_manual \
    step_20_extract_repair_extractions step_21_merge_database_and_extracted
	$(RUN) $(ENV_EXTRACTION) sample-cli incremental-update $(if $(DRY_RUN),--dry-run,)

# ADMET-lite summary for one peptide (sequence-derived physicochemistry + rule flags). SEQ=<seq>.
tool-admet:
	$(RUN) $(ENV_WEB) sample-cli admet "$(SEQ)"

# Benchmark a trained sAMPle model against a reference tool on the same holdout. See the command's
# --help for the model/tool/holdout arguments.
tool-benchmark:
	$(RUN) $(ENV_ML) sample-cli benchmark $(ARGS)

# Re-run ONE external predictor over the already-enriched dataset (step 21's output), filling in
# only peptides missing a result for that tool -- not descriptors, not any other predictor, not a
# full steps-21-27 enrichment re-run. DATASET defaults to step 21's own output
# (data/sample_enriched_release/enriched.csv) if unset; pass ALL=1 to force a full recompute
# instead of just the gap.
#
# AMPlify has no entry here -- retired 2026-08-26 after a sustained upstream crash rate in
# production; see the comment on sample.peptide_predictors.EXTERNAL_PREDICTORS.
tool-enrich-macrel:
	$(RUN) $(ENV_CHEM) sample-cli enrich-macrel $(if $(DATASET),--dataset "$(DATASET)",) $(if $(ALL),--all,)

tool-enrich-anticp2:
	$(RUN) $(ENV_CHEM) sample-cli enrich-anticp2 $(if $(DATASET),--dataset "$(DATASET)",) $(if $(ALL),--all,)

tool-enrich-hemopi2:
	$(RUN) $(ENV_CHEM) sample-cli enrich-hemopi2 $(if $(DATASET),--dataset "$(DATASET)",) $(if $(ALL),--all,)

tool-enrich-algpred2:
	$(RUN) $(ENV_CHEM) sample-cli enrich-algpred2 $(if $(DATASET),--dataset "$(DATASET)",) $(if $(ALL),--all,)

# The Celery worker that runs the webserver's background prediction jobs. Needs a reachable
# Redis broker (SAMPLE_BROKER_URL, default redis://localhost:6379/0). Without it the webserver
# still serves every other page and rejects prediction submissions with a 503.
run-worker:
	$(RUN) $(ENV_WEB) celery -A sample.tasks.celery_app worker \
	    --loglevel=INFO --queues=sample-predictions --concurrency=2 --max-tasks-per-child=25

# Full webserver stack (redis + gunicorn + worker). Add `--profile monitoring` for Flower.
compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f web worker

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info build dist

# Macro to run a step conditionally and check its artifact
# Args: 1=step_number, 2=name, 3=command, 4=check_flag (-f or -d), 5=artifact_path
define run_step
	@set -e; \
	: "Step numbers are PLAIN INTEGERS. Lettered steps (19b) are not allowed: renumber instead." ; \
	: "This used to strip non-digits before the range test, which silently ACCEPTED a lettered" ; \
	: "step and ran it whenever its base number was in range. That is how 13b and 05b came to" ; \
	: "exist, and a lettered step cannot be addressed by START_STEP/END_STEP on its own -- it" ; \
	: "shares its predecessor's number, so no range selects it alone. Rejected loudly instead," ; \
	: "at the moment the step runs, rather than tolerated." ; \
	step_num='$(1)'; \
	case "$$step_num" in \
		''|*[!0-9]*) \
			echo "Error: step id '$(1)' is not a plain integer."; \
			echo "       Lettered steps are not allowed -- renumber the pipeline instead."; \
			exit 1 ;; \
	esac; \
	if [ "$$step_num" -ge $(START_STEP) ] && [ "$$step_num" -le $(END_STEP) ]; then \
		: "One run id for the whole invocation. resolve is idempotent: the first step of a make" ; \
		: "creates the run, every later step finds it RUNNING and gets the same id back. A" ; \
		: "re-invocation adopts the run that was still open, which is what makes the skip below" ; \
		: "mean 'already done in THIS run' rather than 'done at some point, by something'." ; \
		run_id=$$($(REGISTRY) resolve --from-scratch $(FROM_SCRATCH) --step $$step_num \
		                              --start-step $(START_STEP) --end-step $(END_STEP)); \
		if [ -z "$$run_id" ]; then \
			echo "Error: could not resolve a run id from the registry."; \
			exit 1; \
		fi; \
		if $(REGISTRY) should-skip --run-id "$$run_id" --step $$step_num \
		                           --from-scratch $(FROM_SCRATCH); then \
			echo "=== Step $(1) already completed under run $$run_id -- resuming past it ==="; \
			echo "    (FROM_SCRATCH=1 re-runs it; make pipeline-status shows the table)"; \
		else \
			echo "=== Running Step $(1): $(2)  [run $$run_id] ==="; \
			$(REGISTRY) begin-step --run-id "$$run_id" --step $$step_num \
			                       --name "$(2)" --artifact "$(5)"; \
			: "rc is captured rather than let `set -e` abort, so the failure is RECORDED before" ; \
			: "the recipe dies. Without this a failed step leaves a RUNNING row that the next" ; \
			: "invocation can only report as INTERRUPTED, losing the exit code." ; \
			rc=0; \
			$(3) || rc=$$?; \
			if [ $$rc -ne 0 ]; then \
				$(REGISTRY) end-step --run-id "$$run_id" --step $$step_num \
				                     --status FAILURE --exit-code $$rc; \
				echo "Error: Step $(1) failed with exit $$rc (recorded under run $$run_id)."; \
				exit $$rc; \
			fi; \
			if [ ! $(4) $(5) ]; then \
				$(REGISTRY) end-step --run-id "$$run_id" --step $$step_num \
				                     --status FAILURE --exit-code 0; \
				echo "Error: Step $(1) artifact $(5) not found!"; \
				exit 1; \
			fi; \
			$(REGISTRY) end-step --run-id "$$run_id" --step $$step_num \
			                     --status SUCCESS --exit-code 0; \
		fi; \
	else \
		echo "Skipping Step $(1) (START_STEP=$(START_STEP), END_STEP=$(END_STEP))"; \
	fi
endef

step_01_download_pubmed_papers_metadata:
	$(call run_step,1,Download PubMed Papers Metadata,$(RUN) $(ENV_ACQUISITION) sample-cli pubmed,-f,data/literature_data/pubmed_metadata.csv)

step_02_download_scopus_papers_metadata:
	$(call run_step,2,Download Scopus Papers Metadata,$(RUN) $(ENV_ACQUISITION) sample-cli scopus,-f,data/literature_data/scopus_metadata.csv)

step_03_download_and_extract_papers:
	$(call run_step,3,Download & Extract Papers Data,$(RUN) $(ENV_ACQUISITION) sample-cli download-pdfs --source pubmed && $(RUN) $(ENV_ACQUISITION) sample-cli download-pdfs --source scopus,-d,data/literature_data/pubmed)

step_04_download_google_patents_data:
	$(call run_step,4,Download Google Patents Data,$(RUN) $(ENV_ACQUISITION) sample-cli google-patents,-d,data/patent_data/google_patents)

step_05_download_antimicrobial_peptide_metadata:
	$(call run_step,5,Download Antimicrobial Peptide Metadata,$(RUN) $(ENV_ACQUISITION) sample-cli databases --targets adp dbaasp dramp dbamp camp --workers 4 $(if $(filter true,$(REFRESH_SOURCES)),--refresh,) && $(RUN) $(ENV_ACQUISITION) sample-cli databases --download-papers --targets adp dbaasp dramp dbamp camp --workers 4,-d,data/antimicrobial_databases)

# Which release of each source database is on disk. A diagnostic, not a numbered step: it
# downloads nothing, so it can be run safely while anything else is running.
#
# DRAMP is the one that needs asking. It has no version in its download path and updates the
# files in place, so the URL cannot tell you what you have -- this re-derives the release from
# the data (row counts against the published tables, plus the SMILES column DRAMP 5.0 added).
sources-status:
	$(RUN) $(ENV_ACQUISITION) sample-cli databases --dramp-status

# Step 6: import anticancer peptide datasets (AntiCP2 today) BEFORE the merge, so their positives
# flow into both the consensus (step 7 -> features) and the activity table (step 19 -> anticancer
# labels) through the normal pipeline. Runs in the step-5 window; harmless if the hosts are down
# (build_anticancer_table just writes an empty table and the merge sees no AntiCP2 rows).
step_06_import_anticancer_databases:
	$(call run_step,6,Import Anticancer Peptide Databases,$(RUN) $(ENV_ACQUISITION) sample-cli import-anticancer --download,-f,data/anticancer_databases/anticancer_peptides.csv)

step_07_merge_antimicrobial_peptide_data:
	$(call run_step,7,Merge Antimicrobial Peptide Data,$(RUN) $(ENV_EXTRACTION) sample-cli merge,-f,data/sample_raw_release/merged_from_database.csv)

# Step 8 SCREENING: everything except the gemma4/LangExtract backend -- NLP, flashtext, the
# catalogued-sequence automaton and the docling table grids. This is the pass that produces most of
# the corpus's peptides: measured on 24 documents, the table pass alone accounted for all 57 found,
# and `table_match` is the largest discovery path in the database (5,133 records).
#
# It is NOT the cheap half. Measured on those same 24 documents with four workers, the table pass
# is 82% of this step's time (429 of 521s), so screening runs at ~0.05 docs/s -- about 10.7 days
# for the 46,189-document corpus. Disabling the table pass gives 0.26 docs/s and finds nothing.
step_08_screening_extract_peptide_data:
ifdef PDF_PATH
	$(call run_step,8,Screening extraction from a single PDF,$(RUN) $(ENV_EXTRACTION) sample-cli extract-from-pdf $(PDF_PATH) --output $(if $(OUTPUT),$(OUTPUT),extracted_data.json),-f,$(if $(OUTPUT),$(OUTPUT),extracted_data.json))
else
	$(call run_step,8,Screening extraction (NLP + flashtext + tables; no gemma4),LANGEXTRACT_BACKEND=0 $(RUN) $(ENV_EXTRACTION) sample-cli extract --sources $(SOURCES) $(RESUME_EXISTS) $(if $(WORKERS),--workers $(WORKERS),) $(if $(LIMIT),--limit $(LIMIT),) $(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE),) $(if $(DOC_TIMEOUT),--doc-timeout $(DOC_TIMEOUT),),-d,data/literature_data_json/pubmed)
endif

# Step 9 REFINED: re-runs the same documents with gemma4 enabled, adding the records only a model
# finds -- prose-stated measurements and the table rows whose structure the grid parser flattens.
#
# It resumes with --resume-mode refined, NOT --resume. The difference is the whole reason the mode
# exists: `--resume` skips any document whose output already exists, and step 8 has just written
# one for every document, so resuming here on file existence skips the entire corpus and adds no
# gemma4 records at all. `refined` instead reads metadata.langextract_backend, which is "enabled"
# only once the model pass has run for that document and "disabled" after a screening-only pass.
# A document also counts as done when gate 1 marked it `skipped`: gate 1 is a pure function of
# the text, so the refined pass cannot do better on a captcha page or a blank scan. Without that
# clause those 509 documents never acquired a marker and every resume re-queued them -- and they
# sit early in the iteration order (139 of the first 400 queued), so a restart spent minutes
# re-skipping junk while reporting "0 peptides extracted so far". Measured on the real corpus,
# the clause takes the queue from 10,472 to 9,952 and makes the first 200 queued documents all
# real work. A document whose .txt is NEWER than its output is re-queued regardless, so
# re-acquiring a captcha page as the real paper is not silently wasted.
#
# This replaces the old advice of simply re-running the step and letting the payload cache return
# the model calls already made for free. That is true of the Ollama calls but not of the rest of a
# document. Measured on a live re-walk 2026-10-21: an already-refined document costs ~3s to
# re-process (498 outputs rewritten in 25 minutes, 1,195 docs/h) against ~38s for one the model
# has not seen (94 docs/h). The cache does most of the work, but re-walking 11,578 finished
# documents to reach the remainder is still ~9.7 hours that produces nothing.
#
# The gate in sample/langextract_records.py decides which documents reach the model
# (LANGEXTRACT_GATE, default "unit": 8.5% fewer calls for 2.4% of gemma4-unique peptides). Every
# record states which pass produced it via the `langextract_backend` metadata key, so a
# half-finished refined pass stays auditable.
# LANGEXTRACT_MODEL is set here rather than taken from DEFAULT_EXTRACT_MODEL. The refined pass is
# chunk-heavy -- tens of calls per document -- so it is the one stage where model size dominates
# wall clock, and gemma4 is an 8B VISION+AUDIO model whose image and audio weights this text-only
# path never touches. Measured on this host: gemma4 held 9.8GB resident for 4.2GB of GPU work,
# which pushed 11GiB into swap on a 15GB machine. gemma3:4b is the same model family (so the
# worker's tuned prompt and chat template still apply), text-first, and ~3.3GB.
#
# gemma3:4b and NOT qwen3:4b, which is smaller still: sample/llm.py lists both `gemma4` and
# `qwen3` as reasoning families that "must be disabled or it consumes the whole token budget and
# the completion comes back empty", and the LangExtract worker never passes `think` -- it cannot,
# it runs in an environment that cannot import sample.*. gemma3 emits no thinking channel.
#
# Override with LANGEXTRACT_MODEL=<tag> to go back to gemma4 or try another local tag.
# REVERTED to gemma4 on 2026-10-13 after measurement. gemma3:4b was set here on the reasoning that
# it is the same model family, so the worker's tuned prompt and chat template would still apply.
# That reasoning was WRONG and the model does not respect the extraction contract. Measured over
# the 40 documents it had rewritten, against 300 documents from the gemma4 era:
#
#   model        peptides/document   from the LLM pass   without a sequence
#   gemma4                   3.33                   --                  --
#   gemma3:4b               11.00                  99%                 65%
#
# The extra records are not recall. On an HPLC-derivatization paper it returned
# `o-phthalaldehyde`, `fluorescamine`, `4-halo-7-nitro-benzo-2,1,3-oxadiazole` and
# `20 essential amino acids` as peptides; elsewhere `magainin analogues A, B, C, F, G, and H`
# (a phrase), `magainin peptides` (a category) and `Magainin10442437` (a name fused to a PMID).
# It was also SLOWER per call despite being less than half the size -- median generation went from
# 11 tokens to 308, and mean generation time from 1.9s to 12.2s, because it answers at length
# where gemma4 answers "nothing here" and stops.
#
# Size was never the binding constraint; OLLAMA_NUM_PARALLEL was -- see environments/README.md.
#
# gemma3:4b, set 2026-10-14 after measuring three candidates against gemma4 on real output.
# Records per document, share carrying a sequence, and share of names a junk classifier rejects:
#
#   model            rec/doc   with sequence   junk names   fits at -np 4
#   gemma3:4b           8.21          24.4%        13.7%   yes (3.8GB on card)
#   gemma4:latest       6.22          27.4%        15.2%   yes (3.4GB; 9.6GB file, vision+audio)
#   qwen3:14b           0.81           5.9%        17.7%   NO (13%/87% CPU/GPU split)
#
# gemma3:4b wins on recall and on junk rate, and it is 3.3GB rather than 9.6GB -- which matters off
# the card as much as on it: Ollama does not mmap under CUDA, so gemma4 held 8.5GB of ANONYMOUS
# host RSS and drove this 15GB machine into 8.8GiB of swap, stalling the extraction workers until
# the GPU sat 58% idle.
#
# MUST EQUAL sample/config.py's DEFAULT_EXTRACT_MODEL. When the two disagree, Ollama evicts and
# reloads one model for the other continuously -- measured at 31 llama-server restarts and 23
# evictions in 20 minutes, with documents failing on a different symptom each run. That fault has
# been introduced twice; keep them equal.
#
# Set to gemma4:latest on 2026-10-16 by operator decision. The trade-offs are recorded in full in
# sample/config.py beside DEFAULT_EXTRACT_MODEL: gemma4 is the model the worker prompt was tuned
# for and the only one with a measured triage score (12/12), but it yields FEWER records per
# document than gemma3:4b (6.22 vs 8.21) with MORE junk (15.2% vs 13.7%), costs ~8.5GB of
# anonymous host RSS on a 15GB machine, and invalidates the payload cache.
#
# An earlier note here condemned gemma3:4b for returning reagent names (`o-phthalaldehyde`,
# `fluorescamine`) as peptides. That was real but unrepresentative -- a single
# HPLC-derivatization paper -- and the rate comparison offered alongside it was confounded,
# measuring gemma4's refined pass against a sample that was mostly step-8 screening output.
#
# A stale claim also lived here: that gemma4 could not be used through LangExtract because the
# worker cannot pass `think`. It can -- langextract 1.7.0's Ollama provider does
# `combined_kwargs.setdefault('think', False)` itself, so a reasoning family is handled without
# the worker doing anything. Verified 2026-10-16 in the installed package.
#
# Verify with `make curation-model-check MODEL=<tag>` before changing this; two swaps made on
# reasoning alone cost ~680 documents of degraded output on 2026-10-13/14.
LANGEXTRACT_MODEL ?= gemma4:latest

# Validate a candidate LangExtract model against what is already on disk, on the same documents,
# BEFORE letting it near a multi-day run. Not a numbered step -- a diagnostic.
#
#   make curation-model-check MODEL=qwen3:14b [DOCS=30]
#
# It makes real model calls (the cache is disabled for the candidate), so it competes with any
# running extraction. The gemma3:4b incident is what it exists to prevent: a model swapped in on
# reasoning alone wrote ~456 documents of fabricated records before anyone looked at the output.
DOCS ?= 30
curation-model-check:
	@test -n "$(MODEL)" || { echo "Usage: make curation-model-check MODEL=<ollama-tag> [DOCS=30]"; exit 1; }
	$(RUN) $(ENV_EXTRACTION) python -m sample.model_check --model $(MODEL) --limit $(DOCS) \
		--json data/model_check_$(subst :,_,$(MODEL)).json

step_09_refined_extract_peptide_data:
	$(call run_step,9,Refined extraction (adds a local LLM pass; interruptible),LANGEXTRACT_MODEL=$(LANGEXTRACT_MODEL) $(RUN) $(ENV_EXTRACTION) sample-cli extract --sources $(SOURCES) $(RESUME_REFINED) $(if $(WORKERS),--workers $(WORKERS),) $(if $(LIMIT),--limit $(LIMIT),) $(if $(CHUNK_SIZE),--chunk-size $(CHUNK_SIZE),) $(if $(DOC_TIMEOUT),--doc-timeout $(DOC_TIMEOUT),),-d,data/literature_data_json/pubmed)

# Step 10: re-check every sequence that synonym resolution APPLIED during steps 8 and 9, and
# withdraw the ones that do not qualify under the current rules.
#
# This exists because the rules tightened after records had been written, and the first version of
# the resolver applied catalogue sequences without validating them. Measured over 1,172 applied
# resolutions in a real run, it had written: 'Asp' -> 'D' (one amino acid), 'C2' -> ACETYLATED (an
# English word sitting in a sequence column), 30 sequences of 1-4 residues, and 67 keyed on a bare
# table row label ('1b', '4a', 'C2') that means nothing outside its own paper. Those records are on
# disk and have to be withdrawn rather than left to reach the database.
#
# Placed AFTER step 9, not before it: step 9 writes resolutions of its own, so a repair that ran
# first would leave them unchecked.
#
# Idempotent and non-destructive. A record whose resolutions all still hold is not rewritten, so
# this can be re-run or interrupted freely; and a withdrawal clears the SEQUENCE while keeping the
# sequence_resolution entry annotated with withdrawn_reason, so what was removed and why stays
# legible. Run it with --dry-run first to see the counts without writing.
step_10_revalidate_sequence_resolutions:
	$(call run_step,10,Revalidate Applied Sequence Resolutions,$(RUN) $(ENV_EXTRACTION) sample-cli revalidate-resolutions $(if $(DRY_RUN),--dry-run,) $(if $(LIMIT),--limit $(LIMIT),),-d,data/literature_data_json/pubmed)

# Step 11: re-apply the CURRENT activity-field rules to records step 9 already wrote.
#
# Step 9 writes records under whatever rules were in force at the time, and those rules keep
# tightening. A document is only re-extracted when the model pass runs again, which on this corpus
# is a multi-day job -- so without this step a rule fixed today reaches the corpus at the speed of
# re-extraction. Measured on the output on disk when this step was added:
#
#     6,173 of 53,107 activities (11.6%)  a concentration type in a mass/time/energy/count unit
#                                         (`brevinin-1RTb MIC = 2548.2 Da` -- a molecular weight
#                                         read out of a mass-spec table into the MIC column)
#     1,098 of 52,713 units      (2.1%)   U+FFFD where PDF extraction lost a character
#     2,312 value_raw                     a range whose upper bound was parsed and then discarded
#        58 activity_type                 prose describing an assay instead of naming one
#
# Idempotent, and verified so on real files: a second pass over its own output changes nothing and
# leaves the files byte-identical. That is why it sits in `all` rather than being a one-off
# backfill -- it is cheap (no model, no PDF, no network) and it guarantees that whatever is on
# disk conforms to the current rules before step 13 is the first stage to read it.
#
# It does NOT re-run preprocessing.standardize_and_filter_activities, which is not idempotent on
# its own output: that function assigns value_raw_unit from the unit it is currently looking at,
# so a second pass over an already-converted record overwrites the document-printed unit (mM) with
# the canonical one (uM) and loses the fact that value_raw was measured in millimolar -- a 1000x
# misreading for anything pairing those two fields. Each repair in sample/polish.py is instead
# individually guarded to be a no-op once applied.
#
# Sequence-level correction is deliberately not here: that comes from step 9 re-extracting the
# document and from step 10's resolution revalidation. Run with DRY_RUN=1 first to see the counts.
# POLISH_VERSION is 2: it now also restores a unit prefix lost in extraction, but ONLY where the
# record's own JSON proves which one -- the activity's context_sentence printing that value in a
# different unit, or a unit a peptide potency cannot be stated in (g/mL, zero occurrences in
# 182,766 curated measurements). Merely improbable units such as mg/mL go to step 12 with the
# sentence attached, because a paper may genuinely use one. Only the unit string is rewritten; no
# number is touched. See sample/scale.py.
step_11_polish_extraction_records:
	$(call run_step,11,Polish Extraction Records (re-apply current activity rules),$(RUN) $(ENV_EXTRACTION) sample-cli polish-extractions $(if $(DRY_RUN),--dry-run,) $(if $(LIMIT),--limit $(LIMIT),),-d,data/literature_data_json/pubmed)

# Step 12: FINAL POLISH. The fields a rule cannot settle, judged by the model.
#
# Runs after steps 9-11 and the order is the whole economy of it. Step 9 extracts, step 10
# revalidates the sequence resolutions it applied, step 11 makes every repair a RULE can make --
# 1,334 units canonicalized, 931 range bounds filled, 16,927 incoherent activities dropped, with
# no model call at all. What reaches this step is the residue nothing cheaper can decide.
#
# Bounded, and measured on the refined output rather than estimated:
#
#     target absent on a typed measurement                   7,959
#     activity_type is a placeholder the paper could fill     1,458
#     unit is mM on a potency metric (a lost micro sign)        686
#     sequence is three-letter notation that passed all rules   460
#     unit carries a U+FFFD no rule can name                     56
#     --------------------------------------------------------------
#     total                                                  10,619   ~14.7h at 2 Ollama slots
#
# Reviewing every field of every activity would be ~87,000 calls instead. The selection policy in
# sample/field_review_run.py picks a field only when a rule returned no verdict AND a cheap signal
# says the answer is consequential -- and the largest group is the one that matters most, since
# 45.6% of activities carry no target and are excluded from every usable tier for want of it.
#
# gemma4:latest, PINNED in sample/field_review.py rather than inherited from DEFAULT_EXTRACT_MODEL:
# extraction's model is chosen for recall on generation, a reviewer for precision on a verdict, and
# this project has recorded that those rankings disagree. It is also the only model here with a
# measured judgement score (12/12 on the triage verdict).
#
# Corrections are RECORDED by default and written only with APPLY=1, and then only when they
# survive the field's own deterministic check -- so this step cannot introduce a defect step 11
# would have to clean up. Run with DRY_RUN=1 first for the target counts, and MAX_CALLS to run the
# pass in pieces.
#
# The review is only half the stage. The apply-verdicts pass below acts on what it recorded, and
# runs unconditionally because RECORDING a verdict nothing reads is how this corpus came to hold
# 34,071 verdicts and zero corrections after 37 hours of inference. It costs no model calls: the
# model's proposals are already on disk, and this pass only decides which of them an authority
# confirms. It also carries a mechanical layer that needs no verdict at all -- a metric name in
# the unit column, a case variant of a vocabulary term, an abbreviated genus the document itself
# spells out -- which is why it sweeps the whole corpus rather than the documents the review
# reached. Measured on the full tree: 33,523 corrections, 5,888 proposals refused.
#
# apply-verdicts is chained INSIDE the run_step command, not written as a second recipe line. A
# second line runs unconditionally: it sits outside the START_STEP/END_STEP guard and outside the
# registry's skip, so `make all START_STEP=22` printed "Skipping Step 12" and then ran the apply
# pass anyway. Harmless here only because the pass is idempotent. Chained with && it is part of
# step 12: skipped when step 12 is skipped, and not reached if the review itself fails.
step_12_review_extracted_fields:
	$(call run_step,12,Final polish: per-field model review of what rules cannot settle,$(RUN) $(ENV_EXTRACTION) sample-cli review-fields --run $(if $(DRY_RUN),--dry-run,) $(if $(APPLY),--apply,) $(if $(MAX_CALLS),--max-calls $(MAX_CALLS),) $(if $(LIMIT),--limit $(LIMIT),) && $(RUN) $(ENV_EXTRACTION) sample-cli apply-verdicts $(if $(DRY_RUN),--dry-run,--run) $(if $(LIMIT),--limit $(LIMIT),),-d,data/literature_data_json/pubmed)

step_13_train_rf_filter:
	$(call run_step,13,Train RF Filter to Discriminate Peptides vs English Words,$(RUN) $(ENV_ML) sample-cli train-rf-filter,-f,data/rf_filter_model.pkl)

step_14_enrich_extracted_peptides_with:
	$(call run_step,14,Enrich Extracted Peptides with External DBs,$(RUN) $(ENV_EXTRACTION) sample-cli external-enrich --sources $(SOURCES) $(if $(WORKERS),--workers $(WORKERS),) $(if $(LIMIT),--limit $(LIMIT),),-d,data/patent_data_json/google_patents)

step_15_validate_peptide_data_extraction:
	$(call run_step,15,Validate Peptide Data Extraction with LLM,$(RUN) $(ENV_EXTRACTION) sample-cli validate --lit-dir data/literature_data_json --patent-dir data/patent_data_json,-f,data/validation_report.json)

step_16_validate_extractions_using_error:
	$(call run_step,16,Validate Extractions using Error Injection,$(RUN) $(ENV_EXTRACTION) sample-cli validate-error-injection --input-dir data/literature_data_json --samples $(INJECTION_SAMPLES),-f,data/error_injection_report.json)

# Step 15: build the manual-curation review set -- one directory per document, with the primary
# text beside the extraction so a reviewer (or the Gemini script in that directory, which is NOT
# part of this pipeline) can judge the extraction against the document it came from.
#
# Mixes two populations. POSITIVES are documents a released peptide was actually drawn from, so
# they bound precision. NEGATIVES are documents the pipeline reported no peptide for, and are the
# only half that can measure what it missed. They must not be pooled into one accuracy figure --
# the manifest says so too.
#
# REBUILDS BY DEFAULT: this set is regenerated far more often than it is annotated in place, and
# the old guard failed every rebuild with a FileExistsError. Pass NO_OVERWRITE=1 to protect a
# directory a curator is annotating.
step_17_select_curation_papers:
	$(call run_step,17,Select Manual-Curation Paper Set,$(RUN) $(ENV_EXTRACTION) sample-cli select-curation-set --papers $(CURATION_PAPERS) --patents $(CURATION_PATENTS) $(if $(NO_OVERWRITE),--no-overwrite,),-f,data/manual_curation/manifest.json)

# Step 16: audit that set with Gemini -- each document's real PDF, plus the extraction template
# and record schema the pipeline was held to, plus what it actually extracted.
#
# DELIBERATELY NOT in the `all` / step-range chains, and it must stay that way: this target grades
# the pipeline, and a pipeline that runs its own grader is not being graded. It also makes real,
# billed API calls over 345 documents. Run it explicitly:
#
#   make step_18_review_curation_with_gemini                        # the whole set
#   make step_18_review_curation_with_gemini CURATION_LIMIT=5       # a cheap trial first
#   make step_18_review_curation_with_gemini CURATION_MODEL=gemini-2.5-pro
#   make review_curation_dry_run                                     # free: resolves, sends nothing
#
# Results land three ways as each document finishes: a line on stdout, an appended row in
# data/manual_curation/review_results.csv, and review.json in that document's own directory.
step_18_review_curation_with_gemini:
	$(call run_step,18,Review Curation Set with Gemini,$(RUN) $(ENV_EXTRACTION) python data/manual_curation/evaluate_with_gemini.py --model $(CURATION_MODEL) $(if $(CURATION_LIMIT),--limit $(CURATION_LIMIT),),-f,data/manual_curation/review_results.csv)

# The same resolution and prompt assembly with no API call and no key required -- verifies every
# document resolves a PDF before any money is spent.
review_curation_dry_run:
	$(RUN) $(ENV_EXTRACTION) python data/manual_curation/evaluate_with_gemini.py --dry-run --redo

step_19_validate_extractions_using_manual:
	$(call run_step,19,Revise Extractions against the Strict Output Contract,$(RUN) $(ENV_EXTRACTION) sample-cli revise --sources $(SOURCES) $(if $(WORKERS),--workers $(WORKERS),) $(if $(LIMIT),--limit $(LIMIT),),-d,data/literature_data_json_revised)

step_20_extract_repair_extractions:
	$(call run_step,20,Extract Repair Extractions,$(RUN) $(ENV_EXTRACTION) sample-cli extract-repair --input-dir data/literature_data_json_revised --output-dir data/literature_data_json_repaired && $(RUN) $(ENV_EXTRACTION) sample-cli extract-repair --input-dir data/patent_data_json_revised --output-dir data/patent_data_json_repaired,-d,data/literature_data_json_repaired)

# Novelty is judged here, by sample/novelty.py rather than by set membership: the published
# `sequence_novelty` is backed by `novelty_basis`, `novelty_parent` and `name_agreement` so the
# claim is auditable. Reads data/peptide_names.json (step 35) for the name tests when it exists;
# on a first run from scratch it does not, and the name tests stay silent and say so.
step_21_merge_database_and_extracted:
	$(call run_step,21,Merge Database and Extracted Peptides,$(RUN) $(ENV_EXTRACTION) sample-cli merge-extractions --literature-dir data/literature_data_json_repaired --patent-dir data/patent_data_json_repaired --database-dir data/antimicrobial_databases,-f,data/sample_raw_release/activities.csv)

step_22_serialize_raw_datasets:
	$(call run_step,22,Serialize Raw Datasets,$(RUN) $(ENV_CHEM) sample-cli serialize,-f,data/sample_raw_release/merged_from_database.json)

# Steps 20-26 replace what used to be one monolithic step ("Enrich Datasets with Descriptors and
# 3D Structures"). Split so a single slow or failing piece -- almost always one of the external
# predictors -- costs only its own step's time to redo, not the whole enrichment's. Each of these
# reads and writes data/sample_enriched_release/enriched.csv IN PLACE, touching only its own
# columns (sample.peptide_predictors.enrich_one_predictor and friends) -- so re-running any one of
# them alone (`make step_25_enrich_anticp2`) never disturbs what an earlier one already computed.
#
# AMPlify had its own step here (originally 20) until 2026-08-26: across multiple real production
# runs of the full 53,614-peptide corpus it crashed on a sustained 7-12%+ of batches with a genuine
# upstream bug, not a resource or environment issue this project could fix -- see the comment on
# sample.peptide_predictors.EXTERNAL_PREDICTORS. Retired rather than kept as a known-broken step.
step_23_enrich_base:
	$(call run_step,23,Compute Descriptors and Stability Features,$(RUN) $(ENV_CHEM) sample-cli enrich-base,-f,data/sample_enriched_release/enriched.csv)

step_24_enrich_macrel:
	$(call run_step,24,Predict AMP/Hemolytic Activity with Macrel,$(RUN) $(ENV_CHEM) sample-cli enrich-macrel,-f,data/sample_enriched_release/enriched.csv)

step_25_enrich_anticp2:
	$(call run_step,25,Predict Anticancer Activity with AntiCP2,$(RUN) $(ENV_CHEM) sample-cli enrich-anticp2,-f,data/sample_enriched_release/enriched.csv)

step_26_enrich_hemopi2:
	$(call run_step,26,Predict Hemolytic Activity with HemoPI2,$(RUN) $(ENV_CHEM) sample-cli enrich-hemopi2,-f,data/sample_enriched_release/enriched.csv)

step_27_enrich_algpred2:
	$(call run_step,27,Predict Allergenicity with AlgPred2,$(RUN) $(ENV_CHEM) sample-cli enrich-algpred2,-f,data/sample_enriched_release/enriched.csv)

step_28_enrich_frequency_embeddings:
	$(call run_step,28,Compute Amino-Acid Frequency Embeddings,$(RUN) $(ENV_CHEM) sample-cli enrich-frequency-embeddings,-f,data/sample_enriched_release/enriched.csv)

step_29_enrich_structures:
	$(call run_step,29,Predict/Embed 3D Structures,$(RUN) $(ENV_CHEM) sample-cli fold && $(RUN) $(ENV_CHEM) sample-cli enrich-structures-fallback,-d,data/sample_enriched_release/structures)

step_30_generate_decoy_datasets:
	$(call run_step,30,Generate Decoy Datasets,$(RUN) $(ENV_CHEM) sample-cli decoys,-f,data/sample_decoy_databases/decoys.csv)

step_31_assemble_final_master_dataset:
	$(call run_step,31,Assemble Final Master Dataset,$(RUN) $(ENV_CHEM) sample-cli assembly,-f,data/sample_final_release/final_dataset.csv)

step_32_update_final_dataset:
	$(call run_step,32,Update Final Dataset,$(RUN) $(ENV_CHEM) sample-cli update-dataset --new-data data/sample_final_release/final_dataset.csv,-f,data/sample_final_release/final_dataset.csv)

step_33_validate_smiles:
	$(call run_step,33,Validate Published SMILES Column,$(RUN) $(ENV_CHEM) sample-cli validate-smiles,-f,data/sample_final_release/final_dataset.csv)

step_34_build_embedding_index:
	$(call run_step,34,Build ESM-2 Embedding Index for Embedding Search,$(RUN) $(ENV_CHEM) sample-cli build-embedding-index,-d,data/embeddings/chroma)

step_35_build_peptide_names:
	$(call run_step,35,Harvest Peptide Names and Synonyms,$(RUN) $(ENV_EXTRACTION) sample-cli build-names,-f,data/peptide_names.json)

step_36_build_bibliography:
	$(call run_step,36,Build Bibliography from Sources and CrossRef,$(RUN) $(ENV_EXTRACTION) sample-cli build-references,-f,data/references.json)

step_37_build_families:
	$(call run_step,37,Assign Peptide Families (Pfam anchors + PepMatch clusters),$(RUN) $(ENV_EXTRACTION) sample-cli build-families,-f,data/families/families.json)

step_38_build_family_hmms:
	$(call run_step,38,Build Profile HMMs for SAMPLE-Derived Family Clusters,$(RUN) $(ENV_EXTRACTION) sample-cli build-family-hmms --press $(NO_RESUME),-d,data/families/hmms)

step_39_link_bioactivity:
	$(call run_step,39,Link Peptides to ChEMBL/PubChem BioAssay/BindingDB,$(RUN) $(ENV_EXTRACTION) sample-cli link-bioactivity,-f,data/bioactivity.json)

step_40_build_modifications:
	$(call run_step,40,Harvest Chemical Modifications,$(RUN) $(ENV_EXTRACTION) sample-cli build-modifications,-f,data/modifications.json)

step_41_build_source_organisms:
	$(call run_step,41,Harvest Peptide Source Organisms,$(RUN) $(ENV_EXTRACTION) sample-cli build-source-organisms,-f,data/source_organisms.json)

step_42_build_cross_references:
	$(call run_step,42,Harvest Peptide Cross-References (UniProt/PDB),$(RUN) $(ENV_EXTRACTION) sample-cli build-cross-references,-f,data/cross_references.json)

step_43_build_gene_ontology:
	$(call run_step,43,Harvest Gene Ontology Terms,$(RUN) $(ENV_EXTRACTION) sample-cli build-gene-ontology,-f,data/gene_ontology.json)

step_44_build_go_obo:
	$(call run_step,44,Slice go-basic.obo to SAMPLE's Harvested GO Terms,$(RUN) $(ENV_EXTRACTION) sample-cli build-go-obo,-f,data/gene_ontology/sample_go_slice.obo)

# Runs before both the SQLite load (step 44) and ML training (step 45), not after, so both
# downstream consumers see a fresh is_anomaly/anomaly_score pair: `load-sql` copies them into the
# `descriptors` table (surfaced on the peptide page as a "potential outlier" badge, never dropped
# from the release), and `train-ml --filter-anomalies` reads the same column from
# final_dataset.csv. Ordered after step 49 before, both were silently unfed on a from-scratch
# `make all` run -- the column didn't exist yet when either ran.
step_45_anomaly_detection:
	$(call run_step,45,Anomaly Detection,$(RUN) $(ENV_ML) sample-cli anomaly-detection,-f,data/sample_final_release/anomaly_report.json)

step_46_load_sqlite_database:
	$(call run_step,46,Load SQLite Database,$(RUN) $(ENV_CHEM) sample-cli load-sql,-f,data/sample.db)

# The ML pipeline is scoped to antimicrobial (antibacterial) peptide prediction only, by
# instruction -- the one case study while the FLAML-searched training path is proved out.
# Anticancer, hemolytic-as-a-training-target, and synergy prediction are deliberately not part of
# this numbered pipeline: `sample-cli train-ml`/`train-synergy` still exist as manual/diagnostic
# commands (the full per-block matrix, and DBAASP synergy respectively), just not wired in here.
# Both step_43 and step_44 go through sample.flaml_xgboost.train_flaml_tasks -- FLAML-searched
# XGBoost (sample.flaml_xgboost.TIME_BUDGET_SECONDS, 600s) on the union of every block
# sample.ml_models.feature_blocks offers, with y-randomization, K-fold cross-validation and the
# single holdout all reported (sample.ml_models.fit_and_validate) -- rather than the fixed-
# hyperparameter baseline. Namespaces are resolved directly from build_tasks() at run time, not
# read from a prior report. Both share one progress file
# (data/sample_final_release/flaml_rollout_progress.json), so running them in any order or
# combination never re-trains a namespace twice; the artifact check below only confirms this run's
# log recorded the namespace as processed at all, not that it passed the quality gate (only
# roc_auc >= 0.75, among sample.ml_models.DEFAULT_QUALITY_GATE's other criteria, is published) --
# see the run's own log for that.
#
# Train/holdout splitting is grouped by cd-hit identity clusters at
# sample.ml_models.GROUPING_IDENTITY (40 %) and NOT unioned with the curated Pfam families step 35
# builds -- see sample.ml_models.UNION_FAMILIES_IN_GROUPING. Unioning the two is single-linkage
# transitive closure and it collapsed 81 % of the antibacterial positives into one unsplittable
# group, leaving a holdout at 4.4 % prevalence against training's 22.0 %. So steps 35/36 are NOT
# prerequisites of steps 45-47 any more; they still feed the family figures in step 50, the
# database and the webserver. Each grouping pass logs its largest group as a percentage of real
# peptides, which is the number to watch: a few percent is healthy, tens of percent is a collapse.
step_47_train_antibacterial_ml_model:
	$(call run_step,47,Train Antibacterial ML Model,$(RUN) $(ENV_ML) sample-cli train-antibacterial-ml-model $(RESUME_FLAG),-f,data/sample_final_release/flaml_rollout_progress.json)

step_48_train_organism_specific_ml_models:
	$(call run_step,48,Train Organism-Specific Antimicrobial ML Models,$(RUN) $(ENV_ML) sample-cli train-organism-specific-ml-models $(RESUME_FLAG),-f,data/sample_final_release/flaml_rollout_progress.json)

step_49_benchmark:
	$(call run_step,49,Benchmark Models Against External Antimicrobial Peptide Prediction Tools,$(RUN) $(ENV_ML) sample-cli benchmark-all,-f,data/sample_final_release/benchmark_report.json)

step_50_exploratory_data_analysis:
	$(call run_step,50,Exploratory Data Analysis,$(RUN) $(ENV_ML) sample-cli eda,-f,sample/static/img/eda/sequence_length_dist.png)

step_51_compare_chemical_spaces:
	$(call run_step,51,Compare Chemical Spaces,$(RUN) $(ENV_ML) sample-cli compare-chemical-spaces --dataset data/sample_final_release/final_dataset.csv --output-dir data/sample_final_release/eda,-f,data/sample_final_release/eda/chemical_space_all.png)

# The paper's figures and tables. Consumes step 47's benchmark_report.json (section 16 / Table 2
# is skipped with a warning when it is absent) plus final_dataset.csv and the step-19 activity
# table, and writes paper/figures/*.{png,pdf} and paper/dataset_stats.json. Read-only with respect
# to the release data. Numbered here, alongside the other reporting steps, because it was
# previously run by hand -- which meant the published figures could silently lag a retrain, and
# did: the grouping defect described above step 45 was only visible once Figure 16's numbers were
# read back against the holdout they came from.
#
# Then renders the manuscript to PDF (paper/build_pdf.py -> paper/sAMPle.pdf, plus
# paper/sAMPle_supplementary.pdf) in ENV_DOCS, which is where pandoc and tectonic live. The two
# halves run in DIFFERENT environments -- figures need plotly/kaleido from ENV_ML, the PDF needs a
# TeX stack that ENV_ML does not carry -- hence the two `conda run` invocations rather than one.
# The artifact check is the PDF because it is produced last: a missing PDF means the build broke
# after the figures were written, which a dataset_stats.json check would not catch.
step_52_generate_paper_report:
	$(call run_step,52,Generate Paper Figures and Tables,$(RUN) $(ENV_ML) python paper/generate_dataset_report.py && $(RUN) $(ENV_DOCS) python paper/build_pdf.py,-f,paper/sAMPle.pdf)

step_53_run_webserver:
	$(call run_step,53,Run Webserver,$(RUN) $(ENV_WEB) python -m sample.worker & WORKER_PID=$$!; trap "kill $$WORKER_PID 2>/dev/null || true" EXIT INT TERM; $(RUN) $(ENV_WEB) sample-cli webserver,-d,data)

# Build the Zenodo archive (sample/zenodo_dump.py). This was a `cp -r` chain until 2026-11-24;
# `cp -r` MERGES into the existing directory and never deletes, so every file ever removed from the
# project stayed in the archive for good -- 11 modules that no longer existed, 126 .pyc files, and
# 28 step_*.sh scripts from a 28-step shell pipeline this has not been for a long time.
# The command mirrors instead (rsync --delete --delete-excluded), excludes bytecode and the retired
# models, resolves the environment.yml symlink, and writes VERSION + MANIFEST.json.
#
# BUMP raises the release version first, writing it to pyproject.toml, sample/__init__.py and
# docs/conf.py together, and names the tarball after it (data/zenodo_dump-v<version>.tar.gz) so two
# releases cannot share a filename:
#     make step_54_generate_zenodo_dump BUMP=minor
#     make step_54_generate_zenodo_dump BUMP=patch INCLUDE_RETIRED=1
# Without BUMP the version is left alone and the archive is rebuilt in place.
# The artifact check is MANIFEST.json, not the tarball, whose name now varies with the version.
step_54_generate_zenodo_dump:
	$(call run_step,54,Generate Zenodo Dump,$(RUN) $(ENV_WEB) sample-cli generate-zenodo-dump $(if $(BUMP),--bump $(BUMP),) $(if $(INCLUDE_RETIRED),--include-retired,),-f,data/zenodo_dump/MANIFEST.json)

docs:
	@echo "=== Building HTML documentation ==="
	$(RUN) $(ENV_DOCS) python -m sphinx -b html docs docs/_build/html
	@echo "HTML docs: docs/_build/html/index.html"

# PDF is produced through LaTeX in two stages: Sphinx renders the documentation to .tex with its
# stock `manual` template, then a TeX engine compiles that to PDF.
#
# Tectonic is the engine rather than latexmk/pdflatex. Two reasons, both practical: conda's
# texlive-core installs without the format files and cannot build them (`fmtutil` dies on a
# missing mktexlsr.pl), so pdflatex reports "I can't find the format file"; and Tectonic fetches
# exactly the packages a document needs on first build instead of requiring a full TeX Live. It
# is a XeTeX engine, which is also what the Unicode in these docstrings needs -- see
# `latex_engine` in docs/conf.py.
#
# Override with TEX_ENGINE=latexmk on a machine with a working TeX Live.
TEX_ENGINE ?= tectonic

docs-pdf:
	@echo "=== Building PDF documentation via LaTeX ($(TEX_ENGINE)) ==="
	$(RUN) $(ENV_DOCS) python -m sphinx -b latex docs docs/_build/latex
ifeq ($(TEX_ENGINE),tectonic)
	cd docs/_build/latex && $(RUN) $(ENV_DOCS) tectonic -X compile sAMPle.tex --outdir .
else
	cd docs/_build/latex && $(RUN) $(ENV_DOCS) latexmk -pdf -interaction=nonstopmode sAMPle.tex
endif
	@echo "PDF: docs/_build/latex/sAMPle.pdf"

docs-clean:
	rm -rf docs/_build docs/readme.rst

extract_from_pdf:
	@if [ -z "$(FILE)" ]; then echo "Usage: make extract_from_pdf FILE=path/to/file.pdf"; exit 1; fi
	python -m sample.cli extract-from-pdf "$(FILE)" --output -

clear_extraction_data:
	@find data/literature_data_json/pubmed/ data/literature_data_json/scopus/ data/patent_data_json/google_patents/ -type f -name "*.json" -delete 2>/dev/null || true