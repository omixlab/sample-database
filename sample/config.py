import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SAMPLE_DATA_DIR", BASE_DIR / "data"))

# Subdirectories
PROMPTS_DIR = DATA_DIR / "prompts"
LITERATURE_DIR = DATA_DIR / "literature_data"
PATENTS_DIR = DATA_DIR / "patent_data"
LOGS_DIR = DATA_DIR / "repaired_logs"
RESOURCES_DIR = DATA_DIR / "resources"

# Prompts
RECOVERY_PROMPT = PROMPTS_DIR / "03_recovery_prompt.txt"
RECOVERY_SCHEMA = PROMPTS_DIR / "03_recovery_prompt_response_format.json"
EXTRACTION_PROMPT = PROMPTS_DIR / "01_extraction_prompt.txt"
EXTRACTION_SCHEMA = PROMPTS_DIR / "01_extraction_prompt_response_format.json"
SCREENING_PROMPT = PROMPTS_DIR / "00_screening_text_prompt.txt"
VERIFICATION_PROMPT = PROMPTS_DIR / "02_verification_prompt.txt"

# Default Models
#
# Extraction uses gemma4 (8B, Q4_K_M). The judgement calls in this pipeline -- "is this name a
# peptide or a plasmid/strain code/instrument?", "is this number a measured result or a buffer
# concentration?" -- were at the edge of what qwen2.5-coder:1.5b could do: it needed heavy
# few-shot prompting to reach 7/8 on candidate triage and still kept obvious non-peptides.
#
# Note gemma4:12b is ALSO installed but cannot load: it requires Ollama >= 0.30.5 and this host
# runs 0.24.0 ("unable to load model"). Upgrading Ollama is the prerequisite for using it.
#
# Screening stays on the small model: it is a cheap yes/no relevance gate run over every document,
# where throughput matters more than judgement, and it does not decide what lands in the database.
DEFAULT_SCREEN_MODEL = os.getenv("DEFAULT_SCREEN_MODEL", "qwen2.5-coder:1.5b")
#
# Changed gemma4:latest -> gemma3:4b on 2026-10-15, and the reason is ONE MODEL PER HOST, not model
# quality. This project has seven independent Ollama consumers -- candidate triage, activity
# verification, organism normalisation, activity recovery, extraction validation, concentration-unit
# adjudication, and the langextract backend. They all share one 12GB card. When this constant and
# the langextract backend's model disagreed, Ollama evicted and reloaded one for the other
# continuously: measured over 20 minutes, 31 llama-server restarts and 23 explicit "predicted to
# exceed available memory, evicting" (gemma4 8.95GiB + gemma3 3.11GiB = 12.1GiB on a 12.3GiB card).
# The GPU read as empty whenever it was sampled, the CPU sat idle, and individual documents failed
# with a different symptom on every run -- a 300s timeout, then a 600s timeout, then HTTP 500 --
# because the failure depended on which eviction a request landed in, not on the document.
#
# That fault was introduced twice: once by pointing langextract at qwen3:14b, then again by
# pointing it at gemma3:4b, both times leaving this constant on gemma4. Keep them equal.
#
# Back to gemma4:latest on 2026-10-16, by operator decision. Keep this EQUAL to the Makefile's
# LANGEXTRACT_MODEL -- that is the one rule in this block that is not a judgement call.
#
# What the switch buys:
#   - the triage benchmark below is MEASURED for gemma4 at 12/12 in 3.2s. gemma3:4b was never run
#     against it, so that was a standing unknown and is now settled.
#   - the langextract worker's prompt and few-shot examples were tuned against gemma4.
#
# What it costs, all measured on this host:
#   - RECORD YIELD AND JUNK RATE ARE WORSE, not better: on real extraction output gemma4 gave
#     6.22 records/document against gemma3:4b's 8.21, with a HIGHER junk-name rate (15.2% vs
#     13.7%). The widely-repeated claim that gemma3:4b fabricates more came from a single
#     HPLC-derivatisation paper and does not survive measurement.
#   - MEMORY. 9.6GB on disk, and Ollama does not mmap under CUDA, so the weights are anonymous
#     host RSS: gemma4 held 8.5GB and drove this 15GB machine into 8.8GiB of swap. On 2026-10-16
#     that thrashing was traced as the cause of a 16% HTTP 500 rate on /api/generate and of
#     sustained page-fault latency across the whole run; moving to gemma3:4b with
#     OLLAMA_NUM_PARALLEL=2 and OLLAMA_KV_CACHE_TYPE=q8_0 brought llama-server to 910MB and 5.3GB
#     free. gemma4 spends that headroom again. The q8_0 and 2-slot settings shrink the KV and
#     prompt caches but NOT the weights.
#   - THE PAYLOAD CACHE IS KEYED ON THE MODEL TAG, so every already-extracted document re-runs
#     the model from scratch. There is no fast cached prefix on the next run.
#
# Verify before committing a multi-day run: `make curation-model-check MODEL=gemma4:latest`.
# Two model swaps made on reasoning instead of measurement cost ~680 documents of degraded
# output on 2026-10-13/14.
DEFAULT_EXTRACT_MODEL = os.getenv("DEFAULT_EXTRACT_MODEL", "gemma4:latest")

# Candidate triage runs on the same model as everything else. It was briefly pinned to the small
# model because gemma4 returned an EMPTY completion for the triage prompt -- but the cause was not
# the model's ability, it was that reasoning models split their output into message.thinking and
# message.content while num_predict caps the two together, so the whole budget went to thinking
# and no answer was ever emitted. llm_call now sends think=False for reasoning families, and with
# that fixed gemma4 scores 12/12 on the triage benchmark in 3.2s, against 9/12 for
# qwen2.5-coder:1.5b. Kept as a separate setting so the two can still be pointed at different
# models without code changes.
DEFAULT_TRIAGE_MODEL = os.getenv("DEFAULT_TRIAGE_MODEL", DEFAULT_EXTRACT_MODEL)

# Whether reasoning families (gemma4, qwen3, deepseek-r1, phi4) may emit a chain of thought.
#
# Lives here rather than in sample.llm because the context rungs below depend on it: a trace
# shares the window with the prompt and the answer, so this one switch has to be readable before
# DEFAULT_NUM_CTX is decided. sample.llm re-exports it as THINK_ENABLED.
#
# MEASURED COST, recorded because it is the reason this was False for most of this project's
# life: these models write the trace to message.thinking and the answer to message.content, and
# num_predict caps the two TOGETHER. On one verification prompt, thinking enabled returned
# content="" in 7.1s with done_reason="length" -- the trace spent the whole budget -- against the
# correct {"supported": false} in 0.3s with think=False. That symptom is what "gemma4 returns an
# empty completion" was, and turning thinking off is what took gemma4 to 12/12 on the triage
# benchmark. THINK_TOKEN_FACTOR below is what makes enabling it survivable.
#
# DEFAULT OFF. Briefly defaulted on, on 2026-10-16, and turned back off the same day: nothing
# measured it against a real extraction run, and the only measurement that exists says this model
# returns an empty completion with thinking on. SAMPLE_LLM_THINK=1 turns it on, and the token
# budget and context rung move with it automatically -- so the switch is one variable, not four.
LLM_THINK = os.getenv("SAMPLE_LLM_THINK", "0").strip().lower() not in (
    "0", "false", "off", "no", "")

#: Multiplier on each call's completion budget while thinking is on, so the trace cannot starve
#: the answer. 6x is chosen to clear the observed failure comfortably, not tuned: the traces that
#: broke it were consuming budgets of 384 tokens entirely.
THINK_TOKEN_FACTOR = float(os.getenv("SAMPLE_THINK_TOKEN_FACTOR", "6"))

#: Floor for that enlarged budget, for call sites that ask for very few tokens.
THINK_MIN_TOKENS = int(os.getenv("SAMPLE_THINK_MIN_TOKENS", "1536"))


# Context window every LLM consumer asks for, and the reason it is PINNED rather than fitted.
#
# Ollama sizes the runner as num_ctx x OLLAMA_NUM_PARALLEL and RESTARTS it whenever a request needs
# more context than the running one provides. So a per-prompt "fitted" num_ctx -- which llm_call
# used to compute, to avoid reserving a full window for a 300-token verification prompt -- makes
# Ollama rebuild the runner continuously. Measured on this host with a single model and zero memory
# evictions: 7 runner restarts in 10 minutes at three distinct sizes (-c 8192, -c 11848, -c 16384),
# which is the model appearing to "stop and start" with nothing wrong in any pipeline log.
#
# The fitting optimisation was sound about KV cache in isolation and wrong about the whole system:
# the reload discards the cache it saved. Pinning costs a fixed reservation (4096 x slots) and buys
# a runner that loads once per run.
#
# 4096 was tuned by measurement against the 3000-character chunk size.
# sample.langextract_records.NUM_CTX now DERIVES from this constant rather than repeating it --
# the comment above used to say the two "MUST stay equal" while they were two independent
# literals, which is a rule with nothing enforcing it.
#
# THE RUNG STEPS UP WHEN THINKING IS ON. A chain of thought shares the window with the prompt and
# the answer, so a 3000-character chunk (~1,000 tokens) plus the worker's prompt and few-shot
# examples (~1,500) plus a trace and the answer does not fit 4096 with any margin -- and
# overflowing it truncates silently, which is the failure this project least wants back.
#
# 8192 x 2 slots is 16,384 of runner context, the same total this host already ran successfully
# at 4096 x 4, and with OLLAMA_KV_CACHE_TYPE=q8_0 it is half the bytes of that run.
#
# IF YOU CHANGE THIS, change OLLAMA_CONTEXT_LENGTH on the server to match. Ollama sizes the
# runner from the request, so a client asking for 8192 against a server default of 4096 rebuilds
# the runner on every call -- the exact fault the pinning above exists to prevent.
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "8192" if LLM_THINK else "4096"))

#: Larger window used only when a prompt genuinely will not fit DEFAULT_NUM_CTX. Escalating costs
#: one reload, after which the bigger runner serves the small calls too, so it converges upward
#: instead of oscillating. Truncating the prompt silently would be the worse trade.
FALLBACK_NUM_CTX = int(os.getenv("FALLBACK_NUM_CTX", "16384" if LLM_THINK else "8192"))


# How long Ollama keeps the model resident after a request. This is the single most important
# setting for a model this size: gemma4 is 9.6GB, so an eviction between calls costs a multi-second
# reload from disk into VRAM, and the pipeline issues many short calls with gaps between them.
# Ollama's default (5m) is long enough for steady work but will evict during slow stretches; 1h
# keeps it pinned across a whole run. Set OLLAMA_KEEP_ALIVE=0 to release the GPU immediately after
# each call (useful when sharing the box), or e.g. "10m" for a middle ground.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "1h")

# External resources
PEPTIDE_NAMES_TXT = RESOURCES_DIR / "peptide_names.txt"
UMLS_LEXICON_TXT = DATA_DIR / "umls_lexicon.txt"
NAME_SEQUENCE_MAP_TSV = RESOURCES_DIR / "name_to_sequence.tsv"

# Raw source database dumps used to build NAME_SEQUENCE_MAP_TSV (see
# scripts/build_name_sequence_lookup.py) -- each entry names a real, well-characterized peptide
# whose canonical sequence a paper may reference by name only, without restating the sequence.
ANTIMICROBIAL_DB_DIR = DATA_DIR / "antimicrobial_databases"
CAMP_DIR = ANTIMICROBIAL_DB_DIR / "camp"
DBAASP_DIR = ANTIMICROBIAL_DB_DIR / "dbaasp"
DRAMP_GENERAL_TSV = ANTIMICROBIAL_DB_DIR / "dramp" / "Datasets" / "general_amps.txt"
ADP_CSV = ANTIMICROBIAL_DB_DIR / "adp" / "adb.csv"

# The source subdirectories a pipeline stage may read extraction JSONs from. Stages glob
# `<dir>/*/*.json`, which silently descends into ANY subdirectory -- including the manual backups
# that accumulate beside real output. Observed in data/literature_data_json_repaired: `pubmed` and
# `scopus` were empty while `pubmed.bak`, `pubmed_old`, `scopus.bak` and `scopus_old` held 40,799
# stale files, so a merge over that directory would have been built entirely from superseded
# extractions with nothing to indicate it. Naming the valid sources makes that impossible.
EXTRACTION_SOURCE_DIRS: Tuple[str, ...] = ("pubmed", "scopus", "google_patents", "patents")

# `peptides` columns an administrator may correct by hand through `/admin`, and the only ones
# `sample.serialize.load_sql_database` reapplies from the `peptide_overrides` table after a
# rebuild. `entry_id` is the primary key every other table joins on and `is_decoy` decides whether
# a fabricated training sequence is served as real, so neither is here. Kept in this module rather
# than `sample.admin` (which imports SQLAlchemy) so `sample.serialize` -- run in `sample-chem`,
# which does not carry that dependency -- can share the one allow-list instead of maintaining a
# second copy that could drift from it.
EDITABLE_PEPTIDE_FIELDS: Tuple[str, ...] = (
    "sequence", "SMILES", "source", "Biological_Activity", "Reference")


def iter_extraction_files(base_dir: str) -> List[str]:
    """
    Extraction JSONs under `base_dir`, from recognized source subdirectories only.

    Also accepts files sitting directly in `base_dir`, which is how the per-source output
    directories (…/pubmed) are addressed when a stage is pointed at one of them directly.

    :param base_dir: Directory to scan. A missing or non-directory path yields an empty list
        rather than raising, so a stage can be pointed at an output that does not exist yet.
    :returns: Sorted absolute-or-relative paths to every extraction JSON found.
    """
    import glob as _glob
    import os as _os

    if not base_dir or not _os.path.isdir(base_dir):
        return []
    files = sorted(_glob.glob(_os.path.join(base_dir, "*.json")))
    for source in EXTRACTION_SOURCE_DIRS:
        files.extend(sorted(_glob.glob(_os.path.join(base_dir, source, "*.json"))))
    return files
