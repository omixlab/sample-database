# Engineering decisions and principles

Permanent rules first, then a dated log of specific decisions and the incidents that motivated
them. Task status does not belong here — see `TODO.md` for what is left to do, `KNOWN_ISSUES.md`
for known limitations.

## Project rules

Each rule below caused a real, measured problem before it became a rule. The incident is the
reason to keep following it, not the rule's justification in the abstract.

- **Never wrap a loop body in `except Exception: pass` without counting and logging what was
  caught.** A bare `except` inside the DBAASP ingest loop (`sample/serialize.py`) hid a 60% loss
  of the activity table — 152,308 rows — with no error anywhere in the log. A total failure looked
  identical to a partial one. An AST audit (2026-08-11) found 70 loop-wrapping `except` blocks
  codebase-wide, 28 of them silent; the four highest-risk ones in `sample/merge.py` were fixed the
  same way (count, log once at the end). 19 lower-risk sites remain — see `KNOWN_ISSUES.md`.
- **Extraction output must be read through `config.iter_extraction_files`'s allowlist, never
  `os.walk`.** `literature_data_json_repaired/` holds `pubmed`, `scopus` *and* `pubmed.bak`,
  `pubmed_old`, `scopus.bak`, `scopus_old` — 128,049 files where the live set is 46,189. Walking
  the raw tree once shipped a database where 22.7% of activity rows were duplicates, and
  separately poisoned `peptides.LITERATURE` with backup-directory filenames presented as
  citations.
- **A new database column or table must degrade gracefully against an older schema.**
  `_peptide_select()` exists so search survives a database missing a column, but `get_peptide`
  built its select list directly from `PEPTIDE_COLUMNS` instead of calling it — the one site that
  still broke with `no such column: source_organism` the moment that column shipped. Grep for
  `", ".join(PEPTIDE_COLUMNS)` before adding a fourth site like this.
- **Every third-party import must be declared in the specific conda environment(s) that actually
  run it.** Three separate incidents: `pyahocorasick` missing from `sample-chem` cost 60% of the
  activity table silently (same incident as above); `sqlalchemy` was declared in *no* environment
  file at all, only present as a transitive dependency; `python-multipart` was missing from
  `sample-web` because Starlette 1.3 parsed urlencoded bodies without it and 1.4 does not.
  `make check-envs` compares versions of packages that are present — it cannot catch one that is
  absent everywhere it matters.
- **Prefer nulling an unparseable value over guessing at it.** The SMILES validator
  (`sample/smiles_validation.py`) rejects a value it cannot classify to empty rather than
  attempting a repair — an empty cell says "unknown"; a wrong guess says something false with the
  same confidence as a right one. Applied the same way to junk peptide names
  (`sample/names.py:looks_like_junk_name`) and internal-filename references
  (`sample/db.py:is_internal_reference`).
- **Long-running or destructive actions always get an explicit go-ahead before starting.**
  Multi-hour/day GPU jobs (ColabFold over the corpus), large downloads (RefSeq genomes), and
  bulk-destructive cleanup (`ollama rm` across a model library) are never started from a backlog
  entry alone, even when clearly the next unblocked item.
- **Verify a live render, not just a unit test, before calling a feature done.** Every new
  database column or table added this session (`peptide_modifications`, `source_organism`,
  `cross_references`, `peptide_go_terms`) was checked with a throwaway probe against the real
  `data/sample.db` and the actual rendered HTML before being counted as finished, in addition to
  permanent unit tests. A passing unit test proves the function works in isolation; it does not
  prove the route wires it in or the template renders it.

### 2026-10-07 — Rebuilding the docs exposed that the first numbering sweep had missed two thirds of the drift

**The docs build was the thing that caught it.** `docs/_build` was seven weeks old (2026-08-18) and
still rendered "the strict output contract (step 13, revise)" when `revise` is step 17. Rebuilding
it published the corrected README, and a grep of the result for stale phrasing turned up
`Run step 27 first` — a `:raises:` line in `sample/families.py` that the first sweep never looked at.

**Why the first sweep missed it, stated plainly.** That sweep flagged a reference only when a
*conflicting command name appeared within 130 characters of it*. It is a good heuristic for
`build-families (step 33)` and it found 35 real defects, but it is silent on the much commoner
form: prose that names a step number and describes the artifact in words, with no command nearby.
`Run step 27 first`, `Step 17's activity table`, `step 42 builds it`, `steps 24-34 rebuild` — none
of those have a command token to collide with. Re-running the sweep against verb patterns
(`run step N`, `step N writes/builds/reads/produces`, `step N's`, `during/after/before step N`) and
then against the *complement* — every remaining prose reference, reviewed by hand — found 70 more.

**Ownership was resolved from the Makefile's own artifact arguments, not from memory.** Each
`run_step` call names the file the step must produce, which settles who owns what: `data/sample.db`
is step 44 (`load-sql`), not 42 (`build-go-obo`); `benchmark_report.json` is 47, not 45; model
pickles and `<block>.dataset.json` are 45-46, not 43 (`anomaly-detection`); `families.json` is 35
(`build-families`), not 33 (`build-names`); `activities.csv` is 19 (`merge-extractions`), not 17
(`revise`); the external-database cache is 12 (`external-enrich`), not 10; the `*_validated` tree is
13 (`validate`), not 11 (`train-rf-filter`); predicted structures and their SMILES are 27 (`fold`),
not 25. Those eight confusions accounted for most of the 70. Two references were left alone as
genuine false positives — `sample/nlp.py` and `sample/cli.py` number the steps of a local
algorithm, not the pipeline — and the retired AMPlify anecdotes in `peptide_predictors.py` lost
their step number entirely, since AMPlify has no step any more and the date is what identifies the
run. `train-synergy` likewise stopped claiming to be "step 6"; it has never been a numbered step.

**The docs were also incomplete, not just stale.** `docs/sample.rst` is hand-maintained, and nine
modules were missing from it — `synonyms`, `docling_tables`, `langextract_records`, `concentration`
and `document_quality` among them, which is to say most of what this session built. Rebuilding
documentation that does not mention the new modules is not a rebuild, so they were added
(77 -> 86 `automodule` directives, verified by diffing the directive lists rather than by counting).

**Documenting them surfaced two real defects, which is the argument for documenting them.**
`sample.flaml_xgboost.train_flaml_tasks` annotated its arguments as the *strings* `"List[str]"` and
`"Dict[str, Any]"` while importing neither name, so `typing.get_type_hints()` on it raised
`NameError` — a latent bug with nothing to do with Sphinx, found only because autodoc tried to
resolve it. `ESM2FineTuneEstimator.predict_proba`'s docstring opened directly into a `:returns:`
field, which docutils read as a field list that never closed. Both fixed; the build is back to its
24-warning baseline, all of which are third-party deprecation noise, SQLAlchemy/pydantic forward
references, and one pre-existing block-quote nit in `sample/decoys.py`.

**State after both sweeps:** 282 prose step references outside `DECISIONS.md`, every one checked
against the step that owns the artifact it describes; 1,946 tests passing; all 94 `sample` modules
importing; HTML docs rebuilt from the corrected sources with 85 module pages.

### 2026-10-07 — Cross-surface numbering sweep: 35 stale step references, and one test that had drifted twice

**The renumbers were applied to the Makefile and then propagated by hand, so every surface that
names a step by number drifted independently.** A sweep of the Makefile, README, the other docs,
`sample/`, `tests/` and the CLI found 35 references pointing at a step that had become something
else. The Makefile itself was already correct -- 52 contiguous integer targets, every `run_step`
argument matching its target, `.PHONY` exact, `all` ascending and omitting only step 16 -- which is
exactly why the drift went unnoticed: the executable surface was right and only the descriptive
ones were wrong.

**What was wrong, by kind.** User-facing error and help text was the worst category, because it
sends a reader to the wrong command: `sample-cli build-families` was announced as step 33 (it is
35), `load-sql` as 42 (44), `assembly` as 27 (29), `link-bioactivity` as 35 (37),
`build-references` as 32 (34), `decoys` as 20 (28), `select-curation-set`'s own `--help` as step 13
(15), and both FLAML trainers as steps 43/44 (45/46). The README carried two independent
range-indexed tables -- the environment table and the CLI phase table -- that had never been
regenerated after the 1--52 renumber; the environment table put step 9 in `sample-ml` when step 9
is the refined extraction, and the CLI table's "Extraction & merge (7--16)" both understated its
range and omitted `revalidate-resolutions` entirely. The README's big step table was correct for 51
of 52 rows; row 8 quoted the `ifdef PDF_PATH` branch's description instead of the default corpus
branch that `make all` actually runs.

**A test that asserted a stale number and passed anyway.** `test_missing_dataset_names_the_step_to_run`
pinned the string `"step 27"` in `export_peptides_fasta`'s error. The message and the test drifted
together through two renumbers, so the test stayed green while the message sent users to a step
that had become `enrich-structures`. It now resolves the expected number from the Makefile target
that actually runs `sample-cli assembly`, so the assertion fails when the message and the pipeline
disagree -- which is the only thing worth testing about a message whose whole job is to name a step.
Hardcoding the corrected number would have rebuilt the same trap.

**`DECISIONS.md`'s own past entries were deliberately left alone.** This is a dated, append-only
log, and an entry that recorded `step_17_extract_repair_extractions` announcing itself as "Step 16"
is a true account of a bug that existed when it was written; renumbering it would falsify the
record. Two exceptions were corrected, because they were standing claims about current
configuration rather than history: that `step_15_review_curation_with_gemini` is absent from `all`
(it is step 16) and that the anticancer import runs as `step_05b_import_anticancer_databases` (it
is step 06, and lettered steps no longer exist).

**Left unresolved, and recorded in `KNOWN_ISSUES.md` instead of guessed at.**
`data/zenodo_dump/scripts/` holds 28 untracked one-line `sample-cli` wrappers from the old 28-step
pipeline. 18 of them carry a number that now means a different step, 2 call a command that no
longer exists, and step 52 tars the directory into the published Zenodo artifact although nothing
generates it. Renumbering is not available as a fix: the numbers and the bodies disagree about what
the steps *are*, so renaming to 1--52 would produce filenames that misdescribe their own contents.
Deleting it is the recommendation and generating it from the Makefile is the alternative; both are
the user's call, and untracked files are unrecoverable, so neither was done unilaterally.

**Mechanical checks now stated as invariants**, all passing: 52 contiguous targets with no lettered
ids; every `step_NN_*` token in the tree resolving to a real target, with one documented exception
(`revision.py`'s persisted `step_13_automated_revision` provenance label, which is written into
records and must not be renumbered); all 52 README rows matching the Makefile's own descriptions;
and all 57 Makefile `sample-cli` invocations resolving against the 70 subcommands the parser
registers at runtime. 1,946 tests pass.

### 2026-10-07 — Synonym resolution shipped, then caught fabricating from the catalogue; step 10 added to withdraw what it wrote

**`sample.synonyms` gives name-only peptides a sequence from re-checkable evidence.** 72% of
extracted peptides carry a name and an activity but no sequence (14,784 of 20,604 in the current
pubmed run), and `merge-extractions` discards every one -- `if not seq: continue`. Five tiers run
strongest-first; measured recovery is ~6-7% of name-only records, which is about 1,100 peptides,
and rejected attempts are recorded so the rest can be revisited without redoing the work. It never
joins on a name alone, because **9.4% of catalogue names map to more than one distinct sequence**.

**The first version fabricated sequences out of English prose, and the measurement flattered it.**
Accepting any residue-alphabet run "recovered" `THEPEPTIDES` from "...thepeptides beh", `CASEICIN`,
`ASMELITTIN` from "as melittin", `ACATHELICIDIN` from "a cathelicidin" and `TREATEDSAMPLES` from
"Magainin-treatedsamples", reporting **12% recovery**. This is the third time this project has hit
the lost-space-prose trap (see 2026-09-18's `SCATTERINGLENGTH`), and it happened because a new
validator was written instead of reusing `docling_tables.sequence_from_cell` -- which also rejects
PCR primers, since A, C, G and T are all residue codes. Honest recovery after delegating to it:
**7.2%**, then 6.4% once two further corrections landed. Proximity is NOT evidence: 1,188 of 1,245
"a sequence sits near the name" hits were coincidence inside flattened table text.

**Then the catalogue tiers turned out to bypass validation entirely.** Auditing 1,172 applied
resolutions in the live run found `'Asp' -> 'D'` (one amino acid), `'C2' -> ACETYLATED` (an English
word in a sequence column, carried by no dictionary because it is technical vocabulary), 30
sequences of 1-4 residues, and 67 keyed on a bare table row label. A catalogue hit is evidence
about WHICH entry matches; it is never evidence that the entry's sequence field holds a sequence.
Catalogue sequences now pass a floor of 5 (matching `nlp.MIN_CATALOGUED_SEQUENCE_LEN`), the residue
alphabet, the dictionary/prose filter and an explicit denylist, and bare row labels are treated as
document-local.

**Two fixes measured as wrong before being adopted, and one caught by its own test.** A "starts
with an English function word" filter would have rejected real peptides (`AFTCHCRRSC...` is human
defensin HD6). Looser name normalisation gained 1.7% but mapped `'HAL-1/28'` to the parent's
sequence -- every analogue in a series collapsing onto one molecule -- so it was rejected. And the
row-label pattern plus a unit-letter pattern together read **`LL-37` as litre+litre** and discarded
one of the most studied human antimicrobial peptides; the test written for that regression is what
surfaced it.

**Step 10 withdraws what the buggy version wrote.** `step_10_revalidate_sequence_resolutions` /
`sample-cli revalidate-resolutions` re-checks every applied resolution against the current rules.
A dry run over the live corpus found **83 withdrawals across 32 records, 1,092 still qualifying**.
It is placed AFTER step 9 because step 9 writes resolutions of its own, it is idempotent (a record
whose resolutions hold is not rewritten), and a withdrawal clears the SEQUENCE while keeping the
`sequence_resolution` entry annotated with `withdrawn_reason` -- the repair is auditable and
reversible, which is the same standard gate 3 is held to. It reads through
`config.iter_extraction_files`, never `os.walk`.

**Pipeline renumbered 1-52.** Inserting step 10 shifted the old 10-51 up by one. Done with one
`re.sub` per pattern and a callback, so each match moves exactly once. The prose is the part that
does not move with the tokens, and a single-number regex **missed comma-separated lists**
("Steps 9, 41, 43-48" -- only the first number is adjacent to the keyword), so five such lines were
resolved by hand against the real numbering. The new step-10 comment block was protected from the
shift so its "Step 10" did not become "Step 11".

**`run_step` now REJECTS a lettered step id** rather than stripping it with `tr -dc '0-9'`. That
strip is how `13b` and `05b` came to exist, and a lettered step cannot be selected by
`START_STEP`/`END_STEP` at all because it shares its predecessor's number.

**A provenance label has now been bumped by accident twice.** `revision.py` persists
`"stage": "step_13_automated_revision"` into every peptide record. It has never matched a Makefile
target, and shifting it makes new records disagree with every record on disk. Restored and marked
NOT A STEP REFERENCE in place, since two mechanical renumbers have each caught it.

**Documentation regenerated rather than patched.** The README's step table described a numbering
several renumbers old (step 9 as `train-rf-filter`, step 13 as `select-curation-set`). It is now
generated from the Makefile, so it cannot drift again, and the README documents the 8/9/10
extraction trio, the synonym tiers with their confidences, the two caches, the OCR rescue and the
environment knobs with their measured costs. Fifteen stale step references were corrected across
`environments/README.md`, `TODO.md`, `KNOWN_ISSUES.md` and `data/manual_curation/README.md`.

### 2026-10-16 — DRAMP tracked to 5.0; the release is detected from the data because the URL does not carry it

Prompted by the DRAMP 5.0 paper (*Nucleic Acids Research*, doi:10.1093/nar/gkag989, advance article
15 Oct 2026). Reading it exposed a provenance hole that mattered more than the new data.

**DRAMP's download path does not name the release it serves.** Every file in this project comes
from `download_data/DRAMP3.0_new/`, and DRAMP updates those files in place. Verified 2026-10-16 by
request: that path answers 200 with a 28-column DRAMP 5.0 header and 13,221,293 bytes, while
`DRAMP5.0/` and `DRAMP4.0_new/` both 404. There is no version-pinned URL to move to.

**The consequence was already live and silent.** The URLs say 3.0. The files fetched through them on
2026-03-23 are **4.0** — general 11,612 and patent 17,886 match 4.0's published table exactly. The
same URLs now serve 5.0. Nothing recorded any of this, so the question "which DRAMP release is this
database built from?" had no answer short of counting rows by hand, and the URL invited the wrong
one.

- **`detect_dramp_release` identifies the release from the data**, not the path: `general` and
  `patent` row counts against the published tables, plus the `SMILES` column 5.0 introduced. It
  returns `None` when the two signals disagree rather than picking one — for a provenance question
  a confident wrong answer is worse than an admitted unknown. Only `general` and `patent` are used
  for matching: the published "specific" row counts the *stapled* dataset (377 in 4.0), which is not
  what `specific_amps.txt` contains (6,286 rows), so keying on it would mis-identify every release.
- **Every download writes `dramp/MANIFEST.json`** — per-file size, SHA-256, row and column counts,
  the detected release, and an explicit note that the "3.0" in the URL does not indicate it.
- **A refresh is opt-in** (`--refresh` / `REFRESH_SOURCES=true`). The old loop did
  `if os.path.exists(dest): continue`, so once a file was on disk no later release could replace
  it, and a new upstream release was indistinguishable from an up-to-date local copy. Skipping is
  still right for resuming an interrupted step 5; picking up a release is now a request.
- **A refresh archives rather than overwrites**, to `dramp_<release>_<date>/`. The current database
  was built from those files; discarding them would make it irreproducible, and a source dump is
  small next to the corpus it annotates.

**Two readers of this dump had drifted apart again.** `sample.merge` and
`database_activities.load_dramp` read the same files through the same iterator but chose columns
independently, which is the mistake the shared `_dramp_rows` was introduced to end:

- `merge` preferred `Original_Sequence`; the activity loader took `Sequence`. In
  `specific_amps.txt` those differ in **366 of 366** rows that populate both — `Sequence` marks
  stapling positions with Unicode circled letters (`KFFⓀKLKⓀAVKKGFKKFAKV`) against
  `Original_Sequence`'s `KFFKKLKKAVKKGFKKFAKV`. So the same peptides entered the consensus clean
  and the activity table decorated.
- The loader read `Hemolytic_activity` and `Swiss_Prot_Entry`; `specific_amps.txt` spells them
  `Hemolytic_Activity` and `Uniprot_Entry`. Both returned nothing for all 6,286 of its rows —
  indistinguishable, in the output, from "this entry has no hemolytic data". These are
  long-standing upstream inconsistencies, present in both the March and October exports, not a 5.0
  change.

Fixed by moving the column choice into `dramp_sequence` / `dramp_field` beside `_dramp_rows`, so
there is one definition of each logical DRAMP field and both readers use it.

**The new SMILES is a trust tier, placed above a derived structure for modified peptides.** 5.0
annotates 22,298 of its 32,296 entries, appended as a last column on the general, patent and
specific exports (not stability) — an append-only schema change, so no parsing had to move. In step
31 the order is now: sequence-derived structure, curated DRAMP structure, stored value if
plausible — **except** that curated wins when DRAMP records the entry as modified. A derived
structure is computed from the sequence, so it describes a linear unmodified molecule; for a
cyclic, stapled or terminally-modified peptide it is wrong in the worst way available, since it
parses, round-trips and looks exactly as trustworthy as a right answer. The paper scopes its SMILES
to precisely those cases. Both branches are counted in the step log so the trust decision is
visible.

**Deliberately not done:** DRAMP's separate `*_smiles.*` downloads are not fetched. The same
strings are already the exports' last column, so it would duplicate ~100 MB — and extra `.txt`
files inside `Datasets/` would be read as rows by `_dramp_rows`, which de-duplicates by `DRAMP_ID`
keeping the first row seen, so a two-column SMILES stub could shadow a peptide's full record.

**Also not done yet: the 5.0 data has not been downloaded.** The code can do it; the fetch replaces
the dump the current database was built from, so it waits for an explicit go-ahead, and the
downstream re-run it implies (merge → activities → enrichment) is a separate decision. The FASTS
format 5.0 introduces is a further trap if anyone switches to it: it replaces non-standard residues
with `X`, and `sample.merge` drops every sequence containing `X`, so reading FASTS instead of the
`.txt` exports would silently discard exactly the modified peptides the release is about.

### 2026-10-16 — Extraction runs in chunks of 100 with an Ollama reset between them, and each document gets a 300s budget

**The problem was a decay, not a failure.** Refined extraction held near 1 doc/s and fell to
~0.04 docs/s after about 2,000 documents. Nothing raised, nothing logged a failure, and a manual
`ollama` restart restored full speed. Every client-side structure is bounded and observably drains
(the thread pool, the submission window, the per-document subprocesses); the `llama-server` runner
is the only thing that lives for the whole run while holding a KV cache per slot and a request queue
this project has no way to inspect.

**So the fix is structural rather than diagnostic.** `SAMPLE_EXTRACT_CHUNK_SIZE` (100) documents
are processed, then the runtime is returned to its start-of-run state, then the next chunk starts.
This does not identify the cause; it bounds how long the cause can act. That is a deliberate
trade: the cause is not observable from here, and three earlier attempts to tune this pipeline by
reasoning about the server rather than measuring it cost ~680 documents of degraded output.

- **The reset is an unload, not `systemctl restart`.** `keep_alive: 0` makes Ollama terminate the
  runner subprocess, and the next request starts a fresh one -- a genuinely new process with new
  slots and an empty queue. It needs no privileges, and a reset that needed a sudo password would
  not happen on an overnight run. `SAMPLE_OLLAMA_RESET_CMD` exists for anyone who wants the
  restart instead.
- **A chunk boundary is a real barrier.** A fresh `ThreadPoolExecutor` per chunk, joined with
  `wait=True` before the reset. Unloading the model while a worker still has a request outstanding
  would hand that document a hard failure for no reason of its own. The barrier also checks
  `sample.llm.inflight_calls()`, which can disagree with the pool: the timeout path abandons a
  thread that outlives its caller by design.
- **Chunk boundaries are invisible downstream.** Consecutive slices of the same task list, so
  `--resume` and Ctrl-C behave exactly as before.

**The per-document budget is not a timeout, and the output records which.** `SAMPLE_DOCUMENT_BUDGET_SECONDS`
(300) cannot abort a document: CPython cannot interrupt a thread, and abandoning the future would
leave the worker running and still holding an Ollama slot -- the exact backlog a timeout is meant to
prevent. What it does is refuse to *start* new expensive work: each LLM call is clamped to the
remainder and refused below a 5s floor, and the langextract subprocess is killed outright. A
document that hits the cap is **written, with fewer records, flagged `metadata.budget_exceeded`**.
A missing file would be indistinguishable from "not reached yet" and re-attempted forever by
`--resume`; a flagged one can be counted and re-run deliberately.

Three things had to be fixed for that cap to mean anything:

- **`llm_call`'s timeout never fired.** It ran inside `with ThreadPoolExecutor(...)`, whose
  `__exit__` calls `shutdown(wait=True)` -- so `future.result(timeout=...)` raised `TimeoutError`
  and the context manager then blocked until the very call that had timed out returned. The
  deadline was reported at the right moment and enforced at none. Now the executor is shut down
  with `wait=False`.
- **The budget is carried thread-locally** (`sample/deadlines.py`), not passed as an argument. A
  document's work spans six modules and about a dozen LLM call sites; threading a deadline through
  all of them would be correct only as long as no caller ever forgot, and a forgotten one fails
  silently by reverting to unbounded.
- **Budget refusals must not look like an outage.** The circuit breakers open after three
  *consecutive* failures, globally, for 30s. One over-budget document refuses several calls in a
  row, so counting them would disable a verification layer for every document in flight beside it
  -- precisely the cross-document contamination the breaker was built to stop.
  `deadlines.BudgetExhausted` subclasses `TimeoutError` (so existing handlers still degrade
  gracefully) and is exempted from the breakers.

**What this does not claim.** The 25x decay has not been explained, only bounded. If it recurs
*within* a 100-document chunk, chunking is the wrong remedy and the cause is on the client side
after all -- the per-chunk rate lines now in the log are what would show that.

### 2026-10-05 — Extraction split into step 8 "screening" and step 9 "refined"; pipeline renumbered to 1-51, and the renumber exposed pre-existing reference drift

**Step 8 became two numbered steps rather than one target with a mode flag**, because the two
passes have different costs, failure modes and outputs, and because screening is a usable result on
its own:

* `step_08_screening_extract_peptide_data` -- NLP, flashtext, the catalogued-sequence automaton and
  the docling table grids, with `LANGEXTRACT_BACKEND=0`.
* `step_09_refined_extract_peptide_data` -- the same documents again with gemma4 enabled.

Old steps 9-50 shifted to 10-51 and `END_STEP` to 51. The `PHASED` flag added earlier the same day
was removed: the two numbered steps *are* the two phases.

**Step 9 deliberately does not pass `--resume`.** `--resume` skips any document whose output file
exists, and step 8 has just written one for every document -- resuming step 9 would skip the entire
corpus and add no gemma4 records at all. An interrupted refined pass is continued by re-running it;
the payload cache returns the gemma4 calls already made for free.

**The renumber was done with one `re.sub` per pattern, each with a callback.** The previous
renumber in this project (2026-09-15) used two sequential regexes that both matched the same text
and shifted it twice, destroying the zero-padding. A single callback sub transforms every match
exactly once. 189 target tokens and 53 `run_step` numbers across 10 files, then verified: 51
targets, contiguous 1-51, no duplicates, padding intact, every `run_step` number equal to its
target's number, `.PHONY` complete with no duplicates, and `make` parses.

**Two things were deliberately NOT renumbered.**

* `DECISIONS.md` -- a dated log of what happened. Renumbering its entries would falsify history, so
  entries above this one refer to pre-renumber numbers.
* The `"stage": "step_13_automated_revision"` label in `sample/revision.py`. It is a provenance
  value written into every peptide record, not a Makefile reference (it matched no target before
  the renumber either). Shifting it would make records written from now on disagree with every
  record already on disk, for no benefit.

**The audit found drift that predated this change, and fixing it was most of the work.** Comparing
every `step_NN_*` token against `git show HEAD` separated new damage from old rot: all of the
non-resolving tokens were already broken. Fixed where the intended target was unambiguous --
`anomaly_detection`, `review_curation_with_gemini`, two extraction references, an
`environments/README.md` example naming a target that has never existed, and a
`model_registry`/webserver pickle note that pointed at the SQLite load (which trains nothing)
instead of the ML steps.

Prose step numbers were a second, larger class: 25 references across the Makefile,
`environments/README.md`, `TODO.md` and `KNOWN_ISSUES.md` that the token shift did not touch. These
had to be resolved individually rather than shifted, because the `step 8`/`step 9` prose written
for the new split is already correct while a `step 9` elsewhere (RF-filter training) needed to
become 10. Three disagreed with the code they documented even before the renumber -- including the
`incremental-update` comment, which claimed the chain runs "(1-8, 10, 14-16)" when its own
prerequisite list runs 1-9, 11, 16-18.

**`step_16_review_curation_with_gemini` is absent from `all` on purpose** and the audit confirmed
it should stay that way: it grades the pipeline, and a pipeline that runs its own grader is not
being graded. It also makes billed API calls.

`docs/_build/` still contains pre-renumber names; it is generated output and regenerates from
source. 1,838 tests pass.

### 2026-10-22 — Step 12: the field review becomes the final polish; gemma4:latest pinned; prompts measured, not asserted

**The field review is now step 12, after 9-11, and steps 12-53 became 13-54.** Its position is
the economy of it: step 11 makes every repair a RULE can make -- 1,334 units canonicalized, 931
range bounds filled, 16,927 incoherent activities dropped, no model call -- so what reaches step
12 is only the residue nothing cheaper can decide. Measured on the refined output:

| reviewed | count | why no rule can settle it |
|---|---|---|
| target absent on a typed measurement | 7,959 | only the document names the organism |
| `activity_type` is a placeholder | 1,458 | the methods section names the assay |
| unit is `mM` on a potency metric | 686 | a lost micro sign and a real millimolar read identically |
| sequence is three-letter notation | 460 | `CVS` for `CYS` passes every structural rule |
| unit carries a U+FFFD no rule can name | 56 | the lost character was not a micro sign |
| **total** | **10,619** | ~14.7h at two Ollama slots |

Those are the figures as measured when the step was designed. Re-measured 2026-11-04 against a
corpus grown to 38,548 refined documents, the same selection yields 34,273 targets, with the
absent-target group at 24,638; the scale populations added that day account for 3,331. The table
above is left as written, because it is the measurement the design was argued from. See the README
for the current numbers.

A sixth population is selected here but costs no model call: ~887 records whose extracted name
identifies a peptide this project catalogues under a different sequence. The name catalogue settles
those by rule, so step 12 records the finding rather than asking the model. Which of the two fields
is wrong is not recoverable from the record, so they are flagged, not corrected — see
[What counts as a new sequence](#what-counts-as-a-new-sequence).

Reviewing every field of every activity would be ~87,000 calls. The selection policy picks a field
only when a rule returned no verdict AND a cheap signal says the answer is consequential; the
largest group is also the most valuable, since 45.6% of activities carry no target and are excluded
from every usable tier for want of that one field.

**Corrections are recorded by default and written only with `APPLY=1`, and then only when they
survive the field's own deterministic check.** A model's proposed value is evidence, not an
instruction, and the re-check means this step cannot introduce a defect step 11 would have to clean
up -- a proposed `activity_unit` of `Da` on an MIC is refused and counted, not written.

**gemma4:latest is PINNED in `field_review.py`, not inherited from `DEFAULT_EXTRACT_MODEL`.** Set
by operator decision. The two settings answer different questions: extraction's model is chosen for
recall on generation, a reviewer for precision on a verdict, and this project has already recorded
that those rankings disagree -- `gemma3:4b` wins on extraction recall (8.21 vs 6.22 records per
document) and is a judgement failure, returning `o-phthalaldehyde` and `fluorescamine` as peptides
with 65% of its records carrying no sequence. gemma4 is also the only model here with a measured
judgement score, 12/12 on the triage verdict. Inheriting would have meant a future extraction-model
change silently re-choosing the reviewer as a side effect, which is how `TestModelSplitGuard` came
to exist; `warn_if_review_model_differs_from_extraction` states the eviction cost if they ever
diverge.

**The prompt "optimization" made it worse, and the measurement is what caught it.** `v1`
measured 67% recall on known-bad with 0% false alarms and 67% `unsure` -- a reviewer that declined
rather than one that erred. `v2` bundled three hypotheses about that `unsure` rate: `reason` before
`verdict` (property order IS generation order under constrained decoding, so this is free
chain-of-thought), a positive criterion for `ok` (v1 never said what makes a value *correct*,
leaving `ok` the one option with no stated grounds), and a `settles_it` clause naming the evidence.

At n=10 per cell v2 looked better and was briefly shipped. At n=40 on the same cases it is clearly
worse:

| field | variant | caught | false | unsure |
|---|---|---|---|---|
| activity_unit | **v1** | **75%** | 5% | 60% |
| activity_unit | v2 | 10% | 0% | 70% |
| activity_type | **v1** | 35% | 20% | 72% |
| activity_type | v2 | 30% | 20% | 60% |

75% against 10% is 15 of 20 known-bad cases against 2, which is not noise. And v2's apparent
"consistent zero false alarms" -- the one signal the n=10 decision rested on -- was an artifact of
that sample size: at n=40 it matches v1's 20% on `activity_type`.

The likeliest mechanism is v2's own positive criterion. Telling the model that "the text does not
have to restate the value verbatim" bought leniency rather than decisiveness, and a unit of `min`
on an MIC does appear somewhere in a paper about minutes. An instruction written to cut the
`unsure` rate instead taught the model to accept.

`PROMPT_VARIANT` is back to `v1`, the measured best. `v3` isolates the change most likely to have
helped from the one that hurt -- `v1` verbatim plus the `settles_it` clause, with no positive-`ok`
criterion and no reordering -- and ships only if it measures better. The losers are kept so every
comparison can be re-run after a model change.

**Two lessons recorded rather than the result alone.** n=10 per cell cannot separate these
variants: five known-bad cases make a two-case difference look like a tripling. And the bundling
was the error -- v2 changed three things at once, so when it regressed, nothing said which change
did it, and a fourth variant was needed to find out.

**`FailureMode.measured` separates observation from reasoning.** Four modes in the first draft
carried neither an example nor a count -- real classes of error reasoned about, not counted here. A
test caught them; rather than delete useful prompt content the flag makes the distinction explicit,
and a second test bounds the reasoned modes to a minority so the prompts cannot drift into
describing an imagined failure surface.

### 2026-10-22 — Per-field review specs: prompt, schema, evidence budget and model, one set per field

A single "check this record" prompt cannot be good at seven fields that fail in seven different
ways. `sample/field_review.py` holds one `FieldSpec` per field, each carrying the failure modes
measured on this corpus, a deterministic pre-check, a JSON schema, an evidence budget and a model.

**Three measurements decided the design, and each changed the answer.**

*The model is the exception, not the mechanism.* A rule already settles 1,423 incoherent units,
1,163 mojibake units, 882 unbounded ranges and 68 prose types — 3,536 defects a per-record LLM
pass would have spent calls rediscovering at ~38s of model time per document.
`deterministic_verdict` is consulted first for every field; the prompts are written only for the
residue it cannot decide. Of 35 failure modes across the seven fields, 10 are rule-settled and
are deliberately left OUT of the prose the model sees, so it is not invited to re-litigate a
settled question.

*The per-field context knob is evidence, not `num_ctx`.* `llm_call` pins the window to a rung and
escalates once, because a continuously varying size makes Ollama rebuild its runner — measured at
7 restarts in 10 minutes. A spec therefore budgets characters of evidence so the finished prompt
lands on the bottom rung, `EVIDENCE_BUDGET_CHARS` derives that budget from the rung rather than
hardcoding it, and a test asserts every field's worst-case prompt fits.

*A field's own value is often the worst way to find its evidence.* Found by running the bench
rather than by reasoning: the first measurement of `activity_unit` returned 100% `unsure`, and the
model's own reason said why — "the document text provided does not contain any information about
the activity unit". The needle was the unit itself, and `min` matches "minute", "determined" and
"amino" throughout a paper, so 1,200 characters of windows contained no units at all. A unit is
written beside its VALUE. Making the needles per-field took that field from 0% to 67% recall on
known-bad cases, and `_MIN_NEEDLE_CHARS` rose from 3 to 4 because `min`, `s`, `h` and `%` are all
real units that match everywhere.

**Every spec names the pipeline model, and that is a starting position rather than a finding.**
Two things make per-field model diversity expensive here: loading a second model evicts the first
(31 llama-server restarts and 23 evictions in 20 minutes when two stages disagreed), and this
project's rule is that two swaps made on reasoning alone cost ~680 documents. So `review_many`
groups work by model, and `sample/field_review_bench.py` exists to produce the justification
before any spec moves off the default.

**The bench labels itself from the corpus.** Known-bad cases are records a deterministic rule
rejects — `validate_extracted_sequence` refusing `C10H22N6O4` is ground truth, not an opinion —
with the rule suppressed for the duration of the call so the model has to reach the verdict
itself. Known-good cases come from the trustworthy tier. Degenerate cases are excluded: an empty
field gives a reviewer nothing to judge, and without that filter `peptide_sequence`'s known-bad
population is dominated by the 26,716 empty values measured on this corpus, which would have made
the measurement meaningless. The summary states that this is a NECESSARY condition only —
reproducing the rules says nothing about the ambiguous cases the prompts actually run on.

**The report initially lied about the result, in a way worth recording.** The first version
computed an "accuracy" that counted `unsure` as a miss and ranked models on it. gemma4:latest
measured 67% recall on known-bad with ZERO false alarms on known-good and was reported as "33%
accurate", because it declined to affirm the good cases rather than asserting them. That is
exactly the conservatism the prompt asks for. The report now ranks on recall with the false-alarm
rate as the tiebreak, and labels the decided share a cost rather than an accuracy.

**`FailureMode.measured` separates observation from reasoning.** Four modes in the first draft
carried neither an example nor a count — they were real classes of error I had reasoned about,
not ones counted here. A test caught them. Rather than delete useful prompt content, the flag
makes the distinction explicit and a second test bounds the reasoned modes to a minority, so the
prompts cannot drift into describing an imagined failure surface. 2,419 tests pass.

### 2026-10-21 — Step 11 polishes already-extracted records; steps 11-52 renumbered to 12-53

**A rule fixed today otherwise reaches the corpus at the speed of re-extraction.** Step 9 writes
records under whatever rules were in force when it ran, and a document is only rewritten when the
model pass reaches it again — a multi-day job on this corpus. Four rules tightened on 2026-10-21
alone, and the output on disk showed what that costs:

| defect | count | share |
|---|---|---|
| concentration type in a mass/time/energy/count unit | 6,173 | 11.6% of activities |
| unit carrying U+FFFD from PDF decoding | 1,098 | 2.1% of units |
| `value_raw` range whose upper bound was never stored | 2,312 | — |
| `activity_type` holding prose instead of a type | 58 | — |

**New step 11 (`sample/polish.py`, `sample-cli polish-extractions`)** re-applies the current
activity-field rules to records already written. It calls no model, reads no PDF and touches no
sequence: sequence-level correction comes from step 9 re-extracting the document and from step
10's resolution revalidation, and conflating them here would hide how much of the corpus is still
stale.

**It does NOT re-run `preprocessing.standardize_and_filter_activities`, and that is the central
design decision.** That function is not idempotent on its own output — it assigns `value_raw_unit`
from the unit it is currently looking at, so a second pass over an already-converted record
overwrites the document-printed unit (`mM`) with the canonical one (`uM`) and destroys the record
of what `value_raw` was measured in. Anything pairing those two fields is then wrong by 1000x,
which is the exact failure `value_raw_unit` exists to prevent. Verified before writing a line of
the step: pass 1 leaves `value_raw_unit` as `mM`, pass 2 as `uM`. Each repair in `polish.py` is
instead individually guarded to be a no-op once applied, and idempotence is verified on real
files — a second pass changes nothing and leaves them byte-identical.

**Two safety properties the step needed because it rewrites the only copy of thousands of files.**
Documents are written through a temporary file and `os.replace`, since a plain `open(path, "w")`
truncates before it writes and an interruption would leave truncated JSON. And a file whose mtime
changed between being read and being written is left alone, counted as
`skipped_changed_underneath`: step 9 writes into the same directory for days, and reinstating the
version this step read would discard a fresh re-extraction — permanently, because that older
content still carries the refined-pass marker, so step 9's resume would never revisit it.

**Steps 11-52 became 12-53.** Lettered steps are forbidden by the rule above (they share a
predecessor's number, so no `START_STEP`/`END_STEP` range selects one alone), and appending the
polish as step 53 would have been worse than lettering: the range check is numeric, so
`make all END_STEP=20` would silently skip it. Verified mechanically — all 52 pre-existing targets
map to their old name at the new number, the sequence is contiguous 1-53, every `run_step` call's
first argument matches its target name, and `END_STEP` was raised to 53.

Prose and target references were updated in `README.md`, `environments/README.md`, `sample/cli.py`,
`sample/revision.py`, `data/manual_curation/README.md` and
`data/manual_curation/evaluate_with_gemini.py`. Three things were deliberately NOT renumbered:

- `revision.py`'s `"stage": "step_13_automated_revision"` — a provenance value persisted into
  every peptide record, which has never matched a Makefile target name. Its own comment records
  that two previous renumbers bumped it by accident; this one did not.
- `data/zenodo_dump/README.md` — a published 28-step dump with its own numbering, a historical
  record.
- This file and `KNOWN_ISSUES.md` — their step numbers describe what was true when each entry was
  written. Rewriting them would falsify the history. **Entries dated before 2026-10-21 refer to
  the pre-renumber numbering: subtract one from any step above 10.**

`tests/test_polish.py::TestTheStepNumberingIsConsistent` now pins the invariants each previous
renumber broke: contiguity from 1, no lettered targets, `END_STEP` covering the last step, every
`run_step` argument matching its target, step 11 preceding step 13 in `all`, and no stale
`step_NN_` reference in live documentation.

### 2026-10-21 — Steps resume by default; progress is recorded per run; step 9's resume needed a marker, not a file test

**The pipeline now continues where it stopped.** `make` resumes; `make FROM_SCRATCH=1` starts over.
Before this, an interrupted 52-step run had to be restarted by hand with `START_STEP=`, and the
operator had to remember which step it died on — there was nothing on disk that recorded it.

Two mechanisms, because neither covers both cases:

- **Step level.** `run_step` records every step into `data/pipeline_runs.sqlite`
  (`sample/run_registry.py`) under a run id, and skips a step already recorded `SUCCESS` under the
  current run. This is what makes an interrupted run continue at step 30 instead of step 1.
- **Item level.** The step's own command skips the documents, clusters or namespaces it already
  processed. This is what continues a single long step part-way through.

**Only `SUCCESS` earns a skip.** A step that was `RUNNING` when the run died is recorded
`INTERRUPTED` and re-run, as is a `FAILURE`. Neither has a trustworthy artifact, and treating one
as done would build the rest of the pipeline on a gap — the failure mode this project has already
been bitten by twice in other forms (the `except Exception: pass` rule above, and the backup-
directory walk).

**Step 9 could not use the existing `--resume`, and the reason is worth keeping.** `--resume`
skips any document whose output JSON exists, and step 8 writes one for *every* document — so
resuming step 9 on file existence skips the entire corpus and adds no model records at all. The
Makefile comment had recorded this correctly and therefore passed no resume flag at all, with the
advice to "simply re-run it: the payload cache returns the gemma4 calls already made for free".
That is true of the Ollama calls and partly false of the rest of a document. Measured on a live
step 9 re-walk on 2026-10-21: a document whose refined pass is already done costs ~3s to re-process
(498 outputs rewritten in 25 minutes, 1,195 docs/h) against ~38s for one that is not (94 docs/h).
So the cache does most of the work, but re-walking 11,578 finished documents to reach the
unfinished remainder is still ~9.7 hours that produces nothing, on top of the 4.4 days the
remaining 9,963 documents genuinely need. An earlier draft of this entry put the re-walk at four
days by applying the new-work rate to it; that was wrong, and the measured gap between the two
rates is the whole reason the payload cache was built.

The fix is a third mode, `--resume-mode refined`, which reads `metadata.langextract_backend` from
the output instead of testing for the file. Measured over 4,410 outputs on disk the marker
separates the two passes exactly — 2,322 `enabled` (refined pass done), 1,983 `disabled`
(screening only), 105 absent (documents skipped before the backend was consulted, which are
retried and skip again cheaply) — with no overlap. `--resume` survives as a synonym for
`--resume-mode exists`, which remains right for step 8.

**A deterministically-skipped document counts as done, and that clause was missing at first.**
Gate 1 (`sample.document_quality.is_unusable_document`) is a pure function of the text and
`metadata.skipped` is written at exactly one site, so the refined pass reading the same text
reaches the identical verdict on a captcha page or a blank scan. Without the clause those 509
documents never acquired a marker, so every resume re-queued all of them — and because they sit
early in the iteration order (139 of the first 400 queued), a restart spent its first several
minutes re-skipping junk while reporting `0 peptides extracted so far`. Observed live: 119
documents written after a restart, 113 of them skips, 6 reaching the model, 0 peptides. An
operator watching that restarts again, and the run never reaches real work. With the clause the
queue goes from 10,472 to 9,952 and the first 200 queued documents are all real work.

**A document whose input text is newer than its output is re-queued regardless of any marker.**
This is the general form of the same correctness question: an output describes the input that
existed when it was written. 427 of the pubmed corpus are Cloudflare bot-check pages saved as
PDFs (604 bytes, `Are you a robot?`), and if those are ever re-downloaded as the real papers the
mtime comparison is what stops the resume from skipping them forever on the strength of a stale
`skipped` marker.

**`resumed_count` was wrong for a run interrupted during its first step.** The adoption path
required a step to have reached a terminal status before counting a resume, on the theory that it
was needed to tell "second step of this make" from "second invocation". The extra condition meant
`make step_09` alone, killed during step 9, recorded `resumed_count = 0`. Caught on the live run
(20261021T162333Z-14f7f9, `attempts=2`, `resumed_count=0`), not by the test suite, which only
covered the case where an earlier step had finished.

**The interrupted sweep then had to be scoped to one step, for the same reason in reverse.** The
first fix closed EVERY `RUNNING` row on the theory that a surviving one proves a new invocation.
That is false when two steps of one run are in flight at once: running `make
step_11_polish_extraction_records` by hand while step 9 was genuinely mid-flight — 1h44m elapsed,
five worker subprocesses alive — recorded step 9 as `INTERRUPTED` and bumped `resumed_count`,
purely because another step was being started in a second terminal. `resolve` now takes the step
about to run and closes only that step's stale row; a row belonging to any other step is left
alone. A long step with a quick manual one alongside it is ordinary operation, not an
interruption. Found by using the tool, not by the tests that shipped with it.

**Gate 4b: an activity whose unit cannot express its type is dropped.** Seen on a live run —
`brevinin-1RTb | activity=MIC | value=2548.2 | unit=Da`. That is the peptide's molecular weight,
lifted out of a mass-spec table into the MIC column. Measured over 53,107 activities on disk,
**6,173 (11.6%)** pair a concentration activity type with a unit that is not a concentration:
2,852 mass, 2,078 time, 753 energy, 490 count. Gate 4 cannot catch these — the number really is
in the paper, so it grounds — and once such a row reaches a database no reader can distinguish it
from a real MIC.

`sample.concentration.activity_unit_is_coherent` judges one direction only: a type from a closed
set of concentration measurements (`MIC`, `IC50`, `LD50`, …) paired with a unit from a closed set
that is never a concentration. The reverse is deliberately not judged, because the type
vocabulary is open — the model returns things like `ATP-based assay for amebic viability` — and a
whitelist there would discard real measurements. Three units are deliberately excluded from the
reject set: `mm` (indistinguishable from `mM` under any case-folding, and a legitimate ZOI
diameter), `%` (correct for hemolysis) and `mg/kg` (a real in-vivo dose). The gate applies to new
output; the 6,173 rows already on disk clear as step 9 re-extracts their documents.

**Three further field defects from the same log, all fixed in `preprocessing.py` beside the
existing OCR unit repair rather than as new mechanisms.**

*U+FFFD units (1,098, 2.1%).* PDF extraction emits U+FFFD for several different characters and
only one of them is a micro sign, so the repair is POSITIONAL: U+FFFD followed by `M`, `m`, `g`,
`mol` or `L` becomes `u`, and every other occurrence is left for the caller to reject. A blanket
replacement would turn `�10�4` (a multiplication sign and a superscript minus) into `u10u4` and
invent a unit that was never printed. Measured against the real corpus: **1,045 of 1,098 (95.2%)
fully repaired**, with the remaining 53 correctly untouched — `�10�4`, `CD8�T` (a Greek letter in
a cell-type name) and `ml�1` (a superscript minus).

*Prose activity types (58).* All 30 distinct values longer than 40 characters are assay
descriptions, not types — `hydrophobicity (RP-HPLC retention time tR )`, `Not specified (measured
by percentage relative to control cells)` — and `strict_schema` already rejected every one. The
threshold is safe by construction: the longest member of `VALID_ACTIVITY_TYPES` is `Membrane
Permeabilization` at 25 characters and it is the only one at or above 25. The prose is MOVED to
`activity_type_detail` rather than discarded, and `activity_type` emptied: the value, unit and
target beside it may all be real, so only the mislabelled field is cleared.

*Unparsed ranges (2,312).* `nlp.parse_activity_value_fast` has returned the upper bound all along;
`standardize_and_filter_activities` discarded it with `_`. Two changes: `N to M` is rewritten to
the hyphen form the parser already reads (anchored between digits, so `up to 50` is left alone),
and the bound is published — **scaled by the same factor as the lower bound in each converting
branch**. Taking it straight from the parser would have left a millimolar upper bound beside a
micromolar lower one, which is precisely the 1000x inconsistency this module's `value_raw_unit`
comment was written about. A backwards bound is not asserted, and a recovered CFU exponent carries
none, because that path replaces the value rather than scaling it.

All three, like gate 4b, apply to new output; the rows already on disk clear as step 9
re-extracts their documents.

**The refined pass's per-document fan-out now fits the model server.** `LANGEXTRACT_MAX_WORKERS`
defaulted to 6 regardless of what the server could serve, so a plain `make step_09` with
`--workers 4` issued 24 concurrent requests against this host's 2 slots — a 12x oversubscription,
and silent, because the surplus queues inside Ollama rather than failing. Measured consequences:
documents spent their whole 300s budget waiting, langextract subprocesses hit their own 299s
request timeout and fell back to the non-model path, throughput sat at 0.02 docs/s, and
llama-server's prompt-cache footprint grew from 1,363MB to 9,030MB while available RAM fell from
6,624MB to 1,086MB with 6.8GB of swap in use — recreating the exact condition
`OLLAMA_NUM_PARALLEL=2` and `OLLAMA_KV_CACHE_TYPE=q8_0` were set to escape.

`run_parallel_pipeline` now derives the fan-out from the server's real slot count before it
reports, taking 24 requests to 4. Document-level workers are deliberately NOT clamped: a document
in flight makes progress on page text, tables and gates between model calls, so parallelism there
is useful beyond the slot count, while a second thread inside one document only splits the same
call queue. An explicit `LANGEXTRACT_MAX_WORKERS` is never overridden.

The warning that reported this was itself wrong and had to be fixed first: it compared slots
against `--workers` alone, understating the request count sixfold ("2 of every 4 LLM calls will
queue" while 22 of 24 were queueing), and it advised raising `OLLAMA_NUM_PARALLEL` — the opposite
of the fix on a host with 1GB free. It now reports the product and recommends a pair of values
whose product fits the server.

**`RESUME` was retired rather than redefined.** It was a boolean that applied to step 8 alone and
defaulted to *off*; both are now wrong. The Makefile raises `$(error ...)` if it is still passed,
because silently ignoring it would give an operator who typed `RESUME=true` the opposite of a
resume.

**The registry is standard-library only, and deliberately so.** `run_step` calls it three times
per step through the system `python3`. An import of pandas — or of anything else from
site-packages — would force `conda run` per call, at 1-2s of interpreter and solver startup: a
minute of overhead on every invocation across 52 steps, paid even by a `make` that does nothing.
A test asserts the module stays free of such imports.

**Progress stats live in two tables keyed on the run id**: `runs` (status, start/end,
`from_scratch`, `resumed_count`, git commit, host, user) and `step_progress` (status, duration,
exit code, attempts, artifact path and entry count, and item counts where a command reports them).
`make pipeline-status` renders the last run, `make pipeline-runs` lists recent ones. The database
is machine-local bookkeeping, gitignored, and relocatable with `SAMPLE_RUN_REGISTRY_DB`.

**A resumed invocation adopts the open run's id rather than minting a new one linked to it.** The
alternative spreads one logical run's progress across rows that have to be stitched back together
to answer "did step 30 run yet?"; adoption keeps that a single-row lookup, and `resumed_count`
records how many invocations the run took.

`make pipeline-selftest` exercises the real `run_step` macro five times against a throwaway
registry with `true`/`false` as the step command — running, skipping, `FROM_SCRATCH`, a failing
command, and a missing artifact — so the machinery can be verified without starting a multi-day
stage. 2,216 tests pass.

**Also fixed, found while checking the logged DOI field:** `_DOI_SOURCES` joined on
`pubmed_metadata.csv`, whose `internal_id` is `pubmed_<pmid>`, while a record's `document_id` is
the acquisition hash that only `pubmed_downloaded.csv` carries. The index built without error,
reported a plausible size, and matched nothing — every logged DOI was `-` across 11,698 documents.
`scopus_metadata.csv` holds a single empty row, so that half was dead too. Both now point at
`*_downloaded.csv` (22,050/22,050 pubmed stems resolve). Every existing test in
`TestActivityLogging` monkeypatched `_DOI_INDEX`, so none of them touched the join; the new one
builds it from disk.

### 2026-10-05 — gemma4:latest kept after benchmarking; a gemma4 gate at "unit"; 96.8 GB of unlinkable orphan PDFs removed

**No local model beats gemma4:latest, and the reason is that model size is not the lever.** Measured
on 5 documents holding 29 gold sequences (peptides the Gemini review confirmed present and the
pipeline missed), with the GPU idle and the payload cache disabled:

| model | size | recall | s/doc | speedup |
|---|---|---|---|---|
| **gemma4:latest** | 9.6 GB | **21/29 = 72.4%** | 118.1 | — |
| gemma3:4b | 3.3 GB | 12/29 = 41.4% | 110.6 | 1.07x |

A model **3x smaller ran 6% faster** and gave up **31 points of recall** — and was slower than
gemma4 on two of the five documents. That is consistent with everything else measured here: two
concurrent requests give 1.31x not 2.0x, and six client workers are *slower* than two. The work is
bound by chunk processing on a compute-saturated RTX 3060, not by weights, so shrinking the model
buys almost nothing and costs recall heavily. `qwen3:4b` and `llama3.2:3b` were downloaded and
queued but the comparison was stopped once the decision was made. `gemma4:12b` was removed for
disk: it is a reasoning model that returned an empty `response` with its budget spent on `thinking`.

**The gemma4 gate is "unit", and the aggressive keyword gate was rejected on measurement.** The
cheap passes — NLP, flashtext, the catalogued-sequence automaton, table grids — now always run,
and only the expensive pass is gated. Measured over 63 documents holding 247
LangExtract-attributable peptides against 4,000 others:

| gate | gemma4 calls saved | LangExtract peptides lost |
|---|---|---|
| `assay AND unit` | **55.8%** | **69.2%** |
| `no-cheap-activity AND unit` | 18.7% | 33.2% |
| `unit present` (default) | 8.5% | **2.4%** |

The aggressive gate is not merely lossier, it is lossy in a **biased** way: LangExtract's value is
concentrated in documents whose text lacks clean assay-and-unit strings — scanned, OCR'd and
table-heavy ones — so that gate is ANTI-CORRELATED with where the model helps and discards roughly
seven of every ten peptides only it would have found. An earlier framing of this idea looked far
better (51.5% skipped) because it was measured as "skip the document entirely", which also discards
the cheap passes; gating only the model is the correct comparison and it is much less profitable.

**The gate tests the text the model will READ, not the `.txt`.** For a scanned document the
pdfplumber `.txt` is 0 bytes, so gating on it would reject precisely the 36.7% of patents the OCR
rescue exists to recover. It is applied to the resolved source text, and both the gate name and its
decision are recorded per document (`langextract_gate`, `langextract_skipped`) — a document never
sent to the model must not look like one the model read and found nothing in.

**81,955 orphan PDFs (96.8 GB) removed as unlinkable.** The PDF filename *is* the `internal_id`, and
**0 of 43,443** pubmed orphan ids appear in `pubmed_metadata.csv` despite its 103,269 rows, while
`scopus_metadata.csv` has no `internal_id` column at all. The orphan ids also split across two
generation schemes (37,081 at 44 characters, 6,362 plain 36-character UUIDs), so these files were
written by an acquisition run whose identifiers were never recorded anywhere on disk. After removal
the PDF count equals the manifest row count **exactly** for both sources (22,370 and 19,011), which
is the verification that only orphans went. Disk went from 95% to 75% used. Every removed path and
size is recorded in `logs/orphan_pdfs_removed_20261005_1306.tsv`; the filenames are the only trace
that survives, and they cannot be re-acquired because no DOI was ever associated with them.

**203 MB of stale extraction directories removed.** `pubmed.bak`, `pubmed_old`, `scopus.bak` and
`scopus_old` under `literature_data_json_repaired/` held 40,799 files that
`config.iter_extraction_files` cannot reach — its allowlist saw 41,061 files and none of those. They
dated from July 2026. This is the concrete cost of the `os.walk` hazard the project rule already
records.

**A from-zero re-run is under way and its honest cost is weeks, not days.** Holding at ~0.03 docs/s
with 4 workers, that is **~12.8 days for pubmed and ~26.7 days for the full 46,189-document
corpus**. The 8.5% gate saving is real and barely dents it. The previous output is preserved at
`data/literature_data_json_pre20261005_1303` and `data/patent_data_json_pre20261005_1303` so new and
old yields stay comparable. The run is launched with `setsid nohup` and detached from any shell:
an earlier run died while SIGSTOPped with no controlling shell, which is also the lesson for
suspending long jobs — suspend them by explicit pid, briefly, or not at all.

### 2026-10-05 — Speed work: the LangExtract defaults are already optimal, the prefilter is unsafe, and the GPU is the bound. A result cache is the only real win.

**Three of the four candidate optimizations were measured and REJECTED.** Recording them so they
are not re-attempted.

**A cheap precondition to skip the model does not pay, and the obvious version is a 15% recall
loss.** LangExtract is asked for peptides *with* an experimental measurement, so requiring an assay
token and a concentration unit looks like a free filter: it skips **51.5%** of the corpus for 4.27
ms/document, which at 101.4 s/document is ~669 of the ~1,301 GPU-hours a full pass costs. Validated
against 3,000 documents that really did yield activity-bearing peptides, it would have skipped
**14.93% of them** — including one with 22 peptides. Loosening it collapses the benefit:

| precondition | skips | false negatives |
|---|---|---|
| narrow assay AND unit | 51.5% | **14.93%** |
| broad assay AND unit | 18.4% | **1.87%** |
| unit OR sequence OR "peptide" | 4.1% | 0.03% |

Every variant with near-zero false negatives skips ~4%. The reason is structural: this corpus was
already selected by a peptide/antimicrobial search, so there is no large irrelevant population left
to skip. **Not implemented** — a 2x speedup that silently drops one document in seven is a recall
regression wearing a speedup's clothes.

**Parameter tuning is exhausted; the current 3000/4096/2 is a real optimum.** Swept on a
24,485-character document with 12 gold sequences, measuring recall *and* time, caching disabled so
every row is a genuine model run:

| buffer | num_ctx | workers | time | records | hits |
|---|---|---|---|---|---|
| **3000** | **4096** | **2** | **151.1s** | **19** | **7/12** |
| 3000 | 4096 | 6 | 267.7s | 19 | 7/12 |
| 6000 | 4096 | 2 | 146.3s | 5 | 0/12 |
| 6000 | 4096 | 6 | 144.5s | 5 | 0/12 |
| 10000 | 4096 | 6 | 82.6s | 2 | 0/12 |
| 10000 | 8192 | 6 | 302.4s | 0 | 0/12 |

Two traps here. **`10000/4096` is "1.83x faster" and finds nothing** — a sweep that recorded only
wall time would have shipped a 0%-recall config as the headline win. The cause is the one the
worker docstring already warned about: the chunk plus the prompt and both few-shot examples must
fit `num_ctx`, and past ~3000 characters they do not, so the model silently receives a truncated
chunk. And `10000/8192` hit the 300s per-request timeout added earlier the same day, which
*reported* the failure instead of returning a confident nothing.

**More client workers is SLOWER, and the reason is that the GPU is saturated, not serialized.**
Six workers took 0.56x the throughput of two. Measured directly: two concurrent requests complete
in 16.0s against 21.0s sequentially — a ratio of **1.31**, not 2.0. Ollama does overlap requests;
the RTX 3060 is simply compute-bound (91-99% utilization throughout the sweep), so extra concurrent
slots add KV-cache pressure at `num_ctx` each without adding throughput. Raising
`OLLAMA_NUM_PARALLEL` server-side is therefore worth **at most ~1.3x**, not the 4-6x it appears to
promise, and it needs a systemd change this host cannot make without a password.

**Raising docling concurrency is also unavailable, for a different reason.**
`MAX_CONCURRENT_CONVERSIONS` reads as 1, which looks like a 4x loss on the OCR path. It is correct:
the host has **15 GB of RAM with 2 GB available**, and `_default_conversion_slots` derives the bound
from `MemAvailable`. The failure it prevents is already on record above — swap-out at 43 MB/s with
the CPU 80-95% idle. GPU has headroom here; memory does not.

**What was implemented: a payload cache, which is the only lever with real headroom.** At 101.4
s/document the extraction stage costs ~1,301 GPU-hours per pass, and it is re-run constantly while
the stages after it are developed — every re-run paying in full for documents whose text and model
had not changed. The cache stores the WORKER PAYLOAD rather than the finished records, so
improvements to record assembly take effect on cached documents without re-running anything. It is
keyed on the **text content** (not the path, so an OCR pass that rewrites a document re-extracts,
while an unchanged document is reused even if it moved), the model, the chunk parameters, and a
**digest of the worker script** — the prompt and examples live in that file, so any change to what
the model is asked invalidates every entry automatically, which a hand-maintained version constant
would not. `MAX_WORKERS` is deliberately excluded: it changes throughput, not output, so tuning
concurrency must not discard the cache. Errored payloads are never cached, or a timeout would
freeze in as a document's permanent result.

**`needs_ocr` is memoized.** It was called twice per document — once via `should_ocr` for the OCR
text, once inside `docling_document` for the conversion — each call opening the PDF and reading up
to 12 pages at 24.1 ms. Honest accounting: the duplicate costs ~0.31 hours over the corpus, a
rounding error against the model. Kept because it is free and because it stops the two call sites
ever disagreeing about the same file.

**Two measurement hazards worth remembering.** The cache nearly invalidated its own sweep: the
first two configs differ only in `MAX_WORKERS`, which is excluded from the cache key by design, so
the second would have been a cache hit reporting a false ~0s. That run was discarded and re-run
with `LANGEXTRACT_CACHE=""`. And the suite was seeding **real** cache entries under
`data/langextract_cache` from stubbed payloads until an autouse fixture redirected it — which first
surfaced as the request-timeout test asserting against an empty command list, because its document
never reached the subprocess at all.

**The remaining lever is model choice, and it is unfinished.** `gemma4:latest` is 101.4 s/document
at 63.2% recall on the benchmark. A smaller model could be materially faster, but `gemma4:12b`
emits reasoning tokens, `qwen3:8b` could not be downloaded, and `SparkLLM/Spark-X2.5-4B` has no
engine in this Ollama build (0 occurrences of `spark` in the binary against 317 for `gemma4`).

### 2026-10-05 — Code review of the OCR/LangExtract work: 11 findings, all fixed; four of them recorded failed downloads as successes

**Four defects in `sample/pdf.py` shared one failure mode, and it is the expensive one.**
`download_paper_pdf` short-circuits on a file already at the destination, so anything that records
a failed download as complete is never retried and the corrupt file is what extraction reads. The
43,443 orphaned PDFs in `data/literature_data/pubmed` alone (65,493 files against 22,050 manifest
rows) are what this looks like after the fact.

* **The print-to-PDF result was discarded.** `save_as_pdf(driver, output_path)` already validates
  its capture and returns a bool; the caller threw it away and returned `True` unconditionally, so
  a failed capture became a completed download with *nothing on disk*. That defeated the atomic
  write added the day before. Now returned.
* **The resume short-circuit measured instead of validating.** `os.path.getsize(...) > 1000`
  accepts any partial write or HTML error page over 1 KB as finished, permanently. Now
  `is_valid_pdf`, which is the function that existed for exactly this and was never wired in.
* **`download_direct_pdf` wrote straight to the destination**, truncating it before the first byte
  arrives. Now streamed to `.part` beside the destination and renamed in — atomic, because the
  temporary file shares the destination's directory.
* **`os.replace` cannot cross a filesystem boundary.** The download directory is a
  `TemporaryDirectory` and the destination is under `data/`, so wherever `/tmp` is a separate
  mount — including the Docker setup this repo ships — it raised `EXDEV` and the download was
  reported as a failure at DEBUG level. The new `_move_into_place` tries the rename, then falls
  back to copy-beside-destination-and-rename, keeping atomicity *and* crossing boundaries.

**A failed DOI was charged one attempt per manifest flush, not per try.** The flush loop rebuilt
its frame from the cumulative result list every five results and handed the whole frame to the
failure ledger, which stamps `attempts=1` and sums per DOI. Reproduced: one failed download reached
**`attempts=4`**, so with the default `--max-attempts 3` the DOI was abandoned permanently after a
single real try. The ledger now receives only the rows added since the previous flush, which also
absorbs a second cause — the flush test is `len(processed_rows) % 5`, and a `None` result leaves the
length unchanged, so consecutive flushes could fire at the same length.

**The picogram repair introduced the error it was built to fix.** `adjudicate_picogram_unit`
returned a hardcoded `"ug/mL"` on a "misread micro sign" verdict, but `"pg/l"` is in
`_AMBIGUOUS_MASS_UNITS` — so a `pg/L` reading was rewritten as per-millilitre, a 1000x *denominator*
error introduced while repairing a 1000x numerator error. `micrograms_counterpart` now preserves the
volume: `pg/l` → `ug/L`, `pg ml` → `ug/mL`.

**`_dedupe_records` merged different peptides that shared a generic label.** The lookup fell back to
a name match even when the record's own sequence matched nothing, so "Peptide 1"/`AAAA...` and
"Peptide 1"/`BBBB...` became one record with the second sequence dropped and its activities
re-attributed. Generic labels are rife here. A name match is now trusted only when the two
sequences do not *contradict* each other — the fallback itself has to stay, because joining a
name-only record to a sequence-bearing one is the whole reason these backends are unioned.

**Molar detection disagreed with itself across two modules.** `concentration._MOLAR_UNIT_TO_UM`
reaches its `"m"` entry case-insensitively (it folds the unit first) while `preprocessing` matched
only uppercase `"M"`, so a lowercase `"m"` was converted at 1e6 by one and left unconverted by the
other — and `_MOLAR_UNITS` holds only micromolar spellings, so the unconverted unit bypassed
`_MAX_PLAUSIBLE_POTENCY_UM` and published. Now case-insensitive. Lowercase `"mm"` is unaffected:
`_NON_CONCENTRATION_UNITS` drops it earlier as a length.

**And four regressions in the day's own OCR work.** A missing `_predictor_command` import made
`_pdftoppm_command` raise `NameError` on the one host the branch exists for — no `pdftoppm` on
PATH — taking the whole language-detection path down; fixing it exposed a second bug, that
`_predictor_command` returns a command *list*, not a path. Language resolution sat between
`mkdtemp` and the `try/finally` that removes it, leaking a directory per failure. A failed shared
conversion returned `{}`, whose `.get("tables")` is `None` — the sentinel for "convert it yourself"
— so the most expensive operation in the pipeline was re-run on exactly the documents that had just
failed it. And a restored docstring described gap-fill behaviour the code has never had.

**Three test-isolation defects surfaced only because the fixes needed them.** `_run` in
`test_pipeline_runner.py` assigns over module attributes rather than using monkeypatch, so its
stubs outlived the test that wanted them; and `ocr_languages_for` and `available()` are both
`lru_cache`d — correct in production, order-dependent in tests — without anything clearing them.
All three now have restore fixtures, and the suite was verified to pass in forward order, reverse
order and file-by-file. **1,798 passed, 7 skipped.**

### 2026-10-02 — OCR promoted from a fallback to the text source for scanned documents; the 2026-09-18 "OCR does not pay" finding was papers-only

**The earlier conclusion was right about papers and wrong about patents, and nothing in the code
said which corpus it had been measured on.** The 2026-09-18 entry below records "OCR was
investigated first and does not pay: across 300 sampled PDFs, **0%** have no text layer", and the
docling worker's docstring restated it as "99.7% of this corpus's PDFs carry a healthy text layer
... none had no text layer at all". Re-measured per corpus:

| corpus | sampled | pages | zero text layer | sparse | healthy |
|---|---|---|---|---|---|
| papers | 60 | 663 | **0.0%** | 0.0% | 100.0% |
| patents | 60 | 1,400 | **36.7%** | 0.0% | 63.3% |

There is almost nothing in between — a document either has a good text layer or literally none —
so the cheap per-page test separates the two cleanly. The patent figure is the direct explanation
for that corpus's near-total zero yield: `WO2008043938A3.txt` and `WO2014004719A3.txt` are **0
bytes on disk**. No extractor can be blamed for finding nothing in an empty file.

**Gate 1 guaranteed that the documents needing OCR were the ones that never got it.** Gate 1
rejects an empty or near-empty document and returns immediately, and it ran *before* any
PDF-derived pass. So every zero-text-layer patent was skipped as `empty`, and a table/OCR pass
placed after it was unreachable code for exactly its intended inputs. The OCR pass now runs
*before* gate 1, and its text **replaces** the document's text rather than being handed only to
LangExtract — so the NLP, flashtext and LLM passes inside `extract_peptides_from_text` all read
it, and gate 3 validates each sequence against the same text it was found in. Gate 1 is then
re-run on the recovered text, because OCR output can itself be junk and a rescue that bypassed the
gate would let a blank scan through as a real document. `captcha_or_bot_check` is deliberately not
rescuable: that text came out of the PDF, so OCR-ing the same PDF returns the same interstitial.

Verified end to end on `WO2008043938A3` (0-byte `.txt`): `text_source=docling_ocr`,
`text_layer_chars=0`, `ocr_text_chars=55720`, `ocr_rescue_succeeded=True`, `table_pass_ran=True`.
It yields 0 peptides, which is correct — it is a French cosmetic-composition patent (`A61K 8/22`,
`A61Q 11/00`) with no peptide content. Recall on scanned *peptide* patents is not yet measured.

**The OCR language is DETECTED from the page, by the local model -- the office prefix was only ever
a guess.** The first version of this resolved the language from the patent-office country prefix in
the filename. That is weak evidence in both directions: a CN-office patent may be published in
English, a US patent may quote Chinese prior art, and a journal paper has no office prefix at all,
so the entire literature corpus silently resolved to `eng`. Detection now runs in three stages,
each a fallback for the one before:

1. **Script**, from tesseract's OSD (`--psm 0`) on a 150-dpi render of page one (~0.35s). Verified
   correct on all six documents tried — Han, Cyrillic, Hangul, Japanese, Latin, Latin — but its
   confidence is routinely near zero on these scans (**Hangul 0.00, Japanese 0.67, Han 1.12**), and
   `Han` cannot separate Chinese from Japanese because Japanese kanji *are* Han characters. So the
   script only narrows which packs are worth sampling.
2. **Language**, from the local model (`llm_call`, same Ollama model as the rest of the pipeline)
   reading a one-page OCR sample taken with those packs.
3. **Office prefix**, only if neither of the above produced anything.

Measured on six documents in five scripts: **6/6 linguistically correct in 1.8–4.1s each**, against
~12 s/page for the full OCR pass, so detection is free in context. The clear win over the prefix
table is `WO2008043938A3`, an entirely French PCT publication: the prefix gave `eng,fra,deu`,
detection gives `fra,eng` — French first and the dead German pack dropped.

**Two defects showed up only because the detector was tested against real pages.** The model read
the Japanese patent `JP2025516591A` as "English, Japanese" — both genuinely present on a bilingual
patent, but ordered that way tesseract would have OCR'd a Japanese page English-primary and
mis-segmented the glyphs. And for `CN111718497B` it offered "Chinese, English, Japanese", the third
an artefact of the Han ambiguity. The **script now arbitrates ordering**: the leading pack is the
first the model named that is consistent with the detected script, and the model's extra guesses
are dropped. One language plus English is what this corpus needs — the non-Latin text for context,
Latin for the sequences and reagent names. Result is cached per PDF (`lru_cache`), which is right in
production and made the tests order-dependent until they cleared it.

**The language still has to be right, because getting it wrong is worse than not running OCR at
all.** 52% of the patent corpus is CN, plus 490 KR, 126 JP and 177 SU/RU — about 70% is
non-Latin script. Tesseract asked for `eng` on a Cyrillic page does not fail; it returns Latin
look-alikes. `SU1569953A1` OCR'd under `eng` as `FOCYQAPCTBEHHBHA HOMUTET` and under `rus,eng` as
the correct `ГОСУДАРСТВЕННЫЙ КОМИТЕТ` — **0 Cyrillic characters recovered versus 3,965**. That
output is not merely useless: long runs of capitals are what a sequence detector keys on, and
`CCCP` is four valid amino-acid letters, so eng-only OCR would have manufactured peptides that
were never printed. Languages are resolved from the patent-office prefix, native pack first (it is
primary for tesseract, and an English-primary model mis-segments CJK glyphs), with `eng` always
appended because sequences and reagent names are Latin even in a CN patent. WO and EP are *not*
English offices — the EPO publishes in EN/FR/DE and WO appears in its filing language — so they
get `eng,fra,deu`.

**Two implementation traps, both found by measurement rather than reasoning.** docling validates
every element of its `lang` list individually, so the single token `"rus+eng"` — which tesseract's
own CLI accepts — raises `OcrLanguageNotSupportedError`, exits 1 and loses the document; languages
must be passed separately. And `docling_document()` now returns tables *and* text from **one**
conversion, because the first version of this change called docling twice per document (once for
tables, once for text) and would have doubled the cost of precisely the scanned documents that are
expensive. For the same reason OCR is gated by `should_ocr()` *before* converting: docling runs
TableFormer in ACCURATE mode regardless, so converting the 124,526-paper corpus to recover text
its `.txt` already holds would have cost hundreds of hours and returned nothing.

**Cost, measured.** ~12.3 s/page (5m20s for a 26-page patent), 2.4 GB peak RSS. `DOCLING_OCR=auto`
is the default and spends that only where the text layer is unusable: ~1,882 patents (~37 h at 4
concurrent conversions) and ~0 papers. `always` would pay full OCR on the ~63% of patents and 100%
of papers that do not need it — thousands of hours — and `never` restores the old behaviour. OCR
text is cached under `data/ocr_text/`, mirroring the corpus layout, and deliberately **not** beside
the `.txt` it supplements: several steps glob their text directories, and a second `.txt` per
document there would be ingested as a separate document and double-count every peptide in it.

**LangExtract's per-request timeout was silently costing whole documents.** The Ollama provider
defaults its HTTP timeout to **120s** and raises `InferenceRuntimeError` when it expires; the
worker converts any exception into "no records", so one slow chunk — or a model paying its load
time inside the first request — discards **every peptide in that document**, logged only as a
fallback. Found while benchmarking: `qwen2.5-coder:1.5b` returned **0 records** on its first
document for this reason alone, then **18 records** on the next once warm. It is now set
explicitly via `language_model_params={"timeout": ...}` to 300s
(`LANGEXTRACT_REQUEST_TIMEOUT`), which must stay below the 600s subprocess bound so the specific
error surfaces rather than a generic kill. This is a production defect, not a benchmark artifact —
it has been silently dropping documents on every slow chunk in the corpus.

**Model benchmarking needs a warm-up, or it measures load time.** The first comparison ran a
`gemma4:latest` that an earlier smoke test had already loaded against cold challengers, so the
120s window counted ~20s of model load against the challengers only. Each model is now warmed with
a one-token request before timing. With that fixed, `gemma4:latest` reproduces exactly across runs
at temperature 0 (7/12, 6/6, 3/5 on the first three documents, 113.3s vs 109.1s), so the
comparison is reproducible rather than a single noisy sample.

**A test that reached the network was passing for the wrong reason.** `tests/test_pipeline_runner.py`
claims in its docstring to be "free of LLM calls", but it stubbed only the extractor, not the table
pass — so with Ollama up, three gate assertions passed or failed according to what the model
returned that run. Stubbing `_table_peptides` fixed those and, incidentally, two
`test_langextract_records` failures: `available()` is `lru_cache`d, and the live call was poisoning
its cache for later tests.

### 2026-09-18 — Table extraction rebuilt on docling + Tesseract, after four of my own detectors produced false positives

**The measurements came first, and three of them overturned the premise I started with.**

Sizing the "missed peptides" problem, a regex that accepted any 15+ residue-alphabet run reported
that 79.4% of zero-yield documents contained a recoverable sequence (~16,038 documents). It was
matching English prose with the spaces removed — `SCATTERINGLENGTH`, `PEPTIDETHERAPIES`,
`INTERDIGITATING`. Re-measured with word-boundary isolation and a dictionary filter: **11.4%,
~2,293 documents**, and that is still an upper bound (`EFFECTIVENESSAT` survives it). The same
class of error recurred four times in one session — `REFERENCES`/`TETRAMIC` earlier, pdfplumber's
`find_tables()` reporting 26 "tables" in a 12-page paper of which the first several were 2x2 grids
holding **zero non-empty cells**, and a "Table N caption + numeric rows" heuristic that selected
pages whose only table was the phrase *"(Table 2 and Fig. 1)"* in running prose. **A detector is
not evidence until it has been run against cases it should reject.**

OCR was investigated first and does not pay: across 300 sampled PDFs, **0%** have no text layer and
**0.3%** are sparse (<200 chars/page);
*(Corrected 2026-10-02 — see the entry above. That 300-PDF sample was papers. Re-measured on
patents, **36.7% of 60 sampled (1,400 pages) have a text layer of exactly zero characters**, and
this paragraph's conclusion does not hold for that corpus.)* an earlier probe recovered **0 of 33** missed sequences by
OCR, and conversion loss is rare (median `.txt`/PDF ratio 99.7%). Tesseract is therefore wired as a
narrow fallback, not a headline path.

**What the trained detector actually shows.** Measured with docling's TableFormer over 40 random
PubMed PDFs (437 pages, network blocked): **68%** of documents hold a table, **25%** hold one
carrying peptide data (sequence / MIC / strain / activity) — 19 relevant tables, **214 data rows**,
at **1.14 s/page** (~12.5 s/document; 9.5 h over the corpus at 8 workers). `sample/pdf_tables.py`
was reachable only from `sample-cli extract-tables`, never from `process_all_extractions`, so at
corpus scale tables were never read at all.

The incumbent's yield was then measured directly rather than assumed. Across **6 table-rich PDFs**
— the same documents on which docling recovers 3, 6 and 4 real tables — the pdfplumber path
produced **2 records in total, and 0 on the three richest**. Verified against `git show HEAD:` that
this is pre-existing behaviour and not a regression from the unit fix below. So the honest
statement is not that pdfplumber is noisier; on these papers it returns essentially nothing, which
also means `--backend auto`'s fallback has nothing to fall back to and exists only for hosts where
`sample-docling` has not been built.

**Defects found in the existing table path, each of which produced wrong data rather than none:**

- `activity_unit` was hardcoded to `"ug/mL"` for every activity regardless of the column header.
  This corpus really does contain `MIC ( µ M)` columns, so those were recorded in micrograms per
  millilitre — wrong by ~1000x for a typical peptide. The unit is now **read from the header**, and
  a cell whose column declares no unit yields **no activity at all**. Omitting the field instead —
  the first attempt, and the intuitive "honest" choice — is wrong: `normalize_unit(None)` returns
  `""`, and `""` is absent from `units_for_activity()` for *every* assay, so each such record was
  silently discarded by `enforce_activity` three stages downstream. Measured across 4,000 existing
  extractions, no surviving activity has an absent or empty unit. Dose-response columns are the one
  exception: their values are percentages, so `%` is the unit and the row's own concentration is
  kept as `dose_raw`/`dose_unit`.
- Activities set **both** `target_group` and `activity_group`, which are different fields rather
  than aliases. `target_group` is read at extraction time by `strict_schema`, which uses it to
  reject contradictions such as an antibacterial assay against a fungus; left unset, that check is
  inert on every record and reports nothing. `activity_group` is the persisted column behind the
  activities table, its index, and the `activity_group=` filter the search API and web UI expose.
  Emitting only one of them either disables validation or breaks search.
- `target_organism` was title-cased, which breaks binomial nomenclature (`Escherichia Coli`). Worse,
  even correctly cased, no strain designation resolves against the NCBI taxonomy — `Escherichia
  coli D21` and `Pseudomonas aeruginosa ATCC 15692` are both rejected outright. Labels now go
  through `database_activities.resolve_target_species`, the helper the database ingest already uses,
  which falls back to the binomial and then the genus. It also returns `""` for clade labels, which
  is what stops a table's section-heading rows (`Gram-positive bacteria`, `Yeasts`) from being
  recorded as species.
- `activity_type` defaulted to `"MIC"` for any cell containing a digit, labelling hemolysis
  percentages and dose columns as minimum inhibitory concentrations.
- Rows whose length differed from the header's were dropped outright — exactly the ragged and
  merged cells TableFormer exists to reconstruct.
- `nlp.is_valid_sequence`, which that module applies to **every cell of every row**, returns True
  for `"Magainin 2"`, `">128"`, `"ND"` and `"Escherichia coli"`. Table cells need a stricter test
  than prose does; `sequence_from_cell` requires the residue alphabet with a length floor and a
  dictionary check. It deliberately has **no distinct-residue floor**: `KLLKLLKKLLKLLK` is a real
  designed AMP with two.

**Wired into the extraction step and ON by default** (`SAMPLE_EXTRACT_TABLES=0` disables it). The
table path existed only behind a manual CLI command, so at corpus scale no table was ever read.
`_extract_document` now derives the PDF from the text path by dropping the `_text` suffix — one rule
covering `literature_data_text`, `papers_text` and `patent_data_text` rather than a per-source table
that a later corpus would silently fall out of — and merges table records into the text-derived ones
by sequence, so a peptide reported in both prose and a table stays one peptide. Records are tagged
`discovery_path="table_match"`, normalised to the extractor's own shape, and pass through Gate 3
like everything else. Measured: 100% of extracted `.txt` files have a PDF, so the pass reaches the
whole corpus.

Two things this cost, both measured rather than assumed:

- **Concurrency is memory-bound, not CPU-bound.** One conversion peaks at **2,086 MB** (`/usr/bin/time
  -v`) and this host has ~9.7 GB available with swap already 5 GB in. Four concurrent conversions is
  ~8.3 GB and already at the edge, so `MAX_CONCURRENT_CONVERSIONS` stays at 4; raising it to 12 — the
  intuitive fix for the runtime — would need ~25 GB and drive the machine into swap, on top of the
  extraction step's own multi-GB shared lookups. Subprocesses are also pinned to one thread each
  (`OMP/MKL/OPENBLAS/TORCH_NUM_THREADS=1`): torch otherwise takes one thread per core, and 8 worker
  threads each spawning a 16-thread process puts 128 runnable threads on 24 cores.
- **The runtime dwarfs the step it attaches to.** ~19–27 s/document against a full pubmed extraction
  that takes **4.6 h end to end** — roughly 28 h for pubmed, ~60 h for all 46,189 documents, and it
  cannot be parallelised away on this hardware. Existing outputs carry no table data (0 of 3,000
  sampled), and `--resume` skips any document that already has output, so adopting this means
  genuinely reprocessing rather than resuming.

Measured yield on documents the pipeline actually processes — not on hand-picked PDFs, which is what
every earlier figure in this entry came from: **58%** (7/12) contain at least one table but only
**14%** (2/14) yield peptide records, at ~5 records each. That projects to ~6,600 documents and
~33,000 records corpus-wide, but n=14 puts the 95% interval at roughly 4–43%, i.e. 9k–99k records. A
bounded pilot (`--limit 500 --output-dir …`) narrows that; more reasoning does not. The per-document
metadata (`table_pass_ran`, `table_records_found`, `table_peptides_added`) exists so that pilot can
be read off the outputs directly.

**The subprocess returned its JSON on the same stream a third-party library prints to, and lost
whole documents silently.** TableFormer's `MatchingPostProcessor` writes orphan-cell recovery
warnings through its own writer, and the weight loader prints a tqdm bar; neither goes through
`logging` or `warnings`, so suppressing both — which the worker did — left them in front of the
payload. `json.loads` then failed with `Extra data: line 1 column 5` and the caller logged one
WARNING and reported **zero tables**. It fired on precisely the ragged, merged-cell tables the
docling path exists to recover: one paper whose six tables are real returned nothing. The payload
now travels in its own file, because which library prints what to stdout is not something a caller
can enumerate in advance. Re-measured after the fix, the held-out record counts were **unchanged**
(6 tables recovered on that paper, still 0 records): its tables are clinical — `Characteristic`,
`Patients (N 5 156)`, `P Value` — with no assay column and no unit, so 0 is the correct answer. The
defect was real and cost whole documents, but it did not explain the held-out result, and an earlier
draft of this entry claiming it did was wrong.

**A model decides picogram-vs-microgram, because no rule can.** OCR loses the micro sign in several
ways, and `preprocessing._OCR_UNIT_CANONICAL` repairs the mechanical cases unconditionally — `lM`,
`Mg/mL`, `Lg/mL` are not units in any system, so rewriting them is unambiguous. `pg/mL` is not like
that: it is a perfectly real unit, and measured across 6,000 extraction texts it appears in **4.58%**
of them, overwhelmingly in genuine measurements — cytokine ELISA readouts ("both <8 pg/ml"), limits
of quantification ("was 10 pg/ml"), plasma peptide levels ("ANP, BNP and CNP are 1528 ± 158 pg
mL⁻¹"). A repair-table entry or a value-range floor would have corrupted all of those to catch the
minority that are artefacts. The artefacts are real too: one sentence reads "TBSV (0.007-2 ug/ml)
and TMVP (0.06-16 pg/ml)" — same assay, same list, one unit a million-fold out of line — and OCR of
a scanned paper rendered the same column "0 pg. /ml." on page 4 and "0 ug./ml." on page 5.

`llm_validator.adjudicate_picogram_unit` asks the local model to weigh the assay and the surrounding
text: picograms are routine for immunoassays and implausible for an MIC against an organism.
Verified against the three real cases above — the two genuine ones were kept, the misread was
overridden. It is gated by `is_ambiguous_mass_unit` so no other unit ever reaches the model, carries
its own circuit breaker (a run of failures here must not open the one gating extraction validation),
and **keeps the printed unit on every failure path** — an unreachable backend, an unparseable reply,
a reply missing the verdict field. A judge that cannot be consulted must not rewrite a measurement.
An override records `activity_unit_original` and the model's reason beside the new unit, so the call
is auditable and reversible rather than a silent edit.

The value range would not have caught it either: `UNIT_RANGES` permits `pg/mL` up to 1e10, so a MIC
of "32 pg/mL" passes `normalize_value` unremarked, and zero pg-based units exist among the 147,498
activities in the live database.

**Working around the project's own validator lost the one check that could not be hand-rolled.**
`sequence_from_cell` deliberately avoided `nlp.is_valid_sequence`, because on table cells that
function returns True for `"Magainin 2"`, `">128"`, `"ND"` and `"Escherichia coli"`. Avoiding it also
discarded its composition log-likelihood test (`nlp.py`'s `_DNA_MIN_LEN` block), which exists
because A, C, G and T are all one-letter residue codes — so a PCR primer clears the residue
alphabet, the length floor, the dictionary and the prose rule alike. Methods sections print primer
tables routinely: of the ten records from the first two documents that yielded anything, **seven
were primers**, and they were the records that SURVIVED `merge.py`'s sequence filter, so they would
have entered the database as peptides. The text extractor never had this problem — 0 of 3,000
sampled existing outputs contain a DNA-alphabet peptide — precisely because it uses that test. Both
conditions are now required: the shared test for what only it can catch, the strict local gate for
the prose and organism names it lets through. **Reuse the existing validator and add to it; do not
replace it because it is imperfect in a different direction.**

**A sixth home-made detector produced a false positive, this one in committed code.**
`sequence_from_cell` accepted the clade heading `Gram-negative` as the sequence `GRAMNEGATIVE`:
every letter is a residue code, stripping the hyphen clears the length floor, and `GRAMNEGATIVE` is
absent from the word list. `Gram-positive` escaped only because `o` is not a residue code. Cells are
now split on hyphens and whitespace and rejected when every part is an ordinary word. Running count
for the session: a 15-residue prose regex, `find_tables()` counting empty ruled boxes, a
caption-plus-numeric-rows page selector, `nlp.is_valid_sequence` on table cells, a selection
heuristic that matched *"(Table 2 and Fig. 1)"* in running prose, and now this.

**Every table record was being deleted by the extraction pipeline's third gate, and stage-by-stage
verification never showed it.** 50 unit tests passed, 72/72 activities satisfied `strict_schema`,
units and organisms and assay classes were all correct — and
`document_quality.sequence_supported_by_context` still deleted **4 of 4** records. That gate asks
whether a sequence appears in its own stored context window and is the only irreversible deletion in
the extraction stage. Papers print sequences in three-letter code (`H-Phe 1 -Leu 2 - …`); the
converted one-letter form appears nowhere on the page, not even after whitespace and hyphens are
squashed, so any page-derived window fails. The gate is not at fault — it exists to remove residue
runs a regex captured from prose, and cannot distinguish those from a legitimate conversion. Its own
contract supplies the answer: it returns True for records with no context at all, naming "a database
import, **a table parser**, or an LLM repair". So a table record keeps its window only when the
sequence is genuinely findable in it, and otherwise carries none. Appending the converted sequence to
the stored context would have made the check pass by **fabricating the evidence the gate is asking
for**, which is worse than storing none; only `llm_validator` reads the field and it already falls
back to the activity's own `context_sentence`.

The first attempt at that fix was placed in the per-table record builder and did nothing, because an
assay record has no sequence until `merge_document_records` joins one from the sequence table — the
check saw `sequence=None`, concluded there was nothing to find, and kept the window. Same
compute-then-overwrite shape as the `target_group` defect above. Provenance had the mirror problem:
the first record wins the merge, so a record cited the sequence table's page while every value in it
came from an assay table several pages later.

**Assay tables are frequently transposed** — peptides across the columns, organisms down the rows,
under one spanning `MIC (uM)` header. Read row-wise, every value is attributed to the wrong entity.
Orientation is now detected from the two-row header structure. Where the row-label column declares
a concentration of its own (`peptide concn ( µ M)` with rows `3/6/12/24`), those labels are doses,
not organisms, and `target_organism` is left unset rather than recording `"3"` as a species.

**Three-letter sequences were being lost silently, and the naive fix corrupts them.** Sequence
tables are written `H-Phe 1 -Leu 2 - ... -Leu 13 -NH 2`. Passing that cell straight to
`sequence.sequence_to_1_letter` returns `HFLPLIGRVLSGIL` — the N-terminal `H-` read as a histidine,
a *wrong* peptide rather than a missing one, which no downstream validity check would catch. The
terminus and residue numbering are stripped first, giving `FLPLIGRVLSGIL`. Verified independently:
the analogue named `Gln 3 -TA` converts to `FLQLIGRVLSGIL` against `TA`'s `FLPLIGRVLSGIL` — a P→Q
substitution at position 3, exactly what its name asserts.

**The cheap pre-filter is opt-in, not the default.** Requiring both a `Table N` mention and an
assay keyword in the already-extracted text keeps 21% of documents and would cut the corpus pass to
~2.0 h. But measured against the trained detector, **10% of gate-negative documents still hold a
relevant table** (3 of 30; 20 data rows) — ~1,742 documents corpus-wide. That is a silent loss of
real data to save seven hours of one-time compute, so the full sweep is the default.

**docling does not run locally out of the box; both failures were found only by blocking the
network.** Its default OCR engine (RapidOCR) downloads PP-OCRv6 weights from `modelscope.cn` at
first use — the weights are already on disk under `~/.cache/docling/models/RapidOcr`, but
rapidocr's torch engine looks elsewhere. And `docling-tools models download` populates
`~/.cache/docling` while the conversion pipeline resolves weights through
`huggingface_hub.snapshot_download`, which reads `~/.cache/huggingface/hub`: a machine that has run
only the CLI still reaches the network on its first conversion. On a networked host both would have
fetched silently and the environment would have been declared local while phoning out on every
document. **Verify offline claims by blocking the network and confirming the block works against a
known-reachable URL first** — `unshare -n` is not permitted unprivileged here, so `HF_HUB_OFFLINE=1`
plus proxies pointed at a dead port is the usable harness.

docling is isolated in `sample-docling` because it resolves numpy 2.x and transformers 5.17 against
`sample-extraction`'s pinned numpy<2 (spacy). It is driven as a subprocess against that
environment's interpreter, resolved by absolute path via `_predictor_command`, never `conda run`.
Torch is pinned to the CPU build: the GPU environments already carry cu121 for ESM-2, and a second
CUDA stack costs several GB on a disk at 95%.

### 2026-09-17 — The novelty statistic was measured against a baseline of one sequence
- **"99.89% new peptides discovered" was an artefact of a clobbered input file.** Step 11's report
  divides extracted sequences by those already known, reading the known set from
  `data/antimicrobial_databases/merged_source.csv`. That file held **one row**, whose ADP accession
  was `real_one` -- a test fixture's FASTA header. Restored from
  `sample_raw_release/merged_from_database.csv` (md5-identical, 53,618 rows, 45,333 distinct valid
  sequences), the same extracted corpus now reports **0.69% new: 317 genuinely new sequences of
  1,890, with 1,573 already in the databases**. The headline inverted completely. Nothing else in
  the report was wrong -- the counts reconciled all along; only the denominator was destroyed.
- **The write that destroyed it had a guard, applied to one of two call sites.**
  `merge_database_and_extractions` only writes that compatibility copy when the run is producing
  the canonical release; `merge_databases` wrote it unconditionally, to a hard-coded RELATIVE path,
  ignoring `--output` entirely. So any run with a custom output -- a test, a subset, a comparison --
  replaced the project's baseline with a partial result. Guarded now, matching its twin, with a
  regression test that runs `merge_databases` from a throwaway CWD and asserts nothing appears at
  the shared path. The existing comment on the guarded twin already said this damage "is not
  recoverable from within the pipeline"; that turned out to be too pessimistic, since the file is a
  mirror of one that survives, but it was right that the pipeline cannot repair itself.
- **Fixing the baseline had to be possible without re-running the audit.** Step 11 runs the LLM
  field validation and *then* computes the report, so correcting a denominator would have cost tens
  of thousands of LLM calls -- and worse, an unreachable backend makes every verdict `None`, which
  is a keep-everything no-op that would have REPLACED good validated output with unvalidated
  copies. `validate_extractions(run_llm=False)` / `sample-cli validate --report-only` now recomputes
  from the existing `*_validated` trees. Verified non-destructive: all three trees unchanged in file
  count and newest mtime across the run.
- **A note on reading test failures in this environment.** Running the one affected test file in
  isolation failed seven tests, five of them in a class that never touches the merged file -- all
  with the same `CXXABI_1.3.15` ImportError this log already documents. The failures said nothing
  about the change; confirming it required the whole suite, as the environment note prescribes.

### 2026-09-16 — PubMed extraction reviewed; every PubMed link on the site pointed at the wrong article
- **A well-formed link to the wrong record beats a broken one at hiding.** `papers.source_id` stores
  all 2,217 PubMed accessions in float form (`10075760.0`) because pandas types an integer column
  holding any NaN as float64. `external_links.build_url` cleaned that with
  `re.sub(r"\D", "", identifier)`, which deletes the dot and **keeps its trailing zero**: PMID
  10075760 became 100757600, a different (nonexistent) article. Nothing looked wrong anywhere --
  the URL was syntactically valid, the accession rendered beside it, and `describe()` reported
  `verified: True`, which means "this accession shape was resolved against the live resource", so
  the page actively vouched for a link it had mangled. 13,831 peptide pages, 2,217 distinct wrong
  URLs. Fixed in three places behind one shared `strip_float_suffix()`: the URL, the *displayed*
  accession (cleaning only the URL leaves the page printing `PubMed: 10075760.0` next to a link to
  10075760 -- two identifiers for one reference is worse than either error alone), and
  `serialize.link_refs`, so a rebuilt release stops storing the float form at all. Verified on the
  rendered page, not just in the unit test.
- **Not every outlier is a bug: PMID 29307492 is DRAMP's citation, not a mass-link defect.** One
  paper is attached to 5,908 peptides -- 11% of the release, 43% of all PubMed links, against 123
  for the next busiest. The merge is innocent: all 5,908 rows carry it as the *sole* value of
  `pubmed_ids`, every one of them has `DRAMP` populated (against ADP 0, CAMP 20, DBAMP 32, DBAASP
  33), and their `Reference` cell is the bare string `29307492`. It is a source database's own
  reference, replicated across 5,908 of its 29,668 records and faithfully carried through. What is
  wrong is presentational -- a database-level citation is rendered as if it were this peptide's
  primary literature -- not arithmetic. Left as is; recorded so the next person who finds the
  outlier does not go hunting for a merge bug that is not there.
- **The 92.4% empty extraction rate is mostly real.** Of 22,050 PubMed documents, 20,383 yield no
  peptide. That is not a starved pipeline: acquisition succeeded for 100%, every document has text,
  and the median is 40 KB. A strict probe over 500 empty extractions that had real text found 86%
  containing neither a peptide-shaped sequence nor MIC-with-units -- true negatives -- and ~0.2%
  with both, the genuine misses. The first version of that probe said 2.2%, because its
  sequence regex matched the word `REFERENCES`: every letter of it is in the residue alphabet. A
  detector for biological sequences has to exclude English, or it will confidently measure the
  wrong thing.
- **58% of surviving extractions are name-only, and that is a specificity problem, not a lookup
  gap.** Only 23 of 625 distinct extracted names (3.7%) resolve against the 35,664-name map, but the
  unresolved head is `magainin`, `nisin`, `cecropin`, `cathelicidin`, `defensin` -- family names.
  The map holds `Magainin-2`, `Nisin Z`, `Cecropin C`; it is the extractor that is recording the
  family rather than the peptide. Resolving `magainin` to a sequence would be *wrong*, since
  Magainin-1 and -2 differ. The fix belongs in extraction, and enlarging the name map would not
  touch it.

- **The Ramachandran plot now uses a real reference distribution, and the choice of library was
  decided on what it can say, not on what it draws.** The classifier was a hand-drawn box
  approximation that the page itself described as "not a density-contour method like MolProbity".
  Three candidates were measured rather than assumed. **MDAnalysis** ships a `Rama_ref` grid, but
  it is one pooled 4-degree histogram of ~100k counts: its 98% and 99.95% mass contours collapse to
  densities 3 and 1, which scores the genuinely forbidden (60, -120) as *favored*. **RamachanDraw**
  is cleanly MIT-licensed but carries a single pooled 100x100 KDE with no residue-type split and no
  favoured/allowed thresholds, so using it would mean inventing cutoffs again -- the very thing
  being removed. **pyrama** packages the four Top500 tables (Lovell et al. 2003) with MolProbity's
  own bounds, which is the only option that can express the thing that actually matters:
  (60, -120) is *allowed* for a general residue, *favored* for glycine at density 0.127, and an
  *outlier* for proline at 0.0. One set of boxes cannot say all three. Chosen on that basis.
- **pyrama ships no licence, and that risk was raised and accepted (2026-09-16).** `pip show
  pyrama` reports `License: UNKNOWN` and the wheel carries no LICENSE file, so by default it is
  all-rights-reserved, and `docker compose build` installs it into the image -- which
  redistributes it. Accepted deliberately for a research tool rather than discovered later: the
  dependency is declared and never vendored, only its data files are read, and the fallback means
  dropping it would coarsen the Ramachandran call rather than break the page. Two things follow
  from the decision. The **paper cites Lovell et al. 2003** -- the Top500 tables are independently
  published data, and the citation belongs to them, not to the Python wrapper that happens to
  package them. And if the position ever changes, the exits are unchanged: ask upstream
  (github.com/gerdos/PyRAMA) for a licence, vendor the Richardson-lab tables directly under their
  own terms, or drop to `ramachandraw` (MIT) and accept a pooled KDE with no residue-type split --
  which cannot tell that (60, -120) is an outlier for a general residue and favoured for glycine.
- **Only the data files are read; `pyrama` is never imported.** Its `__init__` pulls in
  `matplotlib.pyplot`, and a request path has no business initialising a plotting backend, so the
  four tables are located through `importlib.metadata.files()` and parsed directly. The regions are
  sent to the page as contour polygons from `contourpy` (~24 KB of vector paths) instead of a
  rendered image (~303 KB, and a bitmap that bakes in one theme's colours and cannot be hovered).
- **The numbers it reports changed, and every change is explainable.** On a 33-residue helix the
  box model said 80.6% favored / 19.4% allowed / 0 outliers; the reference says 96.8% / 0 / one
  outlier -- residue 2 LEU at phi=-21.4, psi=-27.4, reference density 1e-05, which the box's overly
  generous "allowed" alpha rectangle had swallowed. On an 87-residue model with mean pLDDT 46 the
  outliers rise from 5 to 8 and land where they should: the C-terminal tail (residues 83-86) and a
  residue at phi=+164, essentially forbidden outside glycine. More outliers on the lower-confidence
  structure is the right direction.
- **The fallback stays, and the page says which one ran.** Without the tables the box test still
  produces a figure rather than a blank panel, and `analyze_structure` returns a `method` sentence
  that the template prints verbatim -- naming Top500 only when the tables were really read. Two
  tests pin that: a page claiming a rigour the classifier did not use would be worse than either
  method alone.
- **A prototype outside the source tree caught a bug that would have shipped.**
  `tables.get(kind) or tables["General"]` raises `ValueError: truth value of an array ... is
  ambiguous` on NumPy arrays. It surfaced in a scratch script validating the classifier against ten
  hand-checked points, not in `sample/structure_analysis.py`.

### 2026-09-15 — Pipeline renumbered to a flat 1-50; a double-shift bug in the rename, caught by verification

`13b` (the Gemini curation audit, inserted beside step 13) became **14**, and every step from the
old 14 onward moved up one to land at 15-50. `END_STEP` went 49 -> 50; without that the new final
step would have been silently skipped by the range guard rather than failing loudly. The rename
covered the Makefile, `sample/`, `docs/`, `tests/`, `README.md`, `TODO.md`, `KNOWN_ISSUES.md` and
`paper/` — 330 lines across 48 files, plus 25 stale references in `tests/` found only after the
first pass, having been left out of the file list by oversight.

**A real bug, introduced and then corrected in place.** The first pass applied two regexes in
sequence: one for target names (`step_NN_`) and one for prose (`step NN`). The prose pattern then
matched the *already-shifted* target names, so every target at or above 14 moved **+2** instead of
+1, while its own `run_step,NN` argument moved +1 — leaving `step_17_extract_repair_extractions`
announcing itself as "Step 16", and a gap at 15. The same prose substitution reformatted matched
numbers through `str(int(...))`, which stripped the two-digit zero padding and turned `step_01_`
into `step_1_`. Neither defect was visible in the summary output; both were caught by checking that
each target's number equals the `run_step` number it passes, and that the sequence has no gaps.

Reverting was not available: `sample/`, `KNOWN_ISSUES.md` and `TODO.md` carried the same session's
uncommitted work, so `git checkout` would have discarded far more than the rename. Fixed forward
instead — target names at or above 15 shifted back down one and re-padded — which is verifiable in
a way a blind revert-and-retry is not: 50 targets, 50 resolving references, 51 matching
target/`run_step` pairs, no gaps, no duplicates, every touched Python file compiling.

**This file's own step numbers were deliberately NOT shifted.** The 22 "step NN" references in the
entries below are *historical*: statements like "correctly step 25 in the immediately-prior scheme,
now 33" and "the SQLite load moved from step 22 to 24" describe numbering that no longer exists.
Shifting them would rewrite the record of the previous two renumberings into something false. The
lesson from the 2026-08-12 entry — that stale step references were *already wrong before* that
renaming touched them — is the reason this log stays fixed while live references move.

### 2026-09-15 — Peptide optimizer removed; a stale BLAST index and a NULL-descriptor 500 found by probing the live server
- **The peptide optimizer was removed on request**, not deprecated in place: `sample/optimizer.py`,
  `sample/optimizer_store.py`, both templates, both test modules, the three `/api/v1/optimize`
  routes and their six response models, the two HTML routes, the `sample.optimize_peptide` Celery
  task, the nav entry and the peptide page's "Optimize this peptide" button. The dated entries
  below that describe building it are left as they were — this log records what was decided when,
  so removing the history would misrepresent it. `data/optimizer_jobs.db` is left on disk: it is
  untracked local job history, and deleting it is not needed to remove the tool.
- **Index existence is not index freshness.** `init_blast_database()` returned the formatted
  database whenever `peptides_db.pin` existed, and rebuilt only when it was missing. The index on
  this host was built on Jun 30 from an older consensus and held 26,358 sequences; the release it
  was being searched against holds 53,614. Roughly half the corpus — every peptide added since
  June, including all of `SAMPLE00000530xx` — was unreachable by BLAST, and nothing reported it,
  because a stale index answers queries perfectly well and the missing half simply never appears
  in a hit list. It now rebuilds when the index is older than the consensus CSV that feeds it.
  Same failure shape as the silent-`except` rule above: the wrong answer and the right answer are
  indistinguishable from the outside.
- **A guard on one column does not guard the row.** The enriched-descriptor table on the peptide
  page was wrapped in `{% if peptide.molecular_weight %}` and then formatted `logp`,
  `hydrogen_bond_donors` and `hydrogen_bond_acceptors` unguarded. `SAMPLE0000011784` (sequence
  `[Dab]`, a single non-standard residue) has a molecular weight but NULL for the other three, so
  `"%.2f"|format(None)` raised `TypeError` and the whole page 500'd — one released peptide with no
  detail page at all, and no test caught it because the fixtures all carry complete descriptors.
  Every formatted cell in that table now falls back to an em dash, as the physicochemical table
  below it already did.
- **`biopython` was undeclared in `environments/sample-web.yml`**, found by auditing that
  environment's imports directly rather than through `conda run`. Four things the dashboard serves
  import it lazily, inside a function: the peptide page's DSSP/Ramachandran/pLDDT card, its
  PDB-homology superposition, its ADMET summary, and the Expression Wizard's GenBank export. A
  lazy import means the app still starts and serves every page, so the gap presents as three
  peptide-page cards quietly missing (their callers catch and log at debug level) and one 500 on
  `POST /api/v1/express/download?format=genbank` — not as a failure to boot. Neither real run path
  exercises the environment that lacked it (the Dockerfile builds `environments/sample.yml`, and
  on this host `conda run -n sample-web` resolves to `sample`), which is why serving it here looked
  correct throughout. Declared, per the standing rule that an import must be declared in the
  environment that runs it; the third such incident after `sqlalchemy` and `python-multipart`.
- **The predict tab serves what the rollout published, not what is on disk.** It offered 115
  models: 106 from `train-ml`'s per-feature-block `esm2` matrix (a manual diagnostic the Makefile
  deliberately keeps outside the numbered pipeline), one `esm2_xgboost_backup`, and 8 `flaml`
  pickles. But `flaml_rollout_progress.json` records only **2** namespaces as `written` — the
  general antibacterial model and `acinetobacter_baumannii/mic` — against 1 `gate_failed` and 16
  `error`. Six of the 8 pickles on disk belong to namespaces that errored (five with
  `[Errno 5] Input/output error`, one an XGBoost `quantile.h` crash) and are dated Sep 11-12,
  before the Sep 15 pass: leftovers of the failed rollout `KNOWN_ISSUES.md` already describes.
  They carry no holdout metrics and no calibrated threshold, so every one was a prediction request
  answerable by a model of unmeasured quality. `list_models` now takes the served set from the
  progress file's `written` status instead of from the `**/*.pkl` glob. The filter is conditional
  on that file existing, so a pre-rollout release or a test fixture still lists what it has rather
  than going empty, and `SAMPLE_SERVE_ALL_MODELS=1` re-exposes everything for diagnostics.
  Separately, FLAML models had been showing *no* metrics at all, because `_report_metrics` reads
  `ml_report.json`, which only ever describes the `esm2` matrix; their scores and calibrated
  thresholds now come from the progress file.
- **Synergy removed from the served surface, on request; kept as offline tooling.** The
  "Predict Synergy with an Antibiotic" feature had two front ends, not one -- a batch panel on
  `/models` and a per-peptide panel on the detail page -- both posting to `POST
  /api/v1/predict/synergy`, both fed by one `_synergy_context()`. All of that is gone: the two
  panels and their scripts, the helper and its call sites, the two routes and the three response
  models. `sample/synergy.py`, `sample/antibiotics.py` and `sample-cli train-synergy` are kept,
  with their unit tests, because the Makefile already scopes synergy as a manual/diagnostic command
  outside the numbered pipeline -- the decision was to stop *serving* it, not to discard the FICI
  work. Only the two panel test classes were deleted; the module tests still cover the trainer.
  `model_registry`'s explicit synergy exclusion also stays: the trained pickles remain in
  `data/sample_final_release/models/synergy/`, and that branch is what keeps them from being
  offered as ordinary models it cannot actually load.
- **The 3D viewer was drawing nothing, and every HTTP-level check said it was fine.** The panel
  rendered as a flat 350px box of pure background. Status codes, byte counts and an RDKit parse of
  the served SDF were all green -- the molecule really was fetched and really was added to a
  viewer. The defect was that there were *two* viewers: the div carried
  `class="viewer_3Dmoljs"`, which is 3Dmol's auto-load hook, and the 2.x bundle fires it from
  `document.onreadystatechange` at readyState `"complete"` -- strictly after the page's own
  `DOMContentLoaded` handler had built a viewer and drawn the molecule. Autoload then created a
  second, data-less viewer on the same element and appended its canvas after the first, with the
  same `position:absolute; top:0; left:0; z-index:0`, so an empty opaque canvas painted over the
  structure. The class and its `data-backgroundcolor`/`data-style` attributes were vestiges of an
  older autoload-based implementation that had since been converted to a manual `createViewer`
  (see the jQuery note in the same block); the conversion left the hook behind. Removing the class
  takes the canvas count from 2 to 1.
- **A render is not a status code.** This was reported as working earlier in the same session on
  the strength of 200s and non-zero byte counts, which is exactly the evidence that cannot see this
  bug. It was settled by rendering the page in headless Chrome and measuring the pixels inside the
  viewer: 16 distinct colours and 0.02 % non-background before, 3,627 and 6.14 % after. Worth
  reaching for whenever the question is "does the user see it", not "does the server send it" --
  and it extends the "verify a live render" rule, which a server-side HTML assertion also cannot
  satisfy for a WebGL canvas.
- **The 3D panel now draws the fold, not the conformer -- as a cartoon coloured by secondary
  structure.** It had been rendering the RDKit-embedded SDF even for the 83% of peptides that have
  a ColabFold PDB, which is both the less informative view and, for a ribbon, an impossible one: a
  molblock carries atoms and bonds with no residue, chain or atom-name records, and a cartoon is
  interpolated through the backbone. So the viewer prefers `/pdb` and styles it by the `ss` AtomSpec
  selector (helix `#e05252`, sheet `#e0b022`, coil grey -- the same three colours as the 2D track
  above it, so the panels cannot disagree), falling back to SDF sticks for an unfolded peptide or a
  fold too short to trace (<4 CA). ColabFold writes no HELIX/SHEET records, so the assignment is
  3Dmol's own geometric one; a 33-residue dermaseptin renders as a single red helix, and an
  87-residue HTH_3 peptide renders helices plus a yellow sheet arrow. A Cartoon/Sticks toggle
  restyles the loaded model without refetching, and the SDF download now fetches on demand, since
  the conformer is no longer loaded just to show something.
- **DSSP was installed and still reported missing.** `_mkdssp_command()` resolved the binary with
  `shutil.which`, and `dssp` is a conda dependency of the serving environment -- so the binary sat
  in the same `bin/` directory as the python running the request while PATH, which on this host
  starts with `envs/sample/bin` for anything `conda run` launches (see `KNOWN_ISSUES.md`), could
  not see it. The peptide page's DSSP track therefore fell back to nothing while Ramachandran and
  pLDDT disorder, which shell out to no binary, rendered from the same PDB perfectly well: a
  partially-populated card that looked like a modelling limitation and was a PATH lookup. It now
  searches `dirname(sys.executable)` by absolute path first, and -- the half that resolving alone
  does not fix -- passes that path to Biopython as `DSSP(..., dssp=<path>)`, which otherwise
  re-execs the binary by bare name and fails for the same reason.
- **`dssp` was declared in `sample-web.yml` but not in `sample.yml`, and the Dockerfile builds
  `sample.yml`.** Every containerised deployment has therefore been serving peptide pages with an
  empty DSSP track, silently, because an absent binary degrades to `None` by design. Declared.
  The same audit is what turned up `biopython` missing from `sample-web.yml` earlier in this entry:
  two environments, each holding one half of what the structure card needs.
- **Found by probing the running server, not by reading it.** The 500 and the stale index were both
  invisible to the test suite (green before and after) and to code reading; both turned up by
  walking every route against the real `data/sample.db`. This is the "verify a live render" rule
  paying for itself a second time.

### 2026-08-14 — All four working predictors verified end to end; two real invocation bugs found; a self-inflicted sklearn regression found and fixed
The predictor-environment backlog (macrel/AMPlify/AntiCP2/toxinpred3) had never been closed out with
real, running tools — each was "written to the documented install method, not yet built." This
session built all four, and building them surfaced problems that a code review would not have:

- **Two real invocation bugs, both of the same shape**: code that correctly *finds* a tool's
  isolated environment but then does not *use* that resolution when actually calling it.
  `MacrelPredictor._run` hardcoded a bare `"macrel"` instead of `self._command` (the resolved
  absolute path `available()` had already found) — it happened to work whenever `macrel` was
  coincidentally on the caller's `PATH` and silently failed otherwise, which is exactly the shape of
  bug that passes a quick smoke test and fails in production. AMPlify's own upstream script
  (`AMPlify.py`) has a generic shebang (`#!/usr/bin/env python3`) that resolves through the
  *caller's* `PATH` rather than pinning its own environment; invoked from any other active
  environment it fails with `ModuleNotFoundError: No module named 'keras'` even though keras is
  correctly installed in AMPlify's own environment. Both fixed by using the resolved environment
  path explicitly rather than trusting a bare command name or a third-party script's shebang.
- **Two environment files had the wrong Python pin, discovered only by trying to solve them.**
  bioconda's `amplify` package requires python=3.6 specifically (the file guessed 3.8, and the
  conda solver said so outright: `requires python 3.6.*, but none of the providers can be
  installed`). AntiCP2's real PyPI package (`anticp2==1.1`, not the GitHub URL the file had — that
  install fails outright, no setup.py) declares `Requires-Python >=3.3,<=3.7`, and PEP 440 without a
  wildcard is a strict comparison: `3.7.12 <= 3.7` is False, so even python=3.7 rejects it. Only
  python=3.6 satisfies the bound unambiguously. Neither problem is visible by reading the YAML; both
  needed an actual solve/install attempt.
- **toxinpred3 stays genuinely blocked, but now exhaustively so rather than assumed.** Its pickle
  was tested against six real scikit-learn versions (1.1.3, 1.2.2, 1.3.1, 1.4.2, 1.5.2, 1.6.1), each
  verified installed via the environment's own `pip` directly (see the next item for why "directly"
  matters) — all incompatible, one in a different way (a downstream numpy error, not the pickle
  dtype error). The package's own metadata does not declare scikit-learn as a dependency at all.
  This is an upstream defect, verified rather than guessed at, and not fixable by picking a
  different version from this side.
- **Self-inflicted regression, found and fixed the same session: `conda run -n <env>` does not
  reliably isolate environments in this shell.** This shell pre-sets `PATH` with `envs/sample/bin`
  first for every command, and `conda run -n <env>` — even `-p <full-prefix>` — does not override
  that for the subprocess it launches; `conda run -n sample-anticp2 python --version` reported
  `envs/sample`'s python. The real damage: several `conda run -n sample-toxinpred3 pip install
  scikit-learn==X` commands, intended to test toxinpred3 against different versions, actually
  installed into the shared `sample` environment instead — silently downgrading its scikit-learn
  from the required `1.9.0` (the pin `sample-web.yml`/`sample-ml.yml` share, needed to unpickle
  trained models) to `1.4.2`. Caught by noticing `conda run`'s reported python version was identical
  across four different named environments (should have been 3.11/3.9/3.10/3.6), traced to the PATH
  override, and fixed by reinstalling `scikit-learn==1.9.0` directly via
  `envs/sample/bin/pip` and confirming a real trained model unpickles again. **The production code
  was never at risk**: `sample.peptide_predictors._predictor_command` resolves each tool by its
  direct absolute path (`envs/<env>/bin/<executable>`) and never shells through `conda run` — this
  bug only reached *my* interactive verification commands, which is exactly why direct-path
  resolution is the pattern that mattered here, not a coincidence.
- **Pre-computed per-peptide annotation** (`sample.peptide_annotations`, `sample-cli
  build-annotations`): the four working predictors (batched — one model load per tool for the whole
  corpus, not per peptide), Pfam/family/cluster (read from the release database, not recomputed),
  near-identity hits, and family-HMM hits, stored per peptide and resumable. Circular peptides
  (topology `head_to_tail`, or CycPeptMPDB membership) are flagged explicitly (`is_circular` +
  `circular_caveat`) rather than having every sequence-based result presented as if the molecule
  were linear — consistent with the applicability-domain discipline the rest of this project's
  models already follow.

### 2026-08-14 — `spacy` unpinned in `sample-extraction.yml` broke step 8 for real; hemolysis/toxicity ADMET gaps wired to the annotator
Two smaller closes the same day as the predictor work above.

- **`spacy` was left unpinned in `environments/sample-extraction.yml`, and it eventually bit.** A
  fresh solve on this env picked up spaCy 3.8.15, while `en_core_sci_sm` 0.5.4 (the biomedical NER
  model `sample.install_models` downloads) was trained against spaCy 3.7.4 and ships a
  `config.cfg` with `include_static_vectors: 'False'` — a string, not a real bool. spaCy 3.7.x
  reads that loosely; 3.8 tightened config-schema validation and rejects it outright
  (`Config error for 'spacy.MultiHashEmbed.v2' ... 'False' is not <class 'bool'>`), which fails
  every single document in step 8's extraction. Caught earlier in this session by direct testing
  and provisionally judged "not currently reproducible" (both live environments happened to still
  have 3.7.5 at that moment) — then it *did* reproduce, live, when the user ran the real step 8
  command and hit exactly this error. Fixed for real: pinned `spacy>=3.7,<3.8` in the environment
  file, and downgraded the already-built `sample-extraction` env's spaCy from 3.8.15 back to 3.7.5
  directly (`envs/sample-extraction/bin/pip`, never `conda run` — see the entry above for why),
  verified by loading `en_core_sci_sm` and tokenizing a real sentence. The lesson generalizes: an
  environment file that "currently works" with an unpinned transitive dependency is not the same
  claim as "will keep working" — the gap only shows up on the next fresh solve, which can happen
  any time a new environment is built or rebuilt.
- **ADMET's `hemolysis` gap now cross-references the real annotation instead of pointing at a
  future action.** `sample.admet` is sequence-only and cannot query the database, so it could only
  ever say "HemoPI2 predicts this, run the annotator" — a static pointer that stayed true even
  after the annotator existed and had actually scored a peptide. `sample/api/pages.py` (which
  already loads both the ADMET profile and the pre-computed annotation for the peptide page)
  now rewrites that gap's status to `computed` with the real HemoPI2/macrel hemolysis values once
  `build-annotations` has reached that peptide, leaving the static note only for peptides the
  batch hasn't processed yet. `toxicity`'s gap note was also corrected from "pending_adapter" (an
  adapter was never the blocker) to state plainly that ToxinPred3 is upstream-blocked, matching
  the exhaustive verification recorded above — it will not get an adapter because there is nothing
  a working adapter could unpickle.

### 2026-08-14 — Incremental update's acquisition half: a diff, not a new pipeline
`sample.incremental.update_families` already did the hard part (place new peptides into the
existing family map without renumbering); what it explicitly declined to do — by design, per its
own docstring — was decide *which* sequences are new. That question turned out to have a small
answer, not a new pipeline: step 15 (`merge-extractions`) already deduplicates its consensus CSV by
sequence and assigns the next free `SAMPLE*` id to anything genuinely new, so after a fresh run of
steps 1-15 that CSV is a strict superset of what the live release database already has. The only
missing piece was a diff (`sample.acquisition.find_new_peptides`: consensus `entry_id`s absent from
`peptides` in the live database) and a driver to call `update_families` on the result
(`sample-cli incremental-update`) — no new acquisition logic, matching the plan in `TODO.md`.
`make incremental-update` chains the existing step targets (1-8, 10, 13-15); steps 9, 11, 12 are
independent quality checks (RF-filter training, LLM/error-injection *reports*) that step 13 does
not read from, so they are not in the critical path and stay separately runnable. Loading the
diffed rows into the live database itself, and their descriptors/bibliography/embeddings (steps
16-24, 33), is deliberately not chained on — those are full-dataset operations against a database
that may be serving traffic, which belongs to a reviewed step, not an unattended fetch. Like the
RefSeq sweep and the annotation batch, this is built and tested but not run here — it fetches new
papers and hits external databases, which is a network-bound action this project's own rules never
start without an explicit go-ahead.

### 2026-08-14 — AlgPred2 (allergenicity) vetted, fixed, and wired: a fifth working predictor
TODO.md's last "pick a trusted tool per type first" item named allergenicity, immunogenicity and
IEDB epitopes with no candidate chosen. Allergenicity had a real one: `algpred2` (Raghava lab, the
same family as anticp2/hemopi2) is a genuine PyPI package (`pip install algpred2`) with
`Requires-Python >=3.6` and no upper bound -- no PEP-440 strict-`<=` trap like anticp2's.

Building it hit exactly the toxinpred3 shape of failure: its shipped RandomForest pickle
(`algpred2/model/rf_model`, trained with scikit-learn **0.19.2** -- eight years old) fails against
any current scikit-learn with the identical error toxinpred3 gives ("node array from the pickle has
an incompatible dtype": the pickle predates the `missing_go_to_left` Tree field 1.3 introduced).
Unlike toxinpred3, this one turned out to be fixable rather than exhaustively blocked:
`scikit-learn==1.2.2` loads it (with a harmless "unpickling from version 0.19.2" warning) -- but
only once `numpy<2` is *also* pinned, because 1.2.2's compiled Cython extensions were built against
numpy 1.x's C ABI and raise `numpy.dtype size changed` against numpy 2.x on their own. Both pins
together: real predictions confirmed through the actual `AlgPred2Predictor` class and through
`enrich_with_predictors` end to end (`algpred2_score=0.434` for the same magainin-like AMP the
other four predictors were verified against). `environments/sample-algpred2.yml` records both pins
and the version search that found them, so the next person doesn't have to repeat six attempts to
rediscover it.

This is the reason toxinpred3's "genuinely blocked" conclusion couldn't be assumed to generalize to
every Raghava-lab tool sharing the same failure signature: the *symptom* (incompatible pickle dtype)
is the same across old scikit-learn-trained models, but the *fix* depends on which scikit-learn
version the model was actually trained with, and that is only discoverable by trying versions
against the real pickle -- toxinpred3's happened to predate every version tried (1.1.3-1.6.1);
algpred2's did not. `sample.peptide_annotations.PREDICTOR_FIELDS` gained `algpred2_score` /
`algpred2_is_allergen`; `sample/peptide_predictors.py`'s module docstring, which said "HemoPI and
AntiCP are Raghava-lab web servers with no installable package" (stale even before this -- both
were already vetted PyPI packages), was corrected.

### 2026-08-14 — DIAMOND/MMseqs2 investigated head-to-head against real blastp; not wired
TODO.md's last item named DIAMOND/MMseqs2 as blocked on "sizing the install." That turned out to
be a non-issue -- `diamond` is 3 MB, `mmseqs2` 127 MB, both ordinary bioconda packages -- so the
real question worth answering was the one the sizing framing had been standing in for: would either
actually improve on the `blastp` search mode already wired at `/search/blast`?

Installed `diamond` for real (into the shared `sample` env, then removed afterward -- this was a
vetting run, not a permanent addition) and built a database from the actual release FASTA (26,358
sequences). Two things worth testing, both against real data:

- **Short-query sensitivity** -- this database's peptides are often under 15 residues, and BLAST's
  E-value significance model is known to struggle with very short queries. A 6-residue query
  returned **zero hits from blastp, diamond's default mode, and diamond `--ultra-sensitive`
  alike**. Not a `blastp`-specific gap: every local-alignment tool tested shares the same
  statistical-significance floor for a match that short. (This database's actual answer to
  short/exact-sequence lookup is `sample.families`' near-identity PepMatch search, which does not
  depend on E-value significance at all -- the right tool for this problem was already built,
  just not this one.)
- **Realistic-length queries and speed** -- at 23 residues, diamond and blastp returned
  comparably-scored top hits (both correctly finding 100%-identity duplicates of a real cataloged
  AMP) in comparable time, ~0.4s either way -- matching `/search/blast`'s own docstring that blastp
  is already "well under a second" against this corpus. DIAMOND's actual selling point (throughput
  at millions-to-billions of sequences) does not have anything to win against at this corpus's
  scale (tens of thousands).

Closed as investigated, not deferred: building a second, redundant search mode has real ongoing
cost (an API endpoint, an env dependency, tests to maintain) that direct testing did not find a
capability reason to pay. `TODO.md`, `ROADMAP.md` updated to reflect a real decision instead of an
unresolved sizing question. Reopen only with a concrete new reason -- a corpus size where blastp
stops being fast, or a specific DIAMOND/MMseqs2 mode this test didn't try.

### 2026-08-14 — Expression-yield models (RP3Net/MPEPE/TLCP-EPE) vetted: three different blockers
TODO.md's expression-yield item bundled RP3Net, TLCP-EPE and MPEPE under one "needs verification"
line, as if the same investigation would clear all three. It doesn't -- each is blocked for a
different, specific reason, found by actually checking each rather than treating the group as one
task:

- **RP3Net** is real and installable (`pip install RP3Net`, MIT, published with a 97-construct
  holdout AUROC of 0.83) -- but its checkpoint (`rp3net_v0.1_d.ckpt`, EBI FTP) is **2.4 GB**. That
  crosses this project's own large-download line -- the same rule gating the RefSeq sweep -- so it
  is not fetched without an explicit go-ahead, even though building and verifying it afterward
  would be exactly the same discipline the annotator's five predictors already went through.
- **MPEPE** has real, public code (`github.com/BRITian/MPEPE`, the paper's own repo) -- but it is
  written for **Python 2.7.15**, end-of-life since January 2020. Buildable (conda can still supply
  a python=2.7 interpreter), but running new production infrastructure on a dead language runtime
  is a real risk to weigh deliberately, not a quick pin fix like anticp2/algpred2's Python-version
  traps were.
- **TLCP-EPE** is real and published (CaLM + ProtT5 fine-tuned with LoRA, fused through a
  BiGRU-MLP head; AUC 0.835 codon-level / 0.713 protein-level on two independent test sets) -- but
  no public code repository was found anywhere. Not vettable further without the authors releasing
  it; reimplementing a transfer-learning architecture from a paper's description and validating it
  against the paper's own benchmark numbers is a research commitment, not a tool-install task, and
  was not attempted.

`TODO.md`/`ROADMAP.md` updated to name these three separately instead of one shared line, so a
future pass can act on RP3Net (once a go-ahead is given for its checkpoint) without re-deriving
that MPEPE and TLCP-EPE need a different kind of decision first.

### 2026-08-14 — MPEPE actually wired, same day: verification succeeded, so it was built, not left
The entry above stopped at "a real risk to weigh deliberately" for MPEPE's Python 2.7 stack. That
weighing happened the same session: built the environment for real, ran MPEPE's own bundled
example (`Example/high_expression_seq.fa`/`low_expression_seq.fa`) end to end, and it reproduced
the README's published output numbers exactly, cleanly separating the high- (AVE 0.74-0.85) and
low-expression (AVE 0.09-0.46) sets. The stack genuinely works today. Two things resolved that
verification surfaced, neither in MPEPE's own documentation:

- **conda-forge's `python=2.7` ships a broken `pip`** (`ImportError: No module named
  pip._internal.cli.main`) -- fixed by bootstrapping the official legacy pip via
  `bootstrap.pypa.io/pip/2.7/get-pip.py` (frozen at 20.3.4, pip's last Python-2 release).
- **The prediction script imports `pandas`**, which the README's dependency list omits entirely.

With the tool confirmed genuinely correct, the remaining question was integration, not further
vetting -- `sample.expression`'s own module docstring already named MPEPE as one of three planned
successors to the CAI-band `yield_estimate`, and its input (coding DNA) is exactly
`build_construct`'s own `"dna"` field, so the fit was already designed for, just unbuilt.

**A real, previously-unknown upstream bug found while wiring it, not before**: scoring one
construct at a time (the Expression Wizard's actual use case, as opposed to MPEPE's own bundled
6-sequence example files) failed silently -- `available: True`, no error, just no result. Traced
to `MPEPE_pred.codonc4.py`'s `name_seq()`: for a **single**-sequence FASTA it collapses
`name_list`/`seq_list` from lists to scalar strings (`if len(name_list) == 1: name_list, seq_list
= name_list[0], seq_list[0]`), and the caller's `for name, seq in zip(name_list, seq_list):`
then iterates the two STRINGS character by character. Each individual base then fails MPEPE's own
length-3 codon-triplet check in `nucl2codon()`, which prints `"Your sequence is not a cds!!!"` and
calls **bare `sys.exit()`** -- exit code 0, indistinguishable from success to any caller checking
the exit code, which is exactly why `subprocess.run(..., check=True)` did not catch it. Confirmed
by hand: a 2-sequence FASTA (the real construct plus a throwaway duplicate) produces the correct
score for both records. `mpepe_yield_estimate` now always submits a second dummy record and reads
the real one back by matching its id explicitly (not by position, in case a future MPEPE version
reorders its output) -- entirely a workaround in this project's own code; MPEPE's script itself
was not patched.

**Wired as an explicit, opt-in action, not part of the fast construct-design path**: a real call
takes ~4 minutes (TensorFlow, Theano and a 10-model ensemble loaded from disk fresh every time, no
warm server) -- the same shape of cost `fold_construct`/ColabFold already has, so
`POST /express/mpepe-yield` mirrors `/express/fold` exactly: rebuild the construct, run the heavy
step, degrade to `available=false` with a note rather than raise. The result's `note` field
discloses the Python-2.7/TensorFlow-1.15/Theano stack on every successful call too, not only on
failure -- an operational fact about what is now running in production, not a caveat to bury.
MPEPE's own repo (code + ~130 MB of pretrained weights) is cloned on first use into `data/mpepe/`
(gitignored), the same treatment as other external downloads (CycPeptMPDB, RefSeq genomes) rather
than committed to this project's history.

### 2026-08-17 — GPS-SUMO/JASSA: the motif, not the scraper
The request was to "integrate GPS-SUMO on the Expression Wizard" (with a JASSA link). Both servers
are reachable, but both are **web forms around trained models** — no API, no downloadable
predictor. Querying one per design request would make this page's correctness depend on a
third-party academic server staying up, and would hammer it at any volume. Neither is acceptable
for a feature on the construct-design path.

What is implementable honestly is the **published consensus motif**, which is citable and
deterministic: ψ-K-x-D/E (Rodriguez/Sampson 2001), the inverted [DE]-x-K-ψ (Matic et al., *Mol
Cell* 2010), the NDSM with its downstream acidic cluster (Yang et al. 2006), and the
phosphorylation-dependent PDSM (Hietakangas et al., *PNAS* 2006). `sumoylation_sites` scans for
those and **says plainly that it is a motif scan, not GPS-SUMO's or JASSA's scored prediction** —
a motif match is far more permissive than either (most matches are not modified in vivo;
accessibility and context decide), so the report links both tools for a real answer rather than
implying equivalence. That is the same line this project already draws for permeability: surface
an honest, weaker signal rather than a fabricated strong one.

**Host-aware, because the biology is.** *E. coli* has no SUMO and no Ubc9, so a motif hit there is
not a risk at all; the result reports `applicable=False` rather than an empty site list, since
"this host cannot do it" and "no sites found" are different claims. Only yeast, Pichia and human
hosts get a real scan.

**Scanned on the peptide, not the fusion** — and there is a genuine trap here worth naming: this
wizard *builds* a His-SUMO-peptide fusion, where SUMO is a cleavable tag that gets removed and
discarded. Endogenous SUMOylation is the opposite thing: an unwanted modification of a lysine in
the product. Scanning the fusion would report the tag itself (SUMO's own sequence is full of
lysines in acidic context), which would be meaningless. The report's wording calls the distinction
out so a reader does not conflate them.

### 2026-08-17 — Two anticancer sources unblocked; a Redis result cache
- **DCTPep and CancerPPD are now importable; ApInAPDB still is not.** All three were recorded as
  `importable=False` with "host path 404s" — rechecked rather than trusted, and two had genuinely
  changed. DCTPep now publishes a bulk `.xlsx` (6,107 rows, 51 columns) at
  `dctpep.cpu-bioinfor.org/downloads/`; CancerPPD publishes six id/sequence lists split by residue
  chirality (L/D/mixed) and natural/non-natural, 3,141 sequences in total. ApInAPDB's host
  (`bioinf.modares.ac.ir`) does not respond at all — the connection times out, so there is nothing
  to inspect and it stays explicitly not importable rather than quietly dropped. The combined
  table goes from 2,804 sequences (969 positives) to **7,533 (5,730 positives)**.
- **Only a verified subset of DCTPep's 51 columns is imported.** Its computed descriptors (pI,
  Boman index, aliphatic index, half-life) are deliberately left out: `sample.enrichment` already
  computes those from the sequence, and importing a second source's numbers for the same
  quantities would put two conflicting answers in the release. Identities, cross-references,
  cytotoxicity, cyclization and modifications are kept, since this database has no other source
  for them.
- **CancerPPD's real value is on its per-entry detail pages**, not in the bulk lists: the measured
  IC50 (with units), cell line, tissue, assay type and PMID live only there. So there is a
  scraper — rate-limited, resumable, writing after every page, and never run implicitly, the same
  treatment the RefSeq sweep gets, because it is ~3,100 requests against an academic server. The
  parser walks labelled cells rather than positional indexes (a new field is then a no-op, not a
  crash) and was verified against three real, structurally different live pages, including the
  case where `IC50=6.49 µg/ml` renders across three cells because the subscript is its own tag.
- **Positives only reach the release, and that was verified rather than assumed.** Every
  release-facing consumer (`consensus_frame` → features, `iter_activity_records` → the activity
  table, `sample.nlp` → the known-sequence automata) goes through `positive_sequences()`, which
  filters `label == 1`; its raw-file fallback independently skips any file whose label is not 1.
  AntiCP2's negatives exist solely in `anticancer_peptides.csv` as labelled ML hard negatives —
  DCTPep and CancerPPD contribute none, since a database of anticancer peptides does not report
  non-anticancer ones, which is a different fact from a peptide having been tested and found
  inactive.
- **A Redis result cache** (`sample.result_cache`) for the two analyses that cost minutes each:
  ColabFold folding and MPEPE's yield estimate. Seven-day TTL, matching
  `annotation_store.CACHE_TTL_SECONDS`. This is explicitly a *cache*, not a record of truth — a
  distinction this project already draws, keeping authoritative results in SQLite because Redis
  holds transient state — so an eviction, a restart or an unreachable server is a non-event: the
  caller just recomputes. It degrades completely and silently (no `redis` package, no server, a
  corrupt entry: all are cache misses), verified on this host, which has no Redis running.
  **Failures are never cached**: an uninstalled tool or a crashed run reflects the machine, not
  the input, and caching one would keep serving it for a week after the cause was fixed.

### 2026-08-17 — Expression Wizard: real construct design, not just codon optimization
A block of user-driven work on `sample/expression.py`. Each piece was checked against a primary
source before it was written, and three of the requests were pushed back on rather than built.

- **Whole-construct mRNA fold, read over the 5' region.** The old code folded only the first 45 nt
  in isolation, which structurally cannot see a hairpin the start codon forms with a complementary
  stretch further downstream — inside SUMO or a large carrier like MBP is exactly where that
  happens. Now the whole construct is folded and the window's own slice of that global structure
  is what the warning is read from. Deliberately NOT read from the whole-molecule MFE: that number
  scales with construct length, so a fixed "blocking" threshold is not comparable across carriers.
  An unmatched bracket inside the window's slice is reported as `long_range_pairing` — the
  interaction the old windowed fold was blind to. The diagram still renders only the window: a
  full MBP fusion is >1000 nt and would draw an unreadable strip.
- **Optional vector 5' UTR/RBS context.** Requested as "require the vector's UTR"; implemented as
  optional. Requiring it would break the common case (most users designing a quick construct do
  not have their vector's exact sequence to hand), and a fold of the ORF alone is still real —
  it is simply blind to the UTR/RBS interaction, which the report now says explicitly.
- **TEV as an alternative cleavable linker, and SUMO's carrier exclusion.** A request to reduce
  SUMO "to just the cleavage site" was declined: SUMO protease (Ulp1/SENP) recognizes SUMO's
  FOLDED domain, not a linear motif — that is why it leaves no scar, and a truncated SUMO would
  simply not be cleaved. The real want (a short-motif linker) is TEV: `ENLYFQ`, cleaving before
  the peptide's own first residue, sharing SUMO's one documented restriction (no proline at P1').
  Since SUMO is itself a strong solubility partner, carrier+SUMO is now rejected outright rather
  than silently built; TEV is the linker to pick when a carrier's help is actually wanted. This
  surfaced a real pre-existing bug: `ExpressionRequest.carrier` defaulted to `"sumo"`, which was
  never a valid carrier key.
- **Pichia pastoris codon usage** from the Kazusa database (137 CDS / 81,301 codons), converted to
  the same fraction-within-amino-acid form the other three hosts use, verified by checking each
  synonymous group sums to ~1.0 and that the optimizer picks the real argmax.
- **Secretion signals** (pelB, α-factor prepro, Ig-κ, IL-2), each from its primary source — pelB
  from Lei et al. 1987, α-factor from Brake et al. 1984 with the sequence taken from UniProt
  P01149 (residues 1-85: the 19-residue pre plus the pro region ending at its Kex2 KR site), IL-2
  from P60568's exact annotated Signal feature. Every entry carries its citation as a link into
  the UI, so a recommendation is checkable rather than asserted. A signal must be the ribosome's
  first product and supplies its own initiator Met, so it leads the construct and the standalone
  start Met is dropped. Host mismatches are refused, since a signal peptide is read by its own
  host's translocation machinery — with one deliberate exception: the α-factor leader is
  *S. cerevisiae*'s own sequence that P. pastoris adopted, so it is valid for both yeasts.
- **N-end-rule half-life of the RELEASED peptide**, not the fusion: every construct's own
  N-terminus is Met, so reading the rule on the fusion is constant and uninformative. Reuses
  `peptide_predictors.N_END_RULE_HALF_LIFE` rather than duplicating the table, so this and the
  ADMET panel cannot silently disagree.
- **GenBank and FASTA export** (`construct_genbank`/`construct_fasta`, `POST /express/download`).
  GenBank is written with BioPython's own writer rather than hand-formatted — the format's column
  and continuation rules are exactly the kind of thing a hand-rolled writer gets subtly wrong.
  Every component becomes a real INSDC feature (`sig_peptide` for the signal, `CDS` with the
  translation), so it opens in SnapGene/Benchling as a labelled map. Verified by round-tripping
  through a parser and checking the DNA back-translates to precisely the designed protein. FASTA
  is the plain-sequence companion for tools that take nothing else; its header carries the
  composition, since FASTA has nowhere else to put it and stripping the header would otherwise
  lose what the parts were.
- **Structure downloads.** The Expression Wizard keeps the ColabFold PDB it already fetched so it
  can be saved without re-folding (minutes of GPU time), and the peptide page gained
  `GET /search/peptide/{id}/pdb` for the *folded* structure — deliberately distinct from the
  existing `/sdf` route, which serves the RDKit-embedded conformer the viewer draws. The PDB is
  never generated on demand (folding is not a request-time operation) and the button only appears
  when a structure actually exists, rather than linking to a 404. 32,284 peptides currently have
  one.
- **2D construct map** as inline SVG, parts to scale by nucleotide span, in the same palette as
  the 3D viewer; exportable as SVG/PNG/PDF. The PDF is a hand-written minimal single-image file
  rather than a new JS dependency, consistent with this page now vendoring its JS.
- **3Dmol.js vendored locally** (`sample/static/js/`), which fixed two real bugs at once. The CDN
  URL in both `expression.html` and `peptide_detail.html` 404s outright — cdnjs paths are
  case-sensitive and the real one is `/3Dmol/`, not `/3dmol/` — which is the `$3Dmol is not
  defined` error seen in the browser; a local copy also keeps both pages working on a
  network-restricted host. Vendoring then exposed the second: `peptide_detail.html` called
  `$(element)` and `$.get`, which only ever worked because 3Dmol's 1.x bundle shipped jQuery
  inside it. 3Dmol 2.x dropped that, and this project loads no jQuery of its own, so that page is
  now written jQuery-free (`createViewer` takes a plain element; `fetch` replaces `$.get`).

**Not built, with reasons.** Codon *harmonization* by probabilistic sampling needs each peptide's
native source organism's usage profile, which this database mostly does not have (venom/skin
peptides from hundreds of species) — implementing it would mean inventing a reference table. A
joint codon/restriction multi-objective optimizer was declined as a large combinatorial change for
marginal gain, since the existing removal already picks the next-best synonymous codon to minimise
CAI disruption. A "kinetic penalty" for bulky residues at SUMO's P1' was declined outright: the
proline exclusion is documented, but a quantified penalty for Val/Ile is not something with a
citable source, and inventing the number is exactly what this project's rules forbid.

### 2026-08-13 — ML validation rigor, per-assay regression, two new tools, and structure analysis
A large session. The through-line: several capabilities that existed as unwired scaffolding were
finished and wired in, several new tools were built, and two real pre-existing bugs surfaced.

- **`sample.ml_models.train_all` was fully written but never called.** The CLI's `train-ml`
  (step 33) ran `sample.ml`'s single pooled Random Forest instead. `train-ml` now calls
  `train_all`, which trains a matrix of (task × feature-block) models with: clustering-based
  grouped train/test splitting (`sample.families.cluster` at 90% identity, so an analogue of a
  held-out peptide cannot leak into training — `pd.factorize` on the raw sequence only catches
  exact duplicates), cross-validated model selection, holdout as the final word, y-randomization,
  a k-NN applicability domain, and a Shapiro-Wilk-informed choice between Welch's t-test and
  Kruskal-Wallis for the real-vs-null significance gate. Each kept model also writes a
  `<block>.dataset.json` (full validation report + the exact train/test peptide ids), surfaced on
  a new per-model "dataset" webpage. `validate-ml` (step 35) now summarizes that report rather than
  re-validating. Applicability domain is plumbed through `model_registry.load_model` →
  `inference.predict` → the Celery task → `prediction_store` (with a backward-compatible column
  migration) → an "outside domain" warning badge on the results page; a peptide outside the domain
  is still scored, only warned.
- **Regression tasks are now real.** `REGRESSION_ASSAYS`/`learning_task_for` existed but
  `build_tasks` hardcoded classification everywhere. It now emits a `.../mic/regression` model
  alongside the classifier for every potency assay, using R²/pearson/spearman/mae/mape and a
  separate R²-based quality gate. Regression cannot reuse the "peptide vs decoy" framing every
  classification task shares — a shuffled decoy has no measured potency — so those tasks are real
  peptides only, `y` the log10 potency. `sample.synergy` likewise now trains a FICI **regressor**
  next to its classifier, and scores the regressor as a classifier at the 0.5 cutoff so the two are
  comparable.
- **Cell lines are classified via Cellosaurus, not a hand-curated guess.** `sample.cell_lines`
  resolves each `target_cell_line` to a cancer type (from Cellosaurus's disease cross-references)
  or a normal source tissue, and `build_tasks` pools cell lines by that category. This caught a
  real error in an earlier hand-drafted table: BGC-823 and Bel-7402, named as gastric/liver lines,
  are recorded by Cellosaurus as HeLa cross-contaminants (cervical) — the authoritative source was
  right and the by-name guess was wrong. Finding: even pooled by cancer type, the best category
  reaches only ~28 positives against a 60 floor, so cell-line-specific models are not trainable on
  current data — which is exactly why the anticancer-database imports matter (see KNOWN_ISSUES.md).
- **Antibiotic = Morgan fingerprint, not a one-hot column.** `sample.antibiotics` resolves an
  antibiotic name to SMILES via PubChem (cached) and computes its ECFP; `synergy.featurise` now
  uses that instead of a per-antibiotic indicator, so a synergy model can score a drug it never
  trained on by chemical similarity. Verified: ciprofloxacin↔levofloxacin (both fluoroquinolones)
  Tanimoto 0.44 vs ciprofloxacin↔gentamicin 0.14. Synergy models need a retrain to use it.
- **Two new webserver tools.** The **peptide optimizer** (`sample.optimizer`): a random mutation
  library ranked against a chosen model, on a fully client-sortable results table. The
  **Expression Wizard** (`sample.expression`, its own `sample-expression` env for ViennaRNA):
  codon optimization (real Kazusa tables for E. coli/yeast/human) → CAI → yield band, a fixed
  His6–[carrier]–SUMO–peptide fusion (SUMO is the cleavable linker before the peptide, not a
  carrier option; His is always N-terminal), a SUMO-cleavage check that flags the proline
  exception, silent restriction-site removal, selectable stop codon, a 5' mRNA arc-diagram from
  RNAfold, and a ColabFold structure of the whole construct rendered in 3Dmol.js coloured by
  component. Advanced learned yield models (RP3Net/TLCP-EPE/MPEPE) are recorded as the planned
  successor behind `yield_estimate`, not integrated blindly — same discipline as the annotator's
  external predictors.
- **Structure-derived analysis on the peptide page.** `sample.structure_analysis` reads the
  ColabFold PDB for DSSP secondary structure (via `mkdssp`, now in `sample-web`), the Ramachandran
  plot (phi/psi via BioPython) and pLDDT-based disorder — three pure-SVG diagrams, distinct from
  the sequence-propensity strip which stays as the no-fold fallback.
- **Real bug found: synergy models were unloadable outside their own route.** They are saved with
  `joblib.dump` but `model_registry.load_model` used plain `pickle.load` (an `UnpicklingError` on
  every one). The generic model registry now excludes the whole `synergy/` namespace — those models
  need a peptide + antibiotic-fingerprint input `inference.build_feature_frame` cannot construct,
  so they were never usable there regardless, and are served only through `sample.synergy`.
- **ColabFold corpus fold resumed** (authorized as P0), including sequences over the 400-residue
  cap. The GPU folding completed; the SDF/SMILES derivation is the remaining tail.
- **Incremental family assignment (`sample.incremental` / `sample-cli update-families`).** New
  peptides are placed into the *existing* family map instead of triggering a full
  `build_families` rebuild. Why not just rebuild: connected-component numbering is derived from the
  whole corpus, so a rebuild can renumber `SAMPLE_FAM_*` ids that `peptide_families`, the exports
  and any external citation already point at — silent id churn. The incremental path runs the same
  length-bucketed PepMatch near-identity search the batch clusterer uses (shared
  `families.bucket_parameters`/`LENGTH_BUCKETS`/`MIN_IDENTITY`, so the two agree on neighbours),
  builds a graph of edges *touching a new sequence only*, and per connected component either
  **inherits** the family its existing members already carry (majority vote, lowest-id tie-break;
  works for Pfam-anchored members too — near-identity to a curated-family peptide is evidence of
  that family without re-running hmmsearch) or **mints one** new `SAMPLE_FAM_*` id continuing past
  the current maximum, shared by all mutually-similar new members. Existing assignments are read,
  never rewritten. The deliberate limitation, documented in the module: a genuinely novel sequence
  that would only anchor on Pfam *by profile* (not by near-identity to an existing member) gets a
  fresh cluster here and is re-anchored the next time the full `build_families` runs — the price of
  stable ids and no external dependency at update time. The network-bound acquisition half (pull
  new papers, refresh external DBs) is intentionally left as a thin driver over existing pipeline
  steps, kept out of `update_families` so the assignment stays testable and side-effect-free.
- **Reference-tool benchmarking (`sample.benchmark` / `sample-cli benchmark`).** When an established
  tool answers the same question a sAMPle model does — HemoPI2 for hemolytic activity — the
  defensible comparison is both tools on the *same* held-out peptides with the *same* metric code,
  not two numbers from two papers on two different test sets. `benchmark_holdout` reconstructs a
  model's exact holdout (the `test_entry_ids` step 33 records) and its true labels by re-running the
  *same* `ml_models.build_tasks` labeling — never a second, subtly-different re-derivation — then
  scores the sAMPle model and the reference tool over those peptides and reports both metric sets
  plus the per-metric delta. Two deliberate honesty constraints: a metric undefined on a
  single-class holdout is reported as `None`, not a flattering 0.5; and an uninstalled reference
  tool yields *no* comparison (via the same `ExternalPredictor.available()` path the annotator uses)
  rather than a fabricated set of negative predictions. The report also states, in prose, that the
  holdout was withheld from the sAMPle model's *training* but is still this project's own data and
  task definition — so a win is "competitive on our data", not a universal-superiority claim. The
  `REFERENCE_TOOLS` registry currently holds only hemolytic→HemoPI2; macrel (antimicrobial) and
  toxinpred3 (toxicity) join it once their environments are built. Producing actual numbers needs
  the trained models (step 33) and the reference tool installed; the harness and its metric code are
  complete and unit-tested with the tool mocked.
- **Per-tool predictor environments, and the bug that they weren't reachable.** The annotator's
  external predictors each pin a scikit-learn/numpy (or, for AMPlify, TensorFlow) version that would
  fight the others in one shared env — the reason `sample-predictors` deliberately excludes macrel.
  But `_predictor_command` searched only that single shared env, so the dedicated `sample-macrel.yml`
  written last session was **dead weight**: the code could never find macrel installed in an env of
  its own. Fixed by giving each `ExternalPredictor` its own `env` and probing there
  (macrel→`sample-macrel`, amplify→`sample-amplify`, anticp→`sample-anticp2`,
  hemopi2→`sample-predictors`). Added the three env files (AMPlify and AntiCP2 install from their
  GitHub repos, not PyPI, since neither has a real PyPI distribution; AntiCP2 pins the scikit-learn
  its pickles need) plus a `sample-toxinpred3.yml` with the `1.3.1` pin toxinpred3 requires. The env
  files and `env-*` targets are complete; each still needs a *build* to confirm the install and
  entry point, and toxinpred3 additionally needs an adapter (deferred until its real CLI output can
  be parsed against the tool, not guessed). A test now asserts each adapter probes its own env, so
  the wiring can't silently regress to one-env-for-all.
- **RefSeq genomic context (`sample.genomic_context` / `sample-cli genomic-context`).** Answers
  "which sequenced organisms carry this peptide's gene" by tBLASTn against each genome. Built to the
  owner's explicit spec: keep a hit only when it covers ≥95% of the peptide **AND** is ≥95%
  identical — the *conjunction*, because coverage-alone admits a conserved-motif fragment and
  identity-alone admits a full-length lookalike, and only both together support a gene-presence
  claim. The taxonomy of a hit is the *genome's* (its RefSeq tax id / organism name), resolved to a
  broad clade through the existing `data/taxonomy.db` `tax_domain` table, and hits are aggregated
  into a domain→genus→species hierarchy rendered as a Krona-style sunburst SVG (pure Python arc
  geometry, same family as the structure-page SVGs). The download/search loop deletes each genome
  and its BLAST database *as it goes* — peak disk is one genome plus the surviving hit rows, the
  whole reason it is a loop rather than a bulk fetch. Two deliberate separations of concern: the
  loop injects its `download` function so the accounting and delete-as-you-go cleanup are testable
  against local files (no genome is fetched in the suite), and the parse/filter/aggregate/SVG core
  is pure. What is *not* done, on purpose: running the sweep (large downloads → gated on disk/time
  sign-off; `--limit` bounds a first pass) and the per-peptide webserver page to embed the sunburst,
  which has nothing to show until a sweep produces data. This mirrors every compute-heavy step —
  the code is complete and verified, the run is a separate, explicit decision.
- **ADMET-lite (`sample.admet` / `sample-cli admet`).** A per-peptide ADMET summary that reports only
  what the sequence honestly supports. The deliberate scoping decision: full small-molecule ADMET
  suites (SwissADME, pkCSM) are calibrated on drug-like chemistry a peptide is nowhere near, so
  rather than quote their CYP/Caco-2/oral-bioavailability endpoints for a 20-mer, this computes the
  sequence-derivable physicochemistry (MW, pI, net charge, GRAVY, aromaticity, Guruprasad instability
  index, Ikai aliphatic index, Boman index, hydrophobic moment — reusing `sample.descriptors` so the
  numbers match the enrichment step's) plus standard peptide rules (instability cut at 40, the N-end
  rule for half-life, an aggregation-hotspot scan, cationicity), and lists *everything it will not
  fake* in `gaps`. Above all **membrane permeability** is always an explicit `not_integrated` gap —
  it needs a dedicated peptide predictor (candidate: PharmPapp) that is a real tool-trust decision,
  not a number to invent; hemolysis/toxicity are pointed at the annotator's predictors. Non-standard
  residues (ambiguity codes, modified residues) are stripped for the ProtParam metrics and *counted*
  in `non_standard_residues`, so a consumer knows when those values are approximate rather than
  getting a silent wrong answer or a crash.

- **Anticancer import: AntiCP2 wired after the host came back (`sample.anticancer`).** The earlier
  session recorded the anticancer DBs as blocked on unreachable academic hosts; a re-check on
  2026-08-13 found `webs.iiitd.edu.in` answering again, so AntiCP2 was imported the disciplined way —
  its download files were fetched and *inspected* (plain text, one sequence per line, labelled
  pos/neg train/test) before a parser was written, not assumed. Output is a normalised table of 2,804
  unique sequences / 969 anticancer positives + 1,835 hard negatives. The other three
  (CancerPPD/ApInAPDB/DCTPep) are left as `SOURCES` registry entries with `importable=False` and the
  reason (their URLs 404): writing a parser for a format never seen is exactly what this project's
  "null an unparseable value rather than guess" rule forbids at import scale. A negative here means
  "an AMP with no *reported* anticancer activity" (a labelled hard negative), never "proven
  inactive" — encoded as `activity=non-anticancer-amp`, `label=0`, so a later training step cannot
  mistake absence of evidence for evidence of absence. **Wired into the pipeline, not left as a
  loose CSV:** the positives enter through the two paths that actually feed `train_all` --
  `anticancer.consensus_frame` adds them to `merge_databases`' consensus (so each gets an entry id,
  structure and, downstream, descriptor/embedding features in `final_dataset.csv`), and
  `database_activities.load_anticancer` (registered in `_LOADERS` as `ANTICANCER`) emits one
  qualitative `target_group=anticancer` activity per positive into `activities.csv`. A sequence
  already present from an antimicrobial source merges into its existing consensus row rather than
  duplicating. The Makefile runs the import as `step_06_import_anticancer_databases`, before the
  merge, so a normal build integrates it with no manual step. This is why the integration is safe to
  wire on by default: with no import present, `consensus_frame`/`load_anticancer` yield nothing and
  the pipeline is byte-for-byte unchanged.
- **`train_all` validated on the real dataset (bounded smoke run).** It had only ever run on
  synthetic fixtures; a `--tasks-limit 2` run over the real 454 MB / 74,084-row dataset confirmed the
  whole path works end to end — Cellosaurus classified 78/104 real cell-line names, the ≥90%-identity
  PepMatch clustering split ran over 37,042 real peptides, and the quality gate + normality-gated
  null test fired correctly (descriptors→Kruskal, aa_composition→Welch t-test, y-randomisation at
  chance ~0.50, both blocks *dropped* at held-out ROC-AUC ~0.73 < 0.75). This de-risks the full run
  without spending its hours; the full release run remains a separate, explicit go-ahead.

### 2026-08-12 — Annotator scoping found MacrelPredictor had never actually worked, and cost a self-inflicted environment mistake
Starting on TODO.md's P2 "annotator" item required first checking which of its named tools
(anticp2, amplify, macrel, pfam, hemopi, toxinpred) are real and installed. Findings, each checked
against the live tool rather than assumed:

- **`MacrelPredictor` (`sample/peptide_predictors.py`) had two bugs that meant it never produced a
  real prediction, despite `available()` reporting the tool usable.** (1) It did not pass
  `--keep-negatives`, so macrel silently dropped every peptide it did not call an AMP from the
  output file entirely -- verified live: a D/E-only negative control was simply absent, no
  `is_AMP` column at all, next to a real AMP candidate that did appear. (2) Its output-file filter
  (`.tsv`/`.tsv.gz`) never matched macrel's real filename, `<tag>.prediction.gz` -- so even a
  positive call was never parsed. Fixed both; re-verified end to end
  (`GIGKFLHSAKKFGKAFVGEIMNS` -> AMP, 0.950; a D/E-only control -> not-AMP, 0.455), and added
  `tests/test_peptide_predictors.py`, the first tests this adapter ever had -- hermetic, mocking
  `subprocess.run` to write output in macrel's real, verified shape, so they run without the real
  binary installed.
- **The `amplify` PyPI package is not AMPlify.** `pip install amplify` (that literal name) installs
  Fixstars' quantum-annealing optimization SDK (`amplify.fixstars.com`), unrelated to the AMP
  prediction tool this project means. Installed it while checking availability, discovered the
  collision when the resulting binary was a `.so` extension module with no CLI at all, and
  immediately removed it. The real AMPlify (bcgsc) is not on PyPI under any name checked.
- **`toxinpred3`, already installed, cannot actually run.** `toxinpred3 -i ... -m 1` fails loading
  its own shipped model: `ValueError: node array from the pickle has an incompatible dtype` --
  the pickle predates the `missing_go_to_left` field the installed scikit-learn 1.3.1's
  `_tree.pyx` expects, so it was built with an older scikit-learn than the one it is now paired
  with. Not something to silently patch around (guessing at a compatible scikit-learn version
  risks breaking hemopi2's own 1.3.1 pin in the same environment); recorded in `KNOWN_ISSUES.md`
  rather than force-fixed.
- **`anticp2` has no PyPI distribution under that name at all** (`pip index versions anticp2`:
  no matching distribution). Raghava-lab tools are inconsistently packaged; this one is not.

**Self-inflicted mistake, caught and reverted the same session.** Installed `macrel` into
`sample-predictors` to test the fix above without first reading
`environments/sample-predictors.yml`, which already explains -- in a comment written before this
session -- that macrel is deliberately *not* installed there: it "pulls its own pinned
scikit-learn and a conflicting numpy; needs a third environment of its own rather than a pin fight
inside this one." Checked for actual damage before reverting: `scikit-learn` stayed at 1.3.1,
`numpy` at 1.26.4, and `hemopi2` still produced a correct prediction (0.72, Hemolytic, for a known
sequence) -- no version conflict actually occurred this time. Reverted anyway
(`pip uninstall macrel`): a prior, explained, deliberate architectural decision does not stop
being right just because violating it happened not to break anything on one particular run, and
the file's own reasoning is about avoiding future pin fragility, not this one install. The lesson
worth keeping generally, not just here: read the target environment's own file before installing
into it -- it may already explain a constraint that is not visible from the code alone. The
`MacrelPredictor` bug fixes stand regardless of which environment eventually runs it live; see
`KNOWN_ISSUES.md` for what a `sample-macrel` environment would need.

### 2026-08-12 — Pipeline renumbered to a flat 1-40; found and fixed several pre-existing stale step-number references in the process
The letter-suffixed steps this session added (19b, 19c, 22b, 23b-23f) made the sequence read
1..19, 19b, 19c, 20, 21, 22, 22b, 23, 23b..23f, 24..32 -- workable, but the third such insertion
point since the 2026-08-08 flat-1-32 renumbering (see that entry below). Flattened again: steps
1-19 unchanged, everything from the old 19b onward renumbered sequentially to 20-40. Full mapping
in the Makefile's step-target section; the environment-to-step-range comments at the top of the
Makefile and in `environments/README.md` are the authoritative summary.

Mechanical scope: every `step_NN_name` Makefile target, `.PHONY`/`all:` prerequisite list, and
`run_step` call's logging label, plus every prose "step NN" / "steps NN-MM" reference found across
`sample/`, `tests/`, `environments/*.yml` and the top-level docs (`README.md`,
`environments/README.md`) -- done with a script mapping old id to new id, not by hand, since the
old scheme alone had ~380 references across 58 files.

The mechanical pass surfaced something worth recording on its own: a non-trivial number of "step
NN" references were **already wrong before this renumbering touched them** -- comments that said
"step 23" meaning ML model training (correctly step 25 in the immediately-prior scheme, now 33),
or "step 22" meaning the SQLite load (correctly step 24 then, now 32), scattered across
`sample/db.py`, `sample/model_registry.py`, `sample/serialize.py`, `sample/api/app.py`,
`sample/api/v1.py`, `sample/templates/models.html`, `environments/sample-ml.yml`,
`environments/sample-predictors.yml`, `environments/README.md`, and three test files. A purely
mechanical old-number-to-new-number script cannot fix a reference that was already pointing at the
wrong step -- translating a wrong number just produces a differently wrong number -- so each one
found this way was checked against what the code actually does and corrected by hand, not
mechanically re-mapped. `Makefile`'s own environment-range comments
(`# Steps 6-14 and 20-23: ...`) had the same problem for a different reason: they were never
updated when this session's earlier additions (22b, 23b-23f) put more steps into
`sample-extraction` than the comment described, so the comment was stale in *scope*, not just
number. Rewritten from the actual `ENV_EXTRACTION`/`ENV_CHEM`/`ENV_ML`/`ENV_WEB` assignments in
each step target, not copied from the previous (also-wrong) comment.

Also removed `docs/_build/pdf/sAMPle.pdf`: a stale, orphaned PDF (Aug 7) with no Makefile target
that produces it, sitting beside the real one (`docs/_build/latex/sAMPle.pdf`, Aug 11, produced by
`make docs-pdf` via Sphinx + Tectonic) -- two PDFs in the tree with no way to tell which was current
without checking timestamps by hand.

Verified: `make -n` on the first and last step target (`step_01`, `step_40`) resolves correctly
with `END_STEP=40`; all 40 step targets present; full suite still 925 passing (this was a
documentation-and-Makefile-only change, so an unchanged pass count is the expected confirmation,
not a coincidence).

### 2026-08-12 — Review pass found the GO slice was silently presenting 103 obsolete terms as current
A `/goal review all source-code comments, typing, arguments, documentation, unit tests` pass over
this session's own additions re-examined `sample/gene_ontology.py`'s `build_go_slice` (see the
entry below on the feature itself) and found `parse_obo` already extracted `is_obsolete` per term
but `build_go_slice` never read it -- an obsolete GO term resolved exactly like a current one, with
nothing in the output `.obo` or the summary to say otherwise. Checked against the real data rather
than left as a hypothetical: **103 of the 1,600 harvested GO ids (6.4%) resolve to obsolete terms**,
and 102 of those 103 carry no `is_a`/`part_of` at all in go-basic (an obsolete term's hierarchy is
stripped on retirement) -- meaning they were landing in the slice as disconnected orphan nodes,
directly contradicting the module's own stated goal of "a connected sub-hierarchy... a term with no
path to a root is not usably placed in an ontology browser."

Fixed by writing `is_obsolete: true` into the term's stanza -- go-basic.obo's own convention for
exactly this, not a workaround -- rather than dropping the term: CAMP really did annotate peptides
with it at scrape time, and that is a real fact worth keeping, just one that needs to be labelled
rather than presented as current. `resolved_direct`/`resolved_via_alt_id` still count it as
resolved (an obsolete term is not the same claim as an absent one, and conflating them would make
the resolution-rate number -- the check for a scraping regression -- harder to read), with a new
`obsolete_used_terms` field in the summary so the split stays visible. Verified against the real
corpus: the regenerated slice still resolves 1,600/1,600 with the same 0 unresolved, now reporting
103 obsolete, and `grep -c "is_obsolete: true"` on the regenerated file confirms all 103 are marked.

### 2026-08-12 — Embedding search: two corrections to the original plan, both from measuring rather than assuming
TODO.md's line for this read "Embedding search: ESM-2 on GPU → ChromaDB → Celery job (nothing
imports `chromadb` yet)". Built it, and two parts of that framing turned out to be wrong once
measured:

- **"On GPU" implied it needed one.** `facebook/esm2_t6_8M_UR50D` (ESM-2's smallest checkpoint,
  already used elsewhere in this pipeline for the analytics UMAP chart) benchmarked at 8 ms/
  sequence on this host's CPU alone over 500 real peptides. The released corpus is 37,042 real
  peptides -- not 74,084, which is that count *plus* an equal number of decoys (`peptides.is_decoy`
  breaks down as 37,042/37,042 in the live database; the earlier "74,084 peptides" figure recorded
  elsewhere this session was the combined total, not the indexable count) -- so a full CPU-only
  build is under 5 minutes. The real build, run for real against `sample-chem`'s
  `torch==2.5.1+cu121`, did find a GPU on this host despite `sample-ml`'s separate torch build
  reporting `cuda.is_available() == False` for an old-driver reason recorded in that environment's
  own comment -- and finished in under two minutes. Either way, nowhere near the multi-hour/day GPU
  jobs (ColabFold, RefSeq) this project's go-ahead rule exists for, so building the index needed no
  go-ahead, the same reasoning already applied to the CAMP/ADP/DBAASP downloads this session.
- **"Celery job" assumed the query itself would be slow.** It is not: embedding one query sequence
  is a single forward pass through an 8M-parameter model, and a warm-process query (model already
  cached in-process, the normal state of a running web server) measured at ~20 ms end-to-end,
  including the Chroma k-NN lookup over all 37,042 indexed vectors. `/api/v1/search/embedding` is
  synchronous, exactly like `/search/blast` and `/search/hmmer` before it, for the same reason
  documented at each: queueing something this fast would add a poll cycle for no benefit.

Implementation: `sample/embeddings.py` (`build_embedding_index`, `run_embedding_search`,
`embedding_index_path`), wired into `sample/api/common.py` (re-exported, not duplicated -- unlike
BLAST/HMMER it wraps no external binary via subprocess, so the implementation lives in the domain
module rather than in `common.py` directly), the REST endpoint (`/api/v1/search/embedding`) and web
form ("Embedding match"), `sample-cli build-embedding-index` as pipeline step 21 (after step 20's
validated `final_dataset.csv`, which is what carries `is_decoy` -- `merged_from_database.csv` and
`enriched.csv` do not, and indexing against either would have risked a decoy coming back as though
it were a real peptide). `chromadb` added to `sample-chem.yml` (builds the index) and
`sample-web.yml` (serves queries against it) -- the two environments that actually need it, per
this project's rule on declaring a dependency in the specific environment(s) that run it.

Verified against real data: the actual index was built for the full real corpus (37,042 peptides,
confirmed zero decoys present by direct lookup), and a live REST + web-form query against it
(`GIGKFLHSAKKFGKAFVGEIMNS`) returned sensible nearest neighbors (top hit similarity 0.969) with the
result rendered correctly in both the OpenAPI-documented endpoint and the search results table.
Permanent tests mock `sample.esm_enrichment.compute_esm2_embeddings` -- the boundary a real model
load would otherwise cross, including a network call on a cold cache -- the same "mock the
boundary" policy `tests/conftest.py` already documents for PubMed/Ollama/UniProt; what they check
is this project's own logic (decoy exclusion, Chroma wiring, ranking), not ESM-2's or Chroma's own
correctness, which is not this project's to verify.

### 2026-08-12 — ATCC product links read from `target_organism` at display time, no new stored column
DBAASP's raw `targetSpecies.name` ("Staphylococcus aureus ATCC 29213") is stored as-is in
`activities.target_organism` (`sample/serialize.py`'s DBAASP block never resolves it down to a
binomial the way `sample.database_activities.resolve_target_species` does for the *other*
activity-ingestion path) -- confirmed against the live release: 59,904 of 174,163 activities
(34.4%) carry a catalog number this way. `sample.nlp.extract_atcc_id` reads it back out at request
time and `sample.external_links.build_url("atcc", ...)` builds the product URL
(`https://www.atcc.org/products/{id}`); `sample.db._with_taxonomy_link` (extended, not duplicated
-- it already existed for the NCBI Taxonomy link on the same row) sets both
`target_taxonomy_url` and the new `target_atcc_url`. No new database column or pipeline rebuild:
the source string was already there, and a second stored copy of the same fact would just be
another place for the two to disagree -- the same reasoning DECISIONS.md's SMILES-validation entry
already applied to computed-not-stored values.

Verified against the live site rather than assumed: fetched a real product page
(`.../products/25922`) and confirmed it identifies the exact strain ("FDA strain Seattle 1946"),
and fetched a nonexistent catalog number (`.../products/99999999`) and got a genuine 404 -- not
the byte-identical catch-all page APD6 serves for every id, the exact failure mode
`sample/external_links.py`'s own module docstring warns about. Three ATCC numbering series appear
in the real data: plain digits, and two prefixed series -- BAA (Bacteriology and Applied
microbiology Accession) and MYA (mycology) -- each confirmed live against its own product page
(`baa-1605` -> the correct *Acinetobacter baumannii* strain; `mya-4646` -> the correct *Candida
parapsilosis* strain). BAA allows shorter numbers than the plain series (`baa-42` is real, 2
digits), so only the plain series keeps a 3-digit floor.

Extraction rate against the real 59,904 ATCC-qualified activities: 99.80% (59,782) resolve to a
link. The remaining 118 (0.20%, 11 distinct strings) are not left unlinked by a gap in the
regex -- they are not ATCC catalog numbers at all: OCR-style corruption in the source text
("MY A-422" with a stray space, "MAY-646" for "MYA-646"), and internal strain nicknames a
different collection uses ("ATCC Cj1", "ATCC UA159", "ATCC H9812") that happen to sit next to the
word "ATCC" in DBAASP's text without being one of its numbering series. Forcing a match on these
would produce a wrong link with the same visual confidence as a right one -- left unlinked per
this file's "prefer nulling an unparseable value over guessing" rule.

### 2026-08-12 — GO `.obo` slice maps SAMPLE's harvested terms onto the real Gene Ontology, not an invented family hierarchy
The original plan for this item (see the plan file this session generated for the TODO.md
restructure) read "`.obo` ontology for SAMPLE's own Pfam-curated + clustered family hierarchy" and
was built that way first -- a made-up `SAMPLE:NNNNNNN`-numbered vocabulary with two flat branches
(Pfam families, size-≥5 clusters) under an invented root. The user corrected this mid-session: the
`.obo` has to map onto the real Gene Ontology, using the GO terms already harvested from CAMP
(`sample/gene_ontology.py`, `data/gene_ontology.json`, 41,606 terms / 6,016 peptides), not
substitute a project-invented ontology for a real one. The family-hierarchy version
(`sample/family_ontology.py` plus its CLI/Makefile wiring) was removed rather than kept alongside
the corrected version -- it answered a question nobody asked once the real intent was clear.

Built instead: `sample.gene_ontology.download_go_obo` fetches the official `go-basic.obo`
(`https://current.geneontology.org/ontology/go-basic.obo`, the acyclic "basic" edition -- deliberately
not the full `go.obo`, which adds `regulates`-family relations that can cycle) to
`data/gene_ontology/go-basic.obo`. `build_go_slice` looks up every one of this project's harvested
GO ids in it, walks each resolved term's `is_a`/`part_of` ancestor chain up to the ontology roots,
and writes the result to `data/gene_ontology/sample_go_slice.obo` -- a connected sub-hierarchy of
the *real* GO restricted to what this dataset actually annotates, not a flat list and not an
invented vocabulary. `alt_id` resolution handles a harvested id that GO has since merged into
another.

Run against the real data: 1,600 unique harvested GO ids, 1,592 resolved directly and 8 via
`alt_id` -- **zero unresolved**, which is itself informative: it means CAMP's scraped GO ids
(harvested by regex from unstructured text, see that module's own docstring) are all genuine,
current GO accessions, a validation of the harvester this project did not previously have. Pulling
in ancestors added 2,091 terms to the 1,595 directly used, for 3,686 total -- e.g. `GO:0042742`
"defense response to bacterium" (3,148 peptides) correctly resolves `is_a` to both "defense
response" and "response to bacterium". `sample-cli build-go-obo` wires this in as pipeline step
23f (`Makefile`), between the GO harvest (23e) and the SQLite load (24) -- it depends on 23e's
output and nothing downstream depends on it, so its placement does not gate the release.

The raw `go-basic.obo` (~32 MB, one file, downloaded from a stable canonical URL, not a
multi-hour/day job or a hundred-GB dataset) did not need a go-ahead under this project's rule for
long-running or destructive actions -- it is the same class of operation as the CAMP/ADP/DBAASP
downloads already performed autonomously this session. `.obo` was added to `.gitignore` alongside
the existing `*.json`/`*.csv`/`*.fasta` pipeline-output patterns so neither the raw download nor
the derived slice is ever accidentally committed.

Verified three ways: unit tests against a small synthetic `.obo` fixture (`is_a` chains, a
`part_of` relationship, an `alt_id`, an obsolete term, a `[Typedef]` stanza that must not be
mistaken for a `[Term]`) rather than the real 32 MB file, since the suite runs offline
(`tests/conftest.py`'s `no_network`); a real end-to-end run producing the 1,600/0-unresolved/3,686
numbers above; and the CLI command run directly against the real database and inspected by hand.

### 2026-08-12 — The remaining 19 silent-exception sites: fixed or confirmed already safe
Closed out the AST scan's full list (see `KNOWN_ISSUES.md`, now resolved). None turned out to be
another 60%-activity-table-scale loss -- these were mostly single-item fallbacks, as suspected --
but "probably fine" was still just a guess before this pass, and two things were worth finding:

- **`sample/merge.py`'s consensus-building loop** (distinct from the four per-source ingest loops
  already fixed earlier this session) silently dropped every sequence RDKit and its HELM fallback
  both failed to parse, with only the final `entry_count` visible -- no count of how many were
  dropped or why. Same shape as the DBAASP incident, just smaller in practice. Now counts
  `dropped_x_residue` and `dropped_unparseable` separately and logs both alongside the existing
  success count. `merge_database_and_extractions`'s SAMPLE-literature structure-resolution loop had
  the identical gap and got the identical fix.
- **Two sites in `sample/validation_extras.py` were not actually bugs.** `validate_error_injection`
  turns a crashed `_contract_rejects` call into a conservative counted outcome -- treated as a
  false positive or a rejection, which is the safe direction to fail in a suite whose whole job is
  measuring detection -- rather than dropping the trial. The AST scan flags any `except Exception`
  wrapping a loop body regardless of what happens inside it, so this needed a human read to confirm
  rather than a mechanical fix; added a debug line so a spike is traceable to "the checker crashed"
  without changing the (correct) fail-safe behavior.

Two descriptor-computation functions (`sample.enrichment.peptide_descriptors`,
`sample.peptide_predictors.compute_peptides_descriptors`) each had a per-property try/except
already justified by a comment -- one non-standard residue defeating an instability-index lookup
must not cost every other descriptor for that sequence. That reasoning holds; what was missing was
any visibility into *how often* it fires. Both gained an optional `failure_counts: Dict[str, int]`
parameter, incremented per failing property and left `None` (no tracking, unchanged behavior) for
any caller that does not pass one. The one real caller of each (`enrich_with_descriptors`,
`enrich_with_predictors`) now passes a shared dict and logs one summary per run instead of per
sequence.

Everything else in the list (`sample/analysis.py`, `sample/databases.py`, `sample/decoys.py`,
`sample/engine.py`, `sample/llm_validator.py`, `sample/pdf.py`, `sample/serialize.py`'s CAMP
activity block) got the same treatment already established for the `merge.py`/
`database_activities.py` fixes: a counter incremented in the `except`, and one summary `logger`
call instead of the loop-body swallow. Verified with `py_compile` on every touched file, the full
suite (836 passing, up from 830 before this pass -- the 6 new tests cover the two functions with
genuinely new logic: the `failure_counts` threading and the consensus-build drop counters), and,
for the two threaded-counter functions, a forced-failure test that confirms the counter increments
without changing what the caller gets back.

### 2026-08-12 — Admin edits made durable through a replayed-overrides table, not by teaching the pipeline about `peptides`
`/admin`'s entry-edit form used to write straight to `peptides`, which steps 22-32 drop and
rebuild from the CSVs on every run -- so a correction looked like it took effect and then reverted
at the next release with nothing in the log to say why. Rejected the alternative of making the
loader read-merge against the CSV row (`INSERT OR REPLACE` skip logic keyed on entry_id): that
would need every one of the loader's several insert sites to know about overrides, and a mistake
in any one of them fails silently in exactly the way this project keeps finding elsewhere. Instead
`PeptideOverride` (`sample/models.py`) is a new, independently-lifecycled table -- never dropped by
`load_sql_database` -- and `sample.serialize._apply_peptide_overrides` reapplies every row in it
onto the freshly rebuilt `peptides` table in one pass, right after the main load loop, before
bibliography/families/etc. are attached. The admin route (`sample/api/admin.py`) writes both the
live `Peptide` row (so the change is visible immediately) and the override row (so it survives),
through `sample.admin.record_override`, which re-validates the field name against
`EDITABLE_PEPTIDE_FIELDS` before ever writing it -- that table's rows are later replayed as a raw
SQL column name, so an unchecked field there would be a SQL-injection path, not just a
data-quality one.

`EDITABLE_PEPTIDE_FIELDS` moved out of `sample/api/admin.py` into `sample/config.py` rather than
`sample/admin.py`, specifically so `sample/serialize.py` -- run in `sample-chem`, which does not
carry SQLAlchemy (confirmed: `python -c "import sqlalchemy"` fails in that environment) -- can
import the one allow-list instead of maintaining a second copy that could drift from it, or
pulling the ORM into a pipeline stage that has never needed it. This is the same import-scoping
mistake the `pyahocorasick`/`sqlalchemy`/`python-multipart` environment-declaration rule above
exists to prevent, caught here before it happened rather than after.

Verified three ways, from narrowest to widest: a unit test of `_apply_peptide_overrides` against a
bare sqlite3 fixture; an end-to-end test that calls the actual `load_sql_database` step 32 runs,
seeded with an override row, confirming the correction survives a real rebuild and not just the
helper in isolation; and the admin web form through a `TestClient`, confirming the override row is
written on save, a second edit of the same field updates it in place rather than accumulating
history, and a field outside the allow-list is refused with `AdminError`.

### 2026-08-12 — HMMER wired up as a search mode, synchronous, over 519 SAMPLE-derived profiles
Both the REST endpoint (`GET /api/v1/search/hmmer`) and the web form's new "Family match (HMMER
profile scan)" option run `hmmscan` synchronously rather than through Celery, unlike the annotator
work planned for v2. Justified by scale, not by principle: 519 small profiles, pre-`hmmpress`-ed,
score a query in well under a second — closer to the existing synchronous BLAST search than to a
job that needs progress polling. Reuses the exact response shape (`family_id`, `evalue`, `score`,
`bias`) `sample/families.py` already produces for Pfam assignment, so a hit links straight to
`/search?family=<id>`, the same family page Pfam-sourced families use — one page for both kinds of
family evidence rather than a second UI for the second source. `_UNRELIABLE_FAMILY_HMMS` (see the
`SAMPLE_FAM_000001` entry above) is enforced in `run_hmm_search` itself, not filtered later in the
route or template, so no caller of the function can forget to exclude it. Verified against the
running web form (dropdown option present, results table rendered, link resolved to the correct
family) in addition to the REST endpoint and its OpenAPI listing, per this file's "verify a live
render" rule above.

### 2026-08-12 — Corpus-fold status corrected from "not started" to 86.7% done
`TODO.md` said ColabFold over the corpus needed "days of GPU time... hasn't been run" as of the
previous update. Checking `data/predicted_structures.json` directly found 32,097 of 37,042 real
peptides (86.7%) already folded with a genuine ColabFold pLDDT score, most recently dated 2026-08-11.
No fold process was running at the time of the check. This also meant the interactive-3D-structure
feature, previously listed as gated on the fold, was already working and only needed verification.

### 2026-08-12 — Analytics charts: four of five already existed; fixed the stale one, added the sixth
`TODO.md` said "the analytics page currently renders no charts." Checking the live page found four
of five already worked, including amino-acid distribution (`residue_composition.png`) — the same
kind of stale-status pattern as the ColabFold corpus-fold entry above. The one real gap:
`umap_esm2.png` was missing because the file on disk was still named `umap_protbert.png`, dated
from before ProtBERT was removed earlier this session; the code already wrote the correct
filename, it just hadn't been re-run since. Re-ran step 36 to regenerate all five with current
data, deleted the stale ProtBERT file, and added the genuinely missing sixth chart
(target-organism distribution, top 20 by activity count). That one reads a second data source
(`activities.csv`'s `target_organism`) rather than the peptide-level dataset the other five use,
since target organism is an activity-level fact — one peptide can be tested against many
organisms — not a peptide-level one.

### 2026-08-12 — Published SMILES column: replace from structure, then validate, never guess
Two passes in order of trust. First, wherever `predicted_structures.json` has a sequence-derived
structure, its SMILES — guaranteed round-trippable and connected — replaces the stored value
outright. Second, whatever is left is checked against a garbage-detector (sentinels, residue
letters leaked from an LLM pass, truncated generations, model special tokens) and nulled rather
than kept if it fails. Run over the real release: 32,097 replaced with derived structures, 41,949
already-plausible kept, only 10 rejected as unparseable, 29 already empty. The detector logic
itself is not new — it was removed as dead code earlier in the same session when step 16 became
sequence-only and stopped needing it as a *prefilter*; it turned out to still be needed as a
publication-time *validator*, a different job for the same function.

### 2026-08-12 — Profile HMMs built for all 519 SAMPLE-derived family clusters
Aligned each cluster (`mafft --amino --auto`) and built a profile (`hmmbuild --amino`) — 518 of
519 succeeded outright; the one failure was MAFFT auto-detecting a peptide cluster as nucleotide
data (A/C/G/T are valid amino-acid codes too), fixed by forcing `--amino` on MAFFT the same way
`hmmbuild` already was, not just relying on its own heuristic. Verified past "the file exists": a
built profile must actually score a genuine member above `hmmsearch`'s default threshold, checked
directly rather than assumed from a nonzero exit code.

That verification found a real, not cosmetic, limitation: `SAMPLE_FAM_000001` (2,559 members —
27% of every clustered peptide, in one cluster) does not reliably score even its own members.
Every other cluster tested does (confirmed on a 116-member and a 6-member cluster). Cluster sizes
otherwise run tight — median 8, 450 of 519 between 5-20 members — so this one cluster is a clear
outlier, not representative of profile quality generally. It is very likely the same clustering
failure mode already on record: a flat mismatch budget chaining unrelated short peptides into one
component (see the clustering entry below) rather than a genuine shared family, which is also why
a profile built from it does not converge on anything specific enough to match. Not re-clustered
this session — that is `sample/families.py`'s clustering logic, a different piece of work from
building profiles over whatever clusters it already produced. Flagged in `KNOWN_ISSUES.md` so
whatever wires these into a search mode (**C4**) excludes or special-cases it rather than
publishing a family match that means nothing.

### 2026-08-12 — Gene Ontology harvested from CAMP directly, not via Pfam/InterPro mapping
CAMP scrapes UniProt's own GO annotation into a flattened, undelimited text cell that nothing
read. A regex exploiting two small closed vocabularies in the blob (the three GO category names,
and GO evidence codes) finds term boundaries in the unstructured text reliably enough to harvest
41,606 terms across 6,016 peptides. Deliberately did not also parse CAMP's `Pfam`/`InterPro`
fields into a GO mapping: those are free text with no reliable inter-entry delimiter
(`PF16047: Antimicrobial22 PF03032 : ...`), and Pfam family membership is already served far more
reliably by the curated HMMER assignment in `peptide_families`. Re-deriving the same fact from
CAMP's noisier text would create two disagreeing sources for one thing.

### 2026-08-12 — UniProt/PDB identifiers read from `activities.csv`, not re-derived
`activities.csv`'s `cross_references` column (resolved during step 14 from the source databases'
own fields and step 9's recovered external-database hits) never reached `sample.db`: step 16
onward reads a different lineage (`merged_from_database.csv`) that does not carry it. Read
directly from the already-resolved column rather than re-deriving from the source databases,
since the parsing (`sample.external_links.normalize_cross_references`) was already validated on
the way in. 1,939 peptides gained a UniProt/PDB panel, kept separate from the PubChem/ChEMBL
"Assay Archives" panel (sourced independently through step 26) so one accession is never shown as
two independent confirmations of the same fact.

### 2026-08-11 — Bibliography resolution scoped to peptide-linked references only
`harvest_sample_literature` keys the whole downloaded corpus by DOI, taking the reference set from
9,333 to 94,564 entries. Those DOIs match no `papers.source_id`, so none of them reach a page.
Resolving all of them against CrossRef is 35,778 fetches (~2.5 h) for roughly 1,500 records anyone
could ever see. Neither "has an accession" nor "matches `papers.source_id`" is the right filter —
the first marks everything linked (the corpus harvester's accession *is* the DOI), the second
misses DBAASP's activity-level citations. What actually decides it is which harvester produced the
row. Validated by reproducing the exact 9,333-reference set the working bibliography had before
this change.

### 2026-08-11 — Sphinx documentation: apidoc is a generator, not a check
Thirteen modules were absent from `docs/sample.rst` and the docs build stayed green throughout —
`sphinx-apidoc` is run once by hand to generate the `.rst` files; it does not run as part of the
build and so cannot fail it. Re-run `sphinx-apidoc` after adding a module, or diff the documented
module set against `sample/*.py`, rather than trusting a clean build to mean nothing is missing.

Separately, `sample.models`'s SQLAlchemy type annotations pull SQLAlchemy's own docstrings into
the build through `sphinx_autodoc_typehints` — and those docstrings are written for SQLAlchemy's
own Sphinx setup, which enables the third-party `sphinx-paramlinks` extension. Without it, every
`:paramref:` role and `.. legacy::` directive in a pulled-in docstring raised a build error.
`docs/conf.py` now registers both locally rather than adding the dependency. Twelve
duplicate-object-description warnings for the generic Pydantic `Page` type remain and are not
suppressible — Sphinx emits that specific warning with no subtype to filter on.

### 2026-08-11 — Sequenceless activities: dropped, not recovered by name
Extraction yields 12,333 literature/patent activities; 8,000 (65%) have no sequence and are
dropped at merge, since the dataset is keyed on sequence. All 8,000 carry a name, which makes
name-based recovery tempting, but only 35% of those names appear in the harvested name index, and
only 14.5% map to exactly one sequence — and that "unambiguous" subset is itself contaminated with
family labels that happen to match one entry (`temporin` → *Temporin-1Sa*, `tachyplesin`,
`protegrin`, `phylloseptin`, `bombinin`). The most frequent unresolved names are family labels
outright: `magainin` (182 activities), `indolicidin` (157), `cecropin` (152), `cathelicidin`
(135). Resolving wholesale would attach a real measurement to a sequence nobody measured, trading
a known, bounded gap for silent, unbounded contamination. Re-running extraction would not recover
these either — the loss is in the papers (which identify the peptide only by name), not the
parser.

### 2026-08-11 — `TestAgainstTheReleasedDatabase` made hermetic
Used to open the live `data/sample.db` directly, skipping when the file was absent. This raced a
concurrent pipeline rebuild (observed twice — the suite failed mid-`DROP`/`CREATE`) and meant a
clean checkout skipped the test entirely, so an ORM/loader mismatch could go unnoticed
indefinitely. Now builds a throwaway database through `sample.serialize.load_sql_database` — the
actual function step 32 runs — from a two-row fixture CSV, which tests the same "does the ORM
match what the loader produces" question without touching a file the pipeline might be rewriting.
Always runs now; costs about 20s more per suite run.

### 2026-08-10 — Ollama models audited against real call sites, not grepped by name
A 15-model, ~95 GB install had 9 models with zero real code references (~44 GB): `qwen2.5:1.5b`,
`qwen2.5-coder:3b`/`7b`, `qwen3.5:9b`, `qwen3:14b`, `qwen3.5:2b`, `phi4`, `phi3`, `moondream`,
`llava`. One of the kept models (`gemma4:12b`) is referenced only in a comment stating it
*cannot currently load* on this host's Ollama version — kept anyway, per explicit choice, against
a future Ollama upgrade. Removed the 9 confirmed-unused ones. Disk: 22 GB → 66 GB free. Found in
passing: `sample/validation.py`'s comment claiming step 10 validation uses `qwen3.5:9b` is stale —
`run_llm_validation` actually uses `DEFAULT_EXTRACT_MODEL` (`gemma4:latest`); not corrected in the
code, flagged here so it is not mistaken for evidence the model is still load-bearing.

### 2026-08-09 — `pubmed_metadata.csv`, ADP/CAMP/DRAMP/dbAMP link fixes, PMC ID Converter
- `pubmed_metadata.csv` has 0 PMIDs across 99,105 rows — step 1 recorded DOIs and links but never
  the identifier itself, discovered while wiring PMID→DOI resolution.
- The PMC ID Converter returns 403 and fails silently, mapping every PMID to "no DOI" —
  indistinguishable from a genuinely pre-DOI paper. Switched to NCBI E-utilities `esummary`.
- ADP's entry pages require a POST form (`ID=00001` against APD6's search endpoint, not a GET
  URL); CAMP, DRAMP and dbAMP each had a different broken link pattern (wrong host, missing
  parameter, or a renamed parameter). Fixed per-source in `sample/external_links.py`.
- An HTTP 200 from an external database does not mean the link resolved — three source-database
  sites answer 200 for a nonexistent accession, and APD6 serves a byte-identical page for every id
  under its catch-all route. Verified links by checking for expected record content, not status
  code alone.

### 2026-08-08 — Clustering, validation-rule enforcement, pipeline renumbering
- Peptide clustering failed in both directions before landing on the current approach: a flat
  mismatch budget chained everything into one 23,647-member cluster; adding identity/coverage
  floors without handling PepMatch's version-suffix string produced zero edges. See
  `sample/families.py`.
- `ACTIVITY_RULES` declared which activity classes and target kinds each assay permits;
  `enforce_activity` applied none of it — four rules were dead until a graded error-injection
  suite (33 corruption types, 2,000 trials) found them. `organism_contradicts_group` separately
  had its own private dict that claimed to mirror `validation_rules` and did not (no `anticancer`
  entry at all; `antiparasitic` limited to protozoa, which would reject every helminth target). It
  now reads the rules directly instead of maintaining a shadow copy.
- Pipeline steps were renumbered sequentially to 1–32 to make room for new stages
  (`build_peptide_names`, `build_bibliography`, `build_families`, `link_bioactivity`,
  `train_synergy_models`); the SQLite load moved from step 22 to 24.
- `data/literature_data_json_validated/` is an orphan directory — nothing in the Makefile or
  `sample/` reads or writes it, and its counts are higher than the stage it appears to follow, so
  comparing against it misleads. The live chain is step 7 → `literature_data_json`, step 12 →
  `_revised`, step 13 → `_repaired`, step 14 → `activities.csv`.

### 2026-10-29 — Novelty is a graded judgment; only some names may adjudicate
- `sequence_novelty` was `seq not in third_party_sequences`. Over 32,408 refined documents that
  called 975 sequences new; classifying all 975 by hand left ~230 defensible. The rest were
  fragments of catalogued peptides (145), catalogued peptides with flanking text (121),
  three-letter notation that can never match a one-letter set (103), near-neighbours within two
  substitutions (60), identifiable MHC epitopes (46), sequences masked by an ambiguity code (36),
  one-residue indels (5) and names spelled in residue letters (4). Replaced with the graded test
  in `sample/novelty.py`, which on a later 32,473-document snapshot claims 460 where the old test
  claims 969 — a 53% reduction, plus 388 reported as uncountable rather than claimed.
- The measurement that settled it: of the 40 whose extracted name is a peptide this project
  catalogues, the sequence disagreed with the catalogued one **40 times out of 40**. Where novelty
  was checkable it was always an extraction error, so the name had to become part of the test.
- **The sequence is judged before the name.** The reverse was implemented first and measured
  worse: 107 name conflicts in the trustworthy tier against the correct 29. A name test applied
  before the containment tests cannot tell a truncation from a different peptide — `DMS-H4` with
  `GLWSTIKNVGKEAAIAAGKAALGAL` against the catalogued `GLWSTIKNVGKEAAIAAGKAALGALG` is the same
  peptide, one residue short. Novelty and name agreement are now separate axes on the verdict.
- B/Z/X/U/O stay valid in `validate_extracted_sequence` and make a sequence *unjudgeable* for
  novelty. They are legitimate IUPAC codes, so rejecting them would delete real records; but each
  masks a position, and novelty is an identity claim. `not_comparable` replaces a guess in either
  direction. Same for sequences under 8 residues, which is above the validator's floor of 5
  because "is this a sequence" and "does its absence mean anything" are different bars.
- **Only a name that identifies something may contradict a sequence.** Reused
  `names.NAME_SEQUENCE_THRESHOLD` (20) rather than inventing a second threshold: a name attached
  to 20+ sequences is a class label — dbAMP filed `Mouse-ear cress` as a name for 300 peptides.
  `silent` is recorded distinctly from `disagrees`, because "no standing to say" is not
  "contradicts". Measured: 8,850 records have an adjudicating name, 7,682 corroborate, 865 (10%)
  contradict.
- `sequence_of_another_peptide` went from an unmeasurable failure mode to a step-12 rule. It is
  the one defect on this field that neither a structural rule nor a model could catch — the
  sequence is perfect, and a model shown only the document cannot know the name belongs to another
  peptide. The catalogue can, and it fires on 865 records (3.82% of those with a sequence).
- The analogue test cannot brute force: the length window around a 20-mer holds 17,857 catalogued
  sequences and the corpus has 13,183 to judge, which is 235M edit-distance computations. Uses the
  pigeonhole principle instead — `MAX_ANALOGUE_EDITS + 1` blocks per catalogued sequence, so k
  edits always leave one block intact and therefore present verbatim in the query. Exact, not
  heuristic; 28s for the whole corpus, and a test asserts it agrees with brute force on 300
  randomized queries including deliberate 1-2 edit mutants.
- Novelty is not scope. An MHC epitope absent from every AMP database is a genuinely new sequence;
  SIINFEKL (in 76 documents of this corpus) is just not an antimicrobial peptide. Deliberately
  left out of the novelty verdict and not yet implemented anywhere, rather than folded in — that
  would repeat the conflation the name ordering already cost.
- `Catalogue.load` reads `data/peptide_names.json`, which step 35 writes — after step 21 consumes
  it. On a first run from scratch the file is absent, the name tests stay silent, and the step
  logs that rather than failing. Accepted: the degradation is to a weaker claim, and the file
  exists on every subsequent run.

### 2026-11-04 — Scale corrections from the record's own JSON, and the faults that mimic them
- Measured against the curated databases over 4,446 findings both sides report for the same
  peptide, assay and organism: `uM` agrees to a median 1.66x and `ug/mL` to 1.00x, while `mg/mL`
  agrees to a median of exactly **1000.00x** — 642 of its 670 overlapping findings are off by more
  than 10x. Excluding `mg/mL`, agreement is 67.0% within 2x, 86.0% within 4x, 93.4% within 10x.
  Including it, that one unit is 56% of every gross disagreement in the corpus.
- The prior is as lopsided: `mg/mL` appears 38 times in 182,766 curated potency measurements
  (0.02%); this extraction has 1,733.
- **Diagnose, do not rewrite.** Three faults arrive identical — a lost prefix, a stock solution
  recorded as a potency (`2 mg/mL` for a dissolved peptide is an ordinary solution and an
  impossible MIC), and a misread number (`value_lower 0` where the sentence says 4.69). Rescaling
  all three produces well-formed wrong answers and erases the evidence. `Finding.rescalable` is
  therefore a field, and a stock solution is routed to `value_raw`, not `activity_unit`.
- **Impossible is correctable, rare is not.** `g/mL` is three orders past peptide solubility and
  appears zero times in the curated corpus, so its spelling in the document IS the damage and step
  11 corrects it — 227 of the 268 rule corrections. `mg/mL` is improbable but possible: the
  context sentence often confirms it ("IC50 was reported to be 6.62 mg/ml"), a paper may use it
  for one assay, and 12 of the first 14 such records had no context sentence. Those 1,848 go to
  step 12 with the sentence attached.
- **Only the unit changes; no number is touched.** `value_lower` is expressed in `activity_unit`,
  so restoring a dropped prefix is a change of unit at a fixed number. Moving both would move the
  quantity twice.
- **The evidence is the JSON.** `context_sentence` is already saved on 53.2% of activities (32.8%
  with a readable value and unit, median 66 characters), so the decisive passage needs no PDF
  reopened and no passage relocated. `record_context` packages it with the sibling measurements
  and the deterministic diagnosis, and every field spec gets it. It is labelled as the extractor's
  record, not the document's words, because it can itself be the fault.
- Three bugs found by running the thing on real data rather than reasoning about it, each now a
  test: (1) every one of the first 398 "confident" findings was a `mg/L` ↔ `ug/mL` no-op, since
  those are the same quantity spelled differently; (2) choosing the corrected prefix by which
  candidate landed in the plausible band proposed `g/mL → ng/mL` on nine records, a 10^9 fix for a
  10^6 error — the restoration order is now fixed to micro-then-milli with the number held
  constant; (3) comparing the recorded value with the sentence as raw numbers flagged 1,672
  activities as contradicting their own context, when most were honest conversions ("1.43 to 22.85
  mM" stored as 1430 uM) or lists sharing one trailing unit ("4.69 and 0.01 mg/ml") — canonical
  comparison plus shared-unit parsing cut that to 288.
- `scale.canonical_unit` returns lower-case lookup keys so damaged spellings collapse onto one
  entry, but the corpus stores `uM` (7,525) and `ug/mL` (5,379) and `preprocessing.canonicalize_unit`
  is case-preserving for molar units. Writing a key straight into a record introduced a spelling
  nothing else used, which `_repair_unit` then rewrote on the next pass — the repair was not
  idempotent, it only looked that way while the two stages ran in the same order twice. `DISPLAY`
  maps keys back before any write.
- Two corrections that needed no model, found by auditing the 13,258 off-vocabulary units.
  (1) `preprocessing.canonicalize_unit` repaired none of the damage PDF extraction produces --
  `u M` 251, `u g/mL` 142, `AM` 113, `WM` 113, `lgml` 75, `mg/liter` 70, `Ag/mL` 58, `lm` 40,
  `ug mL-1` 33, `microgram/mL` 25, `$\mu$ M` 20, and more: 1,212 activities whose unit the field
  does use, spelled in a way the vocabulary does not contain. Step 11 now runs
  `scale.canonical_unit` on whatever the first canonicalizer leaves off-vocabulary, gated on the
  result BEING in the vocabulary so that `kcal/mol`, `fold`, `Range` and `540 nm` stay as written
  instead of being coerced into the nearest concentration. (2) `metric_as_unit` was declared
  deterministic with count=100 and never implemented; `MIC` is in the unit column 622 times, `MBC`
  111, `MRSA` 89, `MFC` 28, `VRE` 25. Now settled by rule, and flagged rather than guessed,
  because the real unit is not in the record.
- Recovering a lost unit from the context sentence was implemented, measured and **abandoned**.
  Only 219 activities matched by exact value, and the matches were wrong in kind: `50 nmol of
  OVA-I peptide` is an amount and `0.22 mmol/kg` is a dose per body weight, neither a
  concentration. The attempt exposed a worse bug in `QUANTITY` -- the molar alternative matched
  the `nm` inside `nmol`, so a sentence could have been used to "correct" a sound concentration
  into the wrong unit. An amount is now refused unless it carries an explicit per-litre
  denominator, and `mmol/L`/`umol/L` are recognised so that a real spelled-out molarity still
  parses.

### 2026-11-16 — Acting on step 12's verdicts: the model proposes, an authority writes

Step 12 had produced **34,048 verdicts and zero corrections**. `--apply` is a separate flag and
the Makefile recipe never passed it, so 37 hours of local inference produced a diagnosis nothing
read. `sample/verdict_apply.py` is the second half of that stage, wired into step 12's recipe so
it cannot be forgotten again. It costs no model calls.

- **The split that makes it safe: the model proposes, a deterministic authority writes.** The same
  split `llm_validator.fill_missing_fields_from_document` already uses, and it is load-bearing
  because the review's answers are regularly right in substance and unusable as text. It answers
  `target_organism` with `"Acinetobacter baumannii or Pseudomonas aeruginosa"` and the literal
  string `"unsure"`; `peptide_sequence` in three-letter notation; `value_raw` as
  `"3.8 ± 0.4 µg/mL"`, unit included. 30,955 corrections written, **9,992 proposals refused**.
- **A conjunction is refused BEFORE any lookup, not after.** Every multi-organism string has a
  prefix that resolves perfectly well, so a lookup returns a confident answer and files the
  measurement against whichever organism came first. The guard then had to be widened for OCR:
  this corpus fuses words, so `"S. epidermidis andE. faecium"` has no word boundary between the
  `d` and the `E`, `\band\b` does not match, and it resolved cleanly to *Staphylococcus
  epidermidis* alone — found by reading a sample of what a full corpus pass had already written.
  Same fault class as the `againstMicrococcus luteus` case `collect_document_genera` documents.
- **An animal cannot be the target of an MIC.** 253 proposals named the peptide's **source**
  organism as its target — *Rana pretiosa*, *Litoria*, *Phyllomedusa sauvagii*, *Apis mellifera*,
  *Bombyx mori*: the frogs, bees and silkworms AMPs are isolated from. NCBI confirms each is a
  real species, so no name-based check would question it. `validation_rules.TARGET_GROUP_DOMAINS`
  looked like the authority for this and **cannot be used as one**: every `ActivityRule` lists the
  `other` catch-all among its groups and `other` carries no domain restriction, so the permitted-
  domain union is unrestricted for every assay. The guard is therefore scoped to a named set of
  microbial assays and applies only to a value being written where there was none, so a false
  refusal costs an unfilled field.
- **A name/sequence conflict is not a sequence error.** 1,016 verdicts say "the name X identifies
  sequence Y, which this sequence does not match". True, and it does not say which of the two is
  wrong. Those records are the ones where the paper and the catalogue disagree, and the
  disagreement is the finding; rewriting the sequence would delete it.
- **A verdict names its peptide and nothing else, so a repeated name makes it unattributable.**
  Step 12 records `peptide` but no peptide or activity index. 10.6% of verdicts name the empty
  string, 428 documents hold more than one unnamed peptide and 511 hold two peptides with the same
  name. The first corpus pass applied those verdicts to each matching peptide in turn, spreading
  one proposal across peptides the review never looked at; 5,809 verdicts are now refused for this.
  It surfaced as a refusal count of 5,790 where only 461 verdicts were being refused.
- **A strict decoder for writes, the forgiving one only for comparison.**
  `novelty.decode_run_of_three_letter_codes` skips a character that starts no code — right for
  producing a comparison key, wrong for rewriting the sequence of record. Measured: it rewrites
  eight sequences by dropping a residue it could not read (`VAL-PRO-GLY-XAA-GLY` loses its
  ambiguous position, `GLP-HIS-TRP-…` its pyroglutamate, `CYCLO-(-LEU-ARG-LEE-LYS-…)` the residue
  behind the typo `LEE`), each yielding a well-formed sequence one residue shorter than the paper
  reported. Writes now require an exact partition. Separately, the plain-residue test had to run
  **after** the decode, not before: `GLYARGLYSLYSARGARGGLNARGARGARGCYSGLY` is spelled entirely in
  valid one-letter codes, so testing for plain residues first accepts the TAT peptide as a
  36-residue sequence and never decodes it to `GRKKRRQRRRCG`.
- **Deciding the notation before parsing a token.** The separator that makes three-letter code
  readable also splits `D-Ala` into `D` and `Ala`, and `D` is a valid one-letter code for
  aspartate — so a parser accepting both notations at once reads `D-Ala-L-Lys-Gly` as
  Asp-Ala-Leu-Lys-Gly and invents two residues. Caught by a test, not by the corpus.
- **A stray character is a position whose residue is unknown.** Stripping what is not a letter
  turned `KWKLFKKIŸPŸKFLHSAKKF` into `KWKLFKKIPKFLHSAKKF`, two residues shorter, and would turn
  `G[Orn]K` into `GORNK` — inventing an arginine and an asparagine, the exact failure
  `sequence_to_1_letter` documents guarding against. 122 stored sequences reach that path.
- **`scale.to_canonical_value` takes `(unit, value)`.** Written reversed, it returns `None` for
  every input and the plausibility guard silently passes everything. Found by a test that expected
  a refusal and got an acceptance.
- **A logic fix in this module needs a restore, not a re-run.** Every value it writes is
  well-formed, so a later pass with a better rule has nothing to object to and will not undo it.
  Three of the guards above were added after a full corpus pass had already written data; the tree
  was restored from a pre-pass copy and the pass redone. `APPLY_VERSION` records which logic
  produced a tree, and bumping it is necessary but not sufficient.
- **Measured result.** Activities carrying an NCBI taxonomy id: **0 → 22,864** — the corpus had no
  organism identifier under any name, so `escherichia coli` and `e. coli` were two unjoinable
  strings. Abbreviated genus 5,672 → 394; organism placeholders 1,139 → 0; distinct organism
  strings 4,202 → 3,582; unit case variants 380 → 0; activity-type case variants 215 → 0;
  sequences not in one-letter code 929 → 396; distinct sequence strings 15,016 → 14,850.

- **FICI synergy retired outright, 2026-11-23, on request: out of the paper, out of the served
  tree.** The earlier decision above stopped *serving* synergy but deliberately kept the FICI work,
  leaving the pickles in `data/sample_final_release/models/synergy/`. That half is now reversed, on
  the owner's call that the models are "not reliable enough for the current usage." The artifacts
  moved to `data/sample_final_release/models_retired/synergy/` (joining two retired ESM-2 pickles
  already there), and `synergy._model_root` plus `train-synergy --model-dir` now default to that
  retired tree, so a re-train writes outside the served root. `model_registry`'s synergy branch
  stays as a backstop for a `--model-dir` aimed back at `models/`.
- **What justified it, measured rather than asserted.** Only **2 of 18** scopes ever passed their
  own held-out gate: *A. baumannii* (ROC-AUC 0.859) and *K. pneumoniae* (0.834) classification. All
  nine regressors failed, every R2 <= 0.073 and most negative (*S. epidermidis* **-9.99**) against a
  0.3 gate; seven of nine classifiers failed, *S. aureus* at ROC-AUC **0.482** — below chance. The
  full table is in KNOWN_ISSUES.md.
- **The paper had drifted from the code, which is the part worth remembering.** The serving removal
  landed in the codebase and never reached the manuscript, so for several revisions the paper
  claimed in three places that 18 synergy models were "made directly accessible through the web
  portal and REST API" while `model_registry` had been excluding them the whole time, and
  `worker.py`'s docstring advertised "bioactivity and synergy prediction jobs" while calling a
  sequence-only `inference.predict`. A removal is not finished when the code stops doing the thing;
  it is finished when nothing still claims it does. Removing a served feature now means sweeping the
  manuscript and `paper/README.md` in the same change.
- **Kept deliberately: the data, not the models.** The 4,853 audited checkerboard measurements
  (5,507 raw, less 654 discarded for >25% FICI discrepancy) stay in the manuscript as a dataset
  contribution under "Peptide--Antibiotic Combination Assay Curation", which states plainly that no
  predictor ships. The measurements are experimental; their reliability was never what was in
  question. `sample/synergy.py`, `sample/antibiotics.py`, `train-synergy` and the unit tests all
  remain, per the earlier decision.
- **Found while sweeping, left alone:** the served set is the `written` cells of
  `flaml_rollout_progress.json`, currently **18** — which does not include the broad antifungal,
  antiviral, anticancer or antiparasitic models that Table 2 reports. The surviving sentence was
  therefore reworded to "every model that passed its held-out quality gate is directly selectable"
  rather than a count, since the count changes with the next `FROM_SCRATCH=1` rollout.

- **Anomaly detection: ESM-2 was already in the matrix; what was missing was seven descriptors
  and any way to tell.** Asked on 2026-11-23 to make the Isolation Forest "also consider the esm
  features", the first finding was that it already did — `run_anomaly_detection` globbed
  `esm2_dim_` from the frame's columns, and `final_dataset.csv` carries all 320, fully populated
  (0 NaN). The detector had been fitting **354** features: 6 descriptors + 28 composition + 320
  ESM. Nothing in the code, the report or the manuscript said so, which is why the question was
  reasonable to ask.
- **The real defect was the descriptor list, which stopped at `sequence_length`.** Seven
  descriptors sat unused in the CSV: `isoelectric_point`, `net_charge_ph7`, `boman_index`,
  `aliphatic_index`, `instability_index`, `hydrophobicity`, `hydrophobic_moment`. Two of those —
  isoelectric point and net charge at pH 7.4 — were named *in the manuscript* as inputs to a
  matrix they were never in. The block is now the 13-entry `ANOMALY_DESCRIPTOR_COLUMNS`, so the
  paper's claim is true rather than aspirational. Measured on the real release: 361 features, and
  the flagged set moves 7,444 -> 6,224 records (Jaccard 0.658 against the old fit), so this is a
  real change in output, not a cosmetic one.
- **Block influence in an Isolation Forest is column count, not information content.** The split
  feature is drawn uniformly at random, so 320 raw ESM dimensions take **88.6%** of split
  decisions and the 13 descriptors take 3.6%. The detector has therefore always been close to a
  pure ESM-space outlier detector with a decorative descriptor block. `--esm-components N`
  standardizes and PCA-reduces the embedding to rebalance it; **the default is 0 (raw), so no
  release changes behaviour unless someone asks for it.** At N=32 the matrix is 73 features,
  89.4% of embedding variance is retained, and the flagged set drops to 2,883 (2.6%) with Jaccard
  0.305 against the raw fit — a different detector, not a tuning, which is exactly why it is
  opt-in and why the share is now printed at fit time.
- **`fillna(0.0)` is counted now, because on the embedding block it is a trap.** A row of 320
  zeros is itself a strong outlier, so a peptide whose ESM embedding failed would be flagged for
  the failure rather than for its sequence. Fill counts are logged and recorded per block
  (`feature_values_filled`). The current release fills nothing, which is worth knowing rather than
  assuming.
- **The report now states what was fitted**, because it previously could not be reconstructed:
  the embedding block is resolved by prefix, so a release built before step 28 fits a silently
  narrower matrix and writes an `is_anomaly` column that looks identical. `feature_blocks`,
  `n_features_total`, `embedding_feature_share` and `embedding_used` close that. An absent
  embedding block is now a loud warning naming the prefixes, not a shrug.
- **Re-run landed the same day (16:33) and confirmed the prediction exactly:** 6,224 anomalies
  (5.610%) over 361 features, `embedding_used: true`, embedding share 0.8864 — matching the
  read-only forecast of 6,224 to the record. Steps 45 and 46 both ran, so `final_dataset.csv`,
  `anomaly_report.json` and the `descriptors` table in `data/sample.db` all agree at 6,224; the
  manuscript and `paper/README.md` were updated from 7,543/6.8% to 6,224/5.6%.
- **While updating it, found that the paper had been reporting means as medians.** "Anomalies
  possess a median net charge of $+4.97$ ... versus $+1.97$" were the `anomalies_mean` and
  `inliers_mean` fields of the old report, relabelled as medians, with IQRs that match no column
  computed here. The true medians (`net_charge_ph7`) are +4.99 [IQR +1.99, +7.99] against +1.99
  [IQR -0.01, +3.97]. Close enough that the conclusion stands, which is exactly why it survived
  several revisions unnoticed. Statistics quoted in the results are now computed from the release
  CSV rather than read off a report field whose name says what it is.
- **The seven added descriptors paid for themselves in the results.** Isoelectric point (median
  11.50 vs 9.70), hydropathy (-0.76 vs -0.26) and the Boman index (2.32 vs 1.59) all separate the
  flagged set at $p < 10^{-15}$, and they are the evidence for the "highly basic, poly-cationic"
  reading the paper already asserted qualitatively. Conversely they killed one claim: length is
  *not* the discriminant (median 19 residues vs 20, p=0.034), so "extreme-length macrocyclic
  scaffolds" was dropped from the outlier description.

- **`sample.result_cache` deleted, 2026-11-23.** The Redis result cache recorded above had exactly
  two callers -- the Expression Wizard's `/express/fold` and `/express/mpepe-yield` routes -- and
  both left with the wizard when it was split into its own tool. Checked before removing, since
  the instruction was conditional on it breaking nothing: no module imported it, its
  `SAMPLE_CACHE_URL` and `SAMPLE_CACHE_TTL` variables had no other consumer and appeared in no
  config file, and Redis itself stays because Celery uses databases 0 and 1 for the broker and
  result backend (the cache used database 2). Removed with it: `tests/test_result_cache.py`, the
  `docs/sample.rst` autodoc entry -- which would have failed the docs build on a module that no
  longer exists -- and two README passages.
- **Why delete rather than keep it for the next expensive analysis.** It was kept for one day on
  exactly that reasoning, with its docstring amended to say it was unused. That is the state in
  which code rots: it still imports, its tests still pass, and nothing exercises the Redis path,
  so the first caller to adopt it inherits whatever broke in the meantime without knowing. The
  extracted tool now carries an on-disk equivalent of the same `cached()` contract, so the design
  is not lost -- it is in the repository that actually runs it. The git history holds the Redis
  implementation if a future caller wants it back.

## Known environment quirk (not a code bug)

This host's `sample` conda environment intermittently fails to import `sqlite3` with
`ImportError: ...libstdc++.so.6: version 'CXXABI_1.3.15' not found (required by
...libicui18n.so.78)`. Partially worked around: `sample/__init__.py` preloads `sqlite3` before
anything else in the package can pull in a native extension that would otherwise win the symbol
resolution first. The workaround is not complete — a single test file run in isolation
(`pytest tests/test_x.py` alone) can still hit it, non-deterministically, even on an unmodified
file. The full suite (`pytest tests/`) has been reliable every time it was run. Prefer running the
whole suite over isolating one file when this environment is acting up.
