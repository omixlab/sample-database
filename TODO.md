# sAMPle TODO

What is left to do. Completed work is not tracked here — see Git history, or `DECISIONS.md` for
the reasoning behind decisions already made. Milestone grouping is in `ROADMAP.md`; blocked items
and known technical debt are in `KNOWN_ISSUES.md`.

Every item below either has no blocker, or names one explicitly. Compute-heavy or destructive
actions (multi-hour GPU jobs, large downloads, bulk cleanup) are never started from this list
alone — see `DECISIONS.md`'s project rules.

## Feature follow-ups (the core is built; these finish it)

- [x] **ADMET membrane permeability — resolved by indexing CycPeptMPDB, not by a model.** A model
      trained on CycPeptMPDB (the public peptide-permeability data) does not transfer to this
      database's *linear* peptides — different chemistry (cyclic, N-methylated macrocycles) and a
      different mechanism (passive diffusion vs membrane disruption); a quick fit confirmed it (holdout
      R²≈0.24, and linear peptides fall out of its applicability domain). So instead of a misleading
      predictor, CycPeptMPDB is **indexed as a source**: `sample.cycpeptmpdb` / `sample-cli
      import-cycpeptmpdb` normalises its ~8,500 cyclic peptides (SMILES, monomer sequence, `log Papp`
      / PAMPA / Caco2), each tagged `head_to_tail`/circular, and `link_to_release` matches them to
      release peptides by InChIKey where the structures coincide. The ADMET panel's `hemolysis` gap
      now shows the live HemoPI2/macrel values once `build-annotations` has scored a peptide, instead
      of a static pointer; `toxicity` stays an honest gap (ToxinPred3 is upstream-blocked, not
      unimplemented — see `KNOWN_ISSUES.md`). Permeability itself stays a gap: no tool applies here.
- [x] **Incremental update — the acquisition half, built.** `sample.acquisition` / `sample-cli
      incremental-update` / `make incremental-update` chains the existing acquisition and extraction
      steps (1-8, 10, 14-16 — each already resumable on its own) and adds the one piece of real logic
      they didn't have: diffing step 19's consensus CSV against the live release database (by
      `entry_id`) and handing exactly the new rows to `sample.incremental.update_families`. Loading
      the result into the live SQLite database, descriptors, bibliography etc. (steps 19-34, 43)
      stays a separate, deliberate follow-on — not auto-chained onto a network-bound fetch. Like the
      RefSeq sweep, this is a real download/refresh run: built and tested, not started here.

## Annotator: external predictors + pre-computed per-peptide annotation

- [x] **macrel, AMPlify, AntiCP2, HemoPI2, AlgPred2 — all five built, fixed, and verified end to
      end** (2026-08-14) producing real predictions together on a real peptide. Two real invocation
      bugs found and fixed in the process (macrel hardcoded a bare command name; AMPlify's own script
      picks the wrong interpreter outside its environment) and three env files corrected (amplify
      needs python=3.6, not 3.8; anticp2 needs python=3.6, not the "≤3.7" its metadata suggests —
      see `KNOWN_ISSUES.md` for the PEP 440 nuance; algpred2 needs scikit-learn==1.2.2 + numpy<2 to
      unpickle a model trained with sklearn 0.19.2). Not started from a rushed guess: each was
      verified against the real, running tool.
- [x] **toxinpred3 — confirmed genuinely blocked, not attempted further.** Its shipped model pickle
      is incompatible with every scikit-learn version tried (1.1.3 through 1.6.1, each verified
      directly against its own environment) — an upstream packaging defect (it does not even declare
      scikit-learn as a dependency), not something fixable from this project's side. See
      `KNOWN_ISSUES.md`.
- [x] **Pre-computed per-peptide annotation — built.** `sample.peptide_annotations` /
      `sample-cli build-annotations` assembles, per peptide: the five working predictors (batched),
      Pfam domains + SAMPLE family/cluster (from the release database), near-identity hits
      (`sample.similarity.search`), and family-HMM hits (`hmmscan`). Circular peptides are flagged
      explicitly (`is_circular` + a caveat) rather than having their sequence-based results presented
      as if they were linear. Resumable, stored in `data/peptide_annotations.sqlite`, and read by the
      peptide page. **Run in full on 2026-11-24: all 55,879 non-decoy peptides are annotated.**
      36 minutes, because `--predictors-from` reused the predictor columns already in
      `final_dataset.csv` under a sequence-identity guard (55,879 matched, 0 dropped) instead of
      re-running four external tools over the corpus.
- [ ] **Immunogenicity, IEDB epitope hits (B-cell, T-cell, MHC) — two real candidates, each gated
      on a specific thing, confirmed rather than assumed (2026-08-14).** `mhcflurry` (openvax,
      pip-installable, peer-reviewed, actively maintained pan-allele MHC-I binding predictor) is
      genuinely trustworthy — but every prediction needs an explicit HLA allele (`mhcflurry predict
      --alleles HLA-A0201 --peptides ...`), and this database's peptides carry no
      target-allele/patient context. Predicting against an invented allele panel would be exactly
      the unfounded guess `DECISIONS.md`'s rules forbid, so this needs a scoped decision (e.g. a
      justified reference HLA panel from a population-coverage study) before it can be wired.
      IEDB's own "T Cell, Class I" next-gen tool (includes immunogenicity, not just binding) IS
      confirmed real and directly downloadable — `downloads.iedb.org/nextgen-tools/tcell_mhci/
      LATEST/` (not the license-gated legacy `tools.iedb.org` page), Python 3.8+, source + examples
      — but the archive is **841 MB**, the same large-download gate as RP3Net's checkpoint below.
      Not fetched without a go-ahead. Membrane permeability and allergenicity are resolved
      (CycPeptMPDB indexing, AlgPred2, above).

## Later

- [x] **MPEPE — built, fixed, and wired as the Expression Wizard's learned yield estimate
      (2026-08-14).** `sample.expression.mpepe_yield_estimate` / `POST /express/mpepe-yield`
      scores a construct's coding DNA with MPEPE's 10-model ensemble. Verified against its own
      published example (reproduces the documented AVE scores exactly). Runs on Python 2.7 /
      TensorFlow 1.15 / Theano — end-of-life software with no ongoing security patching, kept only
      because it is the working implementation of a real, peer-reviewed model; the risk is
      disclosed in the result's `note` on every call, not only on failure. A real upstream bug was
      found and worked around: MPEPE's own `name_seq()` collapses a single-sequence FASTA's
      name/sequence lists to scalar strings, and the caller's `zip()` over them then iterates
      character-by-character — each single base fails MPEPE's own CDS check and it exits with
      **status 0** (no error a caller can catch from the exit code, no result file). Worked around
      by always submitting a second dummy record and matching the real one back by id. ~4 minutes
      per call (no warm server), so it is never part of the fast `/express` path — an explicit,
      opt-in action like `/express/fold`. See `DECISIONS.md`'s 2026-08-14 entry for the full
      account.
- [ ] **RP3Net and TLCP-EPE — the other two expression-yield candidates, each still blocked.**
      **RP3Net** is real, `pip install RP3Net`, MIT license, published (AUROC 0.83, 97-construct
      holdout) — blocked only on its checkpoint: 2.4 GB, hosted on EBI's FTP, the same
      large-download gate as the RefSeq sweep, not started without a go-ahead. **TLCP-EPE** is
      real, published, validated (AUC 0.835 codon / 0.713 protein) — but no public code repository
      found anywhere; not vettable further without the authors releasing it, or a from-scratch
      reimplementation validated against the paper's own benchmark (a research commitment, not a
      tool install, and not attempted).
- [x] **DIAMOND / MMseqs2 search mode — investigated for real (2026-08-14), not wired: no verified
      benefit over the `blastp` mode already in `/search/blast`.** Sizing turned out to be a
      non-issue (`diamond` 3 MB, `mmseqs2` 127 MB via bioconda, installed and tested directly, not
      guessed at from package listings). The actual hypothesis worth testing was whether either
      solves BLAST's well-known weakness on very short queries — this database's peptides are often
      under 15 residues. Tested head-to-head against the real release database (26,358 sequences):
      a 6-residue query returned **zero hits from blastp, diamond default, and diamond
      `--ultra-sensitive` alike** — an E-value significance-model limitation shared by every
      local-alignment tool tested, not a `blastp`-specific gap. At a realistic query length (23
      residues), diamond and blastp returned comparably-scored top hits in comparable time (~0.4s
      either way) — matching `/search/blast`'s own docstring that blastp is already "well under a
      second" against this corpus. Building a redundant search mode adds real maintenance surface
      (endpoint, env dependency, tests) for a capability difference that direct testing did not
      find. Not reopened without a concrete new reason (e.g. a corpus size where blastp stops being
      fast, or a documented DIAMOND/MMseqs2 sensitivity mode this test didn't try).

## Negative-class redesign — the follow-ups it leaves open (2026-09-15)

The code half is done: `sample.decoys` generates adversarial `uniprot_amp_like` hard negatives and
`method="combined"` is a three-way mix (`DECOY_METHOD_MIX`); `sample.ml_models`
(`USE_MEASURED_INACTIVE_NEGATIVES`) puts real measured-inactive peptides into every classification
task's negative class, and each model's record now reports `n_negatives_decoy` vs
`n_negatives_measured_inactive`. Both are unit-tested. What is left is a run or a decision, not code:

- [ ] **Regenerate the decoy set and retrain** — step 28 (`sample-cli decoys`) → 29 (assembly) →
      42 (`train_all`) → 44 (`benchmark`). Multi-hour; by project rule not started without a
      go-ahead. Until it runs, Sections 3.8-3.10 of the paper stay placeholders, and nothing
      computed under the old decoy-only negative class is comparable to what comes out.
- [ ] **Understand the `[Errno 5] Input/output error` that killed 16 tasks** in the last rollout
      (`flaml_rollout_progress.json`) BEFORE re-running, or the re-run loses the same tasks again.
      See `KNOWN_ISSUES.md`.
- [x] **`negative_type` in `final_dataset.csv` -- done, and the original diagnosis was wrong.**
      It was never dropped at assembly: `pd.concat` unions columns (verified directly), so the
      column always survived; the shipped dataset merely predates it. `sample-cli assembly` now
      writes it explicitly on both sides so every row has a defined value rather than an
      incidental NaN. Stratifying metrics by negative source needs only the regeneration run
      below, not new plumbing.
- [ ] **Re-run the benchmark to settle the Table 2 discrepancy** (`KNOWN_ISSUES.md`): the shipped
      `benchmark_report.json` and the manuscript's Table 2 disagree on both sides of every
      comparison, and the table's numbers match no run on disk. Section 3.10 carries a correction
      notice until this is resolved.
- [ ] **Decide the classifier's operating point, separately from the negatives.** Several shipped
      models pair a high ROC-AUC with near-zero recall, which is a threshold artifact of
      F1-maximisation under heavy imbalance, not a discrimination failure. Options: calibrate on
      F-beta with beta > 1, or serve the threshold as a user-selectable operating point and report
      recall-at-fixed-precision. Needs a call on what the resource's default should optimise for.
- [ ] **Two negative sources named in Section 4 are still unbuilt**: peptides drawn specifically
      from secreted-protein space never annotated as antimicrobial, and the substitution study
      that measures how much of any reported metric was attributable to the negatives.
