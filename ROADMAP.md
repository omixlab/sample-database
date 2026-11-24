# Roadmap

Milestones grouping the backlog in `TODO.md` by dependency and rough sequencing. Not a schedule —
no dates, since several items need an explicit go-ahead (large downloads, unattended GPU time)
rather than a timeline. Re-order freely; the grouping reflects what unblocks what, not fixed
priority. `KNOWN_ISSUES.md` records what specifically blocks each gated item.

## Next release — one manual run away

Everything else in this milestone is built, tested and wired; each remaining item is a deliberate,
explicitly gated run rather than more code:

- **Incremental update** — `sample.acquisition` / `incremental-update` chains the acquisition steps
  and places new peptides via the existing family-assignment step. Built and tested; the actual
  paper-fetching run is, again, a deliberate run rather than code left to write.

## Annotator — predictors

Five external predictors are built, fixed and verified end to end: macrel, AMPlify, AntiCP2,
HemoPI2, AlgPred2 (allergenicity). ToxinPred3 (toxicity) is genuinely blocked upstream — its
pickle is incompatible with every scikit-learn version tried — not merely unbuilt. ADMET's
permeability endpoint is resolved by indexing CycPeptMPDB as a source rather than modelling it (the
wrong chemical space for this database's linear peptides). Immunogenicity and IEDB epitope hits
(B-cell, T-cell, MHC) are the one prediction type still without a wired tool — see
`KNOWN_ISSUES.md` for the two real candidates found and the specific reason each isn't a drop-in.

## Expression Wizard — learned yield estimate

**MPEPE is wired**: `POST /express/mpepe-yield` scores a construct's coding DNA with a 10-model
ensemble, verified to reproduce its own published example exactly. Runs on end-of-life software
(Python 2.7 / TensorFlow 1.15 / Theano) the result's `note` discloses on every call, and takes ~4
minutes with no warm server, so it is an explicit, opt-in action — never part of the fast
`/express` construct-design path — like ColabFold folding already is. A real upstream bug (a
silent, exit-code-0 failure on single-sequence input) was found and worked around; see
`DECISIONS.md`. RP3Net and TLCP-EPE remain the two other candidates, each blocked differently — see
`TODO.md`: RP3Net needs a go-ahead for its 2.4 GB checkpoint, TLCP-EPE has no public code at all.

## Later — needs a scope decision first

- RP3Net (expression-yield) — real and installable, blocked only on its 2.4 GB checkpoint download
  needing a go-ahead, the same gate as the RefSeq sweep.
- Immunogenicity / IEDB epitope hits — needs a justified reference HLA-allele panel (for
  `mhcflurry`) or a confirmed install of IEDB's own allele-independent tool; see `KNOWN_ISSUES.md`.

DIAMOND / MMseqs2 search mode was investigated and closed, not deferred: tested head-to-head
against the real release database, neither offered a verified benefit over the `blastp` mode
already in `/search/blast` — see `TODO.md` for the actual numbers.
