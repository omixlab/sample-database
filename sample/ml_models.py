"""
Per-task, per-feature-block model training with honest validation.

The previous training step fitted one Random Forest on one pooled feature matrix and reported its
accuracy. That answers a question nobody asked: "can a model tell any AMP from any decoy on all
features at once?" It cannot say whether the signal is in the language-model embedding or in the
molecular weight, whether the model works for antifungals as well as antibacterials, or -- the one
that matters -- whether the score is real at all rather than an artifact of leakage.

This module trains a MATRIX of models: every task (each activity class; each well-represented target
organism, exact cell line, or pooled cancer type, one assay at a time; hemolysis; each systemic
toxicity assay) against every feature block. There is no task pooling across activity classes,
across assays, or across targets -- "any real peptide vs. a decoy, regardless of what it's active
against" or "active by MIC, against whatever organism" are not biological questions, so they are not
tasks this module builds. Each cell is validated the same way, so the numbers are comparable and the
comparison is the result.

Four safeguards, each catching a failure the pooled model could not:

* **Sequence-level deduplication and grouped splitting.** Near-identical peptides are pervasive in
  this literature -- analog series differing by one residue. Splitting at random puts an analog in
  train and its sibling in test, and the reported score measures memorization.
* **Cross-validation** rather than a single split, so the number carries a spread.
* **y-randomization.** The labels are shuffled and the model refitted. A sound model collapses to
  chance; one that still scores well is reading something structural, and its headline number is
  meaningless. This is the check that catches leakage, and it is the reason to trust the rest.
* **Bootstrap confidence intervals** on the held-out set, so a difference between two feature blocks
  can be read as real or as noise.
"""
import collections
import json
import logging
import os
import time
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("sample.ml_models")

#: Minimum positives for a task to be worth training. Below this a model is fitting noise, and the
#: holdout number would be dominated by which few peptides landed in the test split.
MIN_POSITIVES_PER_TASK: int = 60

#: Folds the grouped splitter (:func:`fit_and_validate`) is constructed with; only the first split
#: is actually used, so this sets the holdout's approximate size (1/N_SPLITS, ~20%) rather than a
#: repeat count -- there is no cross-validation left in this pipeline to repeat.
N_SPLITS: int = 5

#: Bootstrap resamples used for the held-out confidence interval, run once per task.
N_BOOTSTRAP: int = 100

#: Shuffled-label repeats. The mean of these is the number that must sit at chance.
N_Y_RANDOMIZATIONS: int = 5

RANDOM_STATE: int = 42

#: Sequence identity above which two REAL peptides are grouped together for splitting purposes, so
#: an analog series (point mutants, truncations) cannot land one member in train and its sibling in
#: test -- the leakage `pd.factorize` on the raw sequence string cannot catch, because two 95%-
#: identical sequences are two different strings. See :func:`_cluster_groups`.
#:
#: 40 %, the conventional homology-reduction threshold for reporting a peptide classifier, and the
#: floor cd-hit proper supports. Was 90 %, which only ever caught near-duplicates: this literature
#: is built of analog series, and two members of one family at 60 % identity are not independent
#: observations however different their strings look, so a 90 % threshold left them free to land on
#: opposite sides of the split. Lowering it MERGES groups, which shrinks the effective number of
#: independent examples and will lower every reported metric -- that is the point. The previous
#: numbers were measuring partial memorisation of held-out homologues.
GROUPING_IDENTITY: float = 0.4

#: Whether :func:`_family_groups` also unions curated Pfam families into the split groups.
#:
#: OFF, because unioning two overlapping homology signals is single-linkage transitive closure and
#: it collapses this dataset. Measured on the antibacterial task (67,897 labelled rows, 14,283
#: positives): cd-hit clustering alone yields 55,793 groups whose largest holds 201 rows (1.4 % of
#: positives), and the largest curated family holds 3,841 -- but their union-find closure produces
#: ONE group of 11,589 rows carrying 81.1 % of all positives and no negatives, because peptide A
#: shares a family with B, B shares a cluster with C, C shares another family with D, and the
#: components chain.
#:
#: :class:`~sklearn.model_selection.StratifiedGroupKFold` cannot divide a group, so that blob went
#: whole to the training side and left a holdout of 4.4 % prevalence against training's 22.0 % --
#: a 5x train/test distribution shift that made every holdout metric a measurement of an
#: unrepresentative remnant (and a model never held out against its own dominant positive family).
#:
#: Clustering alone still enforces the leakage guard that matters: no peptide within
#: :data:`GROUPING_IDENTITY` of a training peptide can appear in the holdout. What is given up is
#: the union's stated benefit -- linking divergent members of one Pfam family that fall below the
#: identity threshold -- which is real but not worth a holdout that cannot measure anything.
UNION_FAMILIES_IN_GROUPING: bool = False

#: Neighbours used for the applicability-domain distance. 5 is the conventional choice in the QSAR
#: literature (Roy et al. 2015) balancing a stable distance estimate against sensitivity to local
#: density.
AD_K_NEIGHBORS: int = 5



#: A peptide counts as active on a quantitative assay when its reported concentration, converted to
#: micromolar, is below this bar. One literature-grounded criterion, applied uniformly after unit
#: conversion, rather than a different cutoff per assay's native unit -- "has a recorded MIC" or
#: "has a recorded IC50" says nothing about potency on its own; this is what makes "active" mean
#: something.
ACTIVITY_THRESHOLD_UM: float = 5.0

#: The bar BELOW which a measured peptide is not a negative either. Potency is three bands, not
#: two: high (<= ACTIVITY_THRESHOLD_UM), low (between the two bars) and inactive (> this).
#:
#: Introduced because the binary split was contaminating the negative class with genuinely active
#: peptides. "Measured and not potent" previously meant "median > 5 uM", which swept in everything
#: from 5.1 uM upward -- and the published convention for an inactive AMP is >50-100 uM or
#: >128 ug/mL, not >5. Measured across this corpus's own (peptide, target, assay) medians:
#:
#:     high      <= 5 uM        26,374 pairs   7,271 peptides
#:     low       5-100 uM       67,158 pairs  12,816 peptides
#:     inactive  > 100 uM       25,966 pairs   6,683 peptides
#:
#: So 67,158 of the 93,124 pairs the old rule called negative -- 72% of them -- were peptides with
#: real, moderate activity being asserted as inactive. The low band is now held out of binary
#: classification entirely: it is neither a positive (it is not a high-potency lead) nor a
#: negative (it is demonstrably active). That costs 56% of the measured pairs and buys a negative
#: class that means what it says, and a roughly balanced one (26,374 vs 25,966).
INACTIVE_THRESHOLD_UM: float = 100.0

#: Whether a classification task's negative class also includes REAL peptides that were measured
#: against that exact task and missed :data:`ACTIVITY_THRESHOLD_UM` -- not only the decoy pool.
#:
#: ON, and the reason is that the alternative measures the wrong thing. With decoys as the only
#: negatives, a task's question is "is this a real peptide or a shuffled/UniProt-fragment string?",
#: not "is this peptide active?": the positives are real curated peptides and every negative is a
#: construction, so composition and residue-order artifacts of the construction are available as
#: shortcuts, and the reported metric partly measures how detectable the decoy generator is. The
#: measured-inactive peptides are the negatives the question actually needs -- real sequences, real
#: assays against the same target, a real number that missed the bar -- and they cannot be
#: separated from the positives by any artifact of how a decoy was built.
#:
#: What this does NOT claim: an inactive peptide here is inactive *at this threshold, in this
#: assay, against this target*, which is not the same as inert. That is why membership is decided
#: from the same median-of-repeats pass that decides :data:`ACTIVITY_THRESHOLD_UM` activity
#: (:func:`_label_lookup`'s ``_apply_median_threshold``) and read back off its own ``active_*``
#: sets rather than recomputed -- the two labels are two sides of one comparison and cannot drift
#: apart. A peptide with only a qualitative annotation and no number is never a negative: it is a
#: reported active whose potency simply was not captured, and calling it inactive would invent a
#: measurement that does not exist (``DECISIONS.md``: prefer nulling an unparseable value over
#: guessing at it).
#:
#: Turning this OFF restores the decoy-only negative class, for a like-for-like comparison against
#: any number computed before this existed.
USE_MEASURED_INACTIVE_NEGATIVES: bool = True

#: Assays with a real concentration value this pipeline can convert to micromolar and compare
#: against :data:`ACTIVITY_THRESHOLD_UM`. Built from :data:`REGRESSION_ASSAYS` (defined further
#: below) minus ``ZOI`` -- a zone-of-inhibition diameter in mm, not a concentration, so there is
#: nothing to convert -- plus ``Hemolysis``, which carries a real potency value but was never a
#: regression target because "how hemolytic" was never asked as a continuous question; "is it
#: hemolytic below the same bar as everything else" is exactly what this axis asks.
_THRESHOLD_ASSAYS: FrozenSet[str] = frozenset(
    {"MIC", "MIC50", "MIC80", "MIC90", "MIC100", "MBC", "MBC50", "MBC90", "MFC", "MLC", "MHC",
     "IC50", "IC90", "EC50", "EC90", "LD50", "LD90", "LC50", "CC50", "CC90", "ED50", "ED90",
     "GI50", "TC50", "TD50", "HC50", "MBIC", "MBEC", "Hemolysis"})

#: Of the assays above, the subset allowed to form a task with no organism or cell line at all --
#: measured against the host in general (hemolysis, systemic toxicity) rather than against a
#: specific pathogen or cancer target. Everything else (MIC, IC50, ...) needs a real target: a
#: peptide "active" against no one in particular is not a task worth training, it is exactly the
#: pooled-per-metric model this axis exists to not build.
_NO_TARGET_ASSAYS: FrozenSet[str] = frozenset(
    {"Hemolysis", "LD50", "LD90", "LC50", "TC50", "TD50", "HC50", "MHC"})


def _species_level_organism(name: str) -> str:
    """
    Collapses a strain/serovar/subspecies-qualified organism name to its genus + species binomial.

    ``target_organism`` carries whatever granularity the source database recorded -- "Escherichia
    coli", "Escherichia coli W", "Escherichia coli B", and "Escherichia coli C" all name the same
    organism_specific question this axis asks, but as four distinct strings they fragment one real,
    threshold-clearing signal into several that individually may not -- measured directly: real
    corpus data has ``Salmonella enterica`` MIC activity split across 8 differently-qualified
    strings this way. Keeping only the first two whitespace-separated tokens pools them back into
    one task, the same principle :func:`build_cell_line_categories` already applies to individual
    cell lines. A name with fewer than two tokens (rare) is returned unchanged.

    :param name: A ``target_organism`` value, already stripped.
    :returns: ``"Genus species"``, or ``name`` unchanged if it has fewer than two words.
    """
    parts = name.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name


def feature_blocks(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Selects the physicochemical-descriptor and amino-acid-composition blocks to train on --
    ESM-2 is deliberately excluded, by instruction, after this project's own head-to-head
    comparison (antibacterial and anticancer, holdout ROC-AUC with bootstrap CIs, both the fixed-
    hyperparameter XGBoost and a full end-to-end ESM-2 fine-tune) found ESM-2 ahead on ROC-AUC/
    F1/MCC but BEHIND on recall in every comparison -- the metric that mattered more here than the
    metrics ESM-2 was winning.

    Kept as separate blocks (not one combined block) so `train_all`'s existing per-(task, block)
    matrix keeps whichever wins for a given task, the same reason this function used to offer
    multiple blocks before being narrowed to ESM-2 alone (a decision this one reverses).
    A task whose ROC-AUC does not clear :data:`DEFAULT_QUALITY_GATE` on ANY block is rejected
    outright rather than falling back to ESM-2 -- by instruction, uniformly, even where that
    means a task (e.g. anticancer, whose descriptor/composition ROC-AUC ran 0.67-0.81 against
    ESM-2's 0.80-0.87 in that same comparison) ends up with no shipped model at all.

    ``dpc`` and ``ctd`` are order-aware, which the other two blocks are not: amino-acid
    composition and the descriptor scalars are both invariant to shuffling a sequence, so neither
    can express amphipathicity -- the alternation of cationic and hydrophobic residues that
    actually drives membrane activity. They are only available once
    :func:`add_derived_feature_columns` has run, since they are derived from ``sequence`` rather
    than stored in the release CSV.

    :param df: Dataset carrying the descriptor and amino-acid-composition columns, and (after
        :func:`add_derived_feature_columns`) the ``dpc_*``/``ctd_*`` ones.
    :returns: ``{"descriptors": [...], "composition": [...], "dpc": [...], "ctd": [...]}``, each
        omitted if `df` does not carry its columns (empty dict overall -- :func:`train_all` raises
        on that -- if none do).
    """
    # Taken from the serving module's list rather than a second copy of the names here: a column
    # this block trains on that `build_feature_frame` cannot rebuild at scoring time is a model
    # that trains cleanly and then raises on its first real query. Sharing the tuple makes that
    # failure impossible to introduce by editing one site and not the other.
    #
    # That list is NAMED EXPLICITLY and never globbed by substring, because the dataset carries a
    # SECOND, parallel family of biophysical columns under `pep_`/`stab_` prefixes (pep_boman,
    # pep_charge, pep_hydrophobic_moment, stab_isoelectric_point, ...) that is populated for real
    # peptides and 100% NULL for every decoy -- measured directly: 100.0% vs 0.0% coverage. Any of
    # those reaching a feature block hands the model a perfect label proxy: `isnan()` alone
    # separates the classes, and it shows up as a ~1.000 AUC that means nothing, the same way
    # ESM-2 scores 1.000 against shuffled-only decoys by detecting shuffling rather than activity.
    # A substring glob like `"charge" in c` is all it would take.
    from sample.inference import DESCRIPTOR_COLUMNS

    descriptors = [c for c in DESCRIPTOR_COLUMNS if c in df.columns]
    composition = [c for c in df.columns
                   if c.startswith("aa_freq_") or c.startswith("aa_group_freq_")]
    dpc = [c for c in df.columns if c.startswith("dpc_")]
    ctd = [c for c in df.columns if c.startswith("ctd_")]
    blocks: Dict[str, List[str]] = {}
    for name, columns in (("descriptors", descriptors), ("composition", composition),
                          ("dpc", dpc), ("ctd", ctd)):
        if columns:
            blocks[name] = columns
    return blocks


def add_derived_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the dipeptide-composition and CTD columns onto `df`, in place of storing them.

    These two blocks are derived from the ``sequence`` column alone and are therefore NOT written
    into ``final_dataset.csv``: 547 extra columns would add several hundred megabytes to a release
    file that is already 774MB, to carry numbers that are a pure function of a column the file
    already has. Deriving them here instead costs about a minute over the full dataset and keeps
    the training values bit-identical to what :func:`sample.inference.build_feature_frame` will
    recompute at scoring time, since both call the same row builders.

    :param df: Dataset carrying ``sequence``.
    :returns: `df` with the ``dpc_*`` and ``ctd_*`` columns added. Returned unchanged when they are
        already present, so a caller may apply it more than once.
    """
    from sample.inference import CTD_COLUMNS, DPC_COLUMNS, _ctd_row, _dpc_row
    from sample.sequence import sequence_to_1_letter

    if all(c in df.columns for c in (DPC_COLUMNS[0], CTD_COLUMNS[0])):
        return df

    logger.info(f"Deriving {len(DPC_COLUMNS)} dipeptide-composition and {len(CTD_COLUMNS)} CTD "
                f"columns from {len(df)} sequences...")
    seqs = [sequence_to_1_letter(str(s)).upper() for s in df["sequence"]]
    derived = pd.concat(
        [pd.DataFrame(np.asarray([_dpc_row(s) for s in seqs], dtype=np.float32),
                      columns=list(DPC_COLUMNS), index=df.index),
         pd.DataFrame(np.asarray([_ctd_row(s) for s in seqs], dtype=np.float32),
                      columns=list(CTD_COLUMNS), index=df.index)],
        axis=1)
    return pd.concat([df, derived], axis=1)


def _label_lookup(activities_csv: str,
                  mw_by_sequence: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, set]]:
    """
    Reads step 19's activity table into per-sequence label sets.

    :param activities_csv: Path to ``activities.csv``.
    :param mw_by_sequence: ``{sequence: molecular_weight}`` (g/mol), used to convert a mass-per-
        volume reading (µg/mL, mg/mL, ng/mL) to micromolar for the ``active_*`` fields below; a
        sequence missing from this map simply cannot use a mass-unit measurement (a molar-unit one,
        µM, still works without it). Omitted entirely when the caller has no molecular weights.
    :returns: ``{"groups": {...}, "assays": {...}, "organisms": {...}, "cell_lines": {...},
        "cell_categories": {...}, "active_organism_assay": {...}, "active_cell_line_assay": {...},
        "active_cell_category_assay": {...}, "active_no_target_assay": {...},
        "potency_organism_assay": {...}, "potency_cell_line_assay": {...},
        "potency_cell_category_assay": {...}, "potency_no_target_assay": {...}}``; empty dicts when
        the file is absent, which leaves :func:`build_tasks` nothing to build a task from at all --
        every task this module trains is scoped to an activity class, a target, or a host-general
        assay, and none of those are knowable without this file. ``cell_lines`` keys are
        cleaned via :func:`sample.cell_lines.clean_cell_line_name`; ``cell_categories`` additionally
        maps every distinct cell line through :func:`sample.cell_lines.build_cell_line_categories`
        (Cellosaurus-backed: cancer type from its disease cross-references, or source tissue when no
        disease is recorded) so peptides tested against different cell lines from the same tissue of
        origin can be pooled into one task -- real data has as few as 1-22 rows per exact cell line,
        far below :data:`MIN_POSITIVES_PER_TASK`, but often enough per tissue once several lines are
        pooled. ``groups``/``assays``/``organisms``/``cell_lines`` record raw presence -- "this
        peptide was tested this way" -- and are what they always were; the four ``active_*`` fields
        are new and record actual potency: ``{seq: {(target, assay)}}`` for
        ``active_organism_assay``/``active_cell_line_assay``/``active_cell_category_assay``
        (``category`` in the last one, not an exact cell line; ``active_organism_assay``'s target
        is likewise collapsed to its genus + species binomial via
        :func:`_species_level_organism`, so "Escherichia coli W"/"...B"/"...C" and every
        Salmonella serovar/subspecies string pool into one organism-specific task per species
        rather than fragmenting the same real signal across strain-qualified names that
        individually may never clear :data:`MIN_POSITIVES_PER_TASK`), and ``{seq: {assay}}`` for
        ``active_no_target_assay`` -- populated only when the reported concentration, converted to
        micromolar via :func:`sample.concentration.concentration_conversions`, is below
        :data:`ACTIVITY_THRESHOLD_UM`, and only for assays in :data:`_THRESHOLD_ASSAYS`
        (:data:`_NO_TARGET_ASSAYS` for the no-target field). Grouped by peptide sequence first:
        when a peptide has more than one measurement for the same (target, assay) pair, the
        threshold is checked against their MEDIAN, not against whichever single measurement is
        most potent -- one noisy reading should not by itself flip a peptide active (or a lone
        outlier weak one flip it inactive) against several measurements that agree. The four
        ``potency_*`` fields mirror the ``active_*`` ones key-for-key, but unconditionally --
        ``{seq: {(target, assay): [potency, ...]}}`` (``{seq: {assay: [...]}}`` for the no-target
        field) -- every measured potency for that pair, regardless of whether it clears
        :data:`ACTIVITY_THRESHOLD_UM`, since a regression task needs the full range of measured
        values, not just the potent ones; this same list is what the median-based ``active_*``
        determination above reads from, and :func:`build_tasks` separately median-aggregates it
        again for the regression target itself.
        ``"Homo sapiens"`` never appears in ``organisms`` or as an ``active_organism_assay`` target:
        DBAASP/DRAMP record some anticancer IC50s (and general LD50/EC50/Hemolysis toxicity assays)
        with the target organism literally as the human host and no cell-line identity, and a
        per-organism model built from that would misread as "this peptide's activity against
        humans" rather than what it actually is. The anticancer share of those rows
        (``target_group == "anticancer"``) is instead pooled into ``cell_categories`` /
        ``active_cell_category_assay`` under ``"cancer_unspecified_cell_line"``; the rest, when the
        assay is one of :data:`_NO_TARGET_ASSAYS`, becomes a no-target hemolysis/toxicity task
        instead of being silently dropped or shipped as a "Homo sapiens" model.
    """
    from sample.cell_lines import build_cell_line_categories, clean_cell_line_name
    from sample.concentration import concentration_conversions

    empty = {"groups": {}, "assays": {}, "organisms": {}, "cell_lines": {}, "cell_categories": {},
             "active_organism_assay": {}, "active_cell_line_assay": {},
             "active_cell_category_assay": {}, "active_no_target_assay": {},
             "potency_organism_assay": {}, "potency_cell_line_assay": {},
             "potency_cell_category_assay": {}, "potency_no_target_assay": {},
        "inactive_organism_assay": {}, "inactive_cell_line_assay": {},
        "inactive_cell_category_assay": {}, "inactive_no_target_assay": {}}
    if not activities_csv or not os.path.exists(activities_csv):
        logger.warning(
            f"{activities_csv} not found -- no activity labels means no tasks to train at all. "
            "Run step 21 to enable per-class, per-assay-per-target, hemolysis and toxicity models.")
        return empty
    try:
        df = pd.read_csv(activities_csv, low_memory=False)
    except Exception as e:
        logger.error(f"Could not read {activities_csv}: {e}")
        return empty

    out: Dict[str, Dict[str, Any]] = {
        "groups": {}, "assays": {}, "organisms": {}, "cell_lines": {}, "cell_categories": {},
        "active_organism_assay": {}, "active_cell_line_assay": {},
        "active_cell_category_assay": {}, "active_no_target_assay": {},
        "potency_organism_assay": {}, "potency_cell_line_assay": {},
        "potency_cell_category_assay": {}, "potency_no_target_assay": {},
        "inactive_organism_assay": {}, "inactive_cell_line_assay": {},
        "inactive_cell_category_assay": {}, "inactive_no_target_assay": {}}
    for column, key in (("target_group", "groups"), ("activity_type", "assays"),
                        ("target_organism", "organisms"), ("target_cell_line", "cell_lines")):
        if column not in df.columns or "sequence" not in df.columns:
            continue
        sub = df[["sequence", column]].dropna()
        for seq, value in zip(sub["sequence"], sub[column]):
            if key == "cell_lines":
                value = clean_cell_line_name(value)
                if not value:
                    continue
            else:
                value = str(value).strip()
                if not value:
                    continue
            # "Homo sapiens" in target_organism is never the pathogen this axis models -- it is
            # DBAASP/DRAMP recording an anticancer IC50 (or a general LD50/EC50 toxicity assay)
            # against "cancer cells"/humans with no cell-line identity captured. Building an
            # "organism_specific/Homo sapiens" model from it would misread as "this peptide's
            # activity against humans" (i.e. toxicity) rather than what it actually is (cancer
            # potency with unspecified cell line); the anticancer share of these rows is instead
            # routed into cell_categories below, correctly labeled as cancer, not species.
            if key == "organisms" and value.lower() == "homo sapiens":
                continue
            seq_key = str(seq).strip().upper()
            out[key].setdefault(seq_key, set()).add(value)

    # Threshold-based activity (classification) AND the continuous potency behind it (regression):
    # every row with a usable concentration contributes its potency to the potency_* fields
    # unconditionally -- one pass over the data feeding both axes, since which target a row
    # carries (a cell line, when present, always wins over an organism string, matching how
    # `cell_lines` is already treated as more specific than `organisms` elsewhere in this module)
    # is exactly the same question either way. A row with neither target goes to the no-target
    # fields only when its assay is one the host-in-general axis recognizes. The active_* sets
    # are NOT decided per-row here -- see the median-based pass at the end of this function, after
    # every row (from here and the cancer_unspecified_cell_line block below) has contributed its
    # potency, so a peptide's classification reflects all of its repeat measurements together.
    needed = {"sequence", "activity_type", "value_lower", "activity_unit"}
    if needed <= set(df.columns):
        cols = list(needed) + [c for c in ("target_organism", "target_cell_line")
                               if c in df.columns]
        sub = df[cols].dropna(subset=list(needed))
        for row in sub.itertuples(index=False):
            assay = str(row.activity_type).strip()
            if assay not in _THRESHOLD_ASSAYS:
                continue
            value = pd.to_numeric(row.value_lower, errors="coerce")
            if pd.isna(value) or value <= 0:
                continue
            seq_key = str(row.sequence).strip().upper()
            conc = concentration_conversions(float(value), row.activity_unit,
                                             (mw_by_sequence or {}).get(seq_key))
            um, potency = conc["uM"], conc["potency"]
            if um is None:
                continue

            raw_cell_line = getattr(row, "target_cell_line", None)
            cell_line = (clean_cell_line_name(raw_cell_line) if pd.notna(raw_cell_line) else "")
            raw_organism = getattr(row, "target_organism", None)
            organism = str(raw_organism).strip() if pd.notna(raw_organism) else ""

            if cell_line:
                active_key, potency_key, pair = "active_cell_line_assay", "potency_cell_line_assay", (cell_line, assay)
            elif organism and organism.lower() != "homo sapiens":
                pair = (_species_level_organism(organism), assay)
                active_key, potency_key = "active_organism_assay", "potency_organism_assay"
            elif assay in _NO_TARGET_ASSAYS:
                active_key, potency_key, pair = "active_no_target_assay", "potency_no_target_assay", assay
            else:
                # No cell line, organism blank or "Homo sapiens", and the assay is not one of the
                # no-target-eligible ones -- no usable target on this axis at all.
                continue

            if potency is not None:
                out[potency_key].setdefault(seq_key, {}).setdefault(pair, []).append(potency)

    # Classification activity is decided on the MEDIAN of a peptide's repeat measurements for the
    # same (target, assay) pair -- grouped by sequence, exactly as the regression target already
    # is (see _regression_labels_for) -- not on whichever single measurement happens to be most
    # potent. One noisy reading (a plausible OCR/transcription error, or genuine assay-to-assay
    # variance) should not by itself flip a peptide to "active" when its other measurements
    # against the same target and assay disagree; a run of consistently potent measurements
    # should not be discounted to "inactive" by one outlier weak one either. Reuses the potency_*
    # lists already collected rather than re-deriving anything: potency is a strictly monotonic
    # (decreasing) transform of the micromolar value (higher potency == lower concentration), so
    # comparing the median POTENCY against the threshold's own potency equivalent is exactly
    # equivalent to comparing a median MICROMOLAR value against ACTIVITY_THRESHOLD_UM, without a
    # second, lossy round trip through the conversion.
    threshold_potency = -np.log10(ACTIVITY_THRESHOLD_UM * 1e-6)
    # Potency is a DECREASING function of concentration, so the inactive bar -- a HIGHER
    # micromolar value -- is a LOWER potency. A median below this is > INACTIVE_THRESHOLD_UM.
    inactive_potency = -np.log10(INACTIVE_THRESHOLD_UM * 1e-6)

    def _apply_median_threshold(potency_key: str, active_key: str,
                                inactive_key: str = "") -> None:
        for seq_key, pairs in out[potency_key].items():
            for pair, potencies in pairs.items():
                median = float(np.median(potencies))
                # >= , not >: the bar is "<= ACTIVITY_THRESHOLD_UM is high", so a median of
                # exactly 5 uM is a positive. Strict > put it in the low band instead, which
                # contradicted the stated band definition at the boundary itself.
                if median >= threshold_potency:
                    out[active_key].setdefault(seq_key, set()).add(pair)
                elif inactive_key and median < inactive_potency:
                    out[inactive_key].setdefault(seq_key, set()).add(pair)
                # Between the two bars: the low band. Deliberately recorded in neither set, so it
                # is a positive nowhere and a negative nowhere.

    # Organism/cell-line/no-target potency is already fully collected above, so their active_*
    # sets can be decided now -- specifically BEFORE the cell-line -> Cellosaurus-category mapping
    # just below, which reads active_cell_line_assay and must see it already populated.
    _apply_median_threshold("potency_organism_assay", "active_organism_assay",
                            "inactive_organism_assay")
    _apply_median_threshold("potency_cell_line_assay", "active_cell_line_assay",
                            "inactive_cell_line_assay")
    _apply_median_threshold("potency_no_target_assay", "active_no_target_assay",
                            "inactive_no_target_assay")

    categories: Dict[str, Any] = {}
    if out["cell_lines"]:
        all_cell_lines = {v for values in out["cell_lines"].values() for v in values}
        categories = build_cell_line_categories(all_cell_lines)
        logger.info(f"Cellosaurus classified {len(categories)}/{len(all_cell_lines)} distinct "
                   f"cell line names into a cancer type or source tissue.")
        for seq_key, cell_line_values in out["cell_lines"].items():
            for cell_line in cell_line_values:
                classified = categories.get(cell_line)
                if classified:
                    out["cell_categories"].setdefault(seq_key, set()).add(classified[0])
        for seq_key, pairs in out["active_cell_line_assay"].items():
            for cell_line, assay in pairs:
                classified = categories.get(cell_line)
                if classified:
                    out["active_cell_category_assay"].setdefault(seq_key, set()).add(
                        (classified[0], assay))
        # The same lift for the inactive band, or a cell-category task would have positives
        # decided per category and negatives decided per exact cell line, which never match up.
        for seq_key, pairs in out["inactive_cell_line_assay"].items():
            for cell_line, assay in pairs:
                classified = categories.get(cell_line)
                if classified:
                    out["inactive_cell_category_assay"].setdefault(seq_key, set()).add(
                        (classified[0], assay))

    # DBAASP/DRAMP anticancer rows commonly report an IC50 against "cancer cells"
    # (target_organism == "Homo sapiens") with no cell-line identity captured -- excluded from
    # `organisms`/`active_organism_assay` above so no model ever gets built and named as if "Homo
    # sapiens" were the target, but the potency signal is real, so it is pooled here under its own
    # explicit cancer category instead of being dropped outright, alongside the
    # Cellosaurus-classified breast_cancer/leukemia_lymphoma/... categories -- membership as
    # before, plus the same per-assay activity threshold as every other target-bearing field.
    needed = {"sequence", "target_group", "target_organism"}
    if needed <= set(df.columns):
        cols = list(needed) + [c for c in ("target_cell_line", "activity_type", "value_lower",
                                            "activity_unit") if c in df.columns]
        sub = df[cols].copy()
        sub["target_group"] = sub["target_group"].astype(str).str.strip()
        sub["target_organism"] = sub["target_organism"].astype(str).str.strip().str.lower()
        no_cell_line = (sub["target_cell_line"].isna() if "target_cell_line" in sub.columns
                        else pd.Series(True, index=sub.index))
        mask = (sub["target_group"] == "anticancer") & (sub["target_organism"] == "homo sapiens") \
            & no_cell_line
        matched = sub.loc[mask]
        for seq in matched["sequence"]:
            seq_key = str(seq).strip().upper()
            out["cell_categories"].setdefault(seq_key, set()).add("cancer_unspecified_cell_line")

        value_cols = {"activity_type", "value_lower", "activity_unit"}
        if value_cols <= set(matched.columns):
            for row in matched.dropna(subset=list(value_cols)).itertuples(index=False):
                assay = str(row.activity_type).strip()
                if assay not in _THRESHOLD_ASSAYS:
                    continue
                value = pd.to_numeric(row.value_lower, errors="coerce")
                if pd.isna(value) or value <= 0:
                    continue
                seq_key = str(row.sequence).strip().upper()
                conc = concentration_conversions(float(value), row.activity_unit,
                                                 (mw_by_sequence or {}).get(seq_key))
                um, potency = conc["uM"], conc["potency"]
                if um is None:
                    continue
                pair = ("cancer_unspecified_cell_line", assay)
                if potency is not None:
                    out["potency_cell_category_assay"].setdefault(seq_key, {}).setdefault(
                        pair, []).append(potency)

    # This is the ONLY source of potency_cell_category_assay entries (the cell-line -> category
    # mapping above works by relabeling already-decided active_cell_line_assay pairs, not by
    # collecting potency itself), so this median pass has to run after it, not alongside the
    # others above.
    _apply_median_threshold("potency_cell_category_assay", "active_cell_category_assay")
    return out


class TaskSpec(NamedTuple):
    """
    One binary task, described by the dimensions that place it in the model namespace.

    :param peptide_type: Activity class the model predicts, e.g. ``antibacterial``; ``general``
        for the task spanning every real peptide.
    :param scope: ``organism_specific``, ``cell_type_specific`` or ``general`` -- whether the task
        is restricted to one target organism, one cell line, or neither.
    :param assay_type: Assay the task is restricted to, e.g. ``MIC``; ``any`` when unrestricted.
    :param target: The specific organism or cell line, when `scope` is not ``general``.
    :param task: Learning task -- ``classification`` (real peptide vs. decoy) or ``regression``
        (predict the continuous potency of a real peptide). It is the last namespace segment, so a
        given assay can carry both a ``.../mic/classification`` and a ``.../mic/regression`` model
        side by side.
    :param y: Labels. For classification: 1 positive, 0 decoy, ``NaN`` for rows in neither class.
        For regression: the continuous potency (-log10 molar concentration -- pIC50/pMIC-style,
        see :func:`sample.concentration.concentration_conversions`) for real peptides that have
        one, ``NaN`` for everyone else -- decoys included, since a shuffled sequence has no
        measured potency to regress toward.
    :param n_measured_inactive: How many of this task's negatives are REAL peptides measured
        against it and found not potent (:data:`USE_MEASURED_INACTIVE_NEGATIVES`) rather than
        decoys. 0 for a regression task, and 0 when the negative class is decoys alone. Carried on
        the spec so the training report can state each model's negative-class composition instead
        of leaving "negatives" to be read as "decoys", which is what it used to mean.
    """
    peptide_type: str
    scope: str
    assay_type: str
    target: str
    task: str
    y: pd.Series
    n_measured_inactive: int = 0

    @property
    def namespace(self) -> str:
        """
        :returns: The task's path within the model directory, as
            ``<peptide_type>/<scope>/<assay_type>/<task>``. Mirrors how a consumer looks a model up:
            what kind of peptide, tested against what, measured how.
        """
        parts = [_slug(self.peptide_type), _slug(self.scope)]
        if self.target:
            parts.append(_slug(self.target))
        parts += [_slug(self.assay_type), _slug(self.task)]
        return "/".join(parts)

    @property
    def name(self) -> str:
        """:returns: A flat identifier for the task, for logs and report keys."""
        return self.namespace.replace("/", "__")


def _slug(value: str) -> str:
    """
    :param value: Any label going into a directory name.
    :returns: A filesystem-safe lowercase slug.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()
    return cleaned or "unspecified"


def build_tasks(df: pd.DataFrame, activities_csv: str,
                min_task_examples: Optional[int] = None,
                min_target_examples: Optional[int] = None) -> List[TaskSpec]:
    """
    Builds the list of tasks to train: classification everywhere, plus a regression task
    wherever enough real peptides have a measured continuous potency for the same (target, assay).

    Every classification task shares the same negatives -- the decoys -- so the models are
    comparable and each one answers "does this peptide look like an X-active peptide, as opposed
    to a shuffled sequence of the same composition". Using shuffled decoys rather than "peptides
    with a different activity" is deliberate: the latter would train a model to tell antibacterials
    from antifungals, which is a different and much harder question than the one the webserver
    asks. A regression task has no decoys at all -- a shuffled sequence has no measured potency to
    regress toward -- so it is trained on real peptides only.

    No task pooled across a quantitative assay's whole target range ("this peptide's MIC, whatever
    it was tested against") or across every target a metric happens to have been measured for
    ("active by MIC or by MBC or ..., against whatever organism"), for either learning task. A
    quantitative assay (MIC, IC50, LD50, Hemolysis, ...) only ever contributes a task scoped to one
    target (organism, exact cell line, or pooled cancer type) and one assay at a time, or -- for
    the handful of assays measured against the host rather than a specific target (hemolysis,
    systemic toxicity) -- with no target at all; see :data:`_THRESHOLD_ASSAYS` and
    :data:`_NO_TARGET_ASSAYS` in :func:`_label_lookup`. For classification, "positive" means the
    reported concentration, converted to micromolar and median-aggregated across a peptide's
    repeat measurements for the same (target, assay) exactly as the regression target below is, is
    below :data:`ACTIVITY_THRESHOLD_UM` -- an actual potency criterion, not "this assay was run on
    this peptide at all", and not decided by whichever single repeat measurement happens to be most
    potent. For regression, the
    target is potency itself (-log10 of the molar concentration -- pIC50/pMIC-style, the standard
    pharmacological transform; see :func:`sample.concentration.concentration_conversions`), median-
    aggregated across a peptide's several measurements for the same (target, assay) when it has
    more than one. Regressing on this rather than the raw reported number is what makes a
    regression task meaningful at all: the old version of this pipeline regressed on
    log10(value_lower) directly, silently mixing measurements reported in µM with measurements
    reported in µg/mL as if they were the same scale, which they are not without knowing each
    peptide's own molecular weight -- exactly what :func:`sample.concentration.to_micromolar`
    exists to convert through before any number is fit to.

    :param df: Dataset with ``sequence``, ``is_decoy``, and, when available, ``molecular_weight``
        (used to convert a mass-per-volume reading to micromolar; a peptide missing it can still
        use a molar-unit reading, just not a mass-unit one).
    :param activities_csv: Step 19 activity table supplying per-sequence class labels.
    :param min_task_examples: Minimum positives (classification) or minimum peptides with a
        measured potency (regression) for a general/activity-class/pooled-cancer-type/hemolysis/
        toxicity task to be worth training; :data:`MIN_POSITIVES_PER_TASK` when omitted.
    :param min_target_examples: The same, for an organism- or exact-cell-line-specific task;
        ``min_task_examples * 2`` when omitted, matching this module's long-standing default (a
        per-target model trained on a handful of peptides reports a number driven by fold
        assignment, not by biology, so this axis is held to a higher bar than the others).
    :returns: One :class:`TaskSpec` per task that clears its threshold. For classification, ``y``
        is 1 for a positive, 0 for a decoy, ``NaN`` for every other real peptide (some other task's
        positive, or a peptide never tested this way). For regression, ``y`` is the continuous
        potency for a real peptide that has one, ``NaN`` for every decoy and for a real peptide
        without a measurement for this exact (target, assay). Either way the caller drops the
        ``NaN`` rows.
    """
    # Resolved here rather than defaulted in the signature: a default bound at function-definition
    # time would freeze the value MIN_POSITIVES_PER_TASK had when this module was first imported,
    # not its value at call time -- and a caller (a test, a smaller/experimental corpus) that
    # reassigns the module constant expects that reassignment to actually take effect.
    if min_task_examples is None:
        min_task_examples = MIN_POSITIVES_PER_TASK
    target_threshold = (min_target_examples if min_target_examples is not None
                        else min_task_examples * 2)

    mw_by_sequence: Dict[str, float] = {}
    if "molecular_weight" in df.columns:
        mw = pd.to_numeric(df["molecular_weight"], errors="coerce")
        seqs_for_mw = df["sequence"].astype(str).str.strip().str.upper()
        mw_by_sequence = dict(zip(seqs_for_mw[mw.notna()], mw[mw.notna()]))

    labels = _label_lookup(activities_csv, mw_by_sequence=mw_by_sequence)
    is_decoy = df["is_decoy"].astype(int) == 1
    seqs = df["sequence"].astype(str).str.strip().str.upper()

    tasks: List[TaskSpec] = []

    def _labels_for(positive_mask: pd.Series,
                    inactive_mask: Optional[pd.Series] = None) -> pd.Series:
        y = pd.Series(np.nan, index=df.index, dtype=float)
        y[positive_mask.values] = 1.0
        # Real, measured-inactive peptides join the decoys as negatives when the caller found any
        # (see USE_MEASURED_INACTIVE_NEGATIVES). Written BEFORE the decoy assignment so that a
        # decoy sequence which somehow also carries a measurement still ends up labelled 0 either
        # way, rather than depending on which of the two masks won.
        if inactive_mask is not None and USE_MEASURED_INACTIVE_NEGATIVES:
            y[inactive_mask.values] = 0.0
        y[is_decoy.values] = 0.0
        return y

    # Sequences with at least one measurement that DID clear the activity bar, and sequences with
    # at least one usable number at all, pooled across every target axis. The pair is what makes
    # "measured and not potent" decidable for the activity-class tasks below, which are keyed on
    # `groups` (raw "was tested this way" presence) and so carry no potency axis of their own:
    # a class negative has to be in `any_potency` (a real number exists) and out of `any_active`
    # (no number anywhere cleared the bar). Requiring `any_potency` is the guard that keeps a
    # qualitatively-annotated active with no reported value out of the negative class.
    any_active = (set(labels["active_organism_assay"]) | set(labels["active_cell_line_assay"])
                  | set(labels["active_cell_category_assay"])
                  | set(labels["active_no_target_assay"]))
    any_potency = (set(labels["potency_organism_assay"]) | set(labels["potency_cell_line_assay"])
                   | set(labels["potency_cell_category_assay"])
                   | set(labels["potency_no_target_assay"]))
    # Peptides with at least one median in the INACTIVE band. A class negative is drawn from this
    # rather than from "has a number and never cleared the bar": the latter counted the whole
    # 5-100 uM low band as inactive, which is 72% of the pairs it was yielding.
    any_inactive = (set(labels["inactive_organism_assay"])
                    | set(labels["inactive_cell_line_assay"])
                    | set(labels["inactive_cell_category_assay"])
                    | set(labels["inactive_no_target_assay"]))

    def _class_inactive_for(group_value: str) -> pd.Series:
        """
        :param group_value: A ``target_group`` activity class, e.g. ``antibacterial``.
        :returns: Mask of real peptides tested for that class, carrying at least one convertible
            measurement, none of which reached :data:`ACTIVITY_THRESHOLD_UM` against any target.
            Deliberately conservative: a peptide active against anything at all is excluded, so a
            peptide that is a weak antibacterial but a potent antifungal is dropped from this
            task rather than asserted to be an antibacterial negative.

            This mask deliberately OVERRIDES the class task's own positive mask, which is keyed on
            ``groups`` -- mere presence, "this peptide was tested in this class". Presence is not
            activity: a peptide measured against this class and found not potent was landing in
            the positive class purely for having been tested, which is the single largest label
            defect this function exists to correct. :func:`_labels_for` applies the masks in that
            order for exactly this reason.
        """
        return seqs.map(lambda s, v=group_value: (v in labels["groups"].get(s, ())
                                                  and s in any_inactive
                                                  and s not in any_active))

    def _measured_inactive_for(potency: Dict[str, Dict[Any, List[float]]],
                               active: Dict[str, set], key: Any,
                               inactive: Optional[Dict[str, set]] = None) -> pd.Series:
        """
        :param potency: A ``potency_*`` map -- every measured potency, threshold or not.
        :param active: The ``active_*`` map decided from it by the median pass in
            :func:`_label_lookup`.
        :param key: The ``(target, assay)`` pair, or bare assay for the no-target axis.
        :returns: Mask of real peptides measured for exactly this `key` whose median missed the
            activity bar. Read as "has potency for this key, is not active for this key" off the
            already-decided sets rather than re-medianing, so this can never disagree with the
            positive label it is the complement of.
        """
        return seqs.map(lambda s, k=key: (k in inactive.get(s, ())
                                          and k not in active.get(s, ())))

    def _counts(mapping: Dict[str, set]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for value_set in mapping.values():
            for v in value_set:
                out[v] = out.get(v, 0) + 1
        return out

    # Per activity class -- the "main type of peptide" axis. "other" is a residual bucket (mixed
    # hemolysis, systemic toxicity, and qualitative measurements that resolve to none of the five
    # recognized classes), not a coherent class of its own, so it is excluded here; its real signal
    # is what feeds the hemolysis and toxicity tasks below instead.
    for value, _ in sorted(_counts(labels["groups"]).items(), key=lambda kv: -kv[1]):
        if value == "other":
            continue
        mask = seqs.map(lambda s, v=value: v in labels["groups"].get(s, ()))
        if int(mask.sum()) >= min_task_examples:
            inactive = _class_inactive_for(value)
            tasks.append(TaskSpec(value, "general", "any", "", "classification",
                                  _labels_for(mask, inactive),
                                  int(inactive.sum()) if USE_MEASURED_INACTIVE_NEGATIVES else 0))

    def _regression_labels_for(potency: Dict[str, Dict[Any, List[float]]], key: Any) -> pd.Series:
        # Continuous potency (median across a peptide's several measurements for this exact
        # (target, assay), or plain assay for the no-target axis) for real peptides that have one;
        # NaN for everyone else -- decoys included, since a shuffled sequence has no measured
        # potency, which is the whole reason regression cannot reuse the vs-decoy framing
        # classification tasks share.
        y = pd.Series(np.nan, index=df.index, dtype=float)
        values = seqs.map(lambda s: float(np.median(potency[s][key]))
                          if s in potency and key in potency[s] else np.nan)
        y[values.notna().values] = values[values.notna()].values
        return y

    def _add_target_assay_tasks(scope: str, active: Dict[str, set],
                                potency: Dict[str, Dict[Any, List[float]]], threshold: int,
                                inactive_map: Optional[Dict[str, set]] = None) -> None:
        # One task per (target, assay) pair actually present, e.g. "active against E. coli by MIC"
        # and "active against E. coli by MBC" are different questions with different answers, and
        # pooling either across assays or across targets would blur both -- for the regression
        # version of that same pair too, not just classification.
        pairs = sorted({pair for values in active.values() for pair in values} |
                       {pair for values in potency.values() for pair in values})
        for target, assay in pairs:
            key = (target, assay)
            mask = seqs.map(lambda s, k=key: k in active.get(s, ()))
            if int(mask.sum()) >= threshold:
                inactive = _measured_inactive_for(potency, active, key, inactive_map or {})
                tasks.append(TaskSpec("general", scope, assay, target, "classification",
                                      _labels_for(mask, inactive),
                                      int(inactive.sum()) if USE_MEASURED_INACTIVE_NEGATIVES else 0))
            reg_y = _regression_labels_for(potency, key)
            if int(reg_y.notna().sum()) >= threshold:
                tasks.append(TaskSpec("general", scope, assay, target, "regression", reg_y))

    # Per target organism -- the pathogen-specific antibacterial/antifungal/antiparasitic axis.
    _add_target_assay_tasks("organism_specific", labels["active_organism_assay"],
                            labels["potency_organism_assay"], target_threshold,
                            labels["inactive_organism_assay"])

    # Per exact cell line -- includes non-cancer lines used as cytotoxicity/selectivity controls
    # (e.g. fibroblasts), since those carry a real target_cell_line too.
    _add_target_assay_tasks("cell_type_specific", labels["active_cell_line_assay"],
                            labels["potency_cell_line_assay"], target_threshold,
                            labels["inactive_cell_line_assay"])

    # Per pooled cancer type / source tissue (Cellosaurus-classified, plus the
    # "cancer_unspecified_cell_line" bucket for anticancer rows recorded against the human host
    # with no cell-line identity). Held to the lower, general-axis bar rather than the exact-target
    # one above: real per-cell-line counts (1-22 rows) never clear target_threshold on their own,
    # so pooling several lines by what they actually are is what gives this axis a real chance of
    # training at all.
    _add_target_assay_tasks("cancer_type_specific", labels["active_cell_category_assay"],
                            labels["potency_cell_category_assay"], min_task_examples,
                            labels["inactive_cell_category_assay"])

    # Assays measured against the host in general rather than a specific pathogen or cancer
    # target: hemolysis gets its own explicit peptide type -- it is a distinct, named safety
    # question, not a flavor of generic "toxicity" -- and each systemic LD50/LC50/TC50/... assay
    # gets its own task under "toxicity", one assay at a time like every other axis here.
    no_target = labels["active_no_target_assay"]
    no_target_potency = labels["potency_no_target_assay"]
    for assay in sorted({a for values in no_target.values() for a in values} |
                        {a for values in no_target_potency.values() for a in values}):
        peptide_type = "hemolytic" if assay == "Hemolysis" else "toxicity"
        mask = seqs.map(lambda s, a=assay: a in no_target.get(s, ()))
        if int(mask.sum()) >= min_task_examples:
            inactive = _measured_inactive_for(no_target_potency, no_target, assay,
                                              labels["inactive_no_target_assay"])
            tasks.append(TaskSpec(peptide_type, "general", assay, "", "classification",
                                  _labels_for(mask, inactive),
                                  int(inactive.sum()) if USE_MEASURED_INACTIVE_NEGATIVES else 0))
        reg_y = _regression_labels_for(no_target_potency, assay)
        if int(reg_y.notna().sum()) >= min_task_examples:
            tasks.append(TaskSpec(peptide_type, "general", assay, "", "regression", reg_y))

    return tasks


def _is_sequence_input(columns: Sequence[str]) -> bool:
    """
    :param columns: A feature block's column list.
    :returns: Whether this block's contract is the raw sequence string itself (a fine-tuned
        transformer's own input) rather than a row of already-computed numeric features. Every
        other block in this project is numeric-per-column and goes through the ordinary
        ``pd.to_numeric(...).fillna(0.0)`` coercion; that coercion would silently zero out a
        sequence string rather than raise, which is why this needs an explicit branch at every
        call site that builds a feature matrix (:func:`train_task`,
        :func:`sample.benchmark._score_model_from_dataset`,
        :func:`sample.inference.build_feature_frame`) instead of being caught generically.
    """
    return list(columns) == ["sequence"]


def _make_model(is_regression: bool = False, **overrides: Any) -> Any:
    """
    :param is_regression: Build a regressor (potency prediction) rather than a classifier.
    :param overrides: Hyperparameters to override on the base configuration. Applied to whichever
        estimator class is actually in use, so an override meant for XGBoost degrades gracefully to
        the RandomForest fallback's own meaning of the same parameter name (``scale_pos_weight`` ->
        ``class_weight``, specifically -- RandomForestClassifier has no parameter of that name).
        :func:`fit_and_validate` is the one caller that overrides anything today, passing a
        per-task ``scale_pos_weight`` (every classification task here is imbalanced, and none of
        that imbalance is otherwise corrected); there is no hyperparameter search left to feed this
        mechanism beyond that, and it stays available for a caller that wants to experiment
        manually.
    :returns: The base estimator -- a gradient-boosted tree ensemble. XGBoost handles the wide,
        correlated, partly-redundant embedding blocks here better than a random forest and trains
        far faster on 1,300+ columns; it falls back to a random forest when xgboost is unavailable
        so the step still runs. A regressor uses squared-error loss; a classifier, logloss.
    """
    if is_regression:
        try:
            from xgboost import XGBRegressor
            params = dict(n_estimators=400, max_depth=6, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         eval_metric="rmse", tree_method="hist",
                         random_state=RANDOM_STATE, n_jobs=os.cpu_count())
            params.update(overrides)
            return XGBRegressor(**params)
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            logger.warning("xgboost unavailable; falling back to RandomForestRegressor.")
            params = dict(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
            params.update(overrides)
            return RandomForestRegressor(**params)
    try:
        from xgboost import XGBClassifier
        params = dict(n_estimators=400, max_depth=6, learning_rate=0.08,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                     eval_metric="logloss", tree_method="hist",
                     random_state=RANDOM_STATE, n_jobs=os.cpu_count())
        params.update(overrides)
        return XGBClassifier(**params)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        logger.warning("xgboost unavailable; falling back to RandomForestClassifier.")
        params = dict(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
        params.update(overrides)
        # RandomForestClassifier has no scale_pos_weight parameter -- passing it through
        # unmodified would raise a TypeError. Its own class-imbalance correction is
        # class_weight; the same negative:positive ratio means the same thing there (positives
        # weighted up by that factor, negatives left at 1.0) since both work by inflating each
        # positive sample's contribution to the loss.
        spw = params.pop("scale_pos_weight", None)
        if spw is not None:
            params["class_weight"] = {0: 1.0, 1: spw}
        return RandomForestClassifier(**params)


def _predict_scores(model: Any, X: np.ndarray, is_regression: bool) -> np.ndarray:
    """
    The one prediction call both holdout scoring and bootstrapping go through, so a classifier and
    a regressor are read the same way everywhere downstream.

    :param model: A fitted estimator.
    :param X: Feature matrix.
    :param is_regression: Whether `model` is a regressor.
    :returns: For a regressor, its predicted value; for a classifier, the positive-class
        probability. Either way, one number per row, higher meaning "more".
    """
    if is_regression:
        return np.asarray(model.predict(X)).ravel()
    return np.asarray(model.predict_proba(X))[:, 1]


def _score(y_true: np.ndarray, y_pred: np.ndarray, is_regression: bool,
          threshold: float = 0.5) -> Dict[str, float]:
    """
    :param threshold: Classifier decision cutoff, forwarded to :func:`_scores`; ignored for a
        regressor (no classes to threshold).
    :returns: :func:`_regression_scores` for a regressor, :func:`_scores` for a classifier -- the
        single dispatch point so every caller (holdout, CV, bootstrap, y-randomization) reports
        the metric set appropriate to the task without each re-deciding.
    """
    return _regression_scores(y_true, y_pred) if is_regression else _scores(y_true, y_pred, threshold)


def _primary_metric(is_regression: bool) -> str:
    """
    :returns: The metric model selection maximizes and the quality gate keys on -- ``"r2"`` for a
        regressor (1.0 perfect, 0.0 no better than the mean, negative worse), ``"roc_auc"`` for a
        classifier (1.0 perfect, 0.5 chance). Both are "higher is better", which is what lets the
        selection and gate logic stay shared.
    """
    return "r2" if is_regression else "roc_auc"


def _cdhit_available_for_grouping() -> bool:
    """
    :returns: Whether cd-hit is installed, and therefore whether :func:`_cluster_groups` can
        actually deliver :data:`GROUPING_IDENTITY` rather than falling back to PepMatch (which
        under-clusters above ~14 residues -- see that function's own fallback branch). Wrapped
        here so :func:`train_all` can record the answer on every model without importing
        ``sample.families`` at module scope.
    """
    from sample.families import cdhit_available

    return cdhit_available()


def _cluster_groups(df: pd.DataFrame, identity: float = GROUPING_IDENTITY,
                    work_dir: Optional[str] = None) -> np.ndarray:
    """
    Assigns every row a group id such that no two REAL peptides more than `identity` similar to
    each other land in different groups.

    Deliberately not `pd.factorize` on the raw sequence string: that only catches EXACT duplicates,
    and this literature is full of analog series -- point mutants and truncations of one parent
    peptide -- that are near-identical strings, not identical ones. Splitting those at random puts
    an analog in train and its sibling in test, and the reported score then measures memorization
    of the parent, not generalization.

    Decoys are each given their own singleton group rather than also being clustered: they are
    synthetic negatives (shuffled sequences or UniProt fragments) constructed to NOT resemble real
    peptides or each other, so the near-duplicate leakage this function exists to prevent is not a
    decoy-side concern, and clustering ~74k of them through the same PepMatch pass real peptides use
    would roughly double this step's runtime for no protection anything actually needs.

    :param df: Dataset carrying ``sequence`` and ``is_decoy``.
    :param identity: Required fraction of identical positions for two real peptides to share a
        group; :data:`GROUPING_IDENTITY` when omitted. Delivered by cd-hit where it is installed;
        see :func:`sample.families.cluster_cdhit` for why the PepMatch fallback cannot reach it.
    :param work_dir: Scratch directory for PepMatch's index files, forwarded to
        :func:`sample.families.cluster`; that function's own default (``data/families``, step 35's
        working directory) when omitted. Step 35 and this function search at different identity
        thresholds and would otherwise race to overwrite the same scratch filenames if ever run
        concurrently; a caller that wants isolation (a test, or a dedicated step-32 scratch
        directory) can pass its own path.
    :returns: One group id per row of `df`, aligned to `df`'s positional order (not its index).
    """
    from sample.families import cdhit_available, cluster as near_identity_cluster, cluster_cdhit

    is_decoy = df["is_decoy"].astype(int) == 1
    seqs = df["sequence"].astype(str).str.strip().str.upper()

    groups = np.empty(len(df), dtype=np.int64)

    # Decoys: singleton groups, offset into a range no real-peptide cluster id will ever reach.
    decoy_positions = np.flatnonzero(is_decoy.to_numpy())
    groups[decoy_positions] = np.arange(len(decoy_positions))

    # Real peptides: cluster the UNIQUE sequences once, then broadcast each row's cluster id --
    # rows sharing an identical sequence share a cluster automatically, since PepMatch would find
    # them as a 100%-identity pair anyway, so no separate exact-dedup pass is needed here.
    real_positions = np.flatnonzero(~is_decoy.to_numpy())
    real_seqs = seqs.iloc[real_positions]
    unique_seqs = {f"p{i}": s for i, s in enumerate(dict.fromkeys(real_seqs))}
    seq_to_key = {s: k for k, s in unique_seqs.items()}

    if len(unique_seqs) <= 1:
        assignment = {k: 1 for k in unique_seqs}
    elif cdhit_available():
        assignment = cluster_cdhit(unique_seqs, identity=identity, work_dir=work_dir)
    else:
        # PepMatch cannot reach the low identity this split targets -- its mismatch budget is
        # capped at 6, so a 50-residue peptide is clustered at 88% however low `identity` is set.
        # Falling back to it is better than not clustering, but it silently under-clusters, which
        # inflates every score computed off the resulting split, so it says so.
        logger.warning(
            f"cd-hit is not installed; falling back to PepMatch near-identity clustering, which "
            f"cannot deliver {identity:.0%} identity for peptides longer than ~14 residues (its "
            f"mismatch budget is capped at 6). Homologues WILL remain split across train and "
            f"test, and the resulting scores are optimistic. Install cd-hit for a real "
            f"homology-reduced split.")
        assignment = near_identity_cluster(unique_seqs, identity=identity, work_dir=work_dir)

    # Cluster ids from near_identity_cluster start at 1 and are not guaranteed disjoint from the
    # decoy range above without an explicit offset.
    cluster_offset = len(decoy_positions)
    seq_cluster_id = {s: assignment[seq_to_key[s]] + cluster_offset for s in unique_seqs.values()}
    for pos, seq in zip(real_positions, real_seqs):
        groups[pos] = seq_cluster_id[seq]

    n_clusters = len(set(seq_cluster_id.values()))
    logger.info(f"Grouping for train/test splitting: {len(real_positions)} real peptides in "
               f"{n_clusters} clusters at >={identity:.0%} identity "
               f"({len(decoy_positions)} decoys as singletons).")
    return groups


def _family_groups(df: pd.DataFrame, families_json: Optional[str] = None,
                   identity: float = GROUPING_IDENTITY,
                   work_dir: Optional[str] = None,
                   union_families: bool = UNION_FAMILIES_IN_GROUPING) -> np.ndarray:
    """
    Assigns every row a group id for leakage-free train/test splitting, from a :func:`_cluster_groups`
    pass at `identity` over every real peptide and -- only when `union_families` is set -- a union
    with the curated family assignment (``data/families/families.json``, built by step 35's
    ``build_families``, Pfam-anchored where a curated family exists).

    The union is OFF by default. Combining two overlapping homology signals through union-find is
    single-linkage transitive closure, and on this dataset it collapses 81 % of the antibacterial
    positives into a single unsplittable group -- see :data:`UNION_FAMILIES_IN_GROUPING` for the
    measured numbers and what that did to the holdout. Clustering alone still guarantees the
    property this function exists for: nothing within `identity` of a training peptide reaches the
    holdout.

    The cost of leaving it off, stated plainly: two divergent members of one Pfam family that fall
    below `identity` are no longer forced onto the same side of the split, so a small amount of
    remote-homology leakage is possible. That is a bounded, measurable risk; the union's
    alternative was a holdout whose class balance made every metric uninterpretable.

    :param df: Dataset carrying ``sequence``, ``is_decoy``, and ideally ``entry_id`` (the key
        ``families.json`` is written under -- without it no row draws a curated family and the
        grouping rests on :func:`_cluster_groups` alone).
    :param families_json: The family assignment; ``data/families/families.json`` when omitted.
        Read only when `union_families` is set.
    :param identity: Forwarded to :func:`_cluster_groups`.
    :param work_dir: Forwarded to :func:`_cluster_groups`.
    :param union_families: Union curated families into the groups. Defaults to
        :data:`UNION_FAMILIES_IN_GROUPING` (False); pass True to reproduce the previous behaviour.
    :returns: One group id per row of `df`, aligned to `df`'s positional order (not its index).
    """
    from sample.families import FAMILY_DIR

    families_json = families_json or os.path.join(FAMILY_DIR, "families.json")
    assignment: Dict[str, Dict[str, Any]] = {}
    if not union_families:
        pass  # families are not consulted at all; see UNION_FAMILIES_IN_GROUPING
    elif os.path.exists(families_json):
        with open(families_json, encoding="utf-8") as fh:
            assignment = json.load(fh)
    else:
        logger.warning(f"No family assignment at {families_json}; grouping falls back to plain "
                       f"near-identity clustering for every real peptide. Run "
                       f"`sample-cli build-families` (step 37) for curated Pfam-family grouping.")

    is_decoy = df["is_decoy"].astype(int) == 1
    entry_ids = (df["entry_id"].astype(str) if "entry_id" in df.columns
                else df.index.astype(str)).to_numpy()

    family_id_by_pos: Dict[int, str] = {}
    n_pfam = n_file_cluster = 0
    real_positions = np.flatnonzero(~is_decoy.to_numpy())
    for pos in real_positions:
        record = assignment.get(entry_ids[pos])
        family_id = record.get("family_id") if record else None
        if family_id:
            family_id_by_pos[pos] = family_id
            if record.get("source") == "pfam":
                n_pfam += 1
            else:
                n_file_cluster += 1

    # Signal two: sequence-identity clustering over every real peptide (decoys come back as
    # singletons from here, and stay singletons -- they are synthetic negatives built not to
    # resemble anything, so they need no leakage guard).
    cluster_groups = _cluster_groups(df, identity=identity, work_dir=work_dir)

    parent: Dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    # Each row is a node; a representative node per family id and per cluster id links the rows
    # that share it. Representatives live in disjoint integer ranges above every row index so they
    # cannot collide with a row or with each other.
    family_base = len(df)
    family_ids = sorted(set(family_id_by_pos.values()))
    family_node = {fid: family_base + i for i, fid in enumerate(family_ids)}
    cluster_base = family_base + len(family_ids)
    cluster_node: Dict[int, int] = {}

    for pos in real_positions:
        fid = family_id_by_pos.get(int(pos))
        if fid is not None:
            union(family_node[fid], int(pos))
        cid = int(cluster_groups[pos])
        if cid not in cluster_node:
            cluster_node[cid] = cluster_base + len(cluster_node)
        union(cluster_node[cid], int(pos))

    groups = np.empty(len(df), dtype=np.int64)
    decoy_positions = np.flatnonzero(is_decoy.to_numpy())
    groups[decoy_positions] = cluster_groups[decoy_positions]
    # Real-peptide group ids are offset past the decoy singleton range so the two cannot collide.
    real_offset = int(cluster_groups.max()) + 1 if len(cluster_groups) else 0
    root_to_group: Dict[int, int] = {}
    for pos in real_positions:
        root = find(int(pos))
        if root not in root_to_group:
            root_to_group[root] = real_offset + len(root_to_group)
        groups[pos] = root_to_group[root]

    if union_families:
        provenance = (f", unioned with {len(family_ids)} curated families covering "
                      f"{len(family_id_by_pos)} of them ({n_pfam} Pfam-anchored, "
                      f"{n_file_cluster} from step 35's own clustering)")
    else:
        provenance = " (curated families NOT unioned -- see UNION_FAMILIES_IN_GROUPING)"
    largest = max(collections.Counter(groups[real_positions].tolist()).values(), default=0)
    logger.info(
        f"Grouping for train/test splitting: {len(real_positions)} real peptides into "
        f"{len(root_to_group)} groups at >={identity:.0%} identity{provenance}; "
        f"largest group {largest} rows "
        f"({largest / max(len(real_positions), 1):.1%} of real peptides); "
        f"{len(decoy_positions)} decoys as singletons.")
    return groups


def _applicability_domain(X_scaled: np.ndarray, k: int = AD_K_NEIGHBORS) -> Dict[str, Any]:
    """
    Defines the model's domain of applicability as a standardized k-NN distance bound in the
    training feature space (Roy et al. 2015's leverage/distance approach, the standard QSAR
    formulation): a query is IN the domain when its mean distance to its `k` nearest training
    points does not exceed what the training set's own internal neighbor distances already show is
    typical.

    :param X_scaled: The training/selection portion's feature matrix, already
        :class:`~sklearn.preprocessing.StandardScaler`-transformed -- the model itself
        (tree-based) needs no scaling, but a raw-unit Euclidean distance would let a
        thousands-wide `molecular_weight` column dominate a [0, 1]-ranged frequency column.
    :param k: Neighbours to average over; :data:`AD_K_NEIGHBORS` when omitted.
    :returns: ``{"index": fitted NearestNeighbors, "k", "threshold", "training_mean_distance",
        "training_std_distance"}``. `threshold` is ``training_mean_distance +
        3 * training_std_distance`` over each training point's own distance to its `k` nearest
        OTHER training points (excluding itself) -- three standard deviations past what the
        training set's own internal density looks like.
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(X_scaled) - 1)
    if k < 1:
        # A task too small to have a meaningful neighbourhood at all; every query is reported
        # out-of-domain rather than the check being silently skipped.
        return {"index": None, "k": 0, "threshold": 0.0,
               "training_mean_distance": float("nan"), "training_std_distance": float("nan")}

    index = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    distances, _ = index.kneighbors(X_scaled)
    own_distances = distances[:, 1:].mean(axis=1)  # column 0 is each point's distance to itself
    mean_d, std_d = float(own_distances.mean()), float(own_distances.std())
    return {"index": index, "k": k, "threshold": mean_d + 3.0 * std_d,
           "training_mean_distance": mean_d, "training_std_distance": std_d}


def in_domain(X_scaled: np.ndarray, ad: Dict[str, Any]) -> np.ndarray:
    """
    Checks query points against a model's applicability domain.

    :param X_scaled: Query feature matrix, scaled with the SAME
        :class:`~sklearn.preprocessing.StandardScaler` fitted alongside `ad` (stored beside it in
        the model pickle as ``"scaler"``) -- a query scaled any other way is being compared against
        a distance bound it cannot mean anything against.
    :param ad: An applicability-domain dict from :func:`_applicability_domain`.
    :returns: One bool per row of `X_scaled`; ``True`` when within the domain. Every row is
        ``False`` when the model recorded no domain at all (``ad["index"] is None``), which fails
        toward the more visible warning rather than silently skipping the check.
    """
    if not ad or ad.get("index") is None:
        return np.zeros(len(X_scaled), dtype=bool)
    distances, _ = ad["index"].kneighbors(X_scaled, n_neighbors=ad["k"])
    return distances.mean(axis=1) <= ad["threshold"]


#: Assays whose published value is a continuous potency, so predicting them is a REGRESSION
#: problem, not a classification one. "How strong is this peptide against E. coli" and "is this
#: peptide antibacterial" are different questions and cannot share an evaluation.
REGRESSION_ASSAYS: Tuple[str, ...] = (
    "MIC", "MIC50", "MIC80", "MIC90", "MIC100", "MBC", "MBC50", "MBC90", "MFC", "MLC", "MHC",
    "IC50", "IC90", "EC50", "EC90", "LD50", "LD90", "LC50", "CC50", "CC90", "ED50", "ED90",
    "GI50", "TC50", "TD50", "HC50", "MBIC", "MBEC", "ZOI",
)


def learning_task_for(assay_type: str) -> str:
    """
    Decides whether an assay is predicted by regression or classification.

    :param assay_type: Assay name, or ``any`` when the task is not assay-specific.
    :returns: ``regression`` when the assay has a continuous potency value, otherwise
        ``classification``. A class label like "is this anticancer" has no value to regress on.
    """
    return "regression" if str(assay_type).strip() in REGRESSION_ASSAYS else "classification"


def _best_fbeta_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0) -> float:
    """
    The score cutoff that maximizes F-beta on `(y_true, y_prob)`.

    XGBoost's default operating point is 0.5, which is the right cutoff for a balanced problem but
    not generally for one of this project's tasks: `scale_pos_weight` corrects the TRAINING loss
    for class imbalance (positives routinely a small fraction of the decoy-heavy negative class),
    but does nothing to the decision boundary the fitted model reports at serving time -- the two
    are independent knobs. Left at 0.5, a scale_pos_weight-corrected model still under-predicts the
    minority class: high precision, suppressed recall, exactly the pattern this project's own
    benchmark against external tools kept surfacing even after the imbalance fix landed. Uses
    :func:`~sklearn.metrics.precision_recall_curve` rather than a manual sweep so every candidate
    cut point actually present in the data is considered, not just a fixed grid.

    `beta` sets which point on the SAME precision-recall curve counts as "best" -- it does not
    change what the model learned, only which of its own tradeoffs gets reported and served.
    ``beta=1`` (F1, the default -- equal weight) is what every caller used before this parameter
    existed. ``beta>1`` weights recall more heavily: the reference antibacterial tools this project
    benchmarks against (Macrel, AMPlify, AMPidentifier, amPEPpy) all operate at recall 0.78-0.94
    against sAMPle's F1-calibrated 0.42 or worse, at correspondingly lower precision than sAMPle's
    -- evidence that those tools' own authors chose a more permissive operating point on their own
    curves, not that their curves discriminate better (this project's own ROC-AUC has stayed
    competitive with or ahead of most of them throughout). Moving `beta` up is choosing a
    comparable operating point on sAMPle's curve, the same lever, not a different one.

    :param y_true: Binary ground truth.
    :param y_prob: Predicted probability of the positive class.
    :param beta: Recall weight relative to precision in the optimized score
        (``(1+beta**2)*P*R / (beta**2*P+R)``). 1.0 is F1; above 1.0 favors recall, below 1.0 favors
        precision.
    :returns: The F-beta-maximizing threshold, or 0.5 when `y_true` has only one class (F-beta is
        undefined without both).
    """
    from sklearn.metrics import precision_recall_curve

    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    # precision/recall carry one more entry than thresholds (the last pair is threshold=+inf, i.e.
    # predicting nothing positive) -- drop it so the two arrays align with `thresholds` 1:1.
    p, r = precision[:-1], recall[:-1]
    beta2 = beta * beta
    fbeta = (1 + beta2) * p * r / np.clip(beta2 * p + r, 1e-12, None)
    return float(thresholds[int(np.argmax(fbeta))])


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """F1 special case (``beta=1``) of :func:`_best_fbeta_threshold` -- kept as its own name since
    every existing caller and test refers to F1 specifically, not the general F-beta family."""
    return _best_fbeta_threshold(y_true, y_prob, beta=1.0)


def _oof_calibrate_threshold(X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
                             factory: Callable[..., Any], model_params: Optional[Dict[str, Any]],
                             n_folds: int) -> float:
    """
    K-fold, grouped out-of-fold threshold calibration -- every row of `X_train` scored by a fold
    model that did not see it during that fold's fit, pooled into one F1-optimal cutoff
    (:func:`_best_f1_threshold`) over the whole training portion rather than 20% of it from a
    single split. Shared by :func:`fit_and_validate`'s two calibration paths: the
    fixed-hyperparameter estimator's own real per-fold refits, and a cheap-refit-capable
    architecture's frozen refits of its already-found configuration (see that function's own
    docstring for why the second case no longer needs to fall back to a noisier single split).

    :param X_train: The training portion only -- calibration must never see a holdout row.
    :param y_train: Training labels.
    :param groups_train: Cluster group id per training row, so no analog of a fold's held-out
        peptide leaks into that same fold's fit.
    :param factory: Builds each fold's estimator, called as ``factory(is_regression=False,
        **model_params)``.
    :param model_params: Hyperparameters (e.g. ``scale_pos_weight``) to fit each fold with.
    :param n_folds: Requested fold count. Below 2, calibration is skipped entirely (returns 0.5) --
        callers wanting the single-split fallback build it themselves, as
        :func:`fit_and_validate`'s non-cheap-refit path still does.
    :returns: The pooled F1-optimal threshold, or 0.5 when there are too few groups for even one
        fold, every fold's fit portion collapsed to a single class, or `n_folds` < 2.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if n_folds < 2:
        return 0.5
    try:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(X_train, y_train, groups_train))
    except ValueError:
        folds = []

    oof_probs = np.full(len(y_train), np.nan, dtype=np.float64)
    for fold_i, (fit_idx, calib_idx) in enumerate(folds, 1):
        if len(np.unique(y_train[fit_idx])) < 2:
            continue
        t_fold = time.time()
        fold_model = factory(is_regression=False, **(model_params or {}))
        fold_model.fit(X_train[fit_idx], y_train[fit_idx])
        oof_probs[calib_idx] = _predict_scores(fold_model, X_train[calib_idx], is_regression=False)
        logger.info(f"    calibration fold {fold_i}/{len(folds)} done in "
                   f"{time.time() - t_fold:.1f}s")

    covered = ~np.isnan(oof_probs)
    if covered.sum() > 0 and len(np.unique(y_train[covered])) == 2:
        return _best_f1_threshold(y_train[covered], oof_probs[covered])
    return 0.5


def _scores(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Classification metrics.

    Both threshold-free and threshold-based numbers are reported. The threshold-free pair (ROC-AUC,
    average precision) is what ranks models; accuracy on a balanced decoy set is the least
    informative number here and is included only because it is the one people ask for. Balanced
    accuracy is reported alongside it so an imbalanced per-organism task cannot look good by
    predicting the majority class.

    :param y_true: Binary ground truth.
    :param y_prob: Predicted probability of the positive class.
    :param threshold: Score at or above which a prediction is called positive -- the model's own
        calibrated cutoff (see :func:`_best_f1_threshold`) when called from
        :func:`fit_and_validate`, 0.5 for any other caller that does not pass one.
    :returns: Metric name -> value.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 brier_score_loss, cohen_kappa_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
    pred = (y_prob >= threshold).astype(int)
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc = float("nan")
    return {
        "roc_auc": roc,
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
        # Calibration: how well the reported PROBABILITY matches reality, which matters because the
        # webserver shows that probability to a user rather than a bare label.
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Regression metrics for potency prediction.

    Rank correlation is reported next to R^2 on purpose: for screening, ordering peptides correctly
    matters more than predicting the exact micromolar value, and a model can be useful on the first
    while looking poor on the second.

    :param y_true: Observed values, log10-transformed by the caller.
    :param y_pred: Predicted values on the same scale.
    :returns: Metric name -> value.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                                 mean_squared_error, r2_score)
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    try:
        out["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        out["mape"] = float("nan")
    try:
        out["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        out["pearson_r"] = float("nan")
    try:
        out["spearman_r"] = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        out["spearman_r"] = float("nan")
    return out


def _bootstrap_samples(y_true: np.ndarray, y_score: np.ndarray,
                       n_resamples: int = N_BOOTSTRAP,
                       is_regression: bool = False, threshold: float = 0.5) -> Dict[str, List[float]]:
    """
    Raw per-resample metric values -- the shared basis for the confidence interval
    (:func:`_bootstrap_ci`) and for the real-vs-null significance test (:func:`_compare_to_null`).

    :param y_true: Ground truth for the held-out set (class 0/1, or continuous for regression).
    :param y_score: Predicted probabilities (classifier) or predicted values (regressor).
    :param n_resamples: Bootstrap resamples.
    :param is_regression: Score each resample with regression metrics rather than classification.
    :param threshold: Classifier decision cutoff, forwarded to :func:`_score`; ignored for a
        regressor.
    :returns: Metric name -> one value per usable resample. A classification resample of
        all-one-class is skipped (ROC-AUC is undefined for it); a regression resample with no
        variance in `y_true` is skipped (R^2 is undefined for it).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    collected: Dict[str, List[float]] = {}
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if is_regression:
            if float(np.var(y_true[idx])) == 0.0:
                continue
        elif len(np.unique(y_true[idx])) < 2:
            continue
        for metric, value in _score(y_true[idx], y_score[idx], is_regression, threshold).items():
            collected.setdefault(metric, []).append(value)
    return collected


def _bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                  n_resamples: int = N_BOOTSTRAP,
                  is_regression: bool = False, threshold: float = 0.5) -> Dict[str, List[float]]:
    """
    Percentile confidence intervals for the held-out metrics.

    :param y_true: Ground truth for the held-out set.
    :param y_score: Predicted probabilities (classifier) or values (regressor).
    :param n_resamples: Bootstrap resamples.
    :param is_regression: Use regression metrics.
    :param threshold: Classifier decision cutoff, forwarded to :func:`_bootstrap_samples`; ignored
        for a regressor.
    :returns: Metric -> ``[lower, upper]`` at 95%. Without this, two feature blocks differing by
        0.01 on a metric look different when they are not.
    """
    samples = _bootstrap_samples(y_true, y_score, n_resamples, is_regression, threshold)
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for m, v in samples.items() if v}


def _y_randomization(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                     y_test: np.ndarray, model_params: Optional[Dict[str, int]] = None,
                     n_repeats: int = N_Y_RANDOMIZATIONS,
                     is_regression: bool = False, threshold: float = 0.5,
                     model_factory: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """
    Refits on shuffled TRAINING labels and scores on the SAME held-out test rows the real model
    was evaluated on; the score must collapse to chance.

    This is the control that makes the headline number meaningful. A model that still separates
    shuffled labels is reading something structural -- duplicated sequences across folds, an index
    column, a leaked identifier -- and its real score cannot be trusted whatever it says. Evaluating
    on the SAME test rows every repeat (rather than a fresh internal split per repeat) is what makes
    the resulting distribution directly comparable to the real model's own bootstrap distribution
    in :func:`_compare_to_null` -- both describe performance on the identical population, so a
    difference between them is a difference in the model, not in which rows happened to be held out.

    :param X_train: Training features (the real model's own training split).
    :param y_train: Training labels, shuffled internally each repeat.
    :param X_test: The held-out test features -- fixed, never shuffled.
    :param y_test: The held-out test labels -- fixed, real, never shuffled.
    :param model_params: Hyperparameters to fit each repeat with; the real model's own selected
        configuration when given, so the null control uses the same architecture being validated.
    :param n_repeats: Shuffled-label refits; :data:`N_Y_RANDOMIZATIONS` when omitted.
    :param is_regression: Build regressors and score with regression metrics.
    :param threshold: Classifier decision cutoff the real model was scored at (see
        :func:`_best_f1_threshold`), reused here so the null distribution's threshold-based
        metrics (F1, recall, ...) are comparable to the real model's rather than measured at a
        different operating point; ignored for a regressor.
    :param model_factory: Builds each repeat's estimator, called as
        ``factory(is_regression=..., **model_params)`` exactly like :func:`_make_model` --
        :func:`_make_model` itself when omitted. Must match whatever architecture produced
        `model_params`/the real model being validated (see :func:`fit_and_validate`).
    :returns: ``{"mean_<metric>", "max_<metric>", "per_repeat"}`` for the primary metric (ROC-AUC
        for a classifier, R^2 for a regressor) -- summary plus the full per-repeat metric dicts,
        which :func:`_compare_to_null` treats as the null distribution. A collapse to chance
        (AUC near 0.5, or R^2 near 0/negative) is the expected result.
    """
    factory = model_factory if model_factory is not None else _make_model
    metric = _primary_metric(is_regression)
    rng = np.random.default_rng(RANDOM_STATE)
    per_repeat: List[Dict[str, float]] = []
    # tqdm renders live for an interactive terminal; the explicit logger.info below is what
    # actually shows up in a redirected-to-file log (a `make step_N > log 2>&1` run, which is how
    # this pipeline is normally run) -- tqdm's carriage-return updates either vanish
    # (`leave=False` erases the bar on completion) or land as unreadable raw `\r` bytes once
    # there is no terminal to render them, so a refit-per-repeat architecture (a FLAML search, a
    # transformer fine-tune) previously produced total silence for the entire repeat, indistin-
    # guishable in a log file from a hang.
    for i in tqdm(range(n_repeats), desc="y-randomization", unit="repeat", leave=False):
        t_repeat = time.time()
        y_shuffled = rng.permutation(y_train)
        if not is_regression and len(np.unique(y_shuffled)) < 2:
            logger.info(f"    y-randomization repeat {i + 1}/{n_repeats}: skipped "
                       f"(shuffled labels collapsed to a single class)")
            continue
        logger.info(f"    y-randomization repeat {i + 1}/{n_repeats} starting...")
        model = factory(is_regression=is_regression, **(model_params or {}))
        model.fit(X_train, y_shuffled)
        preds = _predict_scores(model, X_test, is_regression)
        scores = _score(y_test, preds, is_regression, threshold)
        per_repeat.append(scores)
        logger.info(f"    y-randomization repeat {i + 1}/{n_repeats} done in "
                   f"{time.time() - t_repeat:.1f}s: {metric}={scores[metric]:.3f}")
    if not per_repeat:
        return {f"mean_{metric}": float("nan"), f"max_{metric}": float("nan"), "per_repeat": []}
    values = [s[metric] for s in per_repeat]
    return {f"mean_{metric}": float(np.nanmean(values)), f"max_{metric}": float(np.nanmax(values)),
           "per_repeat": per_repeat}


def _cross_validate(X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
                    is_regression: bool, threshold: float,
                    model_factory: Optional[Callable[..., Any]],
                    model_params: Optional[Dict[str, Any]], n_folds: int) -> Dict[str, Any]:
    """
    K-fold, grouped cross-validation over the TRAINING portion only -- a robustness check on the
    SAME configuration the holdout score already validated, not a second round of model selection.
    Nothing here chooses a different architecture or hyperparameter set; every fold refits
    `model_factory` with the identical `model_params` the real model was fit with.

    Deliberately NOT the mechanism that picks a model: this project's own history removed
    selection-oriented cross-validation once already, because a fixed-hyperparameter estimator has
    nothing to select among (see :func:`fit_and_validate`'s own docstring). This is the opposite
    use of the same technique -- reporting how stable the chosen configuration's score is across
    resamples of the training data, alongside the single holdout split and the y-randomization
    null, not instead of either.

    `model_factory` is expected to already be resolved to whatever's cheap for repeated refitting
    -- callers pass the same cheap factory used for y-randomization
    (:meth:`sample.flaml_xgboost.FlamlXGBoostEstimator.y_randomization_factory`) when the real
    factory is an expensive search, for the identical reason: N_SPLITS additional real FLAML
    searches would cost N_SPLITS x TIME_BUDGET_SECONDS for a number that is a robustness check,
    not the headline result.

    :param X_train: The real model's own training split (not the full dataset -- CV must not see
        any holdout row).
    :param y_train: Training labels.
    :param groups_train: Cluster group id per training row, so no analog of a fold's held-out
        peptide leaks into that same fold's fit.
    :param is_regression: Build regressors and score with regression metrics.
    :param threshold: Classifier decision cutoff the real model was scored at, reused here for the
        same reason :func:`_y_randomization` reuses it -- comparable operating point.
    :param model_factory: Builds each fold's estimator; :func:`_make_model` when ``None``.
    :param model_params: Hyperparameters (e.g. ``scale_pos_weight``) to fit each fold with.
    :param n_folds: Requested fold count.
    :returns: ``{"mean_<metric>", "std_<metric>"}`` for every key :func:`_score` reports, plus
        ``"per_fold"``. Empty (only ``"per_fold": []``) when there are too few groups for even one
        fold, or every fold's fit portion collapsed to a single class.
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    factory = model_factory if model_factory is not None else _make_model
    try:
        if is_regression:
            folds = list(GroupKFold(n_splits=n_folds).split(X_train, y_train, groups_train))
        else:
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                            random_state=RANDOM_STATE)
            folds = list(splitter.split(X_train, y_train, groups_train))
    except ValueError:
        folds = []

    per_fold: List[Dict[str, float]] = []
    for fold_i, (fit_idx, val_idx) in enumerate(folds, 1):
        if not is_regression and len(np.unique(y_train[fit_idx])) < 2:
            logger.info(f"    cross-validation fold {fold_i}/{len(folds)}: skipped "
                       f"(fit portion collapsed to a single class)")
            continue
        t_fold = time.time()
        model = factory(is_regression=is_regression, **(model_params or {}))
        model.fit(X_train[fit_idx], y_train[fit_idx])
        preds = _predict_scores(model, X_train[val_idx], is_regression)
        scores = _score(y_train[val_idx], preds, is_regression, threshold)
        per_fold.append(scores)
        logger.info(f"    cross-validation fold {fold_i}/{len(folds)} done in "
                   f"{time.time() - t_fold:.1f}s: "
                   f"{_primary_metric(is_regression)}={scores[_primary_metric(is_regression)]:.3f}")

    result: Dict[str, Any] = {"per_fold": per_fold}
    if not per_fold:
        return result
    for key in per_fold[0]:
        values = [f[key] for f in per_fold if f.get(key) == f.get(key)]  # drop NaN
        if values:
            result[f"mean_{key}"] = float(np.mean(values))
            result[f"std_{key}"] = float(np.std(values))
    return result


#: Below this Shapiro-Wilk p-value, a distribution is treated as non-normal. The conventional
#: significance level; :func:`_normality_test` is a screening step for which comparison test to
#: trust, not the headline result, so there is no reason to depart from it.
NORMALITY_ALPHA: float = 0.05


def _normality_test(values: np.ndarray, alpha: float = NORMALITY_ALPHA) -> Dict[str, Any]:
    """
    Shapiro-Wilk test for whether a metric distribution looks normal.

    This exists because :func:`_compare_to_null` has two ways to compare the real model's bootstrap
    distribution against the y-randomization null, and they answer the same question differently:
    Welch's t-test compares means and is most powerful when both distributions are roughly normal;
    Kruskal-Wallis compares ranks and makes no shape assumption but has less power when normality
    actually holds. Picking one without checking is picking blind, so this runs first and the
    result is stored alongside the two test outcomes rather than discarded once a choice is made.

    :param values: One metric's samples (bootstrap resamples, or y-randomization repeats).
    :param alpha: Significance level; :data:`NORMALITY_ALPHA` when omitted.
    :returns: ``{"statistic", "pvalue", "is_normal", "n"}``. ``is_normal`` is ``True`` when Shapiro-
        Wilk does NOT reject normality (``pvalue > alpha``) -- the conventional reading, though it
        means "not shown to be non-normal" rather than "confirmed normal". Fewer than 3 finite
        values, or a constant array (Shapiro is undefined for zero variance), reports
        ``is_normal=False`` so the caller falls back to the assumption-free test rather than
        dividing by zero or crashing.
    """
    from scipy.stats import shapiro

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3 or np.allclose(finite, finite[0]):
        return {"statistic": float("nan"), "pvalue": float("nan"), "is_normal": False,
               "n": int(len(finite))}
    try:
        stat, p = shapiro(finite)
    except Exception:
        return {"statistic": float("nan"), "pvalue": float("nan"), "is_normal": False,
               "n": int(len(finite))}
    return {"statistic": float(stat), "pvalue": float(p), "is_normal": bool(p > alpha),
           "n": int(len(finite))}


def _compare_to_null(bootstrap_samples: Dict[str, List[float]],
                     null_per_repeat: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Tests whether the real model's bootstrap metric distribution is significantly different from
    the y-randomization (null) distribution of the same metric, matching what the user asked this
    validation to establish beyond a bare "mean is near 0.5" read of :func:`_y_randomization`.

    Both a parametric (Welch's t-test, which does not assume equal variance -- the two
    distributions have very different sample sizes, :data:`N_BOOTSTRAP` versus
    :data:`N_Y_RANDOMIZATIONS`) and a non-parametric (Kruskal-Wallis, which assumes nothing about
    the shape of either distribution) test are computed and reported side by side. Which one is
    actually load-bearing is decided by :func:`_normality_test` run on each distribution first: the
    t-test is trusted only when BOTH the real and null distributions fail to reject normality,
    otherwise Kruskal-Wallis is -- the statistically appropriate choice given what the normality
    check found, rather than always reading both and hoping they agree.

    :param bootstrap_samples: The real model's per-metric bootstrap resamples, from
        :func:`_bootstrap_samples`.
    :param null_per_repeat: The y-randomization control's per-repeat metric dicts, from
        :func:`_y_randomization`'s ``"per_repeat"``.
    :returns: Metric -> ``{"real_mean", "null_mean", "t_statistic", "t_pvalue",
        "kruskal_statistic", "kruskal_pvalue", "real_normality", "null_normality",
        "primary_test", "primary_pvalue"}``. ``primary_test``/``primary_pvalue`` are the
        normality-selected pair; :func:`assess_quality` gates on those rather than requiring both
        tests to agree.
    """
    from scipy.stats import kruskal, ttest_ind

    if not null_per_repeat:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    metrics = set(bootstrap_samples) & set(null_per_repeat[0])
    for metric in metrics:
        real_vals = np.asarray([v for v in bootstrap_samples[metric] if np.isfinite(v)])
        null_vals = np.asarray([r[metric] for r in null_per_repeat
                                if np.isfinite(r.get(metric, np.nan))])
        if len(real_vals) < 2 or len(null_vals) < 2:
            continue
        try:
            t_stat, t_p = ttest_ind(real_vals, null_vals, equal_var=False)
        except Exception:
            t_stat, t_p = float("nan"), float("nan")
        try:
            h_stat, h_p = kruskal(real_vals, null_vals)
        except Exception:
            h_stat, h_p = float("nan"), float("nan")

        real_normality = _normality_test(real_vals)
        null_normality = _normality_test(null_vals)
        both_normal = real_normality["is_normal"] and null_normality["is_normal"]
        primary_test = "t_test" if both_normal else "kruskal"
        primary_pvalue = t_p if both_normal else h_p

        out[metric] = {
            "real_mean": float(np.mean(real_vals)), "null_mean": float(np.mean(null_vals)),
            "t_statistic": float(t_stat), "t_pvalue": float(t_p),
            "kruskal_statistic": float(h_stat), "kruskal_pvalue": float(h_p),
            "real_normality": real_normality, "null_normality": null_normality,
            "primary_test": primary_test, "primary_pvalue": float(primary_pvalue),
        }
    return out


class QualityGate(NamedTuple):
    """
    The bar a model must clear to be published.

    :param min_holdout_auc: Minimum held-out ROC-AUC.
    :param min_ci_lower: Minimum lower bound of the 95% bootstrap interval. This is the one that
        rejects a good-looking score computed on too few samples: a wide interval reaching toward
        chance means the number is not established, whatever its point estimate.
    :param max_y_random_auc: Maximum mean ROC-AUC under shuffled labels. A model that still
        separates randomized labels is reading leakage, and its real score cannot be trusted.
    :param max_null_pvalue: Maximum p-value for ROC-AUC against the y-randomization null
        distribution (:func:`_compare_to_null`). A point estimate can clear every threshold above
        by chance on a small holdout; requiring the real distribution to be statistically
        distinguishable from the null is the check that catches that. The p-value checked is
        whichever of the t-test or Kruskal-Wallis test :func:`_normality_test` selected as
        appropriate for the two distributions being compared (``"primary_pvalue"``), not a
        `require-both` combination of the two -- running an assumption-laden test and an
        assumption-free one on the same data and requiring both to agree is not more rigorous
        than picking the one whose assumptions actually hold.
    """
    min_holdout_auc: float = 0.75
    min_ci_lower: float = 0.65
    max_y_random_auc: float = 0.60
    max_null_pvalue: float = 0.05


#: The gate applied by :func:`train_all`. Deliberately strict: this pipeline exists to publish a
#: database, and a weak model shipped alongside it is read as a claim about the data.
DEFAULT_QUALITY_GATE = QualityGate()


class RegressionQualityGate(NamedTuple):
    """
    The bar a REGRESSION model must clear to be published, keyed on R^2 rather than ROC-AUC.

    Thresholds are deliberately lower than the classifier's AUC thresholds, and not because the bar
    is softer: predicting a continuous potency (MIC, IC50, FICI) from sequence is a genuinely harder
    problem than separating actives from shuffled decoys, and published potencies carry real
    measurement noise (different labs, media, inocula) that caps the achievable R^2 well below 1. An
    R^2 of 0.3 -- the model explaining 30 % of the variance in log potency -- is a real, useful
    signal here, where the same number would be a weak classifier.

    :param min_holdout_r2: Minimum held-out R^2.
    :param min_ci_lower_r2: Minimum lower bound of the 95% bootstrap R^2 interval; above 0 means
        the model beats predicting the training mean even at the pessimistic end of the interval.
    :param max_y_random_r2: Maximum mean R^2 under shuffled labels. A sound regressor collapses to
        R^2 at or below 0 (no better than the mean) on shuffled potencies.
    :param max_null_pvalue: As :attr:`QualityGate.max_null_pvalue`, but for R^2.
    """
    min_holdout_r2: float = 0.30
    min_ci_lower_r2: float = 0.10
    max_y_random_r2: float = 0.10
    max_null_pvalue: float = 0.05


DEFAULT_REGRESSION_QUALITY_GATE = RegressionQualityGate()


def assess_quality(record: Dict[str, Any], gate: QualityGate = DEFAULT_QUALITY_GATE,
                   regression_gate: RegressionQualityGate = DEFAULT_REGRESSION_QUALITY_GATE
                   ) -> Tuple[bool, List[str]]:
    """
    Judges one trained model against the quality gate appropriate to its learning task.

    :param record: A result record from :func:`train_task`.
    :param gate: Classifier thresholds (keyed on ROC-AUC).
    :param regression_gate: Regressor thresholds (keyed on R^2).
    :returns: ``(passed, reasons)``; `reasons` lists every criterion failed, so a rejected model
        says why rather than merely disappearing.
    """
    if str(record.get("learning_task", "classification")).lower() == "regression":
        return _assess_regression_quality(record, regression_gate)

    reasons: List[str] = []
    holdout = record.get("holdout", {}).get("roc_auc", float("nan"))
    ci = record.get("holdout_ci95", {}).get("roc_auc", [float("nan"), float("nan")])
    yrand = record.get("y_randomization", {}).get("mean_roc_auc", float("nan"))

    if not np.isfinite(holdout) or holdout < gate.min_holdout_auc:
        reasons.append(f"held-out ROC-AUC {holdout:.3f} < {gate.min_holdout_auc}")
    if ci and np.isfinite(ci[0]) and ci[0] < gate.min_ci_lower:
        reasons.append(f"95% CI lower bound {ci[0]:.3f} < {gate.min_ci_lower}")
    if np.isfinite(yrand) and yrand > gate.max_y_random_auc:
        reasons.append(f"y-randomized ROC-AUC {yrand:.3f} > {gate.max_y_random_auc} (leakage)")

    null_auc = record.get("null_comparison", {}).get("roc_auc", {})
    primary_p = null_auc.get("primary_pvalue", float("nan"))
    primary_test = null_auc.get("primary_test", "kruskal")
    if np.isfinite(primary_p) and primary_p >= gate.max_null_pvalue:
        reasons.append(
            f"real vs. y-randomized ROC-AUC not significant ({primary_test} p={primary_p:.3g} "
            f">= {gate.max_null_pvalue}, chosen by a normality test on the two distributions) -- "
            f"the point-estimate gap above could be chance on this holdout")
    return (not reasons), reasons


def _assess_regression_quality(record: Dict[str, Any],
                               gate: RegressionQualityGate) -> Tuple[bool, List[str]]:
    """
    The regression branch of :func:`assess_quality`, keyed on R^2.

    :param record: A regression result record from :func:`train_task`.
    :param gate: R^2-based thresholds.
    :returns: ``(passed, reasons)``.
    """
    reasons: List[str] = []
    holdout = record.get("holdout", {}).get("r2", float("nan"))
    ci = record.get("holdout_ci95", {}).get("r2", [float("nan"), float("nan")])
    yrand = record.get("y_randomization", {}).get("mean_r2", float("nan"))

    if not np.isfinite(holdout) or holdout < gate.min_holdout_r2:
        reasons.append(f"held-out R2 {holdout:.3f} < {gate.min_holdout_r2}")
    if ci and np.isfinite(ci[0]) and ci[0] < gate.min_ci_lower_r2:
        reasons.append(f"95% CI lower bound R2 {ci[0]:.3f} < {gate.min_ci_lower_r2}")
    if np.isfinite(yrand) and yrand > gate.max_y_random_r2:
        reasons.append(f"y-randomized R2 {yrand:.3f} > {gate.max_y_random_r2} (leakage)")

    null_r2 = record.get("null_comparison", {}).get("r2", {})
    primary_p = null_r2.get("primary_pvalue", float("nan"))
    primary_test = null_r2.get("primary_test", "kruskal")
    if np.isfinite(primary_p) and primary_p >= gate.max_null_pvalue:
        reasons.append(
            f"real vs. y-randomized R2 not significant ({primary_test} p={primary_p:.3g} "
            f">= {gate.max_null_pvalue}) -- the point-estimate gap above could be chance")
    return (not reasons), reasons


def train_task(df: pd.DataFrame, y: pd.Series, columns: Sequence[str],
               task_name: str, block_name: str, groups: np.ndarray,
               learning_task: str = "classification",
               model_factory: Optional[Callable[..., Any]] = None,
               calibrate_threshold: bool = True,
               n_calibration_folds: int = N_SPLITS,
               n_y_randomizations: int = N_Y_RANDOMIZATIONS,
               n_cv_folds: int = N_SPLITS) -> Optional[Dict[str, Any]]:
    """
    Trains and validates one (task, feature block) cell of the matrix.

    Order of operations, and why it is this order: (1) split off a holdout set, grouped by
    near-identity cluster so no analog of a held-out peptide leaks into training; (2) fit on the
    train portion with fixed hyperparameters; (3) score the fitted model on the holdout set, which
    nothing before this point has touched -- the final word, per this project's validation policy;
    (4) K-fold cross-validate the same chosen configuration on the train portion, as a robustness
    diagnostic, never a different model than the one the holdout scored; (5) run y-randomization
    evaluated on that SAME holdout set, so its null distribution is directly comparable to the real
    model's bootstrap distribution
    (:func:`_compare_to_null`); (6) fit the model's applicability domain on the train/selection
    portion's feature space.

    Classification and regression differ in three places, all downstream of the same split: the
    estimator (:func:`_make_model`), the metric set (:func:`_score`), and the holdout splitter
    (a regressor's continuous `y` cannot be stratified, so grouped-but-unstratified k-fold is used).

    :param df: Full dataset.
    :param y: Task labels; ``NaN`` rows are dropped. For classification these are 0/1 (decoy /
        positive); for regression they are the continuous target (e.g. log10 MIC), and decoys
        already carry ``NaN`` because a shuffled sequence has no measured potency.
    :param columns: Feature columns for this block.
    :param task_name: Task identifier, for the report.
    :param block_name: Feature block identifier, for the report.
    :param groups: Cluster group id per row of `df` (same length and row order), from
        :func:`_cluster_groups` -- computed once for the whole dataset and passed in, rather than
        recomputed per task, since the clustering does not depend on which task is being trained.
    :param learning_task: ``"classification"`` or ``"regression"``.
    :param model_factory: Forwarded to :func:`fit_and_validate`; ``None`` (the default XGBoost/
        RandomForest estimator) for every existing caller. A non-default factory is also how
        ``columns == ["sequence"]`` is expected to be paired with an estimator that actually
        accepts raw sequence strings (e.g. a fine-tuned transformer) rather than a numeric matrix.
    :param calibrate_threshold: Forwarded to :func:`fit_and_validate`; ``True`` (F1-calibrated)
        when omitted. ``False`` pins the conventional 0.5 instead -- fixed, not derived from any
        fold of the training data, so it cannot land at an unstable extreme calibration itself
        picked (see that function's own docstring on this tradeoff).
    :param n_calibration_folds: Forwarded to :func:`fit_and_validate`.
    :param n_y_randomizations: Forwarded to :func:`fit_and_validate`.
    :param n_cv_folds: Forwarded to :func:`fit_and_validate`.
    :returns: The result record, or ``None`` when the task has too little data after cleaning.
        Includes ``_train_entry_ids``/``_test_entry_ids`` -- the peptide ids on each side of the
        holdout split -- alongside the usual ``_model``/``_columns``/``_scaler``/
        ``_applicability_domain`` private keys :func:`train_all` pops before writing the report.
    """
    is_regression = str(learning_task).lower() == "regression"

    keep = y.notna().to_numpy()
    sub = df.loc[keep]
    task_groups = groups[keep]
    labels = (y.loc[keep].to_numpy(dtype=float) if is_regression
              else y.loc[keep].astype(int).to_numpy())
    if is_regression:
        if len(labels) < MIN_POSITIVES_PER_TASK or float(np.var(labels)) == 0.0:
            return None
    elif len(np.unique(labels)) < 2:
        return None

    # Deduplicate EXACT-identical sequences (near-identical ones are not split apart at all -- see
    # `groups` -- so this only removes true repeats, which would otherwise be counted twice).
    seqs = sub["sequence"].astype(str).str.strip().str.upper()
    dedup = (~seqs.duplicated()).to_numpy()
    sub, labels, task_groups = sub.loc[dedup], labels[dedup], task_groups[dedup]
    if is_regression:
        if len(labels) < MIN_POSITIVES_PER_TASK:
            return None
    elif int(labels.sum()) < MIN_POSITIVES_PER_TASK or int((labels == 0).sum()) < MIN_POSITIVES_PER_TASK:
        return None

    # A sequence-input block's estimator takes the raw string, not a computed numeric feature --
    # pd.to_numeric(...).fillna(0.0) would silently coerce every sequence to 0.0 rather than raise,
    # scoring on garbage without any visible sign of it (see _is_sequence_input).
    if _is_sequence_input(columns):
        X = sub[["sequence"]].astype(str).to_numpy(dtype=object)
    else:
        X = sub[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    # Which peptides actually landed in each split -- published per model so a claim like "0.91
    # AUC" can be checked against the real train/test composition rather than taken on faith.
    if "entry_id" in sub.columns:
        entry_ids = sub["entry_id"].astype(str).to_numpy()
    else:
        entry_ids = sub.index.astype(str).to_numpy()

    record = fit_and_validate(X, labels, task_groups, entry_ids, is_regression=is_regression,
                              model_factory=model_factory, skip_scaling=_is_sequence_input(columns),
                              calibrate_threshold=calibrate_threshold,
                              n_calibration_folds=n_calibration_folds,
                              n_y_randomizations=n_y_randomizations, n_cv_folds=n_cv_folds)
    record.update({"task": task_name, "feature_block": block_name, "n_features": len(columns)})
    record["_columns"] = list(columns)
    return record


def fit_and_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, entry_ids: np.ndarray,
                     is_regression: bool = False, calibrate_threshold: bool = True,
                     model_factory: Optional[Callable[..., Any]] = None,
                     skip_scaling: bool = False,
                     n_calibration_folds: int = N_SPLITS,
                     n_y_randomizations: int = N_Y_RANDOMIZATIONS,
                     n_cv_folds: int = N_SPLITS) -> Dict[str, Any]:
    """
    Fits and fully validates one model against an already-prepared feature matrix: a single
    grouped train/holdout split, fit with :func:`_make_model`'s fixed hyperparameters, scored on
    the untouched holdout, characterized with a bootstrap 95% confidence interval, checked against
    a y-randomization null, and given an applicability domain.

    Cross-validation is NOT selection here -- this project fixes its hyperparameters (or, for a
    FLAML-searched architecture, fixes them once via the real search) rather than choosing among
    candidates per fold, so :func:`_cross_validate` reports a fold-to-fold spread as an extra
    diagnostic alongside the single holdout number, never a different claim about the model and
    never a different model than the one the holdout scored. The holdout stays the one number
    directly comparable to a head-to-head external-tool comparison (Macrel, AntiCP2, ...): one
    fixed train/test split, one bootstrap 95% CI on each side, on the identical held-out sequences
    (see :mod:`sample.benchmark`) -- cross-validation and y-randomization are both computed
    alongside it, not instead of it.

    This is the validated-fit core :func:`train_task` uses -- everything from the split onward,
    extracted so a second caller with its own row shape (:mod:`sample.synergy`'s peptide+antibiotic
    rows, where the same peptide legitimately repeats across several rows and must not be
    deduplicated the way :func:`train_task` dedupes its own one-row-per-peptide input) can reuse
    the identical validation machinery without re-implementing any of it a second time.

    Order of operations, and why it is this order: (1) one grouped train/holdout split, so no
    analog of a held-out row leaks into training; (2) calibrate a decision threshold from a further
    internal split of the train portion ONLY (see :func:`_best_f1_threshold`) -- must happen before
    the holdout is ever scored, or the threshold would be tuned against the very number it is meant
    to validate; (3) fit on the full train portion with fixed hyperparameters; (4) score the fitted
    model on the holdout, which nothing before this point has touched -- the final word, per this
    project's validation policy; (5) bootstrap that holdout score for a 95% CI; (6) run
    y-randomization evaluated on the SAME holdout, so its null distribution is directly comparable
    to the real model's bootstrap distribution (:func:`_compare_to_null`); (7) fit the applicability
    domain on the train portion's feature
    space, never the holdout, for the same reason the holdout is never used for anything upstream
    of scoring: a domain fit including the evaluation rows would make every one of them trivially
    "in domain" by construction.

    :param X: Feature matrix, one row per training example.
    :param y: Labels for the same rows -- class 0/1 for a classifier, continuous for a regressor.
    :param groups: Group id per row (e.g. near-identity cluster id), same length as `X`/`y`. Rows
        sharing a group id never split across train and holdout.
    :param entry_ids: An identifier per row (e.g. peptide entry id), same length as `X`/`y`, for
        recording which rows landed on which side of the split.
    :param is_regression: Whether to build a regressor and score with regression metrics.
    :param calibrate_threshold: Calibrate the decision cutoff (see :func:`_best_f1_threshold`)
        rather than leaving it at the conventional 0.5. ``True`` for every caller except
        :mod:`sample.synergy`'s FICI classifier, whose probability is deliberately constructed so
        0.5 lands exactly on the domain's own synergy cutoff (see that module's
        ``predict_synergy``, which serves at a hardcoded 0.5 independent of anything this function
        returns) -- recalibrating that model's reported threshold away from 0.5 would describe
        holdout performance at an operating point the served predictions never actually use.
        Ignored for a regressor, which has no decision boundary to calibrate.
    :param model_factory: Builds the estimator, called as ``factory(is_regression=..., **params)``
        exactly like :func:`_make_model` -- :func:`_make_model` itself when omitted, so every
        existing caller (the XGBoost/RandomForest path, :mod:`sample.synergy`) is unaffected. A
        non-default factory is what lets the null-control refit (:func:`_y_randomization`) and
        every internal calibration/final fit here use an architecture other than the fixed
        XGBoost hyperparameters -- e.g. a fine-tuned transformer -- while every other safeguard
        (grouped split, threshold calibration, bootstrap CI, y-randomization) stays identical.
    :param skip_scaling: Skip fitting a :class:`~sklearn.preprocessing.StandardScaler` and
        applicability domain on `X_train`. Needed whenever `X` is not a numeric matrix (a raw
        sequence-string column, for a `model_factory` whose estimator embeds sequences itself) --
        both assume numeric input and would raise or produce a meaningless "domain" otherwise.
        ``False`` for every existing caller; the returned ``_scaler``/``_applicability_domain`` are
        ``None``/the same "no domain" shape :func:`_applicability_domain` already returns for a
        too-small task, which :func:`sample.inference._in_domain_flags` already null-guards on.
    :param n_calibration_folds: Folds for the out-of-fold threshold calibration
        (:data:`N_SPLITS` -- 5 -- when omitted, matching the outer holdout split). Meaning depends
        on whether `model_factory` exposes a cheap-refit hook (see the ordering note above): with
        one, this is raised to at least :data:`N_SPLITS` regardless (a lower value tuned for the
        old expensive-search cost is no longer honored as a reason to calibrate worse, now that
        calibration refits are cheap); without one, a value of 1 falls back to a single 80/20
        calibration split instead of K-fold OOF -- noisier (see :func:`_best_f1_threshold`'s
        calling comment), a deliberate, explicit compute-cost tradeoff for a genuinely expensive,
        cheap-refit-less architecture (a from-scratch search or fine-tune with no found-
        configuration hook -- not one this project currently ships).
    :param n_y_randomizations: Forwarded to :func:`_y_randomization`; :data:`N_Y_RANDOMIZATIONS`
        when omitted. Same tradeoff as `n_calibration_folds`: fewer repeats weakens the
        null-comparison's own statistical power and should be treated as a known, reported
        reduction in confidence, not silently equivalent to the default.
    :param n_cv_folds: Forwarded to :func:`_cross_validate`; :data:`N_SPLITS` when omitted. Folds
        below 2 skip cross-validation entirely (an empty ``per_fold``) rather than raising, the
        same degrade-gracefully policy `n_calibration_folds`/`n_y_randomizations` already follow.
    :returns: ``learning_task``, ``n_samples``, ``n_positives``, ``n_negatives``, ``threshold``
        (the calibrated decision cutoff -- ``0.5`` when `calibrate_threshold` is ``False``, ``None``
        for a regressor), ``holdout``,
        ``holdout_ci95``, ``cross_validation``, ``y_randomization``, ``null_comparison``, plus the
        private ``_model``, ``_scaler``, ``_applicability_domain``, ``_train_entry_ids``,
        ``_test_entry_ids`` keys a caller pops before writing the report (mirroring
        :func:`train_task`'s own contract).
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split
    from sklearn.preprocessing import StandardScaler

    factory = model_factory if model_factory is not None else _make_model
    labels = y

    # One grouped train/holdout split -- stratified for a classifier, plain grouped for a
    # regressor (a continuous target has no classes to stratify on). Built from a 5-fold grouped
    # splitter and keeping only its first split (an 80/20-ish partition) rather than a plain
    # `train_test_split`, so the split still respects `groups` -- no analog of a held-out peptide
    # can land in training. Falls back to an ungrouped 80/20 split only when grouping itself is
    # infeasible (too few groups for even one fold).
    try:
        if is_regression:
            train_idx, test_idx = next(GroupKFold(n_splits=N_SPLITS).split(X, labels, groups))
        else:
            splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            train_idx, test_idx = next(splitter.split(X, labels, groups))
    except ValueError:
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)), test_size=0.2,
            stratify=None if is_regression else labels, random_state=RANDOM_STATE)

    X_train, y_train = X[train_idx], labels[train_idx]
    X_test, y_test = X[test_idx], labels[test_idx]
    train_entry_ids = entry_ids[train_idx].tolist()
    test_entry_ids = entry_ids[test_idx].tolist()

    # Every classification task here is imbalanced by construction (a task's negatives are ALL
    # decoys -- 53,614 of them -- while its positives are only the peptides real for that specific
    # activity, target or assay, routinely a small fraction of that), and XGBoost's default
    # decision threshold (0.5 on an uncorrected model) systematically under-predicts the minority
    # (positive) class as a result: high precision, suppressed recall. scale_pos_weight -- the
    # negative:positive ratio actually present in THIS task's training split, not a fixed constant
    # -- is XGBoost's standard correction for exactly this, shifting the fitted model's operating
    # point back toward balanced rather than changing what it has learned to discriminate.
    # Regression has no classes to weight.
    model_params: Dict[str, Any] = {}
    if not is_regression:
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        if n_pos > 0:
            model_params["scale_pos_weight"] = n_neg / n_pos

    # Decision threshold: calibrated from OUT-OF-FOLD predictions on the training portion only,
    # never X_test/y_test -- picking the cutoff that maximizes a holdout's own F1 and then
    # reporting that same holdout's F1 would be leakage (tuning a hyperparameter against the
    # number it is meant to validate). scale_pos_weight above corrects the training loss for class
    # imbalance but leaves the fitted model's decision boundary at whatever 0.5 means for it, which
    # is a separate, uncorrected source of the same under-prediction (see
    # :func:`_best_f1_threshold`).
    #
    # A SINGLE throwaway train/calibration split (the first version of this) turned out too noisy
    # to trust: direct comparison against the true F1-optimal threshold on a real task's actual
    # holdout showed a single split can land far from it in either direction depending on which
    # ~20% of rows it happened to draw for calibration, an error large enough to cost real
    # recall/F1/MCC. Pooling out-of-fold predictions across every fold instead -- every row of
    # X_train gets scored by a model that did not see it during that fold's fit, so this remains
    # leakage-free -- uses all of X_train for the threshold estimate rather than 20% of it, which
    # is what actually stabilizes it. Regression has no decision boundary to calibrate.
    #
    # ORDERING: a factory that exposes the cheap-refit hook (see the cross-validation/
    # y-randomization comment below for what that is) gets the model fit FIRST, calibration
    # SECOND -- the opposite of the fixed-hyperparameter path immediately below, and worth being
    # explicit about why. The single 80/20 fallback existed only because a second real search per
    # calibration fold was too expensive to afford N_SPLITS times over; once the real model has
    # already been searched for once, calibrating against CHEAP refits of that SAME found
    # configuration costs no more than the fixed-hyperparameter path's own OOF calibration always
    # has, so there is no remaining reason to accept the single-split's extra noise for this
    # architecture. This also removes what used to be a SEPARATE "final fit" search entirely: the
    # model calibration refits are cloned from is the same one scored on the holdout below, not a
    # second, independently-searched model that calibration's threshold was never guaranteed to
    # transfer to exactly.
    threshold = 0.5
    # `getattr(..., "func", model_factory)` unwraps a `functools.partial` (e.g. a caller pinning
    # `time_budget=...` for a quick test) before the hasattr check -- `partial` does not proxy
    # attribute lookups to the callable it wraps, so checking `model_factory` directly would
    # silently read as "no cheap-refit hook" for a partial-wrapped FlamlXGBoostEstimator and fall
    # back to the slower path below with no error at all.
    supports_cheap_refit = model_factory is not None and hasattr(
        getattr(model_factory, "func", model_factory), "y_randomization_factory")
    cheap_model_factory = model_factory
    using_cheap_factory = False

    if supports_cheap_refit and not is_regression:
        logger.info("  Fitting model (real search) -- calibration reuses its found "
                   "configuration via cheap refits, not a second search...")
        t_search = time.time()
        model = factory(is_regression=is_regression, **model_params)
        model.fit(X_train, y_train)
        logger.info(f"  Model fit in {time.time() - t_search:.1f}s")

        cheap_hook = getattr(model, "y_randomization_factory", None)
        using_cheap_factory = callable(cheap_hook)
        cheap_model_factory = cheap_hook() if using_cheap_factory else model_factory

        if calibrate_threshold:
            t_calib = time.time()
            # No reason left to accept fewer than N_SPLITS folds here -- the caller's
            # n_calibration_folds exists to cap REAL search cost, which this path no longer pays
            # per fold, so a value below N_SPLITS from a caller still tuned for the old,
            # expensive-search cost is not honored as a reason to calibrate worse.
            effective_folds = max(n_calibration_folds, N_SPLITS) if using_cheap_factory \
                else n_calibration_folds
            logger.info(f"  Calibrating decision threshold: {effective_folds}-fold "
                       f"out-of-fold pooling"
                       f"{' via cheap frozen refits' if using_cheap_factory else ''}...")
            threshold = _oof_calibrate_threshold(X_train, y_train, groups[train_idx],
                                                 cheap_model_factory, model_params,
                                                 effective_folds)
            logger.info(f"  Threshold calibration done in {time.time() - t_calib:.1f}s: "
                       f"threshold={threshold:.4f}")
    else:
        if not is_regression and calibrate_threshold:
            t_calib = time.time()
            if n_calibration_folds >= 2:
                logger.info(f"  Calibrating decision threshold: {n_calibration_folds}-fold "
                           f"out-of-fold pooling ({n_calibration_folds} refits)...")
                threshold = _oof_calibrate_threshold(X_train, y_train, groups[train_idx], factory,
                                                     model_params, n_calibration_folds)
            else:
                # n_calibration_folds <= 1: a single 80/20 calibration split rather than K-fold
                # OOF -- noisier (this is the exact tradeoff the OOF version above replaced), but
                # a deliberate concession for an estimator with no cheap-refit hook that is too
                # expensive to refit n_calibration_folds extra times per task (a genuine
                # from-scratch search or fine-tune, not one this project currently ships).
                logger.info("  Calibrating decision threshold: single 80/20 split (1 refit)...")
                try:
                    thr_splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                                        random_state=RANDOM_STATE)
                    thr_fit_idx, thr_calib_idx = next(thr_splitter.split(X_train, y_train,
                                                                          groups[train_idx]))
                except ValueError:
                    thr_fit_idx, thr_calib_idx = train_test_split(
                        np.arange(len(y_train)), test_size=0.2, stratify=y_train,
                        random_state=RANDOM_STATE)
                y_calib = y_train[thr_calib_idx]
                if len(np.unique(y_train[thr_fit_idx])) == 2 and len(np.unique(y_calib)) == 2:
                    calib_model = factory(is_regression=False, **model_params)
                    calib_model.fit(X_train[thr_fit_idx], y_train[thr_fit_idx])
                    calib_probs = _predict_scores(calib_model, X_train[thr_calib_idx],
                                                  is_regression=False)
                    threshold = _best_f1_threshold(y_calib, calib_probs)
            logger.info(f"  Threshold calibration done in {time.time() - t_calib:.1f}s: "
                       f"threshold={threshold:.4f}")

        logger.info("  Fitting final model on the full training split...")
        t_final = time.time()
        model = factory(is_regression=is_regression, **model_params)
        model.fit(X_train, y_train)
        logger.info(f"  Final model fit in {time.time() - t_final:.1f}s")

    # HOLDOUT: the final word. Nothing above this line has seen X_test/y_test.
    preds = _predict_scores(model, X_test, is_regression)
    holdout = _score(y_test, preds, is_regression, threshold)
    bootstrap_samples = _bootstrap_samples(y_test, preds, is_regression=is_regression,
                                           threshold=threshold)

    # Cross-validation and y-randomization both refit the real model's OWN architecture many times
    # over (N_SPLITS folds, n_y_randomizations repeats) -- for an expensive-to-fit factory (a FLAML
    # search, a transformer fine-tune), re-running the SEARCH that many times answers a question
    # neither diagnostic asks (cross-validation is a robustness check on the chosen configuration,
    # not a second round of model selection; a y-randomization repeat only needs to show a
    # shuffled-label refit collapses to chance). Such a factory may expose a
    # `y_randomization_factory()` hook returning a cheap, no-search factory bound to the real
    # model's OWN found configuration (e.g. FlamlXGBoostEstimator.y_randomization_factory) --
    # resolved ONCE here and reused for both diagnostics below. Every other caller (the fixed-
    # hyperparameter XGBoost path has no search to skip in the first place) has no such hook, so
    # `getattr(..., None)` falls through to the unchanged `model_factory`.
    cheap_hook = getattr(model, "y_randomization_factory", None)
    using_cheap_factory = callable(cheap_hook)
    cheap_model_factory = cheap_hook() if using_cheap_factory else model_factory

    logger.info(f"  Running {n_cv_folds}-fold cross-validation on the training split"
               f"{' (cheap frozen refit, no search)' if using_cheap_factory else ''}...")
    t_cv = time.time()
    cross_validation = _cross_validate(X_train, y_train, groups[train_idx], is_regression,
                                       threshold, cheap_model_factory, model_params, n_cv_folds)
    logger.info(f"  Cross-validation done in {time.time() - t_cv:.1f}s")

    # Y-randomization, evaluated on the identical holdout rows, with the SAME scale_pos_weight and
    # the SAME calibrated threshold: a label permutation preserves the class counts exactly, so
    # scale_pos_weight is the same value that would be recomputed from any shuffled y_train, and the
    # null control must use the same architecture AND operating point being validated for the
    # comparison to mean anything.
    logger.info(f"  Running {n_y_randomizations} y-randomization repeat(s)"
               f"{' (cheap frozen refit, no search)' if using_cheap_factory else ''}...")
    t_yrand = time.time()
    yrand = _y_randomization(X_train, y_train, X_test, y_test, model_params=model_params,
                            is_regression=is_regression, threshold=threshold,
                            n_repeats=n_y_randomizations, model_factory=cheap_model_factory)
    logger.info(f"  Y-randomization done in {time.time() - t_yrand:.1f}s")
    null_comparison = _compare_to_null(bootstrap_samples, yrand.get("per_repeat", []))

    if skip_scaling:
        scaler = None
        ad = {"index": None, "k": 0, "threshold": 0.0,
             "training_mean_distance": float("nan"), "training_std_distance": float("nan")}
    else:
        scaler = StandardScaler().fit(X_train)
        ad = _applicability_domain(scaler.transform(X_train))

    return {
        "learning_task": "regression" if is_regression else "classification",
        "n_samples": int(len(labels)),
        "n_positives": None if is_regression else int(labels.sum()),
        "n_negatives": None if is_regression else int((labels == 0).sum()),
        "threshold": None if is_regression else float(threshold),
        "holdout": holdout,
        "holdout_ci95": {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                        for m, v in bootstrap_samples.items() if v},
        "cross_validation": {k: v for k, v in cross_validation.items() if k != "per_fold"},
        "y_randomization": {k: v for k, v in yrand.items() if k != "per_repeat"},
        "null_comparison": null_comparison,
        "_model": model,
        "_scaler": scaler,
        "_applicability_domain": ad,
        "_train_entry_ids": train_entry_ids,
        "_test_entry_ids": test_entry_ids,
    }


def train_all(dataset_csv: str, model_dir: str, report_path: str,
              activities_csv: str = "data/sample_raw_release/activities.csv",
              filter_anomalies: bool = False,
              tasks_limit: Optional[int] = None,
              gate: QualityGate = DEFAULT_QUALITY_GATE,
              cluster_work_dir: Optional[str] = None,
              min_task_examples: Optional[int] = None,
              min_target_examples: Optional[int] = None,
              families_json: Optional[str] = None,
              model_factory: Optional[Callable[..., Any]] = None,
              calibrate_threshold: bool = True,
              n_calibration_folds: int = N_SPLITS,
              n_y_randomizations: int = N_Y_RANDOMIZATIONS,
              n_cv_folds: int = N_SPLITS) -> Dict[str, Any]:
    """
    Trains the full task x feature-block matrix and writes the models and the report.

    Each kept model gets two files in its namespace directory: ``<block>.pkl`` (the estimator,
    columns, scaler and applicability domain -- what :mod:`sample.inference` loads to predict) and
    ``<block>.dataset.json`` (the full validation record plus the exact ``train_entry_ids``/
    ``test_entry_ids`` -- what :mod:`sample.model_registry`'s ``get_model_dataset`` and the
    webserver's per-model "dataset" page read, so a published AUC can be checked against which
    peptides were actually held out rather than taken on faith).

    :param dataset_csv: Assembled dataset with features, ``sequence`` and ``is_decoy``.
    :param model_dir: Directory for the pickled models, one per (task, block).
    :param report_path: Where to write the JSON report.
    :param activities_csv: Step 19 activity table supplying per-class labels.
    :param filter_anomalies: Drop rows flagged by the anomaly detector before training.
    :param tasks_limit: Train only the first N tasks; for smoke tests.
    :param gate: Quality thresholds. Models that fail are reported but not written to disk.
    :param cluster_work_dir: Scratch directory for :func:`_cluster_groups`'s PepMatch index files;
        that function's own default (``data/families``, shared with step 35) when omitted. A test
        harness should pass an isolated directory so it does not overwrite step 35's real working
        files with synthetic fixture data.
    :param min_task_examples: Forwarded to :func:`build_tasks`; :data:`MIN_POSITIVES_PER_TASK`
        when omitted.
    :param min_target_examples: Forwarded to :func:`build_tasks`; ``min_task_examples * 2`` when
        omitted.
    :param families_json: Forwarded to :func:`_family_groups`;
        ``data/families/families.json`` when omitted. A test harness should pass an isolated path
        (even a path to a file that does not exist, which degrades cleanly to plain near-identity
        clustering) so a run against synthetic fixture data does not silently pick up the real
        release's curated family assignment.
    :param model_factory: Forwarded to :func:`train_task` for every cell; the default XGBoost/
        RandomForest estimator (:func:`_make_model`) when omitted, unchanged from this function's
        original behavior. Pass :class:`sample.flaml_xgboost.FlamlXGBoostEstimator` for a
        FLAML-searched cell instead of a fixed-hyperparameter one -- ``sample-cli train-ml``
        does this by default now; the fixed-hyperparameter path remains reachable (its own
        ``--fixed-hyperparameters`` flag, or calling this function directly) as a fast, single-
        model-family baseline.
    :param calibrate_threshold: Forwarded to :func:`train_task` for every cell; ``True``
        (F1-calibrated) when omitted. ``False`` pins the conventional 0.5 for every model in the
        matrix instead of deriving one from any split of the training data.
    :param n_calibration_folds: Forwarded to :func:`train_task` for every cell;
        :data:`N_SPLITS` (5-fold OOF) when omitted. Reduce this alongside `model_factory` for an
        expensive architecture the same way :data:`sample.flaml_xgboost.FLAML_N_CALIBRATION_FOLDS`
        does, or the full matrix multiplies an expensive search by 5 calibration folds on top of
        the real fit. Meaningless when `calibrate_threshold` is ``False``.
    :param n_y_randomizations: Forwarded to :func:`train_task` for every cell;
        :data:`N_Y_RANDOMIZATIONS` when omitted.
    :param n_cv_folds: Forwarded to :func:`train_task` for every cell; :data:`N_SPLITS` when
        omitted.
    :returns: The report dict, with ``models`` (kept) and ``rejected`` lists.
    """
    import pickle

    # Resolved here (not defaulted in the signature -- see build_tasks' own comment on why) and
    # reused below for the report's "settings" block, so a caller who omitted both sees the actual
    # thresholds that were applied, not the sentinel None they passed.
    if min_task_examples is None:
        min_task_examples = MIN_POSITIVES_PER_TASK
    if min_target_examples is None:
        min_target_examples = min_task_examples * 2

    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Assembled dataset not found at {dataset_csv}")
    df = pd.read_csv(dataset_csv, low_memory=False)
    if "is_decoy" not in df.columns or "sequence" not in df.columns:
        raise ValueError("Dataset needs both 'sequence' and 'is_decoy' columns.")
    df = df.dropna(subset=["is_decoy", "sequence"])
    if filter_anomalies and "is_anomaly" in df.columns:
        before = len(df)
        df = df[df["is_anomaly"] == 0]
        logger.info(f"Filtered {before - len(df)} anomalies before training.")

    df = add_derived_feature_columns(df)
    blocks = feature_blocks(df)
    if not blocks:
        raise ValueError("No recognized feature columns; run the enrichment step first.")
    tasks = build_tasks(df, activities_csv, min_task_examples=min_task_examples,
                        min_target_examples=min_target_examples)
    if tasks_limit:
        tasks = tasks[:tasks_limit]

    # Computed ONCE for the whole dataset: which family/cluster a real peptide belongs to does not
    # depend on which task is being trained, and re-deriving it per (task, block) cell would run
    # PepMatch dozens of times over the same peptides for no different answer.
    groups = _family_groups(df, families_json=families_json, work_dir=cluster_work_dir)

    # Also computed once and reused per cell below, to write each kept model's holdout as FASTA
    # alongside its dataset.json -- see the write site further down for why. Keyed the same way
    # train_task derives entry_ids from `sub` (a row-filtered view of this same `df`, so its index
    # is a subset of df's): by the entry_id column when present, by the row index otherwise.
    id_key = df["entry_id"].astype(str) if "entry_id" in df.columns else df.index.astype(str)
    seq_by_id = dict(zip(id_key, df["sequence"].astype(str)))

    logger.info(f"Training {len(tasks)} task(s) x {len(blocks)} feature block(s): "
                f"{', '.join(blocks)}")
    os.makedirs(model_dir, exist_ok=True)

    # Every (task, feature block) cell mapped out up front, so the progress bar below shows real
    # progress through a known-size plan rather than an open-ended nested loop.
    cells = [(spec, block_name, columns) for spec in tasks for block_name, columns in blocks.items()]

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    with tqdm(cells, desc="Training task x feature-block matrix", unit="cell") as pbar:
        for spec, block_name, columns in pbar:
            pbar.set_description(f"{spec.name}/{block_name}")
            try:
                record = train_task(df, spec.y, columns, spec.name, block_name, groups,
                                    learning_task=spec.task, model_factory=model_factory,
                                    calibrate_threshold=calibrate_threshold,
                                    n_calibration_folds=n_calibration_folds,
                                    n_y_randomizations=n_y_randomizations,
                                    n_cv_folds=n_cv_folds)
            except Exception as e:
                logger.error(f"{spec.name}/{block_name} failed: {e}")
                continue
            if record is None:
                continue
            model = record.pop("_model")
            cols = record.pop("_columns")
            scaler = record.pop("_scaler")
            ad = record.pop("_applicability_domain")
            train_entry_ids = record.pop("_train_entry_ids")
            test_entry_ids = record.pop("_test_entry_ids")
            record.update({
                "peptide_type": spec.peptide_type, "scope": spec.scope,
                "assay_type": spec.assay_type, "target": spec.target,
                "learning_task": spec.task, "namespace": spec.namespace,
            })
            # Which homology-reduction mechanism actually produced this model's split, recorded on
            # the model itself rather than left to a log line nobody reads six months later.
            # _cluster_groups falls back to PepMatch when cd-hit is missing, and PepMatch cannot
            # deliver GROUPING_IDENTITY above ~14 residues (its mismatch budget is capped at 6),
            # so the fallback silently UNDER-clusters and inflates every score computed off the
            # resulting split. A reader of these metrics needs to know which of the two ran.
            record["grouping_identity"] = GROUPING_IDENTITY
            record["grouping_mechanism"] = ("cdhit" if _cdhit_available_for_grouping()
                                            else "pepmatch_fallback_under_clusters")
            # Negative-class composition, stated per model rather than left implicit. "n_negatives"
            # alone used to mean "decoys" because that was the only kind there was; now that a
            # negative can also be a real measured-inactive peptide, a reader (and Section 3.8 of
            # the paper) needs the split to interpret the metric at all. Derived from the record's
            # own n_negatives so the two can never be inconsistent.
            if record.get("n_negatives") is not None:
                n_inactive = min(int(spec.n_measured_inactive), int(record["n_negatives"]))
                record["n_negatives_measured_inactive"] = n_inactive
                record["n_negatives_decoy"] = int(record["n_negatives"]) - n_inactive
            passed, reasons = assess_quality(record, gate)
            record["quality_passed"] = passed
            record["quality_failures"] = reasons

            metric = _primary_metric(spec.task == "regression")
            null_cmp = record.get("null_comparison", {}).get(metric, {})
            ci = record.get("holdout_ci95", {}).get(metric, [float("nan"), float("nan")])
            status = "keep" if passed else "drop"
            logger.info(
                f"  [{status}] {spec.namespace}/{block_name}  n={record['n_samples']:>6} "
                f"{metric}={record['holdout'].get(metric, float('nan')):.3f} "
                f"CI95=[{ci[0]:.3f}, {ci[1]:.3f}] "
                f"yrand={record['y_randomization'].get(f'mean_{metric}', float('nan')):.3f} "
                f"vs-null({null_cmp.get('primary_test', 'kruskal')} "
                f"p={null_cmp.get('primary_pvalue', float('nan')):.3g})"
                + ("" if passed else f"  <- {reasons[0]}"))

            if not passed:
                # Recorded but NOT written. A model that cannot be trusted should not be sitting in
                # the release directory where something will eventually load it.
                rejected.append(record)
                continue

            # <peptide_type>/<scope>[/<target>]/<assay_type>/<task>/<feature_block>.pkl
            out_dir = os.path.join(model_dir, *spec.namespace.split("/"))
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{block_name}.pkl")
            with open(path, "wb") as fh:
                pickle.dump({"model": model, "columns": cols, "namespace": spec.namespace,
                             "feature_block": block_name, "metrics": record["holdout"],
                             # The scaler and AD travel together: a query must be scaled the same
                             # way before its distance to the training set means anything -- see
                             # sample.ml_models.in_domain.
                             "scaler": scaler, "applicability_domain": ad}, fh)
            record["model_path"] = path

            # Full validation report AND the exact train/test peptide split, one JSON file beside
            # the pickle -- what the webserver's per-model "dataset" page reads. Kept separate from
            # the aggregate `report` below rather than folded into it: `report["models"]` is read on
            # every models-page load, and a model's train/test id lists (up to tens of thousands of
            # peptides for the largest activity-class tasks) would make that load slow for a page that
            # mostly wants to compare AUCs, not enumerate ids.
            model_id = f"{spec.namespace}/{block_name}"
            dataset_path = os.path.join(out_dir, f"{block_name}.dataset.json")
            dataset_record = dict(record)
            dataset_record.update({
                "model_id": model_id,
                "n_train": len(train_entry_ids),
                "n_test": len(test_entry_ids),
                "train_entry_ids": train_entry_ids,
                "test_entry_ids": test_entry_ids,
                # The fitted NearestNeighbors index is not JSON-serializable and is not needed to
                # read the domain -- only its summary statistics are.
                "applicability_domain": {k: v for k, v in ad.items() if k != "index"},
            })
            with open(dataset_path, "w", encoding="utf-8") as fh:
                json.dump(dataset_record, fh, indent=2)
            record["dataset_path"] = dataset_path

            # Both sides of the split as FASTA, ready to hand to a tool this project has no wrapper
            # for (a web upload form, an unpackaged CLI): the SAME peptides step 47's benchmark
            # scores this model and its reference tool against, so an outside comparison stays
            # apples-to-apples rather than running against a fresh, differently-composed sample.
            # Written once here, at train time, rather than regenerated per download --
            # sample.benchmark.holdout_fasta falls back to computing the test side on demand for a
            # release built before this existed. `out_dir` is already the organized, namespaced
            # directory every other artifact for this (task, block) cell lands in
            # (<peptide_type>/<scope>[/<target>]/<assay_type>/<task>/), so both files land beside
            # the model and its dataset.json rather than a separate parallel tree.
            from sample.families import write_fasta

            fasta_path = write_fasta(
                {entry_id: seq_by_id[entry_id] for entry_id in test_entry_ids if entry_id in seq_by_id},
                os.path.join(out_dir, f"{block_name}.holdout.fasta"))
            record["holdout_fasta_path"] = fasta_path

            train_fasta_path = write_fasta(
                {entry_id: seq_by_id[entry_id] for entry_id in train_entry_ids if entry_id in seq_by_id},
                os.path.join(out_dir, f"{block_name}.train.fasta"))
            record["train_fasta_path"] = train_fasta_path

            kept.append(record)

    report = {
        "dataset": dataset_csv,
        "n_rows": int(len(df)),
        "feature_blocks": {k: len(v) for k, v in blocks.items()},
        "n_tasks": len(tasks),
        "n_models_kept": len(kept),
        "n_models_rejected": len(rejected),
        "settings": {
            "base_model": "XGBClassifier (falls back to RandomForestClassifier)",
            "n_splits": N_SPLITS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_y_randomizations": N_Y_RANDOMIZATIONS,
            "min_task_examples": min_task_examples,
            "min_target_examples": min_target_examples,
            "grouping": f"curated Pfam family (or step 35's own near-identity cluster) from "
                       f"data/families/families.json where available, else near-identity "
                       f"clustering at >={GROUPING_IDENTITY:.0%} sequence identity "
                       f"(sample.families.cluster, PepMatch) over real peptides not covered by "
                       f"that file; decoys as singleton groups; grouping performed once, before "
                       f"any split",
            "split": "single grouped train/holdout split (first fold of a "
                    f"StratifiedGroupKFold/GroupKFold with n_splits={N_SPLITS}, ~"
                    f"{100 / N_SPLITS:.0f}% held out) grouped by cluster; fit: fixed "
                    "hyperparameters (no grid search, no cross-validation) on the train portion "
                    "only, holdout never seen until final scoring",
            "hyperparameters": "fixed -- sample.ml_models._make_model's own base configuration, "
                              "unmodified; no hyperparameter search or cross-validation runs here",
            "y_randomization": "shuffled TRAINING labels, refit, scored on the SAME holdout rows "
                              "as the real model, so its null distribution is population-matched "
                              "for the significance test below",
            "null_significance_test": "real holdout bootstrap distribution vs. y-randomization "
                                      "per-repeat distribution; a Shapiro-Wilk normality test "
                                      "(alpha=" + repr(NORMALITY_ALPHA) + ") is run on each "
                                      "distribution first, and Welch's t-test is used as the "
                                      "gating p-value only when both are not shown non-normal, "
                                      "otherwise Kruskal-Wallis is; both tests and both normality "
                                      "results are kept in null_comparison for transparency",
            "applicability_domain": f"standardized k={AD_K_NEIGHBORS} nearest-neighbour distance "
                                    f"in the training/selection feature space (Roy et al. 2015); "
                                    f"threshold = training mean + 3 SD of training points' own "
                                    f"k-NN distance",
            "quality_gate": gate._asdict(),
        },
        "models": kept,
        "rejected": rejected,
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info(
        f"Kept {len(kept)} model(s) meeting the quality gate; rejected {len(rejected)}. "
        f"Report written to {report_path}")
    return report
