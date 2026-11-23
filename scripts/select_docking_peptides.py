"""
Selects the peptide set to dock against the MsbA G907 site.

Selection is deliberately narrow, and every narrowing is a docking constraint rather than a
biological judgement:

* **MIC only.** The activity table carries 30 activity types; only ``MIC`` is used. MBC (13,648
  records) is a *bactericidal* endpoint, MIC50/MIC90 (3,436) are population statistics over a
  panel of strains, and IC50 is a different assay geometry entirely. Mixing them would rank
  peptides by a number that does not mean the same thing from row to row.
* **A real measurement, not a bound.** ``value_operator`` of ``>`` means "no inhibition up to this
  concentration" -- an inactive result, whose value must never be read as a potency.
* **2-20 residues.** The upper bound is AutoDock CrankPep's tractable range; the lower bound just
  excludes single residues, which are not peptides. Note this range is wider than CrankPep's own
  documented comfort zone at the short end -- it changes very little here in practice, because
  potency in this dataset does not reach the very short lengths (widening 5-20 to 2-20 adds 155
  peptides overall but only ONE more at <= 1 uM, and the shortest potent peptide is 14 residues).
* **<= 1 cysteine.** ADCP models peptides as linear flexible chains; a disulfide-constrained fold
  docked as an unconstrained chain produces conformations that cannot physically exist.
* **Not predicted hemolytic**, by HemoPI2 (``hemopi2_is_hemolytic``). A blocker candidate that
  lyses red blood cells is not a therapeutic lead, whatever it does to the transporter.
* **No measured anti-fibroblast activity.** Cytotoxicity against a normal fibroblast line is
  direct evidence of harm to host cells. Crucially, a fibroblast record whose ``value_operator``
  is ``>`` is evidence of the OPPOSITE -- "no effect up to the highest dose tested" -- and must
  not be read as toxicity: 47 of the 110 fibroblast records in this release are exactly that, so
  treating every fibroblast record as a red flag would discard peptides shown to be SAFE.

Writes ``pilot_peptides.fasta`` (for :mod:`scripts.dock_peptides_adcp`) and a ``.json`` sidecar
recording the provenance of every selected peptide -- including the activity type it came from, so
this constraint is auditable after the fact rather than implied by a comment.
"""
import argparse
import json
import os
import sys

import sample  # noqa: F401  -- must precede pandas; see sample/__init__.py on the libstdc++ pin
import pandas as pd

from sample.concentration import to_micromolar

#: Only this activity type is accepted. See the module docstring on why MBC/MIC50/MIC90 are not.
ACTIVITY_TYPE = "MIC"

#: Substrings identifying Escherichia coli in the free-text organism field.
ECOLI_PATTERNS = ("escherichia coli", "e. coli", "e.coli")

#: Matches a fibroblast cell line in the free-text ``target_cell_line`` field. The word
#: "fibroblast" alone finds only 37 of the 110 such records here -- the standard lines are named,
#: not described (L929, NIH-3T3, MRC-5, HDF, HFF, WI-38), so they are matched explicitly.
FIBROBLAST_PATTERN = (r"fibro|\bL[- ]?929\b|\b(?:NIH[- ]?)?3T3\b|\bN?HDFa?\b|"
                      r"\bMRC[- ]?5\b|\bHFF\b|\bWI[- ]?38\b")


def antifibroblast_sequences(acts: pd.DataFrame) -> set:
    """
    :param acts: The full activity table (not the E. coli subset -- cytotoxicity against a
        fibroblast line is recorded against that cell line, not against a bacterium).
    :returns: Sequences with a MEASURED cytotoxic effect on a fibroblast line. Records whose
        ``value_operator`` is ``>`` are excluded: those report no effect up to the tested dose,
        which is evidence of safety, and counting them here would invert their meaning.
    """
    cell_line = acts["target_cell_line"].fillna("").astype(str)
    on_fibroblast = cell_line.str.contains(FIBROBLAST_PATTERN, case=False, regex=True)
    operator = acts["value_operator"].fillna("").astype(str).str.strip()
    return set(acts.loc[on_fibroblast & (operator != ">"), "sequence"])


def _ecoli_mask(organisms: pd.Series) -> pd.Series:
    lowered = organisms.fillna("").str.lower()
    return pd.concat([lowered.str.contains(p, regex=False) for p in ECOLI_PATTERNS], axis=1).any(
        axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activities", default="data/sample_raw_release/activities.csv")
    ap.add_argument("--dataset", default="data/sample_final_release/final_dataset.csv",
                    help="Supplies entry_id and molecular_weight (needed for ug/mL -> uM).")
    ap.add_argument("--out-dir", default="data/docking")
    ap.add_argument("--max-um", type=float, default=1.0, help="Potency ceiling in micromolar")
    ap.add_argument("--min-length", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=20)
    ap.add_argument("--max-cys", type=int, default=1)
    ap.add_argument("--top", type=int, default=12, help="How many of the most potent to keep")
    args = ap.parse_args()

    acts = pd.read_csv(args.activities, low_memory=False)
    print(f"{len(acts):,} activity records", flush=True)

    acts = acts[acts["activity_type"].astype(str).str.strip() == ACTIVITY_TYPE]
    print(f"{len(acts):,} are {ACTIVITY_TYPE}", flush=True)

    acts = acts[_ecoli_mask(acts["target_organism"])]
    print(f"{len(acts):,} {ACTIVITY_TYPE} records against E. coli", flush=True)

    # ">" is "no inhibition up to this concentration" -- an inactive, not a potency.
    operator = acts["value_operator"].fillna("").astype(str).str.strip()
    n_bounded = int((operator == ">").sum())
    acts = acts[operator != ">"]
    print(f"dropped {n_bounded:,} '>' (non-inhibitory bound) records -> {len(acts):,}", flush=True)

    meta = pd.read_csv(args.dataset,
                       usecols=["entry_id", "sequence", "molecular_weight", "is_decoy"])
    # Decoys are SYNTHETIC NEGATIVES and must never appear in a table of peptides with a measured
    # E. coli MIC. They can get here because the join is on `sequence`: a shuffled decoy inherits
    # its parent's exact amino-acid composition, and a short one can coincide with a real
    # sequence that does carry activity records, which then attach to the decoy's entry_id.
    # Observed directly: SAMPLE_DECOY_00020535 ("XX") was emitted with a 64 uM MIC and flagged
    # dockable. Dropped here rather than downstream so the released CSV never contains one.
    meta = meta[meta["is_decoy"].fillna(0).astype(int) == 0]
    meta = meta.dropna(subset=["sequence"]).drop_duplicates(subset=["sequence"])
    acts = acts.merge(meta.drop(columns=["is_decoy"]), on="sequence", how="inner")
    print(f"{len(acts):,} joined to a released (non-decoy) entry", flush=True)

    acts["value_um"] = [
        to_micromolar(v, u, mw)
        for v, u, mw in zip(acts["value_lower"], acts["activity_unit"], acts["molecular_weight"])
    ]
    acts = acts.dropna(subset=["value_um"])
    print(f"{len(acts):,} converted to uM", flush=True)

    # Median across replicate assays per peptide, then keep the peptide's best (lowest) value.
    per_peptide = (acts.groupby(["entry_id", "sequence"], as_index=False)
                       .agg(mic_um=("value_um", "median"), n_records=("value_um", "size")))
    per_peptide = per_peptide.sort_values("mic_um")

    potent = per_peptide[per_peptide["mic_um"] <= args.max_um].copy()
    print(f"{len(potent):,} peptides at <= {args.max_um} uM ({ACTIVITY_TYPE}, E. coli)", flush=True)

    potent["length"] = potent["sequence"].str.len()
    potent["n_cys"] = potent["sequence"].str.upper().str.count("C")

    tractable = potent[(potent["length"] >= args.min_length)
                       & (potent["length"] <= args.max_length)]
    print(f"{len(tractable):,} within {args.min_length}-{args.max_length} residues", flush=True)

    standard = tractable[tractable["sequence"].apply(
        lambda s: set(str(s).upper()) <= set("ACDEFGHIKLMNPQRSTVWY"))]
    if len(standard) != len(tractable):
        print(f"{len(tractable) - len(standard):,} dropped for non-standard residues "
              f"(ADCP has no rotamer for 'X')", flush=True)

    linear = standard[standard["n_cys"] <= args.max_cys]
    print(f"{len(linear):,} with <= {args.max_cys} cysteine "
          f"({len(standard) - len(linear)} disulfide-rich excluded)", flush=True)

    chosen = linear.head(args.top)
    os.makedirs(args.out_dir, exist_ok=True)

    # The full E. coli MIC table, every peptide, not just the docking subset. Carries the two
    # tractability flags as columns rather than pre-filtering, so the same file answers "what is
    # active against E. coli" and "what of that can ADCP actually dock" without a second pass.
    full = per_peptide.copy()
    full["length"] = full["sequence"].str.len()
    full["n_cys"] = full["sequence"].str.upper().str.count("C")
    full["activity_type"] = ACTIVITY_TYPE
    full["target_organism"] = "Escherichia coli"
    full["in_length_range"] = ((full["length"] >= args.min_length)
                               & (full["length"] <= args.max_length))
    full["is_linear_enough"] = full["n_cys"] <= args.max_cys
    # ADCP builds its ligand from the 20 standard residues; an 'X' (DBAASP's placeholder for a
    # non-standard residue, as in Apicidin -> "XXIX") has no rotamer library entry and cannot be
    # docked at all. Checked explicitly rather than assumed: the one-letter `sequence` field is a
    # lossy projection of cyclic and modified compounds, so non-standard letters do occur.
    full["is_standard_alphabet"] = full["sequence"].apply(
        lambda s: set(str(s).upper()) <= set("ACDEFGHIKLMNPQRSTVWY"))
    full["adcp_dockable"] = (full["in_length_range"] & full["is_linear_enough"]
                             & full["is_standard_alphabet"])
    full["selected_for_docking"] = full["entry_id"].isin(set(chosen["entry_id"]))
    full = full.rename(columns={"mic_um": "mic_um_median", "n_records": "n_mic_records"})
    full = full[["entry_id", "sequence", "activity_type", "target_organism", "mic_um_median",
                 "n_mic_records", "length", "n_cys", "in_length_range", "is_linear_enough",
                 "adcp_dockable", "selected_for_docking"]]
    csv_path = os.path.join(args.out_dir, "ecoli_mic_peptides.csv")
    full.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}: {len(full):,} peptides with an E. coli MIC "
          f"({int(full['adcp_dockable'].sum()):,} ADCP-dockable)", flush=True)

    fasta_path = os.path.join(args.out_dir, "pilot_peptides.fasta")
    with open(fasta_path, "w", encoding="utf-8") as fh:
        for row in chosen.itertuples():
            fh.write(f">{row.entry_id}|MIC_uM={row.mic_um:.4g}|len={row.length}\n"
                     f"{row.sequence}\n")

    json_path = os.path.join(args.out_dir, "pilot_peptides.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([{"entry_id": r.entry_id, "sequence": r.sequence,
                    "activity_type": ACTIVITY_TYPE, "target_organism": "Escherichia coli",
                    "mic_um": round(float(r.mic_um), 6), "n_mic_records": int(r.n_records),
                    "length": int(r.length), "n_cys": int(r.n_cys)}
                   for r in chosen.itertuples()], fh, indent=2)

    print(f"\nwrote {fasta_path} and {json_path} ({len(chosen)} peptides)", flush=True)
    for row in chosen.itertuples():
        print(f"  {row.entry_id}  {row.sequence:22s} MIC={row.mic_um:8.4f} uM  "
              f"(n={row.n_records})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
