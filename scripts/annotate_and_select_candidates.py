"""
Annotates the E. coli docking candidates with every available third-party predictor, then selects
the top N for docking.

Each predictor lives in its own conda environment and is reached by ``conda run``, because these
tools pin mutually incompatible scikit-learn versions -- the same reason the rest of this project
isolates them (see ``environments/README.md``).

    python scripts/annotate_and_select_candidates.py --top 100

WHAT EACH TOOL CONTRIBUTES, and how far it should be trusted here:

* **EIPpred** (``pip install eippred``, env ``sample-eippred``) -- predicted MIC against E. coli,
  reported directly in uM. Recorded as an ANNOTATION, never a filter, for two measured reasons.

  (1) On THIS candidate set it carries no usable signal. Measured against all 138 candidates it
  could score (its input floor is 8 residues, so one 3-mer is unscored): Spearman rank
  rho = -0.064 (p = 0.46) and Pearson r on log10 = -0.131 -- i.e. no monotonic relationship with
  the measured MIC at all, not merely a biased one. Nor is there a constant offset to correct for:
  the predicted/measured ratio has median 0.9x but spans 0.3x to 10.5x. Read this as a statement
  about this set rather than about the tool: every candidate here is sub-uM BY CONSTRUCTION
  (measured 0.050-1.0 uM), and that compressed range attenuates any correlation, while EIPpred's
  own predictions for them span only 0.30-1.38 uM. The tool reports R = 0.78 across its full
  training range; it simply has no resolution inside the narrow window this selection lives in.

  (2) Circularity: EIPpred was trained on DBAASP v3, and 136 of these 139 candidates are
  DBAASP-sourced, so its "prediction" for them is substantially recall of its own training set.
* **antifungal** (``pip install antifungal``, env ``sample-antifungal``) -- per-species antifungal
  MIC (C. albicans / krusei / parapsilosis, C. neoformans) plus an antifungal index. This is the
  closest available tool to an MFC predictor, and it is NOT one: it predicts MIC, not minimum
  FUNGICIDAL concentration, and it is off-target for an E. coli transporter. Annotation only.
  No CLI, so it is driven through a small inline script.
* **HemoPI2** -- hemolysis-specific, and the predictor this project already uses live for that
  endpoint. Used as the FILTER.
* **Macrel / AlgPred2 / ToxinPred2** -- recorded, not filtered. See the module-level note below.

WHY ONLY HemoPI2 FILTERS. Applied together these tools leave nothing: of 139 candidates, 118 pass
HemoPI2, 16 pass Macrel, 14 pass ToxinPred2, 27 pass AlgPred2 -- and 0 pass all four. That is not
a safe set of candidates being found; it is four tools disagreeing. The disagreement has a visible
cause: the top candidates are proline-rich AMPs (Bac7/oncocin/drosocin), documented in the
literature as non-lytic and low-toxicity, yet Macrel scores them 0.51-0.82 hemolytic and AlgPred2
calls every one an allergen. These tools are trained largely on lytic, membrane-active cationic
peptides, and a highly cationic proline-rich peptide resembles one compositionally while acting by
an entirely different (intracellular-target) mechanism. Filtering on them would discard the best
candidates for a property they are not believed to have. ToxinPred2 additionally scores GENERAL
toxicity rather than hemolysis, and this project already wires it in only as a benchmark
comparator, never as a live annotator (see ToxinPred2Predictor's own docstring).

All four columns are written out regardless, so a stricter filter is one pandas expression away.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import sample  # noqa: F401  -- must precede pandas; see sample/__init__.py
import pandas as pd

FIBRO_SCRIPT = "scripts/select_docking_peptides.py"


def run_eippred(sequences, ids, env="sample-eippred"):
    """
    :returns: ``{entry_id: predicted_MIC_uM}``; empty when the tool is unavailable.

    EIPpred chdir's into a working directory it creates and writes its output THERE, not at the
    path given to ``-o`` relative to the caller's cwd -- so the output is located by search rather
    than assumed. Its documented input floor is 8 residues; shorter sequences are not submitted.
    """
    usable = [(i, s) for i, s in zip(ids, sequences) if len(s) >= 8]
    if not usable:
        return {}
    work = tempfile.mkdtemp(prefix="eippred_run_")
    fasta = os.path.join(work, "in.fa")
    with open(fasta, "w", encoding="utf-8") as fh:
        for entry_id, seq in usable:
            fh.write(f">{entry_id}\n{seq}\n")
    try:
        subprocess.run(["conda", "run", "-n", env, "eippred", "-i", "in.fa",
                        "-o", "out.csv", "-j", "1"],
                       cwd=work, capture_output=True, text=True, timeout=3600, check=False)
    except Exception as exc:
        print(f"  EIPpred failed: {exc}", flush=True)
        return {}
    hits = {}
    for root, _dirs, files in os.walk(work):
        if "out.csv" in files:
            with open(os.path.join(root, "out.csv"), encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    key = (row.get("ID") or "").lstrip(">").strip()
                    try:
                        hits[key] = float(row["MIC(uM)"])
                    except (KeyError, TypeError, ValueError):
                        continue
    print(f"  EIPpred: {len(hits)} of {len(usable)} submitted sequences scored", flush=True)
    return hits


def run_antifungal(sequences, env="sample-antifungal"):
    """
    :returns: ``{sequence: {field: value}}`` from ``antifungal.predict.predict_MIC``.

    The package exposes no console script, so it is driven through an inline program whose only
    output on stdout is one JSON document -- parsed rather than scraped.

    Input travels via a temp FILE named in argv, not stdin: ``conda run`` does not forward the
    parent's stdin to the child, so ``json.load(sys.stdin)`` here read empty input and died with
    "Expecting value: line 1 column 1".
    """
    if not sequences:
        return {}
    program = (
        "import json,sys\n"
        "from antifungal.predict import predict_MIC\n"
        "seqs=json.load(open(sys.argv[1]))\n"
        "r=predict_MIC(seqs)\n"
        "print('@@JSON@@'+json.dumps(r))\n"
    )
    fd, seq_path = tempfile.mkstemp(suffix=".json", prefix="antifungal_in_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(list(sequences), fh)
    try:
        proc = subprocess.run(["conda", "run", "-n", env, "python", "-c", program, seq_path],
                              capture_output=True, text=True, timeout=3600)
        marker = [l for l in (proc.stdout or "").splitlines() if l.startswith("@@JSON@@")]
        if not marker:
            print(f"  antifungal produced no JSON; stderr tail: "
                  f"{(proc.stderr or '')[-300:]}", flush=True)
            return {}
        payload = json.loads(marker[0][len("@@JSON@@"):])
    except Exception as exc:
        print(f"  antifungal failed: {exc}", flush=True)
        return {}
    seqs = payload.get("peptide_seq", [])
    fields = [k for k in payload if k != "peptide_seq"]
    out = {}
    for idx, seq in enumerate(seqs):
        out[seq] = {f"af_{k}": payload[k][idx] for k in fields
                    if isinstance(payload[k], list) and idx < len(payload[k])}
    print(f"  antifungal: {len(out)} sequences scored", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="data/docking/ecoli_mic_peptides.csv")
    ap.add_argument("--dataset", default="data/sample_final_release/final_dataset.csv")
    ap.add_argument("--activities", default="data/sample_raw_release/activities.csv")
    ap.add_argument("--toxinpred2", default="data/docking/candidates_toxinpred2.csv",
                    help="Precomputed ToxinPred2 output; skipped when absent.")
    ap.add_argument("--out-dir", default="data/docking")
    ap.add_argument("--max-um", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--skip-eippred", action="store_true")
    ap.add_argument("--skip-antifungal", action="store_true")
    args = ap.parse_args()

    d = pd.read_csv(args.candidates)
    cand = d[(d.mic_um_median <= args.max_um) & d.adcp_dockable].copy()
    print(f"{len(cand)} candidates (<= {args.max_um} uM MIC, ADCP-dockable)", flush=True)

    # --- predictors already materialised in the release ---
    cols = ["entry_id", "hemopi2_is_hemolytic", "hemopi2_score",
            "macrel_hemolytic_probability", "algpred2_is_allergen", "algpred2_score"]
    cand = cand.merge(pd.read_csv(args.dataset, usecols=cols, low_memory=False),
                      on="entry_id", how="left")

    if os.path.exists(args.toxinpred2):
        tp = pd.read_csv(args.toxinpred2, usecols=["entry_id", "toxinpred2_score",
                                                   "toxinpred2_is_toxic"])
        cand = cand.merge(tp, on="entry_id", how="left")
    else:
        print(f"  {args.toxinpred2} absent -- ToxinPred2 columns omitted", flush=True)

    # --- measured anti-fibroblast cytotoxicity (evidence, not prediction) ---
    sys.path.insert(0, os.path.dirname(os.path.abspath(FIBRO_SCRIPT)))
    from select_docking_peptides import antifibroblast_sequences
    acts = pd.read_csv(args.activities, low_memory=False)
    fibro = antifibroblast_sequences(acts)
    cand["has_antifibroblast_activity"] = cand["sequence"].isin(fibro)
    print(f"  measured anti-fibroblast cytotoxicity: "
          f"{int(cand['has_antifibroblast_activity'].sum())} candidate(s)", flush=True)

    # --- third-party tools run here ---
    if not args.skip_eippred:
        pred = run_eippred(cand["sequence"].tolist(), cand["entry_id"].tolist())
        cand["eippred_mic_um"] = cand["entry_id"].map(pred)
    if not args.skip_antifungal:
        af = run_antifungal(cand["sequence"].tolist())
        if af:
            af_frame = pd.DataFrame.from_dict(af, orient="index")
            af_frame.index.name = "sequence"
            cand = cand.merge(af_frame.reset_index(), on="sequence", how="left")

    # A dockability guard, not an activity filter: a sequence of one or two distinct residues is
    # not a credible sub-micromolar antibacterial, and one reached the selection. "AAA"
    # (SAMPLE0000038085) carried a 0.52 uM E. coli MIC from DBAASP record DBAASPS_23450, whose
    # peptide_name is "AAAA(P)" -- a MODIFIED compound. Its 0.12 ug/mL converts against a 3-mer's
    # tiny molecular weight into an apparently potent 0.52 uM, which is how trialanine came to
    # rank among PrAMPs, on one unreplicated record.
    #
    # This is not a parsing bug. DBAASP legitimately catalogues short cyclic and modified
    # compounds, and its one-letter `sequence` field is a LOSSY PROJECTION of them: Cyclo(Gly-Pro)
    # appears as "GP", Apicidin as "XXIX", HC-toxin as "XPAA" (X = non-standard residue). 9,024
    # DBAASP activity records carry sequences under 5 residues for this reason. The projection is
    # faithful to what DBAASP stores; it just is not a linear peptide ADCP can dock.
    #
    # Expressed as sequence complexity rather than a length floor deliberately: the length bounds
    # are a docking constraint set elsewhere (2-20 residues), and raising them to hide this would
    # silently discard genuinely short peptides too.
    cand["is_low_complexity"] = cand["sequence"].apply(lambda s: len(set(str(s).upper())) <= 2)
    n_low = int(cand["is_low_complexity"].sum())
    if n_low:
        print(f"  low-complexity (<=2 distinct residues, excluded as suspect): {n_low}", flush=True)

    # --- the filter: HemoPI2 plus measured fibroblast harm. See the module docstring. ---
    keep = cand[(cand.hemopi2_is_hemolytic == 0)
                & (~cand.has_antifibroblast_activity)
                & (~cand.is_low_complexity)].copy()
    print(f"\nafter filtering (HemoPI2 non-hemolytic, no measured anti-fibroblast harm, "
          f"not low-complexity): {len(keep)}", flush=True)
    chosen = keep.sort_values("mic_um_median").head(args.top)

    os.makedirs(args.out_dir, exist_ok=True)
    annotated = os.path.join(args.out_dir, "candidates_annotated.csv")
    cand.sort_values("mic_um_median").to_csv(annotated, index=False)

    selected_csv = os.path.join(args.out_dir, "docking_selection.csv")
    chosen.to_csv(selected_csv, index=False)
    fasta = os.path.join(args.out_dir, "pilot_peptides.fasta")
    with open(fasta, "w", encoding="utf-8") as fh:
        for row in chosen.itertuples():
            fh.write(f">{row.entry_id}|MIC_uM={row.mic_um_median:.4g}|"
                     f"len={len(row.sequence)}\n{row.sequence}\n")

    print(f"\nwrote {annotated} ({len(cand)} rows, all tool columns)")
    print(f"wrote {selected_csv} and {fasta} ({len(chosen)} selected for docking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
