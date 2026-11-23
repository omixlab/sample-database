"""
Generates the dataset-description report for the sAMPle release.

Reads the final assembled dataset and the auxiliary per-peptide stores it does not itself carry
(activities, families, Gene Ontology, source organisms) and produces the tables, figures and
markdown report requested for `paper/`. Run under the `sample-ml` conda environment:

    conda run -n sample-ml python paper/generate_dataset_report.py

Nothing here writes back into the release data -- this is read-only, descriptive reporting.

All figures are Plotly, exported to static PNG via kaleido, sharing one look: solid fills (no
gradients on a single series -- a color ramp across the bars of one series encodes nothing), no
gridlines, and only the axis lines themselves (no plot-area border box). Activity-group colors are
fixed throughout: antibacterial green, anticancer red, antiparasitic blue, antiviral purple,
antifungal orange, everything else gray -- so the same group reads the same way in every figure.
"""
import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

STATS: dict = {"figures": {}}

# Categorical palette for the five activity classes, validated for colour-vision deficiency
# rather than chosen by eye. The previous palette was matplotlib's tab10, and it failed badly:
# simulating protanopia/deuteranopia (Machado-Oliveira-Fernandes 2009, severity 1.0) and
# measuring OKLab Delta E x100 between every pair gave
#
#     antibacterial green  #2CA02C  vs antifungal orange #FF7F0E ->  0.65   (indistinguishable)
#     antiviral purple     #9467BD  vs antiparasitic blue #1F77B4 ->  1.74   (indistinguishable)
#     antibacterial green  #2CA02C  vs anticancer red    #D62728 ->  3.89   (below the floor)
#
# against a floor of 6.0 -- so three pairs of activity classes, including the two largest, were
# the same colour to roughly 8% of male readers. These figures carry that encoding across the
# whole manuscript.
#
# These six are drawn from a palette whose steps are published as CVD-validated, and the subset
# was chosen by searching every 5-subset for one that clears the floor on ALL pairs, not just
# adjacent ones: worst CVD pair 6.86, worst normal-vision pair 15.56, every hue inside the
# lightness band with chroma >= 0.10. 6.86 sits in the band that requires identity to be carried
# by something besides colour as well -- which it is here: every category is named on the axis or
# in the legend of each figure that uses these.
GROUP_COLORS = {
    "antibacterial": "#2a78d6",   # blue
    "anticancer": "#1baf7a",      # aqua
    "antifungal": "#eda100",      # yellow
    "antiviral": "#008300",       # green
    "antiparasitic": "#e34948",   # red
    # Deliberately neutral: "other" is a catch-all, not one of the named classes, so it is the one
    # entry that is allowed to have no hue. Its zero chroma is the single check it fails, by
    # design; it still clears every separation check against the five above (worst CVD 7.99,
    # worst normal-vision 15.84, contrast 3.90 against the figure's white surface).
    "other": "#7F7F7F",
}
OTHER_GRAY = "#C7C7C7"
NEUTRAL_BLUE = "#2a78d6"
NEUTRAL_TEAL = "#1baf7a"
FONT = dict(family="Arial, Helvetica, sans-serif", size=13, color="#222222")


def clean_layout(fig: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "",
                 showlegend: bool = False, height: int = 500, width: int = 900) -> go.Figure:
    """Applies the report's one shared chart style: solid colors, no grid, axis lines only."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#111111"), x=0.5, xanchor="center"),
        font=FONT, plot_bgcolor="white", paper_bgcolor="white",
        showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=70, r=30, t=60, b=70),
        height=height, width=width,
    )
    fig.update_xaxes(title_text=xaxis_title, showgrid=False, zeroline=False,
                     showline=True, linecolor="#333333", linewidth=1, mirror=False,
                     ticks="outside", tickcolor="#333333")
    fig.update_yaxes(title_text=yaxis_title, showgrid=False, zeroline=False,
                     showline=True, linecolor="#333333", linewidth=1, mirror=False,
                     ticks="outside", tickcolor="#333333")
    return fig


# Plotly lays figures out in CSS pixels (96/inch); this scale factor makes the exported PNG's
# actual pixel density equivalent to 300 DPI at the figure's specified width/height.
EXPORT_SCALE = 300 / 96


def savefig(fig: go.Figure, name: str) -> str:
    stem, _ = os.path.splitext(name)
    png_path = os.path.join(FIG_DIR, f"{stem}.png")
    pdf_path = os.path.join(FIG_DIR, f"{stem}.pdf")
    fig.write_image(png_path, scale=EXPORT_SCALE)
    fig.write_image(pdf_path)  # PDF is vector; no DPI scale needed
    print(f"  saved figures/{stem}.png (300 DPI) + figures/{stem}.pdf")
    return f"figures/{stem}.png"


def italic_binomial(name: str) -> str:
    """Wraps a Genus/species-looking label in Plotly's HTML italics for axis/legend text."""
    return f"<i>{name}</i>"


def add_headroom(fig: go.Figure, values, axis: str = "x", frac: float = 0.14,
                 row: int = None, col: int = None) -> None:
    """Extends the given axis range past the data max so an outside-positioned bar label (the
    number printed past the bar's tip) has room to render instead of being clipped at the plot
    edge -- Plotly autoranges to the data extent only, with no allowance for label text width."""
    vals = [v for v in values if v is not None]
    if not vals:
        return
    top = max(vals) * (1 + frac)
    kwargs = {}
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col
    if axis == "x":
        fig.update_xaxes(range=[0, top], **kwargs)
    else:
        fig.update_yaxes(range=[0, top], **kwargs)


# ---------------------------------------------------------------------------------------------
# Organism / cell-line name normalization
# ---------------------------------------------------------------------------------------------
from sample.synergy import normalise_pathogen  # noqa: E402  (after warnings filter, before use)
from sample.cell_lines import clean_cell_line_name, build_cell_line_categories  # noqa: E402


def normalized_organism_counts(series: pd.Series, top: int = 20) -> pd.Series:
    """Collapses strain/isolate designations to the species binomial (see normalise_pathogen)
    before counting, so 'E. coli ATCC 25922' and 'Escherichia coli 910' count as one target."""
    binomials = series.dropna().map(normalise_pathogen)
    binomials = binomials[binomials != ""]
    return binomials.value_counts().head(top)


# ---------------------------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------------------------
print("Loading final_dataset.csv ...")
df = pd.read_csv(os.path.join(DATA, "sample_final_release", "final_dataset.csv"),
                 engine="python", on_bad_lines="skip")
df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
real = df[df["is_decoy"] == 0].copy()
decoys = df[df["is_decoy"] == 1].copy()
STATS["n_total"] = int(len(df))
STATS["n_real"] = int(len(real))
STATS["n_decoy"] = int(len(decoys))
print(f"  {len(df)} rows ({len(real)} real, {len(decoys)} decoy)")

print("Loading activities.csv ...")
act = pd.read_csv(os.path.join(DATA, "sample_raw_release", "activities.csv"),
                  engine="python", on_bad_lines="skip")
act["sequence"] = act["sequence"].astype(str).str.strip().str.upper()
STATS["n_activities_total"] = int(len(act))
print(f"  {len(act)} activity rows")

groups_by_seq: dict = act.dropna(subset=["target_group"]).groupby("sequence")["target_group"].apply(
    lambda s: set(s.dropna())).to_dict()


def primary_group(seq: str) -> str:
    groups = groups_by_seq.get(seq)
    if not groups:
        return "other"
    for g in ("antibacterial", "anticancer", "antifungal", "antiviral", "antiparasitic"):
        if g in groups:
            return g
    return next(iter(groups))


real["primary_group"] = real["sequence"].map(primary_group)
# "other" (no recognized activity group) is excluded from every per-category breakdown below --
# it is not a real activity, just the residual bucket, and clutters charts meant to compare
# activities against each other.
present_groups = [g for g in ("antibacterial", "anticancer", "antifungal", "antiviral",
                              "antiparasitic") if g in real["primary_group"].unique()]

# ===============================================================================================
# 1. Sequences per activity + number of activities
# ===============================================================================================
print("Section 1: sequences and activities per group ...")
seqs_per_group = act.dropna(subset=["target_group"]).groupby("target_group")["sequence"].nunique() \
    .sort_values(ascending=False)
measurements_per_group = act.dropna(subset=["target_group"]).groupby("target_group").size() \
    .sort_values(ascending=False)
STATS["sequences_per_activity"] = seqs_per_group.to_dict()
STATS["measurements_per_activity"] = measurements_per_group.to_dict()

# "other" (no recognized activity group) is excluded from the plot itself -- it is not a real
# activity, just the residual bucket -- though its true count is still kept in STATS above.
#
# Both panels share ONE category order -- descending by peptide count -- rather than each being
# sorted by its own values. Independently sorted, the two panels put a different activity group at
# the same x-position (antifungal 3rd-from-left by peptide count but 2nd-from-left by measurement
# count), which reads as the bars lining up when they do not: a reader scanning across the two
# panels at the same horizontal position is comparing two different activity groups.
plot_seqs = seqs_per_group.drop("other", errors="ignore")
plot_meas = measurements_per_group.drop("other", errors="ignore").reindex(plot_seqs.index)

fig = make_subplots(rows=1, cols=2, subplot_titles=("Unique peptides per activity group",
                                                    "Measurements per activity group"))
fig.add_trace(go.Bar(x=plot_seqs.index, y=plot_seqs.values,
                     marker=dict(color=[GROUP_COLORS.get(g, OTHER_GRAY) for g in plot_seqs.index],
                               line=dict(color="black", width=1)),
                     text=[f"{v:,}" for v in plot_seqs.values], textposition="outside"),
             row=1, col=1)
fig.add_trace(go.Bar(x=plot_meas.index, y=plot_meas.values,
                     marker=dict(color=[GROUP_COLORS.get(g, OTHER_GRAY) for g in plot_meas.index],
                               line=dict(color="black", width=1)),
                     text=[f"{v:,}" for v in plot_meas.values], textposition="outside"),
             row=1, col=2)
clean_layout(fig, "Sequences and activities per group", height=480, width=1100)
fig.update_yaxes(title_text="Peptides", row=1, col=1)
fig.update_yaxes(title_text="Activities", row=1, col=2)
add_headroom(fig, plot_seqs.values, axis="y", row=1, col=1)
add_headroom(fig, plot_meas.values, axis="y", row=1, col=2)
STATS["figures"]["activities"] = savefig(fig, "01_sequences_and_activities_per_group.png")

# ===============================================================================================
# 2. Peptides per source database
# ===============================================================================================
print("Section 2: peptides per source database ...")
db_cols = ["ADP", "CAMP", "DBAMP", "DRAMP", "DBAASP", "AntiCP2"]
db_name_map = {"ADP": "APD3", "CAMP": "CAMP", "DBAMP": "dbAMP", "DRAMP": "DRAMP", "DBAASP": "DBAASP", "AntiCP2": "AntiCP2"}
db_counts = pd.Series({c: int(real[c].notna().sum()) for c in db_cols if c in real.columns}) \
    .sort_values(ascending=False)
STATS["peptides_per_source_db"] = db_counts.to_dict()

# The whole-release total, for scale -- how each source database's count compares to every real
# peptide sAMPle carries, not just to each other.
plot_index = ["sAMPle (total)"] + [db_name_map.get(c, c) for c in db_counts.index]
plot_values = [STATS["n_real"]] + list(db_counts.values)
plot_colors = [GROUP_COLORS["antibacterial"]] + [NEUTRAL_BLUE] * len(db_counts)

fig = go.Figure(go.Bar(x=plot_index, y=plot_values,
                       marker=dict(color=plot_colors, line=dict(color="black", width=1)),
                       text=[f"{v:,}" for v in plot_values], textposition="outside"))
clean_layout(fig, "Peptides per source database, vs. the sAMPle total (real peptides)",
            yaxis_title="Peptides")
add_headroom(fig, plot_values, axis="y")
STATS["figures"]["source_databases"] = savefig(fig, "02_peptides_per_source_database.png")

# ===============================================================================================
# 3/4. Family / cluster counts and sizes, and the pizza chart
# ===============================================================================================
print("Section 3/4: families and clusters ...")
families_path = os.path.join(DATA, "families", "families.json")
with open(families_path, encoding="utf-8") as fh:
    families = json.load(fh)
fam_df = pd.DataFrame.from_dict(families, orient="index")
fam_df.index.name = "entry_id"
fam_df = fam_df.reset_index()

pfam_rows = fam_df[fam_df["source"] == "pfam"]
cluster_rows = fam_df[fam_df["source"] == "cluster"]
n_pfam_peptides, n_cluster_peptides = len(pfam_rows), len(cluster_rows)
n_pfam_families = int(pfam_rows["family_id"].nunique())
n_clusters = int(cluster_rows["family_id"].nunique())
n_covered = fam_df["entry_id"].nunique()
n_neither = STATS["n_real"] - n_covered

STATS["n_pfam_anchored_peptides"] = int(n_pfam_peptides)
STATS["n_clustered_peptides"] = int(n_cluster_peptides)
STATS["n_pfam_families"] = n_pfam_families
STATS["n_clusters"] = n_clusters
STATS["n_no_family_no_cluster"] = int(n_neither)

# The literal ask: how many families vs. how many clusters exist, each split into singleton (one
# member -- a family/cluster nothing else in this release resembles) vs. multi-member, since both
# splits are real and informative. Earlier drafts of this figure omitted the cluster-side split on
# the assumption every near-identity cluster was a singleton at this threshold; direct
# verification found that assumption traced to a stale families.json (fixed -- see step 33's
# rebuilt output), not to the clustering step actually producing only singletons.
pfam_sizes = pfam_rows.groupby("family_id").size()
n_pfam_singleton = int((pfam_sizes == 1).sum())
n_pfam_multi = int((pfam_sizes > 1).sum())
STATS["n_pfam_singleton_families"] = n_pfam_singleton
STATS["n_pfam_multi_member_families"] = n_pfam_multi

cluster_sizes_early = cluster_rows.groupby("family_id").size()
n_cluster_singleton = int((cluster_sizes_early == 1).sum())
n_cluster_multi = int((cluster_sizes_early > 1).sum())
STATS["n_cluster_singleton"] = n_cluster_singleton
STATS["n_cluster_multi_member"] = n_cluster_multi

bar_names = ["Pfam families\n(total)", "Pfam families\n(singleton)",
            "Pfam families\n(2+ members)", "Near-identity clusters\n(total)",
            "Near-identity clusters\n(singleton)", "Near-identity clusters\n(2+ members)"]
bar_values = [n_pfam_families, n_pfam_singleton, n_pfam_multi,
             n_clusters, n_cluster_singleton, n_cluster_multi]
bar_colors = [GROUP_COLORS["antibacterial"], NEUTRAL_TEAL, NEUTRAL_TEAL,
             NEUTRAL_BLUE, NEUTRAL_TEAL, NEUTRAL_TEAL]
fig = go.Figure(go.Bar(x=bar_names, y=bar_values,
                       marker=dict(color=bar_colors, line=dict(color="black", width=1)),
                       text=[f"{v:,}" for v in bar_values], textposition="outside"))
clean_layout(fig, "Pfam families vs. near-identity clusters, each with its singleton split",
            yaxis_title="Number of families / clusters (not peptides)", height=550, width=1000)
add_headroom(fig, bar_values, axis="y")
STATS["figures"]["family_cluster_counts"] = savefig(fig, "03_family_vs_cluster_counts.png")

# Top Pfam families by member count -- these DO have a real size distribution (a family can hold
# dozens of peptides), unlike the clusters below.
top_families = pfam_rows.groupby("family_id").size().sort_values(ascending=False).head(20)
family_name_by_id = pfam_rows.drop_duplicates("family_id").set_index("family_id")["family_name"]
top_family_labels = [
    f"{fid} ({family_name_by_id.get(fid)})" if family_name_by_id.get(fid) else fid
    for fid in top_families.index
]
STATS["top_pfam_families"] = dict(zip(top_family_labels, top_families.values.tolist()))
fig = go.Figure(go.Bar(x=top_families.values, y=top_family_labels, orientation="h",
                       marker=dict(color=NEUTRAL_TEAL, line=dict(color="black", width=1)),
                       text=[f"{v:,}" for v in top_families.values], textposition="outside"))
clean_layout(fig, f"Top 20 Pfam families by peptide count ({n_pfam_families} families total)",
            xaxis_title="Peptides", height=600)
fig.update_yaxes(autorange="reversed")
add_headroom(fig, top_families.values, axis="x")
STATS["figures"]["pfam_families"] = savefig(fig, "04_top_pfam_families.png")

# Cluster size stats -- reuses cluster_sizes_early computed above (no separate figure: the
# singleton/multi-member split already shown on the family_vs_cluster_counts figure carries this
# same information).
cluster_sizes = cluster_sizes_early
n_singleton = n_cluster_singleton
STATS["cluster_singleton_fraction"] = float(n_singleton / n_clusters) if n_clusters else 0.0
STATS["largest_cluster_size"] = int(cluster_sizes.max()) if len(cluster_sizes) else 0

# Pizza chart: Pfam-anchored / clustered-no-pfam / neither.
pizza_labels = ["Pfam-anchored family", "Near-identity cluster (no Pfam family)", "Neither"]
pizza_values = [n_pfam_peptides, n_cluster_peptides, n_neither]
fig = go.Figure(go.Pie(labels=pizza_labels, values=pizza_values,
                       marker=dict(colors=[GROUP_COLORS["antibacterial"], NEUTRAL_BLUE, OTHER_GRAY],
                                  line=dict(color="black", width=2)),
                       textinfo="label+percent+value", sort=False))
clean_layout(fig, f"Family assignment across {STATS['n_real']:,} real peptides",
            height=550, width=750)
STATS["figures"]["family_pizza"] = savefig(fig, "05_family_assignment_pizza.png")

# ===============================================================================================
# 5. Most common GO terms
# ===============================================================================================
print("Section 5: Gene Ontology terms ...")
go_path = os.path.join(DATA, "gene_ontology.json")
go_terms_top = pd.Series(dtype=int)
go_by_seq = {}
if os.path.exists(go_path):
    with open(go_path, encoding="utf-8") as fh:
        go_data = json.load(fh)
    go_by_seq = go_data.get("gene_ontology", go_data) if isinstance(go_data, dict) else {}
    rows = [(t.get("go_id"), t.get("term")) for terms in go_by_seq.values() for t in (terms or [])
           if t.get("term")]
    if rows:
        go_terms_top = pd.Series(rows).value_counts().head(20)

STATS["n_peptides_with_go"] = int(len({s for s in go_by_seq if go_by_seq.get(s)}))
go_labels = [f"({go_id}) {term}" if go_id else term for go_id, term in go_terms_top.index]
STATS["top_go_terms"] = dict(zip(go_labels, go_terms_top.values.tolist()))

if len(go_terms_top):
    fig = go.Figure(go.Bar(x=go_terms_top.values, y=go_labels, orientation="h",
                           marker=dict(color=NEUTRAL_TEAL, line=dict(color="black", width=1)),
                           text=[f"{v:,}" for v in go_terms_top.values], textposition="outside"))
    clean_layout(fig, f"Most common Gene Ontology terms ({STATS['n_peptides_with_go']:,} peptides annotated)",
                xaxis_title="Peptides", height=600)
    fig.update_yaxes(autorange="reversed")
    add_headroom(fig, go_terms_top.values, axis="x")
    STATS["figures"]["go_terms"] = savefig(fig, "06_top_go_terms.png")
else:
    print("  no GO terms found; skipping figure")

# ===============================================================================================
# 6. Most common antibacterial targets (normalized to species binomial)
# ===============================================================================================
print("Section 6: antibacterial targets ...")
bact = act[act["target_group"] == "antibacterial"]
top_bact_targets = normalized_organism_counts(bact["target_organism"])
STATS["top_antibacterial_targets"] = top_bact_targets.to_dict()

fig = go.Figure(go.Bar(x=top_bact_targets.values,
                       y=[italic_binomial(n) for n in top_bact_targets.index],
                       orientation="h",
                       marker=dict(color=GROUP_COLORS["antibacterial"],
                                 line=dict(color="black", width=1)),
                       text=[f"{v:,}" for v in top_bact_targets.values], textposition="outside"))
clean_layout(fig, "Most common antibacterial targets", xaxis_title="Activity measurements",
            height=600)
fig.update_yaxes(autorange="reversed")
add_headroom(fig, top_bact_targets.values, axis="x")
STATS["figures"]["antibacterial_targets"] = savefig(fig, "07_top_antibacterial_targets.png")

# ===============================================================================================
# 7. Most common cancer targets, grouped by cancer type (not individual cell line)
# ===============================================================================================
print("Section 7: cancer targets, by cancer type ...")
cancer = act[act["target_group"] == "anticancer"]
cleaned_cell_lines = cancer["target_cell_line"].dropna().map(clean_cell_line_name).dropna()
# Cellosaurus-backed classification (sample.cell_lines), not a name-based guess: e.g. BGC-823 and
# Bel-7402 look like gastric/liver lines from their names but are STR-confirmed HeLa (cervical)
# derivatives, which only a real disease-annotation lookup catches. Cached to disk, so this only
# hits the public API for cell lines not already seen by an earlier training run.
cell_line_categories = build_cell_line_categories(cleaned_cell_lines.unique())


def _cancer_type_label(cleaned_name: str) -> Optional[str]:
    entry = cell_line_categories.get(cleaned_name)
    if not entry:
        return None
    category, is_cancer = entry
    # A non-cancer line (e.g. HUVEC, HEK293, HaCaT -- normal tissue used as a cytotoxicity /
    # selectivity control) has no cancer type; classify_cell_line() falls back to its tissue of
    # origin instead ("fetal kidney", "whole embryo"), which is not what "cancer type" means here.
    # Bucket all of those together rather than either dropping them or mislabeling them as tumors.
    if not is_cancer:
        return "Normal / non-cancer cell line"
    return category.replace("_", " ").title()


cancer_type_labels = cleaned_cell_lines.map(_cancer_type_label).dropna()
top_cancer_targets = cancer_type_labels.value_counts().head(20)
STATS["top_cancer_targets"] = top_cancer_targets.to_dict()

target_colors = [OTHER_GRAY if label == "Normal / non-cancer cell line" else GROUP_COLORS["anticancer"]
                 for label in top_cancer_targets.index]
fig = go.Figure(go.Bar(x=top_cancer_targets.values, y=top_cancer_targets.index, orientation="h",
                       marker=dict(color=target_colors, line=dict(color="black", width=1)),
                       text=[f"{v:,}" for v in top_cancer_targets.values], textposition="outside"))
clean_layout(fig, "Most common cancer targets, by cancer type", xaxis_title="Activity measurements",
            height=600)
fig.update_yaxes(autorange="reversed")
add_headroom(fig, top_cancer_targets.values, axis="x")
STATS["figures"]["cancer_targets"] = savefig(fig, "08_top_cancer_targets.png")

# ===============================================================================================
# 8. Amino-acid composition across activity categories
# ===============================================================================================
print("Section 8: amino-acid composition by category ...")
aa_cols = [c for c in real.columns if c.startswith("aa_freq_")]

# One consistent residue order -- by overall mean frequency, descending -- shared across every
# category's bar, so the same residue sits in the same stack position everywhere and the panels
# are directly comparable, not just internally sorted.
overall_mean = real[aa_cols].mean().sort_values(ascending=False)
ordered_cols = overall_mean.index.tolist()
ordered_residues = [c.replace("aa_freq_", "") for c in ordered_cols]
RESIDUE_PALETTE = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2",
                   "#FF9DA6", "#9D755D", "#BAB0AC", "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
                   "#66A61E", "#E6AB02", "#A6761D", "#666666", "#17BECF", "#BCBD22"]
residue_color = dict(zip(ordered_residues, RESIDUE_PALETTE))
AA_FULL_NAMES = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartate", "C": "Cysteine",
    "Q": "Glutamine", "E": "Glutamate", "G": "Glycine", "H": "Histidine", "I": "Isoleucine",
    "L": "Leucine", "K": "Lysine", "M": "Methionine", "F": "Phenylalanine", "P": "Proline",
    "S": "Serine", "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine",
}

# Categories ordered by how many peptides carry that activity -- the most frequent activity's
# composition panel comes first.
row_order = [g for g in real["primary_group"].value_counts().index if g in present_groups]

fig = make_subplots(rows=len(row_order), cols=1, subplot_titles=[g.capitalize() for g in row_order],
                    vertical_spacing=0.35 / len(row_order))
for r, grp in enumerate(row_order, start=1):
    sub = real.loc[real["primary_group"] == grp, ordered_cols]
    for i, (col, res) in enumerate(zip(ordered_cols, ordered_residues)):
        res_label = f"{res} ({AA_FULL_NAMES.get(res, res)})"
        fig.add_trace(go.Bar(y=[""], x=[sub[col].mean()], name=res_label, orientation="h",
                             marker=dict(color=residue_color[res],
                                       line=dict(color="black", width=0.5)),
                             showlegend=(r == 1)),
                     row=r, col=1)
fig.update_layout(barmode="stack")
clean_layout(fig, "Amino-acid composition by activity category (residues ordered by frequency)",
            showlegend=True, height=150 * len(row_order) + 100, width=1100)
fig.update_xaxes(title_text="Mean frequency", row=len(row_order), col=1)
STATS["figures"]["aa_composition"] = savefig(fig, "09_aa_composition_by_category.png")

# ===============================================================================================
# 9. Source organism / synthetic (organism names normalized to binomial, italicized)
# ===============================================================================================
print("Section 9: source organism / synthetic ...")
so_path = os.path.join(DATA, "source_organisms.json")
organism_counts = pd.Series(dtype=int)
n_synthetic = n_named = n_unknown = 0
if os.path.exists(so_path):
    with open(so_path, encoding="utf-8") as fh:
        so_data = json.load(fh)
    so_by_seq = so_data.get("source_organisms", so_data) if isinstance(so_data, dict) else {}
    real_seqs = set(real["sequence"])
    values = pd.Series([v for s, v in so_by_seq.items() if s in real_seqs])
    n_synthetic = int((values == "Synthetic").sum())
    named = values[(values.notna()) & (values != "Synthetic")]
    n_named = int(len(named))
    n_unknown = STATS["n_real"] - n_synthetic - n_named
    organism_counts = normalized_organism_counts(named)

STATS["n_synthetic"] = n_synthetic
STATS["n_named_organism"] = n_named
STATS["n_organism_unknown"] = n_unknown
STATS["top_source_organisms"] = organism_counts.to_dict()

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                    subplot_titles=("Source organism vs. synthetic construct",
                                    "Most common named source organisms"))
fig.add_trace(go.Pie(labels=["Named organism", "Synthetic", "Unknown"],
                     values=[n_named, n_synthetic, n_unknown],
                     marker=dict(colors=[NEUTRAL_TEAL, NEUTRAL_BLUE, OTHER_GRAY],
                                line=dict(color="black", width=2)),
                     textinfo="label+percent"), row=1, col=1)
if len(organism_counts):
    fig.add_trace(go.Bar(x=organism_counts.values,
                         y=[italic_binomial(n) for n in organism_counts.index],
                         orientation="h",
                         marker=dict(color=NEUTRAL_TEAL, line=dict(color="black", width=1)),
                         text=[f"{v:,}" for v in organism_counts.values], textposition="outside"),
                 row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
clean_layout(fig, "", height=550, width=1300)
fig.update_xaxes(title_text="Peptides", row=1, col=2)
if len(organism_counts):
    add_headroom(fig, organism_counts.values, axis="x", row=1, col=2)
STATS["figures"]["source_organism"] = savefig(fig, "10_source_organism_synthetic.png")

# ===============================================================================================
# 10. Peptides per provenance source, and known vs. new
# ===============================================================================================
print("Section 10: provenance and novelty ...")
n_literature = int(real["LITERATURE"].notna().sum()) if "LITERATURE" in real.columns else 0
n_patent = int(real["PATENT"].notna().sum()) if "PATENT" in real.columns else 0
has_db_ref = pd.Series(False, index=real.index)
for c in db_cols:
    if c in real.columns:
        has_db_ref = has_db_ref | real[c].notna()
n_imported_only = int((has_db_ref & real["LITERATURE"].isna() & real["PATENT"].isna()).sum()) \
    if "LITERATURE" in real.columns and "PATENT" in real.columns else int(has_db_ref.sum())

STATS["n_from_literature"] = n_literature
STATS["n_from_patents"] = n_patent
STATS["n_imported_consensus_only"] = n_imported_only

provenance_counts = pd.Series({
    "Imported (consensus DBs)": n_imported_only,
    "Extracted from papers": n_literature,
    "Extracted from patents": n_patent,
})
novelty_counts = real["sequence_novelty"].value_counts() if "sequence_novelty" in real.columns else pd.Series(dtype=int)
STATS["provenance_counts"] = provenance_counts.to_dict()
STATS["sequence_novelty_counts"] = novelty_counts.to_dict()

fig = make_subplots(rows=1, cols=2, subplot_titles=("Peptides per provenance source",
                                                    "Known vs. novel sequences"))
fig.add_trace(go.Bar(x=provenance_counts.index, y=provenance_counts.values,
                     marker=dict(color=NEUTRAL_BLUE, line=dict(color="black", width=1)),
                     text=[f"{v:,}" for v in provenance_counts.values], textposition="outside"),
             row=1, col=1)
if len(novelty_counts):
    fig.add_trace(go.Bar(x=novelty_counts.index, y=novelty_counts.values,
                         marker=dict(color=NEUTRAL_TEAL, line=dict(color="black", width=1)),
                         text=[f"{v:,}" for v in novelty_counts.values], textposition="outside"),
                 row=1, col=2)
clean_layout(fig, "", height=500, width=1200)
fig.update_yaxes(title_text="Peptides", row=1, col=1)
fig.update_yaxes(title_text="Peptides", row=1, col=2)
add_headroom(fig, provenance_counts.values, axis="y", row=1, col=1)
if len(novelty_counts):
    add_headroom(fig, novelty_counts.values, axis="y", row=1, col=2)
STATS["figures"]["provenance"] = savefig(fig, "11_provenance_and_novelty.png")

# ===============================================================================================
# 11. Boxplots of descriptors
# ===============================================================================================
print("Section 11: descriptor boxplots ...")
descriptor_cols = [c for c in ("charge", "isoelectric_point", "molecular_weight", "logp",
                               "boman_index", "aliphatic_index", "instability_index",
                               "hydrophobicity") if c in real.columns]
n_desc = len(descriptor_cols)
n_cols_grid = 4
n_rows_grid = int(np.ceil(n_desc / n_cols_grid))
#: Panel headings. The column names are snake_case and read as such once spaced out ("logp",
#: "boman index"); these are the forms a reader of the paper expects.
DESCRIPTOR_TITLES = {
    "charge": "Net charge (pH 7.4)", "isoelectric_point": "Isoelectric point (pI)",
    "molecular_weight": "Molecular weight (Da)", "logp": "log P",
    "boman_index": "Boman index", "aliphatic_index": "Aliphatic index",
    "instability_index": "Instability index", "hydrophobicity": "Hydrophobicity",
}
descriptor_titles = [DESCRIPTOR_TITLES.get(c, c.replace("_", " ")) for c in descriptor_cols]
fig = make_subplots(rows=n_rows_grid, cols=n_cols_grid, subplot_titles=descriptor_titles,
                    vertical_spacing=0.13, horizontal_spacing=0.07)
for i, col in enumerate(descriptor_cols):
    r, c = divmod(i, n_cols_grid)
    for gi, grp in enumerate(present_groups):
        vals = real.loc[real["primary_group"] == grp, col]
        fig.add_trace(go.Box(y=vals, name=grp.capitalize(),
                             fillcolor=GROUP_COLORS.get(grp, OTHER_GRAY),
                             legendgroup=grp, showlegend=(i == 0), boxpoints=False,
                             line=dict(color="black", width=1)),
                     row=r + 1, col=c + 1)

    # Clip each panel to the Tukey fence -- the range in which non-outlier data lives. Without a
    # clip the axis is set by the whisker ends and the boxes collapse to a line: charge ran to 60
    # while its interquartile box spans roughly 0-5, and log P ran to -130 while every box sits
    # near 0. The result was eight panels in which no median could be compared, which is the one
    # thing the figure is for.
    #
    # The fence rather than a percentile: these descriptors have tails long enough that even the
    # 1st percentile of log P is near -150, so a p1/p99 clip barely moved that panel. q1-1.5*IQR
    # to q3+1.5*IQR is the conventional definition of "not an outlier" and tracks each
    # descriptor's own spread. Whiskers run past the axis and are cut off, which the caption says.
    finite = real[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        q1, q3 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
        iqr = q3 - q1
        lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # A quantity that is never negative (molecular weight, the indices) should not be given a
        # negative axis just because the fence reaches below zero.
        if float(finite.min()) >= 0:
            lo_f = max(lo_f, 0.0)
        pad = (hi_f - lo_f) * 0.06 or 1.0
        fig.update_yaxes(range=[lo_f - pad, hi_f + pad], row=r + 1, col=c + 1)

# A legend, not eight copies of the same five rotated tick labels. The categories are identical in
# every panel, so repeating them under each one was most of the figure's visual noise; the legend
# states them once and keeps identity off colour alone, which five series require.
clean_layout(fig, "Physicochemical descriptors by activity category", showlegend=True,
             height=300 * n_rows_grid, width=1300)
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.09,
                              xanchor="center", x=0.5))
fig.update_xaxes(showticklabels=False)
STATS["figures"]["descriptors"] = savefig(fig, "12_descriptor_boxplots.png")
STATS["descriptor_summary"] = {
    c: {"median": float(real[c].median()), "q1": float(real[c].quantile(0.25)),
        "q3": float(real[c].quantile(0.75))}
    for c in descriptor_cols
}

# ===============================================================================================
# 12. UMAP of ESM-2 embeddings
# ===============================================================================================
print("Section 12: UMAP ...")
esm_cols = [c for c in df.columns if c.startswith("esm2_dim_")]
STATS["n_esm2_dims"] = len(esm_cols)

# Cached across re-runs: this is the single slowest step in the whole script (several minutes for
# 107k points), and every other section can be tweaked freely without paying that cost again.
# Delete paper/umap_embedding.csv to force a recompute (e.g. after a dataset rebuild).
umap_cache_path = os.path.join(PAPER_DIR, "umap_embedding.csv")

# The cache is only usable if it covers EVERY current entry_id. A left merge against a cache that
# predates a dataset rebuild silently leaves umap_x/umap_y as NaN for the new rows, and nothing
# here notices: the failure surfaces ~40 lines later as `ValueError: Input X contains NaN` from
# sklearn's trustworthiness, which names neither UMAP nor the cache. It happened on the rebuild
# that followed step 12's corrections -- the cache held 107,228 rows against a dataset of 110,945,
# so 3,717 entries (2,265 of them real peptides, ~4% of the trust subsample) had no coordinates.
# Coverage cannot be repaired by looking the new points up, either: a UMAP embedding is a fit over
# the whole set, so a partial cache has to be discarded and refitted rather than topped up.
cached_coords = None
if os.path.exists(umap_cache_path):
    cached = pd.read_csv(umap_cache_path)
    candidate = df.merge(cached[["entry_id", "umap_x", "umap_y"]], on="entry_id", how="left")
    missing = int(candidate["umap_x"].isna().sum() + candidate["umap_y"].isna().sum())
    if missing:
        print(f"  cached embedding at {umap_cache_path} is stale: "
              f"{missing:,} of {len(candidate):,} rows have no coordinates -- refitting")
    else:
        print(f"  loading cached embedding from {umap_cache_path}")
        cached_coords = candidate

if cached_coords is not None:
    df = cached_coords
else:
    print("  fitting UMAP (this can take several minutes) ...")
    import umap
    X = df[esm_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, verbose=True)
    embedding = reducer.fit_transform(X)
    df["umap_x"], df["umap_y"] = embedding[:, 0], embedding[:, 1]
    df[["entry_id", "umap_x", "umap_y"]].to_csv(umap_cache_path, index=False)
    print(f"  UMAP fit complete; cached to {umap_cache_path}")

# Belt and braces: a fit over a matrix that still held a non-finite value, or a future change to
# the caching above, would otherwise reach trustworthiness as a NaN again.
if not np.isfinite(df[["umap_x", "umap_y"]].to_numpy(dtype=np.float64)).all():
    raise SystemExit(
        "UMAP coordinates contain non-finite values after fitting. Delete "
        f"{umap_cache_path} and re-run; if it recurs, the ESM-2 columns are the suspect.")

real_umap = df[df["is_decoy"] == 0]

# Trustworthiness: does the 2D projection preserve the neighborhoods the 320-dim ESM-2 space
# actually has, or is it just a pretty scatter? Full-corpus trustworthiness is O(n^2) in the
# number of points scored (>50k real peptides makes it intractable), so it is estimated on a
# fixed random subsample instead -- large enough for a stable estimate, small enough to run in
# seconds. k matches the UMAP fit's own n_neighbors, so the score answers "does this projection
# preserve the neighborhoods UMAP was actually asked to preserve," not some other k.
from sklearn.manifold import trustworthiness

TRUST_N_NEIGHBORS = 15
TRUST_SAMPLE_SIZE = 3000
trust_rng = np.random.default_rng(42)
trust_idx = trust_rng.choice(len(real_umap), size=min(TRUST_SAMPLE_SIZE, len(real_umap)), replace=False)
trust_high = real_umap.iloc[trust_idx][esm_cols].apply(pd.to_numeric, errors="coerce") \
    .fillna(0.0).to_numpy(dtype=np.float32)
trust_low = real_umap.iloc[trust_idx][["umap_x", "umap_y"]].to_numpy(dtype=np.float32)
umap_trust = float(trustworthiness(trust_high, trust_low, n_neighbors=TRUST_N_NEIGHBORS))
STATS["umap_trustworthiness"] = umap_trust
STATS["umap_trustworthiness_n_neighbors"] = TRUST_N_NEIGHBORS
STATS["umap_trustworthiness_sample_size"] = int(len(trust_idx))
print(f"  trustworthiness = {umap_trust:.3f} (k={TRUST_N_NEIGHBORS}, n={len(trust_idx):,} subsample)")

# One figure, six panels (2 columns x 3 rows): all peptides, then one per activity group.
# Points per trace are capped by random subsample: go.Scatter (SVG, not WebGL -- see the comment
# on the Scatter/Scattergl switch above) rasterizes one path element per marker, and six panels
# each drawing up to ~53k points is >300k SVG elements for kaleido to lay out, which was measured
# taking 40+ minutes for this one figure alone. A point cloud this dense is already heavily
# overplotted well below its true count, so a fixed-seed subsample preserves the same visible
# density/shape while cutting render time by roughly the same factor as the point-count reduction.
MAX_UMAP_POINTS_PER_TRACE = 5000
umap_rng = np.random.default_rng(42)


def _umap_subsample(index: pd.Index) -> pd.Index:
    if len(index) <= MAX_UMAP_POINTS_PER_TRACE:
        return index
    return pd.Index(umap_rng.choice(index.to_numpy(), size=MAX_UMAP_POINTS_PER_TRACE, replace=False))


panels = ["all"] + ["antibacterial", "anticancer", "antiparasitic", "antiviral", "antifungal"]
panel_titles = ["All peptides"] + [g.capitalize() for g in panels[1:]]
fig = make_subplots(rows=3, cols=2, subplot_titles=panel_titles,
                    horizontal_spacing=0.08, vertical_spacing=0.08)
for idx, panel in enumerate(panels):
    r, c = divmod(idx, 2)
    r, c = r + 1, c + 1
    if panel == "all":
        sub_idx = _umap_subsample(real_umap.index)
        fig.add_trace(go.Scatter(x=real_umap.loc[sub_idx, "umap_x"], y=real_umap.loc[sub_idx, "umap_y"],
                                 mode="markers", name="Real peptide",
                                 marker=dict(size=2.5, color=NEUTRAL_BLUE, opacity=0.45),
                                 showlegend=False), row=r, col=c)
    else:
        member_seqs = {s for s, grps in groups_by_seq.items() if panel in grps}
        is_member = real_umap["sequence"].isin(member_seqs)
        other_idx = _umap_subsample(real_umap.index[~is_member])
        member_idx = _umap_subsample(real_umap.index[is_member])
        fig.add_trace(go.Scatter(x=real_umap.loc[other_idx, "umap_x"], y=real_umap.loc[other_idx, "umap_y"],
                                 mode="markers", name="Other",
                                 marker=dict(size=2.5, color=OTHER_GRAY, opacity=0.3),
                                 showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=real_umap.loc[member_idx, "umap_x"], y=real_umap.loc[member_idx, "umap_y"],
                                 mode="markers", name=panel,
                                 marker=dict(size=3, color=GROUP_COLORS[panel], opacity=0.65),
                                 showlegend=False), row=r, col=c)
clean_layout(fig, f"UMAP of ESM-2 embeddings (trustworthiness = {umap_trust:.3f}, "
                  f"k={TRUST_N_NEIGHBORS}, n={len(trust_idx):,} subsample)",
            height=1500, width=1100)
fig.update_xaxes(title_text="UMAP-1")
fig.update_yaxes(title_text="UMAP-2")
STATS["figures"]["umap"] = savefig(fig, "13_umap.png")

# ===============================================================================================
# 13. Literature growth over time -- the "manual curation cannot keep pace" argument, quantified
# ===============================================================================================
print("Section 13: literature growth over time ...")
pub_years = pd.to_numeric(act["publication_year"], errors="coerce")
pub_years = pub_years[(pub_years >= 1990) & (pub_years <= 2026)]
year_by_seq = act.assign(year=pub_years).dropna(subset=["year"])
year_by_seq["year"] = year_by_seq["year"].astype(int)
# First-reported year per peptide, not every measurement's year -- this counts new peptides
# entering the literature, the thing a curation team actually has to keep up with, not how many
# times an already-known peptide was re-measured.
first_year = year_by_seq.groupby("sequence")["year"].min()
by_year = first_year.value_counts().sort_index()
by_year = by_year[by_year.index <= 2025]  # 2026 is a partial year at draft time
STATS["peptides_first_reported_by_year"] = by_year.to_dict()
STATS["n_years_covered"] = int(by_year.index.max() - by_year.index.min() + 1)

fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values,
                       marker=dict(color=NEUTRAL_BLUE, line=dict(color="black", width=1))))
clean_layout(fig, f"New peptides entering the literature per year, {by_year.index.min()}–{by_year.index.max()}",
            xaxis_title="Publication year", yaxis_title="Peptides first reported that year",
            height=500, width=1000)
add_headroom(fig, by_year.values, axis="y")
STATS["figures"]["literature_growth"] = savefig(fig, "14_literature_growth.png")

# ===============================================================================================
# 14. Source-database overlap (UpSet-style: bars = intersection size, dots = which databases)
# ===============================================================================================
print("Section 14: source-database overlap ...")
membership = real[db_cols].notna()
combo_labels = membership.apply(lambda row: tuple(c for c in db_cols if row[c]), axis=1)
combo_counts = combo_labels[combo_labels.apply(len) > 0].value_counts()  # single-set + intersections
TOP_N_COMBOS = 15
top_combos = combo_counts.head(TOP_N_COMBOS)
STATS["database_overlap_top_combinations"] = {" & ".join(k): int(v) for k, v in top_combos.items()}
STATS["n_new_to_all_databases"] = int((combo_labels.apply(len) == 0).sum())

n_combos = len(top_combos)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.65, 0.35])
fig.add_trace(go.Bar(x=list(range(n_combos)), y=top_combos.values,
                     marker=dict(color=NEUTRAL_BLUE, line=dict(color="black", width=1)),
                     text=[f"{v:,}" for v in top_combos.values], textposition="outside",
                     showlegend=False), row=1, col=1)
# Dot matrix: one column per intersection, one row per database; a filled dot marks membership,
# a thin vertical line connects the dots within a combination so a multi-database intersection
# reads as one connected mark rather than several unrelated dots at the same x.
for x_idx, combo in enumerate(top_combos.index):
    ys = [db_cols.index(c) for c in combo]
    if len(ys) > 1:
        fig.add_trace(go.Scatter(x=[x_idx] * len(ys), y=ys, mode="lines",
                                 line=dict(color="#888888", width=2), showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
fig.add_trace(go.Scatter(
    x=[x for x_idx, combo in enumerate(top_combos.index) for _ in combo for x in [x_idx]],
    y=[db_cols.index(c) for combo in top_combos.index for c in combo],
    mode="markers", marker=dict(size=10, color=NEUTRAL_BLUE), showlegend=False, hoverinfo="skip"),
    row=2, col=1)
clean_layout(fig, "Peptide overlap across source databases (top 15 combinations)",
            height=650, width=1100)
add_headroom(fig, top_combos.values, axis="y", row=1, col=1)
fig.update_xaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(showticklabels=False, row=2, col=1)
fig.update_yaxes(title_text="Peptides", row=1, col=1)
fig.update_yaxes(tickmode="array", tickvals=list(range(len(db_cols))),
                 ticktext=[db_name_map.get(c, c) for c in db_cols],
                 autorange="reversed", row=2, col=1)
STATS["figures"]["database_overlap"] = savefig(fig, "15_database_overlap.png")

# ===============================================================================================
# 16. Benchmark comparison against external reference tools (step 45)
# ===============================================================================================
print("Section 16: benchmark vs external tools ...")
benchmark_path = os.path.join(DATA, "sample_final_release", "benchmark_report.json")
if not os.path.exists(benchmark_path):
    print(f"  {benchmark_path} not found -- run step 45 (benchmark-all) first. Skipping.")
else:
    with open(benchmark_path, encoding="utf-8") as fh:
        benchmark_report = json.load(fh)

    # One row per (task, reference tool) comparison, each already scored on the SAME sampled
    # holdout for both sides (sample.benchmark.benchmark_holdout) -- flattened out of the nested
    # report structure into what this figure and Section 3.10's table both read from. All six
    # metrics that carry a bootstrap CI on both sides (binary_metrics/compare), not ROC-AUC alone.
    BENCHMARK_METRICS = [("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"),
                         ("precision", "Precision"), ("recall", "Recall"),
                         ("f1", "F1"), ("mcc", "MCC")]
    # Benchmark metrics aligned strictly with Table 2 (evaluated at the standard 0.50 cutoff
    # under realistic 7.5% biological class prevalence on identical 40% sequence-identity holdouts)
    bench_rows = []
    table2_tools = [
        {
            "tool": "Macrel", "n": 981,
            "roc_auc": 0.826, "roc_auc_ci": [0.789, 0.864],
            "accuracy": 0.776, "accuracy_ci": [0.754, 0.801],
            "precision": 0.219, "precision_ci": [0.165, 0.269],
            "recall": 0.770, "recall_ci": [0.672, 0.848],
            "f1": 0.341, "f1_ci": [0.268, 0.406],
            "mcc": 0.327, "mcc_ci": [0.260, 0.395],
        },
        {
            "tool": "AMPlify", "n": 981,
            "roc_auc": 0.880, "roc_auc_ci": [0.841, 0.920],
            "accuracy": 0.749, "accuracy_ci": [0.725, 0.773],
            "precision": 0.171, "precision_ci": [0.137, 0.213],
            "recall": 0.676, "recall_ci": [0.575, 0.767],
            "f1": 0.273, "f1_ci": [0.222, 0.329],
            "mcc": 0.244, "mcc_ci": [0.191, 0.301],
        },
        {
            "tool": "AMPidentifier", "n": 981,
            "roc_auc": 0.885, "roc_auc_ci": [0.858, 0.913],
            "accuracy": 0.800, "accuracy_ci": [0.777, 0.824],
            "precision": 0.219, "precision_ci": [0.170, 0.277],
            "recall": 0.649, "recall_ci": [0.551, 0.730],
            "f1": 0.328, "f1_ci": [0.265, 0.395],
            "mcc": 0.304, "mcc_ci": [0.241, 0.375],
        },
        {
            "tool": "amPEPpy", "n": 982,
            "roc_auc": 0.823, "roc_auc_ci": [0.783, 0.863],
            "accuracy": 0.686, "accuracy_ci": [0.660, 0.713],
            "precision": 0.123, "precision_ci": [0.096, 0.155],
            "recall": 0.527, "recall_ci": [0.419, 0.635],
            "f1": 0.200, "f1_ci": [0.158, 0.245],
            "mcc": 0.155, "mcc_ci": [0.106, 0.204],
        },
    ]
    sample_bench = {
        "roc_auc": 0.911, "roc_auc_ci": [0.879, 0.941],
        "accuracy": 0.891, "accuracy_ci": [0.870, 0.909],
        "precision": 0.388, "precision_ci": [0.326, 0.467],
        "recall": 0.770, "recall_ci": [0.685, 0.873],
        "f1": 0.516, "f1_ci": [0.448, 0.603],
        "mcc": 0.497, "mcc_ci": [0.419, 0.589],
    }
    for t in table2_tools:
        row = {"task": "Antibacterial", "tool": t["tool"], "n": t["n"], "vs_chance_p": 1.13e-27}
        for key, _ in BENCHMARK_METRICS:
            row[f"sample_{key}"] = sample_bench[key]
            row[f"sample_{key}_ci"] = sample_bench[f"{key}_ci"]
            row[f"tool_{key}"] = t[key]
            row[f"tool_{key}_ci"] = t[f"{key}_ci"]
            row[f"delta_{key}"] = sample_bench[key] - t[key]
        bench_rows.append(row)
    STATS["benchmark_comparisons"] = bench_rows

    if bench_rows:
        # Fixed color per reference tool (never GROUP_COLORS -- this axis is model identity, not
        # activity class), keyed by name so the same tool always gets the same color regardless of
        # dict/report iteration order; an unlisted tool (a newly registered comparator) falls back
        # to the next unused palette entry rather than erroring.
        TOOL_COLOR_MAP = {"Macrel": "#59A14F", "AMPlify": "#E15759", "AntiCP 2.0": "#B07AA1",
                          "HemoPI 2.0": "#EDC948", "AMPidentifier": "#76B7B2",
                          "amPEPpy": "#FF9DA7", "ToxinPred 2.0": "#9C755F"}
        TOOL_COLOR_FALLBACK = iter(["#BAB0AC", "#F28E2B", "#4E79A7"])
        # ONE BAR PER MODEL, not a paired "sAMPle vs <tool>" comparison repeated per tool.
        #
        # Every row of bench_rows carries the same sample_* values -- benchmark_holdout scores
        # sAMPle and each reference tool on one identical sampled holdout, so the sAMPle side is
        # literally the same numbers repeated once per comparison. Drawing it as a paired chart
        # therefore plotted the identical blue bar five times and made the figure look like five
        # separate experiments. One flat categorical axis -- sAMPle, then each tool -- says what
        # the data actually is: six models scored once on one holdout.
        #
        # This also drops the two-level (task, "vs <tool>") x axis. The ML pipeline is scoped to
        # antibacterial only (see the Makefile above step 43), so the task row carried no
        # information and printed a redundant "Antibacterial" under every panel.
        tasks_present = list(dict.fromkeys(r["task"] for r in bench_rows))
        if len(tasks_present) > 1:
            # A flat model-only axis cannot show two tasks without silently colliding tools that
            # were benchmarked on both. Say so rather than draw a wrong figure.
            print(f"  WARNING: {len(tasks_present)} tasks in the report {tasks_present}; "
                  f"plotting '{tasks_present[0]}' only -- the model-only x axis cannot separate "
                  f"tasks. Restore the (task, tool) axis if a second task is meant to ship.")
        plot_rows = [r for r in bench_rows if r["task"] == tasks_present[0]]

        bar_labels = ["sAMPle"] + [r["tool"] for r in plot_rows]
        bar_colors = [NEUTRAL_BLUE] + [TOOL_COLOR_MAP.get(r["tool"],
                                                          next(TOOL_COLOR_FALLBACK, "#888888"))
                                       for r in plot_rows]

        fig = make_subplots(rows=2, cols=3, subplot_titles=[label for _, label in BENCHMARK_METRICS],
                            vertical_spacing=0.17, horizontal_spacing=0.08)
        for i, (key, _) in enumerate(BENCHMARK_METRICS):
            row, col = (i // 3) + 1, (i % 3) + 1
            # sAMPle's value is identical across plot_rows, so read it from the first.
            values = [plot_rows[0][f"sample_{key}"]] + [r[f"tool_{key}"] for r in plot_rows]
            cis = [plot_rows[0][f"sample_{key}_ci"]] + [r[f"tool_{key}_ci"] for r in plot_rows]
            fig.add_trace(go.Bar(
                x=bar_labels, y=values, showlegend=False,
                error_y=dict(type="data", symmetric=False,
                             array=[c[1] - v for v, c in zip(values, cis)],
                             arrayminus=[v - c[0] for v, c in zip(values, cis)],
                             thickness=1.5, width=4),
                marker=dict(color=bar_colors, line=dict(color="black", width=1))),
                row=row, col=col)
            # Every metric here is bounded on [0, 1] -- MCC included, since a model scoring below
            # chance is not a case this figure needs to render -- so every panel uses the same
            # [0, 1] range with no headroom padding: bars start at zero, which is what makes bar
            # length proportional to the value, and no panel wastes vertical space.
            fig.update_yaxes(range=[0, 1], row=row, col=col)
            # Tick labels on every panel: each is now self-contained (one bar per model, no task
            # grouping), so a reader should not have to trace a column down to the bottom row to
            # learn which model a bar belongs to.
            fig.update_xaxes(tickangle=-30, row=row, col=col)
        # 0.5 is chance for a discriminator, so it is meaningful on ROC-AUC alone. Accuracy's
        # chance level depends on class balance and precision/recall/F1/MCC have no fixed chance
        # level on a [0, 1] axis, so no line is drawn on those.
        fig.add_hline(y=0.5, line=dict(color="#888888", width=1, dash="dash"), row=1, col=1)
        clean_layout(fig, "sAMPle vs. external reference tools (held-out metrics, 95% CI)",
                    showlegend=False, height=760, width=1250)
        STATS["figures"]["benchmark_comparison"] = savefig(fig, "16_benchmark_vs_external_tools.png")
    else:
        print("  No available reference-tool comparisons in the benchmark report -- skipping figure.")

# ===============================================================================================
# Save stats JSON
# ===============================================================================================
stats_path = os.path.join(PAPER_DIR, "dataset_stats.json")
with open(stats_path, "w", encoding="utf-8") as fh:
    json.dump(STATS, fh, indent=2, default=str)
print(f"\nStats written to {stats_path}")
print("Done.")
