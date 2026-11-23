# sAMPle: A Continuously Updatable Knowledgebase, Multi-Activity Predictor, and Homology-Aware Benchmarking Platform for Bioactive Peptides

**Daniela Peres Martinez¹**, **Isabel Ladeira Pereira¹**, **Rafaella Sinnott Dias¹**, **Luiza Perez Casagrande¹**, **Gratchela Rodrigues Dutra¹**, **João Pedro Gomes Greco¹**, **Déborah Trota Farias de Albernaz¹**, **Ana Lucia Chaves²**, **Daiane Drawanz Hartwig³**, **Luciano da Silva Pinto¹**, **Frederico Schmitt Kremer¹***

¹ Centro de Desenvolvimento Tecnológico (CDTec), Universidade Federal de Pelotas (UFPel), Capão do Leão, Brazil  
² Centro de Ciências Químicas, Farmacêuticas e de Alimentos (CCQFA), Universidade Federal de Pelotas (UFPel), Capão do Leão, Brazil  
³ Instituto de Biologia (IB), Universidade Federal de Pelotas (UFPel), Capão do Leão, Brazil  
* Corresponding author: fred.s.kremer@hotmail.com

---

## Abstract
Antimicrobial resistance (AMR) threatens global public health, driving urgent demand for novel peptide therapeutics. While thousands of new bioactivity measurements enter the scientific literature and patent record annually, existing antimicrobial peptide (AMP) repositories remain static, manually updated snapshots prone to curation lag. Automating biocuration through unconstrained large language models (LLMs) introduces high hallucination risks, unverified assertions, and transcription errors. Here, we introduce **sAMPle** (https://sample-database.org), a continuously updatable knowledgebase, predictive modeling engine, and machine-learning benchmarking platform for bioactive peptides. sAMPle couples deterministic, pattern-based candidate discovery (Aho–Corasick and syntactic regex parsers) with bounded LLM claim verification and strict JSON-Schema gates, guaranteeing that every extraction satisfies deterministic biological rules. Integrating PubMed, Scopus, Google Patents, and six primary repositories (dbAMP 3.0, DRAMP 5.0, DBAASP, CAMP, AntiCP2, APD3), sAMPle unifies **55,879 curated bioactive peptides** (110,945 total records including matched decoys) and **225,858 bioactivity records** across antibacterial, anticancer, antifungal, antiviral, and antiparasitic domains, recovering **830 novel peptides** absent from all precursor databases. Sequences are systematically enriched with physicochemical descriptors, AlphaFold2/ColabFold structures, ESM-2 320-dimensional embeddings, Pfam/near-identity families, and Cellosaurus-standardized cell lines. Crucially, sAMPle exposes and resolves fundamental shortcut-learning and homology-leakage failures in peptide ML: we demonstrate that protein language models (ESM-2) achieve an artifactual ROC-AUC of 0.9999 against shuffled decoys purely by recognizing generator patterns, prompting the implementation of adversarial hard negatives and experimentally measured sub-threshold potency controls under strict 40% CD-HIT sequence-similarity partitioning. On identical held-out sequences evaluated under biological class imbalance (7.5% positive prevalence), sAMPle achieves an ROC-AUC of 0.911 [0.879, 0.941] and a precision-recall AUC (PR-AUC) of 0.469 [0.372, 0.604], markedly outperforming Macrel (PR-AUC 0.275), AMPlify (PR-AUC 0.399), AMPidentifier (PR-AUC 0.332), and amPEPpy (PR-AUC 0.304) at the standard 0.50 operational cutoff. sAMPle further deploys 29 validated models across diverse activities. sAMPle is openly accessible via an interactive web server with 3D pLDDT confidence coloring, DOI/patent links, a versioned REST API, and permanent archives.

---

## 1. Introduction
The relentless dissemination of antimicrobial resistance (AMR) represents one of the paramount crises facing global healthcare, accounting for an estimated 4.95 million associated deaths annually. Conventional small-molecule antibiotics are increasingly compromised by multi-drug-resistant (MDR) bacterial pathogens, prompting widespread exploration of alternative therapeutic modalities. Naturally occurring and rationally engineered antimicrobial peptides (AMPs) have emerged as privileged scaffolds for novel drug development. These evolutionarily ancient host-defense molecules typically disrupt bacterial cell membranes via non-specific electrostatic and hydrophobic interactions, or target conserved intracellular components, thereby presenting a substantially higher barrier to resistance mutation compared to conventional antibiotics.

Despite intense academic and industrial activity, realizing the translational promise of therapeutic peptides is constrained by a fundamental bioinformatics bottleneck: the absence of a unified, continuously updated, machine-learning-ready repository of experimentally measured bioactivities. Established repositories—such as APD3, DRAMP, DBAASP, dbAMP, and CAMP—have served as foundational resources for the computational biology community. However, these databases rely almost exclusively on periodic manual curation. Manual literature extraction is a fixed-rate endeavor operating against an exponential publication trajectory: the rate of newly described peptide bioactivities has expanded by more than two orders of magnitude since the 1990s, exceeding 1,200 newly reported peptides annually in peak years. While resources such as DRAMP index patent-derived entries (e.g., 17,886 in DRAMP 4.0 and 18,715 in the DRAMP 5.0 release integrated here), primary patent gazettes contain vast unstructured bioactivity experimental assays, specialized chemical modifications, and emerging proprietary sequences that remain largely disconnected from standardized ontologies. Most academic biocuration efforts focus on peer-reviewed literature, leaving primary patent claims under-utilized for quantitative potency modeling.

To resolve this challenge, we developed **sAMPle** (https://sample-database.org), an automated, auditable knowledgebase, predictive modeling platform, and benchmarking engine for bioactive peptides. Crucially, sAMPle is not a passive aggregator or meta-database; rather, it is designed from the ground up as an active biocuration and machine-learning infrastructure. Rather than relying on unconstrained generative LLM extraction, sAMPle is constructed around an "NLP-first, LLM-verified" architecture: candidate sequences and assay values are first extracted through deterministic regular expressions, Aho–Corasick automata, and schema-enforced syntactic parsers. Generative models are invoked exclusively for narrow semantic adjudication (e.g., verifying that a quantitative claim directly references the co-extracted peptide and resolving taxonomic synonymy against NCBI databases), and all LLM outputs must satisfy strict deterministic JSON-Schema filters before database ingestion.

Beyond automated mining and multi-database harmonization (integrating 55,879 curated peptides and 225,858 activities), sAMPle addresses severe methodological deficiencies endemic to peptide machine learning. Most contemporary AMP classifiers report exceptional performance (ROC-AUC > 0.95) evaluated on naive random splits or against simplistic shuffled decoys. We demonstrate that this apparent accuracy frequently reflects *shortcut learning* and *homology leakage*: deep protein language models such as ESM-2 separate shuffled decoys from real peptides at an artifactual ROC-AUC of 0.9999 solely by recognizing unnatural n-gram statistics introduced by the generator. To establish true generalization, sAMPle implements an adversarial negative curation framework, experimentally measured sub-threshold potency controls (peptides requiring > 5 μM for antimicrobial effect or showing confirmed lack of inhibition up to assay testing limits, defining a strict high-potency lead threshold), and strict 40% CD-HIT sequence-identity clustering with Pfam superfamily isolation.

---

## 2. Materials and Methods

### 2.1. Literature and Patent Extraction
Automated pipelines harvest scientific literature (PubMed/MEDLINE, Scopus APIs) and patent specifications (Google Patents, USPTO, EPO, WIPO under IPC codes A61K38/00, C07K7/00, C07K14/00). Pattern-based tokenizers and regular expressions extract IUPAC amino acid sequences (5–100 residues) alongside quantitative bioactivity values (MIC, IC50, MBC, EC50, Ki, percent inhibition).

### 2.2. Deterministic Verification and Curation Audit
Candidate sequence–assay assertions are paired with full sentence context and submitted to a containerized LLM validator. The validator verifies direct attribution and standardizes organism and cell line names to NCBI Taxonomy and Cellosaurus identifiers. LLM outputs must pass JSON-Schema gates enforcing biological plausibility (positive concentration values, valid amino acids, structural length limits).

In a stratified human curation audit of 1,000 documents (800 articles, 200 patents; 2,991 activity rows):
- **Sequence String Fidelity:** 98.4%
- **Taxonomic Assignment Accuracy:** 92.6%
- **Quantitative Potency and Unit Resolution:** 91.2%
- **Spurious Entity Rate:** < 0.5%
- **Extraction Recall on Negative Strata:** Document-level recall was 88.7% and peptide-level extraction recall was 86.4% on author-tested therapeutic sequences.
- **Inter-Annotator Agreement:** Double-blind evaluation of a 100-document subset between two independent biocurators achieved Cohen's $\kappa = 0.89$ on sequence entity identification and $\kappa = 0.85$ on quantitative potency/unit assignments.

### 2.3. Multi-Database Harmonization
Ingests six foundational databases: dbAMP 3.0 (32,958 peptides), DRAMP 5.0 (31,255), DBAASP v3.0 (16,754), CAMP v4.0 (16,136), AntiCP2 (4,709), and APD3 (3,306). Sequences are normalized to uppercase IUPAC single-letter codes; duplicates are mapped to stable sAMPle accessions (`SAMPLE0000000001` to `SAMPLE0000055879`).

### 2.4. Physicochemical Profiling and Structural Modeling
Every peptide is enriched across four analytical layers:
- **Physicochemical Descriptors:** Isoelectric point (pI), net charge at pH 7.4 accounting for C-terminal amidation via the Henderson–Hasselbalch equation, Boman index, instability index, aliphatic index, and Kyte–Doolittle GRAVY hydropathicity computed via `peptides.py` and Biopython's `ProtParam`. Molecular weight, partition coefficient (log P), and topological polar surface area (TPSA) calculated with RDKit from sequence-derived SMILES graphs.
- **Secondary Structure and Amphipathicity:** Hydrophobic moments for $\alpha$-helical and $\beta$-sheet conformations, alongside Eisenberg hydrophobicity scales.
- **Deep Protein Language Representations:** Dense 320-dimensional contextual embeddings from ESM-2 (8M checkpoint, `facebook/esm2_t6_8M_UR50D`).
- **3D Conformation and pLDDT Confidence:** AlphaFold2/ColabFold atomic coordinates with per-residue pLDDT confidence scores (very high >90, confident 70–90, low 50–70, very low <50).

### 2.5. Dataset Serialization and Multi-Modal Packaging
To adhere strictly to FAIR data stewardship principles and facilitate high-throughput offline re-analysis without database bottlenecks, sAMPle provides a unified serialized distribution package (`sample_records_json_pdb_sdf.zip`, 557.07 MB). The archive synchronizes:
1. **Structured JSON Metadata (`json/<entry_id>.json`):** Individual JSON records for all 55,879 curated bioactive peptides, containing primary sequence strings, biophysical descriptors, curated quantitative bioactivity tables (target organisms, normalized potency in $\mu\text{M}$ and $-\log_{10}[\text{M}]$, assay types), NCBI taxonomy IDs, Cellosaurus mappings, and complete literature/patent bibliographic cross-references.
2. **AlphaFold2 Tertiary Coordinates (`pdb/<entry_id>.pdb`):** Predicted 3D atomic coordinates for 48,664 peptides with per-residue pLDDT confidence metrics recorded in the B-factor column.
3. **3D Molecular Conformers (`sdf/<entry_id>.sdf`):** 55,874 energy-minimized small-molecule/peptide conformers in MDL SDF format generated via RDKit with MMFF94 force-field refinement.

This archive is generated via `sample.serialize_archive` and indexed on both Zenodo and the Datasets tab of the web portal.

### 2.6. Negative Class Design and Adversarial Hard Negatives
The negative evaluation space comprises two complementary tiers:
1. **Tripartite Synthetic Decoy Library (55,066 sequences):**
   - *Composition-Matched Shuffled Sequences (17,922 seqs; 32.5%):* Permuted residues preserving exact composition and length.
   - *Filtered Swiss-Prot Non-AMP Fragments (18,593 seqs; 33.8%):* Fragments devoid of antimicrobial, toxin, defense, or cytolytic annotations.
   - *Adversarial Hard Negatives (18,551 seqs; 33.7%):* Swiss-Prot fragments specifically filtered to mimic the cationic and amphipathic properties of real AMPs (net charge ≥ +2 at pH 7.4, hydrophobic fraction ≥ 0.30, and <70% identity to any known AMP via CD-HIT).
2. **Experimental Potency Stratum (three bands):**
   - Every (peptide, target, assay) triple is assigned a potency band from the median of its repeat measurements: **high** (≤ 5 μM), **low** (> 5 and ≤ 100 μM) and **inactive** (> 100 μM). Positives are the high band; negatives are the inactive band plus the decoys. The **low band is excluded from classification entirely** — it is neither a therapeutic lead nor an inactive, and putting it in either class would be a false statement about a real measurement.
   - Why the middle band exists: a single bar at 5 μM produced 93,124 negative pairs of which **67,158 (72%) sat between 5 and 100 μM** — peptides with real, moderate, published activity being asserted as inactive, against a literature convention that puts the inactive bar at > 50–100 μM or > 128 μg/mL. Excluding the band costs 56% of measured pairs and gives a near-balanced design (26,374 high vs 25,966 inactive) in which a negative means a peptide that was tested and genuinely did not work.

### 2.7. Homology-Aware Partitioning
- Clustered with CD-HIT at strict 40% sequence identity (`c = 0.40`).
- 5-fold cross-validation and independent holdouts partitioned via `StratifiedGroupKFold` ensuring homologous clusters reside strictly on either the training or testing partition.
- Pfam domains mapped via HMMER3 (`hmmsearch`) against Pfam-A (646 families identified).

### 2.8. Machine Learning Pipeline
XGBoost classifiers and regressors optimized with FLAML:
- Hyperparameter tuning optimizes balanced objective loss functions preserving probability calibration.
- Models operate at the standard operational decision cutoff of $\tau = 0.50$ across all classification tasks and external benchmark comparisons, ensuring fair and unbiased evaluation under realistic class prevalence without artificial post-hoc threshold shifting.
- Statistical significance is assessed via 100-iteration bootstrap 95% confidence intervals and 5-repeat y-randomization null models (models rejected unless holdout ROC-AUC ≥ 0.75 with p ≤ 0.05).
- Applicability domain is parameterized via 5-nearest-neighbor Euclidean distance in standardized feature space (flagging test queries > 3 SDs from training mean).

### 2.9. Peptide–Antibiotic Combination Assay Curation
An initial extraction of 5,507 experimental checkerboard combination assays from DBAASP was filtered using automated quality criteria: entries with missing FICI, peptides < 5 amino acids, or unresolvable bacterial species were excluded, and all records reporting both individual and combination MICs were cross-checked against the theoretical definition ($\text{FICI} = \text{MIC}_{\text{comb}}^{\text{pep}}/\text{MIC}_{\text{alone}}^{\text{pep}} + \text{MIC}_{\text{comb}}^{\text{abx}}/\text{MIC}_{\text{alone}}^{\text{abx}}$), discarding 654 records exhibiting > 25% mathematical discrepancy or missing single-agent controls. This yielded a rigorously audited benchmark dataset of 4,853 experimental synergy measurements (2,394 synergistic at $\text{FICI} \le 0.5$, 2,459 non-synergistic). These curated measurements are released as a benchmark dataset for downstream method development; sAMPle does not deploy a synergy predictor, as models trained on this set did not meet the held-out validation thresholds required for release.

### 2.10. Web Server Architecture and Asynchronous Prediction Worker
Built with FastAPI, Uvicorn, Jinja2, TailwindCSS, and SQLAlchemy ORM backed by SQLite:
- **Asynchronous Task Architecture:** Heavy compute tasks (ESM-2 embedding generation and ML scoring across 29 bioactivity models) are decoupled from the HTTP cycle via a dedicated worker process (`sample.worker`). Submissions are assigned UUIDs and enqueued (Redis/Celery with automated SQLite fallback), allowing the web interface to display non-blocking real-time status transitions from `PENDING` to `SUCCESS`.
- **Search Services:** Local NCBI BLAST+ (`blastp-short`), HMMER3 (`hmmsearch`) against 646 Pfam-A defense domains, and 320-dimensional cosine vector search.
- **Curated Provenance:** External DOIs resolved to publisher journal pages; patent disclosures resolved to Google Patents.
- **3D Visualization:** Client-side Mol* and 3Dmol.js rendering AlphaFold structures with pLDDT confidence coloring.
- **OpenAPI REST API:** OpenAPI/Swagger documented endpoints (`/api/v1`) for programmatic batch retrieval and prediction.
- **5-Year Maintenance Commitment:** Hosted at https://sample-database.org with mobile and tablet responsive layouts.

### 2.11. Unsupervised Anomaly Detection
An unsupervised Isolation Forest algorithm fitted across all 110,945 records on a joint 361-feature matrix — 13 physicochemical descriptors, 28 amino-acid composition features (single-residue and residue-group frequency spectra), and the 320 dimensions of the ESM-2 embedding — detects atypical sequences (`contamination='auto'`). Because the forest draws its split feature uniformly at random, the embedding accounts for 88.6% of split decisions; the per-block composition is recorded in the run's report, and `--esm-components N` reduces the embedding block to N principal components when a more even balance is wanted. Anomalies are surfaced on detail cards as quality caveats and can be excluded via `--filter-anomalies`.

---

## 3. Results

### 3.1. Corpus Composition and Functional Distribution
The release comprises **110,945 total records**: **55,879 unique curated bioactive peptides** and 55,066 matched decoy sequences, encompassing **225,858 distinct bioactivity measurements** across pathogens, cancer cell lines, and control tissues (Figure 1). That count is the released activity table (`activities.csv`), which retains every harmonized measurement, including those whose peptide is not itself a released entry; the queryable database and REST API expose the 204,370 of them that attach to a peptide record, so the two figures differ by construction rather than by loss.

By provenance:
- 50,744 peptides originated exclusively from consensus biological databases;
- 5,027 solely from peer-reviewed literature mining;
- 82 solely from patent extractions;
- 26 were co-identified across both literature and patent disclosures.
- **Sum:** 50,744 + 5,027 + 82 + 26 = **55,879** total curated active peptides (5,053 literature-supported and 108 patent-supported).

A total of 23,646 curated peptides carry quantitative experimental bioactivity measurements in the database. Because antimicrobial peptides are frequently multi-functional, individual peptides frequently span multiple activity categories:
- Antibacterial: 17,495 unique peptides (168,943 measurements);
- Antifungal: 5,245 unique peptides (20,926 measurements);
- Anticancer: 7,502 unique peptides (17,270 measurements);
- Antiviral: 351 unique peptides (551 measurements);
- Antiparasitic: 308 unique peptides (744 measurements);
- Other: 4,332 unique peptides (17,424 measurements).
The remaining 32,233 peptides in the curated corpus represent qualitative AMP entries, natural defense sequences, or structural records from precursor databases without parsed quantitative MIC concentration assays.

Source database distribution (Figure 2): dbAMP 3.0 (32,958), DRAMP 5.0 (31,255), DBAASP (16,754), CAMP (16,136), AntiCP2 (4,709), APD3 (3,306).

![**Figure 1.** Sequences and bioactivity measurements across functional classes.](figures/01_sequences_and_activities_per_group.png)

![**Figure 2.** Peptide contribution across integrated precursor databases.](figures/02_peptides_per_source_database.png)

### 3.2. Literature Growth Dynamics and Novel Mining Yield
Literature output surged past 500 new peptides per year in 2005, peaking at 1,238 in 2018 (Figure 14). Over the past decade, output sustained between 600 and 1,200 novel peptides annually.

Cross-database intersection analysis reveals significant sequence redundancy across legacy databases (Figure 15). Crucially, sAMPle's continuous literature and patent mining engines surfaced **830 newly discovered peptides** entirely absent from all six precursor databases (with 1,699 newly reported sequences across cross-database combinations), alongside 3,915 corroborated sequences (Figure 11).

![**Figure 14.** Annual publication trajectory of newly discovered bioactive peptides.](figures/14_literature_growth.png)

![**Figure 15.** Sequence overlap across six integrated precursor databases.](figures/15_database_overlap.png)

![**Figure 11.** Peptide provenance and sequence novelty.](figures/11_provenance_and_novelty.png)

### 3.3. Peptide Family Landscape and Sequence Diversity
Pfam-A profiling anchored 14,544 peptides (26.0% of the active corpus) to 646 distinct Pfam families (Figure 4), comprising 218 singleton families and 428 multi-member families. Dominant families include Defensin_beta_2 (PF13841, 746 members), Gamma-thionin (PF00304, 709), Thaumatin (PF00314, 706), FSAP signal propeptide (PF03032, 692), Antimicrobial_2 (PF08023, 578), Cyclotide (PF03784, 516), Defensin_beta (PF00711, 473), and Cathelicidin (PF00666, 193).

Of the peptides lacking Pfam domain hits, 38,064 were grouped using CD-HIT into 20,763 near-identity clusters at ≥ 80% identity and coverage (2,428 multi-member families), while 3,271 represent sequence singletons possessing neither Pfam domain hits nor clustering partners (14,544 + 38,064 + 3,271 = 55,879 total curated peptides).

![**Figure 4.** Top Pfam families represented in sAMPle.](figures/04_top_pfam_families.png)

![**Figure 5.** Proportion of active peptides assigned to Pfam families, clusters, and singletons.](figures/05_family_assignment_pizza.png)

### 3.4. Physicochemical Landscape and Structural Embeddings
Across the 55,879 curated peptides, median values are: net charge at pH 7.4 of +2.61 (IQR: 0.05 to +4.94), isoelectric point (pI) of 9.84 (IQR: 7.55 to 11.65), molecular weight of 2,248.5 Da (IQR: 1,478.9 to 3,782.8 Da), and partition coefficient log P of **-7.20** (Figure 12).

2D UMAP projection of ESM-2 contextual embeddings and biophysical features achieves a neighborhood trustworthiness of 0.926 ($k=15$, sample size 3,000; Figure 13).

![**Figure 12.** Physicochemical property distributions across major bioactivity classes. Boxes give the interquartile range and median; whiskers follow the Tukey convention. Each panel's vertical axis is clipped to that descriptor's Tukey fence (Q1 − 1.5·IQR to Q3 + 1.5·IQR), so whiskers reaching beyond it are cut off — these descriptors have tails long enough (log P to −150) that an unclipped axis compresses every box to a line.](figures/12_descriptor_boxplots.png)

![**Figure 13.** UMAP structural landscape of 55,879 bioactive peptides.](figures/13_umap.png)

### 3.5. Isolation Forest Identifies Mechanistic Outliers
Unsupervised Isolation Forest flagged **6,224 sequences (5.6% of release records)** as significant anomalies (anomaly score < 0). Against the non-anomalous remainder, anomalies show a median net charge at pH 7.4 of +4.99 (IQR: +1.99 to +7.99) versus +1.99 (IQR: -0.01 to +3.97), a median isoelectric point of 11.50 (IQR: 9.54 to 12.81) versus 9.70 (IQR: 6.53 to 11.29), median hydropathy of -0.76 versus -0.26, and a median Boman index of 2.32 versus 1.59 (all $p < 10^{-15}$, Mann–Whitney U-test) — reflecting poly-cationic synthetic constructs and proline-rich peptides. Length is not the discriminant: median 19 residues among anomalies versus 20 among the rest.

### 3.6. Exposing Shortcut Learning: Shuffled Decoys vs. Real Peptides
In controlled benchmarks evaluating feature blocks against length- and composition-matched shuffled decoys (40% CD-HIT split), ESM-2 embeddings achieved a near-perfect holdout ROC-AUC of **0.9999** (95% CI [1.000, 1.000]) and PR-AUC of **0.9995** on antibacterial prediction (0.980 precision, 0.992 recall), and ROC-AUC **0.9999** / PR-AUC **0.9987** on anticancer prediction. Conversely, physicochemical descriptors and composition features reached ROC-AUCs of only 0.660 and 0.584 on antibacterial, and 0.615 and 0.577 on anticancer data.

This demonstrates that deep neural language models memorize non-natural higher-order n-gram transitions created by random sequence permutation, establishing the necessity of sAMPle's adversarial hard negatives and experimentally measured sub-threshold biological controls.

### 3.7. Benchmarking Against External Predictors
We benchmarked sAMPle's FLAML-searched antibacterial classifier against four established predictive algorithms: Macrel, AMPlify, AMPidentifier, and amPEPpy on a reproducible random holdout sample (N = 1,000, seed 0, 40% sequence-identity cutoff; Table 2, Figure 16). Sequences unhandled by reference tool input parsers were filtered, yielding n = 981 valid evaluation sequences for Macrel, AMPlify, and AMPidentifier (74 active AMPs, 907 negatives; 7.54% positive prevalence) and n = 982 for amPEPpy. Both sAMPle and all reference predictors were evaluated at the standard operational decision cutoff of 0.50, alongside threshold-free ranking metrics (ROC-AUC and Precision-Recall AUC).

**Table 2. Head-to-head benchmarking of sAMPle against external state-of-the-art predictors on identical held-out sequences (40% sequence-identity cutoff).** Values in brackets indicate bootstrap 95% confidence intervals; Δ denotes sAMPle minus reference tool. Both sAMPle and reference models operate at the standard 0.50 decision threshold under realistic biological class imbalance (7.5% positive prevalence).
| Reference Comparison | Metric | sAMPle [95% CI] | Reference Tool [95% CI] | Difference (Δ) |
|---|---|---|---|---|
| **vs. Macrel** (n = 981) | ROC-AUC | **0.911** [0.879, 0.941] | 0.826 [0.789, 0.864] | **+0.085** |
| | PR-AUC | **0.469** [0.372, 0.604] | 0.275 [0.194, 0.366] | **+0.194** |
| | Accuracy | **0.891** [0.870, 0.909] | 0.776 [0.754, 0.801] | **+0.115** |
| | Precision | **0.388** [0.326, 0.467] | 0.219 [0.165, 0.269] | **+0.169** |
| | Recall | 0.770 [0.685, 0.873] | 0.770 [0.672, 0.848] | 0.000 |
| | F1-score | **0.516** [0.448, 0.603] | 0.341 [0.268, 0.406] | **+0.175** |
| | MCC | **0.497** [0.419, 0.589] | 0.327 [0.260, 0.395] | **+0.170** |
| **vs. AMPlify** (n = 981) | ROC-AUC | **0.911** [0.879, 0.941] | 0.880 [0.841, 0.920] | **+0.031** |
| | PR-AUC | **0.469** [0.372, 0.604] | 0.399 [0.332, 0.513] | **+0.070** |
| | Accuracy | **0.891** [0.870, 0.909] | 0.749 [0.725, 0.773] | **+0.142** |
| | Precision | **0.388** [0.326, 0.467] | 0.171 [0.137, 0.213] | **+0.217** |
| | Recall | 0.770 [0.685, 0.873] | 0.676 [0.575, 0.767] | +0.094 |
| | F1-score | **0.516** [0.448, 0.603] | 0.273 [0.222, 0.329] | **+0.243** |
| | MCC | **0.497** [0.419, 0.589] | 0.244 [0.191, 0.301] | **+0.253** |
| **vs. AMPidentifier** (n = 981) | ROC-AUC | **0.911** [0.879, 0.941] | 0.885 [0.858, 0.913] | **+0.026** |
| | PR-AUC | **0.469** [0.372, 0.604] | 0.332 [0.256, 0.445] | **+0.137** |
| | Accuracy | **0.891** [0.870, 0.909] | 0.800 [0.777, 0.824] | **+0.091** |
| | Precision | **0.388** [0.326, 0.467] | 0.219 [0.170, 0.277] | **+0.169** |
| | Recall | 0.770 [0.685, 0.873] | 0.649 [0.551, 0.730] | +0.121 |
| | F1-score | **0.516** [0.448, 0.603] | 0.328 [0.265, 0.395] | **+0.188** |
| | MCC | **0.497** [0.419, 0.589] | 0.304 [0.241, 0.375] | **+0.193** |
| **vs. amPEPpy** (n = 982) | ROC-AUC | **0.911** [0.879, 0.941] | 0.823 [0.783, 0.863] | **+0.088** |
| | PR-AUC | **0.469** [0.372, 0.604] | 0.304 [0.224, 0.415] | **+0.165** |
| | Accuracy | **0.891** [0.870, 0.909] | 0.686 [0.660, 0.713] | **+0.205** |
| | Precision | **0.388** [0.326, 0.467] | 0.123 [0.096, 0.155] | **+0.265** |
| | Recall | 0.770 [0.685, 0.873] | 0.527 [0.419, 0.635] | +0.243 |
| | F1-score | **0.516** [0.448, 0.603] | 0.200 [0.158, 0.245] | **+0.316** |
| | MCC | **0.497** [0.419, 0.589] | 0.155 [0.106, 0.204] | **+0.342** |

*Note:* sAMPle demonstrates overwhelming statistical significance against the empirical y-randomization chance null baseline ($p = 1.13 \times 10^{-27}$, Welch's t-test; empirical null mean ROC-AUC = 0.515). While bootstrap 95% confidence intervals for ROC-AUC overlap between sAMPle, AMPlify, and AMPidentifier, sAMPle achieves substantial gains in PR-AUC (+0.070 to +0.194) and operational precision under biological class imbalance.

![**Figure 16.** Six-panel comparison of sAMPle against Macrel, AMPlify, AMPidentifier, and amPEPpy across ROC-AUC, Accuracy, Precision, Recall, $F_1$-score, and Matthews correlation coefficient (MCC) on strictly partitioned held-out peptides ($n=981$ to $982$, 40% sequence-identity cutoff) with 95% bootstrap confidence intervals. All predictors operate at the standard 0.50 decision threshold under biological class imbalance (7.5% positive prevalence).](figures/16_benchmark_vs_external_tools.png)

### 3.8. Stratified Negative-Stratum Ablation
To verify that sAMPle does not rely on trivial physicochemical shortcuts across its negative classes, model specificity and false-positive rates were evaluated across each negative stratum on held-out test sequences ($N=13,188$):
- **Filtered Swiss-Prot Non-AMP Fragments ($n=500$):** Specificity = **0.998** (False Positive rate = 0.002, Mean Probability = 0.053). The model decisively rejects natural non-antimicrobial sequences with near-zero false alarms.
- **Adversarial Hard Negatives ($n=500$):** Specificity = **0.982** (False Positive rate = 0.018, Mean Probability = 0.080). Even when faced with challenging decoys designed to mimic real AMPs in charge (≥ +2) and hydrophobicity (≥ 0.30), the model maintains an exceptionally high 98.2% specificity.
- **Composition-Matched Shuffled Sequences ($n=500$):** Specificity = **0.934** (False Positive rate = 0.066, Mean Probability = 0.168).
- **Authentic Active Positives ($n=500$):** Sensitivity (Recall) = **0.502** at the uncalibrated 0.50 threshold (Mean Probability = 0.490; full-spectrum ROC-AUC = 0.861–0.911).

This ablation directly proves that sAMPle has learned genuine antimicrobial sequence motifs rather than relying on cationic charge or hydrophobic fraction heuristics.

### 3.9. Multi-Target Bioactivity and Pathogen-Specific MIC Models
sAMPle trains and validates 29 independent machine-learning models across broad functional categories and pathogen-specific minimum inhibitory concentrations:
- **Broad Functional Classes (Global Holdouts):** Antibacterial (ROC-AUC 0.861, PR-AUC 0.435, Acc 0.895, F1 0.455), Antifungal (ROC-AUC 0.911, PR-AUC 0.683, Acc 0.942, F1 0.598), Antiviral (ROC-AUC 0.904, PR-AUC 0.525, Acc 0.997, F1 0.607), Anticancer (ROC-AUC 0.801, PR-AUC 0.494, Acc 0.885, F1 0.452), Antiparasitic (ROC-AUC 0.913, PR-AUC 0.154, Acc 0.996, F1 0.151).
- **Pathogen-Specific MIC Models (MIC ≤ 5 μM):** High discriminatory accuracy predicting potent activity: *A. baumannii* (ROC-AUC 0.977, PR-AUC 0.498, Acc 0.991), *C. albicans* (ROC-AUC 0.967, PR-AUC 0.663, Acc 0.993), *B. subtilis* (ROC-AUC 0.964, PR-AUC 0.555, Acc 0.984), *K. pneumoniae* (ROC-AUC 0.948, PR-AUC 0.392, Acc 0.994), *E. coli* (ROC-AUC 0.944, PR-AUC 0.614, Acc 0.971), *S. aureus* (ROC-AUC 0.950, PR-AUC 0.621, Acc 0.978), *P. aeruginosa* (ROC-AUC 0.937, PR-AUC 0.542, Acc 0.987), *S. epidermidis* (ROC-AUC 0.971, PR-AUC 0.453, Acc 0.990).

### 3.10. Web Platform, Interactive Visualizations, and Usability
The sAMPle knowledgebase is freely accessible through https://sample-database.org:
1. Dynamic Sequence Search: exact sequence matching, local NCBI BLAST+, HMMER3 Pfam-A search, and 320-dimensional cosine vector similarity search;
2. Peptide Explorer: multi-criteria filtering by target organism, minimum quantitative potency, activity category, source database, with instant CSV and FASTA export;
3. Interactive 3D Structure Viewer: integrated Mol* and 3Dmol.js rendering AlphaFold2 atomic structures with residue pLDDT confidence coloring;
4. Auditable Provenance Links: external DOIs resolved to publisher journal pages and patent disclosures resolved to Google Patents;
5. Real-Time Asynchronous Prediction Workbench: interactive ML interface accepting custom peptide sequences for real-time bioactivity scoring, powered by `sample.worker` with live `PENDING` to `SUCCESS` updates;
6. FAIR Datasets Hub and Multi-Modal Serialized Downloads: dedicated `/datasets` tab providing direct one-click downloads for tabular TSV/CSV files, SQLite relational databases, FASTA sequence catalogs, and the unified 557.07 MB multi-modal archive (`sample_records_json_pdb_sdf.zip`) containing individual JSON metadata for all 55,879 curated peptides, 48,664 AlphaFold2 PDB structures, and 55,874 RDKit SDF conformers;
7. RESTful API: OpenAPI/Swagger-documented endpoints (`/api/v1`) for programmatic sequence retrieval and prediction;
8. Long-term maintenance commitment of at least 5 years with mobile and tablet responsive layouts.

![**Figure 17.** The sAMPle web portal and interactive exploration interface. (A) Explore and Search dashboard with real-time multi-criteria filtering across 55,879 peptides and 225,858 bioactivity records. (B) Interactive 3D molecular conformation viewer rendering AlphaFold2 structures with per-residue pLDDT confidence coloring and structural validation. (C) Detailed experimental assay table with target organisms, standardized potency ($-\log_{10}[\text{M}]$), and direct literature provenance links to PubMed DOIs and Google Patents.](figures/17_web_portal_interface.png)

---

## 4. Discussion
Automated extraction of biomedical data has reached a pivotal juncture. While large language models present unprecedented natural-language fluency, their unconstrained deployment in biocuration carries unacceptable risks of error propagation. By relegating the generative model to bounded semantic triage and enforcing strict JSON-Schema and regex gating, sAMPle establishes a reliable paradigm for continuous biocuration, demonstrated by >98% sequence fidelity across our 1,000-document audit.

A second central contribution is exposing shortcut learning in peptide machine learning: ESM-2 embeddings achieve an ROC-AUC of 0.9999 when classifying authentic peptides against shuffled decoys, revealing that language models detect grammatical regularities of synthetic generators rather than functional antimicrobial motifs. By introducing adversarial hard negatives and experimentally measured sub-threshold potency controls under strict 40% sequence-identity partitioning, sAMPle provides a benchmark that measures authentic therapeutic generalization.

Finally, while sAMPle currently incorporates 830 newly mined peptides not present in precursor databases, its primary scientific value lies not merely in sequence aggregation, but in providing an open, reproducible biocuration infrastructure that bridges literature text-mining, structural modeling, and multi-activity prediction. The inclusion of a unified multi-modal archive (`sample_records_json_pdb_sdf.zip`) containing individual JSON records alongside AlphaFold2 PDB coordinates and RDKit SDF conformers empowers offline machine-learning workflows without external dependencies, while the asynchronous worker engine ensures reliable online execution.

---

## 5. Conclusion
sAMPle provides a continuously extensible, transparent, and statistically validated knowledgebase bridging the divide between literature mining and predictive peptide engineering. By aggregating 55,879 curated peptides, 225,858 bioactivity measurements, patent dark data, and structural confidence profiles into an open platform, sAMPle establishes a new benchmark for biocuration integrity in the era of automated biomedical text mining.

---
*Code and Data Availability: The entire sAMPle codebase, web server, and extraction scripts are freely available on GitHub (https://github.com/firas-bioinfo/sAMPle and mirror https://github.com/omixlab/sample-database). Release datasets, embeddings, trained models, and the unified serialized multi-modal archive (`sample_records_json_pdb_sdf.zip`, 557.07 MB) are permanently hosted on Zenodo (https://doi.org/10.5281/zenodo.14885834) and accessible via the web portal Datasets tab (https://sample-database.org/datasets).*
