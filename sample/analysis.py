import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModel
import umap
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("sample.analysis")

#: Protein language model used for the embedding projection. ESM-2's 8M checkpoint: small enough
#: to run on any GPU here, distributed as safetensors, and takes raw sequences. It replaced
#: ProtBERT, which ships only a `pytorch_model.bin` that transformers 5.x will not load below
#: torch 2.6 -- a version this project cannot take without giving up its CUDA 12.1 pin.
ESM2_MODEL = "facebook/esm2_t6_8M_UR50D"

#: Physicochemical descriptors the anomaly detector fits. The list used to stop at
#: ``sequence_length``, which left isoelectric point, net charge at pH 7.4, the Boman index,
#: hydrophobicity and three others sitting unused in the release CSV -- while the manuscript
#: named two of them as inputs. Every column here is present in `final_dataset.csv`.
ANOMALY_DESCRIPTOR_COLUMNS: Tuple[str, ...] = (
    "molecular_weight", "logp", "hydrogen_bond_donors", "hydrogen_bond_acceptors",
    "charge", "sequence_length", "isoelectric_point", "net_charge_ph7", "boman_index",
    "aliphatic_index", "instability_index", "hydrophobicity", "hydrophobic_moment",
)

#: Prefixes of the amino-acid composition block (frequency spectra).
ANOMALY_COMPOSITION_PREFIXES: Tuple[str, ...] = ("aa_freq_", "aa_group_freq_")

#: Prefixes of the protein-language-model block. ESM-2 is what the current release carries;
#: ``protbert_dim_`` is kept because releases built before the ProtBERT->ESM-2 switch have it.
ANOMALY_EMBEDDING_PREFIXES: Tuple[str, ...] = ("esm2_dim_", "protbert_dim_")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# CSS and design palette settings for plots (matching web theme)
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = '#0b0f19'
plt.rcParams['axes.facecolor'] = '#111827'
plt.rcParams['axes.edgecolor'] = '#374151'
plt.rcParams['axes.labelcolor'] = '#f3f4f6'
plt.rcParams['xtick.color'] = '#9ca3af'
plt.rcParams['ytick.color'] = '#9ca3af'
plt.rcParams['text.color'] = '#f3f4f6'
plt.rcParams['grid.color'] = '#374151'

# Color Palette: Indigo & Purple (Active), Slate & Gray (Decoy)
PALETTE = {
    "Active": "#6366f1",  # Indigo
    "Decoy": "#94a3b8"   # Slate
}

from sample.sequence import get_peptide_charge

def run_exploratory_data_analysis(dataset_csv: str, output_dir: str, sample_size: int = 500,
                                  activities_csv: Optional[str] = None) -> bool:
    """
    Generates exploratory data analysis plots, including:
      - Distribution of peptide sequence lengths
      - Frequency distributions of amino acid residues
      - Distribution of net charges
      - UMAP of Morgan fingerprints (2D structure similarity)
      - UMAP of ESM-2 embeddings (sequence similarity)
      - Target-organism distribution (top pathogens by activity count)

    :param dataset_csv: Peptide-level dataset (sequence, charge, decoy label). This is
        `final_dataset.csv` and does not carry `target_organism` -- that is an *activity*-level
        field (one peptide can be tested against many organisms), so the target-organism plot
        reads a second, separate source.
    :param output_dir: Where to write the PNG files.
    :param sample_size: Rows sampled for the two UMAP projections.
    :param activities_csv: Wide activity table carrying `target_organism`; the packaged release
        location when omitted. Missing the file skips only the target-organism plot, not the rest.
    """
    logger.info(f"Running Exploratory Data Analysis on {dataset_csv}...")
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset CSV not found at {dataset_csv}")
        
    df = pd.read_csv(dataset_csv)
    if "sequence" not in df.columns:
        logger.error("Dataset lacks 'sequence' column required for EDA.")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize label mapping for coloring
    if "is_decoy" in df.columns:
        df["Class"] = df["is_decoy"].map({0: "Active", 1: "Decoy"})
    else:
        df["Class"] = "Active"
        
    # Standardize sequences
    df["sequence"] = df["sequence"].apply(lambda x: str(x).upper().strip())
    df["sequence_length"] = df["sequence"].apply(len)
    
    # If charges aren't pre-computed, compute them
    if "charge" not in df.columns:
        df["charge"] = df["sequence"].apply(get_peptide_charge)
        
    logger.info("Generating distribution plots...")
    
    # Plot 1: Distribution of Length
    plt.figure(figsize=(10, 6))
    if "Class" in df.columns and df["Class"].nunique() > 1:
        sns.histplot(data=df, x="sequence_length", hue="Class", kde=True, 
                     palette=PALETTE, bins=30, alpha=0.6, element="step")
    else:
        sns.histplot(data=df, x="sequence_length", kde=True, color="#6366f1", bins=30, alpha=0.6)
    plt.title("Distribution of Peptide Sequence Lengths", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sequence Length (AA)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sequence_length_dist.png"), dpi=150, facecolor='#0b0f19')
    plt.close()
    
    # Plot 2: Net Charge Distribution
    plt.figure(figsize=(10, 6))
    if "Class" in df.columns and df["Class"].nunique() > 1:
        sns.kdeplot(data=df, x="charge", hue="Class", fill=True, palette=PALETTE, alpha=0.4, linewidth=2)
    else:
        sns.kdeplot(data=df, x="charge", fill=True, color="#6366f1", alpha=0.4, linewidth=2)
    plt.title("Net Charge Distribution at pH 7.0", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Net Charge", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "net_charge_dist.png"), dpi=150, facecolor='#0b0f19')
    plt.close()
    
    # Plot 3: Residue Frequencies
    logger.info("Computing residue frequencies...")
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Compute mean frequency per class
    residue_data = []
    for cls in df["Class"].unique():
        sub_df = df[df["Class"] == cls]
        # Count all occurrences
        all_seqs = "".join(sub_df["sequence"])
        total_len = len(all_seqs) if len(all_seqs) > 0 else 1
        for aa in amino_acids:
            freq = (all_seqs.count(aa) / total_len) * 100
            residue_data.append({
                "Residue": aa,
                "Frequency (%)": freq,
                "Class": cls
            })
            
    df_res = pd.DataFrame(residue_data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_res, x="Residue", y="Frequency (%)", hue="Class", palette=PALETTE, alpha=0.85)
    plt.title("Amino Acid Residue Frequencies Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Amino Acid Residue", fontsize=12)
    plt.ylabel("Percentage Frequency (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residue_composition.png"), dpi=150, facecolor='#0b0f19')
    plt.close()
    
    # Sample for dimensionality reductions to prevent OOM / CPU timeout
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    logger.info(f"Using a sampled subset of {len(sample_df)} sequences for UMAP projections.")
    
    # Plot 4: UMAP based on Morgan Fingerprints
    logger.info("Generating Morgan Fingerprints...")
    fingerprints = []
    labels_fp = []
    unparseable_smiles = 0
    fingerprint_failures = 0

    for idx, row in sample_df.iterrows():
        seq = row["sequence"]
        mol = Chem.MolFromSequence(seq)
        if not mol:
            # Try HELM or fallback smiles
            smiles = row.get("SMILES")
            if pd.notna(smiles) and smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                except Exception:
                    unparseable_smiles += 1
        if mol:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                arr = np.zeros((1,))
                Chem.DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr)
                labels_fp.append(row["Class"])
            except Exception:
                fingerprint_failures += 1

    if unparseable_smiles or fingerprint_failures:
        logger.warning(
            f"Morgan fingerprint plot: {unparseable_smiles} SMILES fallback(s) unparseable, "
            f"{fingerprint_failures} fingerprint computation failure(s), of {len(sample_df)} "
            f"sampled sequences.")

    if len(fingerprints) > 5:
        logger.info("Computing UMAP for Morgan Fingerprints...")
        X_fp = np.vstack(fingerprints)
        reducer_fp = umap.UMAP(n_neighbors=min(15, len(X_fp)-1), min_dist=0.1, random_state=42)
        umap_fp = reducer_fp.fit_transform(X_fp)
        
        plt.figure(figsize=(10, 8))
        df_umap_fp = pd.DataFrame(umap_fp, columns=["UMAP 1", "UMAP 2"])
        df_umap_fp["Class"] = labels_fp
        
        sns.scatterplot(data=df_umap_fp, x="UMAP 1", y="UMAP 2", hue="Class", palette=PALETTE, alpha=0.7, edgecolor='none', s=40)
        plt.title("UMAP Projection of Morgan Fingerprints", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_morgan.png"), dpi=150, facecolor='#0b0f19')
        plt.close()
        logger.info("Morgan Fingerprint UMAP generated.")
    else:
        logger.warning("Insufficient valid molecules found to generate Morgan Fingerprints UMAP.")
        
    # Plot 5: UMAP based on Peptide Embeddings (ESM-2)
    #
    # ESM-2 replaces ProtBERT here. ProtBERT ships only a `pytorch_model.bin`, and
    # transformers 5.x refuses to torch.load such a checkpoint below torch 2.6, while this
    # project pins torch 2.5.1+cu121 for the CUDA 12.2 driver. ESM-2 loads from safetensors,
    # needs no space-separated input, and is the stronger protein language model.
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Loading ESM-2 model from Hugging Face for peptide embeddings...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ESM-2 running on device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)
        try:
            model = AutoModel.from_pretrained(ESM2_MODEL).to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                logger.warning("CUDA Out of Memory when loading ESM-2 model. Falling back to CPU.")
                device = torch.device("cpu")
                model = AutoModel.from_pretrained(ESM2_MODEL).to(device)
            else:
                raise e

        model.eval()
        
        # ESM-2 takes the raw sequence; the space separation ProtBERT required is not used.
        sequences = sample_df["sequence"].tolist()
        processed_sequences = list(sequences)
        
        logger.info("Generating ESM-2 sequence embeddings in batches...")
        embeddings = []
        batch_size = 8
        
        try:
            with torch.no_grad():
                for i in range(0, len(processed_sequences), batch_size):
                    batch = processed_sequences[i:i+batch_size]
                    inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    
                    # Perform mean pooling over sequence length, respecting attention mask
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch_size, seq_len, 1)
                    masked_embeds = outputs.last_hidden_state * attention_mask
                    sum_embeds = torch.sum(masked_embeds, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    mean_embeds = sum_embeds / sum_mask
                    
                    embeddings.append(mean_embeds.cpu().numpy())
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                logger.warning("CUDA Out of Memory during inference. Falling back to CPU for embedding generation.")
                device = torch.device("cpu")
                model = model.to(device)
                embeddings = []
                with torch.no_grad():
                    for i in range(0, len(processed_sequences), batch_size):
                        batch = processed_sequences[i:i+batch_size]
                        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        
                        attention_mask = inputs["attention_mask"].unsqueeze(-1)
                        masked_embeds = outputs.last_hidden_state * attention_mask
                        sum_embeds = torch.sum(masked_embeds, dim=1)
                        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                        mean_embeds = sum_embeds / sum_mask
                        
                        embeddings.append(mean_embeds.cpu().numpy())
            else:
                raise e
                
        X_bert = np.vstack(embeddings)
        logger.info(f"ESM-2 embeddings calculated with shape: {X_bert.shape}")
        
        logger.info("Computing UMAP for ESM-2 Embeddings...")
        reducer_bert = umap.UMAP(n_neighbors=min(15, len(X_bert)-1), min_dist=0.1, random_state=42)
        umap_bert = reducer_bert.fit_transform(X_bert)
        
        plt.figure(figsize=(10, 8))
        df_umap_bert = pd.DataFrame(umap_bert, columns=["UMAP 1", "UMAP 2"])
        df_umap_bert["Class"] = sample_df["Class"].tolist()
        
        sns.scatterplot(data=df_umap_bert, x="UMAP 1", y="UMAP 2", hue="Class", palette=PALETTE, alpha=0.7, edgecolor='none', s=40)
        plt.title("UMAP Projection of ESM-2 Embeddings", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_esm2.png"), dpi=150, facecolor='#0b0f19')
        plt.close()
        logger.info("ESM-2 UMAP generated successfully.")
        
    except Exception as e:
        logger.error(f"Failed to generate ESM-2 embeddings or UMAP: {e}")
        # Write dummy placeholder image if failed to avoid UI breaking
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"ESM-2 UMAP Unavailable\n({str(e)})", ha="center", va="center", color="red", fontsize=14)
        plt.title("UMAP Projection of ESM-2 Embeddings", fontsize=14, fontweight="bold")
        plt.savefig(os.path.join(output_dir, "umap_esm2.png"), dpi=150, facecolor='#0b0f19')
        plt.close()

    # Plot 6: Target-organism distribution
    activities_csv = activities_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
        "sample_raw_release", "activities.csv")
    if os.path.exists(activities_csv):
        logger.info("Generating target-organism distribution...")
        try:
            organisms = pd.read_csv(activities_csv, usecols=["target_organism"],
                                    low_memory=False)["target_organism"].dropna()
            top = organisms.value_counts().head(20).sort_values()
            plt.figure(figsize=(10, 8))
            plt.barh(top.index, top.values, color="#6366f1", alpha=0.85)
            plt.title("Top 20 Target Organisms by Activity Count", fontsize=14,
                     fontweight="bold", pad=15)
            plt.xlabel("Number of Activities", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "target_organism_dist.png"), dpi=150,
                       facecolor='#0b0f19')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate target-organism distribution: {e}")
    else:
        logger.warning(f"No activity table at {activities_csv}; skipping the "
                       f"target-organism distribution plot.")

    logger.info("Exploratory Data Analysis completed successfully.")
    return True

def _resolve_prefixed(df: "pd.DataFrame", prefixes: Tuple[str, ...]) -> List[str]:
    """
    :param df: Frame to search.
    :param prefixes: Column-name prefixes defining one feature block.
    :returns: Matching column names, grouped by prefix in the order given and numerically sorted
        within each prefix, so ``esm2_dim_2`` precedes ``esm2_dim_10`` and the column order a
        fitted model saw is reproducible rather than lexicographic.
    """
    out: List[str] = []
    for prefix in prefixes:
        hits = [c for c in df.columns if c.startswith(prefix)]

        def _key(name: str) -> Tuple[int, str]:
            tail = name[len(prefix):]
            return (int(tail), "") if tail.isdigit() else (1 << 30, tail)

        out.extend(sorted(hits, key=_key))
    return out


def run_anomaly_detection(dataset_csv: str, output_csv: str, report_json: str,
                          contamination: Union[str, float] = "auto",
                          esm_components: int = 0) -> bool:
    """
    Fits an Isolation Forest to detect anomalous peptide records, over three named feature
    blocks: physicochemical descriptors, amino-acid composition spectra, and the ESM-2
    protein-language-model embedding.

    The block composition is written into the report, because it is not otherwise recoverable
    from the output: the embedding block is resolved by prefix, so a release built before the
    embeddings existed fits a silently narrower matrix and produces an `is_anomaly` column that
    looks identical. ``feature_blocks`` in the report says what the detector actually saw.

    One consequence worth knowing when reading the flags: an Isolation Forest picks the split
    feature uniformly at random, so a block's influence is its share of the column *count*, not
    its information content. With the 320 raw ESM-2 dimensions in, the embedding is ~90% of the
    matrix and the descriptors are close to decorative. ``esm_components`` exists to rebalance
    that -- see the parameter.

    :param contamination: Forwarded to :class:`~sklearn.ensemble.IsolationForest` unchanged.
        ``"auto"`` (scikit-learn's own default) rather than a fixed fraction, so the flagged
        proportion is whatever the fitted decision-function threshold implies for this data,
        not a number chosen in advance.
    :param esm_components: When > 0, the embedding block is standardized and reduced to this
        many principal components before fitting, so it stops outvoting the other two blocks by
        sheer column count. ``0`` (the default) fits the raw dimensions, which is what every
        release to date was built with.
    :returns: ``True`` on success; ``False`` if no feature block could be resolved.
    """
    logger.info(f"Running Unsupervised Anomaly Detection (Isolation Forest) on {dataset_csv}...")
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset CSV not found at {dataset_csv}")

    df = pd.read_csv(dataset_csv)

    descriptor_cols = [c for c in ANOMALY_DESCRIPTOR_COLUMNS if c in df.columns]
    missing_descriptors = [c for c in ANOMALY_DESCRIPTOR_COLUMNS if c not in df.columns]
    composition_cols = _resolve_prefixed(df, ANOMALY_COMPOSITION_PREFIXES)
    embedding_cols = _resolve_prefixed(df, ANOMALY_EMBEDDING_PREFIXES)

    if missing_descriptors:
        logger.warning("%d of %d physicochemical descriptors are absent and will not be fitted: "
                       "%s", len(missing_descriptors), len(ANOMALY_DESCRIPTOR_COLUMNS),
                       ", ".join(missing_descriptors))
    if not embedding_cols:
        # Loudly, because the block is resolved by prefix: absent embeddings do not raise, they
        # just quietly shrink the matrix, and the resulting `is_anomaly` column is
        # indistinguishable from one fitted with them. Run step 28 before this step.
        logger.warning("No ESM-2 embedding columns (%s) in %s -- fitting on %d descriptor and "
                       "%d composition features ONLY. The flags will not reflect "
                       "language-model sequence structure.",
                       "/".join(ANOMALY_EMBEDDING_PREFIXES), dataset_csv,
                       len(descriptor_cols), len(composition_cols))

    if not (descriptor_cols or composition_cols or embedding_cols):
        logger.error("No valid features found for anomaly detection.")
        return False

    # fillna(0.0) is load-bearing and dangerous on the embedding block: a row of 320 zeros is
    # itself a strong outlier signal, so a peptide whose embedding failed would be flagged for
    # the failure rather than for its sequence. Counted and reported per block, never silent.
    fills: Dict[str, int] = {}
    frames: List["pd.DataFrame"] = []
    blocks: Dict[str, Dict[str, Any]] = {}

    for name, cols in (("physicochemical", descriptor_cols),
                       ("composition", composition_cols)):
        if not cols:
            continue
        part = df[cols].apply(pd.to_numeric, errors="coerce")
        fills[name] = int(part.isna().to_numpy().sum())
        frames.append(part.fillna(0.0))
        blocks[name] = {"n_features": len(cols), "columns": list(cols),
                        "values_filled": fills[name]}

    embedding_variance: Optional[float] = None
    if embedding_cols:
        part = df[embedding_cols].apply(pd.to_numeric, errors="coerce")
        fills["embedding"] = int(part.isna().to_numpy().sum())
        part = part.fillna(0.0)
        blocks["embedding"] = {"n_source_columns": len(embedding_cols),
                               "prefixes": list(ANOMALY_EMBEDDING_PREFIXES),
                               "values_filled": fills["embedding"],
                               "pca_components": None,
                               "explained_variance_ratio": None}
        if esm_components > 0:
            k = min(esm_components, len(embedding_cols), len(df))
            scaled = StandardScaler().fit_transform(part.to_numpy(dtype="float64"))
            pca = PCA(n_components=k, random_state=42)
            reduced = pca.fit_transform(scaled)
            embedding_variance = float(pca.explained_variance_ratio_.sum())
            part = pd.DataFrame(reduced, index=df.index,
                                columns=[f"esm_pc_{i}" for i in range(k)])
            blocks["embedding"]["pca_components"] = k
            blocks["embedding"]["explained_variance_ratio"] = embedding_variance
            logger.info("Reduced %d embedding dimensions to %d principal components "
                        "(%.1f%% of embedding variance retained).",
                        len(embedding_cols), k, embedding_variance * 100.0)
        blocks["embedding"]["n_features"] = part.shape[1]
        frames.append(part)

    X = pd.concat(frames, axis=1)
    total_filled = sum(fills.values())
    if total_filled:
        logger.warning("Filled %d non-numeric/missing feature values with 0.0 across blocks: %s",
                       total_filled, ", ".join(f"{k}={v}" for k, v in fills.items() if v))

    composition = ", ".join(f"{n}={b['n_features']}" for n, b in blocks.items())
    emb_features = blocks.get("embedding", {}).get("n_features", 0)
    emb_share = (emb_features / X.shape[1]) if X.shape[1] else 0.0
    logger.info("Fitting Isolation Forest on %d samples and %d features (%s); the embedding "
                "block is %.1f%% of the matrix and therefore of the random split choices.",
                X.shape[0], X.shape[1], composition, emb_share * 100.0)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X)  # 1 for inlier, -1 for outlier/anomaly
    scores = iso.decision_function(X)
    
    # Add anomaly columns to DataFrame
    df["anomaly_score"] = scores
    df["is_anomaly"] = (preds == -1).astype(int)
    
    # Write updated dataset back
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved updated dataset with anomaly labels to {output_csv}")
    
    # Extract details of anomalies
    anomalies_df = df[df["is_anomaly"] == 1]
    inliers_df = df[df["is_anomaly"] == 0]
    
    total = len(df)
    num_anomalies = len(anomalies_df)
    pct_anomalies = (num_anomalies / total) * 100
    
    # Compute stats for normal vs anomalous
    summary_stats = {}
    for col in ["sequence_length", "charge", "molecular_weight", "logp"]:
        if col in df.columns:
            summary_stats[col] = {
                "inliers_mean": float(inliers_df[col].mean()),
                "anomalies_mean": float(anomalies_df[col].mean())
            }
            
    # Sample top anomalous examples (lowest anomaly score)
    top_anomalies = []
    if "sequence" in df.columns:
        anom_sorted = anomalies_df.sort_values(by="anomaly_score")
        for idx, row in anom_sorted.head(10).iterrows():
            top_anomalies.append({
                "entry_id": str(row.get("entry_id", "N/A")),
                "sequence": str(row.get("sequence", "")),
                "sequence_length": int(row.get("sequence_length", 0)),
                "charge": float(row.get("charge", 0.0)),
                "anomaly_score": float(row.get("anomaly_score", 0.0)),
                "is_decoy": int(row.get("is_decoy", 0))
            })
            
    report = {
        "total_records": total,
        "num_anomalies": num_anomalies,
        "percentage_anomalies": pct_anomalies,
        "contamination_rate": contamination,
        # What the detector actually fitted. Recorded because the embedding block is resolved by
        # prefix: without this, a run that silently missed it is indistinguishable from one that
        # did not. `embedding_feature_share` is also the embedding's share of the random split
        # choices an Isolation Forest makes, which is how much it influenced the flags.
        "n_features_total": int(X.shape[1]),
        "feature_blocks": blocks,
        "embedding_feature_share": emb_share,
        "embedding_used": bool(embedding_cols),
        "feature_values_filled": fills,
        "summary_features": summary_stats,
        "top_anomalies": top_anomalies
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(report_json)), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Anomaly detection report saved to {report_json}. Detected {num_anomalies} anomalies ({pct_anomalies:.2f}%).")
    return True

def compare_chemical_spaces(dataset_csv: str, out_dir: str) -> None:
    """
    Compares the chemical space of the extracted peptides against the database baseline.

    Projects molecular descriptors with UMAP and writes the comparison figures, which is how a
    reviewer sees whether newly extracted peptides occupy the same region as known AMPs or a
    distinct one.

    :param dataset_csv: Final dataset CSV carrying descriptors and a source label.
    :param out_dir: Directory for the generated figures.
    :returns: ``None``; figures are written to ``out_dir``.
    """
    logger.info(f"Comparing chemical spaces from {dataset_csv}...")
    if not os.path.exists(dataset_csv):
        logger.error(f"Dataset not found at {dataset_csv}")
        return False
        
    df = pd.read_csv(dataset_csv)
    
    # Identify sources
    db_cols = ['ADP', 'CAMP', 'DBAMP', 'DRAMP', 'DBAASP']
    
    sources = []
    for _, row in df.iterrows():
        is_db = any(pd.notna(row.get(c)) for c in db_cols if c in df.columns)
        is_lit = pd.notna(row.get('LITERATURE')) if 'LITERATURE' in df.columns else False
        is_pat = pd.notna(row.get('PATENT')) if 'PATENT' in df.columns else False
        
        if is_db:
            sources.append('Database')
        elif is_lit:
            sources.append('Literature')
        elif is_pat:
            sources.append('Patent')
        else:
            sources.append('Unknown')
            
    df['Source_Category'] = sources
    
    # Filter out unknown/decoy if needed (we only want actual extracted/curated AMPs)
    df = df[(df['Source_Category'] != 'Unknown') & (df['is_decoy'] == 0)].copy()
    
    if df.empty:
        logger.error("No valid sources found for comparison.")
        return False
        
    feature_cols = [
        "molecular_weight", "logp", "hydrogen_bond_donors", 
        "hydrogen_bond_acceptors", "charge", "sequence_length"
    ]
    emb_cols = [c for c in df.columns if c.startswith("aa_freq_") or c.startswith("aa_group_freq_") or c.startswith("esm2_dim_") or c.startswith("protbert_dim_")]
    all_features = feature_cols + emb_cols
    
    # Filter available
    all_features = [c for c in all_features if c in df.columns]
    X = df[all_features].copy().fillna(0.0)
    
    logger.info(f"Computing UMAP embedding on {X.shape[0]} samples and {X.shape[1]} features...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    embedding = reducer.fit_transform(X)
    
    df['UMAP_1'] = embedding[:, 0]
    df['UMAP_2'] = embedding[:, 1]
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Kolmogorov-Smirnov Test
    from scipy.stats import ks_2samp
    import json
    
    logger.info("Computing Kolmogorov-Smirnov tests for feature distributions...")
    ks_results = {}
    sources_unique = df['Source_Category'].unique()
    for feat in feature_cols:
        if feat not in df.columns: continue
        ks_results[feat] = {}
        for i in range(len(sources_unique)):
            for j in range(i+1, len(sources_unique)):
                s1, s2 = sources_unique[i], sources_unique[j]
                dist1 = df[df['Source_Category'] == s1][feat].dropna()
                dist2 = df[df['Source_Category'] == s2][feat].dropna()
                if len(dist1) > 0 and len(dist2) > 0:
                    stat, pval = ks_2samp(dist1, dist2)
                    ks_results[feat][f"{s1}_vs_{s2}"] = {"statistic": stat, "p_value": pval}
                    
    with open(os.path.join(out_dir, "ks_test_results.json"), "w") as f:
        json.dump(ks_results, f, indent=4)
        
    # HDBSCAN Clustering
    logger.info("Running HDBSCAN clustering based on base features...")
    try:
        from sklearn.cluster import HDBSCAN
        hdbscan_clusterer = HDBSCAN(min_cluster_size=15)
    except ImportError:
        try:
            import hdbscan
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
        except ImportError:
            logger.warning("HDBSCAN not available. Falling back to KMeans for demonstration.")
            from sklearn.cluster import KMeans
            hdbscan_clusterer = KMeans(n_clusters=5, random_state=42)
            
    from sklearn.preprocessing import StandardScaler
    X_base = df[[c for c in feature_cols if c in df.columns]].copy().fillna(0.0)
    X_base_scaled = StandardScaler().fit_transform(X_base)
    try:
        df['HDBSCAN_Cluster'] = hdbscan_clusterer.fit_predict(X_base_scaled)
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        df['HDBSCAN_Cluster'] = -1
    
    with plt.style.context("default"):
        # Plot 1: All sources overlaid
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='Source_Category', palette='Set2', s=30, alpha=0.7)
        plt.title("Chemical Space Comparison (UMAP) - All Sources", color='black')
        plt.legend(title='Source')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chemical_space_all.png"), dpi=300)
        plt.close()
        
        # Separate plots for each category
        categories = df['Source_Category'].unique()
        palette = sns.color_palette("Set2", n_colors=len(categories))
        cat_color = dict(zip(categories, palette))
        
        for cat in categories:
            plt.figure(figsize=(10, 8))
            df_cat = df[df['Source_Category'] == cat]
            sns.scatterplot(data=df_cat, x='UMAP_1', y='UMAP_2', color=cat_color[cat], s=30, alpha=0.8)
            plt.title(f"Chemical Space - {cat} only", color='black')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"chemical_space_{cat.lower()}.png"), dpi=300)
            plt.close()
            
        # Plot HDBSCAN Clusters
        plt.figure(figsize=(10, 8))
        # Ensure cluster is categorical for discrete coloring
        df['Cluster_Label'] = df['HDBSCAN_Cluster'].astype(str)
        sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='Cluster_Label', palette='tab20', s=30, alpha=0.7)
        plt.title("Chemical Space Comparison (UMAP) - Feature Clusters", color='black')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chemical_space_clusters.png"), dpi=300)
        plt.close()
        
    logger.info(f"Chemical space plots and KS test results saved to {out_dir}")
    return True
