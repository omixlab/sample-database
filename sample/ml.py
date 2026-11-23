import os
import pickle
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, GroupShuffleSplit, GroupKFold, StratifiedGroupKFold
)
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("sample.ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_ml_models(dataset_csv: str, model_out_path: str, filter_anomalies: bool = False) -> bool:
    """
    Trains a Random Forest classifier to predict antimicrobial peptide activity
    based on molecular descriptors and frequency embeddings.
    """
    logger.info(f"Training machine learning model from dataset {dataset_csv}...")
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Assembled dataset not found at {dataset_csv}")
        
    df = pd.read_csv(dataset_csv)
    
    # Drop rows without labels
    if "is_decoy" not in df.columns:
        logger.error("Dataset lacks 'is_decoy' binary label column.")
        return False
        
    df = df.dropna(subset=["is_decoy"])
    
    # Filter anomalies if requested and present
    if filter_anomalies and "is_anomaly" in df.columns:
        before_count = len(df)
        df = df[df["is_anomaly"] == 0]
        logger.info(f"Filtered out {before_count - len(df)} anomalies before training.")
    
    # Positive class: is_decoy = 0 (means it's a real antimicrobial peptide!)
    # Negative class: is_decoy = 1 (means decoy)
    # Let's map target: 1 = Active (AMP), 0 = Inactive (Decoy)
    y = (df["is_decoy"] == 0).astype(int)
    
    # Identify feature columns
    feature_cols = [
        "molecular_weight", "logp", "hydrogen_bond_donors", 
        "hydrogen_bond_acceptors", "charge", "sequence_length"
    ]
    
    # Add frequency embedding columns if present
    emb_cols = [c for c in df.columns if c.startswith("aa_freq_") or c.startswith("aa_group_freq_")]
    # Add ProtBERT embeddings if present
    pb_cols = [c for c in df.columns if c.startswith("protbert_dim_")]
    # Add ESM-2 embeddings if present
    esm2_cols = [c for c in df.columns if c.startswith("esm2_dim_")]
    
    feature_cols.extend(emb_cols)
    feature_cols.extend(pb_cols)
    feature_cols.extend(esm2_cols)
    
    # Filter dataset
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(0.0)
    
    if X.empty:
        logger.error("No valid features found for training.")
        return False
        
    logger.info(f"Training features shape: {X.shape}. Target shape: {y.shape}")
    
    # Train/Test Split: Use sequence similarity clustering (CD-HIT 40% identity) to prevent homology data leakage
    if "sequence" in df.columns:
        from sample.ml_models import _cluster_groups
        groups = _cluster_groups(df, identity=0.4)
    elif len(emb_cols) > 0:
        n_clusters = max(2, len(X) // 20)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        groups = kmeans.fit_predict(X[emb_cols])
    else:
        groups = np.arange(len(X))
        
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y, groups))
    except ValueError:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
        )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Initialize Random Forest inside a scaling pipeline
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    logger.info(f"Model trained successfully. Test Accuracy: {acc:.4f}, Test ROC-AUC: {auc:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(os.path.abspath(model_out_path)), exist_ok=True)
    with open(model_out_path, 'wb') as f:
        pickle.dump({
            "model": rf,
            "feature_columns": feature_cols,
            "metrics": {
                "test_accuracy": acc,
                "test_auc": auc
            }
        }, f)
        
    logger.info(f"Model saved to {model_out_path}")
    return True

def validate_ml_models(dataset_csv: str, model_path: str, report_out_path: str, filter_anomalies: bool = False) -> dict:
    """
    Validates model using cross-validation and generates a detailed report.
    """
    logger.info(f"Validating model {model_path} against dataset {dataset_csv}...")
    if not os.path.exists(dataset_csv) or not os.path.exists(model_path):
        raise FileNotFoundError("Dataset or model file missing.")
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data["model"]
    feature_cols = model_data["feature_columns"]
    
    df = pd.read_csv(dataset_csv).dropna(subset=["is_decoy"])
    
    # Filter anomalies if requested and present
    if filter_anomalies and "is_anomaly" in df.columns:
        before_count = len(df)
        df = df[df["is_anomaly"] == 0]
        logger.info(f"Filtered out {before_count - len(df)} anomalies before validation.")
        
    y = (df["is_decoy"] == 0).astype(int)
    X = df[feature_cols].copy().fillna(0.0)
    
    if "sequence" in df.columns:
        from sample.ml_models import _cluster_groups
        groups = _cluster_groups(df, identity=0.4)
    elif len(emb_cols) > 0:
        n_clusters = max(5, len(X) // 20)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        groups = kmeans.fit_predict(X[emb_cols])
    else:
        groups = np.arange(len(X))
        
    try:
        cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        next(cv_splitter.split(X, y, groups))
        cv_groups = groups
    except ValueError:
        cv_splitter = 5
        cv_groups = None
        
    # Run 5-fold cross-validation grouped by sequence similarity
    cv_scores = cross_val_score(model, X, y, cv=cv_splitter, groups=cv_groups, scoring='accuracy')
    cv_auc = cross_val_score(model, X, y, cv=cv_splitter, groups=cv_groups, scoring='roc_auc')
    
    # Run full set predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary')
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)
    
    report = {
        "overall_accuracy": acc,
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1_score": f1,
        "overall_auc_score": auc,
        "cv_accuracy_scores": list(cv_scores),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_auc_scores": list(cv_auc),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std())
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(report_out_path)), exist_ok=True)
    with open(report_out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Validation report saved to {report_out_path}. CV Accuracy Mean: {report['cv_accuracy_mean']:.4f}")
    return report
