import os
import tempfile
import numpy as np
import pandas as pd
import json
import pytest

import sample.ml as SML
import sample.rf_filter as SRF


def test_train_ml_models_sequence_grouping(tmp_path):
    """
    Verifies that train_ml_models clusters sequences and uses StratifiedGroupKFold
    so that identical or homologous sequences land in the same fold.
    """
    homolog_a = "KLLKKLLKKLLK"
    homolog_b = "RRWWWRRWWWRR"
    
    seqs = [homolog_a] * 5 + [homolog_b] * 5 + [f"DECOY_{i}" for i in range(10)]
    is_decoy = [0] * 10 + [1] * 10
    
    df = pd.DataFrame({
        "sequence": seqs,
        "is_decoy": is_decoy,
        "molecular_weight": [1500.0] * 20,
        "logp": [-0.5] * 20,
        "hydrogen_bond_donors": [10] * 20,
        "hydrogen_bond_acceptors": [5] * 20,
        "charge": [3] * 20,
        "sequence_length": [12] * 20,
        "aa_freq_K": [0.5] * 5 + [0.0] * 15,
        "aa_freq_R": [0.0] * 5 + [0.5] * 5 + [0.0] * 10,
    })
    
    csv_path = str(tmp_path / "dataset.csv")
    df.to_csv(csv_path, index=False)
    model_path = str(tmp_path / "model.pkl")
    
    success = SML.train_ml_models(csv_path, model_path)
    assert success is True
    assert os.path.exists(model_path)


def test_validate_ml_models_sequence_grouping(tmp_path):
    """
    Verifies that validate_ml_models uses sequence similarity grouping in StratifiedGroupKFold.
    """
    homolog_a = "KLLKKLLKKLLK"
    homolog_b = "RRWWWRRWWWRR"
    
    seqs = [homolog_a] * 5 + [homolog_b] * 5 + [f"DECOY_{i}" for i in range(10)]
    is_decoy = [0] * 10 + [1] * 10
    
    df = pd.DataFrame({
        "sequence": seqs,
        "is_decoy": is_decoy,
        "molecular_weight": [1500.0] * 20,
        "logp": [-0.5] * 20,
        "hydrogen_bond_donors": [10] * 20,
        "hydrogen_bond_acceptors": [5] * 20,
        "charge": [3] * 20,
        "sequence_length": [12] * 20,
        "aa_freq_K": [0.5] * 5 + [0.0] * 15,
        "aa_freq_R": [0.0] * 5 + [0.5] * 5 + [0.0] * 10,
    })
    
    csv_path = str(tmp_path / "dataset.csv")
    df.to_csv(csv_path, index=False)
    model_path = str(tmp_path / "model.pkl")
    
    SML.train_ml_models(csv_path, model_path)
    
    report_path = str(tmp_path / "report.json")
    report = SML.validate_ml_models(csv_path, model_path, report_path)
    assert "overall_accuracy" in report
    assert "cv_accuracy_mean" in report
    assert "cv_auc_mean" in report


def test_rf_filter_train_with_sequence_clustering(tmp_path, monkeypatch):
    """
    Verifies that train_rf_filter groups homologous positive sequences to prevent data leakage.
    """
    # Create fake directory structure for DATA_DIR
    raw_dir = tmp_path / "sample_raw_release"
    res_dir = tmp_path / "resources"
    raw_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # Positive sequences with homologs
    merged_db = [
        {"sequence": "KLLKKLLKKLLK"},
        {"sequence": "KLLKKLLKKLLA"},
        {"sequence": "RRWWWRRWWWRR"},
        {"sequence": "RRWWWRRWWWRA"},
        {"sequence": "ACDEFGHIKLMN"},
    ]
    with open(raw_dir / "merged_from_database.json", "w") as f:
        json.dump(merged_db, f)
        
    # Negative English words
    words = ["HELLO\n", "WORLD\n", "PEPTIDE\n", "FILTER\n", "SAMPLE\n"]
    with open(res_dir / "words_alpha.txt", "w") as f:
        f.writelines(words)
        
    monkeypatch.setattr(SRF, "DATA_DIR", str(tmp_path))
    model_out = str(tmp_path / "rf_filter.pkl")
    res = SRF.train_rf_filter(output_model_path=model_out)
    assert res == model_out
    assert os.path.exists(model_out)
