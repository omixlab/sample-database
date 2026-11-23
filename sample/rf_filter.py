import json
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
from sample.config import DATA_DIR
import logging

from typing import Any, List, Optional, Sequence

logger = logging.getLogger("sample.rf_filter")

def extract_features(seq: str) -> List[float]:
    """
    Turns a sequence into the 27 features the filter is trained on.

    The feature vector is the sequence length followed by the relative frequency of each of the
    26 uppercase letters. Frequencies rather than counts, so that a short and a long sequence with
    the same composition land in the same region of the space.

    :param seq: Amino-acid sequence. Case, spaces and hyphens are normalized away.
    :returns: 27 floats -- ``[length, freq_A, freq_B, ..., freq_Z]``. An empty sequence yields
        27 zeros rather than raising, so a caller can score a batch without pre-filtering it.
    """
    seq = seq.upper().replace("-", "").replace(" ", "")
    length = len(seq)
    if length == 0:
        return [0] * 27
    
    features = [length]
    counts = {chr(i): 0 for i in range(65, 91)}
    for char in seq:
        if char in counts:
            counts[char] += 1
            
    for char in sorted(counts.keys()):
        features.append(counts[char] / length)
        
    return features

def train_rf_filter(output_model_path: Optional[str] = None) -> Optional[str]:
    """
    Trains the random-forest filter that separates real peptide sequences from English words.

    Positives come from the merged peptide database, negatives from an English word list; the
    discriminator matters because extraction picks up capitalized words that happen to spell valid
    one-letter amino-acid codes.

    :param output_model_path: Where to write the pickled classifier. Defaults to
        ``<DATA_DIR>/rf_filter_model.pkl``.
    :returns: The path written, or ``None`` if training data was unavailable.
    """
    if not output_model_path:
        output_model_path = os.path.join(DATA_DIR, "rf_filter_model.pkl")
        
    print("Loading positive examples...")
    merged_db_path = os.path.join(DATA_DIR, "sample_raw_release", "merged_from_database.json")
    with open(merged_db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    positives = set()
    for entry in data:
        seq = entry.get('sequence', '')
        if seq:
            seq = seq.strip().upper().replace("-", "").replace(" ", "")
            if seq.isalpha() and len(seq) >= 3:
                positives.add(seq)
                
    import re
    AA1 = r'\[?(?:[DLdl]-?)?[ACDEFGHIKLMNPQRSTVWY]\]?'
    SEQ_EXACT = re.compile(rf'\b(?:[Nn][Hh]2-|[Aa]c-)?((?:{AA1}-?){{4,}}{AA1})(?:-[Nn][Hh]2|-[Cc][Oo][Oo][Hh])?\b')
    SEQ_ALPHANUM = re.compile(r'\b(?:[A-Z]{2,6}|[a-z][A-Z]{1,5})(?:-?\d{1,4}|-alpha|-beta|-gamma)\b')
    
    
    print("Loading negative examples (hard negatives matching NLP patterns)...")
    negatives = set()
    
    # 1. Parse UNIX dictionary
    with open("/usr/share/dict/words", "r", encoding="utf-8") as f:
        for line in f:
            raw_word = line.strip()
            test_word = raw_word.upper()
            
            if SEQ_EXACT.search(test_word) or SEQ_ALPHANUM.search(test_word):
                word = test_word.replace("-", "").replace(" ", "").replace("'", "")
                if word.isalpha() and len(word) >= 3:
                    negatives.add(word)
                    
    # 2. Parse Surnames/Family Names
    import urllib.request
    surname_path = os.path.join(DATA_DIR, "surnames.txt")
    if not os.path.exists(surname_path):
        print("Downloading surnames dataset...")
        try:
            url = "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/surnames/us.txt"
            urllib.request.urlretrieve(url, surname_path)
        except Exception as e:
            print(f"Failed to download surnames: {e}")
            
    if os.path.exists(surname_path):
        print("Adding peptide-like surnames to negative dataset...")
        with open(surname_path, "r", encoding="utf-8") as f:
            for line in f:
                raw_word = line.strip()
                test_word = raw_word.upper()
                
                # Treat surnames just like dictionary words: if they look like peptides, they are hard negatives!
                if SEQ_EXACT.search(test_word) or SEQ_ALPHANUM.search(test_word):
                    word = test_word.replace("-", "").replace(" ", "").replace("'", "")
                    if word.isalpha() and len(word) >= 3:
                        negatives.add(word)
                
    overlap = negatives.intersection(positives)
    if overlap:
        print(f"Removing {len(overlap)} overlapping sequences from negatives.")
    negatives = negatives - positives
    print(f"Found {len(negatives)} hard negative examples.")
    
    pos_list = list(positives)
    neg_list = list(negatives)
    all_seqs = pos_list + neg_list
    X = np.array([extract_features(seq) for seq in all_seqs])
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    
    # Cluster positive sequences by homology (40% identity) to prevent data leakage into test set
    from sample.families import cdhit_available, cluster_cdhit, cluster as near_identity_cluster
    pos_dict = {f"p_{i}": s for i, s in enumerate(pos_list)}
    if cdhit_available() and len(pos_dict) > 1:
        pos_clusters = cluster_cdhit(pos_dict, identity=0.4)
    elif len(pos_dict) > 1:
        try:
            pos_clusters = near_identity_cluster(pos_dict)
        except Exception:
            pos_clusters = {k: i for i, k in enumerate(pos_dict)}
    else:
        pos_clusters = {k: 0 for k in pos_dict}
        
    pos_groups = [pos_clusters.get(f"p_{i}", i) for i in range(len(pos_list))]
    neg_groups = [len(pos_list) + i for i in range(len(neg_list))]
    groups = np.array(pos_groups + neg_groups)
    
    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = next(sgkf.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    with open(output_model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"Model trained and saved to {output_model_path}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    return output_model_path

#: Process-wide cache of the unpickled classifier, so worker threads share one instance.
_loaded_rf_clf: Optional[Any] = None

def filter_sequences(seqs: Sequence[str], model_path: Optional[str] = None) -> List[bool]:
    """
    Scores a batch of sequences, keeping the loaded model in a module-level cache.

    Batched because the per-call overhead of a scikit-learn ``predict`` dominates for single
    sequences, and extraction scores many candidates per document.

    :param seqs: Sequences to score.
    :param model_path: Pickled classifier. Defaults to ``<DATA_DIR>/rf_filter_model.pkl``.
    :returns: One bool per input, ``True`` meaning "looks like a real peptide". If the model file
        does not exist every sequence is accepted -- an untrained filter must not silently delete
        the corpus.
    """
    global _loaded_rf_clf
    
    if not model_path:
        model_path = os.path.join(DATA_DIR, "rf_filter_model.pkl")
        
    if not os.path.exists(model_path):
        return [True] * len(seqs) # Default to True if model not trained
        
    if _loaded_rf_clf is None:
        with open(model_path, 'rb') as f:
            _loaded_rf_clf = pickle.load(f)
    
    if not seqs:
        return []
        
    feats = np.array([extract_features(seq) for seq in seqs])
    preds = _loaded_rf_clf.predict(feats)
    return [bool(p == 1) for p in preds]

def filter_sequence(seq: str, model_path: Optional[str] = None) -> bool:
    """
    Single-sequence convenience wrapper around :func:`filter_sequences`.

    :param seq: Sequence to score.
    :param model_path: Pickled classifier. Defaults to ``<DATA_DIR>/rf_filter_model.pkl``.
    :returns: ``True`` if the sequence looks like a real peptide.
    """
    return filter_sequences([seq], model_path)[0]
