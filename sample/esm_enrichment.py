import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("sample.esm_enrichment")

_ESM_TOKENIZER = None
_ESM_MODEL = None
_DEVICE = None

def get_esm_device() -> torch.device:
    """
    :returns: The device ESM-2 should run on -- CUDA when available, otherwise CPU.
    """
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE

def get_esm_model() -> tuple:
    """
    Loads the ESM-2 model and tokenizer once and caches them for the process.

    Cached because the weights are large and every worker thread shares one instance; reloading
    per call would dominate runtime and multiply memory.

    :returns: ``(model, tokenizer)``, both already moved to :func:`get_esm_device`.
    """
    global _ESM_TOKENIZER, _ESM_MODEL
    if _ESM_MODEL is None:
        try:
            from transformers import AutoTokenizer, EsmModel
            device = get_esm_device()
            model_name = "facebook/esm2_t6_8M_UR50D"
            logger.info(f"Loading ESM-2 Protein Language Model '{model_name}' on {device}...")
            _ESM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _ESM_MODEL = EsmModel.from_pretrained(model_name).to(device)
            _ESM_MODEL.eval()
        except Exception as e:
            logger.warning(f"Failed to load ESM-2 model: {e}. ESM-2 embeddings disabled.")
            _ESM_MODEL = False
            _ESM_TOKENIZER = False
    return _ESM_TOKENIZER, _ESM_MODEL

def compute_esm2_embeddings(sequences: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Computes 320-dimensional ESM-2 protein language model contextual embeddings
    for a batch of amino acid sequences using Meta's esm2_t6_8M_UR50D model.
    """
    if not sequences:
        return None

    tokenizer, model = get_esm_model()
    if not model or not tokenizer:
        return None

    device = get_esm_device()
    all_embeddings = []

    clean_sequences = [s.strip().upper() for s in sequences]

    try:
        for i in range(0, len(clean_sequences), batch_size):
            batch_seqs = clean_sequences[i:i + batch_size]
            inputs = tokenizer(batch_seqs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean-pooling over token representations (excluding CLS/SEP tokens)
                seq_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(seq_embeddings)

        return np.vstack(all_embeddings)

    except Exception as e:
        logger.error(f"Error computing ESM-2 embeddings: {e}")
        return None
