"""
Antibiotic name -> Morgan (ECFP) fingerprint, for the peptide-antibiotic synergy models.

The synergy data names its partner antibiotic (Gentamicin, Vancomycin, ...) but carries no
structure. A one-hot indicator per antibiotic -- what :func:`sample.synergy.featurise` used before
-- cannot generalize: a drug the model never saw in training just lands in an "other" column and
the model has nothing to reason with. A Morgan/ECFP fingerprint of the drug's actual structure lets
the model reason by chemical similarity, so a new fluoroquinolone is scored like the
fluoroquinolones it has seen. That is what the peptide-plus-antibiotic-fingerprint input is for.

SMILES are resolved once from PubChem's PUG REST (name -> ConnectivitySMILES) and cached to
``data/antibiotics/smiles_cache.json``, so a training run does not re-hit the API per row and an
offline run reuses the cache. A name PubChem cannot resolve, or a run with no network and no cached
entry, yields an all-zero fingerprint -- the honest "unknown structure" state, distinct from a real
fingerprint, exactly as the old "other" column was but now without pretending unrelated drugs are
the same.
"""
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger("sample.antibiotics")

#: PubChem PUG REST base. name -> ConnectivitySMILES is a single GET.
PUBCHEM_BASE: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

#: ECFP length. 256 bits keeps the synergy feature matrix small next to the peptide descriptors
#: while still separating the antibiotic classes present here.
FP_BITS: int = 256

#: ECFP radius. 2 is ECFP4, the usual default for drug-like small molecules.
FP_RADIUS: int = 2


def _cache_path() -> str:
    """:returns: The antibiotic SMILES cache file; ``SAMPLE_ANTIBIOTIC_CACHE`` overrides it."""
    override = os.environ.get("SAMPLE_ANTIBIOTIC_CACHE")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "antibiotics", "smiles_cache.json")


def _load_cache() -> Dict[str, Optional[str]]:
    """:returns: The name -> SMILES cache (a cached ``None`` means "PubChem had no match")."""
    path = _cache_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_cache(cache: Dict[str, Optional[str]]) -> None:
    """:param cache: The name -> SMILES cache to persist."""
    path = _cache_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)


def resolve_smiles(name: str, timeout: float = 15.0) -> Optional[str]:
    """
    Resolves an antibiotic name to a SMILES string via PubChem.

    :param name: The antibiotic's name, as it appears in the synergy data.
    :param timeout: Per-request timeout.
    :returns: The SMILES, or ``None`` when PubChem has no match or the request fails -- the caller
        treats that as an unknown structure rather than raising.
    """
    name = str(name or "").strip()
    if not name:
        return None
    url = (f"{PUBCHEM_BASE}/compound/name/{urllib.parse.quote(name)}"
           f"/property/ConnectivitySMILES,CanonicalSMILES/JSON")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
    except Exception as e:
        logger.info(f"PubChem lookup failed for {name!r}: {e}")
        return None
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None
    # PubChem renamed CanonicalSMILES -> ConnectivitySMILES; accept either.
    return props[0].get("ConnectivitySMILES") or props[0].get("CanonicalSMILES")


def morgan_fingerprint(smiles: str, n_bits: int = FP_BITS,
                       radius: int = FP_RADIUS) -> Optional[np.ndarray]:
    """
    :param smiles: A SMILES string.
    :param n_bits: Fingerprint length.
    :param radius: Morgan radius (2 = ECFP4).
    :returns: The ECFP as a 0/1 float array of length `n_bits`, or ``None`` when RDKit cannot parse
        the SMILES.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as e:
        logger.warning(f"RDKit unavailable, cannot fingerprint antibiotics: {e}")
        return None
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vect, array)
    return array


def fingerprint_columns(n_bits: int = FP_BITS) -> List[str]:
    """:returns: The column names for an antibiotic fingerprint block, ``abx_fp_0..n``."""
    return [f"abx_fp_{i}" for i in range(n_bits)]


def build_fingerprints(names: Iterable[str], n_bits: int = FP_BITS, radius: int = FP_RADIUS,
                       use_network: bool = True) -> Dict[str, np.ndarray]:
    """
    Builds a Morgan fingerprint per distinct antibiotic name, resolving SMILES through PubChem
    (cached) and returning a zero vector for anything that cannot be resolved or parsed.

    :param names: Antibiotic names (duplicates collapsed).
    :param n_bits: Fingerprint length.
    :param radius: Morgan radius.
    :param use_network: When ``False``, only the on-disk SMILES cache is used -- for a reproducible
        offline training run.
    :returns: ``name -> fingerprint`` (float array of length `n_bits`).
    """
    cache = _load_cache()
    distinct = sorted({str(n or "").strip() for n in names if str(n or "").strip()})
    zero = np.zeros((n_bits,), dtype=np.float32)
    out: Dict[str, np.ndarray] = {}
    resolved_this_run = False

    for name in distinct:
        if name not in cache:
            if not use_network:
                cache[name] = None
            else:
                cache[name] = resolve_smiles(name)
                resolved_this_run = True
                time.sleep(0.2)  # be gentle with the shared PubChem service
        smiles = cache.get(name)
        fp = morgan_fingerprint(smiles, n_bits, radius) if smiles else None
        out[name] = fp if fp is not None else zero

    if resolved_this_run:
        _save_cache(cache)
        n_known = sum(1 for n in distinct if cache.get(n))
        logger.info(f"Antibiotic fingerprints: resolved {n_known}/{len(distinct)} names to a "
                   f"structure via PubChem; the rest fall back to a zero fingerprint.")
    return out
