import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("sample.descriptors")

# Kyte-Doolittle Hydrophobicity Scale
_KD_HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Eisenberg Hydrophobicity Scale for Hydrophobic Moment
_EISENBERG_SCALE = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

def calculate_hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """
    Calculates hydrophobic moment (uH) assuming an alpha-helical conformation (angle = 100 degrees/residue).
    """
    if not seq:
        return 0.0
    
    rad_angle = math.radians(angle)
    sin_sum = 0.0
    cos_sum = 0.0
    n = 0

    for i, aa in enumerate(seq.upper()):
        h_val = _EISENBERG_SCALE.get(aa, 0.0)
        sin_sum += h_val * math.sin(i * rad_angle)
        cos_sum += h_val * math.cos(i * rad_angle)
        n += 1

    # `n` is len(seq), and the empty case already returned above, so no zero-division guard is
    # needed here. The one that used to sit here was unreachable.
    return round(math.sqrt(sin_sum**2 + cos_sum**2) / n, 4)

def calculate_net_charge(seq: str, ph: float = 7.4) -> float:
    """
    Calculates net charge of a peptide sequence at a given pH.
    """
    if not seq:
        return 0.0
    
    # pKa values of ionizable groups
    pKa_Nterm = 9.69
    pKa_Cterm = 2.34
    pKa_side = {'K': 10.53, 'R': 12.48, 'H': 6.00, 'D': 3.86, 'E': 4.25, 'C': 8.33, 'Y': 10.07}

    # N-terminal positive charge
    charge = 1.0 / (1.0 + 10.0**(ph - pKa_Nterm))
    # C-terminal negative charge
    charge -= 1.0 / (1.0 + 10.0**(pKa_Cterm - ph))

    for aa in seq.upper():
        if aa in ('K', 'R', 'H'):
            charge += 1.0 / (1.0 + 10.0**(ph - pKa_side[aa]))
        elif aa in ('D', 'E', 'C', 'Y'):
            charge -= 1.0 / (1.0 + 10.0**(pKa_side[aa] - ph))

    return round(charge, 2)

def calculate_boman_index(seq: str) -> float:
    """
    Calculates the Boman (potential protein-binding) index.
    """
    if not seq:
        return 0.0
    
    # Boman solubility values (kcal/mol)
    boman_values = {
        'L': -4.92, 'I': -4.92, 'V': -4.04, 'F': -2.98, 'M': -2.35,
        'W': -2.33, 'A': -1.81, 'C': -1.28, 'G': -0.94, 'Y': -0.14,
        'T': 2.57, 'S': 3.40, 'H': 4.66, 'Q': 5.54, 'K': 5.55,
        'N': 6.64, 'E': 6.81, 'D': 8.72, 'R': 9.38, 'P': 0.00
    }
    
    total = sum(boman_values.get(aa, 0.0) for aa in seq.upper())
    return round(total / len(seq), 3)

def compute_all_descriptors(seq: str) -> Dict[str, Any]:
    """
    Computes a comprehensive set of physicochemical and biophysical descriptors for a peptide sequence.
    """
    if not seq:
        return {}

    seq = seq.upper().strip()
    length = len(seq)
    
    kd_vals = [_KD_HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    mean_hydrophobicity = round(sum(kd_vals) / length, 3) if length > 0 else 0.0
    uH = calculate_hydrophobic_moment(seq)
    charge_74 = calculate_net_charge(seq, ph=7.4)
    boman = calculate_boman_index(seq)

    # Basic amino acid counts & percentages
    pos_count = sum(1 for aa in seq if aa in 'KRH')
    neg_count = sum(1 for aa in seq if aa in 'DE')
    hydrophobic_count = sum(1 for aa in seq if aa in 'AILMFWV')

    return {
        "sequence_length": length,
        "mean_hydrophobicity_kd": mean_hydrophobicity,
        "hydrophobic_moment_uH": uH,
        "net_charge_ph74": charge_74,
        "boman_index": boman,
        "positive_residue_ratio": round(pos_count / length, 3),
        "negative_residue_ratio": round(neg_count / length, 3),
        "hydrophobic_residue_ratio": round(hydrophobic_count / length, 3)
    }
