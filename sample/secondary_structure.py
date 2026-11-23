"""
Per-residue secondary-structure propensity from sequence composition alone.

Replaces the RDKit 2D chemical-graph depiction on the peptide page. A bond diagram of a
30-residue peptide is unreadable at any reasonable size and answers a question nobody visiting
this page is asking; what the page needs is the same thing a biologist reaches for first when
looking at a new AMP sequence -- is this region likely helical, is it likely to form a strand.

**This is not a structure prediction.** It is the classical Chou-Fasman (1978) propensity scale,
averaged over a sliding window and thresholded -- the same kind of estimate EMBOSS's ``chofam`` or
a textbook exercise produces. It needs nothing but the sequence, so it covers every peptide in the
release including the ~38 % that ColabFold cannot fold (too short to mean anything, or non-standard
residues) and the corpus-wide fold run that has not happened yet (**C1**). Where a real prediction
exists -- an actual solved or ColabFold-predicted 3D structure -- that is what the 3D viewer below
this panel shows; the two are not in competition, they answer different questions at different
levels of confidence and this one never claims to be the other.

Chou & Fasman, *Prediction of the Secondary Structure of Proteins from their Amino Acid Sequence*,
Adv. Enzymol. Relat. Areas Mol. Biol. 47:45-148 (1978). Reproduced in essentially every
protein-structure-prediction reference since; the specific numbers below are the widely-cited
consensus table (e.g. as tabulated in EMBOSS's ``chofam.dat``).
"""
from typing import Dict, List, NamedTuple

#: (helix propensity, sheet propensity), Chou-Fasman 1978. Turn propensity is not used here: a
#: three-way split (helix / sheet / coil) is the level of detail a residue-height display can show
#: without becoming noise, and "turn" would mostly relabel short coil stretches.
_PROPENSITY: Dict[str, "_Propensity"] = {}


class _Propensity(NamedTuple):
    helix: float
    sheet: float


_TABLE = {
    "E": (1.51, 0.37), "A": (1.42, 0.83), "L": (1.21, 1.30), "H": (1.00, 0.87),
    "M": (1.45, 1.05), "Q": (1.11, 1.10), "W": (1.08, 1.37), "V": (1.06, 1.70),
    "F": (1.13, 1.38), "K": (1.16, 0.74), "I": (1.08, 1.60), "D": (1.01, 0.54),
    "T": (0.83, 1.19), "S": (0.77, 0.75), "R": (0.98, 0.93), "C": (0.70, 1.19),
    "N": (0.67, 0.89), "Y": (0.69, 1.47), "P": (0.57, 0.55), "G": (0.57, 0.75),
}
_PROPENSITY = {residue: _Propensity(*values) for residue, values in _TABLE.items()}

#: A propensity right at 1.0 -- the scale's own break-even point between "favours" and "disfavours"
#: -- for a residue outside the standard 20, so it never wins a comparison and just dilutes its
#: window toward coil, which is the honest answer for a modification the scale has no data for.
_NEUTRAL = _Propensity(1.0, 1.0)

#: Standard Chou-Fasman nucleation window: four consecutive residues averaging above threshold.
WINDOW = 4

#: Chou-Fasman's own break-even value for calling a window helix- or sheet-favouring.
THRESHOLD = 1.00


def residue_states(sequence: str) -> List[str]:
    """
    Classifies every residue as ``H`` (helix), ``E`` (sheet) or ``C`` (coil).

    :param sequence: One-letter sequence, any case.
    :returns: One state per residue, same length and order as the input. A residue is windowed
        with its neighbours -- secondary structure is a property of a stretch, not a single
        residue in isolation -- using whichever window fits (shrinking near the ends rather than
        padding with an invented value).
    """
    sequence = str(sequence or "").upper().strip()
    n = len(sequence)
    if n == 0:
        return []

    half = WINDOW // 2
    states = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + (WINDOW - half))
        window = sequence[start:end]
        propensities = [_PROPENSITY.get(r, _NEUTRAL) for r in window]
        avg_helix = sum(p.helix for p in propensities) / len(propensities)
        avg_sheet = sum(p.sheet for p in propensities) / len(propensities)

        if avg_helix >= THRESHOLD and avg_helix >= avg_sheet:
            states.append("H")
        elif avg_sheet >= THRESHOLD and avg_sheet > avg_helix:
            states.append("E")
        else:
            states.append("C")
    return states


def summarize(sequence: str) -> Dict[str, float]:
    """
    :param sequence: One-letter sequence.
    :returns: ``{"helix": fraction, "sheet": fraction, "coil": fraction}`` of residues in each
        state, for a one-line caption above the per-residue diagram.
    """
    states = residue_states(sequence)
    if not states:
        return {"helix": 0.0, "sheet": 0.0, "coil": 0.0}
    n = len(states)
    return {
        "helix": states.count("H") / n,
        "sheet": states.count("E") / n,
        "coil": states.count("C") / n,
    }
