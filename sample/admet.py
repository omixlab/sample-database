"""
ADMET-lite: an absorption/distribution/metabolism/excretion/toxicity summary for a peptide, built
only from what the sequence itself supports.

Full small-molecule ADMET suites (SwissADME, pkCSM) lean on models trained on drug-like small
molecules; most of their endpoints -- CYP metabolism, Caco-2 permeability, oral bioavailability --
are calibrated on chemistry a 20-residue peptide is nowhere near, so quoting them for a peptide is
worse than admitting the gap. This module therefore reports only properties that are either directly
computable from the sequence (molecular weight, isoelectric point, GRAVY, instability index,
aliphatic index, net charge, Boman index, hydrophobic moment) or standard, peptide-appropriate
rules of thumb (Guruprasad's instability cut, the N-end rule for in-vivo half-life, an aggregation-
hotspot scan). Everything it cannot honestly compute is listed explicitly in ``gaps`` rather than
filled with a number -- above all **membrane permeability**, which needs a dedicated predictor
(PharmPapp was vetted and is KNIME-only, so not directly integrable; no Python-native model chosen
yet), and the **toxicity/hemolysis** endpoints, which the annotator's external predictors (HemoPI2,
and ToxinPred once wired) answer.

Reusing the pipeline's own descriptors (:mod:`sample.descriptors`) keeps the numbers on this page
identical to the ones the enrichment step already computed for the same peptide, rather than a
second, subtly different implementation.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from sample import descriptors

logger = logging.getLogger("sample.admet")

#: The 20 standard amino acids. ProtParam raises on anything else (ambiguity codes, modified
#: residues), so metrics that go through it are computed on the sequence reduced to these.
_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")

#: Guruprasad et al. (1990): a computed instability index above this predicts an unstable protein.
INSTABILITY_CUTOFF: float = 40.0

#: Residues counted as aggregation-prone for the hotspot scan (aliphatic + aromatic hydrophobics).
_AGG_PRONE = set("AILMFVWY")

#: Minimum run length of aggregation-prone residues to call a hotspot.
_AGG_MIN_RUN: int = 3

#: N-end rule (mammalian): the N-terminal residue predicts metabolic half-life. Destabilizing
#: residues target a protein for rapid degradation; stabilizing ones leave it long-lived. This is
#: the coarse two-class form -- enough to flag "this peptide's own N-terminus makes it short-lived",
#: which is what a designer changes with an N-terminal cap.
_N_END_DESTABILIZING = set("RKHFWYLIDENQ")


def _clean(sequence: str) -> str:
    """
    :param sequence: A raw peptide sequence.
    :returns: The sequence upper-cased and reduced to the 20 standard residues, for ProtParam.
    """
    return "".join(a for a in str(sequence or "").upper() if a in _STANDARD)


def _protein_analysis(clean_seq: str):
    """
    :param clean_seq: A sequence of standard residues only.
    :returns: A ``ProteinAnalysis`` for it, or ``None`` when it is empty.
    """
    if not clean_seq:
        return None
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    return ProteinAnalysis(clean_seq)


def aliphatic_index(sequence: str) -> Optional[float]:
    """
    Ikai's (1980) aliphatic index: the relative volume of aliphatic side chains, a thermostability
    proxy.

    :param sequence: A peptide sequence.
    :returns: ``100 * (X_A + 2.9 X_V + 3.9 (X_I + X_L))`` over mole fractions, or ``None`` for an
        empty sequence.
    """
    seq = _clean(sequence)
    n = len(seq)
    if n == 0:
        return None
    a = seq.count("A") / n
    v = seq.count("V") / n
    i = seq.count("I") / n
    ll = seq.count("L") / n
    return round(100.0 * (a + 2.9 * v + 3.9 * (i + ll)), 2)


def physicochemical(sequence: str) -> Dict[str, Any]:
    """
    Computes the physicochemical block of the profile.

    :param sequence: A peptide sequence.
    :returns: Molecular weight, isoelectric point, net charge at pH 7, GRAVY, aromaticity,
        instability index, aliphatic index, Boman index and hydrophobic moment. ProtParam-derived
        values are ``None`` when the sequence has no standard residues.
    """
    clean = _clean(sequence)
    pa = _protein_analysis(clean)

    block: Dict[str, Any] = {
        "molecular_weight": round(pa.molecular_weight(), 2) if pa else None,
        "isoelectric_point": round(pa.isoelectric_point(), 2) if pa else None,
        "net_charge_ph7": descriptors.calculate_net_charge(clean, ph=7.0) if clean else None,
        "gravy": round(pa.gravy(), 3) if pa else None,
        "aromaticity": round(pa.aromaticity(), 3) if pa else None,
        "instability_index": round(pa.instability_index(), 2) if pa else None,
        "aliphatic_index": aliphatic_index(sequence),
        "boman_index": descriptors.calculate_boman_index(clean) if clean else None,
        "hydrophobic_moment": descriptors.calculate_hydrophobic_moment(clean) if clean else None,
    }
    return block


def instability_flag(instability_index: Optional[float]) -> Dict[str, Any]:
    """
    :param instability_index: The computed instability index, or ``None``.
    :returns: Whether the peptide is predicted stable (index below :data:`INSTABILITY_CUTOFF`).
    """
    if instability_index is None:
        return {"stable": None, "index": None, "cutoff": INSTABILITY_CUTOFF}
    return {
        "stable": instability_index < INSTABILITY_CUTOFF,
        "index": instability_index,
        "cutoff": INSTABILITY_CUTOFF,
        "note": "Guruprasad instability index; below 40 predicts a stable protein.",
    }


def half_life_class(sequence: str) -> Dict[str, Any]:
    """
    Classifies in-vivo half-life from the N-terminal residue (N-end rule, mammalian).

    :param sequence: A peptide sequence.
    :returns: The N-terminal residue and whether it is stabilizing or destabilizing.
    """
    clean = _clean(sequence)
    if not clean:
        return {"n_terminal": None, "class": None}
    residue = clean[0]
    destabilizing = residue in _N_END_DESTABILIZING
    return {
        "n_terminal": residue,
        "class": "destabilizing" if destabilizing else "stabilizing",
        "note": ("N-end rule: a destabilizing N-terminus predicts rapid degradation; cap or "
                 "substitute it to extend half-life." if destabilizing else
                 "N-end rule: a stabilizing N-terminus predicts a longer half-life."),
    }


def aggregation_hotspots(sequence: str) -> Dict[str, Any]:
    """
    Scans for runs of aggregation-prone residues.

    A coarse sequence-only proxy: maximal runs of at least :data:`_AGG_MIN_RUN` consecutive
    aliphatic/aromatic hydrophobic residues. It flags the hydrophobic stretches that drive peptide
    self-association, not a calibrated aggregation score.

    :param sequence: A peptide sequence.
    :returns: The fraction of the sequence inside a hotspot and the hotspot spans (1-based,
        inclusive).
    """
    clean = _clean(sequence)
    n = len(clean)
    if n == 0:
        return {"hotspot_fraction": None, "hotspots": []}
    hotspots: List[Dict[str, int]] = []
    covered = 0
    for match in re.finditer(f"[{''.join(sorted(_AGG_PRONE))}]{{{_AGG_MIN_RUN},}}", clean):
        hotspots.append({"start": match.start() + 1, "end": match.end()})
        covered += match.end() - match.start()
    return {
        "hotspot_fraction": round(covered / n, 3),
        "hotspots": hotspots,
        "note": (f"Runs of >={_AGG_MIN_RUN} consecutive hydrophobic residues (coarse aggregation "
                 f"proxy, not a calibrated score)."),
    }


def net_charge_class(charge: Optional[float]) -> Dict[str, Any]:
    """
    :param charge: Net charge at pH 7, or ``None``.
    :returns: A cationic/anionic/near-neutral label. Cationicity is the property most tied to an
        AMP's membrane activity and to non-specific toxicity, so it is surfaced as a flag.
    """
    if charge is None:
        return {"charge": None, "class": None}
    if charge >= 2.0:
        label = "cationic"
    elif charge <= -2.0:
        label = "anionic"
    else:
        label = "near-neutral"
    return {"charge": charge, "class": label}


#: ADMET endpoints this framework deliberately does not compute, each with why. Surfaced so the
#: profile is honest about its own boundary rather than silently partial.
def _gaps() -> List[Dict[str, str]]:
    return [
        {"property": "membrane_permeability", "status": "not_predicted",
         "note": ("Not predicted for linear peptides: the public peptide-permeability data "
                  "(CycPeptMPDB) is cyclic macrocycles whose passive-diffusion permeability does not "
                  "transfer to these linear, membrane-disrupting peptides. CycPeptMPDB is indexed as "
                  "a separate cyclic-peptide source instead (sample.cycpeptmpdb).")},
        {"property": "hemolysis", "status": "available_via_annotator",
         "note": ("HemoPI2 (and macrel's own hemolysis head) predict this; see the Annotation panel "
                  "below once `build-annotations` has run for this peptide.")},
        {"property": "toxicity", "status": "blocked_upstream",
         "note": ("ToxinPred 3.0 predicts this, but its shipped model pickle does not unpickle under "
                  "any scikit-learn version tried (1.1.3-1.6.1) -- a real upstream packaging defect, "
                  "not something fixable from this project's side. See KNOWN_ISSUES.md.")},
        {"property": "cyp_metabolism", "status": "not_applicable",
         "note": ("CYP450 metabolism models are calibrated on drug-like small molecules, not "
                  "peptides; a number here would be misleading.")},
    ]


def admet_profile(sequence: str) -> Dict[str, Any]:
    """
    Builds the full ADMET-lite profile for one peptide.

    :param sequence: A one-letter peptide sequence.
    :returns: ``{"sequence", "length", "non_standard_residues", "physicochemical", "flags",
        "gaps"}``. ``non_standard_residues`` counts residues excluded from the ProtParam-based
        metrics, so a caller can see when those numbers are approximate. Returns an empty dict for an
        empty sequence.
    """
    raw = str(sequence or "").upper().strip()
    if not raw:
        return {}
    clean = _clean(raw)

    phys = physicochemical(raw)
    return {
        "sequence": raw,
        "length": len(raw),
        "non_standard_residues": len(raw) - len(clean),
        "physicochemical": phys,
        "flags": {
            "instability": instability_flag(phys["instability_index"]),
            "half_life": half_life_class(raw),
            "aggregation": aggregation_hotspots(raw),
            "net_charge": net_charge_class(phys["net_charge_ph7"]),
        },
        "gaps": _gaps(),
    }
