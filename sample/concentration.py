"""
Concentration-unit conversion: reported value + unit -> micromolar, millimolar, micrograms/mL,
and potency (-log10 molar).

Deliberately dependency-free (stdlib only). This logic is shared by two callers in different conda
environments -- sample.ml_models (the activity threshold and, now, the regression targets, run
under ``sample-ml``) and sample.serialize (the value_mM/value_ug_ml/potency columns written into
the SQLite database, run under ``sample-chem``) -- and started out living in sample.preprocessing,
which imports sample.nlp for its own OCR-repair work. sample.nlp imports ``ahocorasick``, a real
dependency of the extraction environment that neither ml_models nor serialize otherwise needs;
importing preprocessing from ml_models pulled that whole chain in and broke `train-ml` in an
environment that never installs it. A module with no imports beyond the standard library cannot
have this problem again, regardless of which environment either caller runs in.
"""
import math
from typing import Any, Dict, Optional, Tuple

# Concentration-unit synonyms recognized for molarity conversion, keyed lowercase with either
# unicode "micro" glyph (µ U+00B5, μ U+03BC) folded to ASCII "u". Molar-family units convert
# directly to micromolar; mass-family units convert to micrograms/mL first and then need the
# peptide's own molecular weight to reach micromolar -- mass and molar concentrations are not
# interchangeable without it.
#: Molar units, as multiples of micromolar. "m" (molar) is included so that a value reported in it
#: can be CONVERTED and therefore plausibility-checked: without it, a table reporting "0.1 M"
#: produced an unconvertible record that slipped past every magnitude guard downstream. 1 M is
#: 1e6 uM, so any potency metric genuinely in molar units lands far above
#: :data:`sample.preprocessing._MAX_PLAUSIBLE_POTENCY_UM` and is dropped -- which is the correct
#: outcome, since a peptide MIC is never molar (measured: an "Arg" row at 0.1-5 M, read out of a
#: buffer-composition table rather than an assay).
_MOLAR_UNIT_TO_UM: Dict[str, float] = {"pm": 1e-6, "nm": 1e-3, "um": 1.0, "mm": 1e3, "m": 1e6}
_MASS_UNIT_TO_UG_ML: Dict[str, float] = {"ng/ml": 1e-3, "ug/ml": 1.0, "mg/ml": 1e3}


def _fold_micro_glyph(unit: Any) -> str:
    """:returns: ``unit`` lowercased with either unicode "micro" glyph folded to ASCII "u"."""
    return str(unit).strip().lower().replace("µ", "u").replace("μ", "u")


def to_micromolar(value: float, unit: Any, molecular_weight: Optional[float] = None) -> Optional[float]:
    """
    Converts one reported concentration to micromolar.

    :param value: The reported numeric value.
    :param unit: Its unit, in any spelling :func:`_fold_micro_glyph` and case-folding normalize to
        one of :data:`_MOLAR_UNIT_TO_UM`/:data:`_MASS_UNIT_TO_UG_ML`'s keys.
    :param molecular_weight: The peptide's molecular weight (g/mol) -- needed only to convert a
        mass-per-volume unit; ``None`` or non-positive makes any mass-unit conversion impossible.
    :returns: The micromolar concentration, or ``None`` when the unit is not a recognized
        concentration (e.g. ``%``, or ``mg/kg`` -- a dose, not a concentration) or a mass unit's
        molecular weight is unavailable.
    """
    key = _fold_micro_glyph(unit)
    if key in _MOLAR_UNIT_TO_UM:
        return float(value) * _MOLAR_UNIT_TO_UM[key]
    if key in _MASS_UNIT_TO_UG_ML:
        if not molecular_weight or molecular_weight <= 0:
            return None
        ug_per_ml = float(value) * _MASS_UNIT_TO_UG_ML[key]
        return ug_per_ml * 1000.0 / float(molecular_weight)
    return None


def concentration_conversions(value: float, unit: Any,
                              molecular_weight: Optional[float]) -> Dict[str, Optional[float]]:
    """
    Expresses one reported concentration in every unit this project displays or trains on.

    A single micromolar conversion (:func:`to_micromolar`) feeds every other figure, so a
    measurement reported in either a molar or a mass-per-volume unit ends up with the same four
    numbers -- there is exactly one place a unit-conversion mistake could hide, not four.

    :param value: The reported numeric value.
    :param unit: Its unit.
    :param molecular_weight: The peptide's molecular weight (g/mol); ``None`` when unknown.
    :returns: ``{"uM": ..., "mM": ..., "ug_ml": ..., "potency": ...}``. Every value is ``None``
        when the unit is not a recognized concentration. ``uM``/``mM``/``potency`` are available
        from a molar-unit reading even without ``molecular_weight``; ``ug_ml`` needs it regardless
        of the original unit (going from a molar reading to mass, or preserving a mass reading,
        both require knowing how much one mole of this particular peptide weighs). ``potency`` is
        the standard pharmacological transform -log10([M]) -- e.g. IC50 5 uM -> pIC50 5.3 -- used
        so activity spanning orders of magnitude compares on one linear scale, the same reason
        pH represents [H+], and the same reason it -- not raw mM -- is what
        :func:`sample.ml_models.build_tasks` regresses on.
    """
    out: Dict[str, Optional[float]] = {"uM": None, "mM": None, "ug_ml": None, "potency": None}
    um = to_micromolar(value, unit, molecular_weight)
    if um is None:
        return out
    out["uM"] = um
    out["mM"] = um / 1000.0
    if molecular_weight and molecular_weight > 0:
        out["ug_ml"] = um * float(molecular_weight) / 1000.0
    if um > 0:
        out["potency"] = -math.log10(um * 1e-6)
    return out


#: Activity types whose value IS a concentration. A measurement of one of these cannot be
#: expressed in a unit of mass, time, energy or sequence length, whatever the extractor says.
CONCENTRATION_ACTIVITY_TYPES: frozenset = frozenset({
    "mic", "mic50", "mic90", "mbc", "mbec", "mfc", "ic50", "ic90", "ec50", "ed50",
    "lc50", "ld50", "cc50", "gi50", "bmc", "mlc",
})

#: Units that are never a concentration, grouped by what they actually measure. Each group is
#: here because it was observed standing in for a concentration on real output, not on principle.
#:
#: Measured over 53,107 activities on disk: 6,173 (11.6%) paired a concentration activity type
#: with one of these -- 2,852 mass, 2,078 time, 753 energy, 490 count. The mass cases are the
#: clearest: `MIC = 4113.18 Da` for peptide HG1 is its molecular weight lifted from a mass-spec
#: table, and `brevinin-1RTb MIC = 2548.2 Da` likewise. A reader cannot tell those from a real
#: MIC once they are in a database.
#:
#: Deliberately NOT here:
#:   `mm`  -- ambiguous with `mM` (millimolar) under any case-folding, and a zone-of-inhibition
#:            diameter is a legitimate `ZOI` measurement. Dropping it risks real data. No
#:            concentration type was observed carrying it, so it costs nothing to leave out.
#:   `%`   -- correct for Hemolysis and for growth-inhibition percentages.
#:   `mg/kg` -- an in-vivo DOSE, not an in-vitro concentration, but a real measurement that
#:            belongs with its activity rather than being discarded.
NON_CONCENTRATION_UNITS: Dict[str, str] = {
    "da": "mass", "kda": "mass", "g/mol": "mass", "amu": "mass", "dalton": "mass",
    "min": "time", "mins": "time", "minute": "time", "minutes": "time",
    "s": "time", "sec": "time", "secs": "time", "second": "time", "seconds": "time",
    "h": "time", "hr": "time", "hrs": "time", "hour": "time", "hours": "time",
    "kcal/mol": "energy", "kj/mol": "energy",
    "bp": "count", "aa": "count", "residues": "count", "nt": "count",
}


def activity_unit_is_coherent(activity_type: Any, activity_unit: Any) -> Tuple[bool, str]:
    """
    Whether a unit can express the activity it is attached to.

    Only one direction is judged: a type that must be a concentration paired with a unit that
    cannot be one. The reverse -- a concentration unit on an unrecognized type -- is left alone,
    because the type vocabulary is open (the model returns things like "ATP-based assay for
    amebic viability") and a strict whitelist there would discard real measurements.

    :param activity_type: The extracted ``activity_type``, e.g. ``"MIC"``.
    :param activity_unit: The extracted ``activity_unit``, e.g. ``"Da"``.
    :returns: ``(True, "")`` when the pairing is possible, else ``(False, reason)`` where the
        reason names what the unit actually measures, e.g. ``"MIC_in_mass_unit"``.
    """
    kind = (str(activity_type) if activity_type is not None else "").strip().lower()
    if kind not in CONCENTRATION_ACTIVITY_TYPES:
        return True, ""
    unit = _fold_micro_glyph(activity_unit).strip().lower()
    if not unit:
        return True, ""
    measures = NON_CONCENTRATION_UNITS.get(unit)
    if measures is None:
        return True, ""
    return False, f"{kind.upper()}_in_{measures}_unit"
