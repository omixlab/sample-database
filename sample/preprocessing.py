import re
import logging
from typing import List, Dict, Any

from sample.concentration import NON_CONCENTRATION_UNITS
from sample.nlp import parse_activity_value_fast

logger = logging.getLogger(__name__)

# Assay types that express potency as a concentration, so an implausibly large value is
# meaningful evidence of a parsing/OCR error rather than a weak-but-real result.
_POTENCY_METRICS = {"MIC", "MIC50", "MIC90", "MIC100", "IC50", "IC90", "EC50", "EC90",
                    "MBC", "MBC50", "MBC90", "MFC", "CC50", "LC50", "ED50"}
_MOLAR_UNITS = {"um", "µm", "μm"}          # everything is normalized to uM before this check
_MAX_PLAUSIBLE_POTENCY_UM = 10_000.0       # 10 mM -- far beyond any peptide reported as active

# Scanned-literature renderings of the micro sign, which OCR loses in several distinct ways. These
# are matched CASE-SENSITIVELY and deliberately so: "Mg/mL" and "mg/mL" differ by one letter's case
# but by a factor of 1000, and publishing both spellings unconverted leaves a consumer unable to
# tell a microgram from a milligram. "MM" and "Lg/ml" are not real units at all -- no unit system
# writes them -- so reading them as micro is unambiguous rather than a guess.
_OCR_UNIT_CANONICAL = {
    # The four spellings of a two-letter unit whose case is its entire meaning. "mM" (millimolar)
    # and "mm" (millimeter -- a real zone-of-inhibition unit) are both genuine and must survive
    # distinct; "MM" and "Mm" are neither, and are the OCR renderings of "uM" and "mm" respectively.
    "MM": "uM", "LM": "uM", "IM": "uM",
    # Lowercase "l" is the single most common OCR rendering of the micro sign in this corpus --
    # 1,303 records shipped with the literal unit "lM", which is not a unit in any system. The
    # uppercase "LM" was mapped here from the start and the lowercase form, which is what the OCR
    # actually produces (see VALUE_PATTERN's comment in extractor.py), was the one left out.
    "lM": "uM", "Im": "uM",
    "Mm": "mm",
    "Mg/mL": "ug/mL", "Mg/ml": "ug/mL", "MG/ML": "ug/mL",
    "Lg/mL": "ug/mL", "Lg/ml": "ug/mL", "lg/mL": "ug/mL", "lg/ml": "ug/mL",
    "Mmol/L": "umol/L", "lmol/L": "umol/L",
}

# The micro sign arrives as three distinct code points -- U+00B5 MICRO SIGN, U+03BC GREEK SMALL
# LETTER MU, and plain ASCII "u" -- because PDF producers disagree about which to embed. All three
# mean micro, but they are different strings, so the same measurement shipped under "uM"/"μM"/"µM"
# as three separate units: 11,667 micromolar records split across four spellings (those three plus
# "lM"), and 5,824 microgram-per-millilitre records across seven. A consumer grouping by
# activity_unit saw four unrelated units where there is one.
_MICRO_CHARS = "µμ"

# Case of the denominator carries no meaning ("ug/ml" and "ug/mL" are the same unit), unlike the
# scalar molar units above, so it is normalized to one spelling. Litres are conventionally
# capitalized in this literature to keep "l" from being read as a digit 1.
_DENOMINATOR_FORM = {"ml": "mL", "l": "L", "dl": "dL", "ul": "uL", "kg": "kg", "g": "g", "spot": "spot"}
_NUMERATOR_FORM = {
    "ug": "ug", "mg": "mg", "ng": "ng", "pg": "pg", "g": "g", "kg": "kg",
    "mol": "mol", "mmol": "mmol", "umol": "umol", "nmol": "nmol", "pmol": "pmol",
    "cfu": "CFU", "log10 cfu": "log10 CFU", "log10cfu": "log10 CFU",
    "iu": "IU", "u": "U", "mu": "mU",
}
# Units with no denominator, where case IS meaning and may not be folded away.
_SCALAR_FORM = {"uM": "uM", "mM": "mM", "nM": "nM", "pM": "pM", "M": "M",
                "mm": "mm", "cm": "cm", "%": "%", "IU": "IU", "U": "U"}


def canonicalize_unit(unit: str) -> str:
    """
    Reduces a unit to one spelling per physical quantity, so `activity_unit` can be grouped and
    joined on directly.

    Order is load-bearing. The case-sensitive OCR repair runs FIRST, while "Mg/mL" (an OCR'd
    microgram) is still distinguishable from "mg/mL" (a real milligram) -- the two differ by one
    letter's case and by a factor of 1000. Only afterwards is the denominator's case folded, once
    nothing depends on it.
    """
    if not unit:
        return unit
    u = unit.strip()
    # Mojibake repair runs before everything else, so the exact-string map and the micro-glyph
    # fold below see a unit spelled in real characters. See _MOJIBAKE_MICRO for why this is a
    # lookahead and not a blanket replacement.
    if "\ufffd" in u:
        u = _MOJIBAKE_MICRO.sub("u", u)
    # Parenthesised and spaced forms reach here as "( ug/mL )"; the partition below would then
    # read "( ug" as the numerator and leave it unrecognised.
    u = u.strip("() ").strip()
    u = _OCR_UNIT_CANONICAL.get(u, u)
    for ch in _MICRO_CHARS:
        u = u.replace(ch, "u")
    if "/" in u:
        num, _, den = u.partition("/")
        num, den = num.strip(), den.strip()
        return f"{_NUMERATOR_FORM.get(num.lower(), num)}/{_DENOMINATOR_FORM.get(den.lower(), den)}"
    return _SCALAR_FORM.get(u, u)


# A concentration metric measured in a unit that cannot express a concentration is a mislabel, not
# a measurement: the assay name and the number came from different parts of the text. Colony counts
# (CFU/mL) are inoculum densities from a methods sentence -- a MIC's p99 across the corpus was
# 1e7 purely because of these. Lengths (mm) are a mix of figure scale bars ("Scale bars = 20 mm"),
# biofilm thickness, and genuine micromolar values whose micro sign OCR'd to "m" ("MIC < 5 mm").
# Neither the label nor the unit can be trusted over the other, so the record is dropped rather
# than repaired by guesswork.
#: Units that cannot express a concentration, so a POTENCY metric reported in one is a mislabel --
#: the assay name and the number came from different parts of the source.
#:
#: "mm"/"cm" are zone-of-inhibition diameters. "%" joined them from measured table output: a
#: docling-parsed table of percent killing ("20.2 +/- 15.5", "89.6 +/- 4.8", "95.0 +/- 1.5" for
#: hBD-2) was emitted with activity_type MIC, because the column header carried the assay name
#: while the values were percentages. A MIC is a concentration; a percentage is a response at some
#: concentration, and the two are not interchangeable. 2,401 such records were produced by the
#: table backend in the last full run.
# Units a potency metric cannot be measured in. Split across two tables ON PURPOSE, by whether
# case carries meaning:
#
#   * These three are matched CASE-SENSITIVELY and stay local. "mm" (millimetre, a real
#     zone-of-inhibition unit) and "mM" (millimolar, a real concentration) differ only by case, so
#     a case-folding lookup would reject every millimolar measurement in the corpus. "%" is
#     correct for hemolysis but not for an MIC.
#   * Everything in `sample.concentration.NON_CONCENTRATION_UNITS` -- mass, time, energy and
#     count units -- is matched case-INSENSITIVELY, which is safe because no member collides with
#     a concentration unit under folding ("Da", "min", "kcal/mol", "bp"; note bare "m" is
#     deliberately absent, since "M" is molar).
#
# That shared table is what was missing here: this set held only the three case-sensitive entries,
# so a mass unit on a potency metric was never rejected. Measured over 53,107 activities on disk,
# 6,173 (11.6%) were such pairings -- 2,852 mass, 2,078 time, 753 energy, 490 count -- including
# `brevinin-1RTb MIC = 2548.2 Da` and `HG1 MIC = 4113.18 Da`, which are molecular weights read out
# of a mass-spec table into the MIC column.
_NON_CONCENTRATION_UNITS = {"mm", "cm", "%"}

#: U+FFFD standing in for the micro sign, repaired POSITIONALLY rather than by blanket
#: replacement.
#:
#: PDF text extraction emits U+FFFD for several different characters, and only one of them is a
#: micro sign. Measured over the 1,098 corrupted unit strings on disk: 1,027 (93.5%) are U+FFFD
#: immediately before a unit token -- `�M`, `�g/mL`, `� M`, `�mol/L` -- where
#: micro is the only character that can appear. The remainder are a different character entirely:
#: `�10�4` is a multiplication sign and a superscript minus, `�1` in `�g
#: ml�1` is a superscript minus, `CD8�T` is a Greek letter in a cell-type name, and a
#: leading `�50` in a VALUE is an unrecoverable comparison or +/- operator.
#:
#: So the lookahead is the whole point: replacing U+FFFD unconditionally would turn `�10
#: �4` into `u10u4` and invent a unit. Only U+FFFD followed by M, m, g, mol or L is read as
#: micro; every other occurrence is left alone for the caller to reject.
_MOJIBAKE_MICRO = re.compile(r"\ufffd\s*(?=(?:mol|[MmgL]))")

#: Longest string still plausibly an activity TYPE rather than a description of one.
#:
#: The longest member of `validation_rules.VALID_ACTIVITY_TYPES` is "Membrane Permeabilization"
#: at 25 characters, and only that one is 25 or longer, so 40 clears every real type with room to
#: spare. Above it the model is describing the assay instead of naming it: measured over 52,713
#: activities on disk, all 58 values longer than this are prose -- "hydrophobicity (RP-HPLC
#: retention time tR )", "Intracellular staining for IFN-g following peptide stimulation",
#: "Not specified (measured by percentage relative to control cells)". None is a bioactivity, and
#: `strict_schema` already rejects every one of them.
_MAX_ACTIVITY_TYPE_CHARS = 40

#: A range written with the word "to", which the shared value parser does not recognise.
#:
#: `nlp.parse_activity_value_fast` reads "5-50", "5 - 50" and "5–50" as ranges and returns both
#: bounds, but "5 to 50" yields only the lower one, so 2,312 activities on disk carry a range in
#: `value_raw` with `value_upper` unset. Rewriting it to the hyphen form the parser already
#: handles is cheaper and safer than a second range grammar.
#:
#: Anchored between digits so "up to 50" -- an upper bound, not a range -- is left alone.
_WORD_RANGE = re.compile(r"(?<=\d)\s+to\s+(?=\d)")

def raw_value_and_unit(act: Dict[str, Any]) -> tuple:
    """
    The (value, unit) pair as the source document printed it.

    `activity_unit` is the CANONICAL unit and belongs with `value_lower`. Once a magnitude
    conversion has run (mM -> uM), it no longer describes `value_raw`, so reading those two fields
    together -- the obvious thing to do, and what five call sites in this codebase did -- reports a
    "0.3 mM" measurement as "0.3 uM", off by a factor of 1000. Any consumer that shows a value to a
    human, or checks one against the source text, needs this pair rather than a mixture of the two.

    Falls back to `activity_unit` for records built outside standardize_and_filter_activities
    (pdf_tables.py assembles its own, with no conversion, so there the two units are the same).
    """
    return act.get('value_raw', ''), (act.get('value_raw_unit') or act.get('activity_unit') or '')


def standardize_and_filter_activities(activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts values and units to standard formats and filters false positives.
    """
    valid_activities = []

    for act in activities:
        val = act.get('value_lower')
        if val is None:
            val = act.get('value_raw')

        unit = act.get('activity_unit')

        # If there's no value at all, it's structurally incomplete.
        # It should be flagged for LLM recovery or dropped.
        if val is None or val == "":
            continue

        # Use the same operator/range/±/scientific-notation-aware parser as the rest of the
        # pipeline (nlp.py's parse_activity_value_fast, already used by extractor.py's percentage
        # sanity bound and pdf_tables.py) instead of a standalone strip-non-digits-and-float()
        # pass. The standalone version couldn't parse a real range ("5-10"), a "±" value
        # ("5.0 ± 1.0"), or "x 10^" scientific notation -- all of which VALUE_PATTERN itself can
        # legitimately capture as value_raw -- and silently dropped every one of those records.
        atype_raw = str(act.get('activity_type') or '')
        if len(atype_raw) > _MAX_ACTIVITY_TYPE_CHARS:
            # The prose is kept rather than discarded: it describes a real assay, and the
            # value/unit/target beside it may well be a real measurement. Only the TYPE field is
            # wrong, so it is emptied -- the pipeline's existing spelling for "unknown type" --
            # and the description moved somewhere it does not corrupt grouping or joins.
            act['activity_type_detail'] = atype_raw
            act['activity_type'] = ''

        upper_float = None
        cfu_exponent_recovered = False
        if isinstance(val, str):
            val = _WORD_RANGE.sub("-", val)
            operator, val_float, upper_float = parse_activity_value_fast(val)
            if val_float is None:
                logger.debug(f"Could not parse activity value: {val}")
                continue
            # Preserve censoring. "<100" and "100" are different measurements -- one is an upper
            # bound, the other exact -- but value_lower stored 100.0 for both, so a consumer
            # reading only the numeric column cannot tell them apart and would treat a bound as a
            # point estimate. value_raw kept the operator all along; this surfaces it in a field
            # that survives aggregation.
            if operator:
                act['value_operator'] = operator
        else:
            try:
                val_float = float(val)
            except (TypeError, ValueError):
                continue

        # Very basic unit standardization (e.g. mM to uM). Checked case-sensitively: "mM"
        # (millimolar) and "mm" (millimeter -- a real ZOI/zone-of-inhibition unit in this domain)
        # are different units that collapse to the same string under .lower() -- see the
        # VALUE_PATTERN comment in extractor.py for the real-output example this caused.
        if unit:
            # Canonicalize spelling first, so the magnitude conversions below see one known form
            # rather than any of "MM"/"lM"/"μM"/"Mg/mL". See canonicalize_unit.
            unit = canonicalize_unit(unit)

            # A concentration metric in a colony-count or length unit is a mislabel; see
            # _NON_CONCENTRATION_UNITS.
            if act.get('activity_type') in _POTENCY_METRICS and (
                    'CFU' in unit.upper() or unit in _NON_CONCENTRATION_UNITS
                    or unit.strip().lower() in NON_CONCENTRATION_UNITS):
                logger.debug(
                    f"Dropping {act.get('activity_type')} reported in {unit!r} -- that unit cannot "
                    f"express a concentration, so the metric label and the value disagree"
                )
                continue

            # value_raw preserves the literal string the document printed, which is measured in the
            # unit the document printed beside it -- NOT in the canonical unit this record ships.
            # Recording it keeps the raw pair self-consistent: a "0.3 mM" measurement stores
            # value_raw "0.3" + value_raw_unit "mM" alongside value_lower 300.0 + activity_unit
            # "uM". Without this field the two most obviously-paired columns (value_raw,
            # activity_unit) read as "0.3 uM" -- a 1000x error, and the unit value_raw was actually
            # measured in was recorded nowhere.
            act['value_raw_unit'] = unit

            if unit.strip().lower() == 'm':
                # Case-insensitively, because sample.concentration._MOLAR_UNIT_TO_UM reaches its
                # "m" entry case-insensitively too. Matching only uppercase "M" here meant a
                # lowercase "m" was converted at 1e6 by that table while this stage left it
                # unconverted -- and _MOLAR_UNITS holds only micromolar spellings, so the
                # unconverted unit then bypassed _MAX_PLAUSIBLE_POTENCY_UM entirely and published.
                # Lowercase "mm" cannot reach here: _NON_CONCENTRATION_UNITS drops it above as a
                # length.
                #
                # Molar. Converted rather than passed through so the plausibility bound below can
                # see it: _MOLAR_UNITS holds only micromolar spellings, so an unconverted "M" was
                # invisible to every magnitude guard. Measured in table output -- an "Arg" row at
                # 0.1-5 M, read out of a buffer-composition table rather than an assay. After
                # conversion 0.1 M is 100,000 uM, an order of magnitude past
                # _MAX_PLAUSIBLE_POTENCY_UM, so such rows are now dropped instead of published.
                val_float *= 1_000_000.0
                if upper_float is not None:
                    upper_float *= 1_000_000.0
                unit = 'uM'
            elif unit == 'mM':
                val_float *= 1000.0
                if upper_float is not None:
                    upper_float *= 1000.0
                unit = 'uM'
                # The CONVERSION above is correct and must stay: 5.52 mM really is 5520 uM, and
                # 342 of the 354 such rows in the current release are sound. What is suspect is the
                # SOURCE unit. No one reports an active peptide's MIC/IC50 in millimolar, and the
                # micro sign is routinely lost by OCR ("µM" -> "mM"), so a potency metric whose
                # document-printed unit was mM is more likely a damaged micro sign than a genuinely
                # millimolar result. Measured: the curation audit found a record storing 5520 uM
                # for a paper that printed 4.83-5.52 uM.
                #
                # FLAGGED, not dropped or silently corrected. The two readings are indistinguishable
                # once the character is lost, so "repairing" it would invent a measurement; and the
                # existing _MAX_PLAUSIBLE_POTENCY_UM guard below only catches values at or above
                # 10 mM, which this class of error usually sits under. The flag lets a consumer
                # exclude or review them without this stage guessing.
                if act.get('activity_type') in _POTENCY_METRICS:
                    act['value_scale_suspect'] = True
                    act['value_scale_suspect_reason'] = (
                        'potency metric reported in mM; the micro sign is commonly lost by OCR, '
                        'so this may be a 1000x overstatement of a uM value')
            elif unit.lower() == 'mg/l':
                # Numerically identical (1 mg/L == 1 ug/mL), so this is a relabel, not a conversion.
                unit = 'ug/mL'

        # Recover a lost exponent in colony counts. Superscripts do not survive PDF extraction, so
        # an inoculum of 10^6 CFU/ml arrives as the literal "106 CFU/ml" -- seen throughout the
        # corpus as 106/107/104. Bacterial counts are always reported as powers of ten in this
        # literature, and no assay reports a raw count of 106 cells per ml, so a 3-digit value of
        # the form "10n" in CFU units is unambiguously 10^n.
        if unit and unit.upper().startswith("CFU"):
            digits = str(act.get('value_raw', '')).strip()
            if re.fullmatch(r'10\d', digits):
                val_float = 10.0 ** int(digits[2])
                # This REPLACES the value rather than scaling it, so there is no factor to apply
                # to a bound parsed from the same string. Dropping the bound keeps the pair in one
                # magnitude; a colony-count range with both bounds written "10n" has not been
                # observed.
                cfu_exponent_recovered = True

        # Physical-plausibility bound on reported potency. A peptide whose MIC/IC50-class value is
        # in the millimolar range is not an active compound anyone reports as a finding, and in
        # this corpus such values are overwhelmingly an OCR artifact: the micro sign is frequently
        # rendered as a latin "m", so a genuine "10 µM" IC50 arrives as "10 mM" and the mM->µM
        # conversion above then inflates it a further 1000x. Observed in real output: IC50 values
        # of 10000 and 500000 µM for MMP-inhibitor peptides whose published potencies are 10 µM
        # and 500 µM. The values cannot be repaired (the two readings are indistinguishable once
        # the character is lost) so the affected records are dropped rather than published wrong.
        if act.get('activity_type') in _POTENCY_METRICS and unit and unit.lower() in _MOLAR_UNITS:
            if val_float >= _MAX_PLAUSIBLE_POTENCY_UM:
                logger.debug(
                    f"Dropping implausible {act.get('activity_type')} of {val_float} uM "
                    f"(likely an OCR'd micro sign read as milli)"
                )
                continue

        act['value_lower'] = val_float
        act['activity_unit'] = unit
        # The upper bound of a range, in the SAME unit as value_lower -- which is why it is scaled
        # in each converting branch above rather than taken straight from the parser. Written only
        # when the parser found one and nothing already set it, and never when the bounds would
        # disagree in magnitude. `parse_activity_value_fast` has returned this third value all
        # along; this stage discarded it with `_`, which is why 2,312 activities on disk carry a
        # range in value_raw with value_upper unset. A backwards or implausible bound is caught
        # downstream by strict_schema, which validates value_upper when it is present.
        if (upper_float is not None and not cfu_exponent_recovered
                and act.get('value_upper') in (None, "")
                and upper_float >= val_float):
            act['value_upper'] = upper_float
        valid_activities.append(act)

    return valid_activities
