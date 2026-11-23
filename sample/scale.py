"""
Diagnosing and correcting unit-scale errors, using the record's own JSON as evidence.

Measured against the curated databases on 2026-11-03, over 4,446 findings this project extracted
that a source database also reports for the same peptide, assay and organism:

    unit      overlap   >10x off    ~1000x off    median ratio
    uM          2,322        149             2           1.66x
    ug/mL       1,416         83             6           1.00x
    mg/mL         670        642           485        1000.00x
    nM             31         13             4           1.15x

`uM` and `ug/mL` agree with the curated value as well as two labs measuring the same MIC ever do.
`mg/mL` does not agree at all: its median ratio is exactly 1000, which is not a measurement
difference but a lost micro prefix. Excluding it, agreement is 67.0% within 2x, 86.0% within 4x
and 93.4% within 10x, median 1.43x; including it, 56% of every gross disagreement in the corpus
is this one unit.

The prior is just as lopsided. Across 182,766 curated potency measurements, `mg/mL` appears 38
times -- 0.02%. This extraction has 1,733, forty-five times the entire curated corpus. A unit the
field barely uses cannot suddenly be the fourth most common one in a literature scrape.

**Why this is not simply "rewrite mg/mL as ug/mL".** Pulling one real record apart shows three
different faults wearing the same clothes::

    MIC          value_raw "2"     mg/mL   "cathelicidin-RC1 solution (2 mg/ml, dissolved in
                                            sterile deionized water) was incubated at ..."
    MIC          value_raw "0"     uM      "The MICs of cathelicidin-RC1 and meropenem against
                                            E. coli ATCC25922 are 4.69 ..."
    Bactericidal value_raw "75"    mg/mL   "Among the 48 microorganism strains, 34 strains could
                                            be killed by cathelicidin-..."

The first is a **stock solution** captured as a potency: 2 mg/mL is the concentration the peptide
was dissolved at, and no rescaling makes it an MIC. The second is a **parse failure**: the unit is
right, the sentence states 4.69, and the field says 0. The third is a number that is not a
concentration at all. Rescaling all three would turn one wrong answer into a differently wrong
answer and lose the evidence that anything was amiss.

So this module diagnoses rather than rewrites, and it separates what it can prove from what it
cannot. The evidence it uses is the record's own JSON -- `context_sentence`, present on 53.2% of
activities with a median length of 66 characters, and the sibling measurements on the same record.
That is better evidence than the document text for this question and far cheaper: the sentence the
value came from is already saved, so no PDF is reopened and no passage has to be located.

:data:`CONFIDENT` findings are applied by step 11 (:mod:`sample.polish`) with no model call.
:data:`AMBIGUOUS` findings are escalated to step 12 (:mod:`sample.field_review`), which puts the
same context sentence in front of a local model. Nothing here guesses.
"""
import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger("sample.scale")

#: Activity types whose value is a concentration at which something happens, so a magnitude is
#: meaningful. Deliberately narrower than every concentration-bearing assay: a scale judgment
#: needs a band to judge against, and these are the ones the curated databases populate densely
#: enough to have measured one.
POTENCY_TYPES: frozenset = frozenset({
    "mic", "mic50", "mic90", "mbc", "mfc", "ic50", "ec50", "lc50", "ld50", "gi50", "cc50",
    "bactericidal", "fungicidal", "minimum inhibitory concentration",
})

#: Canonical unit per basis. Molar and mass concentrations are never compared with each other:
#: converting between them needs a molecular weight, and assuming one manufactures agreement or
#: disagreement out of nothing.
MOLAR_UNITS: Dict[str, float] = {"pm": 1e-6, "nm": 1e-3, "um": 1.0, "µm": 1.0, "mm": 1e3, "m": 1e6}
MASS_UNITS: Dict[str, float] = {
    "ng/ml": 1e-3, "ug/ml": 1.0, "µg/ml": 1.0, "mg/ml": 1e3, "g/ml": 1e6,
    "ug/l": 1e-3, "mg/l": 1.0, "g/l": 1e3,
}

#: Plausible potency range, in uM for the molar basis and ug/mL for the mass basis. Taken from
#: the 1st and 99th percentiles of the curated databases' own potency measurements: molar
#: p1 0.116 / median 22.0 / p99 1,141 over 98,610 values, mass p1 0.12 / median 20.0 / p99 1,000
#: over 83,602. Rounded outward, so the band is wider than the measurement rather than narrower.
PLAUSIBLE: Dict[str, Tuple[float, float]] = {"molar": (0.05, 2000.0), "mass": (0.05, 2000.0)}

#: Curated share of potency measurements reported in each unit, for units the field effectively
#: does not use. A unit this rare in 182,766 curated measurements is a priori suspect, whatever
#: its magnitude. `mg/ml` appears 38 times there; this extraction has 1,733.
RARE_IN_CURATED: Dict[str, float] = {"mg/ml": 0.0002, "g/ml": 0.0, "g/l": 0.0, "mm": 0.0}

#: Units in which a peptide potency cannot be stated, whatever the document says. A gram of
#: peptide per millilitre is three orders of magnitude past solubility, and these units appear
#: zero times in 182,766 curated potency measurements -- so when one shows up, the prefix was
#: lost in extraction and the document's own spelling is evidence of the damage, not of intent.
#: Kept apart from :data:`RARE_IN_CURATED` because the two license different actions: a rare unit
#: the document confirms has to be escalated, an impossible one can be corrected.
IMPOSSIBLE_POTENCY_UNITS: frozenset = frozenset({"g/ml", "g/l"})

#: For an impossible unit, the prefixes to try putting back, in order. A lost micro sign is the
#: measured failure -- `Ag/mL` 32, `lm` 30, `WM` 21, and `g/mL` 155 where the glyph vanished
#: entirely -- so the micro restoration is tried first and the milli one second, and the NUMBER is
#: never touched. Choosing by which candidate happens to land in the plausible band instead
#: proposed `g/mL -> ng/mL` on nine records, a 10^9 correction to fix a 10^6 error.
PREFIX_RESTORATION: Dict[str, Tuple[str, ...]] = {"g/ml": ("ug/ml", "mg/ml"),
                                                  "g/l": ("ug/l", "mg/l")}

#: Lookup keys are lower-case so that damaged spellings collapse onto one entry; the corpus
#: stores units in mixed case (`uM` 7,525 times, `ug/mL` 5,379). This maps a key back to the
#: spelling the rest of the pipeline uses, because `preprocessing.canonicalize_unit` is
#: case-preserving for molar units -- it maps `uM` to `uM` and `um` to `um` -- so writing a key
#: straight into a record would introduce a unit string the corpus has never otherwise used.
DISPLAY: Dict[str, str] = {
    "pm": "pM", "nm": "nM", "um": "uM", "mm": "mM", "m": "M",
    "ng/ml": "ng/mL", "ug/ml": "ug/mL", "mg/ml": "mg/mL", "g/ml": "g/mL",
    "ug/l": "ug/L", "mg/l": "mg/L", "g/l": "g/L",
}


def display_unit(key: str) -> str:
    """
    :param key: A canonical lookup key.
    :returns: The spelling to store in a record, or ``key`` when it is not a known unit.
    """
    return DISPLAY.get(key, key)


#: How a finding may be acted on.
CONFIDENT: str = "confident"      # the JSON proves it; step 11 applies it
AMBIGUOUS: str = "ambiguous"      # suspicious, unprovable here; step 12 asks a model
CLEAN: str = "clean"              # nothing to answer for

#: What is wrong, when something is.
SCALE_PREFIX_LOST: str = "scale_prefix_lost"
NOT_A_POTENCY_VALUE: str = "not_a_potency_value"
VALUE_CONTRADICTS_CONTEXT: str = "value_contradicts_context"
IMPLAUSIBLE_MAGNITUDE: str = "implausible_magnitude"

#: Phrasing that marks a concentration as something prepared rather than something measured. The
#: stock-solution confusion is the one fault here that rescaling would actively hide, so it is
#: detected first and never rescaled. Measured: 1,248 activities (2.5%) carry such a sentence.
PREPARATION = re.compile(
    r"\b(dissolv\w*|stock|diluted?|dilution|prepared?|suspend\w*|reconstitut\w*|"
    r"stored?|final\s+concentration\s+of|working\s+(?:solution|concentration))\b", re.I)

#: A quantity as written in prose. The unit alternatives spell out the ways a micro sign survives
#: PDF extraction -- as itself, as `u`, split by a space, or OCR'd into `A`, `l` or `W`.
#:
#: **An amount is not a concentration**, and the trailing-letter lookahead is what keeps the two
#: apart. Without it the molar alternative matched the `nm` inside `nmol` and read "50 nmol of
#: OVA-I peptide" -- a dose in moles -- as 50 nM, which would have let the sentence "correct" a
#: perfectly good concentration into the wrong unit. `mmol/kg` is refused for the same reason: per
#: body weight is a dose, not a concentration, so only an explicit per-litre denominator qualifies.
QUANTITY = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>"
    r"(?:[munpµAlW]\s?[gG]|[gG])\s*/\s*(?:m\s?[lL]|[lL])"      # mass per volume
    r"|(?:[munpµAlW]\s?[mM]|M)(?![A-Za-z])"                      # molar
    r"|[munp]?mol\s*/\s*[lL](?![A-Za-z])"                        # molar, spelled out
    r")", re.X)

#: A bare number immediately before a quantity, joined to it by list or range punctuation, which
#: therefore carries the same unit. Anchored at the end so only the run adjacent to the unit is
#: picked up and an unrelated number earlier in the sentence is not given a unit it never had.
_SHARED_UNIT_RUN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:,|and|or|to|through|[-–—])\s*(?=(?:\d|$))")

#: Micro-sign spellings, mapped to the canonical prefix. `A`, `l` and `W` are OCR substitutions
#: measured on this corpus (`Ag/mL` 32, `lm` 30, `WM` 21, `u M` 202, `u g/mL` 131).
_MICRO_SPELLINGS = {"u": "u", "µ": "u", "a": "u", "l": "u", "w": "u"}


def canonical_unit(unit: Any) -> str:
    """
    :param unit: A unit as written.
    :returns: A lower-case key for :data:`MOLAR_UNITS` or :data:`MASS_UNITS`, with whitespace and
        micro-sign damage normalised. ``""`` when nothing recognisable is left.

    Repairs notation only. It does not decide that a prefix is *wrong* -- `mg/mL` normalises to
    `mg/ml`, not to `ug/ml`, because whether the m should have been a micro sign is the question
    the rest of this module exists to answer, not an assumption to bake into parsing.
    """
    text = str(unit or "").strip().lower()
    if not text:
        return ""
    text = text.replace("$\\mu$", "u").replace("\\mu", "u")
    # Reciprocal notation: `ug mL-1`, `ug mL−1` (ASCII hyphen and U+2212 both occur), `ug mL^-1`.
    text = re.sub(r"\s*[\^]?\s*[-−]\s*1\b", "", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\bmicrogram\b", "ug", text)
    text = re.sub(r"\bmicromolar\b", "um", text)
    text = re.sub(r"\bmilligram\b", "mg", text)
    text = re.sub(r"\bliter\b|\blitre\b", "l", text)
    text = re.sub(r"\bmolar\b", "m", text)
    # `mmol/L` and friends are molar concentrations spelled out, and the source-database loader
    # already collapses `umol/L` to `uM`; without this they fall through as unrecognised and a
    # sentence stating one cannot correct anything.
    text = re.sub(r"^([munp]?)mol\s*/\s*l$", lambda m: (m.group(1) or "") + "m", text)
    # A split prefix: `u M`, `µ g/mL`. Join it before the space strip, so `u M` becomes `um` and
    # not the bare `m` that a blind strip would leave.
    text = re.sub(r"^([munpµalw])\s+(?=[gm])", r"\1", text)
    text = text.replace(" ", "")
    if text.endswith("ml") and "/" not in text and len(text) > 2:
        text = text[:-2] + "/ml"
    # OCR substitutions for the micro sign, only in the prefix position of a known shape.
    match = re.fullmatch(r"([munpµalw])([gm])(/.*)?", text)
    if match:
        prefix = _MICRO_SPELLINGS.get(match.group(1), match.group(1))
        text = prefix + match.group(2) + (match.group(3) or "")
    return text


def basis_of(unit: str) -> str:
    """:returns: ``"molar"``, ``"mass"`` or ``""`` for a canonical unit."""
    if unit in MOLAR_UNITS:
        return "molar"
    if unit in MASS_UNITS:
        return "mass"
    return ""


def to_canonical_value(unit: str, value: float) -> Optional[float]:
    """
    :param unit: Canonical unit.
    :param value: Value as reported.
    :returns: The value in uM (molar) or ug/mL (mass), or ``None`` when the unit has no basis.
    """
    if unit in MOLAR_UNITS:
        return value * MOLAR_UNITS[unit]
    if unit in MASS_UNITS:
        return value * MASS_UNITS[unit]
    return None


class Quantity(NamedTuple):
    """One number-and-unit pair read out of prose."""
    value: float
    unit: str
    basis: str


def quantities_in(text: Any) -> List[Quantity]:
    """
    :param text: A sentence, usually an activity's ``context_sentence``.
    :returns: Every quantity in it whose unit has a concentration basis, in order.

    This is what makes in-JSON correction possible: the sentence a value was taken from is saved
    on the activity, so the unit the document actually printed is recoverable without reopening
    the PDF. 32.8% of activities carry a sentence with a readable value and unit.
    """
    body = str(text or "")
    out: List[Quantity] = []
    for match in QUANTITY.finditer(body):
        unit = canonical_unit(match.group("unit"))
        basis = basis_of(unit)
        if not basis:
            continue
        try:
            value = float(match.group("value").replace(",", "."))
        except ValueError:
            continue
        out.append(Quantity(value, unit, basis))
        # A list sharing one trailing unit: "MICs ranging from 1.43 to 22.85 mM", "4.69 and 0.01
        # mg/ml". Only the last number is adjacent to the unit, so a match-by-match read sees one
        # quantity and treats the others as values the sentence never mentions. Measured, that
        # misreported 1,672 activities as contradicting their own context.
        for bare in _SHARED_UNIT_RUN.finditer(body[:match.start()][-60:]):
            try:
                out.append(Quantity(float(bare.group(1).replace(",", ".")), unit, basis))
            except ValueError:
                continue
    return out


class Finding(NamedTuple):
    """
    One diagnosis of one activity's scale.

    :ivar status: :data:`CLEAN`, :data:`CONFIDENT` or :data:`AMBIGUOUS`.
    :ivar fault: One of the fault constants, or ``""`` when clean.
    :ivar proposed_unit: The unit this should carry, when that is provable. Empty otherwise --
        including for every fault that rescaling would not fix.
    :ivar factor: ``proposed_unit`` relative to the current one, for the record.
    :ivar why: A short slug naming the evidence, for counters and for the step-12 prompt.
    :ivar rescalable: Whether changing the unit would make the record right. ``False`` for a
        stock solution or a number that is not a concentration, which must be flagged, not fixed.
    """
    status: str
    fault: str = ""
    proposed_unit: str = ""
    factor: float = 1.0
    why: str = ""
    rescalable: bool = True


def sibling_units(record: Dict[str, Any], activity: Dict[str, Any]) -> Dict[str, int]:
    """
    :param record: The peptide record.
    :param activity: The activity being judged, excluded from the result.
    :returns: Canonical unit -> count, over the record's other measurements of the same assay.

    The record is its own control. A paper reports a whole table of MICs in one unit, so when one
    row disagrees with its siblings the siblings are the evidence -- available in the JSON, with
    no model and no document.
    """
    kind = str(activity.get("activity_type") or "").strip().lower()
    counts: Dict[str, int] = {}
    for other in record.get("biological_activities") or ():
        if not isinstance(other, dict) or other is activity:
            continue
        if str(other.get("activity_type") or "").strip().lower() != kind:
            continue
        unit = canonical_unit(other.get("activity_unit"))
        if basis_of(unit):
            counts[unit] = counts.get(unit, 0) + 1
    return counts


def diagnose(record: Dict[str, Any], activity: Dict[str, Any]) -> Finding:
    """
    Judges one activity's unit and magnitude from the JSON alone.

    :param record: The peptide record, for its sibling measurements.
    :param activity: The activity to judge.
    :returns: A :class:`Finding`.

    The order is the order of the faults' severity, and it matters. A stock solution is checked
    before any magnitude test, because `2 mg/mL` for a dissolved peptide is a perfectly plausible
    *solution* and an impossible *MIC*; a magnitude test asked first would propose rescaling it to
    2 ug/mL and produce a confident, well-formed, wrong MIC with the evidence of the mistake
    erased. Only after that does the context sentence get to name the unit, then the siblings, then
    the plausible band.
    """
    kind = str(activity.get("activity_type") or "").strip().lower()
    if kind not in POTENCY_TYPES:
        return Finding(CLEAN)

    unit = canonical_unit(activity.get("activity_unit"))
    basis = basis_of(unit)
    if not basis:
        return Finding(CLEAN)

    value = activity.get("value_lower")
    if not isinstance(value, (int, float)) or value <= 0:
        return Finding(CLEAN)

    sentence = str(activity.get("context_sentence") or "")
    found = quantities_in(sentence)

    # 1. A concentration that was prepared, not measured.
    if sentence and PREPARATION.search(sentence):
        matching = [q for q in found if q.unit == unit and abs(q.value - value) < 1e-9]
        if matching:
            return Finding(AMBIGUOUS, NOT_A_POTENCY_VALUE, why="context_describes_a_preparation",
                           rescalable=False)

    table = MOLAR_UNITS if basis == "molar" else MASS_UNITS
    canonical_value = to_canonical_value(unit, value)

    # 2. The sentence states this value in a different unit of the same basis. Compared as raw
    #    numbers, so that a prefix the extractor misread shows up; a value it CONVERTED correctly
    #    is caught by the canonical test in step 3 and must not land here.
    for quantity in found:
        if quantity.basis != basis or abs(quantity.value - value) > 1e-9:
            continue
        if quantity.unit == unit:
            continue
        factor = table[quantity.unit] / table[unit]
        if abs(factor - 1.0) < 1e-9:
            # mg/L and ug/mL are the same quantity spelled differently. Proposing one for the
            # other changes no number and corrects no error; measured, every one of the 398
            # "confident" findings before this guard was exactly this no-op.
            continue
        return Finding(CONFIDENT, SCALE_PREFIX_LOST, proposed_unit=quantity.unit, factor=factor,
                       why="context_sentence_prints_this_value_in_another_unit")

    # 3. The sentence states no quantity equal to the recorded one, in raw or canonical terms.
    #    Then the value, not the unit, is what went wrong -- but only if the extractor did not
    #    simply convert: "MICs ranging from 1.43 to 22.85 mM" recorded as 1430 uM is correct, and
    #    is identical once both sides are in uM.
    if found and canonical_value is not None:
        same_basis = [q for q in found if q.basis == basis]
        if same_basis and not any(
                abs(q.value - value) < 1e-9
                or abs((to_canonical_value(q.unit, q.value) or -1) - canonical_value)
                <= 1e-6 * max(1.0, canonical_value)
                for q in same_basis):
            return Finding(AMBIGUOUS, VALUE_CONTRADICTS_CONTEXT,
                           why="context_sentence_states_a_different_value", rescalable=False)

    low, high = PLAUSIBLE[basis]
    out_of_band = canonical_value is not None and not (low <= canonical_value <= high)

    # 4. A unit a potency cannot be stated in. The document's own spelling is the damage, so it
    #    is not evidence against correcting; the only question is which prefix was lost, and the
    #    plausible band plus the record's other measurements answer it.
    if unit in IMPOSSIBLE_POTENCY_UNITS:
        for candidate in PREFIX_RESTORATION.get(unit, ()):
            scale = table.get(candidate)
            if scale is None or abs(scale / table[unit] - 1.0) < 1e-9:
                continue
            if low <= value * scale <= high:
                return Finding(CONFIDENT, SCALE_PREFIX_LOST, proposed_unit=candidate,
                               factor=scale / table[unit],
                               why="a_potency_cannot_be_stated_in_this_unit")
        return Finding(AMBIGUOUS, SCALE_PREFIX_LOST, why="impossible_unit_and_no_plausible_prefix")

    # 5. A unit the field barely uses, at a magnitude that is ordinary one prefix down. NOT
    #    corrected here, even when the record's other measurements use the plausible unit. Two
    #    measured reasons. The sentence often confirms the rare unit -- "IC50 was reported to be
    #    6.62 mg/ml" -- and a paper may legitimately report one assay in mg/mL and the rest in
    #    ug/mL, so sibling agreement is suggestive and not proof. And 12 of the first 14 such
    #    rescales had no context sentence at all, which is exactly the population a silent rewrite
    #    should not touch. These go to step 12 with the sentence attached.
    if unit in RARE_IN_CURATED and out_of_band:
        confirmed = any(q.unit == unit and abs(q.value - value) < 1e-9 for q in found)
        return Finding(AMBIGUOUS, SCALE_PREFIX_LOST,
                       why=("context_confirms_a_rare_unit" if confirmed
                            else "rare_unit_at_an_implausible_magnitude"))

    # 6. Out of band on an ordinary unit. Nothing here can say which end is wrong.
    if out_of_band:
        return Finding(AMBIGUOUS, IMPLAUSIBLE_MAGNITUDE,
                       why="outside_the_measured_potency_band")

    return Finding(CLEAN)


def describe_bands() -> str:
    """:returns: The measured bands as text, for a report or a prompt."""
    lines = ["Plausible potency ranges, from the curated databases' own measurements:"]
    for basis, (low, high) in sorted(PLAUSIBLE.items()):
        unit = "uM" if basis == "molar" else "ug/mL"
        lines.append(f"  {basis:6s} {low:g} to {high:g} {unit}")
    lines.append("Units the field effectively does not use for potency: "
                 + ", ".join(sorted(RARE_IN_CURATED)))
    return "\n".join(lines)
