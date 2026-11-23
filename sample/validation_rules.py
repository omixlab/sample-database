"""
The validation rule tables, as plain data.

Every constraint the pipeline enforces on a measurement lives here, as lists of tuples laid out to
read like tables. This file is meant to be INSPECTED and EDITED -- by someone checking whether the
database's rules match the domain, not only by someone reading the code that applies them.

Keeping them here rather than inline in :mod:`sample.strict_schema` matters because these are
scientific claims, not implementation details. "A hemolysis assay is measured on cells, never on a
bacterium" and "a MIC below 1 nM is a parsing error" are statements a reviewer should be able to
check and disagree with, and that is hard when they are spread across half a dozen literals in the
middle of enforcement logic. Changing a rule here changes the database; nothing else needs editing.

Four tables:

``UNIT_RANGES``          -- one row per canonical unit: its quantity and plausible bounds.
``TARGET_GROUP_DOMAINS`` -- one row per activity class: which organism domains it may target.
``ACTIVITY_RULES``       -- one row per assay: the classes, units and target kind it allows.
``UNIT_EQUIVALENCES``    -- one row per redundant unit: what it converts to, and by what factor.
"""
from typing import Dict, NamedTuple, Optional, Set, Tuple


class UnitRange(NamedTuple):
    """
    Plausible value range for one canonical unit.

    :param unit: The canonical unit string, exactly as published.
    :param quantity: Physical quantity, used to decide whether a unit can express a concentration.
        One of ``molar``, ``mass_volume``, ``mass_mass``, ``length``, ``count``, ``proportion``,
        ``activity`` or ``dimensionless``.
    :param min_value: Smallest value that is a measurement rather than a parsing artifact.
    :param max_value: Largest such value.
    """
    unit: str
    quantity: str
    min_value: float
    max_value: float


class TargetGroupRule(NamedTuple):
    """
    Which organism domains an activity class may legitimately be tested against.

    :param target_group: Activity class, e.g. ``anticancer``.
    :param domains: NCBI-lineage domains that are valid targets. Empty means "no restriction".
    :param note: Why the restriction exists, for whoever reviews this table.
    """
    target_group: str
    domains: Tuple[str, ...]
    note: str


class ActivityRule(NamedTuple):
    """
    What one assay type is allowed to look like.

    :param activity_type: Assay name, exactly as published in ``activity_type``.
    :param target_groups: Activity classes this assay can belong to. Empty means "any".
    :param quantities: Physical quantities its unit may express (keys of :class:`UnitRange`).
    :param target_kind: ``organism`` for assays run against a whole organism, ``cells`` for those
        run on cultured cells, ``either`` when both are normal.
    :param note: Why, for review.
    """
    activity_type: str
    target_groups: Tuple[str, ...]
    quantities: Tuple[str, ...]
    target_kind: str
    note: str


class UnitEquivalence(NamedTuple):
    """
    A redundant unit and the canonical one it collapses to.

    :param source: Unit as written by some source.
    :param target: Canonical unit it becomes.
    :param factor: Multiplier applied to the value. ``1.0`` where the two are the same quantity
        written differently (1 umol/L IS 1 uM), otherwise a real conversion.
    :param note: For review.
    """
    source: str
    target: str
    factor: float
    note: str


# --- Units -------------------------------------------------------------------------------------
#
# Bounds are deliberately wide: they exist to catch OCR and parsing damage (a micromolar potency of
# 1e6, a negative concentration), not to express what a "good" peptide looks like. A real but
# unusual result must survive.
#
#            unit              quantity          min       max
UNIT_RANGES = [
    UnitRange("uM",            "molar",          1e-6,     1e5),
    UnitRange("mM",            "molar",          1e-9,     1e3),
    UnitRange("nM",            "molar",          1e-3,     1e8),
    UnitRange("pM",            "molar",          1e-2,     1e10),
    UnitRange("M",             "molar",          1e-12,    1e2),
    UnitRange("umol/L",        "molar",          1e-6,     1e5),
    UnitRange("nmol/L",        "molar",          1e-3,     1e8),
    UnitRange("mmol/L",        "molar",          1e-9,     1e3),
    UnitRange("mol/L",         "molar",          1e-12,    1e2),
    UnitRange("ug/mL",         "mass_volume",    1e-6,     1e5),
    UnitRange("mg/mL",         "mass_volume",    1e-9,     1e3),
    UnitRange("ng/mL",         "mass_volume",    1e-3,     1e8),
    UnitRange("pg/mL",         "mass_volume",    1e-2,     1e10),
    UnitRange("mg/L",          "mass_volume",    1e-6,     1e5),
    UnitRange("ug/uL",         "mass_volume",    1e-9,     1e3),
    UnitRange("g/L",           "mass_volume",    1e-9,     1e3),
    UnitRange("mg/kg",         "mass_mass",      1e-4,     1e4),
    UnitRange("ug/kg",         "mass_mass",      1e-1,     1e7),
    UnitRange("g/kg",          "mass_mass",      1e-7,     1e1),
    # A zone of inhibition cannot exceed the plate it is measured on. Standard discs run on 90 mm
    # or 100 mm plates, so the old 200 mm ceiling admitted values describing no possible assay.
    # 150 mm rather than 100 mm because large plates and bioassay trays are genuinely used, and
    # this bound exists to catch OCR damage, not to referee plate choice. Tighten to 100.0 if this
    # corpus only ever uses standard plates.
    UnitRange("mm",            "length",         0.1,      150.0),
    UnitRange("cm",            "length",         0.01,     20.0),
    UnitRange("CFU/mL",        "count",          1e0,      1e12),
    UnitRange("log10 CFU/mL",  "count",          0.0,      15.0),
    # Above 100% on purpose. Normalized hemolysis and viability assays routinely report 110-120%
    # from background absorbance or a lysis baseline that under-reads, and those are real, usable
    # measurements with noise -- not parsing errors. A 105% ceiling dropped them.
    UnitRange("%",             "proportion",     0.0,      120.0),
    UnitRange("IU",            "activity",       1e-6,     1e9),
    UnitRange("U",             "activity",       1e-6,     1e9),
    UnitRange("U/mL",          "activity",       1e-6,     1e9),
    UnitRange("mU",            "activity",       1e-3,     1e12),
    # The empty unit is a real unit here: a ratio of two concentrations has none. Admitting it as a
    # first-class row lets TI, SI and FIC flow through the same machinery as everything else, while
    # units_for_activity still keeps it out of the assays that DO require a unit -- a MIC with no
    # unit is rejected exactly as before.
    UnitRange("",              "dimensionless",  1e-4,     1e6),
]

#: Quantities that can express a concentration. A potency metric reported in anything else is a
#: mislabel: the assay name and the number came from different parts of the text.
CONCENTRATION_QUANTITIES = ("molar", "mass_volume", "mass_mass", "proportion")


# --- Activity classes vs target organism domains -------------------------------------------------
#
# Domains come from NCBI lineage (see scripts/build_taxonomy_domains.py), so these are checked
# against the taxonomy rather than against a hand-written list of genera.
#
#                   target_group      allowed domains                note
TARGET_GROUP_DOMAINS = [
    TargetGroupRule("antibacterial",  ("bacteria", "archaea"),
                    "bacteria, plus archaea -- rare clinically but real in environmental screens, "
                    "and an archaeal target is a prokaryote by any reading of 'antibacterial'"),
    TargetGroupRule("antifungal",     ("fungi",),                    "tested against fungi"),
    TargetGroupRule("antiviral",      ("virus",),                    "tested against viruses"),
    TargetGroupRule("antiparasitic",  ("protozoa", "animal"),
                    "protozoa AND helminths. Nematodes and flatworms are Metazoa, so every "
                    "anthelmintic target -- Caenorhabditis, Haemonchus, Schistosoma, Brugia -- "
                    "resolves to the animal domain; restricting this to protozoa rejected all of "
                    "them. The cost is that it no longer separates antiparasitic from anticancer "
                    "by domain alone, which the assay type does instead"),
    TargetGroupRule("anticancer",     ("animal",),
                    "runs on cultured cells, so the organism is the species the line came from"),
    TargetGroupRule("other",          (),
                    "catch-all: hemolysis, membrane assays, in vivo dosing -- no restriction"),
]


# --- Assays ---------------------------------------------------------------------------------------
#
# `quantities` restricts the UNIT; `target_kind` restricts WHAT the assay can be run on. A MIC is a
# concentration against a whole organism; a zone of inhibition is a length; hemolysis is measured on
# erythrocytes and never on the bacterium the peptide also kills.
#
#                 activity_type                target_groups                                  quantities                              target_kind  note
ACTIVITY_RULES = [
    ActivityRule("MIC",       ("antibacterial", "antifungal", "antiparasitic", "other"), CONCENTRATION_QUANTITIES, "organism", "minimum inhibitory concentration"),
    ActivityRule("MIC50",     ("antibacterial", "antifungal", "antiparasitic", "other"), CONCENTRATION_QUANTITIES, "organism", "MIC for 50% of isolates"),
    ActivityRule("MIC80",     ("antibacterial", "antifungal", "antiparasitic", "other"), CONCENTRATION_QUANTITIES, "organism", "MIC for 80% of isolates"),
    ActivityRule("MIC90",     ("antibacterial", "antifungal", "antiparasitic", "other"), CONCENTRATION_QUANTITIES, "organism", "MIC for 90% of isolates"),
    ActivityRule("MIC100",    ("antibacterial", "antifungal", "antiparasitic", "other"), CONCENTRATION_QUANTITIES, "organism", "complete inhibition"),
    ActivityRule("MBC",       ("antibacterial", "other"),                                CONCENTRATION_QUANTITIES, "organism", "minimum bactericidal concentration"),
    ActivityRule("MBC50",     ("antibacterial", "other"),                                CONCENTRATION_QUANTITIES, "organism", "MBC for 50% of isolates"),
    ActivityRule("MBC90",     ("antibacterial", "other"),                                CONCENTRATION_QUANTITIES, "organism", "MBC for 90% of isolates"),
    ActivityRule("MFC",       ("antifungal", "other"),                                   CONCENTRATION_QUANTITIES, "organism", "minimum fungicidal concentration"),
    ActivityRule("MLC",       ("antibacterial", "antifungal", "other"),                  CONCENTRATION_QUANTITIES, "organism", "minimum lethal concentration"),
    ActivityRule("MBIC",      ("antibacterial", "antifungal", "other"),                  CONCENTRATION_QUANTITIES, "organism", "minimum biofilm inhibitory concentration"),
    ActivityRule("MBEC",      ("antibacterial", "antifungal", "other"),                  CONCENTRATION_QUANTITIES, "organism", "minimum biofilm eradication concentration"),
    ActivityRule("ZOI",       ("antibacterial", "antifungal", "other"),                  ("length",),              "organism", "zone of inhibition -- the only assay measured as a length"),
    ActivityRule("FIC",       ("antibacterial", "antifungal", "other"),                  ("dimensionless",),       "organism", "fractional inhibitory concentration index -- a ratio"),
    # Ratios of one measured concentration to another, so they carry no unit. Both are standard
    # AMP reporting: a peptide is only useful if it kills the target well below the dose that
    # lyses host cells, and these are how that margin is stated.
    ActivityRule("TI",        (),                                                        ("dimensionless",),       "either",   "therapeutic index -- toxic dose over effective dose"),
    ActivityRule("SI",        (),                                                        ("dimensionless",),       "either",   "selectivity index -- host-cell toxicity over antimicrobial potency"),

    ActivityRule("IC50",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "half-maximal inhibitory concentration"),
    ActivityRule("IC90",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "90% inhibitory concentration"),
    ActivityRule("EC50",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "half-maximal effective concentration"),
    ActivityRule("EC90",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "90% effective concentration"),
    ActivityRule("GI50",      ("anticancer", "other"),                                   CONCENTRATION_QUANTITIES, "cells",    "50% growth inhibition of a cell line"),
    ActivityRule("CC50",      ("anticancer", "other"),                                   CONCENTRATION_QUANTITIES, "cells",    "50% cytotoxic concentration"),
    ActivityRule("CC90",      ("anticancer", "other"),                                   CONCENTRATION_QUANTITIES, "cells",    "90% cytotoxic concentration"),
    ActivityRule("TC50",      ("anticancer", "other"),                                   CONCENTRATION_QUANTITIES, "cells",    "50% toxic concentration"),
    ActivityRule("TD50",      ("anticancer", "other"),                                   CONCENTRATION_QUANTITIES, "cells",    "50% toxic dose"),
    ActivityRule("HC50",      ("other",),                                                CONCENTRATION_QUANTITIES, "cells",    "50% hemolytic concentration -- on erythrocytes"),
    ActivityRule("MHC",       ("other",),                                                CONCENTRATION_QUANTITIES, "cells",    "minimum hemolytic concentration"),

    ActivityRule("LD50",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "median lethal dose, usually in vivo"),
    ActivityRule("LD90",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "90% lethal dose"),
    ActivityRule("LC50",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "median lethal concentration"),
    ActivityRule("ED50",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "median effective dose"),
    ActivityRule("ED90",      (),                                                        CONCENTRATION_QUANTITIES, "either",   "90% effective dose"),

    ActivityRule("Hemolysis",                 ("other",),                                CONCENTRATION_QUANTITIES, "cells",    "measured on erythrocytes, never on the microbe"),
    ActivityRule("Cytotoxicity",              ("anticancer", "other"),                   CONCENTRATION_QUANTITIES, "cells",    "measured on cultured cells"),
    ActivityRule("Growth Inhibition",         (),                                        CONCENTRATION_QUANTITIES, "either",   "proportion inhibited, or the concentration causing it"),
    ActivityRule("Biofilm Inhibition",        ("antibacterial", "antifungal", "other"),  CONCENTRATION_QUANTITIES, "organism", "biofilm biomass reduction"),
    ActivityRule("Membrane Permeabilization", (),                                        CONCENTRATION_QUANTITIES, "either",   "dye leakage or depolarization"),
    ActivityRule("Bactericidal",              ("antibacterial", "other"),                CONCENTRATION_QUANTITIES + ("count",), "organism", "time-kill; a log-reduction readout is legitimate"),
]


# --- Unit equivalences ----------------------------------------------------------------------------
#
# Applied before validation so one physical quantity has one spelling. A factor of 1.0 means the two
# strings denote the same thing (1 umol/L IS 1 uM) and only the label changes.
#
#                 source        target     factor    note
UNIT_EQUIVALENCES = [
    UnitEquivalence("umol/L",   "uM",      1.0,      "micromoles per litre is micromolar"),
    UnitEquivalence("nmol/L",   "nM",      1.0,      "nanomoles per litre is nanomolar"),
    UnitEquivalence("mmol/L",   "uM",      1000.0,   "millimolar to micromolar"),
    UnitEquivalence("mol/L",    "uM",      1e6,      "molar to micromolar"),
    UnitEquivalence("mM",       "uM",      1000.0,   "millimolar to micromolar"),
    UnitEquivalence("mg/L",     "ug/mL",   1.0,      "1 mg/L is 1 ug/mL"),
    UnitEquivalence("g/L",      "mg/mL",   1.0,      "1 g/L is 1 mg/mL"),
    UnitEquivalence("ug/uL",    "mg/mL",   1.0,      "1 ug/uL is 1 mg/mL -- the usual bench unit for peptide stocks"),
    UnitEquivalence("M",        "uM",      1e6,      "molar to micromolar, so all molarity lands on one unit"),
]


# --- Derived lookups ------------------------------------------------------------------------------
#
# Built from the tables above so the tables stay the single source of truth. Nothing below should be
# edited directly; edit the rows and these follow.

UNIT_BY_NAME: Dict[str, UnitRange] = {r.unit: r for r in UNIT_RANGES}
VALID_UNITS: Set[str] = set(UNIT_BY_NAME)
CONCENTRATION_UNITS: Set[str] = {
    r.unit for r in UNIT_RANGES if r.quantity in CONCENTRATION_QUANTITIES}
RULE_BY_ACTIVITY: Dict[str, ActivityRule] = {r.activity_type: r for r in ACTIVITY_RULES}
VALID_ACTIVITY_TYPES: Set[str] = set(RULE_BY_ACTIVITY)
DOMAINS_BY_GROUP: Dict[str, Tuple[str, ...]] = {
    r.target_group: r.domains for r in TARGET_GROUP_DOMAINS}
VALID_TARGET_GROUPS: Set[str] = set(DOMAINS_BY_GROUP)
EQUIVALENCE_BY_UNIT: Dict[str, UnitEquivalence] = {e.source: e for e in UNIT_EQUIVALENCES}


def derivable_target_group(activity_type: str) -> Optional[str]:
    """
    The activity class an assay determines by itself, when it determines one.

    An assay that permits exactly one class does not need that class to be observed -- it implies
    it. Hemolysis is the case that matters: it is always an ``other`` measurement, but upstream
    extraction inherits the paper's PRIMARY class, so a hemolysis row in an antibacterial paper
    arrives labelled ``antibacterial``. Treating that as a contradiction rejected real hemolysis
    data; deriving the class instead corrects it without consulting anything.

    :param activity_type: Assay name.
    :returns: The implied class, or ``None`` when the assay permits several and the class has to
        come from the target organism instead.
    """
    rule = RULE_BY_ACTIVITY.get(activity_type)
    if rule is None or len(rule.target_groups) != 1:
        return None
    return rule.target_groups[0]


def groups_for_activity(activity_type: str) -> Set[str]:
    """
    The activity classes one assay may be filed under.

    Distinct from :func:`derivable_target_group`, which answers only the special case of an assay
    permitting exactly one class. This returns the whole permitted set, so a class that is merely
    *wrong* -- a MIC filed as ``anticancer`` -- can be rejected even when the assay allows several
    and no single class can be implied.

    :param activity_type: Assay name.
    :returns: Permitted classes, or an empty set when the assay is unknown or unrestricted. An
        empty set means "do not constrain", so an unrecognized assay is not silently narrowed.
    """
    rule = RULE_BY_ACTIVITY.get(activity_type)
    return set(rule.target_groups) if rule is not None else set()


def target_kind_for_activity(activity_type: str) -> Optional[str]:
    """
    What an assay is measured against: ``organism``, ``cells``, ``either``, or ``None``.

    :param activity_type: Assay name.
    :returns: The assay's target kind, or ``None`` when the assay is not in the table.
    """
    rule = RULE_BY_ACTIVITY.get(activity_type)
    return rule.target_kind if rule is not None else None


def units_for_activity(activity_type: str) -> Set[str]:
    """
    The units one assay may be reported in.

    :param activity_type: Assay name.
    :returns: Canonical units allowed for it; every unit when the assay is not in the table, so an
        unrecognized assay is not silently constrained.
    """
    rule = RULE_BY_ACTIVITY.get(activity_type)
    if rule is None:
        return set(VALID_UNITS)
    allowed = {r.unit for r in UNIT_RANGES if r.quantity in rule.quantities}
    # "dimensionless" has no unit row; it means the value is a bare ratio.
    if "dimensionless" in rule.quantities:
        allowed.add("")
    return allowed


def value_range_for_unit(unit: str) -> Optional[Tuple[float, float]]:
    """
    :param unit: Canonical unit.
    :returns: ``(min, max)`` for it, or ``None`` when the unit is unknown and no bound applies.
    """
    row = UNIT_BY_NAME.get(unit)
    return (row.min_value, row.max_value) if row else None


def convert_to_canonical(value: float, unit: str) -> Tuple[float, str]:
    """
    Applies any unit equivalence, so one quantity has one spelling.

    :param value: Numeric value as reported.
    :param unit: Unit as reported.
    :returns: ``(value, unit)`` after conversion; unchanged when no equivalence applies.
    """
    rule = EQUIVALENCE_BY_UNIT.get(unit)
    if rule is None:
        return value, unit
    return value * rule.factor, rule.target
