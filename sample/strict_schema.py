"""
The strict output contract for the peptide database, and the gate that enforces it.

Everything upstream of this module is recall-oriented: extraction casts wide and each later stage
removes a class of error. This is the last stage that decides what the database actually contains,
so it is the one place where a field is either in canonical form or it is not published. A consumer
joining on `target_organism`, grouping by `activity_unit`, or filtering on `value_lower` must be
able to do so without a cleaning pass of their own.

Two properties are deliberate:

- **Field-level, not record-level.** A peptide with a good sequence, a good MIC and one malformed
  organism loses the organism, not the record. Dropping whole records for a single bad field would
  discard far more real data than it protects.
- **The LLM cannot widen the contract.** A revision suggested by a model is re-checked by these
  same rules before it is accepted, so the model can only ever propose a value that the
  deterministic gate independently agrees is valid. That ordering is what keeps a generated value
  from entering the database on the model's authority alone.
"""
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("sample.strict_schema")

# --- Controlled vocabularies -------------------------------------------------------------------
#
# All of these are DERIVED from the rule tables in sample/validation_rules.py, which is the file to
# edit when the rules change. They are re-exported here because this module is the enforcement API
# every caller already imports, and moving the names would break those callers for no benefit.
from sample.validation_rules import (            # noqa: E402  (re-export, after the docstring)
    ACTIVITY_RULES,
    CONCENTRATION_UNITS as _CONCENTRATION_UNITS,
    DOMAINS_BY_GROUP,
    derivable_target_group,
    groups_for_activity,
    target_kind_for_activity,
    RULE_BY_ACTIVITY,
    UNIT_RANGES,
    VALID_ACTIVITY_TYPES,
    VALID_TARGET_GROUPS,
    VALID_UNITS,
    convert_to_canonical,
    units_for_activity,
    value_range_for_unit,
)

VALID_OPERATORS: Set[str] = {"", "<", ">", "<=", ">=", "~", "="}

# Assays whose value is a concentration, so a colony count or a length is a mislabel rather than a
# measurement. Derived from the tables: an assay is a potency metric when its allowed units are
# concentrations and it is not one of the length- or ratio-valued assays.
_POTENCY_METRICS: Set[str] = {
    r.activity_type for r in ACTIVITY_RULES
    if "length" not in r.quantities and "dimensionless" not in r.quantities
}

# Canonical one-letter alphabet. A published sequence is exactly this -- no ambiguity codes (B, Z,
# X, J), no gaps, no modification syntax -- so that every consumer can parse it the same way.
_CANONICAL_SEQUENCE: re.Pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
#: Shortest sequence published. Below this a 'sequence' is a fragment or an acronym.
MIN_SEQUENCE_LEN: int = 4
#: Longest sequence published. Above this the record is a protein, not a peptide.
MAX_SEQUENCE_LEN: int = 50


def published_json_schema() -> Dict[str, Any]:
    """
    JSON Schema for a published extraction document, derived from the constants above so the schema
    and the code that enforces it cannot drift apart.

    This exists because step 16 was validating published records against
    `data/prompts/01_extraction_prompt_response_format.json` -- the schema describing what the
    extraction *prompt* asks the LLM to return, which requires `reasoning`,
    `peptide_sequence_raw` and `peptide_sequence_3_letter`. Real pipeline records have none of
    those, so every document failed, and the failure handling then made it worse (see
    validation.extract_repair_extractions).
    """
    return {
        "type": "object",
        "required": ["peptides"],
        "properties": {
            "metadata": {"type": "object"},
            "peptides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["peptide_sequence", "peptide_name", "biological_activities"],
                    "properties": {
                        "peptide_sequence": {"type": "string"},
                        "peptide_name": {"type": "string"},
                        "peptide_sequence_1letter": {"type": "string"},
                        "peptide_sequence_3letter": {"type": "string"},
                        "peptide_sequence_helm": {"type": "string"},
                        "peptide_sequence_SMILES_format": {"type": "string"},
                        "database_cross_reference": {"type": "array"},
                        "context": {"type": "string"},
                        "document_id": {"type": "string"},
                        "revision": {"type": "object"},
                        "biological_activities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["activity_type", "activity_unit", "value_lower"],
                                "properties": {
                                    "activity_type": {"enum": sorted(VALID_ACTIVITY_TYPES)},
                                    "activity_unit": {"enum": sorted(VALID_UNITS)},
                                    "value_raw_unit": {"enum": sorted(VALID_UNITS)},
                                    "value_raw": {"type": ["string", "number"]},
                                    "value_lower": {"type": "number", "minimum": 0},
                                    "value_operator": {"enum": sorted(VALID_OPERATORS)},
                                    "target_organism": {"type": "string"},
                                    "target_cell_line": {"type": "string"},
                                    "target_group": {"enum": sorted(VALID_TARGET_GROUPS)},
                                    "context_sentence": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


class Violation:
    """One field that failed the contract, and why."""

    __slots__ = ("path", "field", "value", "reason")

    def __init__(self, path: str, field: str, value: Any, reason: str) -> None:
        """
        :param path: Dotted location of the offending field, e.g. ``peptides[0].activities[2]``.
        :param field: Name of the field that failed, e.g. ``activity_unit``.
        :param value: The rejected value, kept verbatim so a report can show what was seen.
        :param reason: Human-readable explanation of why it failed.
        """
        self.path, self.field, self.value, self.reason = path, field, value, reason

    def as_dict(self) -> Dict[str, Any]:
        """
        JSON-serializable form, for the revision report.

        :returns: Mapping with ``path``, ``field``, ``value`` and ``reason`` keys.
        """
        return {"path": self.path, "field": self.field,
                "value": self.value, "reason": self.reason}

    def __repr__(self) -> str:
        """:returns: Compact single-line form, for logs and test failure output."""
        return f"<{self.path}.{self.field}={self.value!r}: {self.reason}>"


# --- Field-level checks ------------------------------------------------------------------------

def normalize_sequence(seq: Any) -> Tuple[str, Optional[str]]:
    """
    Returns (canonical_1_letter, reason_if_rejected). "" with no reason means "no sequence given",
    which is allowed -- a name-only peptide is a legitimate record.
    """
    if not seq or not str(seq).strip():
        return "", None
    from sample.sequence import sequence_to_1_letter
    from sample.nlp import is_likely_dna_sequence

    raw = str(seq).strip().upper()
    one = (sequence_to_1_letter(raw) or "").strip().upper()
    if not one:
        return "", "not convertible to one-letter code"
    if not _CANONICAL_SEQUENCE.match(one):
        return "", f"non-canonical residues in {one[:24]!r}"
    if len(one) < MIN_SEQUENCE_LEN:
        return "", f"shorter than {MIN_SEQUENCE_LEN} residues"
    if len(one) > MAX_SEQUENCE_LEN:
        return "", f"longer than {MAX_SEQUENCE_LEN} residues"
    # A run of ACGT long enough to be meaningful is a nucleotide sequence that happens to spell
    # valid amino-acid letters; publishing it as a peptide is the error this catches.
    try:
        if is_likely_dna_sequence(one):
            return "", "scores as a nucleotide sequence"
    except Exception:
        pass
    try:
        from sample.rf_filter import filter_sequence
        if not filter_sequence(one):
            return "", "rejected by RF peptide/English-word filter"
    except Exception:
        pass
    return one, None


def normalize_unit(unit: Any) -> Tuple[str, Optional[str]]:
    """
    Canonicalizes a unit string.

    :param unit: Unit as extracted, in any spelling.
    :returns: ``(canonical_unit, reason_if_rejected)``. An EMPTY unit is returned as valid, because
        a dimensionless assay (TI, SI, FIC) genuinely has none; whether the empty unit is
        acceptable for a given assay is decided per-assay by :func:`units_for_activity`, which
        still rejects a MIC that arrives without one.
    """
    if unit is None or not str(unit).strip():
        return "", None
    from sample.preprocessing import canonicalize_unit
    canon = canonicalize_unit(str(unit).strip())
    if canon not in VALID_UNITS:
        return "", f"{unit!r} is not a canonical unit"
    return canon, None


def normalize_organism(name: Any) -> Tuple[str, Optional[str]]:
    """
    Returns (canonical_scientific_name, reason_if_rejected). "" with no reason means "not stated".

    The name must be one NCBI knows AND must already be its canonical scientific name -- a common
    name or abbreviation that merely resolves is rewritten to the canonical form, so the published
    column is joinable against the taxonomy directly.
    """
    if not name or not str(name).strip():
        return "", None
    from sample.nlp import resolve_scientific_name
    raw = " ".join(str(name).split())
    resolved = resolve_scientific_name(raw)
    if not resolved:
        return "", f"{raw!r} is not a name in the NCBI taxonomy"
    return resolved, None


def normalize_cell_line(name: Any) -> Tuple[str, Optional[str]]:
    """Returns (cell_line, reason_if_rejected). Must be known to Cellosaurus."""
    if not name or not str(name).strip():
        return "", None
    raw = " ".join(str(name).split())
    try:
        from sample.nlp import get_cellosaurus_kp
        if get_cellosaurus_kp().extract_keywords(raw):
            return raw, None
    except Exception:
        return raw, None       # resource unavailable: do not discard data over it
    return "", f"{raw!r} is not a known cell line"


def normalize_value(value: Any, unit: str, activity_type: str = "") -> Tuple[Optional[float], Optional[str]]:
    """Returns (float, reason_if_rejected)."""
    if value is None or value == "":
        return None, "missing"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None, f"{value!r} is not numeric"
    if not math.isfinite(num):
        return None, "not a finite number"
    if num < 0:
        return None, "negative"
    # A zero concentration is not a measurement. Zero is meaningful only for the readouts that are
    # genuinely reported as a proportion ("0% hemolysis" is a real and useful result).
    if num == 0 and unit != "%":
        return None, "zero"
    bounds = value_range_for_unit(unit)
    if bounds is not None:
        low, high = bounds
        if not (low <= num <= high):
            return None, f"{num:g} outside the plausible range for {unit}"
    return num, None


# --- Record-level enforcement ------------------------------------------------------------------

def enforce_activity(act: Any, path: str = "activity",
                     require_target: bool = False) -> Tuple[Optional[Dict[str, Any]], List["Violation"]]:
    """
    Returns (clean_activity_or_None, [Violation, ...]).

    None means the activity cannot be published: without a recognized assay type, a canonical unit
    and a usable value there is no measurement to report.

    A target is NOT required by default, and the distinction matters. This gate enforces FORMAT --
    a field that is present must be well formed, and a malformed one is removed. An absent target is
    not malformed, it is unstated, and "MIC = 4 uM" remains a real, correctly-typed measurement
    without one. Requiring it dropped 614 of 827 rejected activities on a 250-document sample, the
    large majority of everything this gate removed, none of it for a format problem. Callers that
    genuinely need a target -- an organism-stratified analysis, say -- can ask for it here rather
    than having that policy baked into the published database.
    """
    violations = []
    if not isinstance(act, dict):
        return None, [Violation(path, "activity", act, "not an object")]

    out = dict(act)

    atype = str(out.get("activity_type") or "").strip()
    if atype not in VALID_ACTIVITY_TYPES:
        return None, [Violation(path, "activity_type", atype, "not a recognized assay type")]

    unit, reason = normalize_unit(out.get("activity_unit"))
    if reason:
        return None, [Violation(path, "activity_unit", out.get("activity_unit"), reason)]
    out["activity_unit"] = unit

    raw_unit, raw_reason = normalize_unit(out.get("value_raw_unit") or unit)
    out["value_raw_unit"] = unit if raw_reason else raw_unit

    allowed_units = units_for_activity(atype)
    if unit not in allowed_units:
        return None, [Violation(path, "activity_unit", unit,
                                f"{atype} cannot be expressed in {unit}")]

    value, reason = normalize_value(out.get("value_lower"), unit, atype)
    if reason:
        return None, [Violation(path, "value_lower", out.get("value_lower"), reason)]
    out["value_lower"] = value

    # The upper bound went through no checks at all: not normalized, not range-checked, not
    # ordered against the lower bound, so "MIC 64-2 uM" published unchanged. A transposed range is
    # the commonest form of this and is repaired rather than dropped -- both numbers are real, only
    # their order is wrong -- while an unusable upper bound is cleared and the measurement kept,
    # since a lower bound alone is still a measurement.
    if out.get("value_upper") is not None and str(out.get("value_upper")).strip() != "":
        upper, upper_reason = normalize_value(out.get("value_upper"), unit, atype)
        if upper_reason:
            violations.append(
                Violation(path, "value_upper", out.get("value_upper"), upper_reason))
            out["value_upper"] = None
        else:
            out["value_upper"] = upper
            if value is not None and upper is not None and upper < value:
                violations.append(Violation(
                    path, "value_upper", upper,
                    f"range runs backwards ({value} to {upper}); bounds transposed"))
                out["value_lower"], out["value_upper"] = upper, value
                value = upper

    operator = str(out.get("value_operator") or "").strip()
    if operator not in VALID_OPERATORS:
        violations.append(Violation(path, "value_operator", out.get("value_operator"),
                                    "unrecognized operator"))
        operator = ""
    # Always written back, so the field is a string on every record. Left unset it stayed None on
    # the records that had no operator and "" on the ones that were corrected -- two spellings of
    # "no operator" in the same column, which a consumer has to special-case.
    out["value_operator"] = operator

    organism, reason = normalize_organism(out.get("target_organism"))
    if reason:
        violations.append(Violation(path, "target_organism", out.get("target_organism"), reason))
    out["target_organism"] = organism

    cell_line, reason = normalize_cell_line(out.get("target_cell_line"))
    if reason:
        violations.append(Violation(path, "target_cell_line", out.get("target_cell_line"), reason))
    out["target_cell_line"] = cell_line

    group = str(out.get("target_group") or "").strip()
    # An assay that permits exactly one class implies it, so the class is corrected rather than
    # disputed -- see validation_rules.derivable_target_group for why hemolysis needs this.
    implied = derivable_target_group(atype)
    if implied and group != implied:
        group = implied
    if group not in VALID_TARGET_GROUPS:
        violations.append(Violation(path, "target_group", group, "not a recognized activity class"))
        group = "other"

    # ACTIVITY_RULES states which classes each assay may carry, and that was never checked: a MIC
    # filed as `anticancer` published unchanged, which is the same mis-pairing that put bacteria in
    # the anticancer bucket downstream. The class is re-derived from the organism where possible --
    # a MIC against E. coli is antibacterial whatever the record claimed -- and falls back to the
    # unrestricted `other` rather than to a guess.
    allowed_groups = groups_for_activity(atype)
    if allowed_groups and group not in allowed_groups:
        corrected = ""
        if organism:
            try:
                from sample.nlp import infer_activity_group
                inferred = infer_activity_group(organism, atype)
                if inferred in allowed_groups:
                    corrected = inferred
            except Exception:
                corrected = ""
        violations.append(Violation(
            path, "target_group", group,
            f"{atype} cannot be filed as {group}"))
        group = corrected or "other"
    # An organism that cannot be the target of the class it was filed under is a mis-pairing. The
    # class is kept (the assay really was run); the organism is what is wrong.
    if organism:
        try:
            from sample.nlp import organism_contradicts_group
            if organism_contradicts_group(organism, group):
                violations.append(Violation(path, "target_organism", organism,
                                            f"not a plausible target for {group}"))
                out["target_organism"] = organism = ""
        except Exception:
            pass
    out["target_group"] = group

    # ACTIVITY_RULES also states WHAT each assay runs on. An assay defined against whole organisms
    # cannot report a cell line: "MIC against MCF-7" means the assay name and the target came from
    # different parts of the source. The cell line is cleared and the measurement kept, mirroring
    # how a contradicting organism is handled above -- a MIC with an unstated target is incomplete,
    # which is publishable, whereas a MIC against a cancer cell line is a false statement.
    #
    # Only this direction is enforced. The mirror case (a cells-based assay naming an organism) is
    # routinely legitimate -- hemolysis records name the species the erythrocytes came from -- so
    # clearing it would delete real provenance.
    if target_kind_for_activity(atype) == "organism" and cell_line:
        violations.append(Violation(
            path, "target_cell_line", cell_line,
            f"{atype} is measured against an organism, not a cell line"))
        out["target_cell_line"] = cell_line = ""

    if require_target and not organism and not cell_line:
        return None, violations + [Violation(path, "target", "", "no target organism or cell line")]

    return out, violations


def enforce_peptide(pep: Any, path: str = "peptide") -> Tuple[Optional[Dict[str, Any]], List["Violation"]]:
    """
    Returns (clean_peptide_or_None, [Violation, ...]).

    None means the record cannot be published: it has neither a usable identity (a canonical
    sequence or a valid name) nor a single activity that survived the contract.
    """
    violations = []
    if not isinstance(pep, dict):
        return None, [Violation(path, "peptide", None, "not an object")]

    out = dict(pep)

    sequence, reason = normalize_sequence(out.get("peptide_sequence"))
    if reason:
        violations.append(Violation(path, "peptide_sequence", out.get("peptide_sequence"), reason))
    out["peptide_sequence"] = sequence

    name = str(out.get("peptide_name") or "").strip()
    if name:
        try:
            from sample.nlp import is_valid_peptide_name
            if not is_valid_peptide_name(name):
                violations.append(Violation(path, "peptide_name", name, "not a valid peptide name"))
                name = ""
        except Exception:
            pass
    out["peptide_name"] = name

    if not sequence and not name:
        return None, violations + [Violation(path, "identity", "", "no sequence and no valid name")]

    # Derived representations are regenerated rather than trusted: they are only correct if they
    # were produced from the sequence this record ends up publishing, which is not the sequence the
    # upstream stage saw if the gate above rewrote or cleared it.
    if sequence:
        from sample.sequence import sequence_to_3_letter, sequence_to_helm
        out["peptide_sequence_1letter"] = sequence
        try:
            out["peptide_sequence_3letter"] = sequence_to_3_letter(sequence)
            out["peptide_sequence_helm"] = sequence_to_helm(sequence)
        except Exception:
            out.pop("peptide_sequence_3letter", None)
            out.pop("peptide_sequence_helm", None)
    else:
        for derived in ("peptide_sequence_1letter", "peptide_sequence_3letter",
                        "peptide_sequence_helm", "peptide_sequence_SMILES_format"):
            out.pop(derived, None)

    clean_acts = []
    for i, act in enumerate(out.get("biological_activities") or []):
        clean, act_violations = enforce_activity(act, path=f"{path}.activities[{i}]")
        violations.extend(act_violations)
        if clean is not None:
            clean_acts.append(clean)
    out["biological_activities"] = clean_acts

    if not clean_acts:
        return None, violations + [Violation(path, "biological_activities", [],
                                             "no activity survived the contract")]
    return out, violations


def enforce_document(doc: Any) -> Tuple[Any, List["Violation"]]:
    """
    Applies the contract to one extraction document.

    Returns (clean_document, [Violation, ...]). The document is always returned -- with an empty
    peptide list if nothing survived -- so callers can distinguish "checked, nothing valid" from
    "not checked".
    """
    violations = []
    if isinstance(doc, list):
        peptides, envelope = doc, None
    elif isinstance(doc, dict):
        peptides, envelope = doc.get("peptides") or [], doc
    else:
        return doc, [Violation("document", "root", None, "unrecognized document shape")]

    clean = []
    for i, pep in enumerate(peptides):
        cleaned, pep_violations = enforce_peptide(pep, path=f"peptides[{i}]")
        violations.extend(pep_violations)
        if cleaned is not None:
            clean.append(cleaned)

    if envelope is None:
        return clean, violations
    out = dict(envelope)
    out["peptides"] = clean
    return out, violations


# --- Cross-field coherence ---------------------------------------------------------------------
#
# The checks above validate each field in isolation: is this a real unit, a real organism, a
# number in range. A record can pass all of them and still be nonsense as a MEASUREMENT, because
# the fields contradict each other -- an anticancer peptide whose target is a bacterium, a
# hemolysis assay reported in millimetres, an MIC against a cancer cell line. Each field is
# individually well-formed; the combination is not a thing anyone measured.
#
# These are separated from enforce_activity on purpose. A field-level violation says which value
# is wrong, so the gate can clear that field. A coherence violation says only that two fields
# disagree, and NOT which of them is at fault -- so the deterministic layer flags it and the LLM
# is asked to read the source text and say which one the paper actually supports.

# All three tables the coherence checks consult live in sample/validation_rules.py:
# TARGET_GROUP_DOMAINS (which domains a class may target), ACTIVITY_RULES.quantities (which units
# an assay may use) and ACTIVITY_RULES.target_kind (organism vs cultured cells).


def check_activity_coherence(act: Dict[str, Any], path: str = "activity") -> List["Violation"]:
    """
    Finds fields that are each individually valid but contradict one another.

    Reports rather than repairs. Which field is wrong cannot be decided from the record alone: for
    an anticancer record targeting *Escherichia coli*, either the class or the organism is the
    error, and only the source text says which. :mod:`sample.llm_validator` passes each finding to
    the LLM with the paragraph it came from.

    :param act: A contract-valid activity record.
    :param path: Location label used in the returned violations.
    :returns: One :class:`Violation` per contradiction found; empty when the record is coherent.
        ``field`` names the pair in conflict, e.g. ``"target_group/target_organism"``.
    """
    problems: List["Violation"] = []
    metric = str(act.get("activity_type") or "").strip()
    unit = str(act.get("activity_unit") or "").strip()
    group = str(act.get("target_group") or "").strip()
    organism = str(act.get("target_organism") or "").strip()
    cell_line = str(act.get("target_cell_line") or "").strip()

    # 1. Activity class vs the domain of the organism it was tested against.
    expected = DOMAINS_BY_GROUP.get(group) or ()
    if expected and organism:
        try:
            from sample.nlp import classify_organism_domain
            domain = classify_organism_domain(organism)
        except Exception:
            domain = ""
        # An unclassifiable organism is not evidence of a contradiction, so it is left alone.
        if domain and domain not in expected:
            problems.append(Violation(
                path, "target_group/target_organism", f"{group} vs {organism} ({domain})",
                f"a {group} assay cannot target a {domain}"))

    # 2. Assay vs unit.
    rule = RULE_BY_ACTIVITY.get(metric)
    if rule is not None and unit not in units_for_activity(metric):
        problems.append(Violation(
            path, "activity_type/activity_unit", f"{metric} in {unit}",
            f"{metric} is not measured in {unit}"))

    # 2b. Assay vs activity class, where the table restricts it.
    if rule is not None and rule.target_groups and group and group not in rule.target_groups:
        problems.append(Violation(
            path, "activity_type/target_group", f"{metric} as {group}",
            f"a {metric} result is not a {group} measurement"))

    # 3. Assay vs the KIND of target. A cell line is not an organism a MIC can be run against, and
    #    a bacterium is not something a hemolysis assay is performed on.
    target_kind = rule.target_kind if rule is not None else "either"
    if target_kind == "organism" and cell_line and not organism:
        problems.append(Violation(
            path, "activity_type/target_cell_line", f"{metric} vs {cell_line}",
            f"{metric} is measured against an organism, not a cell line"))
    if target_kind == "cells" and organism:
        try:
            from sample.nlp import classify_organism_domain
            domain = classify_organism_domain(organism)
        except Exception:
            domain = ""
        if domain and domain not in ("animal",):
            problems.append(Violation(
                path, "activity_type/target_organism", f"{metric} vs {organism} ({domain})",
                f"{metric} is measured on cells, not on a {domain}"))

    return problems


def check_peptide_coherence(pep: Dict[str, Any], path: str = "peptide") -> List["Violation"]:
    """
    Runs :func:`check_activity_coherence` over every activity of one peptide record.

    :param pep: A contract-valid peptide record.
    :param path: Location label used in the returned violations.
    :returns: Every contradiction found across the record's activities.
    """
    problems: List["Violation"] = []
    for i, act in enumerate(pep.get("biological_activities") or []):
        if isinstance(act, dict):
            problems.extend(check_activity_coherence(act, path=f"{path}.activities[{i}]"))
    return problems
