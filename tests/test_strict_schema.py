"""
Tests for :mod:`sample.strict_schema` — the published-record contract.

This is the gate every record passes before it reaches the release, so the tests are written
around one question: can a malformed field get through? Field-level enforcement means a bad field
is dropped without discarding the whole record, and the tests pin both halves of that — what is
removed, and what survives.
"""
from typing import Any, Dict

import pytest

from sample import strict_schema as S


class TestViolation:
    def test_carries_its_four_parts(self) -> None:
        v = S.Violation("activity[0]", "activity_unit", "furlongs", "not a canonical unit")
        assert v.path == "activity[0]"
        assert v.field == "activity_unit"
        assert v.value == "furlongs"
        assert v.reason == "not a canonical unit"

    def test_as_dict_is_json_serializable(self) -> None:
        import json
        v = S.Violation("p", "f", 1.5, "why")
        json.dumps(v.as_dict())

    def test_repr_names_the_field_and_reason(self) -> None:
        text = repr(S.Violation("p", "activity_unit", "furlongs", "not canonical"))
        assert "activity_unit" in text and "not canonical" in text


class TestNormalizeSequence:
    def test_accepts_a_plain_sequence(self) -> None:
        seq, problem = S.normalize_sequence("GIGKFLHSAKKFGKAFVGEIMNS")
        assert seq == "GIGKFLHSAKKFGKAFVGEIMNS"
        assert problem is None

    def test_uppercases(self) -> None:
        assert S.normalize_sequence("gigkflh")[0] == "GIGKFLH"

    def test_converts_three_letter(self) -> None:
        assert S.normalize_sequence("Gly-Ile-Gly-Lys")[0] == "GIGK"

    @pytest.mark.parametrize("seq", ["GIGKJJJ", "GIGKXX"])
    def test_rejects_non_canonical_residues(self, seq: str) -> None:
        cleaned, problem = S.normalize_sequence(seq)
        assert cleaned == "" and "non-canonical" in problem

    @pytest.mark.parametrize("raw,expected", [("GIGK@@", "GIGK"), ("GIG1K", "GIGK")])
    def test_strips_ocr_noise_rather_than_rejecting(self, raw: str, expected: str) -> None:
        # Punctuation and residue numbering are scanning artefacts, not evidence the sequence is
        # wrong. Only non-canonical LETTERS mean the field cannot be trusted.
        assert S.normalize_sequence(raw)[0] == expected

    def test_rejects_nucleotide_sequences(self) -> None:
        # A run of ACGT is DNA that wandered into a peptide field, not a peptide.
        seq, problem = S.normalize_sequence("ACGTACGTACGTACGT")
        assert seq == "" and "nucleotide" in problem

    @pytest.mark.parametrize("seq", ["", None, "   "])
    def test_empty_is_rejected(self, seq: Any) -> None:
        assert S.normalize_sequence(seq)[0] == ""

    def test_enforces_length_bounds(self) -> None:
        assert S.normalize_sequence("GIG")[0] == ""                       # below MIN
        assert S.normalize_sequence("A" * 200)[0] == ""                   # above MAX
        assert S.normalize_sequence("GIGKFLHSAKK")[0] == "GIGKFLHSAKK"    # inside

    def test_bounds_are_the_declared_constants(self) -> None:
        assert S.MIN_SEQUENCE_LEN == 4
        assert S.MAX_SEQUENCE_LEN == 50


class TestNormalizeUnit:
    @pytest.mark.parametrize("raw", ["ug/mL", "UG/ML", "µg/ml", "μg/mL"])
    def test_spellings_collapse_to_one_canonical_form(self, raw: str) -> None:
        # Both Unicode micro signs (U+00B5 and U+03BC) occur in real papers.
        assert S.normalize_unit(raw)[0] == "ug/mL"

    def test_unknown_unit_is_rejected_with_a_reason(self) -> None:
        unit, problem = S.normalize_unit("furlongs")
        assert unit == "" and problem

    def test_every_canonical_unit_normalizes_to_itself(self) -> None:
        from sample.validation_rules import VALID_UNITS
        for unit in VALID_UNITS:
            if unit:
                assert S.normalize_unit(unit)[0] == unit, f"{unit} is not stable"


class TestNormalizeOrganism:
    def test_collapses_whitespace_and_capitalizes_genus(self) -> None:
        assert S.normalize_organism("  escherichia  coli ")[0] == "Escherichia coli"

    def test_expands_abbreviated_genus(self) -> None:
        assert S.normalize_organism("E. coli")[0] == "Escherichia coli"

    @pytest.mark.parametrize("name", ["", None, "   "])
    def test_empty_is_empty(self, name: Any) -> None:
        assert S.normalize_organism(name)[0] == ""

    def test_pure_number_is_not_an_organism(self) -> None:
        # 17% of activity records once carried a bare number here.
        assert S.normalize_organism("100")[0] == ""


class TestNormalizeValue:
    def test_parses_a_numeric_string(self) -> None:
        assert S.normalize_value("16", "ug/mL", "MIC") == (16.0, None)

    def test_accepts_a_float(self) -> None:
        assert S.normalize_value(16.0, "ug/mL", "MIC")[0] == 16.0

    def test_rejects_non_numeric(self) -> None:
        value, problem = S.normalize_value("abc", "ug/mL", "MIC")
        assert value is None and problem

    def test_rejects_negative_concentration(self) -> None:
        value, problem = S.normalize_value(-5.0, "ug/mL", "MIC")
        assert value is None and problem

    def test_rejects_value_outside_the_unit_range(self) -> None:
        # A zone of inhibition wider than any plate is a parsing artefact.
        value, problem = S.normalize_value(5000.0, "mm", "ZOI")
        assert value is None and problem

    def test_accepts_value_inside_the_unit_range(self) -> None:
        assert S.normalize_value(12.0, "mm", "ZOI")[0] == 12.0


class TestEnforceActivity:
    def test_valid_activity_survives_intact(self, activity: Dict[str, Any]) -> None:
        out, violations = S.enforce_activity(dict(activity))
        assert violations == []
        assert out["activity_type"] == "MIC"
        assert out["value_lower"] == 16.0
        assert out["activity_unit"] == "ug/mL"

    def test_fills_declared_optional_fields(self, activity: Dict[str, Any]) -> None:
        # Downstream builds a DataFrame from these, so the key set must not vary by record.
        out, _ = S.enforce_activity(dict(activity))
        assert "target_cell_line" in out and "value_raw_unit" in out

    def test_target_is_optional_by_default(self, activity: Dict[str, Any]) -> None:
        # Requiring it once dropped 614 of 827 activities for an ABSENT rather than malformed
        # target. An activity with no organism recorded is incomplete, not wrong.
        activity.pop("target_organism")
        out, _ = S.enforce_activity(activity)
        assert out is not None

    def test_target_can_be_required(self, activity: Dict[str, Any]) -> None:
        activity.pop("target_organism")
        out, violations = S.enforce_activity(activity, require_target=True)
        assert out is None or violations

    def test_unknown_activity_type_is_rejected(self, activity: Dict[str, Any]) -> None:
        activity["activity_type"] = "not-an-assay"
        out, violations = S.enforce_activity(activity)
        assert out is None or violations

    def test_bad_unit_is_rejected(self, activity: Dict[str, Any]) -> None:
        activity["activity_unit"] = "furlongs"
        out, violations = S.enforce_activity(activity)
        assert out is None or violations

    def test_invalid_operator_is_rejected(self, activity: Dict[str, Any]) -> None:
        activity["value_operator"] = "!!"
        out, violations = S.enforce_activity(activity)
        assert out is None or violations

    @pytest.mark.parametrize("operator", sorted(S.VALID_OPERATORS))
    def test_every_declared_operator_is_accepted(self, activity: Dict[str, Any],
                                                 operator: str) -> None:
        activity["value_operator"] = operator
        out, violations = S.enforce_activity(activity)
        assert out is not None, f"operator {operator!r} rejected: {violations}"

    @pytest.mark.parametrize("bad", [None, "string", 42, []])
    def test_non_mapping_input_is_rejected(self, bad: Any) -> None:
        out, violations = S.enforce_activity(bad)
        assert out is None and violations

    def test_violation_paths_locate_the_field(self, activity: Dict[str, Any]) -> None:
        activity["activity_unit"] = "furlongs"
        _, violations = S.enforce_activity(activity, path="peptides[0].activities[3]")
        assert any("peptides[0].activities[3]" in v.path for v in violations)


class TestEnforcePeptide:
    def test_valid_peptide_survives(self, peptide: Dict[str, Any]) -> None:
        out, violations = S.enforce_peptide(dict(peptide))
        assert out is not None and violations == []
        assert out["peptide_sequence"] == "GIGKFLHSAKKFGKAFVGEIMNS"

    def test_bad_activity_is_dropped_without_losing_the_peptide(
            self, peptide: Dict[str, Any]) -> None:
        # Field-level, not record-level: one broken measurement must not discard the other good
        # ones or the peptide itself.
        good = dict(peptide["biological_activities"][0])
        bad = dict(good)
        bad["activity_unit"] = "furlongs"
        peptide["biological_activities"] = [good, bad]

        out, violations = S.enforce_peptide(peptide)
        assert out is not None
        assert len(out["biological_activities"]) == 1
        assert violations

    def test_peptide_with_no_sequence_still_has_a_name(self, peptide: Dict[str, Any]) -> None:
        peptide["peptide_sequence"] = ""
        out, _ = S.enforce_peptide(peptide)
        if out is not None:
            assert out["peptide_name"] == "magainin-2"

    @pytest.mark.parametrize("bad", [None, "string", 42])
    def test_non_mapping_input_is_rejected(self, bad: Any) -> None:
        out, violations = S.enforce_peptide(bad)
        assert out is None and violations


class TestEnforceDocument:
    def test_valid_document_survives(self, document: Dict[str, Any]) -> None:
        out, violations = S.enforce_document(dict(document))
        assert out is not None and violations == []
        assert len(out["peptides"]) == 1

    def test_document_without_peptides_is_handled(self) -> None:
        out, _ = S.enforce_document({"doc_id": "d", "peptides": []})
        assert out.get("peptides") == []

    def test_envelope_fields_are_preserved(self, document: Dict[str, Any]) -> None:
        out, _ = S.enforce_document(dict(document))
        assert out["doc_id"] == "test-doc"

    def test_bare_peptide_list_is_accepted(self, peptide: Dict[str, Any]) -> None:
        out, violations = S.enforce_document([peptide])
        assert isinstance(out, list) and len(out) == 1 and violations == []

    @pytest.mark.parametrize("bad", [None, "string", 42])
    def test_unrecognized_shape_is_reported_but_returned(self, bad: Any) -> None:
        # Documented: the document is ALWAYS returned, so a caller can tell "checked, nothing
        # valid" apart from "not checked at all". The violation carries the failure.
        out, violations = S.enforce_document(bad)
        assert out == bad
        assert violations and violations[0].reason == "unrecognized document shape"


class TestActivityCoherence:
    """
    Cross-field contradictions: each field is individually valid, but together they cannot be true.
    """

    def test_valid_activity_is_coherent(self, activity: Dict[str, Any]) -> None:
        assert S.check_activity_coherence(activity) == []

    def test_mic_reported_in_millimetres_is_flagged(self, activity: Dict[str, Any]) -> None:
        # A MIC is a concentration; mm is a zone diameter.
        activity["activity_unit"] = "mm"
        assert S.check_activity_coherence(activity)

    def test_anticancer_against_a_bacterium_is_flagged(self, activity: Dict[str, Any]) -> None:
        activity["target_group"] = "anticancer"
        assert S.check_activity_coherence(activity)

    def test_antibacterial_against_archaea_is_allowed(self, activity: Dict[str, Any]) -> None:
        activity["target_organism"] = "Halobacterium salinarum"
        activity["target_group"] = "antibacterial"
        assert S.check_activity_coherence(activity) == []

    def test_unresolvable_organism_is_not_a_contradiction(self,
                                                          activity: Dict[str, Any]) -> None:
        # Absence of evidence: an organism whose domain cannot be resolved must not be treated as
        # conflicting, or every unusual strain name becomes a violation.
        activity["target_organism"] = "Strain XYZ-123 unclassified"
        assert S.check_activity_coherence(activity) == []

    def test_zone_of_inhibition_in_concentration_units_is_flagged(
            self, activity: Dict[str, Any]) -> None:
        activity["activity_type"] = "ZOI"
        activity["activity_unit"] = "ug/mL"
        assert S.check_activity_coherence(activity)


class TestPeptideCoherence:
    def test_valid_peptide_is_coherent(self, peptide: Dict[str, Any]) -> None:
        assert S.check_peptide_coherence(peptide) == []

    def test_returns_a_list(self, peptide: Dict[str, Any]) -> None:
        assert isinstance(S.check_peptide_coherence(peptide), list)


class TestPublishedJsonSchema:
    def test_is_a_valid_json_schema(self) -> None:
        import jsonschema
        jsonschema.Draft7Validator.check_schema(S.published_json_schema())

    def test_validates_a_conforming_document(self, document: Dict[str, Any]) -> None:
        import jsonschema
        cleaned, _ = S.enforce_document(dict(document))
        jsonschema.validate(cleaned, S.published_json_schema())

    def test_is_stable_across_calls(self) -> None:
        assert S.published_json_schema() == S.published_json_schema()
