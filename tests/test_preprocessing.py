"""
Tests for :mod:`sample.preprocessing` — unit canonicalization and molar conversion.

The value/unit pair is the part of a record most easily corrupted, because the conversion and the
label are written in two places. These tests pin the invariant that matters: after
standardization, ``value_lower`` and ``activity_unit`` describe the same quantity, and
``value_raw``/``value_raw_unit`` still describe what the paper actually said.
"""
from typing import Any, Dict, List

import pytest

from sample import preprocessing as P


class TestCanonicalizeUnit:
    @pytest.mark.parametrize("raw", ["ug/ml", "UG/ML", "µg/ml", "μg/mL", "ug/mL"])
    def test_micro_spellings_collapse(self, raw: str) -> None:
        # U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU both appear in real papers and are
        # different codepoints; both must land on the same canonical string.
        assert P.canonicalize_unit(raw) == "ug/mL"

    @pytest.mark.parametrize("raw,expected", [
        ("MG/L", "mg/L"), ("mg/l", "mg/L"), ("MG/ML", "ug/mL"),
    ])
    def test_denominator_case_is_folded_for_slash_units(self, raw: str,
                                                        expected: str) -> None:
        assert P.canonicalize_unit(raw) == expected

    @pytest.mark.parametrize("unit", ["uM", "mM", "nM", "pM", "M", "mm", "cm", "%", "IU", "U"])
    def test_canonical_scalar_units_are_stable(self, unit: str) -> None:
        assert P.canonicalize_unit(unit) == unit

    @pytest.mark.parametrize("raw", ["um", "UM", "Um", "NM", "nm"])
    def test_scalar_case_variants_are_deliberately_not_folded(self, raw: str) -> None:
        # This looks like a gap and is not. Scalar units that differ only in case denote
        # different physical quantities: mm/mM is millimetre vs millimolar, nm/nM nanometre vs
        # nanomolar, um/uM micrometre vs micromolar. Case-folding these would merge a length into
        # a concentration. Unrecognised spellings pass through untouched and are rejected by the
        # contract instead, which loses a record rather than inventing a wrong one.
        assert P.canonicalize_unit(raw) == raw

    def test_ocr_repair_runs_before_case_folding(self) -> None:
        # "MM" is an OCR'd micro sign, repaired explicitly by the case-SENSITIVE table, while
        # "mM" is a genuine millimolar and must survive. The two differ by a factor of 1000.
        assert P.canonicalize_unit("MM") == "uM"
        assert P.canonicalize_unit("mM") == "mM"

    def test_unknown_unit_passes_through_unchanged(self) -> None:
        # Refusing to guess. An unrecognised unit is left intact for the contract to reject, not
        # silently rewritten into something plausible.
        assert P.canonicalize_unit("furlongs") == "furlongs"

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_input_is_returned_unchanged(self, value: Any) -> None:
        assert P.canonicalize_unit(value) == value

    def test_is_idempotent(self) -> None:
        for unit in ("ug/mL", "mg/L", "uM", "nM", "%"):
            assert P.canonicalize_unit(P.canonicalize_unit(unit)) == unit


class TestRawValueAndUnit:
    def test_returns_the_reported_pair(self) -> None:
        act = {"activity_unit": "mM", "value_lower": 0.5, "value_raw": "0.5",
               "activity_type": "MIC"}
        assert P.raw_value_and_unit(act) == ("0.5", "mM")

    def test_missing_raw_value_stays_empty(self) -> None:
        # No fallback to the parsed value on purpose: value_raw is the audit trail back to the
        # source text, and synthesising one from the parsed number would fabricate a quotation.
        act = {"activity_unit": "uM", "value_lower": 4.0, "activity_type": "MIC"}
        raw, unit = P.raw_value_and_unit(act)
        assert unit == "uM"
        assert raw == ""

    def test_empty_activity_does_not_raise(self) -> None:
        assert isinstance(P.raw_value_and_unit({}), tuple)


class TestStandardizeAndFilterActivities:
    @pytest.fixture
    def millimolar(self) -> Dict[str, Any]:
        """:returns: A MIC reported in mM, which standardization converts to uM."""
        return {"activity_unit": "mM", "value_lower": 0.5, "value_raw": "0.5",
                "activity_type": "MIC"}

    def test_converts_molar_units_to_micromolar(self, millimolar: Dict[str, Any]) -> None:
        out = P.standardize_and_filter_activities([millimolar])[0]
        assert out["activity_unit"] == "uM"
        assert out["value_lower"] == pytest.approx(500.0)

    def test_value_and_unit_stay_coherent_after_conversion(self,
                                                           millimolar: Dict[str, Any]) -> None:
        # The defect this guards against: converting the value but leaving the old unit label,
        # which yields a record that is individually well-formed and factually wrong.
        out = P.standardize_and_filter_activities([millimolar])[0]
        assert (out["value_lower"], out["activity_unit"]) == (pytest.approx(500.0), "uM")

    def test_preserves_what_the_paper_actually_said(self,
                                                    millimolar: Dict[str, Any]) -> None:
        # value_raw / value_raw_unit are the audit trail back to the source text and must NOT be
        # rewritten by the conversion.
        out = P.standardize_and_filter_activities([millimolar])[0]
        assert out["value_raw"] == "0.5"
        assert out["value_raw_unit"] == "mM"

    def test_records_raw_unit_even_when_no_conversion_applies(self) -> None:
        out = P.standardize_and_filter_activities(
            [{"activity_unit": "furlongs", "value_lower": 1, "activity_type": "MIC"}])[0]
        assert out["value_raw_unit"] == "furlongs"

    def test_empty_input_yields_empty_output(self) -> None:
        assert P.standardize_and_filter_activities([]) == []

    def test_updates_the_activity_dicts_in_place(self, millimolar: Dict[str, Any]) -> None:
        # Documenting observed behaviour rather than asserting a preference: the function edits
        # the dicts it is given as well as returning a list. A caller holding the original for
        # comparison sees it change underneath. Pinned here so a future change to it is a
        # deliberate decision rather than an accident.
        P.standardize_and_filter_activities([millimolar])
        assert millimolar["activity_unit"] == "uM"
        assert millimolar["value_raw_unit"] == "mM"

    def test_is_idempotent(self, millimolar: Dict[str, Any]) -> None:
        # Re-running the pipeline over already-standardized records must not convert twice --
        # 0.5 mM becoming 500 uM becoming 500000 is exactly the kind of silent corruption a
        # resumed run can introduce.
        once = P.standardize_and_filter_activities([millimolar])
        twice = P.standardize_and_filter_activities(once)
        assert twice[0]["value_lower"] == pytest.approx(once[0]["value_lower"])
        assert twice[0]["activity_unit"] == once[0]["activity_unit"]

    def test_handles_records_missing_a_value(self) -> None:
        out = P.standardize_and_filter_activities([{"activity_type": "MIC",
                                                    "activity_unit": "uM"}])
        assert isinstance(out, list)


class TestPotencyUnitGuards:
    """
    Units a potency metric cannot legitimately carry.

    Every case here was produced by the docling table backend in a real run, which bypassed
    standardize_and_filter_activities entirely -- so these guards existed but never ran on 44% of
    the corpus. See pipeline_runner._table_peptides.
    """

    @staticmethod
    def _mic(unit, value=1.0, atype="MIC"):
        return [{"activity_type": atype, "value_raw": str(value),
                 "value_lower": value, "activity_unit": unit}]

    def test_a_percentage_is_not_a_concentration(self) -> None:
        # Measured: a percent-killing table ("20.2 +/- 15.5" for hBD-2) emitted with
        # activity_type MIC because the header carried the assay name.
        assert P.standardize_and_filter_activities(self._mic("%", 20.2)) == []

    def test_molar_is_converted_then_rejected_as_implausible(self) -> None:
        # Measured: an "Arg" row at 0.1-5 M, read out of a buffer-composition table. Without the
        # M->uM conversion this was invisible to the magnitude guard, because _MOLAR_UNITS holds
        # only micromolar spellings.
        for value in (0.1, 5.0):
            assert P.standardize_and_filter_activities(self._mic("M", value)) == []

    def test_a_zone_diameter_is_not_a_concentration(self) -> None:
        assert P.standardize_and_filter_activities(self._mic("mm", 14.0)) == []

    def test_legitimate_concentration_units_survive(self) -> None:
        for unit, value in (("ug/mL", 4.0), ("uM", 8.0), ("nM", 50.0)):
            kept = P.standardize_and_filter_activities(self._mic(unit, value))
            assert kept, f"{unit} should be kept"
            assert kept[0]["value_lower"] == value

    def test_millimolar_still_converts_to_micromolar(self) -> None:
        kept = P.standardize_and_filter_activities(self._mic("mM", 0.3))
        assert kept[0]["value_lower"] == 300.0
        assert kept[0]["activity_unit"] == "uM"
        # The source unit is preserved so a reader can check the printed value.
        assert kept[0]["value_raw_unit"] == "mM"

    def test_percent_is_only_rejected_for_POTENCY_metrics(self) -> None:
        # Hemolysis is legitimately reported as a percentage and must not be swept up.
        kept = P.standardize_and_filter_activities(self._mic("%", 12.0, atype="Hemolysis"))
        assert kept and kept[0]["activity_unit"] == "%"

    def test_a_mass_unit_potency_is_kept(self) -> None:
        # Documented limitation, pinned so a future change is deliberate: mass units need a
        # molecular weight to convert, so the plausibility bound cannot see them. Polymyxin B at
        # 1.56 mg/mL (1000x its real MIC) survives, and so does a genuinely high-concentration
        # result like an essential oil at 200 mg/mL. Dropping both would lose real data.
        assert P.standardize_and_filter_activities(self._mic("mg/mL", 1.56))


class TestMolarUnitsAreCaseInsensitive:
    """
    A lowercase "m" must be treated as molar, exactly as ``concentration._MOLAR_UNIT_TO_UM`` does.

    That table reaches its ``"m"`` entry case-insensitively (it folds the unit first), while this
    stage matched only an uppercase ``"M"``. So a lowercase ``"m"`` was converted at 1e6 by the one
    and left unconverted by the other -- and since ``_MOLAR_UNITS`` holds only micromolar
    spellings, the unconverted unit then bypassed ``_MAX_PLAUSIBLE_POTENCY_UM`` altogether and was
    published.
    """

    def _run(self, unit, value="0.1", metric="MIC"):
        return P.standardize_and_filter_activities(
            [{"activity_type": metric, "value_lower": value, "value_raw": value,
              "activity_unit": unit, "target_organism": "E. coli"}])

    def test_uppercase_molar_is_dropped_as_implausible(self) -> None:
        # 0.1 M is 100,000 uM, an order of magnitude past _MAX_PLAUSIBLE_POTENCY_UM. A peptide MIC
        # is never molar; the measured case was a buffer-composition row read as an assay.
        assert self._run("M") == []

    def test_lowercase_molar_is_dropped_too(self) -> None:
        assert self._run("m") == []

    def test_surrounding_whitespace_does_not_defeat_it(self) -> None:
        assert self._run(" m ") == []

    def test_millimolar_still_converts_rather_than_being_dropped(self) -> None:
        # The conversion is correct and must stay: 0.1 mM really is 100 uM.
        out = self._run("mM")
        assert len(out) == 1
        assert out[0]["value_lower"] == 100.0
        assert out[0]["activity_unit"] == "uM"

    def test_a_genuine_micromolar_value_is_untouched(self) -> None:
        out = self._run("uM", "5")
        assert len(out) == 1 and out[0]["value_lower"] == 5.0

    def test_lowercase_mm_is_still_a_length_and_dropped(self) -> None:
        # _NON_CONCENTRATION_UNITS drops "mm" before the molar branch; case-folding the molar test
        # must not have turned a millimetre into a millimolar.
        assert self._run("mm") == []
