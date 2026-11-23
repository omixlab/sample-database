"""
Three defects in extracted activity fields, each measured on real output before being fixed.

All three were found by reading the `[activity]` log of a live step 9 run:

  * 1,098 of 52,713 units (2.1%) carried U+FFFD where PDF extraction lost a character.
  * 58 `activity_type` values were prose describing an assay rather than naming one.
  * 2,312 `value_raw` values held a range whose upper bound was parsed and then discarded.
"""
import pytest

from sample.preprocessing import (canonicalize_unit, standardize_and_filter_activities,
                                  _MAX_ACTIVITY_TYPE_CHARS)
from sample.validation_rules import VALID_ACTIVITY_TYPES

FFFD = "�"


class TestMojibakeUnitRepair:
    """
    U+FFFD is repaired only where micro is the only character it can be.

    PDF extraction emits U+FFFD for several distinct characters. Measured over the 1,098 corrupted
    units on disk, 1,027 (93.5%) sit immediately before a unit token, where the lost character was
    the micro sign; the rest are a multiplication sign, a superscript minus or a Greek letter in a
    cell-type name.
    """

    @pytest.mark.parametrize("corrupt,expected", [
        (FFFD + "M", "uM"),
        (FFFD + " M", "uM"),
        (FFFD + "g/mL", "ug/mL"),
        (FFFD + " g/mL", "ug/mL"),
        ("( " + FFFD + " g/mL )", "ug/mL"),
        (FFFD + "mol/L", "umol/L"),
        (FFFD + "g", "ug"),
    ])
    def test_the_micro_position_is_repaired(self, corrupt, expected):
        assert canonicalize_unit(corrupt) == expected

    @pytest.mark.parametrize("untouched", [
        FFFD + "10" + FFFD + "4",            # a multiplication sign and a superscript minus
        "of circulating CD8" + FFFD + "T cells",   # a Greek letter in a cell-type name
        FFFD + "1",                          # a superscript minus
    ])
    def test_every_other_occurrence_is_left_alone(self, untouched):
        """
        The lookahead is the whole point. A blanket replacement would turn the first of these into
        "u10u4" and invent a unit that was never printed.
        """
        assert FFFD in canonicalize_unit(untouched), (
            "repairing this position would be a guess, not a repair")

    def test_a_second_corruption_in_the_same_string_is_handled_independently(self):
        """`�g ml�1` is a micro sign AND a superscript minus; only the first is known."""
        out = canonicalize_unit(FFFD + "g ml" + FFFD + "1")
        assert out.startswith("ug ml") and out.endswith(FFFD + "1")

    def test_the_existing_ocr_repairs_still_work(self):
        """The mojibake pass runs first, so it must not shadow the case-sensitive repairs."""
        assert canonicalize_unit("Mg/mL") == "ug/mL"
        assert canonicalize_unit("lM") == "uM"
        assert canonicalize_unit("MM") == "uM"

    def test_millimolar_and_millimetre_remain_distinct(self):
        assert canonicalize_unit("mM") == "mM"
        assert canonicalize_unit("mm") == "mm"


class TestProseActivityTypes:
    """
    A sentence is not an activity type. The prose is kept; only the type field is cleared.
    """

    LONG = "hydrophobicity (RP-HPLC retention time tR )"

    def test_the_threshold_clears_every_real_activity_type(self):
        longest = max(len(t) for t in VALID_ACTIVITY_TYPES)
        assert longest < _MAX_ACTIVITY_TYPE_CHARS, (
            f"the longest valid type is {longest} chars; the bound must not clip it")

    def test_prose_is_moved_out_of_the_type_field(self):
        out = standardize_and_filter_activities(
            [{"activity_type": self.LONG, "value_raw": "12.5", "activity_unit": "min"}])
        assert len(out) == 1
        assert out[0]["activity_type"] == ""
        assert out[0]["activity_type_detail"] == self.LONG

    def test_the_measurement_itself_survives(self):
        """Dropping the row would discard a value, a unit and a target that may all be real."""
        out = standardize_and_filter_activities(
            [{"activity_type": self.LONG, "value_raw": "12.5", "activity_unit": "min",
              "target_organism": "Escherichia coli"}])
        assert out[0]["value_lower"] == 12.5
        assert out[0]["target_organism"] == "Escherichia coli"

    def test_a_real_type_is_untouched(self):
        out = standardize_and_filter_activities(
            [{"activity_type": "Membrane Permeabilization", "value_raw": "8",
              "activity_unit": "ug/mL"}])
        assert out[0]["activity_type"] == "Membrane Permeabilization"
        assert "activity_type_detail" not in out[0]


class TestRangeUpperBound:
    """
    The upper bound of a range, in the same unit as the lower one.

    `nlp.parse_activity_value_fast` has returned it all along; this stage discarded it with `_`,
    so 2,312 activities carried a range in `value_raw` with `value_upper` unset.
    """

    @staticmethod
    def _one(value_raw, unit="uM", atype="MIC"):
        out = standardize_and_filter_activities(
            [{"activity_type": atype, "value_raw": value_raw, "activity_unit": unit}])
        return out[0] if out else None

    @pytest.mark.parametrize("raw", ["5-50", "5 - 50", "5–50", "5 to 50"])
    def test_a_range_populates_both_bounds(self, raw):
        act = self._one(raw)
        assert (act["value_lower"], act["value_upper"]) == (5.0, 50.0)

    def test_the_word_form_is_recognised(self):
        """The parser reads "5-50" but not "5 to 50"; the latter is rewritten to the former."""
        assert self._one("5 to 50")["value_upper"] == 50.0

    def test_an_upper_bound_is_scaled_with_the_lower_one(self):
        """
        The trap this guards. value_lower is converted mM -> uM; a bound taken straight from the
        parser would stay millimolar beside a micromolar lower bound -- a 1000x inconsistency, the
        exact failure this module's value_raw_unit comment documents.
        """
        act = self._one("0.3 to 0.5", unit="mM")
        assert (act["value_lower"], act["value_upper"]) == (300.0, 500.0)
        assert act["activity_unit"] == "uM"

    def test_molar_is_scaled_the_same_way(self):
        act = self._one("0.001 to 0.002", unit="M")
        assert (act["value_lower"], act["value_upper"]) == (1000.0, 2000.0)

    def test_up_to_is_not_a_range(self):
        """An upper bound written as prose is one number, and the rewrite is anchored on a digit."""
        act = self._one("up to 50")
        assert act.get("value_upper") is None

    def test_a_single_value_gets_no_bound(self):
        act = self._one("50")
        assert act.get("value_upper") is None

    def test_a_backwards_range_is_not_published_as_one(self):
        act = self._one("64-2")
        assert act.get("value_upper") is None, (
            "an upper bound below the lower one is not asserted here; strict_schema repairs "
            "transposed bounds when they are present")

    def test_a_bound_already_set_is_not_overwritten(self):
        out = standardize_and_filter_activities(
            [{"activity_type": "MIC", "value_raw": "5-50", "activity_unit": "uM",
              "value_upper": 99.0}])
        assert out[0]["value_upper"] == 99.0

    def test_a_recovered_cfu_exponent_carries_no_bound(self):
        """
        The exponent recovery REPLACES the value (106 -> 10^6) rather than scaling it, so there is
        no factor to apply to a bound parsed from the same string.
        """
        out = standardize_and_filter_activities(
            [{"activity_type": "Growth Inhibition", "value_raw": "106",
              "activity_unit": "CFU/ml"}])
        assert out and out[0]["value_lower"] == 1_000_000.0
        assert out[0].get("value_upper") is None


class TestTheThreeRepairsCompose:
    def test_a_mojibake_range_in_millimolar(self):
        """All three defects in one activity, which is how they appeared in the log."""
        out = standardize_and_filter_activities(
            [{"activity_type": "x" * 50, "value_raw": "0.3 to 0.5",
              "activity_unit": FFFD + "M"}])
        act = out[0]
        assert act["activity_unit"] == "uM"
        assert (act["value_lower"], act["value_upper"]) == (0.3, 0.5)
        assert act["activity_type"] == "" and act["activity_type_detail"] == "x" * 50
