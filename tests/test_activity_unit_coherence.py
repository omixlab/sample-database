"""Type/unit coherence for extracted activities (`sample.concentration`)."""
import pytest

from sample.concentration import (activity_unit_is_coherent, CONCENTRATION_ACTIVITY_TYPES,
                                  NON_CONCENTRATION_UNITS)


class TestActivityUnitCoherence:
    """
    A concentration cannot be expressed in daltons.

    This gate exists because of a specific, unrecoverable corruption: 2,852 of 53,107 activities
    on disk recorded a peptide's molecular weight as its MIC -- `brevinin-1RTb MIC = 2548.2 Da`,
    `HG1 MIC = 4113.18 Da` -- read out of a mass-spec table into the wrong column. Both halves
    appear verbatim in the paper, so the text-grounding gate passes them; only the relationship
    between type and unit is wrong.
    """

    @pytest.mark.parametrize("unit,measures", [
        ("Da", "mass"), ("kDa", "mass"), ("g/mol", "mass"), ("amu", "mass"),
        ("min", "time"), ("s", "time"), ("h", "time"), ("hours", "time"),
        ("kcal/mol", "energy"), ("kJ/mol", "energy"),
        ("bp", "count"), ("aa", "count"),
    ])
    def test_a_concentration_type_cannot_carry_a_non_concentration_unit(self, unit, measures):
        ok, reason = activity_unit_is_coherent("MIC", unit)
        assert ok is False
        assert reason == f"MIC_in_{measures}_unit"

    @pytest.mark.parametrize("unit", ["uM", "µM", "μM", "nM", "mM", "pM", "ug/mL", "mg/mL",
                                      "ng/mL", "mol/L"])
    def test_a_real_concentration_unit_passes(self, unit):
        assert activity_unit_is_coherent("MIC", unit)[0] is True

    @pytest.mark.parametrize("kind", ["MIC", "mic", "MIC50", "MBC", "IC50", "EC50", "LD50",
                                      "MFC", "CC50", "GI50"])
    def test_every_concentration_type_is_judged_case_insensitively(self, kind):
        assert activity_unit_is_coherent(kind, "Da")[0] is False

    def test_mm_is_left_alone_because_it_is_ambiguous_with_millimolar(self):
        """
        `mm` and `mM` differ only by case, so any case-folding conflates a zone-of-inhibition
        diameter with a millimolar concentration. Dropping it would risk real data, and no
        concentration type was observed carrying it, so it is not in the reject set.
        """
        assert activity_unit_is_coherent("MIC", "mm")[0] is True
        assert activity_unit_is_coherent("ZOI", "mm")[0] is True

    def test_percent_and_dose_units_are_left_alone(self):
        assert activity_unit_is_coherent("Hemolysis", "%")[0] is True
        assert activity_unit_is_coherent("MIC", "%")[0] is True
        assert activity_unit_is_coherent("MIC", "mg/kg")[0] is True, (
            "an in-vivo dose is a real measurement, not an incoherent one")

    def test_an_unrecognised_activity_type_is_not_judged(self):
        """
        The type vocabulary is open -- the model returns things like "ATP-based assay for amebic
        viability" -- so a strict whitelist on the type side would discard real measurements.
        Only the closed set of concentration types is checked.
        """
        assert activity_unit_is_coherent("ATP-based assay for amebic viability", "Da")[0] is True
        assert activity_unit_is_coherent("Molecular weight", "Da")[0] is True
        assert activity_unit_is_coherent("N/A", "Da")[0] is True

    def test_a_missing_unit_is_not_an_incoherence(self):
        for unit in (None, "", "   "):
            assert activity_unit_is_coherent("MIC", unit)[0] is True

    def test_a_missing_type_is_not_judged(self):
        assert activity_unit_is_coherent(None, "Da")[0] is True
        assert activity_unit_is_coherent("", "Da")[0] is True

    def test_the_two_tables_do_not_overlap(self):
        assert not (CONCENTRATION_ACTIVITY_TYPES & set(NON_CONCENTRATION_UNITS)), (
            "a token cannot be both an activity type and a unit")
