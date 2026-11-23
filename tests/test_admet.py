"""
Tests for :mod:`sample.admet` -- the sequence-only ADMET-lite profile.

The properties under test are the ones that make this an *honest* summary rather than a partial one:
non-standard residues are counted (so the ProtParam metrics are known to be approximate, not wrong),
the standard peptide rules (Guruprasad instability, the N-end rule, the aggregation-hotspot scan)
fire on the right inputs, and the endpoints the framework refuses to fake -- membrane permeability
above all -- are always present in ``gaps`` rather than silently dropped.
"""
import pytest

from sample import admet


class TestAliphaticIndex:
    def test_all_alanine_equals_one_hundred(self) -> None:
        # AI = 100 * (X_A + 2.9 X_V + 3.9(X_I + X_L)); all-Ala -> X_A = 1 -> 100.
        assert admet.aliphatic_index("AAAA") == pytest.approx(100.0)

    def test_leucine_weighs_more_than_alanine(self) -> None:
        assert admet.aliphatic_index("LLLL") > admet.aliphatic_index("AAAA")

    def test_empty_is_none(self) -> None:
        assert admet.aliphatic_index("") is None


class TestPhysicochemical:
    def test_core_values_present_for_a_standard_peptide(self) -> None:
        block = admet.physicochemical("GIGKFLHSAKKFGKAFVGEIMNS")
        assert block["molecular_weight"] > 2000
        assert 0 < block["isoelectric_point"] <= 14
        assert block["net_charge_ph7"] > 0  # a cationic AMP
        assert block["aliphatic_index"] is not None

    def test_non_standard_only_sequence_gives_none_protparam_metrics(self) -> None:
        # A sequence of only ambiguity codes has nothing ProtParam can measure.
        block = admet.physicochemical("XBZ")
        assert block["molecular_weight"] is None
        assert block["isoelectric_point"] is None


class TestInstabilityFlag:
    def test_below_cutoff_is_stable(self) -> None:
        assert admet.instability_flag(10.0)["stable"] is True

    def test_above_cutoff_is_unstable(self) -> None:
        assert admet.instability_flag(55.0)["stable"] is False

    def test_none_propagates(self) -> None:
        assert admet.instability_flag(None)["stable"] is None


class TestHalfLifeClass:
    def test_destabilizing_n_terminus(self) -> None:
        # Arg is a classic destabilizing N-terminal residue under the N-end rule.
        result = admet.half_life_class("RGGGG")
        assert result["n_terminal"] == "R"
        assert result["class"] == "destabilizing"

    def test_stabilizing_n_terminus(self) -> None:
        # Gly is stabilizing.
        assert admet.half_life_class("GRRRR")["class"] == "stabilizing"

    def test_leading_non_standard_is_skipped(self) -> None:
        # The first *standard* residue is what the rule reads.
        assert admet.half_life_class("XRGGG")["n_terminal"] == "R"


class TestAggregationHotspots:
    def test_finds_a_hydrophobic_run(self) -> None:
        result = admet.aggregation_hotspots("KKKAILFKKK")  # AILF is a 4-run of hydrophobics
        assert result["hotspots"] == [{"start": 4, "end": 7}]
        assert result["hotspot_fraction"] == pytest.approx(4 / 10)

    def test_no_run_shorter_than_the_minimum(self) -> None:
        # Isolated hydrophobics (no run of 3) -> no hotspot.
        assert admet.aggregation_hotspots("KAKAKAKA")["hotspots"] == []

    def test_empty_sequence(self) -> None:
        assert admet.aggregation_hotspots("")["hotspot_fraction"] is None


class TestNetChargeClass:
    @pytest.mark.parametrize("charge,expected", [
        (3.0, "cationic"), (-3.0, "anionic"), (0.5, "near-neutral"), (-1.0, "near-neutral")])
    def test_classes(self, charge: float, expected: str) -> None:
        assert admet.net_charge_class(charge)["class"] == expected


class TestAdmetProfile:
    def test_full_profile_shape(self) -> None:
        p = admet.admet_profile("GIGKFLHSAKKFGKAFVGEIMNS")
        assert set(p) == {"sequence", "length", "non_standard_residues",
                          "physicochemical", "flags", "gaps"}
        assert set(p["flags"]) == {"instability", "half_life", "aggregation", "net_charge"}

    def test_permeability_is_always_an_explicit_gap_never_a_number(self) -> None:
        p = admet.admet_profile("GIGKFLHSAKKFGKAFVGEIMNS")
        gap_props = {g["property"] for g in p["gaps"]}
        assert "membrane_permeability" in gap_props
        # It is never invented as a physicochemical value.
        assert "membrane_permeability" not in p["physicochemical"]
        perm = next(g for g in p["gaps"] if g["property"] == "membrane_permeability")
        assert perm["status"] == "not_predicted"

    def test_counts_non_standard_residues(self) -> None:
        p = admet.admet_profile("GIGKXBZ")
        assert p["non_standard_residues"] == 3
        assert p["length"] == 7

    def test_empty_sequence_is_empty_dict(self) -> None:
        assert admet.admet_profile("") == {}
