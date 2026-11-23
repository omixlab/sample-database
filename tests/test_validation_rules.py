"""
Tests for :mod:`sample.validation_rules` — the inspectable rule tables.

These tables are scientific claims, not implementation detail, so the tests check two distinct
things: that the derived indexes stay consistent with the tables they come from (a structural
property), and that specific domain rules hold (a factual property). The second kind exists so
that silently deleting a rule fails a test rather than quietly widening what the pipeline accepts.
"""
import pytest

from sample import validation_rules as R


class TestTableIntegrity:
    """Structural invariants across the four tables."""

    def test_tables_are_non_empty(self) -> None:
        assert R.UNIT_RANGES and R.TARGET_GROUP_DOMAINS
        assert R.ACTIVITY_RULES and R.UNIT_EQUIVALENCES

    def test_unit_names_are_unique(self) -> None:
        units = [r.unit for r in R.UNIT_RANGES]
        assert len(units) == len(set(units))

    def test_activity_types_are_unique(self) -> None:
        types = [r.activity_type for r in R.ACTIVITY_RULES]
        assert len(types) == len(set(types))

    def test_target_groups_are_unique(self) -> None:
        groups = [r.target_group for r in R.TARGET_GROUP_DOMAINS]
        assert len(groups) == len(set(groups))

    def test_every_range_is_ordered(self) -> None:
        for r in R.UNIT_RANGES:
            assert r.min_value <= r.max_value, f"{r.unit} has an inverted range"

    def test_derived_indexes_match_their_tables(self) -> None:
        assert set(R.UNIT_BY_NAME) == {r.unit for r in R.UNIT_RANGES}
        assert set(R.RULE_BY_ACTIVITY) == {r.activity_type for r in R.ACTIVITY_RULES}
        assert set(R.DOMAINS_BY_GROUP) == {r.target_group for r in R.TARGET_GROUP_DOMAINS}
        assert set(R.VALID_UNITS) == {r.unit for r in R.UNIT_RANGES}
        assert set(R.VALID_ACTIVITY_TYPES) == {r.activity_type for r in R.ACTIVITY_RULES}
        assert set(R.VALID_TARGET_GROUPS) == {r.target_group for r in R.TARGET_GROUP_DOMAINS}

    def test_equivalence_targets_are_themselves_valid_units(self) -> None:
        # Converting into a unit the pipeline does not recognise would strand the record.
        for eq in R.UNIT_EQUIVALENCES:
            assert eq.target in R.VALID_UNITS, f"{eq.source} -> unknown unit {eq.target}"

    def test_equivalence_factors_are_positive(self) -> None:
        for eq in R.UNIT_EQUIVALENCES:
            assert eq.factor > 0

    def test_activity_rules_reference_known_groups(self) -> None:
        for rule in R.ACTIVITY_RULES:
            for group in rule.target_groups:
                assert group in R.VALID_TARGET_GROUPS, f"{rule.activity_type} -> {group}"

    def test_activity_rules_reference_known_quantities(self) -> None:
        known = {r.quantity for r in R.UNIT_RANGES}
        for rule in R.ACTIVITY_RULES:
            for quantity in rule.quantities:
                assert quantity in known, f"{rule.activity_type} -> {quantity}"

    def test_every_rule_documents_itself(self) -> None:
        # The tables are published in the docs; a blank note there is a rule nobody can review.
        for rule in R.ACTIVITY_RULES:
            assert rule.note.strip()
        for rule in R.TARGET_GROUP_DOMAINS:
            assert rule.note.strip()


class TestDomainRules:
    """Specific corrections that must not regress."""

    def test_antiparasitic_admits_helminths(self) -> None:
        # Helminths are animals; restricting antiparasitics to protozoa drops every worm assay.
        assert "animal" in R.DOMAINS_BY_GROUP["antiparasitic"]

    def test_antibacterial_admits_archaea(self) -> None:
        assert "archaea" in R.DOMAINS_BY_GROUP["antibacterial"]

    @pytest.mark.parametrize("assay", ["TI", "SI"])
    def test_therapeutic_and_selectivity_indices_exist(self, assay: str) -> None:
        # These are ASSAYS reported as a bare ratio, not units -- hence "dimensionless" rather
        # than an entry in UNIT_RANGES.
        assert assay in R.VALID_ACTIVITY_TYPES
        assert "dimensionless" in R.RULE_BY_ACTIVITY[assay].quantities
        assert "" in R.units_for_activity(assay)

    def test_dimensionless_unit_is_representable(self) -> None:
        assert "" in R.VALID_UNITS

    def test_percentage_ceiling_allows_measurement_overshoot(self) -> None:
        # Assays legitimately report slightly over 100% of control; 105% is real data, not noise.
        pct = R.UNIT_BY_NAME["%"]
        assert pct.max_value >= 105.0

    def test_zone_of_inhibition_is_bounded_by_plate_size(self) -> None:
        mm = R.UNIT_BY_NAME["mm"]
        assert mm.max_value <= 200.0, "a zone larger than any plate is a parsing artefact"

    @pytest.mark.parametrize("unit", ["ug/uL", "M"])
    def test_missing_equivalences_are_present(self, unit: str) -> None:
        assert unit in R.VALID_UNITS or unit in R.EQUIVALENCE_BY_UNIT


class TestDerivableTargetGroup:
    def test_hemolysis_implies_its_own_group(self) -> None:
        assert R.derivable_target_group("Hemolysis") is not None

    def test_unknown_activity_derives_nothing(self) -> None:
        assert R.derivable_target_group("not-an-assay") is None

    def test_generic_assay_derives_nothing(self) -> None:
        # MIC is run against bacteria, fungi and parasites alike, so it implies no single group.
        assert R.derivable_target_group("MIC") is None


class TestUnitsForActivity:
    def test_mic_permits_concentration_units(self) -> None:
        units = R.units_for_activity("MIC")
        assert "ug/mL" in units and "uM" in units

    def test_mic_excludes_zone_diameter(self) -> None:
        # A MIC is a concentration. Reporting one in millimetres is the metric/unit contradiction
        # the rules exist to catch.
        assert "mm" not in R.units_for_activity("MIC")

    def test_unknown_activity_fails_open(self) -> None:
        # Documented behaviour: an assay absent from the table is NOT silently constrained, so
        # every unit is permitted rather than none. Restricting here would reject real data for
        # assays the table has not caught up with.
        assert R.units_for_activity("not-an-assay") == set(R.VALID_UNITS)

    def test_returns_a_set(self) -> None:
        assert isinstance(R.units_for_activity("MIC"), set)


class TestValueRangeForUnit:
    def test_known_unit_returns_ordered_pair(self) -> None:
        low, high = R.value_range_for_unit("ug/mL")
        assert low <= high

    def test_unknown_unit_returns_none(self) -> None:
        assert R.value_range_for_unit("furlongs") is None


class TestConvertToCanonical:
    def test_identity_for_canonical_unit(self) -> None:
        assert R.convert_to_canonical(10.0, "ug/mL") == (10.0, "ug/mL")

    def test_applies_the_documented_factor(self) -> None:
        for eq in R.UNIT_EQUIVALENCES:
            value, unit = R.convert_to_canonical(1.0, eq.source)
            assert unit == eq.target
            assert value == pytest.approx(eq.factor)

    def test_unknown_unit_passes_through(self) -> None:
        # Refusing to guess: an unrecognised unit is left alone for the contract to reject.
        assert R.convert_to_canonical(5.0, "furlongs") == (5.0, "furlongs")
