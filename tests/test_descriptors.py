"""Tests for :mod:`sample.descriptors` — biophysical descriptors."""
import math

import pytest

from sample.descriptors import (calculate_boman_index, calculate_hydrophobic_moment,
                                calculate_net_charge, compute_all_descriptors)


class TestHydrophobicMoment:
    def test_amphipathic_helix_scores_above_uniform(self) -> None:
        # An idealised amphipathic helix segregates polar and apolar faces, which is exactly what
        # the moment measures; a homopolymer has no face at all.
        amphipathic = calculate_hydrophobic_moment("LKKLLKLLKKLLKL")
        uniform = calculate_hydrophobic_moment("AAAAAAAAAAAAAA")
        assert amphipathic > uniform

    def test_is_non_negative(self) -> None:
        # A vector magnitude cannot be negative.
        for seq in ("GIGKFLHSAKKFGKAFVGEIMNS", "AAAA", "DDDD"):
            assert calculate_hydrophobic_moment(seq) >= 0.0

    def test_empty_is_zero(self) -> None:
        assert calculate_hydrophobic_moment("") == 0.0

    def test_angle_changes_result(self) -> None:
        # 100 deg/residue is the alpha-helix; 180 is a beta-strand. Different geometry, different
        # moment -- if the angle were ignored these would coincide.
        seq = "LKKLLKLLKKLLKL"
        assert calculate_hydrophobic_moment(seq, angle=100.0) != \
               calculate_hydrophobic_moment(seq, angle=180.0)

    def test_unknown_residue_does_not_raise(self) -> None:
        assert isinstance(calculate_hydrophobic_moment("GIGKX"), float)


class TestNetCharge:
    def test_basic_residues_positive(self) -> None:
        assert calculate_net_charge("KKKRRR") > 0

    def test_acidic_residues_negative(self) -> None:
        assert calculate_net_charge("DDDEEE") < 0

    def test_empty_is_zero(self) -> None:
        assert calculate_net_charge("") == 0.0

    def test_monotonic_in_ph(self) -> None:
        charges = [calculate_net_charge("KKKDDD", ph=p) for p in (2.0, 7.0, 12.0)]
        assert charges == sorted(charges, reverse=True)


class TestBomanIndex:
    def test_returns_float(self) -> None:
        assert isinstance(calculate_boman_index("GIGKFLHSAKKFGKAFVGEIMNS"), float)

    def test_empty_is_zero(self) -> None:
        assert calculate_boman_index("") == 0.0

    def test_hydrophobic_lower_than_polar(self) -> None:
        # Boman estimates protein-binding potential: polar/charged sequences score higher.
        assert calculate_boman_index("LLLLIIIIVVVV") < calculate_boman_index("KKKKRRRRDDDD")


class TestComputeAllDescriptors:
    def test_returns_mapping_of_floats(self) -> None:
        result = compute_all_descriptors("GIGKFLHSAKKFGKAFVGEIMNS")
        assert isinstance(result, dict) and result
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float)), f"{key} is {type(value)}"
            assert not math.isnan(float(value)), f"{key} is NaN"

    def test_empty_sequence_does_not_raise(self) -> None:
        assert isinstance(compute_all_descriptors(""), dict)

    def test_keys_stable_across_sequences(self) -> None:
        # Downstream code builds a DataFrame from these, so the key set must not vary by input.
        a = set(compute_all_descriptors("GIGK"))
        b = set(compute_all_descriptors("KKKKRRRR"))
        assert a == b
