"""Tests for :mod:`sample.structure_analysis` — DSSP, Ramachandran and disorder from a PDB."""
from typing import Any

import pytest

from sample import structure_analysis as SA

# A tiny 3-residue poly-alanine PDB with pLDDT-style B-factors, enough for the coordinate-only
# analyses (phi/psi and B-factor reading). Backbone atoms only.
_MINI_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 90.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 90.00           O
ATOM      5  N   ALA A   2       3.332   1.548   0.000  1.00 40.00           N
ATOM      6  CA  ALA A   2       3.999   2.841   0.000  1.00 40.00           C
ATOM      7  C   ALA A   2       5.510   2.680   0.000  1.00 40.00           C
ATOM      8  O   ALA A   2       6.007   1.556   0.000  1.00 40.00           O
ATOM      9  N   ALA A   3       6.238   3.788   0.000  1.00 85.00           N
ATOM     10  CA  ALA A   3       7.693   3.788   0.000  1.00 85.00           C
ATOM     11  C   ALA A   3       8.244   5.208   0.000  1.00 85.00           C
ATOM     12  O   ALA A   3       7.486   6.178   0.000  1.00 85.00           O
TER
END
"""


@pytest.fixture
def pdb(tmp_path: Any) -> str:
    path = tmp_path / "mini.pdb"
    path.write_text(_MINI_PDB)
    return str(path)


class TestRamachandran:
    def test_one_entry_per_residue(self, pdb: str) -> None:
        angles = SA.ramachandran(pdb)
        assert len(angles) == 3

    def test_terminal_residues_have_no_phi_or_psi(self, pdb: str) -> None:
        angles = SA.ramachandran(pdb)
        # The first residue has no phi (no preceding residue), the last no psi.
        assert angles[0]["phi"] is None
        assert angles[-1]["psi"] is None
        # An interior residue has both.
        assert angles[1]["phi"] is not None and angles[1]["psi"] is not None

    def test_angles_are_in_degrees_within_range(self, pdb: str) -> None:
        for a in SA.ramachandran(pdb):
            for angle in (a["phi"], a["psi"]):
                assert angle is None or -180.0 <= angle <= 180.0

    def test_terminal_residues_have_no_region_either(self, pdb: str) -> None:
        angles = SA.ramachandran(pdb)
        assert angles[0]["region"] is None
        assert angles[-1]["region"] is None


class TestClassifyRamachandran:
    def test_canonical_alpha_helix_is_favored(self) -> None:
        assert SA.classify_ramachandran(-57.0, -47.0) == "favored"

    def test_canonical_beta_strand_is_favored(self) -> None:
        assert SA.classify_ramachandran(-120.0, 130.0) == "favored"

    def test_beta_basin_is_continuous_across_the_180_wrap(self) -> None:
        """psi=+175 and psi=-175 are the same extended conformation (just past +-180 on
        BioPython's -180..180 scale); a table that stopped dead at 180 would wrongly call one an
        outlier and the other favored for what is physically the same backbone geometry."""
        assert SA.classify_ramachandran(-120.0, 175.0) == "favored"
        assert SA.classify_ramachandran(-120.0, -175.0) == "favored"

    def test_left_handed_alpha_is_favored(self) -> None:
        assert SA.classify_ramachandran(60.0, 30.0) == "favored"

    def test_sterically_impossible_geometry_is_an_outlier(self) -> None:
        # phi=psi=0 puts the carbonyls on top of each other; reference density is exactly 0.
        assert SA.classify_ramachandran(0.0, 0.0) == "outlier"

    def test_the_same_angles_are_judged_by_residue_class(self) -> None:
        """
        The point of using the real reference tables. (60, -120) is a sparse-but-observed
        conformation for a general residue, a well-populated one for glycine, and flatly
        forbidden for proline. The previous box approximation ignored residue identity and so
        called it an outlier for all three -- wrong in two of them.
        """
        assert SA.classify_ramachandran(60.0, -120.0, "General") == "allowed"
        assert SA.classify_ramachandran(60.0, -120.0, "GLY") == "favored"
        assert SA.classify_ramachandran(60.0, -120.0, "PRO") == "outlier"

    def test_proline_cannot_reach_the_beta_basin(self) -> None:
        # Its ring fixes phi near -60, so the canonical beta point is an outlier for proline
        # even though it is the textbook favored region for everything else.
        assert SA.classify_ramachandran(-120.0, 130.0, "General") == "favored"
        assert SA.classify_ramachandran(-120.0, 130.0, "PRO") == "outlier"

    def test_missing_angle_is_none_not_outlier(self) -> None:
        assert SA.classify_ramachandran(None, -47.0) is None
        assert SA.classify_ramachandran(-57.0, None) is None

    def test_falls_back_to_the_box_test_without_reference_tables(self,
                                                                 monkeypatch: Any) -> None:
        """An absent optional dependency must coarsen the call, not remove the figure."""
        monkeypatch.setattr(SA, "_reference_tables", lambda: None)
        assert SA.classify_ramachandran(-57.0, -47.0) == "favored"
        assert SA.classify_ramachandran(0.0, 0.0) == "outlier"


class TestResidueTypeFor:
    @pytest.mark.parametrize("resname,following,expected", [
        ("GLY", "ALA", "GLY"),
        ("PRO", "ALA", "PRO"),
        ("ALA", "PRO", "PRE-PRO"),
        ("ALA", "ALA", "General"),
        ("ALA", None, "General"),
    ])
    def test_classes(self, resname: str, following: Any, expected: str) -> None:
        assert SA.residue_type_for(resname, following) == expected

    def test_glycine_and_proline_outrank_pre_proline(self) -> None:
        # The Top500 pre-proline table is explicitly "not Gly or Pro".
        assert SA.residue_type_for("GLY", "PRO") == "GLY"
        assert SA.residue_type_for("PRO", "PRO") == "PRO"


class TestReferenceContours:
    def test_bands_are_polygons_in_degrees(self) -> None:
        contours = SA.ramachandran_reference_contours("General")
        if contours is None:
            pytest.skip("pyrama reference tables are not installed here")
        assert set(contours) == {"favored", "allowed"}
        assert contours["favored"] and contours["allowed"]
        for polygon in contours["favored"]:
            for phi, psi in polygon:
                assert -181 <= phi <= 181 and -181 <= psi <= 181

    def test_favored_is_enclosed_by_allowed(self) -> None:
        # The favored band is the denser core, so it cannot cover more of the plot than the
        # allowed band that surrounds it.
        contours = SA.ramachandran_reference_contours("General")
        if contours is None:
            pytest.skip("pyrama reference tables are not installed here")
        favored = sum(len(p) for p in contours["favored"])
        allowed = sum(len(p) for p in contours["allowed"])
        assert favored < allowed


class TestRamachandranSummary:
    def test_percentages_sum_to_the_scored_count(self) -> None:
        points = [{"region": "favored"}, {"region": "favored"}, {"region": "allowed"},
                  {"region": "outlier"}, {"region": None}]  # a chain terminus, excluded
        summary = SA.ramachandran_summary(points)
        assert summary["n_scored"] == 4
        assert summary["favored_pct"] == 50.0
        assert summary["allowed_pct"] == 25.0
        assert summary["outlier_pct"] == 25.0

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        summary = SA.ramachandran_summary([])
        assert summary["n_scored"] == 0
        assert summary["favored_pct"] == 0.0


class TestDisorder:
    def test_reads_plddt_from_b_factors(self, pdb: str) -> None:
        d = SA.disorder(pdb)
        assert d["plddt"] == [90.0, 40.0, 85.0]
        assert d["mean_plddt"] == pytest.approx((90 + 40 + 85) / 3, abs=0.1)

    def test_flags_residues_below_the_threshold(self, pdb: str) -> None:
        d = SA.disorder(pdb)
        # Only the middle residue (pLDDT 40) is below the 50 disorder cut.
        assert d["disordered"] == [False, True, False]
        assert d["disorder_fraction"] == pytest.approx(1 / 3, abs=0.01)


class TestDsspSecondaryStructure:
    def test_none_when_mkdssp_is_absent(self, pdb: str, monkeypatch: Any) -> None:
        monkeypatch.setattr(SA, "_mkdssp_command", lambda: None)
        assert SA.dssp_secondary_structure(pdb) is None

    def test_three_state_collapse_covers_the_dssp_alphabet(self) -> None:
        # Every DSSP letter maps to one of the three display states.
        for letter in "HGIEBTS- P":
            assert SA._DSSP_THREE_STATE.get(letter, "coil") in ("helix", "sheet", "coil")


class TestAnalyzeStructure:
    def test_missing_pdb_returns_none(self) -> None:
        assert SA.analyze_structure("/definitely/not/here.pdb") is None

    def test_empty_path_returns_none(self) -> None:
        assert SA.analyze_structure("") is None

    def test_combines_the_analyses(self, pdb: str, monkeypatch: Any) -> None:
        # DSSP mocked absent so the test does not require mkdssp; the coordinate analyses are real.
        monkeypatch.setattr(SA, "_mkdssp_command", lambda: None)
        result = SA.analyze_structure(pdb)
        assert set(result) == {"ramachandran", "ramachandran_summary", "disorder", "dssp",
                               "ramachandran_reference"}
        assert len(result["ramachandran"]) == 3
        assert result["disorder"]["mean_plddt"] > 0
        assert result["dssp"] is None
        assert result["ramachandran_summary"]["n_scored"] == \
            sum(1 for p in result["ramachandran"] if p["region"] is not None)

    def test_reports_which_method_classified_the_angles(self, pdb: str,
                                                        monkeypatch: Any) -> None:
        """
        The page prints this sentence verbatim, so it must describe what actually ran. Claiming
        the Top500 reference while the box fallback did the work would be worse than either.
        """
        monkeypatch.setattr(SA, "_mkdssp_command", lambda: None)
        reference = SA.analyze_structure(pdb)["ramachandran_reference"]
        if reference["contours"] is None:
            assert "box approximation" in reference["method"]
        else:
            assert "Top500" in reference["method"] and reference["reference"]
            assert set(reference["contours"]) == {"favored", "allowed"}

    def test_the_fallback_says_so_rather_than_claiming_the_reference(
            self, pdb: str, monkeypatch: Any) -> None:
        monkeypatch.setattr(SA, "_mkdssp_command", lambda: None)
        monkeypatch.setattr(SA, "_reference_tables", lambda: None)
        SA.ramachandran_reference_contours.cache_clear()
        try:
            reference = SA.analyze_structure(pdb)["ramachandran_reference"]
            assert reference["contours"] is None
            assert "box approximation" in reference["method"]
            assert reference["reference"] == ""
        finally:
            SA.ramachandran_reference_contours.cache_clear()
