"""
Tests for :mod:`sample.model_check` -- the guard that should have caught the gemma3:4b swap.

The classifier is the load-bearing part: if it does not flag a derivatization reagent or a
category phrase as a non-peptide, the harness would have passed gemma3:4b and the swap would have
gone ahead anyway. These tests pin the failure classes that were actually observed in that
model's output, plus the one false positive that matters -- a long peptide SEQUENCE must never be
flagged merely for being long.

No model is called here. :func:`sample.model_check.run` drives the real backend as a subprocess,
which belongs in a manual diagnostic rather than a test suite.
"""
from typing import Any

from sample import model_check as MC


class TestIsSequence:
    def test_accepts_a_bare_residue_run(self) -> None:
        assert MC._is_sequence("GIGKFLHSAKKFGKAFVGEIMNS") is True

    def test_accepts_a_sequence_sitting_in_the_name_field(self) -> None:
        # 2.7% of records carry a real sequence in peptide_name; the harness must count those as
        # recoverable rather than as junk.
        assert MC._is_sequence("GNPVPILIPCHR") is True
        assert MC._is_sequence("MYFINILTL") is False, "9 residues is below the default floor"
        assert MC._is_sequence("MYFINILTL", minimum=9) is True

    def test_rejects_chemical_names_and_prose(self) -> None:
        for value in ("o-phthalaldehyde", "fluorescamine", "20 essential amino acids",
                      "magainin peptides", "LL-37", ""):
            assert MC._is_sequence(value) is False, value

    def test_rejects_a_run_containing_non_amino_acid_letters(self) -> None:
        # B, J, O, U, X, Z are not standard residues; a "sequence" using them is not one.
        assert MC._is_sequence("BJOUXZBJOUXZ") is False


class TestJunkClassifier:
    @staticmethod
    def _classes(name: str) -> set:
        return {label for label, test in MC._JUNK if test(name)}

    def test_flags_every_class_observed_in_the_gemma3_output(self) -> None:
        observed = {
            "o-phthalaldehyde": "reagent",
            "4-halo-7-nitro-benzo-2,1,3-oxadiazole": "reagent",
            "fluorescamine": "reagent",
            "magainin analogues A, B, C, F, G, and H": "phrase",
            "magainin peptides": "category",
            "Magainin10442437": "fused-digits",
            "20 essential amino acids": "leading-digit",
            "H": "too-short",
            r"Human epithelial $\beta$-defensins 2 and 3": "markup",
        }
        for name, expected in observed.items():
            assert expected in self._classes(name), f"{name!r} should be flagged as {expected}"

    def test_does_not_flag_a_real_peptide_name(self) -> None:
        for name in ("leuprolide", "cupiennins", "MART-1", "LL-37", "magainin-2",
                     "carnobacteriocins BM1", "Osp-247"):
            assert self._classes(name) == set(), f"{name!r} was wrongly flagged"

    def test_a_long_sequence_is_not_flagged_for_its_length(self) -> None:
        # The over-40-chars rule exists for prose, not for peptides. A 57-residue sequence in the
        # name field is recoverable data; flagging it would hide exactly what we want to promote.
        long_sequence = "MSEVIEGNVKIDRISPGDGATFPKTGDLVTIHYTGTLENGQKFDSSVDRGSPFQCNI"
        assert len(long_sequence) > 40
        assert "over-40-chars" not in self._classes(long_sequence)


class TestProfile:
    @staticmethod
    def _record(name: str = "magainin-2", sequence: str = "", activities: Any = None) -> dict:
        return {"peptide_name": name, "peptide_sequence": sequence,
                "biological_activities": activities if activities is not None else []}

    def test_counts_the_four_fates_a_record_can_have(self) -> None:
        records = [
            self._record(sequence="GIGKFLHSAKKFGKAFVGEIMNS", activities=[{"activity_type": "MIC"}]),
            self._record(name="GNPVPILIPCHR"),          # sequence hiding in the name
            self._record(name="o-phthalaldehyde"),      # junk
            self._record(name="leuprolide"),            # clean, but no activity
        ]
        p = MC._profile(records, documents=2, seconds=10.0)
        assert p["records"] == 4
        assert p["per_document"] == 2.0
        assert p["with_sequence_pct"] == 25.0
        assert p["name_is_sequence_pct"] == 25.0
        assert p["no_activity_pct"] == 75.0
        assert p["junk_name_pct"] == 25.0
        assert p["seconds_per_document"] == 5.0

    def test_survives_an_empty_result(self) -> None:
        # A candidate that returns nothing must produce a readable profile, not a ZeroDivisionError
        # -- "the model returned nothing" is itself a finding worth reporting.
        p = MC._profile([], documents=5, seconds=1.0)
        assert p["records"] == 0 and p["per_document"] == 0.0
        assert p["with_sequence_pct"] == 0.0 and p["junk_name_pct"] == 0.0

    def test_exposes_names_for_the_overlap_calculation(self) -> None:
        p = MC._profile([self._record(name="Magainin-2")], documents=1, seconds=1.0)
        assert p["names"] == {"magainin-2"}, "names are lowercased so overlap is case-insensitive"
