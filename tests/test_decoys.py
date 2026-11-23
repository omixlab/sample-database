"""
Tests for :mod:`sample.decoys` — negative-sample generation.

Decoys are the negatives every classification model is trained against, so their properties are
not cosmetic: a shuffled decoy must preserve composition (otherwise the model learns to count
lysines rather than recognise a motif), and no decoy may coincide with a real positive.

The UniProt fetch is mocked. It is a network boundary, not logic under test.
"""
import os
from typing import Any, List
from unittest import mock

# `sample` must be imported before pandas: sample/__init__.py preloads sqlite3 to pin down which
# libstdc++ the process resolves against before pandas' bundled ICU can grab an incompatible one
# first -- see that module's docstring. Importing pandas here first defeats the guard when this
# file is run in isolation (`pytest tests/test_decoys.py`), which is how this was found.
from sample import decoys as D

import pandas as pd
import pytest


@pytest.fixture
def positives_csv(tmp_path: Any) -> str:
    """:returns: A positives CSV of the shape :func:`generate_decoys` expects."""
    frame = pd.DataFrame({
        "entry_id": [f"SAMPLE{i:04d}" for i in range(20)],
        "sequence": ["GIGKFLHSAKKFGKAFVGEIMNS", "ILPWKWPWWPWRR", "KKKKKKKKKK",
                     "GIGAVLKVLTTGLPALISWIKRKRQQ", "ACDEFGHIKL"] * 4,
    })
    path = str(tmp_path / "positives.csv")
    frame.to_csv(path, index=False)
    return path


#: FASTA the fake UniProt endpoint streams back. Two real protein fragments, long enough that
#: length-matched sub-sequences can be cut from them.
_UNIPROT_FASTA = [
    ">sp|P00001|TEST1 Test protein",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKA",
    "LPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERV",
    ">sp|P00002|TEST2 Another protein",
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV",
    "TTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNR",
]


def _fake_uniprot(*args: Any, **kwargs: Any) -> Any:
    """
    Stands in for the UniProt stream endpoint.

    ``requests`` is imported inside :func:`sample.decoys.generate_decoys`, so there is no
    module-level attribute to patch; the patch target is ``requests.get`` itself. The response is
    consumed with ``iter_lines()`` in streaming mode, so that is what the stub provides.
    """
    response = mock.MagicMock()
    response.status_code = 200
    response.iter_lines = lambda **kw: [line.encode() for line in _UNIPROT_FASTA]
    return response


class TestShuffleSequence:
    def test_preserves_length(self) -> None:
        seq = "GIGKFLHSAKKFGKAFVGEIMNS"
        assert len(D.shuffle_sequence(seq)) == len(seq)

    def test_preserves_composition(self) -> None:
        # The whole point of a shuffled decoy: identical amino-acid composition, destroyed order.
        # If composition drifted, a model could separate the classes by counting residues.
        seq = "GIGKFLHSAKKFGKAFVGEIMNS"
        assert sorted(D.shuffle_sequence(seq)) == sorted(seq)

    def test_usually_changes_the_order(self) -> None:
        seq = "GIGKFLHSAKKFGKAFVGEIMNS"
        assert any(D.shuffle_sequence(seq) != seq for _ in range(20))

    def test_homopolymer_is_unchanged(self) -> None:
        assert D.shuffle_sequence("AAAA") == "AAAA"

    @pytest.mark.parametrize("seq", ["", "A"])
    def test_degenerate_input_does_not_raise(self, seq: str) -> None:
        assert D.shuffle_sequence(seq) == seq


class TestGenerateDecoys:
    def test_writes_the_output_file(self, positives_csv: str, tmp_path: Any) -> None:
        out = str(tmp_path / "decoys.csv")
        D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="shuffling")
        assert os.path.exists(out)

    def test_shuffling_method_preserves_composition(self, positives_csv: str,
                                                    tmp_path: Any) -> None:
        out = str(tmp_path / "decoys.csv")
        D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="shuffling")

        positives = set(pd.read_csv(positives_csv)["sequence"])
        composition = {"".join(sorted(s)) for s in positives}
        for seq in pd.read_csv(out)["sequence"]:
            assert "".join(sorted(str(seq))) in composition

    def test_no_decoy_duplicates_a_positive(self, positives_csv: str,
                                            tmp_path: Any) -> None:
        # A decoy that IS a positive is a mislabelled training row.
        out = str(tmp_path / "decoys.csv")
        D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="shuffling")

        positives = set(pd.read_csv(positives_csv)["sequence"].astype(str))
        produced = set(pd.read_csv(out)["sequence"].astype(str))
        assert produced & positives == set()

    def test_ratio_controls_the_count(self, positives_csv: str, tmp_path: Any) -> None:
        n_positives = len(pd.read_csv(positives_csv))
        out = str(tmp_path / "decoys.csv")
        D.generate_decoys(positives_csv, out, decoy_ratio=0.5, method="shuffling")
        assert len(pd.read_csv(out)) <= n_positives

    def test_output_carries_a_sequence_column(self, positives_csv: str,
                                              tmp_path: Any) -> None:
        out = str(tmp_path / "decoys.csv")
        D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="shuffling")
        assert "sequence" in pd.read_csv(out).columns

    def test_uniprot_method_uses_the_fetched_proteins(self, positives_csv: str,
                                                      tmp_path: Any) -> None:
        out = str(tmp_path / "decoys.csv")
        with mock.patch("requests.get", side_effect=_fake_uniprot):
            D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="uniprot")
        assert os.path.exists(out)

    def test_combined_method_runs_both_paths(self, positives_csv: str,
                                             tmp_path: Any) -> None:
        out = str(tmp_path / "decoys.csv")
        with mock.patch("requests.get", side_effect=_fake_uniprot):
            D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="combined")
        assert len(pd.read_csv(out)) > 0

    def test_uniprot_failure_degrades_instead_of_raising(self, positives_csv: str,
                                                         tmp_path: Any) -> None:
        # A network outage must not abort the step; the shuffled half is still usable.
        out = str(tmp_path / "decoys.csv")
        with mock.patch("requests.get",
                        side_effect=OSError("network down")):
            D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="combined")
        assert os.path.exists(out)

    def test_missing_input_does_not_crash(self, tmp_path: Any) -> None:
        out = str(tmp_path / "decoys.csv")
        try:
            D.generate_decoys(str(tmp_path / "absent.csv"), out)
        except FileNotFoundError:
            pass  # An explicit failure is acceptable; a partial output file is not.
        assert not os.path.exists(out) or len(pd.read_csv(out)) == 0

    def test_output_carries_a_negative_type_column(self, positives_csv: str,
                                                    tmp_path: Any) -> None:
        # Provenance a flat, undifferentiated decoy pool could not offer -- see generate_decoys'
        # own docstring on why this was added: the earlier decoy-strategy review needed to know
        # which decoy came from which method, and the shipped output had no way to tell.
        out = str(tmp_path / "decoys.csv")
        D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="shuffling")
        produced = pd.read_csv(out)
        assert "negative_type" in produced.columns
        assert set(produced["negative_type"]) <= {"shuffled", "uniprot_fragment",
                                                   "fallback_random"}
        assert set(produced["negative_type"]) == {"shuffled"}  # method="shuffling" only

    def test_combined_method_tags_both_generation_methods(self, positives_csv: str,
                                                           tmp_path: Any) -> None:
        out = str(tmp_path / "decoys.csv")
        with mock.patch("requests.get", side_effect=_fake_uniprot):
            D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="combined")
        produced = pd.read_csv(out)
        # Both halves of "combined" must be distinguishable after the fact -- the entire point of
        # tagging generation method rather than returning an opaque sequence list.
        assert "shuffled" in set(produced["negative_type"])
        assert "uniprot_fragment" in set(produced["negative_type"])


class TestDropNearPositiveDecoys:
    """
    _drop_near_positive_decoys is the defense-in-depth check against a "hidden positive" -- a
    candidate decoy that survived keyword/name exclusion but is still too close to a real,
    curated-active sequence to trust as a negative.
    """

    def test_a_near_identical_candidate_is_dropped(self) -> None:
        from sample.families import cdhit_available

        if not cdhit_available():
            pytest.skip("cd-hit is not installed in this environment")
        positive = "GIGKFLHSAKKFGKAFVGEIMNS"
        # One residue changed from the positive -- well within DECOY_POSITIVE_IDENTITY (0.7) of
        # the real thing, the exact "hidden positive" shape this check exists to catch.
        near_duplicate = "GIGKFLHSAKKFGKAFVGEIMNA"
        genuinely_different = "MSKGEELFTGVVPILVELDGDVNGH"
        candidates = {near_duplicate: "uniprot_fragment", genuinely_different: "uniprot_fragment"}

        kept = D._drop_near_positive_decoys({positive}, candidates)

        assert near_duplicate not in kept
        assert genuinely_different in kept

    def test_empty_candidates_returns_empty_without_calling_cdhit(self) -> None:
        assert D._drop_near_positive_decoys({"GIGKFLHSAKKFGKAFVGEIMNS"}, {}) == {}

    def test_missing_cdhit_degrades_to_unfiltered_not_a_crash(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("sample.families.cdhit_available", lambda: False)
        candidates = {"MSKGEELFTGVVPILVELDGDVNGH": "uniprot_fragment"}
        kept = D._drop_near_positive_decoys({"GIGKFLHSAKKFGKAFVGEIMNS"}, candidates)
        assert kept == candidates


class TestAmpLikeAdversarialNegatives:
    """
    The adversarial ("uniprot_amp_like") negative source. A length- and composition-matched decoy
    still leaves "is it cationic and amphipathic" available as a stand-in for "is it active"; these
    fragments are non-AMP by UniProt annotation while being AMP-like by physicochemistry, so that
    shortcut stops separating the classes. The scoring helpers are pure and tested directly.
    """

    def test_net_charge_is_cationic_minus_anionic(self) -> None:
        assert D.net_charge("KKRR") == 4
        assert D.net_charge("DDEE") == -4
        assert D.net_charge("KKDD") == 0

    def test_histidine_is_not_counted_as_cationic(self) -> None:
        # Mostly neutral at pH 7.4 -- counting it would inflate the charge of histidine-rich
        # fragments and let non-cationic sequences into the adversarial bucket.
        assert D.net_charge("HHHH") == 0

    @pytest.mark.parametrize("seq,expected", [("AAAA", 1.0), ("KKKK", 0.0), ("", 0.0)])
    def test_hydrophobic_fraction(self, seq: str, expected: float) -> None:
        assert D.hydrophobic_fraction(seq) == expected

    def test_amp_like_needs_both_charge_and_hydrophobicity(self) -> None:
        assert D.is_amp_like("KKRRLLAAFFIIWW")     # cationic AND hydrophobic
        assert not D.is_amp_like("KKKKKKKKKK")     # cationic, no hydrophobic core
        assert not D.is_amp_like("AAAALLLLIIII")   # hydrophobic, not cationic
        assert not D.is_amp_like("DDEEDDEE")       # neither

    def test_combined_method_only_emits_known_negative_types(self, positives_csv: str,
                                                             tmp_path: Any) -> None:
        # The realised mix depends on what the fetched proteins happen to contain (an AMP-like
        # fragment is rare), so this asserts the contract that always holds: every row is tagged
        # with a recognised source and nothing is left untagged.
        out = str(tmp_path / "decoys.csv")
        with mock.patch("requests.get", side_effect=_fake_uniprot):
            frame = D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="combined")
        known = {"shuffled", "uniprot_fragment", "uniprot_amp_like", "fallback_random"}
        assert set(frame["negative_type"]) <= known
        assert frame["negative_type"].notna().all()

    def test_amp_like_method_does_not_emit_plain_uniprot_fragments(self, positives_csv: str,
                                                                   tmp_path: Any) -> None:
        # Whatever it manages to cut must be adversarial or fallback -- never a plain fragment,
        # which would silently dilute the source this method exists to isolate.
        out = str(tmp_path / "decoys.csv")
        with mock.patch("requests.get", side_effect=_fake_uniprot):
            frame = D.generate_decoys(positives_csv, out, decoy_ratio=1.0, method="amp_like")
        assert "uniprot_fragment" not in set(frame["negative_type"])
        for seq, kind in zip(frame["sequence"], frame["negative_type"]):
            if kind == "uniprot_amp_like":
                assert D.is_amp_like(seq)
