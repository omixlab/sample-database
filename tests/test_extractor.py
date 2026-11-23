"""
Tests for the extraction precision gates in :mod:`sample.extractor` and :mod:`sample.nlp`.

Every case here is a MEASURED failure from the first extraction run under the document gates, not
an invented one. The run extracted "IP" from "prostacyclin (IP) receptors", "FG" from "FGF", "PG"
from "protegrin PG-1", "DEVD" from "GDEVDGSGK" and "WGFTLC" from "CTTHWGFTLC" -- 248 nameless
short sequences in 866 documents, half of all sequence-bearing records.

Two distinct causes, two distinct fixes, tested separately:

* catalogued sequences shorter than :data:`sample.nlp.MIN_CATALOGUED_SEQUENCE_LEN` were being used
  as Aho-Corasick search keys, so a 2-4 residue database entry matched ordinary prose;
* a sequence that is only ever a substring of a longer candidate in the same document was kept
  alongside it, so a truncation of a real peptide was recorded as a peptide.

The safety case matters as much as the failing ones: truncation analogues are real in this
literature, so a short sequence the document states on its own must survive.
"""
from sample import nlp as N
from sample.extractor import _is_sequence_fragment


class TestCataloguedSequenceFloor:
    def test_floor_matches_the_regex_floor(self) -> None:
        # SEQ_EXACT requires 5 residues. The automaton must agree, or the two discovery paths
        # disagree about what is too short to be findable by text search.
        assert N.MIN_CATALOGUED_SEQUENCE_LEN == 5

    def test_the_measured_junk_keys_are_not_search_keys(self) -> None:
        kp = N.get_known_seq_1l_kp()
        for junk in ("IP", "PG", "FG", "PP", "DEVD"):
            assert not kp.extract_keywords(junk), f"{junk!r} is still a search key"

    def test_real_catalogued_peptides_are_still_found(self) -> None:
        # The floor must not cost recall on actual peptides -- magainin-2 and a hybrid.
        kp = N.get_known_seq_1l_kp()
        for real in ("GIGKFLHSAKKFGKAFVGEIMNS", "KWKLFKKIGIGKFLHSAKKF"):
            assert kp.extract_keywords(real), f"{real!r} should still be found"

    def test_a_short_sequence_is_still_a_valid_sequence(self) -> None:
        # The floor governs SEARCHING, not validity. A 4-residue peptide remains a legitimate
        # database record; it simply cannot be identified by finding those letters in prose.
        assert N.is_valid_sequence("DEVD")


class TestSequenceFragmentRule:
    def test_fragment_of_a_longer_candidate_is_dropped(self) -> None:
        text = "Our prototype synthetic peptide, CTTHWGFTLC, inhibits the migration of cells."
        assert _is_sequence_fragment(text, "WGFTLC", ["WGFTLC", "CTTHWGFTLC"])

    def test_caspase_motif_inside_a_longer_peptide_is_dropped(self) -> None:
        text = "GDEVDGSGK was used as a caspase-3 cleavable peptide."
        assert _is_sequence_fragment(text, "DEVD", ["DEVD", "GDEVDGSGK"])

    def test_a_truncation_analogue_stated_on_its_own_survives(self) -> None:
        # The expensive failure: this literature is full of truncation series, and dropping the
        # short member would delete a real peptide. One standalone occurrence is enough to keep it.
        text = ("The palindromic peptide RWQWRWQWR (peptide 1) and its truncated analogue RWQWR "
                "were both synthesised. RWQWR showed lower activity.")
        assert not _is_sequence_fragment(text, "RWQWR", ["RWQWR", "RWQWRWQWR"])

    def test_no_longer_candidate_means_not_a_fragment(self) -> None:
        text = "The peptide RWQWR was tested against E. coli."
        assert not _is_sequence_fragment(text, "RWQWR", ["RWQWR"])

    def test_the_longer_candidate_itself_is_never_a_fragment(self) -> None:
        text = "Peptide CTTHWGFTLC was synthesised."
        assert not _is_sequence_fragment(text, "CTTHWGFTLC", ["WGFTLC", "CTTHWGFTLC"])

    def test_absent_sequence_is_not_reported_as_a_fragment(self) -> None:
        # A candidate that does not occur in the text at all is some other stage's problem.
        assert not _is_sequence_fragment("unrelated text", "WGFTLC", ["WGFTLC", "CTTHWGFTLC"])

    def test_empty_sequence_is_not_a_fragment(self) -> None:
        assert not _is_sequence_fragment("some text", "", ["ABCDEFG"])

    def test_equal_length_candidates_do_not_shadow_each_other(self) -> None:
        # Containment requires a strictly longer candidate; two distinct same-length peptides must
        # both survive.
        text = "Peptides RWQWR and KWKWK were compared."
        assert not _is_sequence_fragment(text, "RWQWR", ["RWQWR", "KWKWK"])


class TestDiscoveryProvenance:
    """
    Every candidate records WHICH path found it. Without this, a record from a 2-residue
    catalogued-automaton hit is indistinguishable from one read off a results table, and diagnosing
    a precision problem means re-deriving each record's origin by hand -- which is exactly what the
    2026-09-16 audit had to do.
    """

    def test_every_candidate_type_maps_to_a_provenance_label(self) -> None:
        from sample.extractor import _DISCOVERY_PATHS
        # The types actually produced by candidate discovery in extract_peptides_from_text.
        for seq_type in ("exact", "3letter", "name", "alphanum"):
            assert seq_type in _DISCOVERY_PATHS

    def test_labels_distinguish_sequence_from_name_paths(self) -> None:
        from sample.extractor import _DISCOVERY_PATHS
        assert _DISCOVERY_PATHS["exact"] != _DISCOVERY_PATHS["name"]
        assert "sequence" in _DISCOVERY_PATHS["exact"]
        assert "name" in _DISCOVERY_PATHS["name"]


class TestCandidateConfidence:
    """
    Recorded, NOT yet gating. The threshold has to come from the curation audit's labelled
    findings; calibrating it on the 13 documents audited so far would dress a guess as a
    measurement. These tests pin the ORDERING, which is what the score has to get right before it
    can ever become a gate.
    """

    @staticmethod
    def _pep(seq: str = "", name: str = "", acts: int = 1) -> dict:
        return {"peptide_sequence": seq, "peptide_name": name,
                "biological_activities": [{"x": 1}] * acts}

    def test_prose_noise_scores_near_zero(self) -> None:
        from sample.extractor import candidate_confidence
        for junk in ("IP", "DEVD", "PISEMA"):
            result = candidate_confidence(self._pep(seq=junk))
            assert result["score"] <= 1, f"{junk!r} scored {result['score']}"
            assert "catalogued" not in result["signals"]
            assert "has_name" not in result["signals"]

    def test_a_catalogued_named_peptide_scores_high(self) -> None:
        from sample.extractor import candidate_confidence
        result = candidate_confidence(
            self._pep(seq="GIGKFLHSAKKFGKAFVGEIMNS", name="magainin 2"))
        assert result["score"] >= 7
        assert "catalogued" in result["signals"]
        assert "long_sequence" in result["signals"]

    def test_noise_scores_strictly_below_a_real_peptide(self) -> None:
        # The property that matters: whatever threshold is eventually chosen, it must be able to
        # separate these two.
        from sample.extractor import candidate_confidence
        noise = candidate_confidence(self._pep(seq="IP"))["score"]
        real = candidate_confidence(
            self._pep(seq="GIGKFLHSAKKFGKAFVGEIMNS", name="magainin 2"))["score"]
        assert noise < real

    def test_scoring_never_raises_on_a_malformed_record(self) -> None:
        from sample.extractor import candidate_confidence
        for bad in ({}, {"peptide_sequence": None}, {"peptide_name": None}):
            assert candidate_confidence(bad)["score"] >= 0

    def test_standalone_signal_requires_text_and_others(self) -> None:
        from sample.extractor import candidate_confidence
        pep = self._pep(seq="RWQWR", name="peptide 2")
        assert "standalone" not in candidate_confidence(pep)["signals"]
        withctx = candidate_confidence(
            pep, text="The peptide RWQWR was tested on its own.", others=["RWQWR"])
        assert "standalone" in withctx["signals"]


class TestTriageExemptionIsKeyedOnCatalogueMembership:
    """
    The exemption used to read `not pd.get("peptide_sequence")`, so ANY sequence-bearing candidate
    skipped LLM triage -- including a nameless regex 5-mer scraped from prose. 248 such records
    appeared in 866 documents. Catalogue membership is the signal that actually justifies the
    exemption, and it is what protects LL-37 from a bad model ballot.
    """

    @staticmethod
    def _catalogued(val: str, pep: dict) -> bool:
        from sample.nlp import (_normalize_peptide_key, get_known_seq_1l_kp,
                                get_name_sequence_map)
        if _normalize_peptide_key(pep.get("peptide_name") or val) in get_name_sequence_map():
            return True
        sequence = (pep.get("peptide_sequence") or "").strip()
        if not sequence:
            return False
        return bool(get_known_seq_1l_kp().extract_keywords(sequence))

    def test_a_catalogued_sequence_is_still_exempt(self) -> None:
        assert self._catalogued(
            "x", {"peptide_name": "", "peptide_sequence": "GIGKFLHSAKKFGKAFVGEIMNS"})

    def test_an_uncatalogued_regex_hit_is_no_longer_exempt(self) -> None:
        # The whole point of the change: this is what used to bypass the only check able to reject it.
        assert not self._catalogued("x", {"peptide_name": "", "peptide_sequence": "PISEMA"})

    def test_a_name_only_junk_candidate_is_not_exempt(self) -> None:
        assert not self._catalogued("CO2", {"peptide_name": "CO2", "peptide_sequence": ""})
