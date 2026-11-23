"""
Tests for :mod:`sample.synonyms`, which gives name-only peptides a sequence.

The reason this module exists: 81% of extracted peptides carry a name and an activity measurement
but no sequence (8,270 of 10,193 in the current pubmed run), and ``sample.merge`` discards every
record without one -- ``if not seq: continue``. The reason it is tested this hard: the first
version of it **fabricated sequences out of English prose**, and those would have been written to
the database as peptides.

Most of what follows is regression cover for specific fabrications that were measured, not
hypothetical ones. Each is named after the string it produced.
"""
from typing import Any, Dict, List

import pytest

from sample import synonyms as S


class TestNameFilters:
    """
    Two separate notions, and conflating them costs recall or correctness.

    A JUNK name cannot be resolved by any route -- there is nothing there. A DOCUMENT-LOCAL name
    ("Peptide 1") cannot be looked up in the catalogue, because a match would only mean two papers
    chose the same label; but it is perfectly resolvable from the document itself, and the first
    version of this module short-circuited on it and threw that away.
    """

    @pytest.mark.parametrize("name", [
        "", " ", "2", "x", "0.62 l g ml - 1", "25 ug/mL", "1.5 mg/mL", "10 uM",
    ])
    def test_measurements_and_blanks_are_junk(self, name: str) -> None:
        # "0.62 l g ml - 1" is a real peptide_name in the current run: the table reader captured a
        # measurement as a name. Looking it up is meaningless and resolving it is impossible.
        assert S.is_junk_name(name) is True

    @pytest.mark.parametrize("name", ["gomesin", "PBD-1", "BmKn2", "PAF26", "Drosomycin"])
    def test_real_names_are_not_junk(self, name: str) -> None:
        assert S.is_junk_name(name) is False
        assert S.is_document_local_name(name) is False

    @pytest.mark.parametrize("name", ["Peptide 1", "compound 3", "analog 7", "P40", "pep-2"])
    def test_generic_labels_are_document_local(self, name: str) -> None:
        assert S.is_document_local_name(name) is True

    def test_a_document_local_name_is_still_resolvable_from_the_document(self) -> None:
        # The point of keeping the two notions apart: the paper writing "Peptide 1 (SEQUENCE)" is
        # the strongest evidence available, and the label being generic does not weaken it.
        text = "The lead compound Peptide 1 (GIGKFLHSAKKFGKAFVGEIMNS) was assayed."
        found = S.internal_explicit_sequence("Peptide 1", text)
        assert found is not None
        assert found[0] == "GIGKFLHSAKKFGKAFVGEIMNS"


class TestSequenceValidationRegressions:
    """
    Every string here was actually produced as a "recovered sequence" by the first version.

    The extracted text loses spaces constantly, so runs of residue-alphabet letters are abundant
    and mostly meaningless. These are the ones that got through a naive alphabet-and-length rule.
    """

    @pytest.mark.parametrize("fabrication,origin", [
        ("THEPEPTIDES", "'...thepeptides beh' -- 'the peptides'"),
        ("CASEICIN", "a peptide NAME, not a sequence"),
        ("TREATEDSAMPLES", "'Magainin-treatedsamples' -- 'treated samples'"),
        ("GRAMNEGATIVE", "'Gram-negative'"),
        ("GCGAAGGCATCAACAT", "a PCR primer -- A, C, G and T are all residue codes"),
    ])
    def test_prose_and_names_are_rejected(self, fabrication: str, origin: str) -> None:
        assert S._clean_sequence(fabrication) is None, f"accepted {fabrication} ({origin})"

    def test_seraspenide_is_handled_by_tier_demotion_not_by_string_rules(self) -> None:
        """
        An honest limit of the string validator, and where the protection actually lives.

        'AcSDKP' -> "SERASPENIDE" was a real fabrication from the alias tier. SERASPENIDE is a DRUG
        NAME, but every letter in it is a residue code and no dictionary carries it, so no
        string-level rule can separate it from a genuine sequence -- and inventing one would reject
        real peptides (AFTCHCRRSC... below is human defensin HD6).

        So the string test is EXPECTED to accept it, and the protection is that the tier which
        produced it cannot apply a sequence on its own.
        """
        assert S._clean_sequence("SERASPENIDE") == "SERASPENIDE"
        assert S.CONFIDENCE["internal_alias"] < S.MIN_CONFIDENCE_TO_JOIN

    @pytest.mark.parametrize("sequence", [
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "SVSCLRNKGVCMPGKCAPKMKQIGTCGMPQVKCCKRK",
        "AKRHHGYKRKFH",
        "AFTCHCRRSCYSTEYSYGTCTVMGINHRFCCL",
    ])
    def test_real_sequences_survive(self, sequence: str) -> None:
        # AFTCHCRRSC... is human defensin HD6 and AKRHHGYKRKFH is P-113D. Both begin with letters
        # that spell English function words ("a", "ak"), which is why a "starts with a function
        # word" heuristic was measured and then NOT adopted -- it rejected real peptides.
        assert S._clean_sequence(sequence) == sequence


class TestInternalExplicitBinding:
    def test_parenthesised_sequence_is_accepted(self) -> None:
        text = "Magainin 2 (GIGKFLHSAKKFGKAFVGEIMNS) showed an MIC of 25 ug/mL."
        assert S.internal_explicit_sequence("Magainin 2", text)[0] == "GIGKFLHSAKKFGKAFVGEIMNS"

    def test_colon_form_is_accepted(self) -> None:
        text = "Peptide RC1: RVCSAIPLPICHGGGGG was synthesised."
        assert S.internal_explicit_sequence("RC1", text)[0] == "RVCSAIPLPICHGGGGG"

    def test_sequence_then_name_is_accepted(self) -> None:
        text = "We used GIGKFLHSAKKFGKAFVGEIMNS (magainin 2) throughout."
        assert S.internal_explicit_sequence("magainin 2", text)[0] == "GIGKFLHSAKKFGKAFVGEIMNS"

    def test_mere_proximity_is_NOT_accepted(self) -> None:
        """
        The single most important negative in this file.

        Measured over the extracted records, 1,188 of 1,245 "a sequence sits near the name" hits
        were coincidence inside flattened table text -- a unit column landing beside an unrelated
        residue run. Accepting proximity would invent bindings that look exactly like data.
        """
        text = ("Concn in platelets f 7.55.5 pH: e ADSGEGDFLAEGGGVRGG ND 3.00 2.86 1.0 "
                "FP-B 21,551.6 PBP 410,2")
        assert S.internal_explicit_sequence("PBP", text) is None

    def test_a_dash_does_not_bind(self) -> None:
        # "Magainin-treatedsamples" matched a dash form in the first version and produced
        # TREATEDSAMPLES. Hyphens are part of peptide names and of ordinary prose, so they cannot
        # mark a binding.
        text = "Magainin-treatedsamples and controls were compared."
        assert S.internal_explicit_sequence("Magainin", text) is None


class TestCatalogueTiers:
    def _catalogue(self, monkeypatch, entries: Dict[str, List[Dict[str, Any]]]) -> None:
        monkeypatch.setattr(S, "_catalogue", lambda: entries)

    def _entry(self, seq: str, organism: str = "", activity: str = "",
               kind: str = "primary", entry_id: str = "E1", xrefs: str = "") -> Dict[str, Any]:
        return {"entry_id": entry_id, "sequence": seq, "name": "n", "name_kind": kind,
                "organism": organism, "activity": activity, "cross_references": xrefs}

    def _record(self, name: str, organism: str = None, activity: str = None,
                target: str = None) -> Dict[str, Any]:
        """
        ``organism`` is the SOURCE organism -- where the peptide came from, which is what the
        catalogue's ``source_organism`` records. ``target`` is what it was tested against, and is
        deliberately available separately so a test can assert it does NOT corroborate.
        """
        act = {}
        if organism:
            act["source_organism"] = organism
        if target:
            act["target_organism"] = target
        if activity:
            act["activity_type"] = activity
        return {"peptide_name": name, "peptide_sequence": "",
                "biological_activities": [act] if act else [], "context": ""}

    def test_a_unique_match_with_no_corroboration_is_recorded_but_not_applied(
            self, monkeypatch) -> None:
        # The name agreeing is not evidence that it is the same molecule. 9.4% of catalogue names
        # map to more than one sequence, so a bare name hit is weak on its own.
        self._catalogue(monkeypatch, {"pbd1": [self._entry("SVSCLRNKGVCMPGKCAPKM")]})
        record = self._record("PBD-1")
        resolution = S.resolve(record, "no binding here")
        assert resolution["method"] == "catalogue_unique_bare"
        assert S.apply_resolution(record, resolution) is False
        assert record["peptide_sequence"] == ""
        assert record["sequence_resolution"]["applied"] is False

    def test_organism_agreement_makes_a_unique_match_usable(self, monkeypatch) -> None:
        self._catalogue(monkeypatch, {"drosomycin": [
            self._entry("DCLSGRYKGPCAVWDNETCRRVCKEE", organism="Drosophila melanogaster")]})
        record = self._record("Drosomycin", organism="Drosophila melanogaster")
        resolution = S.resolve(record, "")
        assert resolution["method"] == "catalogue_corroborated"
        assert S.apply_resolution(record, resolution) is True
        assert record["peptide_sequence"].startswith("DCLSGRY")

    def test_the_organism_tested_AGAINST_does_not_corroborate(self, monkeypatch) -> None:
        """
        The distinction that makes organism evidence meaningful.

        The catalogue's ``source_organism`` is where a peptide comes from; an activity's
        ``target_organism`` is what it was assayed against. A frog peptide tested on *E. coli* must
        not "corroborate" a catalogue entry from *E. coli*. The first version compared these two
        fields, which is why organism agreement fired 31 times against the activity signal's 416 --
        and a spurious hit there is worse than a miss, because it is the evidence that promotes a
        bare catalogue match to usable.
        """
        self._catalogue(monkeypatch, {"brevinin": [
            self._entry("GIGKFLHSAKKFGKAFVG", organism="Escherichia coli")]})
        record = self._record("brevinin", target="Escherichia coli")
        resolution = S.resolve(record, "")
        assert resolution["method"] == "catalogue_unique_bare"
        assert S.apply_resolution(record, resolution) is False

    def test_a_source_organism_in_the_context_corroborates(self, monkeypatch) -> None:
        # Taken from the context only where a source cue introduces it.
        self._catalogue(monkeypatch, {"dermaseptin": [
            self._entry("ALWMTLLKKVLKAAAKAL", organism="Phyllomedusa bicolor")]})
        record = self._record("dermaseptin")
        record["context"] = "the peptide was isolated from Phyllomedusa bicolor skin secretions"
        resolution = S.resolve(record, "")
        assert resolution["method"] == "catalogue_corroborated"

    def test_two_sequences_sharing_a_name_are_refused_without_evidence(self,
                                                                       monkeypatch) -> None:
        self._catalogue(monkeypatch, {"ambig": [
            self._entry("GIGKFLHSAKKFGKAFVG", entry_id="A"),
            self._entry("KWKLFKKIGIGKFLHSAK", entry_id="B"),
        ]})
        record = self._record("ambig")
        resolution = S.resolve(record, "")
        assert resolution["method"] == "ambiguous"
        assert resolution["sequence"] is None
        assert S.apply_resolution(record, resolution) is False

    def test_evidence_can_disambiguate_two_candidates(self, monkeypatch) -> None:
        self._catalogue(monkeypatch, {"ambig": [
            self._entry("GIGKFLHSAKKFGKAFVG", entry_id="A", organism="Xenopus laevis"),
            self._entry("KWKLFKKIGIGKFLHSAK", entry_id="B", organism="Apis mellifera"),
        ]})
        record = self._record("ambig", organism="Xenopus laevis")
        resolution = S.resolve(record, "")
        assert resolution["method"] == "catalogue_disambiguated"
        assert S.apply_resolution(record, resolution) is True
        assert record["peptide_sequence"] == "GIGKFLHSAKKFGKAFVG"

    def test_a_document_local_name_is_never_looked_up(self, monkeypatch) -> None:
        self._catalogue(monkeypatch, {"peptide1": [self._entry("GIGKFLHSAKKFGKAFVG")]})
        record = self._record("Peptide 1")
        resolution = S.resolve(record, "no binding in this text")
        assert resolution["method"] == "no_candidate"
        assert S.apply_resolution(record, resolution) is False

    def test_decoys_cannot_be_reached(self) -> None:
        # Guards the catalogue query itself: a fabricated training sequence must never be handed to
        # a real record. Asserted on the SQL rather than the data so it cannot silently drift.
        import inspect

        source = inspect.getsource(S._catalogue)
        assert "is_decoy" in source and "= 0" in source


class TestApplyResolution:
    def test_a_rejected_attempt_is_still_recorded(self) -> None:
        # A rejected attempt says "this was considered and here is why it was left alone", which is
        # what lets a later pass revisit it instead of starting over.
        record = {"peptide_name": "X", "peptide_sequence": ""}
        resolution = {"sequence": "GIGKFLHSAKKFGKAFVG", "method": "catalogue_unique_bare",
                      "confidence": 0.50, "evidence": ["nothing corroborates it"],
                      "cross_references": {}}
        assert S.apply_resolution(record, resolution) is False
        assert record["peptide_sequence"] == ""
        assert record["sequence_resolution"]["method"] == "catalogue_unique_bare"
        assert record["sequence_resolution"]["evidence"]

    def test_cross_references_are_kept_even_when_no_sequence_is_set(self) -> None:
        # An accession the paper cites is useful to a curator whether or not anything local could
        # resolve a sequence from it.
        record = {"peptide_name": "X", "peptide_sequence": ""}
        resolution = {"sequence": None, "method": "no_candidate", "confidence": 0.0,
                      "evidence": [], "cross_references": {"uniprot": "P15450"}}
        S.apply_resolution(record, resolution)
        assert record["cross_references"] == {"uniprot": "P15450"}

    def test_the_alias_tier_cannot_apply_on_its_own(self) -> None:
        assert S.CONFIDENCE["internal_alias"] < S.MIN_CONFIDENCE_TO_JOIN

    def test_the_bare_catalogue_tier_cannot_apply_on_its_own(self) -> None:
        assert S.CONFIDENCE["catalogue_unique_bare"] < S.MIN_CONFIDENCE_TO_JOIN

    def test_the_document_and_accession_tiers_do_apply(self) -> None:
        for method in ("internal_explicit", "accession", "catalogue_corroborated",
                       "catalogue_disambiguated"):
            assert S.CONFIDENCE[method] >= S.MIN_CONFIDENCE_TO_JOIN, method


class TestAccessions:
    def test_an_accession_beside_the_name_is_found(self) -> None:
        text = "Drosomycin (UniProt P41964) is an antifungal defensin."
        assert S.accessions_near("Drosomycin", text).get("uniprot") == "P41964"

    def test_a_pdb_id_is_found(self) -> None:
        text = "The structure of drosomycin (PDB 1MYN) was solved by NMR."
        assert S.accessions_near("drosomycin", text).get("pdb") == "1MYN"

    def test_an_accession_far_from_the_name_is_not_attributed_to_it(self) -> None:
        text = "Drosomycin was assayed. " + ("filler text. " * 40) + "UniProt P41964"
        assert S.accessions_near("Drosomycin", text) == {}


class TestCatalogueSequencesAreValidatedToo:
    """
    A catalogue hit says WHICH entry matches. It is not evidence that the entry's sequence column
    contains a sequence.

    The text tiers were validated from the start; these were not, and the catalogue is not clean.
    Measured over 1,171 applied resolutions in a real run: 30 were 1-4 residues, and these are the
    actual values that reached records.
    """

    @pytest.mark.parametrize("value,why", [
        ("D", "'Asp' resolved to a single amino acid"),
        ("GI", "'G - I' resolved to two residues"),
        ("KGK", "below the catalogued floor of 5"),
        ("ACETYLATED", "'C2' resolved to this literal word; all letters are residue codes"),
        ("AMIDATED", "a modification, not a sequence"),
        ("CYCLIC", "a description"),
        ("ND", "a placeholder for 'not determined'"),
    ])
    def test_non_sequences_in_the_sequence_column_are_refused(self, value, why) -> None:
        assert S._usable_catalogue_sequence(value) is None, f"accepted {value} ({why})"

    @pytest.mark.parametrize("value", [
        "KKVVFKVKFK", "RLLRLLRLL", "GIGKFLHSAKKFGKAFVGEIMNS",
        "KILRGVCKKIMRTFLRRISKDILTGKK",
    ])
    def test_real_catalogue_sequences_survive(self, value) -> None:
        assert S._usable_catalogue_sequence(value) == value

    def test_the_floor_matches_the_project_wide_one(self) -> None:
        from sample.nlp import MIN_CATALOGUED_SEQUENCE_LEN

        assert S._MIN_CATALOGUE_SEQUENCE_LEN == MIN_CATALOGUED_SEQUENCE_LEN

    def test_a_candidate_with_an_unusable_sequence_is_not_offered(self, monkeypatch) -> None:
        monkeypatch.setattr(S, "_catalogue", lambda: {"c2": [
            {"entry_id": "E1", "sequence": "ACETYLATED", "name": "C2", "name_kind": "primary",
             "organism": "", "activity": "", "cross_references": ""}]})
        assert S.catalogue_candidates("C2x") == []


class TestRowLabelsAreDocumentLocal:
    """
    A bare row label is an index into one paper's table, not a name a catalogue can be asked about.
    Measured, 67 of 1,171 applied resolutions were keyed on one, including 'C2' -> ACETYLATED.
    """

    @pytest.mark.parametrize("label", ["1 b", "13a", "4a", "6e", "C2", "T1", "I1", "N17"])
    def test_labels_are_refused_for_catalogue_lookup(self, label) -> None:
        assert S.is_document_local_name(label) is True

    @pytest.mark.parametrize("name", ["NK-2", "LL-37", "HD-6", "PR-39", "PBD-1", "PAF26"])
    def test_hyphenated_two_letter_names_are_REAL_and_kept(self, name) -> None:
        """
        The regression this pins. An earlier version of the row-label pattern matched the
        hyphenated two-letter form, which is exactly how real peptides are named -- it would have
        discarded correct resolutions, including NK-2 -> KILRGVCKKIMRTFLRRISKDILTGKK.
        """
        assert S.is_document_local_name(name) is False
        assert S.is_junk_name(name) is False


class TestRevalidation:
    """
    Step 10 withdraws resolutions that no longer qualify.

    It exists because the rules tightened after records were written: the first resolver applied
    catalogue sequences without validating them, and over 1,172 applied resolutions in a real run
    that produced 'Asp' -> 'D', 'C2' -> ACETYLATED, 30 sequences of 1-4 residues and 67 keyed on a
    bare table row label. A dry run over the live corpus found 83 such resolutions in 32 records.
    """

    def _record(self, name, sequence, method="catalogue_corroborated", conf=0.80):
        return {"peptide_name": name, "peptide_sequence": sequence,
                "sequence_resolution": {"method": method, "confidence": conf,
                                        "evidence": ["x"], "applied": True}}

    def test_a_short_catalogue_sequence_is_withdrawn(self) -> None:
        assert S.revalidate_resolution(self._record("Asp", "D")) is not None

    def test_a_placeholder_sequence_is_withdrawn(self) -> None:
        assert S.revalidate_resolution(self._record("C2x", "ACETYLATED")) is not None

    def test_a_row_label_keyed_catalogue_match_is_withdrawn(self) -> None:
        r = S.revalidate_resolution(self._record("4a", "GIGKFLHSAKKFGKAFVG"))
        assert r is not None and "document-local" in r

    def test_a_sub_threshold_confidence_is_withdrawn(self) -> None:
        r = S.revalidate_resolution(
            self._record("gomesin", "GIGKFLHSAKKFGKAFVG", method="internal_alias", conf=0.70))
        assert r is not None and "below the current threshold" in r

    def test_a_good_resolution_is_kept(self) -> None:
        assert S.revalidate_resolution(
            self._record("NK-2", "KILRGVCKKIMRTFLRRISKDILTGKK")) is None

    def test_an_unapplied_resolution_is_ignored(self) -> None:
        r = self._record("x", "")
        r["sequence_resolution"]["applied"] = False
        assert S.revalidate_resolution(r) is None

    def test_withdrawal_clears_the_sequence_but_keeps_the_evidence(self, tmp_path) -> None:
        import json as _json

        doc = {"metadata": {"sequences_resolved": 2},
               "peptides": [self._record("Asp", "D"),
                            self._record("NK-2", "KILRGVCKKIMRTFLRRISKDILTGKK")]}
        path = tmp_path / "r.json"
        path.write_text(_json.dumps(doc), encoding="utf-8")
        out = S.revalidate_extraction_file(str(path))
        assert out["withdrawn"] == 1 and out["kept"] == 1
        back = _json.loads(path.read_text(encoding="utf-8"))
        bad, good = back["peptides"]
        assert bad["peptide_sequence"] == ""
        assert bad["sequence_resolution"]["applied"] is False
        # The trail survives, so what was removed and why stays legible.
        assert "withdrawn_reason" in bad["sequence_resolution"]
        assert bad["sequence_resolution"]["evidence"] == ["x"]
        assert good["peptide_sequence"] == "KILRGVCKKIMRTFLRRISKDILTGKK"
        assert back["metadata"]["sequences_resolved"] == 1
        assert back["metadata"]["resolutions_withdrawn"] == 1

    def test_it_is_idempotent(self, tmp_path) -> None:
        import json as _json

        doc = {"metadata": {}, "peptides": [self._record("Asp", "D")]}
        path = tmp_path / "r.json"
        path.write_text(_json.dumps(doc), encoding="utf-8")
        first = S.revalidate_extraction_file(str(path))
        second = S.revalidate_extraction_file(str(path))
        assert first["withdrawn"] == 1
        assert second["withdrawn"] == 0, "a second pass withdrew something again"

    def test_a_dry_run_writes_nothing(self, tmp_path) -> None:
        import json as _json

        doc = {"metadata": {}, "peptides": [self._record("Asp", "D")]}
        path = tmp_path / "r.json"
        path.write_text(_json.dumps(doc), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        out = S.revalidate_extraction_file(str(path), dry_run=True)
        assert out["withdrawn"] == 1
        assert path.read_text(encoding="utf-8") == before

    def test_an_unreadable_record_is_reported_not_raised(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        out = S.revalidate_extraction_file(str(path))
        assert out["error"] and out["withdrawn"] == 0

    def test_the_directory_walk_uses_the_allowlist(self) -> None:
        # Never os.walk: literature_data_json_repaired has held .bak/_old siblings with 40,799
        # files the pipeline must not read. The project rule exists for that.
        import inspect

        assert "iter_extraction_files" in inspect.getsource(S.revalidate_extraction_dirs)
