"""
What it takes for an extracted sequence to count as new to the field.

The old test was ``sequence not in third_party_sequences``. Over 32,408 refined documents it
called 975 sequences new; hand-classifying all 975 left roughly 230 defensible, the rest being
fragments of catalogued peptides, catalogued peptides with flanking text, three-letter notation
that cannot match a one-letter set, and sequences masked by an ambiguity code. The sharper
measurement: of the 40 whose extracted name is a peptide this project catalogues, the sequence
disagreed with the catalogued one 40 times out of 40. These tests pin the graded replacement and,
above all, the ORDER it applies -- judging the name before the sequence is a mistake that reads a
truncation as a contradiction.
"""
import pytest

from sample import novelty as N

MAGAININ = "GIGKFLHSAKKFGKAFVGEIMNS"
NISIN = "ITSISLCTPGCKTGALMGCNMKTATCHCSIHVSK"
CLAVANIN = "VFQFLGKIIHHVGNFVHGFSHVF"


@pytest.fixture()
def catalogue():
    """A catalogue shaped like the real one: sequences, plus names of varying usefulness."""
    return N.Catalogue(
        [MAGAININ, NISIN, CLAVANIN, "KWKKLLKKLLKLLKKLL", "RRGWVLALYLRYGRR"],
        names={
            "magainin2": {MAGAININ},
            "nisin": {NISIN},
            "clavanina": {CLAVANIN},
            # An analogue series: one name, a handful of sequences.
            "dermaseptin": {MAGAININ, "KWKKLLKKLLKLLKKLL"},
            # A class label a source filed in a name column, as dbAMP did with 'Mouse-ear cress'.
            "naturalamps": {MAGAININ, NISIN, CLAVANIN, "KWKKLLKKLLKLLKKLL",
                            "RRGWVLALYLRYGRR"} | {f"AAAAAAAAAA{i:02d}".replace("0", "G")
                                                  for i in range(30)},
        },
        ambiguous_at=20,
    )


class TestDecodingThreeLetterRuns:
    """
    The largest single source of false novelty: 103 of the 975. PDF extraction loses the hyphens
    from ``Arg-Leu-Tyr`` and ``sequence_to_1_letter`` has nothing left to split on, so the string
    reaches novelty as ``ARGLEUTYR`` and matches no one-letter sequence in the catalogue.
    """

    def test_an_unseparated_run_decodes(self):
        assert N.decode_run_of_three_letter_codes("ARGLEUTYRLEUARGILEGLY") == "RLYLRIG"

    def test_a_cyclization_marker_and_terminal_group_are_stripped(self):
        assert N.decode_run_of_three_letter_codes("CYCLOLEUARGLEULYSLYSARGNH2") == "LRLKKR"

    def test_a_stray_letter_between_codes_does_not_derail_the_parse(self):
        # Stereochemical prefixes survive OCR as bare letters wedged between codes.
        assert N.decode_run_of_three_letter_codes("ARGLEUDTYRLEUARGILE") == "RLYLRI"

    def test_a_one_letter_sequence_is_not_mistaken_for_a_run(self):
        # GIGKFLHSAKKFGKAFVGEIMNS contains no triplet that spells a residue name, but the point
        # is the coverage floor: a real one-letter sequence must come back untouched.
        assert N.decode_run_of_three_letter_codes(MAGAININ) == ""

    def test_a_short_accidental_triplet_is_rejected(self):
        # 'ALA' is a valid code, but three characters is not notation.
        assert N.decode_run_of_three_letter_codes("ALA") == ""

    def test_decoding_feeds_the_verdict(self, catalogue):
        """The whole point: the decoded form is what gets compared."""
        reverse = {one: three for three, one in N._RUN_CODES.items()}
        spelled = "".join(reverse[residue] for residue in MAGAININ)
        verdict = catalogue.judge(spelled)
        assert verdict.verdict == N.KNOWN_ENCODING
        assert not verdict.is_novel
        assert verdict.parent == MAGAININ

    def test_a_decoded_fragment_is_a_fragment(self, catalogue):
        """Decoding settles the notation; the shape tests still have their say afterwards."""
        reverse = {one: three for three, one in N._RUN_CODES.items()}
        spelled = "".join(reverse[residue] for residue in NISIN[4:24])
        verdict = catalogue.judge(spelled)
        assert verdict.verdict == N.KNOWN_FRAGMENT
        assert verdict.parent == NISIN


class TestComparability:
    """
    A sequence that cannot be compared is not novel. Novelty is an identity claim, and identity
    is undecidable where a position is masked -- 36 of the 975 carried an ambiguity code.
    """

    @pytest.mark.parametrize("code", sorted(N.AMBIGUITY_CODES))
    def test_an_ambiguity_code_makes_a_sequence_unjudgeable(self, catalogue, code):
        verdict = catalogue.judge(f"GIGKFLHSAKKF{code}GKAFVGEIM")
        assert verdict.verdict == N.NOT_COMPARABLE
        assert verdict.note == "ambiguity_code"
        assert not verdict.is_novel
        assert verdict.is_defect

    def test_the_validator_still_accepts_those_codes(self):
        """
        Comparability is not validity. B/Z/X/U/O are legitimate IUPAC codes and the sequence
        validator accepts them on purpose; this module declines to make a novelty claim about
        them. Both behaviours are correct and they must not be conflated.
        """
        from sample.sequence import validate_extracted_sequence
        assert validate_extracted_sequence("GIGKFLHSAKKFXGKAFVGEIM")[0]

    def test_a_sequence_too_short_to_adjudicate_is_not_novel(self, catalogue):
        verdict = catalogue.judge("GIGKF")
        assert verdict.verdict == N.NOT_COMPARABLE
        assert verdict.note == "too_short_to_adjudicate"
        assert verdict.is_defect

    def test_the_floor_is_above_the_validator_s_floor(self):
        """
        The validator asks whether a string is a sequence at all; this asks whether its absence
        from the catalogue is evidence. The second bar is higher, and every one of the 62
        sub-eight-residue 'novel' sequences was a label, an abbreviation or a fragment.
        """
        from sample.sequence import MIN_EXTRACTED_SEQUENCE
        assert N.MIN_COMPARABLE_LENGTH > MIN_EXTRACTED_SEQUENCE


class TestTheGradedVerdicts:
    """Each verdict is a different account of what the catalogue already holds."""

    def test_an_exact_match_is_known(self, catalogue):
        assert catalogue.judge(MAGAININ).verdict == N.KNOWN_EXACT

    def test_a_fragment_of_a_catalogued_sequence_is_not_new(self, catalogue):
        verdict = catalogue.judge(NISIN[4:24])
        assert verdict.verdict == N.KNOWN_FRAGMENT
        assert verdict.parent == NISIN
        assert not verdict.is_novel

    def test_a_catalogued_sequence_with_flanking_text_is_not_new(self, catalogue):
        verdict = catalogue.judge("WWWW" + MAGAININ + "WWWW")
        assert verdict.verdict == N.KNOWN_EXTENSION
        assert verdict.parent == MAGAININ

    def test_a_point_substitution_is_reported_as_an_analogue(self, catalogue):
        verdict = catalogue.judge("GIGKFLHSAKKFGKAFVGEIMNW")
        assert verdict.verdict == N.ANALOGUE
        assert verdict.distance == 1
        assert verdict.parent == MAGAININ

    def test_a_single_inserted_residue_is_caught(self, catalogue):
        """
        The measured OCR failure: ``RRGWVLALYLRLYGRR`` for the catalogued ``RRGWVLALYLRYGRR``.
        A substitution-only test misses it, because the lengths differ.
        """
        verdict = catalogue.judge("RRGWVLALYLRLYGRR")
        assert verdict.verdict == N.ANALOGUE
        assert verdict.distance == 1

    def test_the_nearest_parent_is_reported_not_the_first_found(self, catalogue):
        """Distance 1 and distance 2 are different claims, so the closest one has to win."""
        verdict = catalogue.judge("GIGKFLHSAKKFGKAFVGEIMNW")
        assert verdict.distance == 1

    def test_an_unrelated_sequence_is_novel(self, catalogue):
        verdict = catalogue.judge("WWPEFTCHMDQNTRYVKLWWPEFTC")
        assert verdict.verdict == N.NOVEL
        assert verdict.is_novel
        assert not verdict.is_defect

    def test_a_distant_relative_is_still_novel(self, catalogue):
        """Three edits is past the analogue limit, so the claim stands."""
        verdict = catalogue.judge("GIGKFLHSAKKFGKAFVGWWW")
        assert verdict.verdict == N.NOVEL


class TestNamesAdjudicateOnlyWhenTheyIdentifySomething:
    """
    The name is evidence about the sequence, but only some names are. A name attached to twenty
    or more sequences is a class label, and letting it overrule a sequence would file
    ``Mouse-ear cress`` as a verdict on 300 peptides.
    """

    def test_a_unique_name_that_agrees_corroborates(self, catalogue):
        verdict = catalogue.judge(MAGAININ, "Magainin 2")
        assert verdict.name_status == N.NAME_UNIQUE
        assert verdict.name_agreement == N.NAME_AGREES
        assert not verdict.is_defect

    def test_a_unique_name_on_an_unrelated_sequence_collapses_the_novelty_claim(self, catalogue):
        """
        Measured base rate for this combination: 40 errors out of 40. The sequence looks new and
        the name says the field already knows this peptide, so the claim is not supportable.
        """
        verdict = catalogue.judge("WWPEFTCHMDQNTRYVKLWWPEFTC", "nisin")
        assert verdict.verdict == N.NAME_CONFLICT
        assert not verdict.is_novel
        assert verdict.is_defect
        assert verdict.named == NISIN

    def test_a_sequence_catalogued_under_another_name_is_flagged_but_still_not_novel(
            self, catalogue):
        """
        `sequence_of_another_peptide`: the sequence is real and correctly transcribed, but the
        record pairs it with a different peptide's name. The novelty verdict stands -- the
        catalogue does hold this sequence -- while the record is a defect, because every activity
        on it is attributed to the wrong peptide.
        """
        verdict = catalogue.judge(MAGAININ, "nisin")
        assert verdict.verdict == N.KNOWN_EXACT
        assert verdict.name_agreement == N.NAME_DISAGREES
        assert verdict.is_defect
        assert verdict.named == NISIN
        assert verdict.parent == MAGAININ

    def test_the_named_sequence_is_reported_separately_from_the_matched_one(self, catalogue):
        """
        Reporting ``parent`` as what the name points to produced the message "the name 'Phd-1'
        identifies ACPIFTKIQGTYRGKAKCK, which this sequence does not match" about exactly that
        sequence. They are two different facts and need two fields.
        """
        verdict = catalogue.judge(MAGAININ, "nisin")
        assert verdict.named != verdict.parent

    def test_an_ambiguous_name_says_nothing(self, catalogue):
        """A class label must not contradict a sequence it knows nothing about."""
        verdict = catalogue.judge("WWPEFTCHMDQNTRYVKLWWPEFTC", "natural AMPs")
        assert verdict.name_status == N.NAME_AMBIGUOUS
        assert verdict.name_agreement == N.NAME_SILENT
        assert verdict.verdict == N.NOVEL
        assert not verdict.is_defect

    def test_an_analogue_series_name_still_adjudicates(self, catalogue):
        """A handful of sequences under one name is a real series, and any member counts."""
        verdict = catalogue.judge("KWKKLLKKLLKLLKKLL", "dermaseptin")
        assert verdict.name_status == N.NAME_SERIES
        assert verdict.name_agreement == N.NAME_AGREES

    def test_a_name_the_catalogue_has_never_seen_says_nothing(self, catalogue):
        verdict = catalogue.judge("WWPEFTCHMDQNTRYVKLWWPEFTC", "cnPGRP1")
        assert verdict.name_status == N.NAME_UNKNOWN
        assert verdict.name_agreement == N.NAME_SILENT
        assert verdict.is_novel

    def test_a_missing_name_says_nothing(self, catalogue):
        verdict = catalogue.judge("WWPEFTCHMDQNTRYVKLWWPEFTC", "")
        assert verdict.name_status == N.NAME_ABSENT
        assert verdict.is_novel

    def test_name_lookup_ignores_punctuation_and_case(self, catalogue):
        assert catalogue.name_status("magainin-2") == N.NAME_UNIQUE
        assert catalogue.name_status("MAGAININ 2") == N.NAME_UNIQUE


class TestTheSequenceIsJudgedBeforeTheName:
    """
    The ordering bug, kept as a test because it was shipped and measured. Asking the name first
    reported 107 conflicts among the trustworthy tier; asking the sequence first reported 29, and
    the 78 difference were truncations and fragments of the very peptide the name identified.
    """

    def test_a_truncated_sequence_reads_as_a_fragment_not_a_conflict(self, catalogue):
        """
        The measured case: ``DMS-H4`` carrying ``GLWSTIKNVGKEAAIAAGKAALGAL`` where the catalogued
        sequence is ``GLWSTIKNVGKEAAIAAGKAALGALG`` -- the same peptide, one residue short.
        """
        verdict = catalogue.judge(NISIN[:-1], "nisin")
        assert verdict.verdict == N.KNOWN_FRAGMENT
        assert verdict.name_agreement == N.NAME_AGREES
        assert not verdict.is_defect

    def test_an_analogue_of_the_named_peptide_is_not_a_conflict(self, catalogue):
        """Published analogue series are point mutants of the peptide they are named after."""
        verdict = catalogue.judge("GIGKFLHSAKKFGKAFVGEIMNW", "Magainin 2")
        assert verdict.verdict == N.ANALOGUE
        assert verdict.name_agreement == N.NAME_AGREES


class TestThePigeonholeIndexIsExact:
    """
    The analogue test cannot brute force: the length window around a 20-mer holds 17,857
    catalogued sequences and the corpus has 13,183 to judge, which is 235 million edit-distance
    computations. The index must therefore return everything brute force would -- a sequence it
    misses becomes a false novelty claim, the exact error this module exists to stop.
    """

    def test_blocks_partition_a_sequence_exactly_once(self):
        for length in range(1, 60):
            spans = N._block_spans(length)
            assert sum(stop - start for start, stop in spans) == length
            assert spans[0][0] == 0 and spans[-1][1] == length

    def test_there_is_one_more_block_than_the_edit_limit(self):
        """The pigeonhole: k edits damage at most k blocks, so k+1 blocks leave one intact."""
        assert len(N._block_spans(30)) == N.MAX_ANALOGUE_EDITS + 1

    def test_the_index_finds_everything_brute_force_finds(self):
        import random
        random.seed(11)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        pool = ["".join(random.choice(alphabet) for _ in range(random.randint(8, 34)))
                for _ in range(600)]
        catalogue = N.Catalogue(pool, names={})

        def brute(form):
            best, distance = "", N.MAX_ANALOGUE_EDITS + 1
            for candidate in catalogue.sequences:
                found = N._within(form, candidate, N.MAX_ANALOGUE_EDITS)
                if found is not None and found < distance:
                    best, distance = candidate, found
            return distance if best else 0

        def mutate(sequence, edits):
            chars = list(sequence)
            for _ in range(edits):
                index = random.randrange(len(chars))
                operation = random.choice(("substitute", "insert", "delete"))
                if operation == "substitute":
                    chars[index] = random.choice(alphabet)
                elif operation == "insert":
                    chars.insert(index, random.choice(alphabet))
                elif len(chars) > 9:
                    chars.pop(index)
            return "".join(chars)

        for trial in range(300):
            if trial % 2:
                query = mutate(random.choice(pool), random.randint(1, 2))
            else:
                query = "".join(random.choice(alphabet) for _ in range(random.randint(8, 34)))
            assert catalogue._nearest(query)[1] == brute(query), query


class TestTheCatalogueDegradesRatherThanFailing:
    """
    Novelty is one column of a long merge step. A catalogue that cannot be built should weaken
    the claim and say so, not abort step 21.
    """

    def test_a_missing_names_file_leaves_names_silent(self, tmp_path):
        catalogue = N.Catalogue.load(merged_path="", names_path=str(tmp_path / "absent.json"),
                                     extra_sequences=[MAGAININ])
        assert len(catalogue) == 1
        assert catalogue.name_status("Magainin 2") == N.NAME_UNKNOWN
        assert catalogue.judge(MAGAININ).verdict == N.KNOWN_EXACT

    def test_an_empty_catalogue_calls_a_comparable_sequence_novel(self):
        assert N.Catalogue([], names={}).judge(MAGAININ).verdict == N.NOVEL

    def test_an_empty_catalogue_still_declines_to_judge_the_unjudgeable(self):
        assert N.Catalogue([], names={}).judge("GIGKX").verdict == N.NOT_COMPARABLE


class TestSummarise:
    def test_every_verdict_is_a_key_even_at_zero(self):
        counts = N.summarise([])
        assert set(counts) == set(N.VERDICTS)
        assert sum(counts.values()) == 0

    def test_counts_add_up(self, catalogue):
        verdicts = [catalogue.judge(MAGAININ), catalogue.judge(NISIN),
                    catalogue.judge("WWPEFTCHMDQNTRYVKLWWPEFTC")]
        counts = N.summarise(verdicts)
        assert counts[N.KNOWN_EXACT] == 2
        assert counts[N.NOVEL] == 1


class TestStep21PublishesTheRefinedVerdict:
    """
    The column `sequence_novelty` is what a reader of the release sees, so the refinement has to
    reach it -- and has to stay auditable, which is why the basis travels alongside the label.
    """

    @pytest.fixture()
    def table(self, monkeypatch):
        """One activity row per sequence, through the real :func:`sample.merge._mark_novelty`."""
        pandas = pytest.importorskip("pandas")
        from sample import merge

        def fake_load(merged_path="", names_path="", extra_sequences=None):
            return N.Catalogue(extra_sequences or [],
                               names={"nisin": {NISIN}, "magainin2": {MAGAININ}})

        monkeypatch.setattr(N.Catalogue, "load", staticmethod(fake_load))

        def build(rows, third_party):
            frame = pandas.DataFrame(rows)
            merge._mark_novelty(frame, set(third_party), {}, {})
            return frame

        return build

    def test_a_catalogued_peptide_the_extractor_re_derived_is_confirmed(self, table):
        frame = table([{"sequence": MAGAININ, "peptide_name": "Magainin 2",
                        "provenance": "SAMPLE", "source_organism": ""}], [MAGAININ])
        assert frame["sequence_novelty"][0] == "Confirmed"
        assert frame["novelty_basis"][0] == N.KNOWN_EXACT

    def test_a_fragment_is_confirmed_not_new(self, table):
        """Under the old test this was ``SAMPLE (new)``; 103 of the 975 were this shape."""
        frame = table([{"sequence": NISIN[4:24], "peptide_name": "", "provenance": "SAMPLE",
                        "source_organism": ""}], [NISIN])
        assert frame["sequence_novelty"][0] == "Confirmed"
        assert frame["novelty_basis"][0] == N.KNOWN_FRAGMENT
        assert frame["novelty_parent"][0] == NISIN

    def test_a_masked_sequence_is_unverifiable_rather_than_new(self, table):
        frame = table([{"sequence": "GIGKFLHSAKKFXGKAFVGEIM", "peptide_name": "",
                        "provenance": "SAMPLE", "source_organism": ""}], [NISIN])
        assert frame["sequence_novelty"][0] == "Unverifiable"
        assert frame["novelty_basis"][0] == N.NOT_COMPARABLE

    def test_a_name_conflict_is_unverifiable_rather_than_new(self, table):
        frame = table([{"sequence": "WWPEFTCHMDQNTRYVKLWWPEFTC", "peptide_name": "nisin",
                        "provenance": "SAMPLE", "source_organism": ""}], [NISIN])
        assert frame["sequence_novelty"][0] == "Unverifiable"
        assert frame["name_agreement"][0] == N.NAME_DISAGREES

    def test_a_genuinely_new_sequence_is_still_claimed(self, table):
        frame = table([{"sequence": "WWPEFTCHMDQNTRYVKLWWPEFTC", "peptide_name": "",
                        "provenance": "SAMPLE", "source_organism": ""}], [NISIN])
        assert frame["sequence_novelty"][0] == "SAMPLE (new)"
        assert frame["novelty_basis"][0] == N.NOVEL

    def test_a_third_party_row_is_untouched_by_any_of_this(self, table):
        frame = table([{"sequence": NISIN, "peptide_name": "nisin",
                        "provenance": "Third Party", "source_organism": ""}], [NISIN])
        assert frame["sequence_novelty"][0] == "Third Party"
        assert frame["novelty_basis"][0] == ""

    def test_the_legacy_labels_are_all_still_in_the_vocabulary(self, table):
        """
        `Unverifiable` is an addition, not a replacement: anything already reading this column
        for the original three values keeps working.
        """
        rows = [{"sequence": MAGAININ, "peptide_name": "", "provenance": "SAMPLE",
                 "source_organism": ""},
                {"sequence": NISIN, "peptide_name": "", "provenance": "Third Party",
                 "source_organism": ""},
                {"sequence": "WWPEFTCHMDQNTRYVKLWWPEFTC", "peptide_name": "",
                 "provenance": "SAMPLE", "source_organism": ""}]
        labels = set(table(rows, [MAGAININ, NISIN])["sequence_novelty"])
        assert labels == {"Confirmed", "Third Party", "SAMPLE (new)"}


class TestStep12CatchesASequenceFiledUnderAnotherName:
    """
    `sequence_of_another_peptide` was the one failure mode on this field that nothing could
    detect: the sequence is structurally perfect so no rule objects, and a model shown only the
    document cannot know the name belongs elsewhere. The catalogue can.
    """

    @pytest.fixture()
    def rule(self, monkeypatch):
        from sample import field_review
        catalogue = N.Catalogue([MAGAININ, NISIN],
                               names={"nisin": {NISIN}, "magainin2": {MAGAININ},
                                      "naturalamps": {MAGAININ, NISIN} |
                                      {f"AAAAAAAAAA{index:02d}".replace("0", "G")
                                       for index in range(30)}},
                               ambiguous_at=20)
        monkeypatch.setattr(field_review, "_CATALOGUE", catalogue)
        return field_review.deterministic_verdict

    def test_the_rule_fires_on_a_mismatched_pairing(self, rule):
        verdict = rule("peptide_sequence", MAGAININ, {"peptide_name": "nisin"})
        assert verdict is not None
        assert verdict.failure_mode == "sequence_of_another_peptide"

    def test_the_message_names_both_sequences(self, rule):
        """
        The first version reported the matched sequence as the one the name identified, which
        produced "the name X identifies <this very sequence>, which this sequence does not match".
        """
        verdict = rule("peptide_sequence", MAGAININ, {"peptide_name": "nisin"})
        assert NISIN in verdict.reason and MAGAININ in verdict.reason

    def test_a_correct_pairing_escalates_to_the_model_instead(self, rule):
        assert rule("peptide_sequence", MAGAININ, {"peptide_name": "Magainin 2"}) is None

    def test_an_ambiguous_name_never_fires_the_rule(self, rule):
        """
        A class label must not condemn a sequence. Without the threshold, ``natural AMPs`` would
        be a verdict on every peptide it was ever filed beside.
        """
        assert rule("peptide_sequence", MAGAININ, {"peptide_name": "natural AMPs"}) is None

    def test_an_unknown_name_never_fires_the_rule(self, rule):
        assert rule("peptide_sequence", MAGAININ, {"peptide_name": "cnPGRP1"}) is None

    def test_a_missing_name_never_fires_the_rule(self, rule):
        assert rule("peptide_sequence", MAGAININ, {"peptide_name": ""}) is None

    def test_a_structurally_invalid_sequence_is_still_caught_by_the_earlier_rule(self, rule):
        """The name rule is added after the structural ones, not in place of them."""
        verdict = rule("peptide_sequence", "C10H22N6O4", {"peptide_name": "nisin"})
        assert verdict.failure_mode == "contains_digits"

    def test_the_rule_is_silent_when_no_catalogue_can_be_built(self, monkeypatch):
        """Step 12 running without one rule beats step 12 not running."""
        from sample import field_review
        monkeypatch.setattr(field_review, "_CATALOGUE", False)
        assert field_review.deterministic_verdict(
            "peptide_sequence", MAGAININ, {"peptide_name": "nisin"}) is None

    def test_the_failure_mode_is_now_measured(self):
        """It carried ``measured=False`` because nothing could count it. Now something can."""
        from sample.field_review import FIELD_SPECS
        spec = FIELD_SPECS["peptide_sequence"]
        mode = next(m for m in spec.failure_modes
                    if m.name == "sequence_of_another_peptide")
        assert mode.measured and mode.deterministic and mode.count == 865


class TestStep12ActuallySelectsTheRecordsTheRuleCatches:
    """
    A rule that fires only when something asks it is a rule that never fires. `select_targets`
    consulted the sequence rule only for hyphenated three-letter notation, so a one-letter
    sequence under a conflicting name was never selected and the finding was never recorded --
    865 firings when measured directly, zero in the run.
    """

    @pytest.fixture()
    def select(self, monkeypatch):
        from sample import field_review, field_review_run
        catalogue = N.Catalogue([MAGAININ, NISIN],
                                names={"nisin": {NISIN}, "magainin2": {MAGAININ}})
        monkeypatch.setattr(field_review, "_CATALOGUE", catalogue)
        return field_review_run.select_targets

    def _record(self, name, sequence):
        return {"peptide_name": name, "peptide_sequence": sequence,
                "biological_activities": []}

    def test_a_one_letter_sequence_under_a_conflicting_name_is_selected(self, select):
        targets = select(self._record("nisin", MAGAININ))
        assert [t for t in targets if t[0] == "peptide_sequence"
                and t[2] == "name_identifies_another_peptide"]

    def test_a_correctly_named_one_letter_sequence_is_still_not_selected(self, select):
        assert not [t for t in select(self._record("Magainin 2", MAGAININ))
                    if t[0] == "peptide_sequence"]

    def test_three_letter_notation_is_still_selected_for_the_model(self, select):
        """The pre-existing branch has to survive the restructure."""
        targets = select(self._record("cnPGRP1", "D-PHE-CYS-TYR-D-TRP-LYS-VAL-CYS-TRP-NH2"))
        assert [t for t in targets if t[2] == "three_letter_notation"]

    def test_a_field_is_not_selected_twice(self, select):
        """The two branches are exclusive; a record must not queue two sequence reviews."""
        targets = select(self._record("nisin", "ALA-ARG-LEU-TYR-LEU-ARG-ILE-GLY-NH2"))
        assert len([t for t in targets if t[0] == "peptide_sequence"]) <= 1
