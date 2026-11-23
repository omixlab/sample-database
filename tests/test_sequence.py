"""Tests for :mod:`sample.sequence` — amino-acid notation conversion and charge."""
from typing import List

import pytest

from sample.sequence import (AA_1_TO_3, AA_3_TO_1, get_peptide_charge, is_valid_peptide,
                             validate_extracted_sequence,
                             sequence_to_1_letter, sequence_to_3_letter, sequence_to_helm)


class TestAminoAcidTables:
    """The lookup tables the conversions are built on."""

    def test_tables_are_mutually_inverse(self) -> None:
        for three, one in AA_3_TO_1.items():
            if one in AA_1_TO_3:
                assert AA_1_TO_3[one].upper() == three.upper()

    def test_covers_twenty_standard_residues(self) -> None:
        assert len(set(AA_3_TO_1.values()) & set("ACDEFGHIKLMNPQRSTVWY")) == 20


class TestSequenceTo1Letter:
    def test_passes_through_one_letter(self) -> None:
        assert sequence_to_1_letter("GIGK") == "GIGK"

    def test_converts_three_letter(self) -> None:
        assert sequence_to_1_letter("Gly-Ile-Gly-Lys") == "GIGK"

    def test_converts_unseparated_three_letter(self) -> None:
        assert sequence_to_1_letter("GlyIleGlyLys") == "GIGK"

    def test_strips_whitespace_and_uppercases(self) -> None:
        assert sequence_to_1_letter("  gigk  ") == "GIGK"

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_input_is_empty_output(self, value: object) -> None:
        assert sequence_to_1_letter(value) == ""

    def test_strips_digits_and_punctuation(self) -> None:
        # OCR'd sequences arrive with residue numbering attached.
        assert "1" not in sequence_to_1_letter("G1I2G3K4")

    def test_strips_stereochemical_prefixes(self) -> None:
        assert sequence_to_1_letter("D-Ala-Gly-L-Lys") == "AGK"

    @pytest.mark.parametrize("raw,expected", [
        ("Ac-GIGK-NH2", "GIGK"), ("AC-GIGK", "GIGK"), ("NH2-GIGK", "GIGK"),
        ("NH2GIGK", "GIGK"), ("GIGK-NH2", "GIGK"), ("GIGKNH2", "GIGK"),
        ("GIGK-amide", "GIGK"), ("FITC-GIGK", "GIGK"), ("PEG-GIGK", "GIGK"),
    ])
    def test_strips_terminal_modifiers(self, raw: str, expected: str) -> None:
        assert sequence_to_1_letter(raw) == expected

    @pytest.mark.parametrize("seq", ["ACA", "ACDEFG", "PEGIK", "FITCGIGK", "GIGKAMIDE"])
    def test_does_not_eat_residues_that_spell_a_modifier(self, seq: str) -> None:
        # Regression, and a data-corruption one: the modifier patterns allowed zero separators,
        # so "ACA" (Ala-Cys-Ala) normalized to "A". 258 sequences in the shipped release were
        # truncated by two or three residues this way. AC, PEG, FITC and AMIDE are all valid
        # residue runs, so a separator is now required for them.
        assert sequence_to_1_letter(seq) == seq

    def test_nonstandard_residues_use_bracket_tokens(self) -> None:
        assert sequence_to_1_letter("Gly-Orn-Lys") == "G[Orn]K"

    def test_unseparated_nonstandard_triplets(self) -> None:
        assert sequence_to_1_letter("GlyOrnLys") == "G[Orn]K"

    @pytest.mark.parametrize("seq", ["G[Orn]K", "G[Dab]K", "G[Dap]K", "GIGK"])
    def test_is_idempotent(self, seq: str) -> None:
        # Regression: the bracket strip used to turn "G[Orn]K" into "GORNK", inventing an R and
        # an N that were never in the peptide. Callers cannot always know whether the string they
        # hold has already been normalized, so this must be a no-op.
        assert sequence_to_1_letter(seq) == seq
        assert sequence_to_1_letter(sequence_to_1_letter(seq)) == seq

    def test_normalizes_bracket_token_casing(self) -> None:
        assert sequence_to_1_letter("G[orn]K") == "G[Orn]K"

    def test_unconvertible_three_letter_parts_are_dropped(self) -> None:
        # "Xyz" is not a residue; it is dropped rather than guessed at.
        assert sequence_to_1_letter("Gly-Xyz-Lys") == "GK"

    def test_mixed_one_and_three_letter_parts(self) -> None:
        # OCR produces these: a one-letter residue sitting among hyphenated three-letter codes.
        assert sequence_to_1_letter("Gly-K-Lys") == "GKK"


class TestSequenceTo3Letter:
    def test_round_trips(self) -> None:
        assert sequence_to_1_letter(sequence_to_3_letter("GIGK")) == "GIGK"

    def test_empty(self) -> None:
        assert sequence_to_3_letter("") == ""

    def test_nonstandard_residue_kept_as_its_own_token(self) -> None:
        assert sequence_to_3_letter("Gly-Orn-Lys") == "Gly-Orn-Lys"

    @pytest.mark.parametrize("seq", ["JJJJ", "OOOO", "UUUU"])
    def test_unidentifiable_residue_yields_empty(self, seq: str) -> None:
        # Documented: return "" rather than emit a partial or wrong conversion. J, O and U have
        # no three-letter mapping in this table.
        assert sequence_to_3_letter(seq) == ""


class TestSequenceToHelm:
    def test_produces_helm_notation(self) -> None:
        helm = sequence_to_helm("GIGK")
        assert "PEPTIDE1" in helm
        assert "G.I.G.K" in helm

    def test_empty(self) -> None:
        assert sequence_to_helm("") == ""

    def test_nonstandard_residue_uses_bracketed_monomer_id(self) -> None:
        assert sequence_to_helm("Gly-Orn-Lys") == "PEPTIDE1{G.[Orn].K}$$$$"

    @pytest.mark.parametrize("seq", ["JJJJ", "OOOO", "UUUU"])
    def test_unidentifiable_residue_yields_empty(self, seq: str) -> None:
        assert sequence_to_helm(seq) == ""


class TestIsValidPeptide:
    """
    Note the contract here is deliberately permissive: the docstring admits three-letter codes,
    chirality prefixes (``D-Phe``) and non-proteinogenic residues, so hyphens, spaces and letters
    outside the standard twenty all pass by design.

    This is NOT the same predicate as the strict ``^[ACDEFGHIKLMNPQRSTVWY]+$`` check nested inside
    :func:`sample.merge.merge_database_and_extractions`, which rejects all of those. Two functions
    share the name and differ in contract; the tests pin each to its own.
    """

    @pytest.mark.parametrize("seq", ["GIGKFLHSAKKFGKAFVGEIMNS", "ACDEFGHIKLMNPQRSTVWY", "AA"])
    def test_accepts_standard_sequences(self, seq: str) -> None:
        assert is_valid_peptide(seq) is True

    @pytest.mark.parametrize("seq", ["Gly-Ile-Gly-Lys", "D-Phe", "Ala, Gly"])
    def test_accepts_modified_and_three_letter_forms(self, seq: str) -> None:
        assert is_valid_peptide(seq) is True

    @pytest.mark.parametrize("seq", ["", "   ", "GIGK1", "GIGK@", "GIGK("])
    def test_rejects_empty_and_non_letter_characters(self, seq: str) -> None:
        assert is_valid_peptide(seq) is False

    def test_rejects_sequences_over_fifty_residues(self) -> None:
        assert is_valid_peptide("A" * 50) is True
        assert is_valid_peptide("A" * 51) is False


class TestGetPeptideCharge:
    def test_cationic_peptide_is_positive(self) -> None:
        # Melittin carries 6 basic residues and no acidic ones.
        assert get_peptide_charge("GIGAVLKVLTTGLPALISWIKRKRQQ") > 3.0

    def test_anionic_peptide_is_negative(self) -> None:
        assert get_peptide_charge("DDDDEEEE") < -3.0

    def test_polylysine_scales_with_length(self) -> None:
        assert get_peptide_charge("KKKKKKKKKK") > get_peptide_charge("KKKKK")

    def test_ph_shifts_charge_in_the_right_direction(self) -> None:
        # Raising pH deprotonates: charge must fall, never rise.
        assert get_peptide_charge("KKKKK", ph=10.0) < get_peptide_charge("KKKKK", ph=4.0)

    def test_empty_sequence_is_zero(self) -> None:
        assert get_peptide_charge("") == 0.0

    def test_returns_float(self) -> None:
        assert isinstance(get_peptide_charge("GIGK"), float)

    def test_amidation_removes_the_c_terminal_negative_charge(self) -> None:
        # Regression: the amidation test used to run against the NORMALIZED sequence, from which
        # sequence_to_1_letter has already stripped the "-NH2" modifier, so it could never be
        # true and every amidated peptide was charged ~1.0 too low.
        free_acid = get_peptide_charge("GIGK")
        amidated = get_peptide_charge("GIGK-NH2")
        assert amidated - free_acid == pytest.approx(1.0, abs=0.01)

    def test_magainin_2_matches_its_published_charge(self) -> None:
        # Magainin-2 is C-terminally amidated and carries a net charge of about +4 at pH 7.
        assert get_peptide_charge("GIGKFLHSAKKFGKAFVGEIMNS-NH2") == pytest.approx(4.0, abs=0.2)

    def test_acidic_side_chains_reduce_charge(self) -> None:
        for residue in ("D", "E", "C", "Y"):
            assert get_peptide_charge("A" + residue + "A") < get_peptide_charge("AAA")


class TestValidateExtractedSequence:
    """
    The gate that decides whether an extracted string is usable as a sequence.

    Every rejected value below is verbatim from one real run's activity log. Measured over 1,554
    logged activities from 204 documents, only 14.4% of sequence fields were plain residue
    strings -- and every one of the rest passed the previous check, which asked only whether the
    field was non-empty.
    """

    @pytest.mark.parametrize("sequence,name,reason", [
        # The largest class: the model copying the name into the sequence field. A real
        # peptide's sequence is essentially never its own name.
        ("CHOLESTEROL", "Cholesterol", "name_copied_into_sequence"),
        ("RICIN", "Ricin", "name_copied_into_sequence"),
        ("PRIMAQUINE", "Primaquine", "name_copied_into_sequence"),
        ("TECHNETIUM-99M", "Technetium-99m", "name_copied_into_sequence"),
        ("MYRISTOYLGLYCINE", "myristoylglycine", "name_copied_into_sequence"),
        ("AC-GDLQVLAQK-AMIDE", "Ac-GDLQVLAQK-amide", "name_copied_into_sequence"),
        # Structure.
        ("COMPONENT 3", "", "contains_digits"),
        ("7.515.022.530.07,515.022.530.07,5", "Hormothamnion", "contains_digits"),
        ("21", "magainin-l", "contains_digits"),
        ("THE PEPTIDE WAS SYNTHESIZED IN OUR LABORATORY", "", "contains_whitespace"),
        # Caught by the raw-string digit rule before the length rule, which is the more
        # accurate reason: this is a table label, not a short peptide.
        ("N = 17", "Pseudomonas sp.", "contains_digits"),
        ("C2H5", "", "contains_digits"),
        ("ALA", "", "too_short"),
        ("G", "VSVIF G protein", "too_short"),
        ("PGSA", "magainin 2 amide", "too_short"),
        # Dictionary and nucleotide gates.
        ("UNKNOWN", "HNP 1", "dictionary_word_or_dna"),
        # OCR-noised nucleotide read: 89% ACGT, the Y and F introduced by OCR, so the pure-ACGT
        # test in sample.nlp could not see it.
        ("GCGTYFGCGGCGTGTGACC", "X (21 mer)", "looks_like_dna"),
        ("", "Magainin 2", "empty"),
    ])
    def test_real_junk_is_rejected_with_a_countable_reason(self, sequence, name, reason):
        ok, got = validate_extracted_sequence(sequence, name)
        assert ok is False
        assert got == reason

    @pytest.mark.parametrize("sequence,name", [
        ("GIGKFLHSAKKFGKAFVGEIMNS", "Magainin 2"),
        ("VTCYCRRTRCGFRERLSGACGYRGRIYRLCCR", "RatNP-1"),
        ("CSCRTSSCRFGERLSGACRLNGRIYRLCC", "RatNP-3"),
        ("ACYCRIGACVSGERLTGACGLNGRIYRLCCR", "RatNP-4"),
        ("GDLQVLAQK", "Ac-GDLQVLAQK-amide (I)"),
        ("CYKQKIIAP", "Ac-CYKQKIIAP-amide (IV)"),
        ("VEEVIEMQGLSLDPTSHRVMAGEEPLEMGPTEFKLLHFFMTHPERVYSREQLLN", ""),
    ])
    def test_real_sequences_survive(self, sequence, name):
        """
        The half that matters more. A gate that deletes genuine records to catch junk is worse
        than the junk -- which is why sample.rf_filter is deliberately not used here: measured on
        this corpus it rejects ACYCRIPACIA, the core of human neutrophil peptide 1.
        """
        ok, reason = validate_extracted_sequence(sequence, name)
        assert ok is True, f"rejected a real sequence as {reason!r}"

    @pytest.mark.parametrize("sequence", [
        "GLY-ARG-GLY-ASP-SER",
        "D-PHE-CYS-TYR-D-TRP-LYS-VAL-CYS-TRP-NH2",
    ])
    def test_three_letter_and_chirality_forms_are_normalised_first(self, sequence):
        """These are legitimate renderings and must be judged on what they convert to."""
        ok, reason = validate_extracted_sequence(sequence, "")
        assert ok is True, f"rejected a three-letter form as {reason!r}"

    def test_space_separated_residues_are_accepted(self):
        """
        A figure or table OCR'd one residue at a time is still a sequence; the normaliser
        collapses it before the whitespace rule is applied.
        """
        ok, _ = validate_extracted_sequence("I I G G R E S R P H S R P Y M A", "HNP-4")
        assert ok is True

    def test_a_whole_paragraph_is_rejected_by_length(self):
        ok, reason = validate_extracted_sequence("A" * 500, "")
        assert ok is False
        assert reason == "too_long"

    def test_the_name_check_ignores_case_and_punctuation(self):
        ok, reason = validate_extracted_sequence("ACETYLCYSTEINE", "Acetyl-cysteine")
        assert ok is False
        assert reason == "name_copied_into_sequence"

    def test_a_short_acgt_peptide_is_not_called_dna(self):
        """
        The ratio test applies only from 12 residues up: short peptides genuinely made of these
        four letters exist and are not evidence of anything.
        """
        ok, _ = validate_extracted_sequence("GATCGATC", "")
        # Rejected or not, it must not be rejected *as DNA* on the ratio rule.
        assert _ != "looks_like_dna"

    def test_a_compound_name_in_residue_letters_is_now_rejected(self):
        """
        This test previously asserted the OPPOSITE, as a standing record of a known gap: a drug
        name spelled entirely in residue letters passed every structural test, because every
        letter in it is a residue. ``name_likelihood`` closed it.
        """
        ok, reason = validate_extracted_sequence("ENKEPHALIN", "ME")
        assert ok is False
        assert reason == "looks_like_a_name"

class TestTheRawStringIsJudgedBeforeConversion:
    """
    Regression for a hole that shipped and was caught in a live run's log.

    ``sequence_to_1_letter`` STRIPS digits and chemical punctuation, so judging only the
    converted form means judging a string the extractor never saw:
    ``C10H22N6O4・H2O`` became ``CHNOHO``, ``C31H47N6O10CLC34H34N7O8CL`` became
    ``CHNOCLCHNOCL``, ``2ABCDEFGH`` became ``ABCDEFGH`` -- each then passed every structural
    check and was published as a peptide sequence.
    """

    @pytest.mark.parametrize("sequence", [
        "C10H22N6O4\u30fbH2O",
        "C31H47N6O10CLC34H34N7O8CL",
        "C7H10NO8CL",
        "CH3(CH2)6CH=CHCH=CHCOOH",
        "2ABCDEFGH",
        "1510152023G-I-G-K-F-L-H-S-A-K-K-F-G-K-A-F-V-G-E-I-M-N-S",
        ">10",
        "39\u00b165",
        "ORN(HCO)2'2',APHE4'4'",
        "[D-DPR(Z)4,4']GS",
    ])
    def test_a_formula_or_label_is_rejected_on_the_raw_string(self, sequence):
        ok, reason = validate_extracted_sequence(sequence, "")
        assert ok is False, "the conversion stripped the evidence and the string got through"
        assert reason in ("contains_digits", "non_residue_characters")

    @pytest.mark.parametrize("sequence", [
        # The one place digits are legitimate: a terminal group on a three-letter sequence.
        "D-PHE-CYS-TYR-D-TRP-LYS-VAL-CYS-TRP-NH2",
        "TRP-GLY-LYS-PRO-VAL-NH2",
        "GLY-ARG-GLY-ASP-SER",
    ])
    def test_a_terminal_group_does_not_make_a_sequence_a_formula(self, sequence):
        """
        'NH2' carries a digit. A blanket digit rule would reject octreotide's published form,
        which is why the digit test is conditional on the string parsing as three-letter codes.
        """
        ok, reason = validate_extracted_sequence(sequence, "")
        assert ok is True, f"rejected a published three-letter form as {reason!r}"

    def test_the_three_letter_test_needs_a_majority_not_every_token(self):
        """
        Real published forms carry the occasional modified residue that is in none of the
        lookup tables; demanding every token match would reject them.
        """
        from sample.sequence import _looks_like_three_letter_sequence as L

        assert L("GLY-ARG-GLY-ASP-SER") is True
        assert L("D-PHE-CYS-TYR-D-TRP-LYS-VAL-CYS-TRP-NH2") is True
        assert L("C10H22N6O4") is False
        assert L("RC-160") is False

class TestTheNameDiscriminator:
    """
    Telling a peptide sequence from a compound name spelled in residue letters -- the class no
    structural test can reject, since every letter in it IS a residue.

    Measured on held-out data (40,000 catalogued sequences against 17,300 English words spelled
    only in residue letters): peptide median +1.99, name median -1.27. All 17 offenders observed
    in a real run score between -0.61 and -2.05.
    """

    @pytest.mark.parametrize("name", [
        "ANTIMYCINA", "METHENAMINEMANDELATE", "TETRACYCLINES", "CANDIDASPECIES",
        "SANDOSTATIN", "ASCIDIACYCLAMIDE", "ACETYLATED", "ENKEPHALIN", "LINOLEOYL",
        "RETINAMIDE", "DPHENYLALANINE", "PRIMAQUINE", "DAUNORUBICIN",
    ])
    def test_real_compound_names_score_as_names(self, name):
        from sample.sequence import NAME_LIKELIHOOD_THRESHOLD, name_likelihood

        score = name_likelihood(name)
        assert score is not None, "the model file is missing; the gate is off"
        assert score < NAME_LIKELIHOOD_THRESHOLD, f"{name} scored {score:+.2f}"

    @pytest.mark.parametrize("sequence", [
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "VTCYCRRTRCGFRERLSGACGYRGRIYRLCCR",
        "DCYCRIPACIAGERRYGTCIYQGRLWAFCC",
        "ALWKNMLKGIGKLAGKAALGAVKKLVGAES",
        "KILRGVCKKIMRTFLRRISKDILTGKK",
        "QVYKGGYTRPIPRPPPFVRPLPGGPIGPYNGCPVSCRGISFSQARSCC",
    ])
    def test_real_sequences_score_as_sequences(self, sequence):
        from sample.sequence import NAME_LIKELIHOOD_THRESHOLD, name_likelihood

        score = name_likelihood(sequence)
        assert score is not None
        assert score >= NAME_LIKELIHOOD_THRESHOLD, f"{sequence} scored {score:+.2f}"

    def test_the_hardest_real_case_survives_with_margin(self):
        """
        ACYCRIPACIA is the human neutrophil peptide 1 core -- a real published sequence that
        ``sample.rf_filter`` rejects. It scores about -0.05, which is why the threshold is -0.5
        and not 0: the asymmetry is deliberate, because losing a real sequence is worse than
        keeping a name that other gates will catch.
        """
        from sample.sequence import NAME_LIKELIHOOD_THRESHOLD, name_likelihood

        score = name_likelihood("ACYCRIPACIA")
        assert score > NAME_LIKELIHOOD_THRESHOLD + 0.2, f"margin lost: {score:+.2f}"

    def test_a_string_too_short_to_score_gets_no_opinion(self):
        """
        None, not a verdict. A caller must be able to tell "no opinion" from "looks like a
        name", or a short real sequence would be rejected on no evidence.
        """
        from sample.sequence import name_likelihood

        assert name_likelihood("GIGK") is None

    def test_a_missing_model_disables_the_test_rather_than_the_pipeline(self, monkeypatch):
        """
        A gate that cannot load its model must fail open and say so, not reject everything.
        """
        import sample.sequence as S

        monkeypatch.setattr(S, "_BIGRAM_MODEL", None)
        monkeypatch.setattr(S, "_BIGRAM_MODEL_TRIED", False)
        monkeypatch.setattr(S, "BIGRAM_MODEL_PATH", "/definitely/not/here.json")
        assert S.name_likelihood("GIGKFLHSAKKFGKAFVGEIMNS") is None
        # and the structural layers still work
        ok, _ = S.validate_extracted_sequence("GIGKFLHSAKKFGKAFVGEIMNS", "Magainin 2")
        assert ok is True

    def test_the_model_file_is_committed(self):
        """
        A gate that depends on a build step is a gate that is off on a fresh clone.
        """
        import os

        from sample.sequence import BIGRAM_MODEL_PATH

        assert os.path.exists(BIGRAM_MODEL_PATH), "train with sample-cli train-sequence-discriminator"

