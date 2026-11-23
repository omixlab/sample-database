"""
Tests for :mod:`sample.verdict_apply`.

Weighted towards the refusals. Accepting a good proposal is the easy half; the reason this module
exists is that step 12's proposals are frequently unusable as written while still being right in
substance, and writing them verbatim would damage fields that are currently merely incomplete.
Each refusal test below names a real proposal taken from the corpus's own recorded verdicts.
"""

import json
import os

import pytest

from sample import verdict_apply as va
from sample.validation_rules import VALID_ACTIVITY_TYPES, VALID_UNITS


# --- activity_type --------------------------------------------------------------------------------

def test_exact_vocabulary_type_is_accepted_unchanged():
    assert va.authorize_activity_type("MIC") == ("MIC", "vocabulary:exact")


@pytest.mark.parametrize("spelling, expected", [
    ("mic", "MIC"), ("Mic", "MIC"), ("ic50", "IC50"), ("Ec50", "EC50"),
    ("hemolysis", "Hemolysis"), ("cytotoxicity", "Cytotoxicity"),
    ("growth inhibition", "Growth Inhibition"), ("zoi", "ZOI"),
])
def test_case_variants_fold_to_the_vocabulary_spelling(spelling, expected):
    """3,810 activities in the corpus differ from the vocabulary only in case."""
    value, authority = va.authorize_activity_type(spelling)
    assert (value, authority) == (expected, "vocabulary:case")
    assert value in VALID_ACTIVITY_TYPES


@pytest.mark.parametrize("phrase, expected", [
    ("minimum inhibitory concentration", "MIC"),
    ("minimum bactericidal concentration", "MBC"),
    ("inhibition of growth", "Growth Inhibition"),
    ("zone of inhibition", "ZOI"),
    ("MTT assay", "Cytotoxicity"),
])
def test_prose_proposals_map_through_the_phrase_table(phrase, expected):
    """The review answers this field in prose more often than in vocabulary."""
    value, _ = va.authorize_activity_type(phrase)
    assert value == expected


def test_an_activity_class_is_not_a_measurement():
    """'antimicrobial activity' names a class of activity, not an assay. 2 of the corpus's most
    common proposals for this field read exactly that way."""
    value, why = va.authorize_activity_type("antimicrobial activity")
    assert value == ""
    assert why == "not_in_activity_vocabulary"


def test_context_sentence_settles_a_type_the_proposal_cannot():
    value, authority = va.authorize_activity_type(
        "concentration", context="The minimum inhibitory concentration against E. coli was 8 ug/mL")
    assert (value, authority) == ("MIC", "phrase:context")


# --- activity_unit --------------------------------------------------------------------------------

@pytest.mark.parametrize("proposal, expected", [
    ("mg/ml", "mg/mL"), ("MG/ml", "mg/mL"), ("ug/ml", "ug/mL"), ("µM", "uM"),
    ("umol/L", "umol/L"), ("nM", "nM"),
])
def test_unit_proposals_land_on_the_vocabulary_spelling(proposal, expected):
    value, _ = va.authorize_unit(proposal)
    assert value == expected
    assert value in VALID_UNITS


def test_case_carries_meaning_in_the_numerator():
    """canonicalize_unit maps all-caps MG/ML to ug/mL by an explicit OCR-repair entry: a capital
    numerator is how a mis-OCR'd micro sign reaches this corpus, and the two readings differ by a
    factor of 1000. authorize_unit runs that repair before folding case, so the project's decision
    stands rather than being folded away."""
    assert va.authorize_unit("MG/ML")[0] == "ug/mL"
    assert va.authorize_unit("Mg/mL")[0] == "ug/mL"
    assert va.authorize_unit("mg/mL")[0] == "mg/mL"


def test_ppm_is_converted_because_the_identity_is_exact():
    """1 ppm by mass in a dilute aqueous solution is 1 mg/L, which the vocabulary spells ug/mL.
    The number does not change, so this is a relabelling and not a conversion."""
    value, authority = va.authorize_unit("ppm")
    assert (value, authority) == ("ug/mL", "synonym:exact_identity")


def test_an_assay_name_is_refused_as_a_unit():
    """MIC is a real token the review offers for this field. Moving it to activity_type is a
    different correction with a different precondition, so it is refused here rather than
    quietly rewritten."""
    value, why = va.authorize_unit("MIC")
    assert value == ""
    assert why == "names_an_assay_not_a_unit"


@pytest.mark.parametrize("debris", ["A/CO", "continued", "calc", "m/z", "I", "fH", "PMM"])
def test_table_debris_is_refused(debris):
    """Each of these occupies the unit column in the corpus; none denotes a quantity."""
    value, _ = va.authorize_unit(debris)
    assert value == ""


# --- target_organism ------------------------------------------------------------------------------

def test_a_resolvable_binomial_gets_its_tax_id():
    stored, taxid, authority = va.authorize_organism("Escherichia coli")
    assert stored == "Escherichia coli"
    assert taxid == 562
    assert authority == "ncbi_taxonomy"


@pytest.mark.parametrize("conjunction", [
    "Acinetobacter baumannii or Pseudomonas aeruginosa",
    "Staphylococcus aureus and Escherichia coli",
    "Listeria monocytogenes, Streptococcus suis, Campylobacter jejuni, and ",
    "E. coli / S. aureus",
])
def test_a_conjunction_is_refused_before_any_lookup(conjunction):
    """The decisive case. Every one of these has a prefix that resolves perfectly well, so a
    lookup would return a confident answer and file the measurement against whichever organism
    happened to be written first. All four are real proposals from the corpus."""
    stored, taxid, why = va.authorize_organism(conjunction)
    assert stored == ""
    assert taxid is None
    assert why == "names_more_than_one_organism"


@pytest.mark.parametrize("fused", [
    "S. epidermidis andE. faecium",
    "Staphylococcus aureus andEscherichia coli",
    "S. aureus E. coli",
])
def test_an_ocr_fused_conjunction_is_still_two_organisms(fused):
    """This corpus's OCR drops the space between words, so a conjunction arrives fused to the next
    genus and \\band\\b does not match it. The first of these resolved cleanly to Staphylococcus
    epidermidis in a full corpus pass, filing a two-organism measurement against one of them."""
    stored, _, why = va.authorize_organism(fused)
    assert stored == ""
    assert why == "names_more_than_one_organism"


def test_a_microbial_assay_refuses_an_animal_target():
    """An MIC is a concentration against a microbe. The review's failure mode here is naming the
    peptide's SOURCE organism as its target -- a crab AMP's MIC filed against Scopimera globosa,
    which NCBI confirms is a real species, so no name-based check would question it."""
    record = {"peptide_name": "X", "biological_activities": [
        {"activity_type": "MIC", "activity_unit": "ug/mL", "target_organism": ""}]}
    verdicts = [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                 "corrected": "Scopimera globosa", "peptide": "X"}]
    changes, refusals = va.apply_verdicts_to_record(record, verdicts)
    assert record["biological_activities"][0]["target_organism"] == ""
    assert changes == []
    assert refusals[0].why == "domain_impossible_for_a_microbial_assay"


def test_hemolysis_still_accepts_an_animal_target():
    """The guard is scoped to assays that measure against a microbe. Hemolysis runs on
    erythrocytes, so an animal target is correct there and must not be refused."""
    record = {"peptide_name": "X", "biological_activities": [
        {"activity_type": "Hemolysis", "activity_unit": "ug/mL", "target_organism": ""}]}
    verdicts = [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                 "corrected": "Homo sapiens", "peptide": "X"}]
    changes, _ = va.apply_verdicts_to_record(record, verdicts)
    assert record["biological_activities"][0]["target_organism"] == "Homo sapiens"
    assert changes


def test_a_microbial_assay_accepts_a_microbial_target():
    assert va.domain_permits_target("MIC", "Escherichia coli") is True
    assert va.domain_permits_target("MBC", "Candida albicans") is True
    assert va.domain_permits_target("MIC", "Homo sapiens") is False


@pytest.mark.parametrize("junk", ["unsure", "n/a", "unknown", "not specified", "-", ""])
def test_placeholder_organism_proposals_are_refused(junk):
    stored, _, why = va.authorize_organism(junk)
    assert stored == ""
    assert why == "placeholder"


@pytest.mark.parametrize("clade", [
    "bacteria", "Gram-positive bacteria", "fungi", "cancer cells", "erythrocytes",
])
def test_a_rank_above_species_is_refused_although_it_resolves(clade):
    """These resolve against NCBI, which is why they have to be refused explicitly: a clade is
    not an organism a measurement was made on, and target_group already carries that level."""
    stored, _, why = va.authorize_organism(clade)
    assert stored == ""
    assert why in ("rank_above_species", "resolves_to_a_rank_above_species")


@pytest.mark.parametrize("artifact, taxid", [("Zeta", 743421), ("Helix", 6534)])
def test_a_single_token_name_gets_no_tax_id(artifact, taxid):
    """What stops a table artifact being dignified with an identifier. "Zeta" comes from "zeta
    potential" and "Helix" from "alpha-helix"; both are real NCBI genera, both sat in
    target_organism against an MIC, and a first corpus pass attached their ids -- which makes the
    artifact look like a curated target. A binomial has two tokens; a genus has one."""
    stored, resolved_id, why = va.authorize_organism(artifact)
    assert stored == ""
    assert resolved_id is None
    assert why == "resolves_to_a_single_token_name"


def test_a_multi_token_virus_name_is_still_accepted():
    """PRRSV resolves to a name containing the word "and". The conjunction guard reads the
    proposal, not the resolution, so a legitimate organism whose official name contains "and" is
    not refused for it."""
    stored, taxid, _ = va.authorize_organism("PRRSV")
    assert "Porcine reproductive and respiratory syndrome virus" in stored
    assert taxid is not None


def test_the_documents_own_genera_settle_an_abbreviation():
    """NCBI registers 'm. luteus' as Methylobacter luteus, which is valid and never what this
    literature means. The document spells out Micrococcus; that decides it."""
    stored, taxid, authority = va.authorize_organism(
        "M. luteus", document_genera=frozenset({"Micrococcus", "Staphylococcus"}))
    assert stored == "Micrococcus luteus"
    assert authority == "document_genera"
    assert taxid is not None


def test_a_strain_suffix_is_kept_while_the_binomial_resolves():
    stored, taxid, _ = va.authorize_organism("Escherichia coli ATCC 25922")
    assert stored.startswith("Escherichia coli")
    assert "ATCC 25922" in stored
    assert taxid == 562


# --- peptide_sequence -----------------------------------------------------------------------------

def test_three_letter_notation_decodes():
    value, authority = va.authorize_sequence("Ala-Val-Leu-Gln-Ser")
    assert (value, authority) == ("AVLQS", "three_letter_decode")


def test_terminal_groups_are_stripped_before_decoding():
    value, _ = va.authorize_sequence("Ac-Gly-Ile-Gly-NH2")
    assert value == "GIG"


def test_a_separated_stereodescriptor_is_not_read_as_a_residue():
    """The bug this guards. The separator that makes three-letter code readable also splits
    "D-Ala" into "D" and "Ala", and D is a valid one-letter code for aspartate -- so a parser
    that accepts both notations at once reads this as Asp-Ala-Leu-Lys-Gly and invents two
    residues. The notation is settled before any token is parsed."""
    value, _ = va.authorize_sequence("D-Ala-L-Lys-Gly")
    assert value == "AKG"


def test_a_hyphenated_one_letter_run_is_still_read_as_one_letter():
    value, authority = va.authorize_sequence("G-I-G-K")
    assert (value, authority) == ("GIGK", "hyphenated_one_letter")


def test_a_non_canonical_residue_refuses_the_whole_proposal():
    """The decisive sequence case. sequence_to_1_letter drops Aib, Vxx and Lxxol and returns a
    short, well-formed sequence that is not the peptide in the paper. This is a real trichogin
    analogue proposal from the corpus."""
    value, why = va.authorize_sequence(
        "Ac-Aib-Ala-Aib-Ala-Aib-Gln-Aib-Vxx-Aib-Ala-Aib-Aib-Ala-Gln-Lxxol")
    assert value == ""
    assert why.startswith("unconvertible_residue:")


def test_an_unseparated_three_letter_run_decodes():
    """The real shape: hyphens lost in PDF extraction. This is the TAT peptide."""
    value, authority = va.authorize_sequence("GLYARGLYSLYSARGARGGLNARGARGARGCYSGLY")
    assert value == "GRKKRRQRRRCG"
    assert authority == "unseparated_three_letter_decode"


def test_an_unseparated_run_below_the_residue_floor_is_refused():
    """GLYILEGLY decodes to three residues. Four is the floor, because a shorter run is as
    likely to be a genuine one-letter sequence as a three-letter one."""
    value, _ = va.authorize_sequence("GLYILEGLY")
    assert value == "GLYILEGLY"


def test_an_inexact_partition_is_refused_rather_than_shortened():
    """The forgiving decoder in novelty.py skips a code it cannot read, which would turn this
    into VPGG -- a four-residue peptide where the paper reported five, with the ambiguous
    position silently gone. GLP (pyroglutamate) is the same case."""
    value, _ = va.authorize_sequence("GLPHISTRPSERTYRLEUARGPROGLY")
    assert value == "GLPHISTRPSERTYRLEUARGPROGLY"


@pytest.mark.parametrize("corrupted", [
    "KWKLFKKIŸPŸKFLHSAKKF",
    "ILIKKDLKÜYÇEARANAES",
    "DDAPIIIUBÖGMP",
])
def test_a_stray_character_refuses_rather_than_closing_the_gap(corrupted):
    """Deleting what is not a letter looks harmless and is not: the first of these would become
    KWKLFKKIPKFLHSAKKF, two residues shorter than the peptide the paper reported, and nothing
    downstream could detect it. All three are real stored sequences from the corpus."""
    value, why = va.authorize_sequence(corrupted)
    assert value == ""
    assert why == "non_residue_character_in_sequence"


def test_a_bracketed_non_standard_residue_is_refused():
    """G[Orn]K must not become GORNK, which invents an arginine and an asparagine."""
    value, why = va.authorize_sequence("G[Orn]K")
    assert value == ""
    assert why == "bracketed_non_standard_residue"


@pytest.mark.parametrize("spelled, expected", [
    ("IAWVKAFIRKLRK-AMIDE", "IAWVKAFIRKLRK"),
    ("AC-SYSMEHFRWGKPV-NH2", "SYSMEHFRWGKPV"),
    ("NH2-GLLRKGGEKIGEKLKKIGQKIKNFFQKLVPQPEQ-COOH",
     "GLLRKGGEKIGEKLKKIGQKIKNFFQKLVPQPEQ"),
    ("DKWSAFLKECSTIAQIYPLQEI-OH", "DKWSAFLKECSTIAQIYPLQEI"),
])
def test_stacked_terminal_groups_come_off_both_ends(spelled, expected):
    """A terminal group qualifies the peptide without changing its residues, and they stack."""
    assert va.authorize_sequence(spelled)[0] == expected


def test_a_one_letter_sequence_passes_through():
    value, authority = va.authorize_sequence("GIGKFLHSAK")
    assert (value, authority) == ("GIGKFLHSAK", "already_one_letter")


# --- value_raw ------------------------------------------------------------------------------------

def test_a_proposal_carrying_a_different_unit_is_refused():
    """'3.8 +/- 0.4 ug/mL' against a stored unit of uM is not a corrected number -- it is a
    different measurement, and nothing in the proposal says which row it came from."""
    fields, why = va.authorize_value("3.8 ± 0.4 µg/mL", {"activity_unit": "uM"})
    assert fields == {}
    assert why.startswith("proposal_carries_a_different_unit")


def test_a_bare_number_is_accepted_and_parsed():
    fields, _ = va.authorize_value("36.29", {"activity_unit": "ug/mL"})
    assert fields["value_lower"] == pytest.approx(36.29)
    assert fields["value_raw"] == "36.29"


def test_a_standard_deviation_keeps_only_the_mean():
    fields, _ = va.authorize_value("167 ± 8", {"activity_unit": "ug/mL"})
    assert fields["value_lower"] == pytest.approx(167.0)


def test_a_magnitude_outside_the_plausible_band_is_refused():
    fields, why = va.authorize_value("4000", {"activity_unit": "ug/mL"})
    assert fields == {}
    assert why == "proposed_magnitude_outside_the_plausible_band"


# --- the mechanical layer -------------------------------------------------------------------------

def test_case_variant_type_is_folded_without_a_verdict():
    activity = {"activity_type": "mic", "value_raw": "8", "activity_unit": "ug/mL"}
    changes = va.mechanical_type_fixes(activity)
    assert activity["activity_type"] == "MIC"
    assert [c.layer for c in changes] == ["mechanical"]


def test_a_metric_in_the_unit_column_moves_to_the_type():
    activity = {"activity_type": "N/A", "activity_unit": "MIC", "value_raw": "16"}
    changes = va.mechanical_unit_fixes(activity)
    assert activity["activity_type"] == "MIC"
    assert activity["activity_unit"] == ""
    assert activity["activity_unit_original"] == "MIC"
    assert len(changes) == 2


def test_a_metric_in_the_unit_column_that_conflicts_only_clears_the_unit():
    """activity_type IC50 with activity_unit MBC names two assays. Nothing in the record says
    which was measured, so the type is left alone."""
    activity = {"activity_type": "IC50", "activity_unit": "MBC", "value_raw": "16"}
    va.mechanical_unit_fixes(activity)
    assert activity["activity_type"] == "IC50"
    assert activity["activity_unit"] == ""


def test_ppm_is_relabelled_without_touching_the_number():
    activity = {"activity_type": "MIC", "activity_unit": "ppm", "value_raw": "32",
                "value_lower": 32.0}
    va.mechanical_unit_fixes(activity)
    assert activity["activity_unit"] == "ug/mL"
    assert activity["value_raw"] == "32"
    assert activity["value_lower"] == 32.0


def test_an_abbreviated_genus_is_expanded_and_gets_a_tax_id():
    activity = {"target_organism": "E. coli", "activity_type": "MIC"}
    changes = va.mechanical_organism_fixes(
        activity, document_genera=frozenset({"Escherichia"}))
    assert activity["target_organism"] == "Escherichia coli"
    assert activity["target_taxid"] == 562
    assert activity["target_organism_original"] == "E. coli"
    assert changes


def test_a_resolvable_organism_that_needs_no_rewrite_still_gets_its_tax_id():
    """The identifier is the join key, and 24,399 filled organism fields had none."""
    activity = {"target_organism": "Staphylococcus aureus"}
    changes = va.mechanical_organism_fixes(activity)
    assert activity["target_organism"] == "Staphylococcus aureus"
    assert activity["target_taxid"] == 1280
    assert [c.field for c in changes] == ["target_taxid"]


def test_a_placeholder_organism_is_cleared():
    activity = {"target_organism": "n/a"}
    va.mechanical_organism_fixes(activity)
    assert activity["target_organism"] == ""


def test_cell_line_spelling_is_canonicalized(tmp_path):
    names = tmp_path / "cellosaurus_names.txt"
    names.write_text("RAW 264.7\nMCF-7\nHeLa\n", encoding="utf-8")
    index = va.cellosaurus_index(str(names))
    activity = {"target_cell_line": "raw264.7"}
    va.mechanical_cell_line_fixes(activity, index)
    assert activity["target_cell_line"] == "RAW 264.7"
    assert activity["target_cell_line_original"] == "raw264.7"


def test_three_letter_sequence_is_decoded_in_place_and_the_original_kept():
    record = {"peptide_sequence": "Gly-Ile-Gly-Lys-Phe"}
    changes = va.mechanical_sequence_fixes(record)
    assert record["peptide_sequence"] == "GIGKF"
    assert record["peptide_sequence_3letter"] == "Gly-Ile-Gly-Lys-Phe"
    assert record["peptide_sequence_1letter"] == "GIGKF"
    assert changes


def test_a_non_canonical_sequence_is_left_alone():
    record = {"peptide_sequence": "Ac-Aib-Ala-Vxx-Lol"}
    changes = va.mechanical_sequence_fixes(record)
    assert record["peptide_sequence"] == "Ac-Aib-Ala-Vxx-Lol"
    assert changes == []


# --- verdict application --------------------------------------------------------------------------

def test_a_name_conflict_never_rewrites_the_sequence():
    """1,059 verdicts carry this reason. The verdict says the name and the sequence disagree; it
    does not say which is wrong, and rewriting the sequence would replace what the paper reported
    with what the catalogue holds."""
    record = {"peptide_name": "Defensin 1", "peptide_sequence": "ALVLLAFQVQADPIQNT",
              "biological_activities": []}
    verdicts = [{"field": "peptide_sequence", "verdict": "wrong",
                 "reason": "name_identifies_another_peptide",
                 "corrected": "ACQFWSCNSSCISRGYRQGYCWGIQYKYCQCQ",
                 "peptide": "Defensin 1"}]
    changes, refusals = va.apply_verdicts_to_record(record, verdicts)
    assert record["peptide_sequence"] == "ALVLLAFQVQADPIQNT"
    assert changes == []
    assert refusals[0].why == "name_sequence_conflict_is_not_a_sequence_error"


def test_a_unit_already_in_the_vocabulary_is_not_overridden_by_a_proposal():
    """Step 11 settled these deterministically. A model proposal does not override a unit the
    vocabulary accepts, or this stage would partly undo step 11's 2,797 repairs."""
    record = {"peptide_name": "LL-37", "biological_activities": [
        {"activity_type": "MIC", "activity_unit": "ug/mL", "value_raw": "8"}]}
    verdicts = [{"field": "activity_unit", "verdict": "wrong", "reason": "implausible_magnitude",
                 "corrected": "mM", "peptide": "LL-37"}]
    changes, refusals = va.apply_verdicts_to_record(record, verdicts)
    assert record["biological_activities"][0]["activity_unit"] == "ug/mL"
    assert changes == []
    assert refusals[0].why == "current_unit_already_in_vocabulary"


def test_an_unsure_verdict_is_never_applied():
    record = {"peptide_name": "X", "biological_activities": [
        {"activity_type": "N/A", "activity_unit": "ug/mL"}]}
    verdicts = [{"field": "activity_type", "verdict": "unsure", "reason": "placeholder_type",
                 "corrected": "MIC", "peptide": "X"}]
    changes, _ = va.apply_verdicts_to_record(record, verdicts)
    assert changes == []


def test_an_organism_proposal_fills_an_empty_target_and_attaches_the_id():
    record = {"peptide_name": "X", "biological_activities": [
        {"activity_type": "MIC", "activity_unit": "ug/mL", "target_organism": ""}]}
    verdicts = [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                 "corrected": "Pseudomonas aeruginosa", "peptide": "X"}]
    changes, _ = va.apply_verdicts_to_record(record, verdicts)
    activity = record["biological_activities"][0]
    assert activity["target_organism"] == "Pseudomonas aeruginosa"
    assert activity["target_taxid"] == 287
    assert changes[0].layer == "authorized"


def test_an_organism_proposal_does_not_overwrite_one_that_already_resolves():
    record = {"peptide_name": "X", "biological_activities": [
        {"activity_type": "MIC", "target_organism": "Staphylococcus aureus"}]}
    verdicts = [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                 "corrected": "Escherichia coli", "peptide": "X"}]
    changes, _ = va.apply_verdicts_to_record(record, verdicts)
    assert record["biological_activities"][0]["target_organism"] == "Staphylococcus aureus"
    assert changes == []


# --- the pass as a whole --------------------------------------------------------------------------

def _document(tmp_path, payload, name="doc.json"):
    directory = tmp_path / "pubmed"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_dry_run_writes_nothing(tmp_path):
    payload = {"peptides": [{"peptide_name": "X", "peptide_sequence": "GIGK",
                             "biological_activities": [{"activity_type": "mic",
                                                        "activity_unit": "ppm",
                                                        "value_raw": "8"}]}]}
    path = _document(tmp_path, payload)
    before = open(path, encoding="utf-8").read()
    summary = va.run(base_dir=str(tmp_path), dry_run=True)
    assert summary["changes"] > 0
    assert open(path, encoding="utf-8").read() == before


def test_the_pass_is_idempotent(tmp_path):
    payload = {"peptides": [{"peptide_name": "X", "peptide_sequence": "GIGK",
                             "biological_activities": [{"activity_type": "mic",
                                                        "activity_unit": "ppm",
                                                        "value_raw": "8",
                                                        "target_organism": "E. coli"}]}]}
    path = _document(tmp_path, payload)
    first = va.run(base_dir=str(tmp_path), dry_run=False)
    assert first["documents_changed"] == 1
    second = va.run(base_dir=str(tmp_path), dry_run=False)
    assert second["changes"] == 0
    assert second["skipped_already_applied"] == 1
    stored = json.loads(open(path, encoding="utf-8").read())
    activity = stored["peptides"][0]["biological_activities"][0]
    assert activity["activity_type"] == "MIC"
    assert activity["activity_unit"] == "ug/mL"
    assert stored["metadata"][va.APPLY_KEY]["version"] == va.APPLY_VERSION


def test_a_verdict_is_refused_when_its_peptide_name_is_not_unique(tmp_path):
    """A verdict identifies its peptide by name and nothing else. Two peptides sharing a name --
    and 10.6% of verdicts name the empty string, which 428 documents have more than one of --
    make it unattributable, and applying it to each in turn spreads one proposal across peptides
    the review never looked at."""
    payload = {"peptides": [
        {"peptide_name": "", "peptide_sequence": "GIGKFLHSAK",
         "biological_activities": [{"activity_type": "MIC", "activity_unit": "ug/mL",
                                    "target_organism": ""}]},
        {"peptide_name": "", "peptide_sequence": "KWKLFKKIGIGAVL",
         "biological_activities": [{"activity_type": "MIC", "activity_unit": "ug/mL",
                                    "target_organism": ""}]}]}
    verdicts = [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                 "corrected": "Escherichia coli", "peptide": ""}]
    changes, refusals = va.correct_document(payload, verdicts)
    for record in payload["peptides"]:
        assert record["biological_activities"][0]["target_organism"] == ""
    assert [r.why for r in refusals] == ["peptide_name_is_not_unique_in_this_document"]
    assert not [c for c in changes if c.layer == "authorized"]


def test_a_unique_peptide_name_still_receives_its_verdict(tmp_path):
    payload = {"peptides": [
        {"peptide_name": "LL-37", "peptide_sequence": "GIGKFLHSAK",
         "biological_activities": [{"activity_type": "MIC", "activity_unit": "ug/mL",
                                    "target_organism": ""}]},
        {"peptide_name": "magainin 2", "peptide_sequence": "KWKLFKKIGIGAVL",
         "biological_activities": [{"activity_type": "MIC", "activity_unit": "ug/mL",
                                    "target_organism": ""}]}]}
    verdicts = [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                 "corrected": "Escherichia coli", "peptide": "LL-37"}]
    va.correct_document(payload, verdicts)
    assert payload["peptides"][0]["biological_activities"][0]["target_organism"] \
        == "Escherichia coli"
    assert payload["peptides"][1]["biological_activities"][0]["target_organism"] == ""


def test_one_rejected_proposal_is_one_refusal_not_one_per_activity():
    """The summary reported 5,790 refusals where 461 verdicts were refused, because the loop
    tries a verdict against each of the peptide's activities."""
    record = {"peptide_name": "X", "biological_activities": [
        {"activity_type": "MIC", "activity_unit": "ug/mL", "target_organism": ""}
        for _ in range(20)]}
    verdicts = [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                 "corrected": "Rana pretiosa", "peptide": "X"}]
    _, refusals = va.apply_verdicts_to_record(record, verdicts)
    assert len(refusals) == 1


def test_verdicts_are_read_from_the_document_when_no_snapshot_is_given(tmp_path):
    """How the pass runs inside the pipeline: step 12 has just written the verdicts into each
    document, so no snapshot file exists or is needed."""
    payload = {
        "metadata": {"field_review": {"version": 1, "model": "gemma4:latest", "variant": "v1",
                                      "verdicts": [{"field": "target_organism",
                                                    "verdict": "wrong",
                                                    "reason": "absent_target",
                                                    "corrected": "Pseudomonas aeruginosa",
                                                    "peptide": "X"}]}},
        "peptides": [{"peptide_name": "X", "peptide_sequence": "GIGKFLHSAK",
                      "biological_activities": [{"activity_type": "MIC",
                                                 "activity_unit": "ug/mL",
                                                 "target_organism": ""}]}]}
    path = _document(tmp_path, payload)
    summary = va.run(base_dir=str(tmp_path), dry_run=False)
    assert summary["verdict_documents_seen"] == 1
    activity = json.loads(open(path, encoding="utf-8").read())[
        "peptides"][0]["biological_activities"][0]
    assert activity["target_organism"] == "Pseudomonas aeruginosa"
    assert activity["target_taxid"] == 287


def test_a_snapshot_overrides_the_documents_own_verdicts(tmp_path):
    """Why the snapshot mode exists: step 12 has no resume check and rewrites metadata
    .field_review unconditionally, so a later pass can erase verdicts this stage has not acted on
    yet. A snapshot taken before that lets them still be applied."""
    payload = {
        "metadata": {"field_review": {"version": 1, "verdicts": []}},
        "peptides": [{"peptide_name": "X", "peptide_sequence": "GIGKFLHSAK",
                      "biological_activities": [{"activity_type": "MIC",
                                                 "activity_unit": "ug/mL",
                                                 "target_organism": ""}]}]}
    path = _document(tmp_path, payload)
    snapshot = tmp_path / "verdicts.jsonl"
    snapshot.write_text(json.dumps({
        "path": path,
        "verdicts": [{"field": "target_organism", "verdict": "wrong", "reason": "absent_target",
                      "corrected": "Escherichia coli", "peptide": "X"}]}) + "\n",
        encoding="utf-8")
    va.run(base_dir=str(tmp_path), verdict_file=str(snapshot), dry_run=False)
    activity = json.loads(open(path, encoding="utf-8").read())[
        "peptides"][0]["biological_activities"][0]
    assert activity["target_organism"] == "Escherichia coli"
    assert activity["target_taxid"] == 562


def test_provenance_records_what_changed(tmp_path):
    payload = {"peptides": [{"peptide_name": "X", "peptide_sequence": "Gly-Ile-Gly",
                             "biological_activities": [{"activity_type": "mic",
                                                        "activity_unit": "ug/ml"}]}]}
    path = _document(tmp_path, payload)
    va.run(base_dir=str(tmp_path), dry_run=False)
    stored = json.loads(open(path, encoding="utf-8").read())
    by_field = stored["metadata"][va.APPLY_KEY]["by_field"]
    assert by_field["peptide_sequence"] == 1
    assert by_field["activity_type"] == 1
