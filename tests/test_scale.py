"""
Unit-scale diagnosis and correction, from the record's own JSON.

The measurement behind the module: over 4,446 findings this project extracted that a curated
database also reports for the same peptide, assay and organism, `uM` agrees to a median 1.66x and
`ug/mL` to 1.00x, while `mg/mL` agrees to a median of exactly 1000.00x and accounts for 56% of
every gross disagreement in the corpus. Across 182,766 curated potency measurements `mg/mL`
appears 38 times; this extraction has 1,733.

What these tests mostly pin is the ORDER of the checks and the refusal to guess. Three separate
faults arrive looking identical -- a lost prefix, a stock solution recorded as a potency, and a
misread number -- and only the first is repaired by changing the unit.
"""
import pytest

from sample import scale as S


def _activity(**kwargs):
    activity = {"activity_type": "MIC", "activity_unit": "ug/mL", "value_lower": 16.0,
                "value_raw": "16", "target_organism": "Escherichia coli",
                "context_sentence": ""}
    activity.update(kwargs)
    return activity


def _record(*activities):
    return {"peptide_name": "test peptide", "peptide_sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
            "biological_activities": list(activities)}


class TestCanonicalUnit:
    """
    Notation repair only. Whether a prefix is WRONG is the question the rest of the module
    answers; parsing must not presume it.
    """

    @pytest.mark.parametrize("written,expected", [
        ("uM", "um"), ("µM", "um"), ("u M", "um"), ("ug/mL", "ug/ml"), ("u g/mL", "ug/ml"),
        ("Ag/mL", "ug/ml"), ("lm", "um"), ("WM", "um"), ("$\\mu$ M", "um"),
        ("microgram/mL", "ug/ml"), ("mg/liter", "mg/l"), ("ug mL-1", "ug/ml"),
        ("MG/ML", "mg/ml"),
    ])
    def test_damaged_notation_normalises(self, written, expected):
        assert S.canonical_unit(written) == expected

    def test_a_milli_prefix_is_not_quietly_promoted_to_micro(self):
        """`mg/mL` parses as `mg/ml`. Deciding the m should have been a micro sign is a verdict."""
        assert S.canonical_unit("mg/mL") == "mg/ml"

    def test_an_unrecognisable_unit_yields_nothing(self):
        assert S.basis_of(S.canonical_unit("kcal/mol")) == ""
        assert S.basis_of(S.canonical_unit("fold")) == ""


class TestQuantitiesIn:
    """Reading the sentence the extractor saved, which is where the real unit usually survives."""

    def test_a_plain_quantity_is_read(self):
        found = S.quantities_in("The MIC was 16 ug/mL against E. coli.")
        assert (found[0].value, found[0].unit) == (16.0, "ug/ml")

    def test_a_range_sharing_one_trailing_unit_yields_both_numbers(self):
        """
        "MICs ranging from 1.43 to 22.85 mM": only the last number touches the unit, so a
        match-by-match read sees one quantity and calls the other a value the sentence never
        mentions. Measured, that misreported 1,672 activities as contradicting their own context.
        """
        values = {q.value for q in S.quantities_in("MICs ranging from 1.43 to 22.85 mM")}
        assert {1.43, 22.85} <= values

    def test_a_list_sharing_one_trailing_unit_yields_both_numbers(self):
        values = {q.value for q in S.quantities_in("are 4.69 and 0.01 mg/ml")}
        assert {4.69, 0.01} <= values

    def test_a_micro_sign_lost_to_ocr_is_still_read(self):
        found = S.quantities_in("inhibited at 640 Ag/mL")
        assert found and found[0].unit == "ug/ml"

    def test_a_number_without_a_unit_is_not_invented(self):
        assert S.quantities_in("34 strains could be killed") == []


class TestDiagnoseOrder:
    """
    The order is load-bearing. Each of these would be answered differently, and wrongly, by a
    check that ran earlier than it should.
    """

    def test_a_stock_solution_is_flagged_and_never_rescaled(self):
        """
        2 mg/mL is an ordinary stock solution and an impossible MIC. A magnitude test asked first
        proposes 2 ug/mL and produces a confident, well-formed, wrong MIC with the evidence of the
        mistake erased.
        """
        activity = _activity(activity_unit="mg/mL", value_lower=2.0, value_raw="2",
                             context_sentence="cathelicidin-RC1 solution (2 mg/ml, dissolved in "
                                              "sterile deionized water) was incubated")
        finding = S.diagnose(_record(activity), activity)
        assert finding.fault == S.NOT_A_POTENCY_VALUE
        assert finding.status == S.AMBIGUOUS
        assert not finding.rescalable
        assert finding.proposed_unit == ""

    def test_the_sentence_naming_another_unit_settles_it(self):
        activity = _activity(activity_unit="nM", value_lower=5.0, value_raw="5",
                             context_sentence="an MIC of 5 uM was recorded")
        finding = S.diagnose(_record(activity), activity)
        assert finding.status == S.CONFIDENT
        assert finding.proposed_unit == "um"
        assert S.display_unit(finding.proposed_unit) == "uM"

    def test_a_correctly_converted_value_is_not_called_a_contradiction(self):
        """
        "MICs ranging from 1.43 to 22.85 mM" stored as 1430 uM is RIGHT. Comparing raw numbers
        calls it a contradiction; comparing canonical ones sees the same quantity.
        """
        activity = _activity(activity_unit="uM", value_lower=1430.0, value_raw="1.43",
                             context_sentence="MICs ranging from 1.43 to 22.85 mM")
        assert S.diagnose(_record(activity), activity).fault != S.VALUE_CONTRADICTS_CONTEXT

    def test_a_number_the_sentence_contradicts_is_flagged_as_a_value_fault(self):
        activity = _activity(activity_unit="uM", value_lower=0.0, value_raw="0",
                             context_sentence="The MICs against E. coli are 4.69 and 9.38 uM")
        activity["value_lower"] = 99.0
        finding = S.diagnose(_record(activity), activity)
        assert finding.fault == S.VALUE_CONTRADICTS_CONTEXT
        assert not finding.rescalable

    def test_an_ordinary_measurement_is_clean(self):
        activity = _activity()
        assert S.diagnose(_record(activity), activity).status == S.CLEAN

    def test_a_non_potency_assay_is_not_judged_on_magnitude(self):
        """There is no measured band for these, so there is nothing to judge against."""
        activity = _activity(activity_type="hemolysis", activity_unit="uM", value_lower=99999.0)
        assert S.diagnose(_record(activity), activity).status == S.CLEAN


class TestImpossibleVersusMerelyRare:
    """
    The distinction that decides whether a correction may be applied without a model. A unit the
    document confirms but the field barely uses has to be escalated; a unit no peptide potency can
    be stated in may be corrected, because then the document's own spelling IS the damage.
    """

    def test_an_impossible_unit_is_corrected(self):
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640")
        finding = S.diagnose(_record(activity), activity)
        assert finding.status == S.CONFIDENT
        assert finding.proposed_unit == "ug/ml"

    def test_the_lost_prefix_is_restored_rather_than_the_band_chased(self):
        """
        Both ug/mL and ng/mL put 640 inside the plausible band, and picking by band proposed
        `g/mL -> ng/mL` on nine records: a 10^9 correction for a 10^6 error. The micro prefix is
        what the glyph loss dropped, so the micro prefix is what goes back.
        """
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640")
        assert S.diagnose(_record(activity), activity).proposed_unit == "ug/ml"

    def test_an_impossible_unit_is_corrected_even_when_the_sentence_agrees(self):
        """`(640 and > 640) g/mL` is the document printing the damage, not confirming intent."""
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640",
                             context_sentence="LC50 and LC90 were (640 and > 640) g/mL")
        assert S.diagnose(_record(activity), activity).status == S.CONFIDENT

    def test_a_merely_rare_unit_is_escalated_not_corrected(self):
        """
        mg/mL is improbable, not impossible. A paper may report one assay in it, and 12 of the
        first 14 such records had no context sentence at all.
        """
        activity = _activity(activity_unit="mg/mL", value_lower=49.38, value_raw="49.38")
        finding = S.diagnose(_record(activity), activity)
        assert finding.status == S.AMBIGUOUS
        assert finding.fault == S.SCALE_PREFIX_LOST

    def test_a_rare_unit_the_sentence_confirms_says_so_in_the_reason(self):
        """"IC50 was reported to be 6.62 mg/ml" is the paper's own unit, whatever we think."""
        activity = _activity(activity_type="IC50", activity_unit="mg/mL", value_lower=6.62,
                             value_raw="6.62",
                             context_sentence="IC50 was reported to be 6.62 mg/ml")
        finding = S.diagnose(_record(activity), activity)
        assert finding.status == S.AMBIGUOUS
        assert finding.why == "context_confirms_a_rare_unit"

    def test_a_notation_only_difference_is_never_proposed(self):
        """
        mg/L and ug/mL are the same quantity spelled differently. Before this guard, every one of
        398 "confident" findings was exactly this no-op.
        """
        activity = _activity(activity_unit="ug/mL", value_lower=4.0, value_raw="4",
                             context_sentence="an MIC of 4 mg/L")
        finding = S.diagnose(_record(activity), activity)
        assert finding.status != S.CONFIDENT


class TestSiblingUnits:
    def test_only_the_same_assay_counts(self):
        target = _activity(activity_type="MIC", activity_unit="mg/mL")
        record = _record(target,
                         _activity(activity_type="MIC", activity_unit="ug/mL"),
                         _activity(activity_type="IC50", activity_unit="uM"))
        assert S.sibling_units(record, target) == {"ug/ml": 1}

    def test_the_activity_under_review_is_excluded(self):
        target = _activity(activity_type="MIC", activity_unit="mg/mL")
        assert S.sibling_units(_record(target), target) == {}


class TestStep11AppliesOnlyWhatIsProven:
    """Step 11 makes no model call, so it may only act where the JSON settles the question."""

    def test_a_confident_finding_is_applied(self):
        from sample import polish
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640")
        kept, counts = polish.polish_activities([activity])
        assert counts["scale_repaired"] == 1
        assert kept[0]["activity_unit"] == "ug/mL"

    def test_the_number_is_not_touched(self):
        """
        `value_lower` is expressed in `activity_unit`, so restoring a dropped prefix is a change
        of unit at a fixed number. Moving both would move the quantity twice.
        """
        from sample import polish
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640")
        kept, _ = polish.polish_activities([activity])
        assert kept[0]["value_lower"] == 640.0
        assert kept[0]["value_raw"] == "640"

    def test_the_repair_records_its_provenance(self):
        from sample import polish
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640")
        kept, _ = polish.polish_activities([activity])
        repair = kept[0][polish.SCALE_REPAIR_KEY]
        assert repair["from"] == "g/mL" and repair["to"] == "ug/mL"

    def test_the_repair_is_idempotent(self):
        from sample import polish
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640")
        kept, first = polish.polish_activities([activity])
        kept, second = polish.polish_activities(kept)
        assert first["scale_repaired"] == 1
        assert second["scale_repaired"] == 0
        assert kept[0]["activity_unit"] == "ug/mL"

    def test_an_ambiguous_finding_is_left_alone(self):
        from sample import polish
        activity = _activity(activity_unit="mg/mL", value_lower=49.38, value_raw="49.38")
        kept, counts = polish.polish_activities([activity])
        assert counts["scale_repaired"] == 0
        assert kept[0]["activity_unit"] == "mg/mL"

    def test_a_stock_solution_is_never_rewritten_by_a_rule(self):
        from sample import polish
        activity = _activity(activity_unit="mg/mL", value_lower=2.0, value_raw="2",
                             context_sentence="solution (2 mg/ml, dissolved in water) was "
                                              "incubated")
        kept, counts = polish.polish_activities([activity])
        assert counts["scale_repaired"] == 0
        assert kept[0]["activity_unit"] == "mg/mL"


class TestTheInJsonContextGivenToTheModel:
    """
    The evidence step 12 shows the model for these questions comes out of the record, not the
    PDF: the sentence each value was taken from is already saved on 53.2% of activities, with a
    median length of 66 characters, so the decisive passage costs nothing to retrieve.
    """

    @pytest.fixture()
    def context(self):
        from sample.field_review import record_context
        return record_context

    def test_the_recorded_sentence_is_included(self, context):
        activity = _activity(context_sentence="The MIC was 16 ug/mL against E. coli.")
        assert "16 ug/mL against E. coli" in context(_record(activity), activity, "activity_unit")

    def test_sibling_measurements_are_included(self, context):
        target = _activity(activity_unit="mg/mL", value_raw="49.38")
        record = _record(target, _activity(activity_unit="ug/mL", value_raw="8"))
        block = context(record, target, "activity_unit")
        assert "other measurements on this same peptide" in block
        assert "8 ug/mL" in block

    def test_the_activity_under_review_is_not_listed_as_its_own_sibling(self, context):
        target = _activity(value_raw="49.38", activity_unit="mg/mL")
        block = context(_record(target), target, "activity_unit")
        assert "other measurements" not in block

    def test_the_automated_diagnosis_is_named(self, context):
        activity = _activity(activity_unit="mg/mL", value_lower=49.38, value_raw="49.38")
        block = context(_record(activity), activity, "activity_unit")
        assert S.SCALE_PREFIX_LOST in block

    def test_an_unrescalable_fault_says_so_explicitly(self, context):
        """
        Otherwise the model is shown a proposed unit for a stock solution and invited to accept a
        correction that cannot be right.
        """
        activity = _activity(activity_unit="mg/mL", value_lower=2.0, value_raw="2",
                             context_sentence="solution (2 mg/ml, dissolved in water) was "
                                              "incubated")
        block = context(_record(activity), activity, "value_raw")
        assert "would NOT fix this" in block

    def test_a_clean_measurement_contributes_no_diagnosis(self, context):
        activity = _activity()
        assert "automated check flagged" not in context(_record(activity), activity, "activity_unit")

    def test_the_sentence_is_labelled_as_the_extractor_s_record_not_the_document(self):
        """
        It is extraction output and can itself be the thing that went wrong, so the prompt must
        not present it with the document's authority.
        """
        from sample.field_review import FIELD_SPECS
        activity = _activity(context_sentence="The MIC was 16 ug/mL.")
        messages = FIELD_SPECS["activity_unit"].build_messages(
            "ug/mL", _record(activity), activity, evidence="", json_context="x")
        user = messages[-1]["content"]
        assert "the document wins" in user

    def test_a_long_sentence_is_truncated(self, context):
        """One recorded 'sentence' reached 9,364 characters, which is a page of text."""
        activity = _activity(context_sentence="x" * 5000)
        assert len(context(_record(activity), activity, "activity_unit")) < 1200


class TestStep12SelectsTheAmbiguousFindings:
    """A diagnosis nothing asks for never reaches a reviewer."""

    @pytest.fixture()
    def select(self, monkeypatch):
        from sample import field_review, field_review_run
        monkeypatch.setattr(field_review, "_CATALOGUE", False)
        return field_review_run.select_targets

    def test_a_rare_unit_is_queued_against_the_unit_field(self, select):
        activity = _activity(activity_unit="mg/mL", value_lower=49.38, value_raw="49.38")
        targets = select(_record(activity))
        assert ("activity_unit", activity, "scale_prefix_suspected") in targets

    def test_a_stock_solution_is_queued_against_the_value_field(self, select):
        """The unit is not what is in doubt, so queuing it against the unit would ask the wrong
        question and invite a correction that cannot help."""
        activity = _activity(activity_unit="mg/mL", value_lower=2.0, value_raw="2",
                             context_sentence="solution (2 mg/ml, dissolved in water) was "
                                              "incubated")
        targets = select(_record(activity))
        assert ("value_raw", activity, "stock_solution_suspected") in targets

    def test_a_confident_finding_is_not_queued(self, select):
        """Step 11 has already applied it; a model call would only confirm a rule."""
        activity = _activity(activity_unit="g/mL", value_lower=640.0, value_raw="640")
        assert not [t for t in select(_record(activity)) if t[2].startswith("scale")]

    def test_a_clean_measurement_is_not_queued_for_scale(self, select):
        activity = _activity()
        assert not [t for t in select(_record(activity))
                    if t[2] in ("scale_prefix_suspected", "implausible_magnitude",
                                "stock_solution_suspected", "contradicts_context_sentence")]

    def test_an_already_suspect_unit_is_not_queued_twice(self, select):
        """`value_scale_suspect` and the scale diagnosis are the same doubt about the same field."""
        activity = _activity(activity_unit="mg/mL", value_lower=49.38, value_raw="49.38",
                             value_scale_suspect=True)
        unit_targets = [t for t in select(_record(activity)) if t[0] == "activity_unit"]
        assert len(unit_targets) == 1


class TestAnAmountIsNotAConcentration:
    """
    The bug that nearly let a sentence corrupt a good record. Before the trailing-letter
    lookahead, the molar alternative matched the `nm` inside `nmol`, so "50 nmol of OVA-I peptide"
    -- a dose in moles -- parsed as 50 nM, and a correct concentration elsewhere in the record
    could have been "corrected" to match it.
    """

    @pytest.mark.parametrize("sentence", [
        "50 nmol of OVA-I peptide",
        "at the same dose of 0.22 mmol/kg in 0.1 mL PBS",
        "20 nmol was injected",
    ])
    def test_an_amount_or_a_per_mass_dose_yields_no_concentration(self, sentence):
        assert S.quantities_in(sentence) == []

    @pytest.mark.parametrize("sentence,expected", [
        ("a 2 mmol/L solution", "mm"),
        ("16 umol/L against E. coli", "um"),
        ("5 uM was recorded", "um"),
    ])
    def test_a_real_concentration_is_still_read(self, sentence, expected):
        found = S.quantities_in(sentence)
        assert found and found[0].unit == expected

    def test_mol_per_litre_spellings_canonicalise(self):
        assert S.canonical_unit("mmol/L") == "mm"
        assert S.canonical_unit("umol/L") == "um"


class TestNotationRepairInStep11:
    """
    `preprocessing.canonicalize_unit` handles the spellings the pipeline produces; these are the
    ones PDF extraction produces, and measured over the corpus it repaired none of them -- 1,212
    activities carrying a unit the field does use, written in a way the release vocabulary does
    not contain.
    """

    @pytest.mark.parametrize("written,expected", [
        ("u M", "uM"), ("AM", "uM"), ("WM", "uM"), ("lm", "uM"), ("$\\mu$ M", "uM"),
        ("u g/mL", "ug/mL"), ("Ag/mL", "ug/mL"), ("lgml", "ug/mL"), ("microgram/mL", "ug/mL"),
        ("ug mL-1", "ug/mL"), ("mg/liter", "mg/L"),
    ])
    def test_extraction_damage_is_repaired(self, written, expected):
        from sample.polish import _repair_unit
        activity = {"activity_unit": written}
        assert _repair_unit(activity)
        assert activity["activity_unit"] == expected

    @pytest.mark.parametrize("unit", ["uM", "ug/mL", "mg/mL", "nM"])
    def test_a_canonical_unit_is_untouched(self, unit):
        from sample.polish import _repair_unit
        activity = {"activity_unit": unit}
        assert not _repair_unit(activity)
        assert activity["activity_unit"] == unit

    @pytest.mark.parametrize("unit", ["kcal/mol", "fold", "MIC", "Range", "540 nm", "IS"])
    def test_a_string_that_is_not_a_unit_is_left_exactly_as_written(self, unit):
        """
        Coercing an unrecognised string into the nearest unit is how a count or a wavelength
        becomes a concentration. The repair applies only when it lands in the release vocabulary.
        """
        from sample.polish import _repair_unit
        activity = {"activity_unit": unit}
        _repair_unit(activity)
        assert activity["activity_unit"] == unit

    def test_the_repair_is_idempotent(self):
        from sample.polish import _repair_unit
        activity = {"activity_unit": "u M"}
        assert _repair_unit(activity)
        assert not _repair_unit(activity)
        assert activity["activity_unit"] == "uM"

    def test_the_magnitude_is_never_converted(self):
        """Only the spelling changes, so `value_lower` stays consistent with `activity_unit`."""
        from sample.polish import _repair_unit
        activity = {"activity_unit": "u M", "value_lower": 16.0, "value_raw": "16"}
        _repair_unit(activity)
        assert activity["value_lower"] == 16.0


class TestAUnitThatIsNotAUnit:
    """
    `metric_as_unit` was declared deterministic and counted at 100, and nothing implemented it.
    Measured: `MIC` appears as a unit 622 times, `MBC` 111, `MFC` 28, `MRSA` 89, `VRE` 25 -- 1,028
    in all, every one escalating to a model that cannot recover the real unit either, because it
    is not in the record.
    """

    @pytest.fixture()
    def rule(self):
        from sample.field_review import deterministic_verdict
        return lambda unit: deterministic_verdict(
            "activity_unit", unit, {"peptide_name": "x"},
            {"activity_type": "MIC", "activity_unit": unit})

    @pytest.mark.parametrize("unit", ["MIC", "MBC", "MFC", "MLC", "ZOI"])
    def test_an_assay_name_in_the_unit_field_is_caught(self, rule, unit):
        assert rule(unit).failure_mode == "metric_as_unit"

    @pytest.mark.parametrize("unit", ["MRSA", "VRE", "MSSA", "ESBL"])
    def test_a_strain_in_the_unit_field_is_caught(self, rule, unit):
        assert rule(unit).failure_mode == "metric_as_unit"

    @pytest.mark.parametrize("unit", ["uM", "ug/mL", "mM", "M", "nM", "mg/mL"])
    def test_a_real_unit_is_never_condemned(self, rule, unit):
        verdict = rule(unit)
        assert verdict is None or verdict.failure_mode != "metric_as_unit"

    def test_nothing_is_invented_for_it(self, rule):
        """The real unit is not in the record, so the finding is recorded and left for a human."""
        assert not (rule("MIC").corrected or "").strip()


class TestWhatMayBeWrittenBackIntoTheCorpus:
    """
    A correction is a free choice, so the bar for writing one is higher than the bar for tolerating
    a value already in the corpus. 10,368 activities carry an off-vocabulary unit and flagging every
    one as wrong would change what step 12 looks at; writing a new one has no such excuse.
    """

    @pytest.fixture()
    def acceptable(self):
        from sample.field_review_run import _acceptable_correction
        return _acceptable_correction

    @pytest.mark.parametrize("proposed", ["ug/mL", "uM", "mg/L", "nM", "µg/ml", "u M"])
    def test_a_vocabulary_unit_is_accepted_however_spelled(self, acceptable, proposed):
        assert acceptable("activity_unit", proposed) == ""

    @pytest.mark.parametrize("proposed", ["mcg/ml", "uM/L", "ug/mL (final)", "MIC", "kcal/mol"])
    def test_a_plausible_looking_non_vocabulary_unit_is_refused(self, acceptable, proposed):
        """
        These pass `deterministic_verdict`, which only rejects what it knows to be bad. Writing
        them would spend step 12's calls partly undoing step 11's 2,768 unit repairs.
        """
        assert acceptable("activity_unit", proposed) != ""

    def test_an_empty_proposal_is_refused(self, acceptable):
        assert acceptable("activity_unit", "   ") == "empty"
        assert acceptable("value_raw", "") == "empty"

    @pytest.mark.parametrize("proposed", ["16", "4.69", "5 to 50", ">128"])
    def test_a_numeric_value_is_accepted(self, acceptable, proposed):
        assert acceptable("value_raw", proposed) == ""

    def test_a_value_with_no_number_is_refused(self, acceptable):
        assert acceptable("value_raw", "not reported") != ""

    def test_a_written_unit_is_stored_in_the_corpus_spelling(self):
        """
        A model answering `µg/ml` is right about the unit and wrong about the spelling; writing it
        verbatim would add yet another off-vocabulary variant.
        """
        from sample.field_review import Verdict, VERDICT_WRONG
        from sample.field_review_run import _apply_correction
        activity = {"activity_type": "MIC", "activity_unit": "mg/mL", "value_lower": 16.0,
                    "target_organism": "E. coli"}
        verdict = Verdict("activity_unit", VERDICT_WRONG, "lost micro sign", corrected="µg/ml")
        assert _apply_correction("activity_unit", verdict, {"peptide_name": "x"}, activity)
        assert activity["activity_unit"] == "ug/mL"

    def test_a_refused_correction_leaves_the_value_alone(self):
        from sample.field_review import Verdict, VERDICT_WRONG
        from sample.field_review_run import _apply_correction
        activity = {"activity_type": "MIC", "activity_unit": "mg/mL", "value_lower": 16.0,
                    "target_organism": "E. coli"}
        verdict = Verdict("activity_unit", VERDICT_WRONG, "junk", corrected="mcg/ml")
        assert not _apply_correction("activity_unit", verdict, {"peptide_name": "x"}, activity)
        assert activity["activity_unit"] == "mg/mL"
