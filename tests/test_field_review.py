"""
Per-field review specs: prompts, schemas, context budgets and model choice.

The failure modes in each spec were measured on this corpus, and the prompts are built from them,
so these tests check the machinery AND that the machinery cannot quietly lose the measurements
that make the prompts worth sending.
"""
import json

import pytest

from sample import field_review as F


class TestEveryFieldHasAUsableSpec:
    def test_the_registry_covers_the_fields_that_were_reviewed(self):
        """Every field whose defects were measured on this corpus has a spec."""
        assert set(F.FIELD_SPECS) == {
            "peptide_name", "peptide_sequence", "activity_type", "activity_unit",
            "value_raw", "target_organism", "target_cell_line"}

    def test_scopes_partition_the_registry(self):
        assert set(F.PEPTIDE_FIELDS) | set(F.ACTIVITY_FIELDS) == set(F.FIELD_SPECS)
        assert not set(F.PEPTIDE_FIELDS) & set(F.ACTIVITY_FIELDS)

    @pytest.mark.parametrize("name", sorted(F.FIELD_SPECS))
    def test_each_spec_is_complete(self, name):
        spec = F.FIELD_SPECS[name]
        assert spec.failure_modes, "a spec with no failure modes produces a generic prompt"
        assert spec.question.endswith("?")
        assert spec.model_rationale, "a model choice with no stated reason cannot be argued with"
        assert spec.scope in ("peptide", "activity")
        assert spec.needle_fields, "with no needles the evidence selector returns the head of the document"

    @pytest.mark.parametrize("name", sorted(F.FIELD_SPECS))
    def test_a_measured_mode_carries_its_evidence(self, name):
        """
        A mode claiming to be measured must show the measurement: real examples from this corpus,
        a counted total, or both. A mode included on reasoning alone has to say so via
        `measured=False`, so the two can never be confused in a prompt or in a review of one.
        """
        for mode in F.FIELD_SPECS[name].failure_modes:
            if not mode.measured:
                continue
            assert mode.examples or mode.count is not None, (
                f"{name}:{mode.name} claims to be measured but carries no evidence; "
                f"either add it or mark the mode measured=False")

    def test_reasoned_modes_are_the_minority(self):
        """
        A sanity bound on the prompts: if most modes were reasoned rather than observed, the
        prompts would be describing an imagined failure surface.
        """
        modes = [m for spec in F.FIELD_SPECS.values() for m in spec.failure_modes]
        reasoned = [m for m in modes if not m.measured]
        assert len(reasoned) < len(modes) / 2, (
            f"{len(reasoned)} of {len(modes)} modes are unmeasured")


class TestTheSchemaIsSafeForOllama:
    @pytest.mark.parametrize("name", sorted(F.FIELD_SPECS))
    def test_no_array_length_bounds_anywhere(self, name):
        """
        Ollama's schema-to-grammar converter returns an EMPTY completion when it meets
        minItems/maxItems rather than an error, which a fail-open caller reads as success. This
        codebase has already shipped that bug once, in `llm._verdict_schema`.
        """
        text = json.dumps(F.FIELD_SPECS[name].schema())
        for unsupported in ("minItems", "maxItems", "minLength", "maxLength", "pattern"):
            assert unsupported not in text, f"{name} schema uses {unsupported}"

    @pytest.mark.parametrize("name", sorted(F.FIELD_SPECS))
    def test_the_verdict_is_a_closed_enum(self, name):
        schema = F.FIELD_SPECS[name].schema()
        assert schema["properties"]["verdict"]["enum"] == list(F.VERDICTS)
        assert "verdict" in schema["required"] and "reason" in schema["required"]

    def test_a_correcting_spec_constrains_the_failure_mode(self):
        schema = F.FIELD_SPECS["activity_unit"].schema()
        modes = schema["properties"]["failure_mode"]["enum"]
        assert "other" in modes, "an unlisted error must still be reportable"
        assert "mojibake" in modes


class TestThePromptNamesRealErrors:
    def test_measured_examples_reach_the_prompt(self):
        prompt = F.FIELD_SPECS["peptide_name"].system_prompt()
        for observed in ("MCF-7", "Diclofenac", "892.306"):
            assert observed in prompt, f"{observed} was measured but is not in the prompt"

    def test_rule_settled_modes_are_left_out_of_the_prompt(self):
        """
        A mode a rule already decides never reaches the model, so listing it wastes context and
        invites the model to re-litigate a settled question.
        """
        spec = F.FIELD_SPECS["activity_unit"]
        prompt = spec.system_prompt()
        assert "micro_lost_as_milli" in prompt, "the ambiguous mode must be listed"
        assert "non_concentration_unit" not in prompt.split("Rules:")[0], (
            "a deterministic mode must not be in the prose list")

    def test_the_prompt_asks_for_unsure_rather_than_a_guess(self):
        for name in F.FIELD_SPECS:
            assert F.VERDICT_UNSURE in F.FIELD_SPECS[name].system_prompt()

    def test_the_field_under_review_is_stated_separately_from_the_record(self):
        """So a neighbouring value cannot be mistaken for the one being judged."""
        spec = F.FIELD_SPECS["activity_unit"]
        messages = spec.build_messages("Da", {"peptide_name": "LL-37"},
                                       {"activity_type": "MIC", "activity_unit": "Da"}, "text")
        user = messages[1]["content"]
        assert "Field under review: activity_unit" in user
        assert "do not judge these" in user


class TestTheContextBudget:
    def test_the_budget_is_derived_from_the_context_rung(self):
        from sample.config import DEFAULT_NUM_CTX
        assert F.EVIDENCE_BUDGET_CHARS == int((DEFAULT_NUM_CTX - 384) * 3.5)

    @pytest.mark.parametrize("name", sorted(F.FIELD_SPECS))
    def test_a_full_prompt_fits_the_bottom_rung(self, name):
        """
        The whole point of a per-field evidence budget. A prompt over the rung makes `llm_call`
        escalate, and a continuously varying window rebuilds the Ollama runner -- measured at 7
        restarts in 10 minutes.
        """
        spec = F.FIELD_SPECS[name]
        record = {"peptide_name": "x" * 60, "peptide_sequence": "GIGKFLHSAKKF" * 4}
        activity = {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL",
                    "target_organism": "Escherichia coli", "target_cell_line": ""}
        messages = spec.build_messages("value", record, activity, "E" * 50_000)
        chars = sum(len(m["content"]) for m in messages)
        needed = int(chars / 3.5) + spec.max_output_tokens + 256
        from sample.config import DEFAULT_NUM_CTX
        assert needed <= DEFAULT_NUM_CTX, (
            f"{name} needs ~{needed} tokens against a rung of {DEFAULT_NUM_CTX}")

    def test_evidence_is_truncated_to_the_spec_budget(self):
        spec = F.FIELD_SPECS["activity_unit"]
        messages = spec.build_messages("uM", {}, {"activity_type": "MIC"}, "Z" * 99_999)
        assert messages[1]["content"].count("Z") == spec.evidence_chars


class TestEvidenceSelection:
    def test_passages_around_a_match_are_returned(self):
        text = ("filler " * 300) + "the MIC of LL-37 was 8 ug/mL against E. coli" + (" filler" * 300)
        out = F.select_evidence(text, ["LL-37"], 600)
        assert "LL-37" in out and len(out) <= 600

    def test_short_needles_are_ignored(self):
        """
        Measured directly: reviewing `activity_unit="min"` used "min" as a needle, selected
        windows around "minute" and "determined", and the model answered "unsure -- the document
        text provided does not contain any information about the activity unit".
        """
        text = "determined minimum amino acid content in minutes " * 40
        assert F.select_evidence(text, ["min"], 300) == text[:300], (
            "a 3-char needle must not drive selection")

    def test_overlapping_windows_are_merged(self):
        text = "A" * 100 + "LL-37" + "B" * 20 + "LL-37" + "C" * 100
        out = F.select_evidence(text, ["LL-37"], 5000)
        assert out.count(" ... ") == 0, "adjacent windows must merge, not repeat"

    def test_no_match_falls_back_to_the_head(self):
        assert F.select_evidence("nothing relevant here", ["ZZZZZZ"], 10) == "nothing re"

    def test_empty_text_is_handled(self):
        assert F.select_evidence("", ["LL-37"], 100) == ""

    def test_needles_come_from_the_spec_in_priority_order(self):
        spec = F.FIELD_SPECS["activity_unit"]
        needles = spec.needles({"peptide_name": "LL-37"},
                               {"value_raw": "8", "target_organism": "E. coli"})
        assert needles == ["8", "LL-37", "E. coli"], "value_raw locates a unit, not the unit itself"


class TestTheDeterministicPathDecidesWhatItCan:
    def test_an_incoherent_unit_is_settled_without_a_model(self):
        v = F.deterministic_verdict("activity_unit", "Da", {}, {"activity_type": "MIC"})
        assert v is not None and v.verdict == F.VERDICT_WRONG
        assert v.failure_mode == "non_concentration_unit" and v.decided_by == "rule"

    def test_a_mojibake_unit_is_repaired_without_a_model(self):
        v = F.deterministic_verdict("activity_unit", "�M", {}, {"activity_type": "MIC"})
        assert v is not None and v.corrected == "uM" and v.failure_mode == "mojibake"

    def test_a_mojibake_the_rules_cannot_name_is_escalated(self):
        """`�10�4` is a multiplication sign and a superscript minus; no rule knows."""
        assert F.deterministic_verdict(
            "activity_unit", "�10�4", {}, {"activity_type": "MIC"}) is None

    def test_a_structurally_invalid_sequence_is_settled(self):
        v = F.deterministic_verdict("peptide_sequence", "C10H22N6O4", {}, None)
        assert v is not None and v.verdict == F.VERDICT_WRONG

    def test_a_valid_sequence_is_escalated_not_affirmed(self):
        """
        Passing the rules is not evidence the paper states it for this peptide; that is the
        question the model is for.
        """
        assert F.deterministic_verdict("peptide_sequence", "GIGKFLHSAKKF", {}, None) is None

    def test_prose_in_the_type_field_is_settled(self):
        v = F.deterministic_verdict("activity_type", "x" * 60, {}, {})
        assert v is not None and v.failure_mode == "prose_instead_of_type"

    def test_a_range_with_no_upper_bound_is_settled(self):
        v = F.deterministic_verdict("value_raw", "5 to 50", {},
                                    {"value_raw": "5 to 50", "value_upper": None})
        assert v is not None and v.failure_mode == "range_without_upper"

    def test_a_range_already_bounded_is_not_flagged(self):
        assert F.deterministic_verdict(
            "value_raw", "5 to 50", {}, {"value_raw": "5 to 50", "value_upper": 50.0}) is None

    def test_an_ambiguous_unit_reaches_the_model(self):
        """mM on an MIC may be a lost micro sign; only the document settles it."""
        assert F.deterministic_verdict("activity_unit", "mM", {}, {"activity_type": "MIC"}) is None


class TestReviewWithoutAModel:
    """`use_llm=False` makes the whole system runnable with no model server."""

    def test_a_rule_still_decides(self):
        v = F.review_field("activity_unit", {}, {"activity_type": "MIC", "activity_unit": "Da"},
                           use_llm=False)
        assert v.verdict == F.VERDICT_WRONG and v.decided_by == "rule"

    def test_an_undecided_field_returns_unsure_not_ok(self):
        """
        Failing CLOSED. A review layer that returned `ok` when it had not looked would silently
        confirm every field whenever the model was unavailable.
        """
        v = F.review_field("activity_unit", {}, {"activity_type": "MIC", "activity_unit": "mM"},
                           use_llm=False)
        assert v.verdict == F.VERDICT_UNSURE

    def test_an_activity_field_without_an_activity_is_refused(self):
        with pytest.raises(ValueError):
            F.review_field("activity_unit", {"peptide_name": "LL-37"}, None, use_llm=False)

    def test_an_unknown_field_raises(self):
        with pytest.raises(KeyError):
            F.review_field("not_a_field", {}, None, use_llm=False)

    def test_a_record_review_covers_every_field_and_activity(self):
        record = {"peptide_name": "LL-37", "peptide_sequence": "GIGKFLHSAKKF",
                  "biological_activities": [{"activity_type": "MIC", "activity_unit": "Da"},
                                            {"activity_type": "MIC", "activity_unit": "uM"}]}
        verdicts = F.review_record(record, use_llm=False)
        assert len(verdicts) == len(F.PEPTIDE_FIELDS) + 2 * len(F.ACTIVITY_FIELDS)
        assert any(v.field == "activity_unit" and v.verdict == F.VERDICT_WRONG for v in verdicts)


class TestModelGrouping:
    def test_fields_group_by_their_model(self):
        groups = F.group_by_model(tuple(F.FIELD_SPECS))
        assert sum(len(v) for v in groups.values()) == len(F.FIELD_SPECS)

    def test_every_spec_defaults_to_the_pipeline_model(self):
        """
        Not an accident and not a finding. Loading a second model evicts the first -- measured at
        31 llama-server restarts and 23 evictions in 20 minutes -- so a per-field model must be
        justified by measurement before it is worth the eviction. `review-fields --measure` is
        how that justification is produced.
        """
        from sample.config import DEFAULT_EXTRACT_MODEL
        assert {s.model for s in F.FIELD_SPECS.values()} == {DEFAULT_EXTRACT_MODEL}


class TestTheModelBench:
    """
    The harness that turns the model column into a measurement.

    No model calls here: `review_field` is replaced so the scoring, the label sourcing and the
    rule-suppression can be tested without a server.
    """

    @staticmethod
    def _corpus(tmp_path):
        import json as _json
        source = tmp_path / "pubmed"
        source.mkdir()
        text_dir = tmp_path / "text"
        text_dir.mkdir()
        docs = {
            # known-bad: a mass unit on an MIC
            "bad": {"metadata": {}, "peptides": [{
                "peptide_name": "brevinin-1RTb",
                "peptide_sequence": "FLGSLLGLVGKVVPTLFCKISKKC",
                "biological_activities": [
                    {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}]}]},
            # known-good: a clean, typed, targeted, dose-bearing measurement
            "good": {"metadata": {}, "peptides": [{
                "peptide_name": "LL-37",
                "peptide_sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
                "biological_activities": [
                    {"activity_type": "MIC", "value_raw": "8", "value_lower": 8.0,
                     "activity_unit": "ug/mL", "target_organism": "Escherichia coli"}]}]},
        }
        for name, payload in docs.items():
            (source / f"{name}.json").write_text(_json.dumps(payload))
        return tmp_path

    def test_cases_are_labelled_from_rules_and_the_trustworthy_tier(self, tmp_path):
        from sample import field_review_bench as B
        cases = B.build_cases("activity_unit", limit=10, base_dir=str(self._corpus(tmp_path)))
        by_label = {c.label_source: c for c in cases}
        assert any(k.startswith("rule:") for k in by_label), "no known-bad case"
        assert "trustworthy_tier" in by_label, "no known-good case"
        bad = next(c for c in cases if c.expected == B.VERDICT_WRONG)
        assert bad.activity["activity_unit"] == "Da"
        good = next(c for c in cases if c.expected == B.VERDICT_OK)
        assert good.activity["activity_unit"] == "ug/mL"

    def test_an_empty_field_is_never_a_case(self, tmp_path):
        """
        A degenerate case measures nothing: there is nothing to judge, and the rules settle it in
        production. Without this filter `peptide_sequence`'s known-bad population is dominated by
        empty values -- 26,716 such records were measured on this corpus.
        """
        import json as _json
        root = tmp_path
        source = root / "pubmed"
        source.mkdir()
        (source / "a.json").write_text(_json.dumps({"metadata": {}, "peptides": [
            {"peptide_name": "x", "peptide_sequence": "",
             "biological_activities": [{"activity_type": "MIC", "activity_unit": ""}]}]}))
        from sample import field_review_bench as B
        assert B.build_cases("activity_unit", limit=10, base_dir=str(root)) == []

    def test_the_rule_is_suppressed_so_the_model_is_what_gets_measured(self, tmp_path,
                                                                       monkeypatch):
        """
        Without this the bench scores the rules: `review_field` consults them first by design, so
        every known-bad case would be answered before the model saw it.
        """
        from sample import field_review as FR
        from sample import field_review_bench as B

        seen = []

        def fake_review(field, record, activity=None, text="", model=None, **kwargs):
            # If the rule had run, this would never be reached for the Da case.
            seen.append((activity or {}).get("activity_unit"))
            return FR.Verdict(field, FR.VERDICT_WRONG, "model said so", decided_by=model)

        monkeypatch.setattr(B, "review_field", fake_review)
        monkeypatch.setattr(B, "available_models", lambda: ["gemma4:latest"])
        results = B.measure_field("activity_unit", models=("gemma4:latest",), limit=10,
                                  base_dir=str(self._corpus(tmp_path)))
        assert "Da" in seen, "the known-bad case never reached the model"
        assert results["gemma4:latest"]["caught"] == 1.0
        assert results["gemma4:latest"]["false_alarms"] == 1.0, (
            "answering 'wrong' to everything must show as a false alarm, not as success")

    def test_rules_are_restored_after_a_measurement(self, tmp_path, monkeypatch):
        from sample import field_review as FR
        from sample import field_review_bench as B
        original = FR.deterministic_verdict
        monkeypatch.setattr(B, "review_field",
                            lambda *a, **k: FR.Verdict("activity_unit", FR.VERDICT_UNSURE))
        monkeypatch.setattr(B, "available_models", lambda: ["gemma4:latest"])
        B.measure_field("activity_unit", models=("gemma4:latest",), limit=4,
                        base_dir=str(self._corpus(tmp_path)))
        assert FR.deterministic_verdict is original, "suppression leaked out of the measurement"

    def test_unsure_is_reported_as_a_cost_not_as_an_error(self, tmp_path, monkeypatch):
        from sample import field_review as FR
        from sample import field_review_bench as B
        monkeypatch.setattr(B, "review_field",
                            lambda *a, **k: FR.Verdict("activity_unit", FR.VERDICT_UNSURE))
        monkeypatch.setattr(B, "available_models", lambda: ["gemma4:latest"])
        r = B.measure_field("activity_unit", models=("gemma4:latest",), limit=10,
                            base_dir=str(self._corpus(tmp_path)))["gemma4:latest"]
        assert r["caught"] == 0.0 and r["false_alarms"] == 0.0 and r["decided"] == 0.0
        table = B.format_results("activity_unit", {"gemma4:latest": r})
        assert "a COST, not accuracy" in table

    def test_an_uninstalled_model_is_skipped_not_pulled(self, tmp_path, monkeypatch):
        from sample import field_review_bench as B
        monkeypatch.setattr(B, "available_models", lambda: ["gemma4:latest"])
        results = B.measure_field("activity_unit", models=("no-such-model:1b",), limit=4,
                                  base_dir=str(self._corpus(tmp_path)))
        assert results == {}

    def test_the_ranking_prefers_recall_then_safety(self):
        from sample import field_review_bench as B
        table = B.format_results("activity_unit", {
            "guesser": {"n": 10, "caught": 1.0, "false_alarms": 1.0, "unsure": 0.0,
                        "decided": 1.0, "median_seconds": 1.0},
            "careful": {"n": 10, "caught": 1.0, "false_alarms": 0.0, "unsure": 0.2,
                        "decided": 0.8, "median_seconds": 9.0},
        })
        assert table.index("careful") < table.index("guesser"), (
            "equal recall must be broken by the false-alarm rate")
