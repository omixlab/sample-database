"""
Step 12: the final polish, judging what steps 9-11 could not settle.

The economy of the step is its position. Step 11 makes every repair a rule can make, so what
reaches here is the residue nothing cheaper can decide -- 10,619 targeted reviews measured on the
refined output, against ~87,000 if every field of every activity were reviewed.
"""
import json
import os

import pytest

from sample import field_review as FR
from sample import field_review_run as R


def _doc(activities, name="LL-37", sequence="LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"):
    return {"metadata": {"langextract_backend": "enabled"},
            "peptides": [{"peptide_name": name, "peptide_sequence": sequence,
                          "biological_activities": activities}]}


@pytest.fixture(autouse=True)
def _no_name_catalogue(monkeypatch):
    """
    Selection is what these tests are about, so the name-conflict rule is held off.

    `_doc` overrides the sequence while keeping the default name, which to that rule is a record
    pairing LL-37's name with another peptide's sequence -- a correct firing, but it would settle
    by rule the very fields these tests assert get escalated, and two tests below would then pass
    for a reason they do not claim. The rule's own behaviour is covered in `test_novelty.py`,
    against an explicit catalogue rather than whatever is on disk.
    """
    monkeypatch.setattr(FR, "_CATALOGUE", False)


@pytest.fixture()
def corpus(tmp_path):
    source = tmp_path / "pubmed"
    source.mkdir()
    return tmp_path, source


def _write(source, name, payload):
    path = os.path.join(str(source), name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


class TestSelection:
    """A field is reviewed only when a rule could not decide AND the answer is consequential."""

    def test_an_absent_target_on_a_typed_measurement_is_selected(self):
        record = _doc([{"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL",
                        "target_organism": "", "target_cell_line": ""}])["peptides"][0]
        targets = R.select_targets(record)
        assert ("target_organism", record["biological_activities"][0], "absent_target") in targets

    def test_a_measurement_with_a_target_is_not_selected(self):
        record = _doc([{"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL",
                        "target_organism": "Escherichia coli"}])["peptides"][0]
        assert not [t for t in R.select_targets(record) if t[0] == "target_organism"]

    def test_a_placeholder_type_is_selected(self):
        for placeholder in ("N/A", "unknown", "", "not specified"):
            record = _doc([{"activity_type": placeholder, "value_raw": "8",
                            "activity_unit": "ug/mL"}])["peptides"][0]
            assert [t for t in R.select_targets(record) if t[0] == "activity_type"], placeholder

    def test_a_scale_suspect_unit_is_selected(self):
        """mM on a potency metric: the conversion is right, the SOURCE unit is in doubt."""
        record = _doc([{"activity_type": "MIC", "value_raw": "0.3", "activity_unit": "uM",
                        "value_scale_suspect": True,
                        "target_organism": "E. coli"}])["peptides"][0]
        assert [t for t in R.select_targets(record) if t[0] == "activity_unit"]

    def test_three_letter_notation_is_selected_because_rules_cannot_screen_it(self):
        """
        `CVS` for `CYS`, `CY'S`, `AL~`: OCR damage inside three-letter notation passes every
        structural rule, so this is the one sequence population a rule cannot screen.
        """
        record = _doc([], sequence="D-PHE-CYS-TYR-D-TRP-LYS-VAL-CYS-TRP-NH2")["peptides"][0]
        assert [t for t in R.select_targets(record) if t[0] == "peptide_sequence"]

    def test_a_one_letter_sequence_is_not_selected(self):
        record = _doc([], sequence="GIGKFLHSAKKFGKAFVGEIMNS")["peptides"][0]
        assert not [t for t in R.select_targets(record) if t[0] == "peptide_sequence"]

    def test_a_unit_a_rule_already_settles_is_not_selected(self):
        """`Da` on an MIC is step 11's job, not the model's."""
        record = _doc([{"activity_type": "MIC", "value_raw": "2548.2",
                        "activity_unit": "Da", "target_organism": "E. coli"}])["peptides"][0]
        assert not [t for t in R.select_targets(record) if t[0] == "activity_unit"]

    def test_a_mojibake_no_rule_can_name_is_selected(self):
        record = _doc([{"activity_type": "MIC", "value_raw": "5",
                        "activity_unit": "�10�4",
                        "target_organism": "E. coli"}])["peptides"][0]
        assert [t for t in R.select_targets(record)
                if t[0] == "activity_unit" and t[2] == "unnameable_mojibake"]


class TestTheRun:
    def test_a_dry_run_counts_targets_without_calling_the_model(self, corpus, monkeypatch):
        root, source = corpus
        path = _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        before = open(path, "rb").read()
        called = []
        monkeypatch.setattr(R, "review_field", lambda *a, **k: called.append(1))
        summary = R.review_extractions(base_dir=str(root), dry_run=True)
        assert summary["targets"] >= 1 and summary["calls"] == 0 and not called
        assert open(path, "rb").read() == before

    def test_verdicts_are_recorded_into_metadata(self, corpus, monkeypatch):
        root, source = corpus
        path = _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        monkeypatch.setattr(R, "review_field", lambda field, *a, **k: FR.Verdict(
            field, FR.VERDICT_WRONG, "the table header says E. coli",
            corrected="Escherichia coli", decided_by="gemma4:latest"))
        summary = R.review_extractions(base_dir=str(root))
        assert summary["calls"] >= 1 and summary["wrong"] >= 1
        stored = json.load(open(path))["metadata"][R.REVIEW_KEY]
        assert stored["model"] == FR.REVIEW_MODEL and stored["version"] == R.REVIEW_VERSION
        assert stored["verdicts"][0]["corrected"] == "Escherichia coli"

    def test_a_correction_is_not_written_without_apply(self, corpus, monkeypatch):
        root, source = corpus
        path = _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        monkeypatch.setattr(R, "review_field", lambda field, *a, **k: FR.Verdict(
            field, FR.VERDICT_WRONG, "why", corrected="Escherichia coli"))
        R.review_extractions(base_dir=str(root))
        activity = json.load(open(path))["peptides"][0]["biological_activities"][0]
        assert activity.get("target_organism", "") == "", "recorded, not applied"

    def test_apply_writes_the_correction(self, corpus, monkeypatch):
        root, source = corpus
        path = _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        monkeypatch.setattr(R, "review_field", lambda field, *a, **k: FR.Verdict(
            field, FR.VERDICT_WRONG, "why", corrected="Escherichia coli"))
        summary = R.review_extractions(base_dir=str(root), apply=True)
        activity = json.load(open(path))["peptides"][0]["biological_activities"][0]
        assert activity["target_organism"] == "Escherichia coli"
        assert summary["corrections_applied"] >= 1

    def test_a_correction_the_rules_reject_is_refused(self, corpus, monkeypatch):
        """
        The guard that keeps this step from creating work for step 11. A proposed
        `activity_unit` of `Da` on an MIC is exactly what step 11 drops, so it must never be
        written here.
        """
        root, source = corpus
        path = _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "0.3", "activity_unit": "uM",
             "value_scale_suspect": True, "target_organism": "E. coli"}]))
        monkeypatch.setattr(R, "review_field", lambda field, *a, **k: FR.Verdict(
            field, FR.VERDICT_WRONG, "why", corrected="Da"))
        summary = R.review_extractions(base_dir=str(root), apply=True)
        activity = json.load(open(path))["peptides"][0]["biological_activities"][0]
        assert activity["activity_unit"] == "uM", "a rule-rejected correction must not be written"
        assert summary["corrections_refused"] >= 1 and summary["corrections_applied"] == 0

    def test_an_empty_correction_is_not_applied(self, corpus, monkeypatch):
        root, source = corpus
        path = _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        monkeypatch.setattr(R, "review_field", lambda field, *a, **k: FR.Verdict(
            field, FR.VERDICT_WRONG, "the paper does not say", corrected=""))
        summary = R.review_extractions(base_dir=str(root), apply=True)
        assert summary["corrections_applied"] == 0
        assert json.load(open(path))["peptides"][0]["biological_activities"][0].get(
            "target_organism", "") == ""

    def test_max_calls_bounds_the_pass(self, corpus, monkeypatch):
        """How a 14.7-hour pass is run in pieces."""
        root, source = corpus
        for i in range(6):
            _write(source, f"{i}.json", _doc([
                {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        monkeypatch.setattr(R, "review_field",
                            lambda field, *a, **k: FR.Verdict(field, FR.VERDICT_OK, "fine"))
        summary = R.review_extractions(base_dir=str(root), max_calls=2)
        assert summary["calls"] == 2 and summary["targets"] == 6

    def test_a_file_changed_underneath_is_left_alone(self, corpus, monkeypatch):
        import time
        root, source = corpus
        _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        monkeypatch.setattr(R, "review_field",
                            lambda field, *a, **k: FR.Verdict(field, FR.VERDICT_OK, "fine"))
        real = R._write_atomically

        def racing(target, payload, expect_mtime=None):
            os.utime(target, (time.time() + 5, time.time() + 5))
            return real(target, payload, expect_mtime)

        monkeypatch.setattr(R, "_write_atomically", racing)
        summary = R.review_extractions(base_dir=str(root))
        assert summary["skipped_changed_underneath"] == 1

    def test_only_recognised_source_subdirectories_are_read(self, corpus, monkeypatch):
        root, source = corpus
        backup = root / "pubmed.bak"
        backup.mkdir()
        _write(backup, "stale.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        _write(source, "live.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        summary = R.review_extractions(base_dir=str(root), dry_run=True)
        assert summary["documents"] == 1

    def test_an_unreadable_document_is_counted(self, corpus):
        root, source = corpus
        (source / "broken.json").write_text('{"metadata": {"skipped": tr')
        summary = R.review_extractions(base_dir=str(root), dry_run=True)
        assert summary["unreadable"] == 1

    def test_the_summary_names_the_model_and_prompt_actually_used(self, corpus, monkeypatch):
        root, source = corpus
        _write(source, "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL"}]))
        summary = R.review_extractions(base_dir=str(root), dry_run=True)
        assert summary["model"] == FR.REVIEW_MODEL
        assert summary["variant"] == FR.PROMPT_VARIANT


class TestTheModelIsPinned:
    def test_the_review_model_is_gemma4(self):
        """
        Set by operator decision on 2026-10-22, and pinned rather than inherited from
        DEFAULT_EXTRACT_MODEL: extraction's model is chosen for recall on generation, a reviewer
        for precision on a verdict, and this project has recorded that those rankings disagree.
        """
        assert FR.REVIEW_MODEL == "gemma4:latest"
        assert all(spec.model == "gemma4:latest" for spec in FR.FIELD_SPECS.values())

    def test_a_split_between_review_and_extraction_is_reported(self, monkeypatch):
        """
        Not an error -- a stated cost. Ollama evicts one model to load another, measured at 31
        llama-server restarts and 23 evictions in 20 minutes when two stages disagreed.
        """
        assert FR.warn_if_review_model_differs_from_extraction() is None
        monkeypatch.setattr(FR, "REVIEW_MODEL", "qwen3:4b")
        message = FR.warn_if_review_model_differs_from_extraction()
        assert message and "evict" in message
