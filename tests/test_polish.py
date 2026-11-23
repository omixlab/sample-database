"""
Step 11: re-applying the current activity-field rules to records already on disk.

The step exists because a rule fixed today otherwise reaches the corpus only at the speed of
re-extraction, which on this corpus is a multi-day job. Measured when it was written: 6,173 of
53,107 activities held a concentration type in a mass/time/energy/count unit, 1,098 units carried
U+FFFD, 2,312 value_raw held a range with no upper bound, and 58 activity_type values were prose.
"""
import json
import os

import pytest

from sample import polish as P

FFFD = "�"


def _doc(activities, **meta):
    return {"metadata": dict(meta),
            "peptides": [{"peptide_name": "Magainin 2", "peptide_sequence": "GIGKFLHSAKKF",
                          "biological_activities": activities}]}


def _write(directory, name, payload):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


@pytest.fixture()
def corpus(tmp_path):
    """A directory shaped like real extraction output: a recognized source subdirectory."""
    source = tmp_path / "pubmed"
    source.mkdir()
    return tmp_path, source


class TestUnitRepair:
    def test_a_mojibake_unit_is_repaired_in_both_unit_fields(self):
        act = {"activity_type": "MIC", "value_raw": "37", "value_lower": 37.0,
               "activity_unit": FFFD + "g/mL", "value_raw_unit": FFFD + "g/mL"}
        assert P._repair_unit(act) is True
        assert act["activity_unit"] == "ug/mL" and act["value_raw_unit"] == "ug/mL"

    def test_the_magnitude_is_not_touched(self):
        """Spelling only. A conversion here would desynchronise value_lower from activity_unit."""
        act = {"activity_type": "MIC", "value_lower": 0.3, "activity_unit": "mM",
               "value_raw": "0.3", "value_raw_unit": "mM"}
        P._repair_unit(act)
        assert act["value_lower"] == 0.3 and act["activity_unit"] == "mM"

    def test_a_clean_unit_is_a_no_op(self):
        act = {"activity_unit": "ug/mL", "value_raw_unit": "ug/mL"}
        assert P._repair_unit(act) is False


class TestProseTypeMove:
    LONG = "hydrophobicity (RP-HPLC retention time tR )"

    def test_prose_is_moved_and_the_type_emptied(self):
        act = {"activity_type": self.LONG, "value_raw": "12.5", "activity_unit": "min"}
        assert P._move_prose_type(act) is True
        assert act["activity_type"] == "" and act["activity_type_detail"] == self.LONG

    def test_a_real_type_is_untouched(self):
        act = {"activity_type": "Membrane Permeabilization"}
        assert P._move_prose_type(act) is False

    def test_moving_twice_changes_nothing(self):
        act = {"activity_type": self.LONG}
        P._move_prose_type(act)
        assert P._move_prose_type(act) is False


class TestRangeBound:
    """
    The scale comes from the record, not from the unit tables.

    `value_lower` has already been through whatever conversion its unit required, so the ratio
    between it and the lower bound parsed out of `value_raw` IS that conversion. This is what lets
    the step fill a bound in the right magnitude without knowing what the document printed.
    """

    def test_a_bound_is_added_in_the_unit_value_lower_already_uses(self):
        act = {"activity_type": "MIC", "value_raw": "0.3 to 0.5", "value_lower": 300.0,
               "activity_unit": "uM", "value_raw_unit": "mM"}
        assert P._add_range_bound(act) is True
        assert act["value_upper"] == 500.0, "0.3->300 is a factor of 1000; 0.5 must become 500"

    def test_an_unconverted_range_keeps_its_own_numbers(self):
        act = {"activity_type": "MIC", "value_raw": "5 to 50", "value_lower": 5.0,
               "activity_unit": "uM"}
        P._add_range_bound(act)
        assert act["value_upper"] == 50.0

    def test_the_hyphen_form_works_too(self):
        act = {"activity_type": "MIC", "value_raw": "5-50", "value_lower": 5.0,
               "activity_unit": "uM"}
        P._add_range_bound(act)
        assert act["value_upper"] == 50.0

    def test_an_existing_bound_is_never_overwritten(self):
        act = {"value_raw": "5 to 50", "value_lower": 5.0, "value_upper": 99.0}
        assert P._add_range_bound(act) is False
        assert act["value_upper"] == 99.0

    def test_a_single_value_gets_no_bound(self):
        act = {"value_raw": "50", "value_lower": 50.0}
        assert P._add_range_bound(act) is False

    def test_a_backwards_range_is_refused(self):
        act = {"value_raw": "64-2", "value_lower": 64.0}
        assert P._add_range_bound(act) is False

    def test_a_zero_lower_bound_gives_no_usable_scale(self):
        """0 -> anything is not a ratio, so the factor cannot be established."""
        act = {"value_raw": "0 to 50", "value_lower": 0.0}
        assert P._add_range_bound(act) is False

    def test_a_missing_value_lower_is_skipped(self):
        act = {"value_raw": "5 to 50"}
        assert P._add_range_bound(act) is False


class TestIncoherentUnitsAreDropped:
    def test_a_molecular_weight_filed_as_an_mic_is_dropped(self):
        kept, counts = P.polish_activities(
            [{"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}])
        assert kept == [] and counts["activity_dropped_incoherent_unit"] == 1

    def test_the_unit_is_repaired_before_coherence_is_judged(self):
        """
        Order matters here and nowhere else: a mojibake `�M` must be read as the micromolar
        it is, not rejected as an unrecognised unit.
        """
        kept, counts = P.polish_activities(
            [{"activity_type": "MIC", "value_raw": "5", "value_lower": 5.0,
              "activity_unit": FFFD + "M"}])
        assert len(kept) == 1 and kept[0]["activity_unit"] == "uM"
        assert counts["activity_dropped_incoherent_unit"] == 0

    def test_a_real_measurement_survives(self):
        kept, _ = P.polish_activities(
            [{"activity_type": "MIC", "value_raw": "8", "value_lower": 8.0,
              "activity_unit": "ug/mL"}])
        assert len(kept) == 1


class TestTheStepOverADirectory:
    def test_it_repairs_counts_and_records_what_it_did(self, corpus):
        root, source = corpus
        _write(str(source), "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "5 to 50", "value_lower": 5.0,
             "activity_unit": FFFD + "M"},
            {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"},
        ]))
        summary = P.polish_extractions(base_dir=str(root))
        assert summary["documents"] == 1 and summary["documents_changed"] == 1
        assert summary["unit_repaired"] == 1
        assert summary["range_bound_added"] == 1
        assert summary["activity_dropped_incoherent_unit"] == 1

        payload = json.load(open(os.path.join(str(source), "a.json")))
        acts = payload["peptides"][0]["biological_activities"]
        assert len(acts) == 1
        assert acts[0]["activity_unit"] == "uM" and acts[0]["value_upper"] == 50.0
        assert payload["metadata"]["polish"]["version"] == P.POLISH_VERSION
        assert payload["metadata"]["polish"]["unit_repaired"] == 1

    def test_it_is_idempotent(self, corpus):
        """
        The property that lets it sit in `all`. Verified on real files too: a second pass leaves
        them byte-identical.
        """
        root, source = corpus
        path = _write(str(source), "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "0.3 to 0.5", "value_lower": 300.0,
             "activity_unit": FFFD + "M", "value_raw_unit": "mM"},
        ]))
        P.polish_extractions(base_dir=str(root))
        first = open(path, "rb").read()
        second_summary = P.polish_extractions(base_dir=str(root))
        assert second_summary["documents_changed"] == 0
        assert open(path, "rb").read() == first

    def test_a_dry_run_writes_nothing(self, corpus):
        root, source = corpus
        path = _write(str(source), "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}]))
        before = open(path, "rb").read()
        summary = P.polish_extractions(base_dir=str(root), dry_run=True)
        assert summary["activity_dropped_incoherent_unit"] == 1
        assert open(path, "rb").read() == before

    def test_an_unreadable_document_is_counted_not_skipped_silently(self, corpus):
        root, source = corpus
        (source / "broken.json").write_text('{"metadata": {"skipped": tr')
        _write(str(source), "ok.json", _doc([
            {"activity_type": "MIC", "value_raw": "8", "value_lower": 8.0,
             "activity_unit": "ug/mL"}]))
        summary = P.polish_extractions(base_dir=str(root))
        assert summary["unreadable"] == 1 and summary["documents"] == 2

    def test_only_recognised_source_subdirectories_are_touched(self, corpus):
        """
        The allowlist rule. `pubmed.bak` and `pubmed_old` accumulate beside the live sources, and
        a raw walk over this tree once shipped a database built entirely from superseded
        extractions.
        """
        root, source = corpus
        backup = root / "pubmed.bak"
        backup.mkdir()
        stale = _write(str(backup), "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}]))
        before = open(stale, "rb").read()
        _write(str(source), "b.json", _doc([
            {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}]))
        summary = P.polish_extractions(base_dir=str(root))
        assert summary["documents"] == 1, "the backup directory must not be read"
        assert open(stale, "rb").read() == before

    def test_a_file_changed_underneath_is_left_alone(self, corpus, monkeypatch):
        """
        Step 9 writes into this same directory for days. Reinstating the version this step read
        would discard a fresh re-extraction -- and because that older content still carries the
        refined-pass marker, step 9's resume would never revisit it.
        """
        import time
        root, source = corpus
        path = _write(str(source), "a.json", _doc([
            {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}]))

        real = P._write_atomically

        def racing(target, payload, expect_mtime=None):
            os.utime(target, (time.time() + 5, time.time() + 5))
            return real(target, payload, expect_mtime)

        monkeypatch.setattr(P, "_write_atomically", racing)
        summary = P.polish_extractions(base_dir=str(root))
        assert summary["skipped_changed_underneath"] == 1
        assert summary["documents_changed"] == 0
        assert json.load(open(path))["peptides"][0]["biological_activities"], (
            "the document must keep the content the other writer left")

    def test_the_limit_stops_early(self, corpus):
        root, source = corpus
        for i in range(5):
            _write(str(source), f"{i}.json", _doc([
                {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}]))
        assert P.polish_extractions(base_dir=str(root), limit=2)["documents"] == 2

    def test_a_record_with_no_activities_is_untouched(self, corpus):
        root, source = corpus
        path = _write(str(source), "a.json", _doc([]))
        before = open(path, "rb").read()
        assert P.polish_extractions(base_dir=str(root))["documents_changed"] == 0
        assert open(path, "rb").read() == before

    def test_a_non_dict_activity_is_carried_through(self):
        """Malformed input is preserved rather than dropped; this step repairs, it does not prune."""
        kept, counts = P.polish_activities(["not a dict", None])
        assert kept == ["not a dict", None]
        assert not any(counts.values())


class TestTheStepNumberingIsConsistent:
    """
    Guards the renumber this step required.

    Inserting step 11 shifted 11-52 to 12-53, which is the project-sanctioned way to add a step
    (lettered steps like `19b` are forbidden -- they share a predecessor's number, so no
    START_STEP/END_STEP range can select one alone). It is also the third such renumber, and each
    previous one left stale references behind, so the invariants are pinned here rather than
    re-checked by hand.
    """

    @staticmethod
    def _root():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _makefile(self):
        with open(os.path.join(self._root(), "Makefile"), encoding="utf-8") as handle:
            return handle.read()

    def test_step_numbers_are_contiguous_from_one(self):
        import re
        numbers = sorted(int(n) for n in
                         re.findall(r"^step_(\d\d)_\w+:", self._makefile(), re.M))
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"a gap or duplicate in the step sequence: {numbers}")

    def test_every_target_is_a_plain_integer(self):
        import re
        assert not re.findall(r"^step_(\d+[a-z])_", self._makefile(), re.M), (
            "lettered steps cannot be addressed by START_STEP/END_STEP alone")

    def test_end_step_covers_the_last_step(self):
        import re
        mk = self._makefile()
        last = max(int(n) for n in re.findall(r"^step_(\d\d)_\w+:", mk, re.M))
        declared = int(re.search(r"^END_STEP \?= (\d+)", mk, re.M).group(1))
        assert declared == last, (
            f"END_STEP is {declared} but the last step is {last}; `make all` would skip the tail")

    def test_each_run_step_call_matches_its_target_name(self):
        """
        The macro's first argument is what the range check and the registry see; the target name
        is what an operator types. A mismatch means `make step_30_x` records itself as step 29.
        """
        import re
        mk = self._makefile()
        for block in re.finditer(r"^step_(\d\d)_(\w+):\n(.*?)(?=^\S|\Z)", mk, re.M | re.S):
            number, name, body = block.group(1), block.group(2), block.group(3)
            calls = re.findall(r"\$\(call run_step,(\d+),", body)
            for called in calls:
                assert int(called) == int(number), (
                    f"step_{number}_{name} calls run_step with {called}")

    def test_the_polish_and_review_steps_run_in_the_right_order(self):
        """
        Extraction, then the rule polish, then the model review, then the first consumer.
        
        The order is the economy of the pair: step 11 makes every repair a RULE can make, so what
        reaches step 12 is only the residue nothing cheaper can decide -- 10,619 targeted reviews
        instead of ~87,000. Both must precede step 14 (external-enrich), the first stage to read
        extraction JSON. Checked through `all`'s prerequisite order, which is what determines
        execution.
        """
        import re
        mk = self._makefile()
        all_block = re.search(r"^all: (.*?)(?=^\t|\Z)", mk, re.M | re.S).group(1)
        order = [int(n) for n in re.findall(r"step_(\d\d)_\w+", all_block)]
        for step in (9, 10, 11, 12):
            assert step in order, f"step {step} is not in `all`"
        assert order.index(9) < order.index(10) < order.index(11) < order.index(12), (
            "extract -> revalidate -> rule polish -> model review")
        assert order.index(12) < order.index(14), (
            "the review must precede the first stage that reads extraction JSON")

    def test_live_documentation_names_only_real_targets(self):
        """
        Each previous renumber left stale `step_NN_` references in the docs. `data/zenodo_dump`
        is excluded on purpose: it documents a published 28-step dump and is a historical record.
        """
        import re
        root = self._root()
        real = set(re.findall(r"^(step_\d\d_\w+):", self._makefile(), re.M))
        stale = []
        for relative in ("README.md", "environments/README.md", "docs/md_to_rst.py",
                         "sample/cli.py", "sample/revision.py",
                         "data/manual_curation/README.md",
                         "data/manual_curation/evaluate_with_gemini.py"):
            path = os.path.join(root, relative)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            for ref in sorted(set(re.findall(r"step_\d\d_[a-z_0-9]+", text))):
                # A provenance value persisted into every peptide record, never a target name.
                if ref == "step_13_automated_revision":
                    continue
                if ref not in real:
                    stale.append(f"{relative}: {ref}")
        assert not stale, "stale step references: " + "; ".join(stale)

    def test_runtime_messages_name_the_right_step(self):
        """
        A message whose job is to tell an operator which step to run must name the step that
        actually runs that command.

        Four such messages were stale after this renumber -- `build-families` cited as step 35
        when it had become 36, `link-bioactivity` as 37 when it had become 38 -- and the same
        class of drift survived two earlier renumbers. The expected number is resolved from the
        Makefile, so this fails when the message and the pipeline disagree rather than when a
        hardcoded number changes.
        """
        import glob
        import re
        root = self._root()
        step_of = {}
        for block in re.finditer(r"\$\(call run_step,(\d+),[^,]*,(.*?)(?:,-[fd],)",
                                 self._makefile(), re.S):
            number, body = int(block.group(1)), block.group(2)
            for sub in re.findall(r"sample-cli ([a-z0-9-]+)", body):
                step_of.setdefault(sub, number)

        patterns = (re.compile(r"`sample-cli ([a-z0-9-]+)`\s*\((?:step|Step) (\d+)\)"),
                    re.compile(r"(?:step|Step) (\d+)\s*\(`sample-cli ([a-z0-9-]+)`\)"))
        wrong = []
        for path in sorted(glob.glob(os.path.join(root, "sample", "**", "*.py"),
                                     recursive=True)):
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            for index, pattern in enumerate(patterns):
                for found in pattern.finditer(text):
                    cmd = found.group(1 if index == 0 else 2)
                    said = int(found.group(2 if index == 0 else 1))
                    real = step_of.get(cmd)
                    if real is not None and real != said:
                        wrong.append(f"{os.path.relpath(path, root)}: `{cmd}` cited as step "
                                     f"{said}, runs at {real}")
        assert not wrong, "messages naming the wrong step: " + "; ".join(wrong)
