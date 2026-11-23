"""
Tests for :mod:`sample.langextract_records` -- the LangExtract backend and its fallback contract.

LangExtract itself is never invoked here. The library runs in an isolated environment as a
subprocess, so what matters at this level is the two things that can break without it: the record
assembly (worker payload -> records in the docling backend's shape) and the degradation path (every
failure must produce an empty list, because that is the signal the caller falls back on).

The fallback is the load-bearing behaviour. If this backend ever raised, or returned a partial
record, extraction would lose documents that docling could have handled.
"""
import json
import os
import time
import tempfile

import pytest

from sample import langextract_records as LR


def _payload(*extractions):
    return {"error": "", "extractions": list(extractions)}


def _peptide(name, sequence="", start=None, end=None):
    return {"extraction_class": "peptide", "extraction_text": name,
            "attributes": {"peptide_sequence": sequence}, "start_char": start, "end_char": end}


def _activity(owner, atype="MIC", value="25", unit="ug/mL", organism="Escherichia coli",
              evidence="MIC of 25 ug/mL"):
    return {"extraction_class": "activity", "extraction_text": evidence,
            "attributes": {"peptide_name": owner, "activity_type": atype, "value_raw": value,
                           "activity_unit": unit, "target_organism": organism},
            "start_char": None, "end_char": None}


@pytest.fixture(autouse=True)
def _isolate_payload_cache(tmp_path_factory):
    """
    Point the payload cache at a throwaway directory for every test in this module.

    Two reasons. It must not write into the repository's real ``data/langextract_cache`` -- running
    the suite would seed production cache entries from stubbed payloads. And a test that does not
    think about caching will otherwise get a HIT from an entry an earlier test wrote: that is how
    the request-timeout test started asserting against an empty command list, because the cached
    document never reached the subprocess at all.
    """
    import sample.langextract_records as _LR

    _LR.LANGEXTRACT_CACHE_DIR = str(tmp_path_factory.mktemp("langextract-cache"))
    yield


@pytest.fixture(autouse=True)
def _clear_availability_cache():
    """
    available() is lru_cached -- right in production, since a per-document import probe would cost
    a subprocess each time -- but it makes these tests order-dependent. Any earlier test that
    reached the real backend cached "available", and the fallback assertions below then saw True
    where they require False.
    """
    # getattr: some tests below replace available() with a plain lambda, which has no cache.
    def clear():
        getattr(LR.available, "cache_clear", lambda: None)()

    clear()
    yield
    clear()


class TestRecordAssembly:
    def test_a_peptide_with_its_activity_becomes_one_record(self) -> None:
        text = "Magainin 2 (GIGKFLHSAK) showed an MIC of 25 ug/mL against Escherichia coli."
        records = LR._records_from_payload(
            _payload(_peptide("Magainin 2", "GIGKFLHSAK"), _activity("Magainin 2")), text)
        assert len(records) == 1
        record = records[0]
        assert record["peptide_name"] == "Magainin 2"
        assert record["peptide_sequence"] == "GIGKFLHSAK"
        assert len(record["biological_activities"]) == 1
        activity = record["biological_activities"][0]
        assert activity["activity_type"] == "MIC"
        assert activity["activity_unit"] == "ug/mL"
        assert activity["target_organism"] == "Escherichia coli"

    def test_the_record_shape_matches_the_docling_backend(self) -> None:
        # Both backends feed the same downstream code; a missing key would surface as a KeyError
        # deep in serialisation rather than here.
        records = LR._records_from_payload(
            _payload(_peptide("JG436"), _activity("JG436")), "JG436 was tested.")
        for key in ("peptide_name", "peptide_sequence", "biological_activities", "context",
                    "source_page"):
            assert key in records[0], f"missing {key}"

    def test_source_grounding_offsets_are_preserved(self) -> None:
        # The reason to prefer this backend: a record can be located in the source document.
        records = LR._records_from_payload(
            _payload(_peptide("Magainin 2", "GIGKFLHSAK", start=11, end=21)), "x" * 50)
        assert records[0]["source_char_start"] == 11
        assert records[0]["source_char_end"] == 21

    def test_an_activity_naming_an_unextracted_peptide_still_yields_a_record(self) -> None:
        # The measurement is real evidence. Dropping it because the peptide extraction missed
        # would discard exactly the recall this backend exists to add.
        records = LR._records_from_payload(_payload(_activity("Polymyxin B")), "Polymyxin B 4 ug/mL")
        assert len(records) == 1
        assert records[0]["biological_activities"]

    def test_activities_are_matched_to_their_peptide_case_insensitively(self) -> None:
        records = LR._records_from_payload(
            _payload(_peptide("Magainin 2"), _activity("MAGAININ 2")), "Magainin 2")
        assert len(records) == 1
        assert len(records[0]["biological_activities"]) == 1

    def test_an_activity_without_a_unit_is_dropped(self) -> None:
        # strict_schema has no valid-unit entry for "", so such a record is discarded downstream
        # anyway. Dropping it here makes the loss visible instead of silent.
        records = LR._records_from_payload(
            _payload(_peptide("P1"), _activity("P1", unit="")), "P1 was tested")
        assert records[0]["biological_activities"] == []

    def test_an_activity_without_a_value_is_dropped(self) -> None:
        records = LR._records_from_payload(
            _payload(_peptide("P1"), _activity("P1", value="")), "P1 was tested")
        assert records[0]["biological_activities"] == []

    def test_the_raw_unit_is_stated_alongside_the_canonical_one(self) -> None:
        # Nothing on this path converts magnitudes, so the two must agree -- otherwise
        # preprocessing.raw_value_and_unit pairs a converted value with an unconverted unit.
        records = LR._records_from_payload(
            _payload(_peptide("P1"), _activity("P1", unit="uM")), "P1")
        activity = records[0]["biological_activities"][0]
        assert activity["activity_unit"] == activity["value_raw_unit"] == "uM"

    def test_an_empty_payload_yields_no_records(self) -> None:
        assert LR._records_from_payload(_payload(), "some text") == []


class TestFallbackContract:
    """Every failure mode must return [] -- that empty list is what the caller falls back on."""

    def test_a_missing_text_file_returns_empty(self) -> None:
        assert LR.extract_records_from_text("/nonexistent/path/doc.txt") == []

    def test_an_unbuilt_environment_returns_empty_without_running_anything(self, monkeypatch) -> None:
        monkeypatch.setattr(LR, "langextract_python", lambda: None)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("Magainin 2 showed an MIC of 25 ug/mL.")
            path = handle.name
        try:
            assert LR.extract_records_from_text(path) == []
        finally:
            os.unlink(path)

    def test_a_worker_error_payload_returns_empty(self) -> None:
        assert LR._records_from_payload({"error": "langextract unavailable",
                                         "extractions": []}, "text") == []

    def test_environment_resolution_never_falls_back_to_path(self, monkeypatch) -> None:
        # The bug this guards: peptide_predictors._predictor_command falls back to
        # shutil.which("python"), which resolved to a DIFFERENT environment's interpreter --
        # one with no langextract -- so available() reported True and every document paid for a
        # doomed subprocess. An isolated interpreter must come from its own environment or not
        # at all.
        monkeypatch.setenv("CONDA_ENVS_PATH", "/definitely/not/a/real/envs/dir")
        monkeypatch.setattr(os.path, "expanduser", lambda p: "/definitely/not/real")
        assert LR.langextract_python() is None
        assert LR.available() is False


class TestRequestTimeoutReachesTheWorker:
    """
    The per-request timeout must be passed explicitly, and must sit below the subprocess bound.

    langextract's Ollama provider defaults its HTTP timeout to 120s and raises
    InferenceRuntimeError when it expires. The worker turns any exception into "no records" for the
    WHOLE document, so one slow chunk -- or a model paying its load time inside the first request --
    silently costs every peptide in that document. Measured: qwen2.5-coder:1.5b returned 0 records
    on its first document for this reason alone, then 18 records on the next once warm.
    """

    def test_the_timeout_is_on_the_command(self, monkeypatch, tmp_path) -> None:
        captured = {}

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            captured["command"] = list(command)
            out = command[command.index("--out") + 1]
            with open(out, "w", encoding="utf-8") as handle:
                handle.write('{"error": "", "extractions": []}')
            return _Completed()

        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")
        monkeypatch.setattr(LR.subprocess, "run", fake_run)
        monkeypatch.setattr(LR, "available", lambda: True)
        monkeypatch.setattr(LR, "langextract_python", lambda: ["/env/bin/python"])

        LR.extract_records_from_text(str(text))

        command = captured.get("command", [])
        assert "--request-timeout" in command, "the per-request timeout never reached the worker"
        assert int(command[command.index("--request-timeout") + 1]) == LR.REQUEST_TIMEOUT

    def test_the_request_timeout_stays_below_the_subprocess_timeout(self) -> None:
        # If the per-request bound exceeded the subprocess bound, the subprocess would be killed
        # first and the provider's own timeout could never fire -- the document would be lost with
        # a less specific error.
        assert LR.REQUEST_TIMEOUT < LR.DEFAULT_TIMEOUT

    def test_it_is_longer_than_langextracts_unsafe_default(self) -> None:
        assert LR.REQUEST_TIMEOUT > 120


class TestPayloadCache:
    """
    The worker payload is cached, because this backend is the dominant cost in the pipeline.

    Measured at 101.4 s/document over the curation benchmark -- roughly 1,301 GPU-hours for one
    pass over the 46,189-document corpus. The extraction stage is re-run often while later stages
    are developed, and without a cache each re-run pays that again for documents whose text and
    model have not changed.
    """

    def _stub_worker(self, monkeypatch, payload, calls):
        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            calls.append(command)
            out = command[command.index("--out") + 1]
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return _Completed()

        monkeypatch.setattr(LR.subprocess, "run", fake_run)
        monkeypatch.setattr(LR, "available", lambda: True)
        monkeypatch.setattr(LR, "langextract_python", lambda: ["/env/bin/python"])

    def _payload(self):
        return {"error": "", "extractions": [
            {"extraction_class": "peptide", "extraction_text": "Magainin 2",
             "attributes": {"peptide_sequence": "GIGKFLHSAK"},
             "start_char": 0, "end_char": 10},
        ]}

    def test_a_second_call_does_not_rerun_the_model(self, monkeypatch, tmp_path) -> None:
        calls = []
        self._stub_worker(monkeypatch, self._payload(), calls)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        text = tmp_path / "doc.txt"
        text.write_text("Magainin 2 GIGKFLHSAK showed an MIC of 25 ug/mL", encoding="utf-8")

        first = LR.extract_records_from_text(str(text))
        second = LR.extract_records_from_text(str(text))
        assert len(calls) == 1, "the cached document was sent to the model again"
        assert [r["peptide_sequence"] for r in first] == [r["peptide_sequence"] for r in second]

    def test_changed_text_is_a_cache_miss(self, monkeypatch, tmp_path) -> None:
        # Keyed on CONTENT, not path: an OCR pass that rewrites a document's text must produce a
        # fresh extraction rather than reusing the result for the empty version.
        calls = []
        self._stub_worker(monkeypatch, self._payload(), calls)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        text = tmp_path / "doc.txt"
        text.write_text("first version of the document text", encoding="utf-8")
        LR.extract_records_from_text(str(text))
        text.write_text("OCR recovered an entirely different body of text", encoding="utf-8")
        LR.extract_records_from_text(str(text))
        assert len(calls) == 2

    def test_a_different_model_is_a_cache_miss(self, monkeypatch, tmp_path) -> None:
        calls = []
        self._stub_worker(monkeypatch, self._payload(), calls)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        text = tmp_path / "doc.txt"
        text.write_text("a document body", encoding="utf-8")
        LR.extract_records_from_text(str(text), model_id="gemma4:latest")
        LR.extract_records_from_text(str(text), model_id="qwen3:8b")
        assert len(calls) == 2, "two models must not share one cache entry"

    def test_concurrency_is_not_part_of_the_key(self, monkeypatch, tmp_path) -> None:
        # MAX_WORKERS changes throughput, not output. Including it would discard the whole cache
        # every time concurrency was tuned -- which is exactly when the cache is most wanted.
        calls = []
        self._stub_worker(monkeypatch, self._payload(), calls)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        text = tmp_path / "doc.txt"
        text.write_text("a document body", encoding="utf-8")
        monkeypatch.setattr(LR, "MAX_WORKERS", 2)
        LR.extract_records_from_text(str(text))
        monkeypatch.setattr(LR, "MAX_WORKERS", 8)
        LR.extract_records_from_text(str(text))
        assert len(calls) == 1

    def test_a_changed_prompt_invalidates_the_cache(self, monkeypatch, tmp_path) -> None:
        # The prompt and few-shot examples live in the worker script, so its digest is part of the
        # key. A stale cache after a prompt change is indistinguishable from a prompt that simply
        # did not help.
        calls = []
        self._stub_worker(monkeypatch, self._payload(), calls)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        text = tmp_path / "doc.txt"
        text.write_text("a document body", encoding="utf-8")
        monkeypatch.setattr(LR, "_worker_fingerprint", lambda: "aaaaaaaaaaaa")
        LR.extract_records_from_text(str(text))
        monkeypatch.setattr(LR, "_worker_fingerprint", lambda: "bbbbbbbbbbbb")
        LR.extract_records_from_text(str(text))
        assert len(calls) == 2

    def test_a_failed_run_is_not_cached(self, monkeypatch, tmp_path) -> None:
        # A payload produced by a timeout or a half-answered document must not be frozen in as
        # that document's permanent result.
        #
        # Retries off: this asserts a CACHING property by counting worker invocations, and
        # "Ollama timed out" is a retryable error, so leaving them on would make the count
        # 2 x (1 + RETRIES) and the assertion a statement about retry policy instead.
        # TestTransientFailuresAreRetried covers the retries themselves.
        monkeypatch.setattr(LR, "RETRIES", 0)
        calls = []
        self._stub_worker(monkeypatch, {"error": "Ollama timed out", "extractions": []}, calls)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        text = tmp_path / "doc.txt"
        text.write_text("a document body", encoding="utf-8")
        assert LR.extract_records_from_text(str(text)) == []
        assert LR.extract_records_from_text(str(text)) == []
        assert len(calls) == 2, "an errored payload was cached"

    def test_caching_can_be_disabled(self, monkeypatch, tmp_path) -> None:
        calls = []
        self._stub_worker(monkeypatch, self._payload(), calls)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", "")
        text = tmp_path / "doc.txt"
        text.write_text("a document body", encoding="utf-8")
        LR.extract_records_from_text(str(text))
        LR.extract_records_from_text(str(text))
        assert len(calls) == 2


class TestTheGate:
    """
    Which documents earn a gemma4 call. The CHEAP passes always run regardless.

    Measured over 63 documents holding 247 LangExtract-attributable peptides, against 4,000
    others: "unit" saves 8.5% of calls for 2.4% of those peptides, while "assay+unit" saves 55.8%
    and loses **69.2%**. The aggressive gate is not merely lossier, it is lossy in a biased way --
    LangExtract's value sits in documents whose text lacks clean assay-and-unit strings (scanned,
    OCR'd, table-heavy), so that gate is anti-correlated with where the model helps.
    """

    def test_a_document_with_a_concentration_unit_is_worth_running(self, monkeypatch) -> None:
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "unit")
        assert LR.worth_running("MIC of 25 ug/mL against Escherichia coli") is True
        assert LR.worth_running("IC50 of 12.5 uM in MCF-7 cells") is True
        assert LR.worth_running("an MIC of 4 µg/mL") is True

    def test_a_document_with_no_unit_is_not(self, monkeypatch) -> None:
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "unit")
        assert LR.worth_running("a review of folding thermodynamics, no measurements") is False

    def test_an_empty_document_is_never_worth_running(self, monkeypatch) -> None:
        # Costs no recall: the cheap passes see the same nothing.
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "unit")
        assert LR.worth_running("") is False
        assert LR.worth_running("   \n  ") is False

    def test_the_gate_can_be_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "none")
        assert LR.worth_running("no units anywhere in this text") is True

    def test_the_aggressive_gate_requires_both(self, monkeypatch) -> None:
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "assay+unit")
        assert LR.worth_running("MIC of 25 ug/mL") is True
        assert LR.worth_running("a concentration of 25 ug/mL, no assay named") is False
        assert LR.worth_running("MIC reported, but no unit given") is False

    def test_an_unrecognised_gate_costs_throughput_not_recall(self, monkeypatch) -> None:
        # A typo in the environment must not silently start discarding documents.
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "untit")
        assert LR.worth_running("no units anywhere") is True

    def test_the_default_is_the_conservative_gate(self) -> None:
        import os as _os

        assert _os.getenv("LANGEXTRACT_GATE", "unit").strip().lower() == "unit"


#: A paper-shaped document: a body with a measurement, then a real bibliography.
# Sized so truncation clears MIN_KEPT_FRACTION (62% kept): with a shorter body the bibliography
# dominates, the cut falls below the floor and the full text is sent -- which is the behaviour
# `test_truncation_that_guts_the_document_is_distrusted` covers separately.
_BODY = ("Peptide magainin-2 was assayed against Escherichia coli and showed an MIC of "
         "25 ug/mL under standard broth microdilution conditions. " * 60)
_REFERENCES = ("REFERENCES\n\n"
               + "".join(f"{i}. Smith J, Jones A, et al. Antimicrobial peptides. "
                         f"J Pept Sci. 2019;{i}:100-120.\n" for i in range(1, 60)))


def _text_file(tmp_path, name, content, corpus="literature_data_text/pubmed"):
    directory = tmp_path / corpus
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestReferenceTruncation:
    """
    The bibliography is 20-30% of a paper, cannot hold a bioactivity measurement, and was being
    chunked and sent to the model like everything else. Removing it is the SAFE kind of reduction
    -- structural boilerplate rather than a filter on signal -- unlike the `assay+unit` gate, whose
    own measurements show it discards 69.2% of this backend's peptides for 55.8% of the calls.
    """

    def test_a_paper_loses_its_bibliography(self, tmp_path) -> None:
        path = _text_file(tmp_path, "paper.txt", _BODY + "\n\n" + _REFERENCES)
        prepared = LR.model_input_text(path)
        assert "REFERENCES" not in prepared
        assert "J Pept Sci" not in prepared
        assert "25 ug/mL" in prepared, "the measurement must survive"
        assert len(prepared) < len(_BODY + _REFERENCES)

    def test_a_patent_is_never_truncated(self, tmp_path) -> None:
        # Measured over 198 patents: truncate_references keeps a MEDIAN of 0.0% of a patent,
        # because a patent's prior-art citations sit at the front rather than the end, so the
        # heuristic cuts the body. Applying it would have extracted 3 of every 4 patents from an
        # empty string. This test is the guard on that.
        content = _REFERENCES + "\n\n" + _BODY
        path = _text_file(tmp_path, "pat.txt", content,
                          corpus="patent_data_text/google_patents")
        assert LR.model_input_text(path) == content

    def test_truncation_that_guts_the_document_is_distrusted(self, tmp_path) -> None:
        path = _text_file(tmp_path, "mostly-refs.txt", "Short body with 25 ug/mL.\n\n" + _REFERENCES)
        prepared = LR.model_input_text(path)
        # Keeping a sliver would mean sending the model almost nothing; the full text is safer.
        assert prepared.startswith("Short body")
        assert "REFERENCES" in prepared, "below MIN_KEPT_FRACTION the full text must be sent"

    def test_the_floor_is_honoured_exactly(self, tmp_path, monkeypatch) -> None:
        path = _text_file(tmp_path, "paper.txt", _BODY + "\n\n" + _REFERENCES)
        monkeypatch.setattr(LR, "MIN_KEPT_FRACTION", 0.0)
        assert "REFERENCES" not in LR.model_input_text(path)
        monkeypatch.setattr(LR, "MIN_KEPT_FRACTION", 0.99)
        assert "REFERENCES" in LR.model_input_text(path), "a 99% floor distrusts any real cut"

    def test_can_be_switched_off(self, tmp_path, monkeypatch) -> None:
        path = _text_file(tmp_path, "paper.txt", _BODY + "\n\n" + _REFERENCES)
        monkeypatch.setattr(LR, "TRUNCATE_REFERENCES", False)
        assert "REFERENCES" in LR.model_input_text(path)

    def test_context_window_text_stays_whole(self, tmp_path) -> None:
        # The records' context windows resolve against the FULL text; a record whose context does
        # not contain its own sequence is deleted downstream, so truncating this would throw away
        # records the model got right.
        content = _BODY + "\n\n" + _REFERENCES
        path = _text_file(tmp_path, "paper.txt", content)
        assert LR._document_text(path) == content
        assert LR.model_input_text(path) != content

    def test_is_bounded_by_max_chars(self, tmp_path, monkeypatch) -> None:
        path = _text_file(tmp_path, "paper.txt", "x" * 5000)
        monkeypatch.setattr(LR, "MAX_CHARS", 1000)
        assert len(LR.model_input_text(path)) <= 1000


class TestCacheKeyFollowsWhatTheModelSaw:
    def test_truncation_changes_the_key(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        path = _text_file(tmp_path, "paper.txt", _BODY + "\n\n" + _REFERENCES)
        monkeypatch.setattr(LR, "TRUNCATE_REFERENCES", True)
        truncated = LR._cache_path(LR.model_input_text(path), "m")
        monkeypatch.setattr(LR, "TRUNCATE_REFERENCES", False)
        whole = LR._cache_path(LR.model_input_text(path), "m")
        assert truncated != whole, (
            "keying on the raw file would serve a pre-truncation payload for a "
            "post-truncation request")

    def test_identical_prepared_text_hits_the_same_key(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        a = _text_file(tmp_path, "a.txt", _BODY + "\n\n" + _REFERENCES)
        b = _text_file(tmp_path, "b.txt", _BODY + "\n\n" + _REFERENCES)
        assert (LR._cache_path(LR.model_input_text(a), "m")
                == LR._cache_path(LR.model_input_text(b), "m"))

class TestTheDocumentBudgetBoundsTheModelPass:
    """
    The langextract subprocess is the ONE hard timeout in the pipeline -- a subprocess can be
    killed, unlike a worker thread -- and it is also usually the largest single item in a
    document's budget. DEFAULT_TIMEOUT alone is 1200s against a 300s document budget, so an
    unclamped call would overrun the budget by itself and make every other limit cosmetic.
    """

    @staticmethod
    def _stub(monkeypatch, captured):
        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            captured["command"] = list(command)
            captured["timeout"] = kwargs.get("timeout")
            out = command[command.index("--out") + 1]
            with open(out, "w", encoding="utf-8") as handle:
                handle.write('{"error": "", "extractions": []}')
            return _Completed()

        monkeypatch.setattr(LR.subprocess, "run", fake_run)
        monkeypatch.setattr(LR, "available", lambda: True)
        monkeypatch.setattr(LR, "langextract_python", lambda: ["/env/bin/python"])

    def test_the_subprocess_timeout_is_cut_to_the_remaining_budget(self, monkeypatch, tmp_path):
        from sample import deadlines

        captured = {}
        self._stub(monkeypatch, captured)
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        with deadlines.Budget(42):
            LR.extract_records_from_text(str(text))
        assert captured["timeout"] <= 42
        assert captured["timeout"] > 0

    def test_the_per_request_timeout_follows_it_down(self, monkeypatch, tmp_path):
        """
        A request timeout above the subprocess's own lifetime means a request still running
        when the worker is killed -- which Ollama finishes anyway, charging the next document
        for a result nobody will read.
        """
        from sample import deadlines

        captured = {}
        self._stub(monkeypatch, captured)
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        with deadlines.Budget(42):
            LR.extract_records_from_text(str(text))
        command = captured["command"]
        assert int(command[command.index("--request-timeout") + 1]) <= 42

    def test_an_exhausted_budget_skips_the_pass_without_spawning_anything(self, monkeypatch,
                                                                         tmp_path):
        from sample import deadlines

        captured = {}
        self._stub(monkeypatch, captured)
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        with deadlines.Budget(0.001):
            time.sleep(0.01)
            assert LR.extract_records_from_text(str(text)) == []
        assert "command" not in captured, "a subprocess was spawned with no budget left"

    def test_a_cached_document_is_served_even_with_no_budget_left(self, monkeypatch, tmp_path):
        """
        The cache is consulted BEFORE the budget is. A cache hit costs a file read, so refusing
        it would throw away free work for exactly the document that most needs a cheap path.
        """
        from sample import deadlines

        captured = {}
        self._stub(monkeypatch, captured)
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        LR.extract_records_from_text(str(text))          # populates the cache
        captured.clear()
        with deadlines.Budget(0.001):
            time.sleep(0.01)
            LR.extract_records_from_text(str(text))      # must still be served
        assert "command" not in captured, "a cache hit spawned a subprocess anyway"

    def test_without_a_budget_the_default_timeouts_are_unchanged(self, monkeypatch, tmp_path):
        from sample import deadlines

        deadlines.clear()
        captured = {}
        self._stub(monkeypatch, captured)
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        LR.extract_records_from_text(str(text))
        assert captured["timeout"] == LR.DEFAULT_TIMEOUT
        command = captured["command"]
        assert int(command[command.index("--request-timeout") + 1]) == LR.REQUEST_TIMEOUT

class TestTransientFailuresAreRetried:
    """
    Measured 2026-10-16 over a 12-minute window: 7 of 43 ``/api/generate`` calls -- 16.3% --
    returned HTTP 500, against zero failures on the 172 ``/api/chat`` calls in the same window.
    The Ollama journal attributes it to prompt-cache allocation thrashing (65 evictions of
    220-440 MiB), so the same document succeeds or fails depending on what else is in flight.
    Each unretried 500 cost that document its whole model pass.
    """

    @staticmethod
    def _staged(monkeypatch, outcomes):
        """Stubs the worker so each call produces the next of `outcomes`: an error or a payload."""
        calls = []

        class _Completed:
            def __init__(self, code=0):
                self.returncode = code
                self.stdout = ""
                self.stderr = ""

        def fake_run(command, **kwargs):
            index = len(calls)
            calls.append(list(command))
            outcome = outcomes[min(index, len(outcomes) - 1)]
            out = command[command.index("--out") + 1]
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(outcome, handle)
            return _Completed()

        monkeypatch.setattr(LR.subprocess, "run", fake_run)
        monkeypatch.setattr(LR, "available", lambda: True)
        monkeypatch.setattr(LR, "langextract_python", lambda: ["/env/bin/python"])
        monkeypatch.setattr(LR.time, "sleep", lambda s: None)
        return calls

    def test_a_500_is_retried_and_the_second_attempt_is_used(self, monkeypatch, tmp_path):
        calls = self._staged(monkeypatch, [
            {"error": "InferenceRuntimeError: Bad status code from Ollama: 500"},
            {"error": "", "extractions": []},
        ])
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(tmp_path / "cache"))

        LR.extract_records_from_text(str(text))
        assert len(calls) == 2, "the 500 was not retried"

    def test_a_document_level_failure_is_not_retried(self, monkeypatch, tmp_path):
        """
        A bad document fails identically the second time. Retrying it would spend a slot the next
        document needs to reach the same answer.
        """
        calls = self._staged(monkeypatch, [{"error": "ValueError: could not parse extraction"}])
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        assert LR.extract_records_from_text(str(text)) == []
        assert len(calls) == 1

    def test_retries_are_bounded(self, monkeypatch, tmp_path):
        calls = self._staged(monkeypatch, [{"error": "Bad status code from Ollama: 500"}])
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        assert LR.extract_records_from_text(str(text)) == []
        assert len(calls) == 1 + LR.RETRIES

    def test_a_repeated_failure_is_not_cached_as_that_documents_answer(self, monkeypatch,
                                                                      tmp_path):
        """
        A payload carrying an error must never be frozen in as the document's permanent result --
        the next run would read the cache and never re-attempt it.
        """
        self._staged(monkeypatch, [{"error": "Bad status code from Ollama: 500"}])
        cache = tmp_path / "cache"
        monkeypatch.setattr(LR, "LANGEXTRACT_CACHE_DIR", str(cache))
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        LR.extract_records_from_text(str(text))
        written = list(cache.rglob("*.json")) if cache.exists() else []
        assert not written, "a failed attempt was cached"

    def test_no_retry_once_the_budget_is_spent(self, monkeypatch, tmp_path):
        """
        A retry that cannot finish is worse than none: it occupies a slot the next document needs
        in order to produce a result nobody will read.
        """
        from sample import deadlines

        calls = self._staged(monkeypatch, [{"error": "Bad status code from Ollama: 500"}])
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        with deadlines.Budget(deadlines.MIN_USEFUL_SLICE + 2):
            LR.extract_records_from_text(str(text))
        assert len(calls) == 1, "retried with no budget left to finish"

    def test_the_staged_input_survives_between_attempts(self, monkeypatch, tmp_path):
        """
        Cleanup used to live in a `finally` that ran per attempt. Deleting the staged, truncated
        input between attempts would leave the retry reading nothing.
        """
        seen = []

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            source = command[1 + command.index(command[1]) - 1] if False else command[2]
            seen.append(os.path.exists(source))
            out = command[command.index("--out") + 1]
            with open(out, "w", encoding="utf-8") as handle:
                json.dump({"error": "Bad status code from Ollama: 500"}
                          if len(seen) == 1 else {"error": "", "extractions": []}, handle)
            return _Completed()

        monkeypatch.setattr(LR.subprocess, "run", fake_run)
        monkeypatch.setattr(LR, "available", lambda: True)
        monkeypatch.setattr(LR, "langextract_python", lambda: ["/env/bin/python"])
        monkeypatch.setattr(LR.time, "sleep", lambda s: None)
        text = tmp_path / "doc.txt"
        text.write_text("a peptide was assayed", encoding="utf-8")

        LR.extract_records_from_text(str(text))
        assert seen == [True, True], f"the worker input vanished between attempts: {seen}"

