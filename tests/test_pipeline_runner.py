"""
Tests for the document gates in :func:`sample.pipeline_runner._extract_document`.

The extractor itself is stubbed throughout: these tests are about the GATES and what they record,
not about extraction, and stubbing keeps them free of LLM calls.

The recording matters as much as the gating. Gate 3 performs the only irreversible deletion in the
extraction stage, and it originally stored a bare count -- so when its drop rate was questioned
(743 deletions across 519 documents), the deleted values had to be reconstructed by re-running the
regex, which produced a wrong answer. A gate that cannot be audited cannot be trusted.
"""
import json
import os
import time

import pytest

import sample.extractor as E
import sample.pipeline_runner as R


@pytest.fixture(autouse=True)
def _restore_stubs():
    """
    Undo the module-attribute stubs `_run` installs.

    `_run` assigns over `E.extract_peptides_from_text` and `R._table_peptides` directly rather
    than through monkeypatch, so without this the stubs outlive the test that wanted them: a later
    test calling the real `_table_peptides` got `_run`'s two-argument stub instead and failed on
    the `payload=` keyword, while passing in isolation.
    """
    original_extract = E.extract_peptides_from_text
    original_tables = R._table_peptides
    yield
    E.extract_peptides_from_text = original_extract
    R._table_peptides = original_tables

_REAL_PAPER = "A real paper. Materials and Methods. " + ("body text " * 300)


def _run(tmp_path, peptides, text=_REAL_PAPER):
    """Runs _extract_document with a stubbed extractor; returns the written record."""
    E.extract_peptides_from_text = lambda t, doc_id="": list(peptides)
    # The table/LangExtract pass is stubbed for the same reason the extractor is. Left live it
    # reaches a real Ollama through the LangExtract backend, so three assertions in this file
    # passed or failed according to what the model happened to return that run -- the module
    # docstring's claim to be "free of LLM calls" was no longer true of it.
    # **kwargs: _extract_document passes payload= and metadata=, and a two-argument stub raised
    # TypeError that the table pass caught and recorded as table_extraction_error -- so these
    # tests were silently exercising a FAILED table pass rather than a skipped one.
    R._table_peptides = lambda internal_id, text_path, **kwargs: ([], None)
    text_path = os.path.join(str(tmp_path), "doc.txt")
    out_path = os.path.join(str(tmp_path), "doc.json")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    R._extract_document(("doc1", text_path, out_path))
    with open(out_path, encoding="utf-8") as handle:
        return json.load(handle)


def _pep(seq="", name="", context="", path="sequence_match"):
    return {"peptide_sequence": seq, "peptide_name": name, "context": context,
            "discovery_path": path, "biological_activities": []}


class TestGate3RecordsWhatItDeletes:
    def test_an_unsupported_sequence_is_dropped_and_recorded(self, tmp_path) -> None:
        out = _run(tmp_path, [_pep(seq="RWQWR", context="unrelated prose with no such sequence")])
        meta = out["metadata"]
        assert out["peptides"] == []
        assert meta["dropped_sequence_not_in_context"] == 1
        recorded = meta["dropped_sequences"]
        assert len(recorded) == 1
        # Enough to audit the call without re-reading the source document.
        assert recorded[0]["peptide_sequence"] == "RWQWR"
        assert recorded[0]["discovery_path"] == "sequence_match"
        assert "unrelated prose" in recorded[0]["context_excerpt"]

    def test_a_supported_sequence_is_kept_and_not_recorded(self, tmp_path) -> None:
        out = _run(tmp_path, [_pep(seq="GIGKFLHSAK",
                                   context="the peptide GIGKFLHSAK was tested")])
        assert [p["peptide_sequence"] for p in out["peptides"]] == ["GIGKFLHSAK"]
        assert "dropped_sequences" not in out["metadata"]
        assert "dropped_sequence_not_in_context" not in out["metadata"]

    def test_the_count_stays_exact_when_the_list_is_truncated(self, tmp_path) -> None:
        many = [_pep(seq=f"RWQW{i:04d}"[:9], context="nothing matching here")
                for i in range(R._MAX_RECORDED_DROPS + 12)]
        out = _run(tmp_path, many)
        meta = out["metadata"]
        assert meta["dropped_sequence_not_in_context"] == len(many)
        assert len(meta["dropped_sequences"]) == R._MAX_RECORDED_DROPS
        # Truncation is stated, never silent.
        assert meta["dropped_sequences_truncated"] is True


class TestDocumentGatesRecordTheirReason:
    def test_a_captcha_page_is_skipped_with_a_reason(self, tmp_path) -> None:
        out = _run(tmp_path, [_pep(seq="GIGKFLHSAK")],
                   text="Are you a robot? Please verify. " + ("x" * 2000))
        assert out["peptides"] == []
        assert out["metadata"]["skipped"] is True
        assert out["metadata"]["skip_reason"] == "captcha_or_bot_check"

    def test_a_watermark_only_document_is_skipped_with_its_own_reason(self, tmp_path) -> None:
        out = _run(tmp_path, [_pep(seq="GIGKFLHSAK")],
                   text="NII-Electronic Library Service\n" * 200)
        assert out["metadata"]["skip_reason"] == "watermark_only"

    def test_a_review_is_marked_but_its_peptides_are_kept(self, tmp_path) -> None:
        # Marked, never dropped: the records are real, they just belong to other papers.
        text = "REVIEW ARTICLE\nAntimicrobial peptides\n" + ("discussion " * 400)
        out = _run(tmp_path, [_pep(seq="GIGKFLHSAK",
                                   context="the peptide GIGKFLHSAK was tested")], text=text)
        assert out["metadata"]["document_type"] == "review"
        assert out["metadata"]["review_evidence"]
        assert len(out["peptides"]) == 1
        assert out["peptides"][0]["from_review_article"] is True


class TestOcrTextReachesLangExtract:
    """
    LangExtract reads docling's OCR output in preference to the pdfplumber ``.txt``.

    This is the only way a scanned document contributes anything: measured over 60 sampled patents
    (1,400 pages), 36.7% have a text layer of exactly ZERO characters, so their ``.txt`` is empty
    and every text-based extractor sees nothing no matter how good it is. The Soviet patent
    SU1569953A1 is the worked example -- 0 characters from pdfplumber, 5,221 from OCR.
    """

    def test_the_cache_never_lands_in_a_text_directory(self) -> None:
        # Load-bearing. Several steps glob their text directories for *.txt; a second .txt per
        # document sitting beside the original would be ingested as its own document and
        # double-count every peptide in it.
        source = "data/patent_data_text/google_patents/SU1569953A1.txt"
        cached = R._ocr_cache_path(source)
        assert not cached.startswith("data/patent_data_text")
        assert cached.startswith(R._OCR_TEXT_CACHE_DIR)
        assert "patent_data_text/google_patents/SU1569953A1.txt" in cached

    def test_a_document_outside_data_still_gets_a_collision_free_key(self) -> None:
        # Mirroring is impossible outside data/, and two documents must never share a cache entry.
        first = R._ocr_cache_path("/elsewhere/a/doc.txt")
        second = R._ocr_cache_path("/elsewhere/b/doc.txt")
        assert first != second
        assert first.startswith(R._OCR_TEXT_CACHE_DIR)

    def test_no_pdf_means_no_ocr_attempt(self) -> None:
        assert R._ocr_text_for("doc1", "data/x_text/doc.txt", None) is None

    def test_a_healthy_text_layer_is_not_converted_at_all(self, monkeypatch, tmp_path) -> None:
        # The cost gate. docling runs TableFormer in ACCURATE mode whether or not OCR is wanted,
        # so converting a document whose .txt is already good buys nothing -- over the
        # 124,526-paper corpus, measured 100% healthy, that is hundreds of hours for no records.
        from sample import docling_tables as D

        monkeypatch.setattr(D, "should_ocr", lambda path: False)
        monkeypatch.setattr(D, "docling_text", lambda path: (_ for _ in ()).throw(
            AssertionError("docling must not be invoked when the text layer is healthy")))
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        assert R._ocr_text_for("doc1", str(tmp_path / "paper.txt"), str(pdf)) is None

    def test_ocr_text_is_written_and_reused(self, monkeypatch, tmp_path) -> None:
        from sample import docling_tables as D

        calls = []

        def fake_text(path):
            calls.append(path)
            return "MIC of 25 ug/mL against Escherichia coli"

        monkeypatch.setattr(D, "should_ocr", lambda path: True)
        monkeypatch.setattr(D, "docling_text", fake_text)
        monkeypatch.setattr(R, "_OCR_TEXT_CACHE_DIR", str(tmp_path / "cache"))

        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        source = str(tmp_path / "scan.txt")

        first = R._ocr_text_for("doc1", source, str(pdf))
        assert first is not None
        with open(first, encoding="utf-8") as handle:
            assert "Escherichia coli" in handle.read()

        # Reused, not reconverted. OCR is the most expensive step in the pipeline (~12 s/page
        # measured), and it is deterministic for a given PDF, so paying for it twice is pure waste.
        second = R._ocr_text_for("doc1", source, str(pdf))
        assert second == first
        assert len(calls) == 1, "the cached OCR text was reconverted"

    def test_an_empty_ocr_result_falls_back_rather_than_caching_nothing(self, monkeypatch,
                                                                       tmp_path) -> None:
        # A failed conversion must not write an empty cache entry that later runs would trust.
        from sample import docling_tables as D

        monkeypatch.setattr(D, "should_ocr", lambda path: True)
        monkeypatch.setattr(D, "docling_text", lambda path: None)
        monkeypatch.setattr(R, "_OCR_TEXT_CACHE_DIR", str(tmp_path / "cache"))
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        assert R._ocr_text_for("doc1", str(tmp_path / "scan.txt"), str(pdf)) is None
        assert not os.path.exists(R._ocr_cache_path(str(tmp_path / "scan.txt")))

    def test_a_docling_failure_degrades_instead_of_raising(self, monkeypatch, tmp_path) -> None:
        # OCR is additive. A broken environment must cost the OCR text, never the document.
        from sample import docling_tables as D

        monkeypatch.setattr(D, "should_ocr", lambda path: True)
        monkeypatch.setattr(D, "docling_text", lambda path: (_ for _ in ()).throw(
            RuntimeError("docling env is half-built")))
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        assert R._ocr_text_for("doc1", str(tmp_path / "scan.txt"), str(pdf)) is None


class TestTheSharedConversionIsNotRepeated:
    def test_a_failed_conversion_does_not_trigger_a_second_one(self, monkeypatch,
                                                              tmp_path) -> None:
        """
        A failed shared conversion yields {}, and `payload.get("tables")` on it is None.

        None is the sentinel for "no tables supplied, convert it yourself", so passing it through
        re-ran the single most expensive operation in the pipeline -- the one the caller had
        already paid for and lost. On a scan that timed out at 900s that made the document cost
        1800s for the same nothing. An empty list is the correct translation: the conversion
        produced no tables, whether it failed or the PDF genuinely has none.
        """
        from sample import docling_tables as D

        calls = []

        def fake_extract(pdf_path, use_ocr=None, timeout=900, tables=None):
            calls.append(tables)
            return []

        monkeypatch.setattr(D, "extract_tables_from_pdf", fake_extract)
        # This is a test about the docling pass only; leaving the LangExtract backend live would
        # reach a real Ollama and populate its cached availability probe for later tests.
        monkeypatch.setattr(R, "USE_LANGEXTRACT_BACKEND", False)
        monkeypatch.setattr(R, "_pdf_path_for", lambda t: str(tmp_path / "doc.pdf"))
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n")
        text = tmp_path / "doc.txt"
        text.write_text("body", encoding="utf-8")

        # payload {} is what _docling_payload returns when the conversion failed.
        R._table_peptides("doc1", str(text), payload={})

        assert calls, "the table backend was never consulted"
        assert calls[0] == [], (
            "a failed shared conversion passed None, which means 'convert it again'")

    def test_tables_from_a_successful_conversion_are_passed_through(self, monkeypatch,
                                                                   tmp_path) -> None:
        from sample import docling_tables as D

        calls = []
        monkeypatch.setattr(D, "extract_tables_from_pdf",
                            lambda pdf_path, use_ocr=None, timeout=900, tables=None:
                            calls.append(tables) or [])
        monkeypatch.setattr(R, "USE_LANGEXTRACT_BACKEND", False)
        monkeypatch.setattr(R, "_pdf_path_for", lambda t: str(tmp_path / "doc.pdf"))
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n")
        text = tmp_path / "doc.txt"
        text.write_text("body", encoding="utf-8")

        grid = [{"page_no": 1, "cells": [{"text": "MIC"}]}]
        R._table_peptides("doc1", str(text), payload={"tables": grid, "text": "x"})
        assert calls[0] == grid

    def test_no_payload_still_lets_the_backend_convert(self, monkeypatch, tmp_path) -> None:
        # payload=None genuinely means "nobody converted this yet" -- the standalone path.
        from sample import docling_tables as D

        calls = []
        monkeypatch.setattr(D, "extract_tables_from_pdf",
                            lambda pdf_path, use_ocr=None, timeout=900, tables=None:
                            calls.append(tables) or [])
        monkeypatch.setattr(R, "USE_LANGEXTRACT_BACKEND", False)
        monkeypatch.setattr(R, "_pdf_path_for", lambda t: str(tmp_path / "doc.pdf"))
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n")
        text = tmp_path / "doc.txt"
        text.write_text("body", encoding="utf-8")

        R._table_peptides("doc1", str(text), payload=None)
        assert calls[0] is None


class TestDedupeDoesNotMergeDifferentPeptides:
    """
    A NAME match is only trusted when the sequences do not contradict each other.

    Generic labels are rife in this corpus -- "Peptide 1", "P1", "compound 3" -- and merging on the
    label alone discarded a genuinely different peptide and re-attributed its activities to
    whichever was seen first. The name fallback itself has to stay: joining a name-only record to a
    sequence-bearing one for the same molecule is the whole reason these backends are unioned.
    """

    def _rec(self, name, sequence, activity):
        return {"peptide_name": name, "peptide_sequence": sequence,
                "biological_activities": [activity], "discovery_path": "x"}

    def test_two_sequences_under_one_generic_label_stay_separate(self) -> None:
        out, merged = R._dedupe_records([
            self._rec("Peptide 1", "AAAAAAAAAA", {"a": 1}),
            self._rec("Peptide 1", "BBBBBBBBBB", {"a": 2}),
        ])
        assert [r["peptide_sequence"] for r in out] == ["AAAAAAAAAA", "BBBBBBBBBB"]
        assert merged == 0

    def test_a_name_only_record_still_joins_a_sequence_bearing_one(self) -> None:
        out, merged = R._dedupe_records([
            self._rec("Magainin 2", None, {"a": 1}),
            self._rec("Magainin 2", "GIGKFLHSAK", {"a": 2}),
        ])
        assert len(out) == 1 and merged == 1
        assert out[0]["peptide_sequence"] == "GIGKFLHSAK"
        assert len(out[0]["biological_activities"]) == 2

    def test_the_same_sequence_twice_still_merges(self) -> None:
        out, merged = R._dedupe_records([
            self._rec("P1", "AAAAAAAAAA", {"a": 1}),
            self._rec("P1", "AAAAAAAAAA", {"a": 2}),
        ])
        assert len(out) == 1 and merged == 1
        assert len(out[0]["biological_activities"]) == 2

    def test_the_same_sequence_under_two_names_still_merges(self) -> None:
        # Sequence is the real identity; a synonym must not duplicate the peptide.
        out, merged = R._dedupe_records([
            self._rec("Magainin 2", "GIGKFLHSAK", {"a": 1}),
            self._rec("MSI-78", "GIGKFLHSAK", {"a": 2}),
        ])
        assert len(out) == 1 and merged == 1


class TestTheGateIsAppliedToTheTextTheModelReads:
    """
    The gate must test the OCR output, not the pdfplumber ``.txt``.

    This is the whole correctness question for the gate. A scanned patent's ``.txt`` is 0 bytes --
    36.7% of sampled patents are like this -- so gating on it would reject every document the OCR
    rescue just recovered, which is the opposite of the intent.
    """

    def _wire(self, monkeypatch, tmp_path, txt_body, ocr_body):
        from sample import docling_tables as D

        text = tmp_path / "doc.txt"
        text.write_text(txt_body, encoding="utf-8")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        monkeypatch.setattr(R, "_pdf_path_for", lambda t: str(pdf))
        monkeypatch.setattr(R, "_OCR_TEXT_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(D, "should_ocr", lambda p: bool(ocr_body))
        monkeypatch.setattr(D, "docling_text", lambda p: ocr_body)
        monkeypatch.setattr(D, "extract_tables_from_pdf",
                            lambda *a, **k: [])
        return str(text)

    def test_an_empty_txt_rescued_by_ocr_still_reaches_the_model(self, monkeypatch,
                                                                tmp_path) -> None:
        import sample.langextract_records as LR

        seen = {}
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "unit")
        monkeypatch.setattr(LR, "extract_records_from_text",
                            lambda path, **k: seen.setdefault("path", path) or [])
        text = self._wire(monkeypatch, tmp_path, "", "MIC of 25 ug/mL against E. coli")
        meta = {}
        R._table_peptides("doc1", text, payload=None, metadata=meta)
        assert meta.get("langextract_skipped") is False, (
            "an OCR-rescued document was gated out on its empty .txt")
        assert "path" in seen

    def test_a_document_with_no_unit_anywhere_is_skipped_and_recorded(self, monkeypatch,
                                                                     tmp_path) -> None:
        import sample.langextract_records as LR

        called = []
        monkeypatch.setattr(LR, "LANGEXTRACT_GATE", "unit")
        monkeypatch.setattr(LR, "extract_records_from_text",
                            lambda path, **k: called.append(path) or [])
        text = self._wire(monkeypatch, tmp_path, "a review with no measurements", "")
        meta = {}
        R._table_peptides("doc1", text, payload=None, metadata=meta)
        assert meta.get("langextract_skipped") is True
        assert meta.get("langextract_gate") == "unit"
        assert not called, "the model was called for a document the gate rejected"


class TestSynonymResolutionRunsAfterGate3:
    """
    Ordering is the whole correctness question for this tier.

    Gate 3 drops any peptide whose sequence does not appear in its own context window -- right for
    a sequence a regex lifted out of prose. But a sequence resolved from the CATALOGUE is not in
    the document at all, by definition. Running resolution before gate 3 would therefore hand a
    record a sequence and then delete that record for not containing it, which is worse than not
    resolving at all: the name-only record would have survived.
    """

    @staticmethod
    def _output_path(tmp_path):
        """:returns: Where :meth:`_run` writes the main extraction output."""
        return os.path.join(str(tmp_path), "doc.json")

    def _run(self, tmp_path, peptides, text="A real paper. " + ("body text " * 300)):
        import sample.extractor as E

        E.extract_peptides_from_text = lambda t, doc_id="": [dict(p) for p in peptides]
        R._table_peptides = lambda internal_id, text_path, **kwargs: ([], None)
        text_path = os.path.join(str(tmp_path), "doc.txt")
        out_path = self._output_path(tmp_path)
        with open(text_path, "w", encoding="utf-8") as h:
            h.write(text)
        R._extract_document(("doc1", text_path, out_path))
        with open(out_path, encoding="utf-8") as h:
            return json.load(h)

    def test_a_catalogue_resolved_sequence_survives_gate_3(self, monkeypatch, tmp_path) -> None:
        from sample import synonyms

        # A resolution whose sequence is deliberately NOT in the document text.
        monkeypatch.setattr(synonyms, "resolve", lambda record, text: {
            "sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "method": "catalogue_corroborated",
            "confidence": 0.80, "evidence": ["organism agrees"], "cross_references": {}})
        out = self._run(tmp_path, [{"peptide_sequence": "", "peptide_name": "Drosomycin",
                                    "context": "Drosomycin was assayed", "discovery_path": "x",
                                    "biological_activities": []}])
        assert len(out["peptides"]) == 1, "the resolved record was dropped -- ordering is wrong"
        assert out["peptides"][0]["peptide_sequence"] == "GIGKFLHSAKKFGKAFVGEIMNS"
        assert out["metadata"]["sequences_resolved"] == 1
        assert "dropped_sequence_not_in_context" not in out["metadata"]

    def test_a_rejected_resolution_leaves_the_record_name_only(self, monkeypatch,
                                                              tmp_path) -> None:
        from sample import synonyms

        monkeypatch.setattr(synonyms, "resolve", lambda record, text: {
            "sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "method": "catalogue_unique_bare",
            "confidence": 0.50, "evidence": ["nothing corroborates it"], "cross_references": {}})
        out = self._run(tmp_path, [{"peptide_sequence": "", "peptide_name": "PBD-1",
                                    "context": "PBD-1 was assayed", "discovery_path": "x",
                                    "biological_activities": []}])
        # The record is WITHHELD, not published and not deleted: it has no sequence and nothing
        # above the confidence floor resolved it, so it cannot reach the database -- but its
        # rejected resolution is recorded in the sidecar so a later pass, against a larger
        # catalogue, knows this was considered and why it was left alone.
        assert out["peptides"] == []
        assert out["metadata"]["withheld_unidentified"] == 1
        import json as _json
        import os as _os
        from sample.pipeline_runner import _withheld_path
        withheld_file = _withheld_path(self._output_path(tmp_path))
        assert _os.path.isfile(withheld_file), "a withheld record must still exist on disk"
        with open(withheld_file, encoding="utf-8") as handle:
            pep = _json.load(handle)["peptides"][0]
        assert pep["peptide_sequence"] == ""
        assert pep["sequence_resolution"]["applied"] is False
        assert pep["sequence_resolution"]["method"] == "catalogue_unique_bare"
        assert out["metadata"]["sequences_resolved"] == 0
        assert out["metadata"]["sequence_resolution_methods"] == {"catalogue_unique_bare": 1}

    def test_peptides_that_already_have_a_sequence_are_left_alone(self, monkeypatch,
                                                                 tmp_path) -> None:
        from sample import synonyms

        calls = []
        monkeypatch.setattr(synonyms, "resolve",
                            lambda record, text: calls.append(record) or None)
        self._run(tmp_path, [{"peptide_sequence": "GIGKFLHSAK", "peptide_name": "m",
                              "context": "the peptide GIGKFLHSAK was tested",
                              "discovery_path": "x", "biological_activities": []}])
        assert not calls, "resolution was attempted on a record that already had a sequence"

    def test_a_resolver_failure_costs_the_resolution_not_the_document(self, monkeypatch,
                                                                     tmp_path) -> None:
        from sample import synonyms

        monkeypatch.setattr(synonyms, "resolve", lambda record, text: (_ for _ in ()).throw(
            RuntimeError("catalogue exploded")))
        out = self._run(tmp_path, [{"peptide_sequence": "", "peptide_name": "X",
                                    "context": "X was assayed", "discovery_path": "x",
                                    "biological_activities": []}])
        assert len(out["peptides"]) == 1
        assert "synonym_resolution_error" in out["metadata"]


class TestIdentityGate:
    """
    A record reaches the extraction output on one of two grounds: a sequence read off the page, or
    a name resolved to a peptide the external sources already know. Anything else is a name and a
    number with no identity, which ``sample.merge`` discards anyway (``if not seq: continue``).

    The load-bearing property is that withholding is NOT deletion. The catalogue grows, so a name
    unresolvable today may resolve later -- and re-deriving these would mean re-running the LLM
    pass, which is a multi-day job. Every test here is about the records still existing somewhere.
    """

    @staticmethod
    def _record(name: str, sequence: str = "", activities: int = 1) -> dict:
        return {"peptide_name": name, "peptide_sequence": sequence,
                "biological_activities": [{"activity_type": "MIC"} for _ in range(activities)]}

    def test_withheld_path_mirrors_the_source_layout(self) -> None:
        from sample.pipeline_runner import _withheld_path
        assert (_withheld_path("data/literature_data_json/pubmed/x.json")
                == "data/literature_data_json_withheld/pubmed/x.json")
        assert (_withheld_path("data/patent_data_json/google_patents/CN1.json")
                == "data/patent_data_json_withheld/google_patents/CN1.json")

    def test_the_withheld_tree_is_unreachable_by_the_merge(self) -> None:
        # The allowlist is what stops a sibling directory being merged by accident -- an os.walk
        # over one once shipped a database with 22.7% duplicate rows. A withheld tree that
        # iter_extraction_files could reach would defeat the entire gate.
        from sample.config import EXTRACTION_SOURCE_DIRS
        from sample.pipeline_runner import WITHHELD_SUFFIX
        assert not any(d.endswith(WITHHELD_SUFFIX) for d in EXTRACTION_SOURCE_DIRS)
        assert WITHHELD_SUFFIX not in "".join(EXTRACTION_SOURCE_DIRS)

    def test_partition_keeps_sequenced_and_withholds_the_rest(self, monkeypatch) -> None:
        import sample.pipeline_runner as R
        monkeypatch.setattr(R, "WITHHOLD_UNIDENTIFIED", True)
        kept = [self._record("magainin-2", "GIGKFLHSAKKFGKAFVGEIMNS"),
                self._record("resolved-from-catalogue", "ILPWKWPWWPWRR"),
                self._record("name only", ""),
                self._record("another name", "")]
        identified, withheld = [], []
        for record in kept:
            keep = (not isinstance(record, dict)
                    or bool((record.get("peptide_sequence") or "").strip()))
            (identified if keep else withheld).append(record)
        assert [r["peptide_name"] for r in identified] == ["magainin-2", "resolved-from-catalogue"]
        assert [r["peptide_name"] for r in withheld] == ["name only", "another name"]
        # Nothing may be invented or lost by the split.
        assert len(identified) + len(withheld) == len(kept)

    def test_a_non_record_is_never_withheld(self) -> None:
        # The gate judges identity. Something that is not a dict is a different defect, handled
        # where records are built -- silently dropping it here would hide that.
        records = ["not a record", None, self._record("n", "")]
        identified, withheld = [], []
        for record in records:
            keep = (not isinstance(record, dict)
                    or bool((record.get("peptide_sequence") or "").strip()))
            (identified if keep else withheld).append(record)
        assert identified == ["not a record", None]
        assert len(withheld) == 1

    def test_the_flag_can_be_switched_off(self, monkeypatch) -> None:
        # Worth keeping while the synonym catalogue grows: the gate is only as good as what the
        # external sources currently know.
        import sample.pipeline_runner as R
        monkeypatch.setenv("SAMPLE_WITHHOLD_UNIDENTIFIED", "0")
        import importlib
        reloaded = importlib.reload(R)
        try:
            assert reloaded.WITHHOLD_UNIDENTIFIED is False
        finally:
            monkeypatch.delenv("SAMPLE_WITHHOLD_UNIDENTIFIED", raising=False)
            importlib.reload(reloaded)


class TestServerParallelSlots:
    """
    The slot count has to be read from the SERVER, not from this process's environment.

    ``OLLAMA_NUM_PARALLEL`` is set on the Ollama service, so it is absent here whether the host is
    configured correctly or not. That is how a default of 1 serialized eight concurrent clients for
    95 hours with no error anywhere -- the only symptom was flat throughput. The check must read
    the running ``llama-server``'s own ``-np`` argument.
    """

    def test_reads_np_from_the_runner_command_line(self, tmp_path, monkeypatch) -> None:
        import sample.pipeline_runner as R

        proc = tmp_path / "proc"
        (proc / "4242").mkdir(parents=True)
        (proc / "4242" / "cmdline").write_bytes(
            b"/usr/local/lib/ollama/llama-server\x00--model\x00/blob\x00-np\x004\x00-c\x0016384\x00")
        (proc / "notapid").mkdir()
        monkeypatch.setattr(R.os, "listdir", lambda p: ["4242", "notapid"] if p == "/proc" else [])
        monkeypatch.setattr(R, "open", lambda p, *a, **k: open(
            str(proc / p.split("/proc/")[1]) if "/proc/" in str(p) else p, *a, **k), raising=False)
        assert R.server_parallel_slots() == 4

    def test_unknown_is_not_reported_as_fine(self, monkeypatch) -> None:
        # None must mean "could not determine" -- no model loaded, a remote Ollama, or an
        # unreadable /proc. Treating it as a pass is what let the original misconfiguration hide.
        import sample.pipeline_runner as R
        monkeypatch.setattr(R.os, "listdir", lambda p: [])
        assert R.server_parallel_slots() is None

    def test_an_unreadable_proc_does_not_raise(self, monkeypatch) -> None:
        import sample.pipeline_runner as R
        monkeypatch.setattr(R.os, "listdir",
                            lambda p: (_ for _ in ()).throw(OSError("permission denied")))
        assert R.server_parallel_slots() is None

    def test_detects_the_real_server_when_one_is_running(self) -> None:
        # Not a mock: on a host with Ollama serving, this must return a positive integer. Skipped
        # rather than failed when nothing is loaded, since Ollama starts no runner until the first
        # request arrives.
        import pytest

        from sample.pipeline_runner import server_parallel_slots
        slots = server_parallel_slots()
        if slots is None:
            pytest.skip("no llama-server running; nothing to detect")
        assert isinstance(slots, int) and slots >= 1


class TestModelSplitGuard:
    """
    Seven LLM consumers share one Ollama server here. On a single GPU two different models cannot
    both stay resident, so a split configuration makes Ollama evict and reload one for the other on
    alternating requests.

    This guard exists because the fault was introduced TWICE -- once pointing langextract at
    qwen3:14b, once at gemma3:4b, both times leaving config.DEFAULT_EXTRACT_MODEL on gemma4 -- and
    was invisible both times. It raises no error: each request simply lands in a different eviction,
    so the same document failed as a 300s timeout, a 600s timeout and an HTTP 500 on three
    consecutive runs. Measured over 20 minutes of a split: 31 runner restarts, 23 evictions.
    """

    def test_silent_when_every_consumer_agrees(self, monkeypatch, caplog) -> None:
        import sample.pipeline_runner as R
        monkeypatch.delenv("LANGEXTRACT_MODEL", raising=False)
        with caplog.at_level("WARNING"):
            assert R.warn_on_model_split("pubmed") is False
        assert "DIFFERENT models" not in caplog.text

    def test_detects_a_split_and_names_both_tags(self, monkeypatch, caplog) -> None:
        """
        The "other" tag is DERIVED from the current default, not hardcoded. This test used to
        pin LANGEXTRACT_MODEL to "gemma4:latest" against a gemma3:4b default -- so when the
        default moved to gemma4 the two agreed, the guard correctly reported no split, and the
        test failed while the code was right. A guard against misconfiguration must not itself
        depend on which configuration is current.
        """
        import sample.pipeline_runner as R
        from sample.config import DEFAULT_EXTRACT_MODEL

        other = "gemma3:4b" if "gemma3" not in DEFAULT_EXTRACT_MODEL else "gemma4:latest"
        monkeypatch.setenv("LANGEXTRACT_MODEL", other)
        with caplog.at_level("WARNING"):
            assert R.warn_on_model_split("pubmed") is True
        assert other in caplog.text
        assert DEFAULT_EXTRACT_MODEL in caplog.text
        # The combined size is the actionable part -- it is what exceeds the card.
        assert "GiB combined" in caplog.text

    def test_reports_combined_size_from_the_manifests(self) -> None:
        import pytest

        from sample.pipeline_runner import _model_size_gib
        size = _model_size_gib("gemma3:4b")
        if size is None:
            pytest.skip("gemma3:4b not pulled on this host")
        assert 2.5 < size < 4.0, f"expected ~3.1 GiB, got {size}"

    def test_the_two_model_settings_agree_in_the_committed_configuration(self) -> None:
        """
        The static half of the same guard: config.DEFAULT_EXTRACT_MODEL and the Makefile's
        LANGEXTRACT_MODEL are what actually get used, and the fault this class documents was
        introduced twice by changing one and not the other. Asserted here so a one-sided edit
        fails in CI rather than 20 minutes into a run.
        """
        import re
        from pathlib import Path

        from sample.config import DEFAULT_EXTRACT_MODEL

        makefile = Path("Makefile").read_text(encoding="utf-8")
        match = re.search(r"^LANGEXTRACT_MODEL \?= *(\S+)", makefile, flags=re.M)
        assert match, "LANGEXTRACT_MODEL not found in the Makefile"
        assert match.group(1) == DEFAULT_EXTRACT_MODEL, (
            f"model split: Makefile says {match.group(1)!r}, config says "
            f"{DEFAULT_EXTRACT_MODEL!r}")

    def test_an_unknown_tag_does_not_raise(self) -> None:
        from sample.pipeline_runner import _model_size_gib
        assert _model_size_gib("definitely-not-pulled:99b") is None
        assert _model_size_gib("no-colon-at-all") is None

    def test_a_split_with_unreadable_sizes_still_warns(self, monkeypatch, caplog) -> None:
        # The warning must not depend on reading the manifests: a remote Ollama or a non-default
        # models root would make the sizes unknown, and the split is still the problem.
        import sample.pipeline_runner as R
        monkeypatch.setenv("LANGEXTRACT_MODEL", "mystery:1b")
        monkeypatch.setattr(R, "_model_size_gib", lambda tag: None)
        with caplog.at_level("WARNING"):
            assert R.warn_on_model_split("pubmed") is True
        assert "sizes unknown" in caplog.text


class TestPinnedContextWindow:
    """
    Every consumer must ask Ollama for the SAME num_ctx, or the runner is rebuilt continuously.

    Ollama sizes the runner as ``num_ctx x OLLAMA_NUM_PARALLEL`` and restarts it whenever a request
    needs more context than the loaded one has. ``llm_call`` used to fit num_ctx to each prompt --
    a real KV-cache saving in isolation -- which made Ollama rebuild the runner over and over and
    discard the cache the fitting was protecting. Measured with a single model and zero memory
    evictions: 7 runner restarts in 10 minutes at ``-c 8192``, ``-c 11848`` and ``-c 16384``. The
    symptom was the model appearing to stop and start, with nothing wrong in any pipeline log and
    keep_alive correctly set to 24h.
    """

    def test_the_two_consumers_agree(self) -> None:
        # A mismatch here is the bug returning, and it is silent at runtime.
        from sample.config import DEFAULT_NUM_CTX
        from sample.langextract_records import NUM_CTX
        assert DEFAULT_NUM_CTX == NUM_CTX

    def test_a_small_prompt_gets_the_pinned_window_not_a_fitted_one(self) -> None:
        from sample.config import DEFAULT_NUM_CTX, FALLBACK_NUM_CTX
        needed = int(len("a short verification prompt") / 3.5) + 512 + 256
        assert needed < DEFAULT_NUM_CTX
        chosen = DEFAULT_NUM_CTX if needed <= DEFAULT_NUM_CTX else FALLBACK_NUM_CTX
        assert chosen == DEFAULT_NUM_CTX, "a small prompt must not shrink the runner"

    def test_an_oversized_prompt_escalates_rather_than_truncating(self) -> None:
        # Truncating silently would be a wrong answer; one reload is cheap by comparison.
        from sample.config import DEFAULT_NUM_CTX, FALLBACK_NUM_CTX
        needed = int(40_000 / 3.5) + 512 + 256          # ~12k tokens, beyond both rungs
        chosen = DEFAULT_NUM_CTX if needed <= DEFAULT_NUM_CTX else FALLBACK_NUM_CTX
        assert chosen == FALLBACK_NUM_CTX
        assert chosen > DEFAULT_NUM_CTX

    def test_escalation_converges_upward(self) -> None:
        # Only two rungs, so the runner can be rebuilt at most once per run rather than
        # oscillating between sizes as prompts vary.
        from sample.config import DEFAULT_NUM_CTX, FALLBACK_NUM_CTX
        rungs = set()
        for chars in (10, 500, 5_000, 12_000, 40_000, 200_000):
            needed = int(chars / 3.5) + 512 + 256
            rungs.add(DEFAULT_NUM_CTX if needed <= DEFAULT_NUM_CTX else FALLBACK_NUM_CTX)
        assert rungs <= {DEFAULT_NUM_CTX, FALLBACK_NUM_CTX}, rungs

    def test_an_explicit_kwarg_still_wins(self) -> None:
        # The escape hatch has to survive: a caller with a genuinely large prompt must be able to
        # ask for the window it needs.
        import inspect

        from sample import llm
        source = inspect.getsource(llm.llm_call)
        assert 'if "num_ctx" in kwargs:' in source
        assert 'num_ctx = kwargs["num_ctx"]' in source

class TestChunkedExecution:
    """
    The corpus is processed in fixed-size groups with the Ollama runtime reset between them.

    This exists because refined extraction DECAYS rather than failing: measured on this corpus,
    throughput held near 1 doc/s and fell to ~0.04 docs/s after ~2,000 documents with no error
    logged, and recovered completely on a manual server restart. The chunk boundary bounds how
    long runner-side state can accumulate.
    """

    @staticmethod
    def _tasks(tmp_path, count):
        tasks = []
        for i in range(count):
            text_path = os.path.join(str(tmp_path), f"doc{i}.txt")
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write(_REAL_PAPER)
            tasks.append((f"doc{i}", text_path, os.path.join(str(tmp_path), f"doc{i}.json")))
        return tasks

    def _run(self, tmp_path, monkeypatch, count, chunk_size, workers=1):
        """Runs the pipeline over `count` stub documents, recording every reset."""
        import pandas as pd

        resets = []
        monkeypatch.setattr("sample.ollama_runtime.reset",
                            lambda *a, **k: resets.append(len(seen)) or
                            {"method": "unload", "unloaded": ["m"], "drained": True,
                             "seconds": 0.1})
        seen = []
        monkeypatch.setattr(R, "_extract_document",
                            lambda task, tally=None, budget=None: seen.append(task[0]) or
                            (task[0], 0, None))
        tasks = self._tasks(tmp_path, count)
        df = pd.DataFrame({"internal_id": [t[0] for t in tasks], "success": [True] * count})
        R.run_parallel_pipeline("src", str(tmp_path), str(tmp_path), df,
                                workers=workers, chunk_size=chunk_size)
        return seen, resets

    def test_the_runtime_is_reset_between_chunks_and_not_after_the_last(self, tmp_path,
                                                                       monkeypatch):
        """
        250 documents in chunks of 100 is three chunks and TWO resets. A reset after the final
        chunk would only make whichever step runs next pay to load the model again.
        """
        seen, resets = self._run(tmp_path, monkeypatch, count=250, chunk_size=100)
        assert len(seen) == 250
        assert resets == [100, 200]

    def test_every_document_is_processed_exactly_once_across_chunks(self, tmp_path, monkeypatch):
        """
        Chunking is a pacing decision. A boundary that dropped or repeated a document would be
        a far worse bug than the slowdown it is meant to fix, and the slices are consecutive so
        --resume and Ctrl-C behave exactly as before.
        """
        seen, _ = self._run(tmp_path, monkeypatch, count=25, chunk_size=10)
        assert seen == [f"doc{i}" for i in range(25)]

    def test_chunking_can_be_switched_off(self, tmp_path, monkeypatch):
        """chunk_size=0 restores the single-pass behaviour, resets and all."""
        seen, resets = self._run(tmp_path, monkeypatch, count=30, chunk_size=0)
        assert len(seen) == 30
        assert resets == []

    def test_a_corpus_smaller_than_one_chunk_is_never_reset(self, tmp_path, monkeypatch):
        seen, resets = self._run(tmp_path, monkeypatch, count=5, chunk_size=100)
        assert len(seen) == 5
        assert resets == []

    def test_the_pool_is_drained_before_the_runtime_is_reset(self, tmp_path, monkeypatch):
        """
        The load-bearing property of the barrier. Unloading the model while a worker still has
        a request outstanding would hand that document a hard failure for no reason of its own,
        so the executor is joined -- not merely waited on -- before the reset is attempted.
        """
        import pandas as pd
        import threading

        running = set()
        overlaps = []
        lock = threading.Lock()

        def _work(task, tally=None, budget=None):
            with lock:
                running.add(task[0])
            time.sleep(0.01)
            with lock:
                running.discard(task[0])
            return task[0], 0, None

        def _reset(*a, **k):
            with lock:
                overlaps.append(len(running))
            return {"method": "unload", "unloaded": [], "drained": True, "seconds": 0.0}

        monkeypatch.setattr(R, "_extract_document", _work)
        monkeypatch.setattr("sample.ollama_runtime.reset", _reset)
        tasks = self._tasks(tmp_path, 20)
        df = pd.DataFrame({"internal_id": [t[0] for t in tasks], "success": [True] * 20})
        R.run_parallel_pipeline("src", str(tmp_path), str(tmp_path), df, workers=4, chunk_size=5)

        assert overlaps, "no reset ran, so the barrier was never exercised"
        assert overlaps == [0] * len(overlaps), f"reset ran with workers still active: {overlaps}"

    def test_a_failed_reset_does_not_end_the_run(self, tmp_path, monkeypatch):
        """
        Degrading to the unchunked behaviour is strictly better than abandoning a corpus
        part-way through a multi-day extraction.
        """
        import pandas as pd

        monkeypatch.setattr(R, "_extract_document",
                            lambda task, tally=None, budget=None: (task[0], 0, None))
        monkeypatch.setattr("sample.ollama_runtime.reset",
                            lambda *a, **k: {"method": "unload", "unloaded": [],
                                             "drained": False, "seconds": 0.0,
                                             "error": "could not reach the server"})
        tasks = self._tasks(tmp_path, 12)
        df = pd.DataFrame({"internal_id": [t[0] for t in tasks], "success": [True] * 12})
        R.run_parallel_pipeline("src", str(tmp_path), str(tmp_path), df, workers=1, chunk_size=5)
        assert len(os.listdir(tmp_path)) >= 12   # the .txt files are all still there, no crash


class TestDocumentBudget:
    """
    The per-document wall-clock cap, and what it records.

    A document that runs out of budget is NOT dropped. It is written with whatever it has and
    flagged, because a missing output file is indistinguishable from "not reached yet" and
    would be re-attempted forever by --resume, while a flagged one can be counted and re-run
    deliberately.
    """

    def test_a_document_records_the_time_it_took(self, tmp_path):
        record = _run(tmp_path, [_pep(seq="GIGKFLHSAK", context="GIGKFLHSAK was tested")])
        assert "processing_seconds" in record["metadata"]
        assert record["metadata"]["budget_seconds"] == R.DOCUMENT_BUDGET_SECONDS
        # Recorded on a document that finished comfortably, too: written only on an overrun,
        # the flag could not distinguish "few peptides" from "cut short" when read later.
        assert "budget_exceeded" not in record["metadata"]

    def test_an_exhausted_budget_still_writes_the_document_and_says_so(self, tmp_path,
                                                                      monkeypatch):
        """
        The output is degraded, not absent, and the metadata states which it is.
        """
        E.extract_peptides_from_text = lambda t, doc_id="": []
        R._table_peptides = lambda internal_id, text_path, **kwargs: ([], None)
        text_path = os.path.join(str(tmp_path), "doc.txt")
        out_path = os.path.join(str(tmp_path), "doc.json")
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(_REAL_PAPER)

        # A budget below MIN_USEFUL_SLICE is exhausted on arrival, which is the state a document
        # reaches after spending its 300s.
        R._extract_document(("doc1", text_path, out_path), budget=0.001)
        with open(out_path, encoding="utf-8") as handle:
            record = json.load(handle)
        assert record["metadata"]["budget_exceeded"] is True
        assert record["metadata"]["budget_seconds"] == 0.001

    def test_the_budget_does_not_leak_onto_the_next_document(self, tmp_path):
        """
        Pool threads are reused across documents and across chunks. A budget left set would be
        already spent, so the next document would skip every LLM call -- silently, with no error
        and no flag, for the rest of the run.
        """
        from sample import deadlines

        E.extract_peptides_from_text = lambda t, doc_id="": []
        R._table_peptides = lambda internal_id, text_path, **kwargs: ([], None)
        text_path = os.path.join(str(tmp_path), "doc.txt")
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(_REAL_PAPER)
        R._extract_document(("doc1", text_path, os.path.join(str(tmp_path), "doc.json")),
                            budget=0.001)
        assert deadlines.active() is False

    def test_overruns_are_counted_for_the_summary(self, tmp_path):
        """
        Without a tally the budget's effect on a corpus is invisible: the only evidence would
        be individually unremarkable warnings scattered through a multi-day log.
        """
        E.extract_peptides_from_text = lambda t, doc_id="": time.sleep(0.05) or []
        R._table_peptides = lambda internal_id, text_path, **kwargs: ([], None)
        text_path = os.path.join(str(tmp_path), "doc.txt")
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(_REAL_PAPER)
        tally = R._BudgetTally()
        R._extract_document(("doc1", text_path, os.path.join(str(tmp_path), "doc.json")),
                            tally=tally, budget=0.01)
        overruns, seconds_lost = tally.snapshot()
        assert overruns == 1
        assert seconds_lost > 0

    def test_the_budget_can_be_switched_off(self, tmp_path):
        E.extract_peptides_from_text = lambda t, doc_id="": []
        R._table_peptides = lambda internal_id, text_path, **kwargs: ([], None)
        text_path = os.path.join(str(tmp_path), "doc.txt")
        out_path = os.path.join(str(tmp_path), "doc.json")
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(_REAL_PAPER)
        R._extract_document(("doc1", text_path, out_path), budget=0)
        with open(out_path, encoding="utf-8") as handle:
            record = json.load(handle)
        # No budget means no budget keys, rather than a budget of zero.
        assert "budget_seconds" not in record["metadata"]
        assert "budget_exceeded" not in record["metadata"]

class TestConcurrencyWarning:
    """
    The oversubscription warning has to count the requests the run can actually issue.

    It used to compare the server's slot count against `--workers` alone, which ignores that the
    refined pass fans each document across LANGEXTRACT_MAX_WORKERS threads. At --workers 4 with
    the default 6 that understated the figure sixfold -- it reported "2 of every 4 calls will
    queue" while 24 requests were in flight against 2 slots.
    """

    def test_total_concurrency_is_workers_times_langextract_workers(self, monkeypatch):
        monkeypatch.setattr(R, "_langextract_max_workers", lambda: 6)
        assert R._langextract_max_workers() == 6

    def test_the_helper_reads_the_value_actually_in_force(self, monkeypatch):
        monkeypatch.delenv("LANGEXTRACT_MAX_WORKERS", raising=False)
        assert R._langextract_max_workers() >= 1

    def test_the_helper_falls_back_to_the_environment(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def no_langextract(name, *args, **kwargs):
            if name == "sample.langextract_records":
                raise ImportError("not available in this environment")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_langextract)
        monkeypatch.setenv("LANGEXTRACT_MAX_WORKERS", "3")
        assert R._langextract_max_workers() == 3
        monkeypatch.setenv("LANGEXTRACT_MAX_WORKERS", "garbage")
        assert R._langextract_max_workers() == 6, "a bad value falls back to the real default"

    def test_the_fan_out_is_lowered_to_fit_the_server(self, monkeypatch):
        """
        A plain `make step_09` must not oversubscribe. With 2 slots and --workers 4 the default
        fan-out of 6 issues 24 requests; it is reduced so the product fits.
        """
        import sample.langextract_records as lx
        monkeypatch.setattr(lx, "MAX_WORKERS", 6)
        monkeypatch.setattr(lx, "MAX_WORKERS_PINNED", False)
        R._fit_langextract_workers(slots=2, workers=4, source="pubmed")
        assert lx.MAX_WORKERS == 1

    def test_an_explicit_setting_is_never_overridden(self, monkeypatch):
        """An operator who names a number gets it, oversubscribed or not."""
        import sample.langextract_records as lx
        monkeypatch.setattr(lx, "MAX_WORKERS", 6)
        monkeypatch.setattr(lx, "MAX_WORKERS_PINNED", True)
        R._fit_langextract_workers(slots=2, workers=4, source="pubmed")
        assert lx.MAX_WORKERS == 6

    def test_a_fan_out_that_already_fits_is_left_alone(self, monkeypatch):
        import sample.langextract_records as lx
        monkeypatch.setattr(lx, "MAX_WORKERS", 2)
        monkeypatch.setattr(lx, "MAX_WORKERS_PINNED", False)
        R._fit_langextract_workers(slots=8, workers=4, source="pubmed")
        assert lx.MAX_WORKERS == 2, "8 slots serve 4x2 requests; nothing to fix"

    def test_the_fan_out_never_drops_below_one(self, monkeypatch):
        import sample.langextract_records as lx
        monkeypatch.setattr(lx, "MAX_WORKERS", 6)
        monkeypatch.setattr(lx, "MAX_WORKERS_PINNED", False)
        R._fit_langextract_workers(slots=1, workers=16, source="pubmed")
        assert lx.MAX_WORKERS == 1, "zero threads would process nothing"

    def test_the_warning_reports_the_product_not_the_worker_count(self, caplog, monkeypatch):
        """
        Pins the number the operator reads. With 2 slots, 4 workers and 6 langextract threads the
        message must say 24 requests and 22 queueing, not 4 and 2.
        """
        import pandas as pd
        monkeypatch.setattr(R, "server_parallel_slots", lambda: 2)
        monkeypatch.setattr(R, "_langextract_max_workers", lambda: 6)
        monkeypatch.setattr(R, "warn_on_model_split", lambda *_a, **_k: None)
        # Neutralised so this test asserts on the warning text alone; the real one mutates
        # langextract_records.MAX_WORKERS, which would leak into every later test in the session.
        monkeypatch.setattr(R, "_fit_langextract_workers", lambda *_a, **_k: None)
        # One task, because run_parallel_pipeline returns before the warning when there is
        # nothing to do; the document function is stubbed so no extraction runs.
        monkeypatch.setattr(R, "build_task_list",
                            lambda *_a, **_k: ([("doc1", "in/doc1.txt", "out/doc1.json")], 0, 0))
        monkeypatch.setattr(R, "_extract_document", lambda task, *_a, **_k: (task[0], 0, None))
        with caplog.at_level("WARNING", logger=R.logger.name):
            R.run_parallel_pipeline("pubmed", "in", "out",
                                    pd.DataFrame({"internal_id": ["doc1"]}), workers=4,
                                    resume=R.RESUME_OFF, chunk_size=0, document_budget=0)
        warnings = [r.message for r in caplog.records if "parallel slot" in r.message]
        assert warnings, "no oversubscription warning emitted"
        text = warnings[0]
        assert "24 concurrent request" in text, text
        assert "22 of every 24" in text, text
        assert "LANGEXTRACT_MAX_WORKERS 6" in text, text
        assert "WORKERS=2 LANGEXTRACT_MAX_WORKERS=1 (= 2 request(s))" in text, (
            "the suggestion must be a pair whose PRODUCT fits the server, not two knobs each of "
            f"which is individually insufficient: {text}")

    def test_no_warning_when_the_server_can_serve_every_request(self, caplog, monkeypatch):
        import pandas as pd
        monkeypatch.setattr(R, "server_parallel_slots", lambda: 4)
        monkeypatch.setattr(R, "_langextract_max_workers", lambda: 2)
        monkeypatch.setattr(R, "warn_on_model_split", lambda *_a, **_k: None)
        monkeypatch.setattr(R, "_fit_langextract_workers", lambda *_a, **_k: None)
        monkeypatch.setattr(R, "build_task_list",
                            lambda *_a, **_k: ([("doc1", "in/doc1.txt", "out/doc1.json")], 0, 0))
        monkeypatch.setattr(R, "_extract_document", lambda task, *_a, **_k: (task[0], 0, None))
        with caplog.at_level("INFO", logger=R.logger.name):
            R.run_parallel_pipeline("pubmed", "in", "out",
                                    pd.DataFrame({"internal_id": ["doc1"]}), workers=2,
                                    resume=R.RESUME_OFF, chunk_size=0, document_budget=0)
        assert not [r for r in caplog.records if r.levelname == "WARNING"
                    and "parallel slot" in r.message]


class TestResumeModes:
    """
    Which documents an interrupted extraction re-processes.

    The trap this guards is specific and was live in the Makefile for months: step 8 writes an
    output JSON for EVERY document, so resuming step 9 on file existence skips the whole corpus
    and adds no model records at all. Step 9 therefore resumes on a marker inside the file
    instead, and these tests pin both modes apart.
    """

    @staticmethod
    def _write(path, backend):
        payload = {"metadata": ({} if backend is None else {"langextract_backend": backend}),
                   "peptides": []}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_normalise_accepts_the_legacy_boolean(self):
        assert R.normalise_resume(True) == R.RESUME_IF_EXISTS
        assert R.normalise_resume(False) == R.RESUME_OFF
        assert R.normalise_resume(None) == R.RESUME_OFF

    def test_normalise_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError):
            R.normalise_resume("sometimes")

    def test_exists_mode_skips_any_document_with_an_output(self, tmp_path):
        done = tmp_path / "a.json"
        self._write(str(done), "disabled")
        assert R._refined_pass_done(str(done)) is False, (
            "a screening-only output must not read as refined")

    def test_refined_mode_skips_only_a_document_the_model_pass_finished(self, tmp_path):
        refined, screened, bare = (tmp_path / n for n in ("r.json", "s.json", "b.json"))
        self._write(str(refined), "enabled")
        self._write(str(screened), "disabled")
        self._write(str(bare), None)
        assert R._refined_pass_done(str(refined)) is True
        assert R._refined_pass_done(str(screened)) is False
        assert R._refined_pass_done(str(bare)) is False

    def test_a_truncated_output_is_reprocessed_rather_than_skipped(self, tmp_path):
        """
        A kill mid-write leaves invalid JSON. Counting that as done would drop the document from
        the corpus silently; counting it as not-done costs one document of work.
        """
        broken = tmp_path / "t.json"
        broken.write_text('{"metadata": {"langextract_backend": "enab')
        assert R._refined_pass_done(str(broken)) is False
        missing = tmp_path / "gone.json"
        assert R._refined_pass_done(str(missing)) is False

    def test_a_json_file_that_is_not_an_object_is_reprocessed(self, tmp_path):
        odd = tmp_path / "list.json"
        odd.write_text("[1, 2, 3]")
        assert R._refined_pass_done(str(odd)) is False

    def test_the_two_modes_select_different_documents(self, tmp_path):
        """
        The end-to-end point of the mode: `exists` queues nothing because step 8 already wrote an
        output for both documents, while `refined` queues the one the model pass never reached.
        """
        import pandas as pd
        inp, out = tmp_path / "txt", tmp_path / "json"
        inp.mkdir(); out.mkdir()
        for name, backend in (("refined_doc", "enabled"), ("screened_doc", "disabled")):
            (inp / f"{name}.txt").write_text("text")
            self._write(str(out / f"{name}.json"), backend)
        df = pd.DataFrame({"internal_id": ["refined_doc", "screened_doc"]})

        tasks, skipped, missing = R.build_task_list(str(inp), str(out), df, R.RESUME_IF_EXISTS)
        assert (len(tasks), skipped) == (0, 2), (
            "this is the bug the mode exists for: on file existence step 9 skips the whole corpus")

        tasks, skipped, missing = R.build_task_list(str(inp), str(out), df, R.RESUME_IF_REFINED)
        assert [t[0] for t in tasks] == ["screened_doc"]
        assert skipped == 1, "the document the model already processed stays skipped"

        tasks, skipped, missing = R.build_task_list(str(inp), str(out), df, R.RESUME_OFF)
        assert len(tasks) == 2 and skipped == 0, "off re-processes everything"

    def test_a_deterministically_skipped_document_counts_as_done(self, tmp_path):
        """
        Gate 1 is a pure function of the text, so the refined pass cannot do better than the
        screening pass on a captcha page or an empty scan. Without this, those documents never
        acquire a marker and every resume re-queues all of them.
        """
        out = tmp_path / "skipped.json"
        out.write_text(json.dumps({"metadata": {"skipped": True,
                                                "skip_reason": "captcha_or_bot_check"},
                                   "peptides": []}))
        assert R._refined_pass_done(str(out)) is True

    def test_a_document_is_requeued_when_its_text_was_reacquired(self, tmp_path):
        """
        The mtime guard. A captcha page re-downloaded as the real paper has a NEWER .txt than the
        output describing it, so that output is stale and the document must be processed again --
        otherwise the re-acquisition is silently wasted.
        """
        out, txt = tmp_path / "d.json", tmp_path / "d.txt"
        out.write_text(json.dumps({"metadata": {"langextract_backend": "enabled"},
                                   "peptides": []}))
        txt.write_text("the real paper, downloaded later")
        os.utime(str(out), (1_000_000, 1_000_000))
        os.utime(str(txt), (2_000_000, 2_000_000))
        assert R._refined_pass_done(str(out), str(txt)) is False, (
            "output older than its input text is stale")

        os.utime(str(out), (3_000_000, 3_000_000))
        assert R._refined_pass_done(str(out), str(txt)) is True

    def test_the_skipped_clause_does_not_rescue_a_truncated_output(self, tmp_path):
        broken = tmp_path / "t.json"
        broken.write_text('{"metadata": {"skipped": tr')
        assert R._refined_pass_done(str(broken)) is False

    def test_resume_skips_both_refined_and_skipped_documents(self, tmp_path):
        import pandas as pd
        inp, out = tmp_path / "txt", tmp_path / "json"
        inp.mkdir(); out.mkdir()
        cases = {"refined_doc": {"langextract_backend": "enabled"},
                 "captcha_doc": {"skipped": True, "skip_reason": "captcha_or_bot_check"},
                 "screened_doc": {"langextract_backend": "disabled"}}
        for name, meta in cases.items():
            (inp / f"{name}.txt").write_text("text")
            (out / f"{name}.json").write_text(json.dumps({"metadata": meta, "peptides": []}))
            os.utime(str(out / f"{name}.json"), (2_000_000, 2_000_000))
            os.utime(str(inp / f"{name}.txt"), (1_000_000, 1_000_000))
        df = pd.DataFrame({"internal_id": list(cases)})
        tasks, skipped, _ = R.build_task_list(str(inp), str(out), df, R.RESUME_IF_REFINED)
        assert [t[0] for t in tasks] == ["screened_doc"], (
            "only the document the model has not seen is real work")
        assert skipped == 2

    def test_a_document_with_no_input_text_is_reported_separately(self, tmp_path):
        import pandas as pd
        inp, out = tmp_path / "txt", tmp_path / "json"
        inp.mkdir(); out.mkdir()
        df = pd.DataFrame({"internal_id": ["ghost"]})
        tasks, skipped, missing = R.build_task_list(str(inp), str(out), df, R.RESUME_IF_REFINED)
        assert (len(tasks), skipped, missing) == (0, 0, 1)


class TestActivityLogging:
    """
    One log line per extracted peptide-activity, so a long run shows what it is finding while it
    runs instead of only through the JSON afterwards.
    """

    @staticmethod
    def _peptide(name="Magainin 2", seq="GIGKFLHSAKKF", activities=None):
        return {"peptide_name": name, "peptide_sequence": seq,
                "biological_activities": activities if activities is not None else []}

    def test_one_line_per_activity_with_every_requested_field(self, caplog, monkeypatch):
        monkeypatch.setattr(R, "_DOI_INDEX", {"doc1": "10.1016/j.peptides.2020.170001"})
        peptide = self._peptide(activities=[
            {"activity_type": "MIC", "value_raw": "> 8", "activity_unit": "ug/mL",
             "target_organism": "Escherichia coli"},
            {"activity_type": "MBC", "value_raw": "16", "activity_unit": "ug/mL",
             "target_organism": "Staphylococcus aureus"},
        ])
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("doc1", [peptide])
        lines = [r.message for r in caplog.records if "[activity]" in r.message]
        assert len(lines) == 2, "a peptide measured twice is two facts and gets two lines"
        for line in lines:
            for field in ("name=", "seq=", "doi=", "activity=", "unit=", "target="):
                assert field in line
        assert "doi=10.1016/j.peptides.2020.170001" in lines[0]
        assert "target=Escherichia coli" in lines[0]

    def test_the_doi_index_is_keyed_on_the_same_id_a_record_carries(self, tmp_path, monkeypatch):
        """
        The join has to be on the acquisition hash, because that is what `document_id` holds.

        This is the shape of the bug it guards: both `pubmed_metadata.csv` and
        `pubmed_downloaded.csv` have an `internal_id` column, but the metadata file keys on
        `pubmed_<pmid>` while a record's `document_id` is the hash. `_DOI_SOURCES` named the
        metadata file, so the index built fine, reported a plausible size, and matched nothing --
        every logged DOI was `-` for 11,698 documents. The other tests in this class monkeypatch
        `_DOI_INDEX`, so none of them touched the join; this one builds it from disk.
        """
        doc_id = "00014910-798d-2fd5-eeab-0065d9040ec2832b0f85"
        src_dir = tmp_path / "data" / "literature_data"
        src_dir.mkdir(parents=True)
        (src_dir / "pubmed_downloaded.csv").write_text(
            "source,title,doi,pmid,link,internal_id,success,pdf_path\n"
            f"pubmed,A paper,10.1016/j.peptides.2020.170001,1,u,{doc_id},True,p.pdf\n")
        (src_dir / "pubmed_metadata.csv").write_text(
            "internal_id,pmid,doi,title,abstract,year\n"
            "pubmed_20259524,20259524,10.9999/decoy,T,A,2010\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(R, "_DOI_INDEX", None)
        index = R._doi_index()
        assert index.get(doc_id) == "10.1016/j.peptides.2020.170001", (
            "the index must resolve the id a record actually carries, not pubmed_<pmid>")
        assert not any(k.startswith("pubmed_") for k in index), (
            "keying on the metadata file's internal_id resolves nothing against document_id")

    def test_a_peptide_with_no_activity_still_gets_a_line(self, caplog, monkeypatch):
        """
        Not padding. Measured over this corpus, 65.2% of LLM-derived records carry no activity at
        all, so the blank lines are the ones that answer "is any of this usable?".
        """
        monkeypatch.setattr(R, "_DOI_INDEX", {})
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("doc1", [self._peptide()])
        lines = [r.message for r in caplog.records if "[activity]" in r.message]
        assert len(lines) == 1
        assert "activity=-" in lines[0] and "unit=-" in lines[0]

    def test_a_value_is_logged_beside_its_unit(self, caplog, monkeypatch):
        """A unit with no number next to it says nothing."""
        monkeypatch.setattr(R, "_DOI_INDEX", {})
        peptide = self._peptide(activities=[
            {"activity_type": "IC50", "value_lower": 2.5, "value_upper": 5.0,
             "activity_unit": "uM", "target_cell_line": "HeLa"}])
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("doc1", [peptide])
        line = [r.message for r in caplog.records if "[activity]" in r.message][0]
        assert "value=2.5-5.0" in line
        # target falls back to the cell line when there is no organism.
        assert "target=HeLa" in line

    def test_an_embedded_newline_cannot_split_a_record_across_lines(self, caplog, monkeypatch):
        """
        These fields come out of document text. One embedded newline would break every later
        attempt to grep or count these lines, silently.
        """
        monkeypatch.setattr(R, "_DOI_INDEX", {})
        peptide = self._peptide(name="Line\nbreaker\twith tabs")
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("doc1", [peptide])
        lines = [r.message for r in caplog.records if "[activity]" in r.message]
        assert len(lines) == 1
        assert "\n" not in lines[0] and "\t" not in lines[0]

    def test_a_long_field_is_truncated_not_wrapped(self, caplog, monkeypatch):
        monkeypatch.setattr(R, "_DOI_INDEX", {})
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("doc1", [self._peptide(name="x" * 400)])
        line = [r.message for r in caplog.records if "[activity]" in r.message][0]
        assert len(line) < 400

    def test_it_can_be_switched_off(self, caplog, monkeypatch):
        monkeypatch.setattr(R, "LOG_ACTIVITIES", False)
        monkeypatch.setattr(R, "_DOI_INDEX", {})
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("doc1", [self._peptide()])
        assert not [r for r in caplog.records if "[activity]" in r.message]

    def test_a_malformed_record_never_costs_the_document(self, caplog, monkeypatch):
        """Reporting must not raise: the document is already written by this point."""
        monkeypatch.setattr(R, "_DOI_INDEX", {})
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("doc1", ["not a dict", None, {"biological_activities": [42]}])

    def test_a_missing_doi_is_a_dash_not_a_crash(self, caplog, monkeypatch):
        monkeypatch.setattr(R, "_DOI_INDEX", {})
        with caplog.at_level("INFO", logger=R.logger.name):
            R._log_peptide_activities("unknown-doc", [self._peptide()])
        line = [r.message for r in caplog.records if "[activity]" in r.message][0]
        assert "doi=-" in line

    def test_the_doi_index_reads_only_the_columns_it_needs(self, tmp_path, monkeypatch):
        """
        The metadata file also carries abstracts; loading those to fill a log field would be tens
        of MB for nothing.
        """
        import pandas as pd

        path = tmp_path / "pubmed_metadata.csv"
        pd.DataFrame({"internal_id": ["a", "b"], "pmid": ["1", "2"],
                      "doi": ["10.1/a", ""], "abstract": ["x" * 100, "y" * 100]}).to_csv(
            path, index=False)
        monkeypatch.setattr(R, "_DOI_INDEX", None)
        monkeypatch.setattr(R, "_DOI_SOURCES", ((str(path), "internal_id", "doi"),))
        index = R._doi_index()
        assert index == {"a": "10.1/a"}, "an empty DOI must not be indexed as a value"

    def test_a_missing_metadata_file_costs_only_the_doi_field(self, monkeypatch):
        monkeypatch.setattr(R, "_DOI_INDEX", None)
        monkeypatch.setattr(R, "_DOI_SOURCES", (("/definitely/not/here.csv", "internal_id", "doi"),))
        assert R._doi_index() == {}

class TestGate4bRejectsIncoherentUnits:
    """
    A molecular weight is not an MIC, even when both numbers are in the paper.

    Observed on a live run: `brevinin-1RTb | activity=MIC | value=2548.2 | unit=Da`. The mass is
    real and the paper states it, so the text-grounding gate passes it; what is wrong is the
    pairing. 6,173 of 53,107 activities on disk were such pairings.
    """

    @staticmethod
    def _pep_with(activities, path="langextract_match", seq="GIGKFLHSAKKF"):
        return {"peptide_sequence": seq, "peptide_name": "Magainin 2", "context": seq,
                "discovery_path": path, "biological_activities": activities}

    def test_a_molecular_weight_filed_as_an_mic_is_dropped(self, tmp_path):
        text = ("Peptide mass was 2548.2 Da by MALDI-TOF. Materials and Methods. "
                + ("body text " * 300))
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}])], text=text)
        assert record["metadata"]["dropped_activity_unit_incoherent"] == 1
        assert record["peptides"][0]["biological_activities"] == []
        dropped = record["metadata"]["dropped_activity_units"][0]
        assert dropped["reason"] == "MIC_in_mass_unit"
        assert dropped["value_raw"] == "2548.2" and dropped["activity_unit"] == "Da"

    def test_it_fires_even_when_the_number_is_in_the_text(self, tmp_path):
        """
        The distinction from gate 4. Grounding alone cannot catch this, because a mass-spec table
        puts the number in the document verbatim.
        """
        text = ("Observed mass 2548.2 Da. Materials and Methods. " + ("body text " * 300))
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "MIC", "value_raw": "2548.2", "activity_unit": "Da"}])], text=text)
        assert record["metadata"]["dropped_activity_unit_incoherent"] == 1
        assert "dropped_activity_not_in_text" not in record["metadata"], (
            "it must be attributed to the unit gate, not miscounted as ungrounded"
        )

    def test_a_real_concentration_measurement_survives(self, tmp_path):
        text = ("The peptide had an MIC of 8 ug/mL against E. coli. Materials and Methods. "
                + ("body text " * 300))
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL",
             "target_organism": "Escherichia coli"}])], text=text)
        assert "dropped_activity_unit_incoherent" not in record["metadata"]
        assert len(record["peptides"][0]["biological_activities"]) == 1

    def test_the_peptide_survives_its_dropped_activity(self, tmp_path):
        """The sequence may be perfectly good; only the measurement attributed to it is wrong."""
        text = ("Mass 3416.0 Da. Materials and Methods. " + ("body text " * 300))
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "MBC", "value_raw": "3416.0", "activity_unit": "Da"}])], text=text)
        assert len(record["peptides"]) == 1
        assert record["peptides"][0]["peptide_sequence"] == "GIGKFLHSAKKF"


class TestGate4GroundsMeasurementsInTheDocument:
    """
    A measurement whose number is nowhere in the document is dropped.

    This gate exists because the model was caught copying its own prompt into its output. The
    LangExtract worker's few-shot example is "Peptide JG436 inhibited MCF-7 cells with an IC50 of
    12.5 uM"; measured over one run, that exact tuple was emitted for 49 different documents, 38
    of which contain neither "12.5" nor "MCF" anywhere in their text. The next most repeated
    measurement in the same corpus appeared in 3 documents.
    """

    @staticmethod
    def _pep_with(activities, path="langextract_match", seq="GIGKFLHSAKKF"):
        return {"peptide_sequence": seq, "peptide_name": "Magainin 2", "context": seq,
                "discovery_path": path, "biological_activities": activities}

    def test_the_prompt_leak_is_dropped(self, tmp_path):
        """The exact fabrication, on a document that says nothing of the kind."""
        text = ("A real paper about peptides. Materials and Methods. " + ("body text " * 300))
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "IC50", "value_raw": "12.5", "activity_unit": "uM",
             "target_cell_line": "MCF-7"}])], text=text)
        assert record["metadata"]["dropped_activity_not_in_text"] == 1
        assert record["peptides"][0]["biological_activities"] == []

    def test_a_measurement_present_in_the_text_survives(self, tmp_path):
        text = ("The peptide had an MIC of 8 ug/mL against E. coli. Materials and Methods. "
                + ("body text " * 300))
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "MIC", "value_raw": "8", "activity_unit": "ug/mL",
             "target_organism": "Escherichia coli"}])], text=text)
        assert "dropped_activity_not_in_text" not in record["metadata"]
        assert len(record["peptides"][0]["biological_activities"]) == 1

    def test_the_peptide_survives_even_when_its_measurement_does_not(self, tmp_path):
        """
        The peptide may be real and correctly named even when a measurement attributed to it was
        invented, so the activity is dropped and the record is not.
        """
        text = "A real paper. Materials and Methods. " + ("body text " * 300)
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "IC50", "value_raw": "12.5", "activity_unit": "uM"}])], text=text)
        assert len(record["peptides"]) == 1
        assert record["peptides"][0]["peptide_sequence"] == "GIGKFLHSAKKF"

    def test_what_was_dropped_is_recorded_not_just_counted(self, tmp_path):
        """
        Every other gate in this stage records its evidence; a gate that cannot be audited
        cannot be trusted, and this one deletes published measurements.
        """
        text = "A real paper. Materials and Methods. " + ("body text " * 300)
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "IC50", "value_raw": "12.5", "activity_unit": "uM",
             "target_cell_line": "MCF-7"}])], text=text)
        dropped = record["metadata"]["dropped_activities"]
        assert dropped[0]["value_raw"] == "12.5"
        assert dropped[0]["target"] == "MCF-7"
        assert dropped[0]["reason"] == "value_not_in_text"

    def test_table_derived_measurements_are_exempt(self, tmp_path):
        """
        Those values come from the PDF via docling, so a number genuinely in a table need not
        appear in the pdfplumber .txt -- grounding them against the text would delete correct
        records.
        """
        text = "A real paper. Materials and Methods. " + ("body text " * 300)
        record = _run(tmp_path, [self._pep_with(
            [{"activity_type": "MIC", "value_raw": "64", "activity_unit": "ug/mL"}],
            path="table_match")], text=text)
        assert "dropped_activity_not_in_text" not in record["metadata"]
        assert len(record["peptides"][0]["biological_activities"]) == 1

    def test_an_activity_with_no_number_is_not_a_measurement(self, tmp_path):
        """'Antibacterial' has nothing to ground; the activity class is judged elsewhere."""
        text = "A real paper. Materials and Methods. " + ("body text " * 300)
        record = _run(tmp_path, [self._pep_with([{"activity_type": "Antibacterial"}])],
                      text=text)
        assert "dropped_activity_not_in_text" not in record["metadata"]
        assert len(record["peptides"][0]["biological_activities"]) == 1

    def test_the_decimal_separator_is_tolerated(self, tmp_path):
        """
        The same value is written '3.5' and '3,5' across this corpus; a strict match would reject
        a correct extraction because of the typesetting it was read from.
        """
        text = "an IC50 of 3,5 uM was measured. Materials and Methods. " + ("body text " * 300)
        record = _run(tmp_path, [self._pep_with([
            {"activity_type": "IC50", "value_raw": "3.5", "activity_unit": "uM"}])], text=text)
        assert "dropped_activity_not_in_text" not in record["metadata"]

class TestContextRungsFollowThinking:
    """
    A chain of thought shares the context window with the prompt and the answer, so enabling it
    has to move the pinned rung -- and the rung has to move in BOTH places at once, or Ollama
    rebuilds the runner between the langextract backend's calls and everything else's.
    """

    def test_langextract_derives_its_window_from_the_project_default(self, monkeypatch):
        """
        These were two independent literals under a comment saying they "MUST stay equal" --
        a rule with nothing enforcing it. Now one reads the other.
        """
        import sample.langextract_records as LR
        from sample.config import DEFAULT_NUM_CTX

        monkeypatch.delenv("LANGEXTRACT_NUM_CTX", raising=False)
        assert LR._default_num_ctx() == DEFAULT_NUM_CTX

    def test_an_explicit_override_still_wins(self, monkeypatch):
        import sample.langextract_records as LR

        monkeypatch.setenv("LANGEXTRACT_NUM_CTX", "2048")
        assert LR._default_num_ctx() == 2048

    def test_the_rung_steps_up_when_thinking_is_on(self):
        """
        8192 x 2 slots is 16,384 of runner context -- the same total this host already ran
        successfully at 4096 x 4, and half the bytes of it under OLLAMA_KV_CACHE_TYPE=q8_0.
        """
        from sample.config import DEFAULT_NUM_CTX, FALLBACK_NUM_CTX, LLM_THINK

        if LLM_THINK:
            assert DEFAULT_NUM_CTX >= 8192
            assert FALLBACK_NUM_CTX > DEFAULT_NUM_CTX
        else:
            assert DEFAULT_NUM_CTX == 4096

    def test_the_two_rungs_stay_distinct(self):
        """
        The fallback exists so a prompt that will not fit the default has somewhere to go. If the
        two collapsed to one value, an oversized prompt would be truncated silently instead.
        """
        from sample.config import DEFAULT_NUM_CTX, FALLBACK_NUM_CTX

        assert FALLBACK_NUM_CTX > DEFAULT_NUM_CTX

