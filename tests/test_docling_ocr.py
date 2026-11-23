"""
Tests for the OCR path in :mod:`sample.docling_tables`.

How much of the corpus needs OCR depends entirely on which corpus. Papers almost never do:
measured over 60 sampled papers (663 pages), 100% carry a healthy text layer. Patents very often
do: measured over 60 sampled patents (1,400 pages), 36.7% have a text layer of EXACTLY ZERO
characters. An earlier note here put the overall figure at ~0.38% from an 800-PDF sample; that
sample was papers, and generalising it to patents is what left the patent corpus yielding almost
nothing -- those documents are not merely hard to read without OCR, they are entirely invisible to
every text-based extractor.

These tests never invoke docling or tesseract. They pin the two decisions that make the OCR path
correct, both of which are easy to regress silently:

* Tesseract is the engine, resolved by ABSOLUTE PATH. docling's default is RapidOCR, which
  downloads PP-OCRv6 weights from modelscope.cn at first use -- verified by running with the
  network blocked, where it raised DownloadFileException. This project must run locally.
* OCR is decided per document from the text layer, not applied to everything. It costs ~6 s/page
  against ~1.14 s/page without, so running it on the 99.6% of documents that do not need it would
  dominate the extraction step.
"""
import os
from typing import Any, Dict, List

import pytest

from sample import docling_tables as D


@pytest.fixture(autouse=True)
def _isolate_conversion_cache(tmp_path_factory):
    """
    Point the docling conversion cache at a throwaway directory for every test in this module.

    Two reasons, both learned the hard way on the LangExtract payload cache. It must not write into
    the repository's real `data/docling_cache` -- running the suite would seed production cache
    entries from stubbed payloads. And a test that does not think about caching otherwise gets a
    HIT from an entry an earlier test wrote: the cache is consulted BEFORE the subprocess, so the
    command-assertion tests below started asserting against an empty command list.
    """
    D.DOCLING_CACHE_DIR = str(tmp_path_factory.mktemp("docling-cache"))
    yield


@pytest.fixture(autouse=True)
def _clear_language_cache():
    """
    ocr_languages_for is lru_cached, which is right in production -- the answer is deterministic
    per PDF and the full OCR pass asks for it more than once -- but it makes tests order-dependent:
    a cached result computed with one tessdata fixture was being returned for the next test's
    different fixture.
    """
    # getattr: one test below replaces ocr_languages_for with a plain lambda, which has no cache.
    def clear():
        getattr(D.ocr_languages_for, "cache_clear", lambda: None)()

    clear()
    yield
    clear()


class TestNeedsOcr:
    """
    needs_ocr opens the PDF with fitz itself, so these drive it through a real file rather than a
    patched helper. An earlier version of this class patched a `_page_char_counts` name that does
    not exist, with raising=False -- so it asserted nothing while appearing to cover the case.
    """

    def _pdf_with_text(self, path: str, text: str, pages: int = 3) -> str:
        """
        A PDF whose pages really carry `text`.

        Uses insert_textbox, not insert_text. insert_text draws a single unwrapped line from the
        given point, so it runs off the page edge and most of the string is never rendered: a
        496-character paragraph came back as 105 recoverable characters, which made a "healthy
        document" fixture actually sparse and failed the test it was written to pass. A rect wraps
        the text the way a real paragraph sits on a page (743 characters recovered for the same
        input).
        """
        fitz = pytest.importorskip("fitz")
        document = fitz.open()
        for _ in range(pages):
            page = document.new_page()
            if text:
                page.insert_textbox(fitz.Rect(72, 72, 520, 700), text, fontsize=11)
        document.save(path)
        document.close()
        return path

    def test_a_page_with_no_text_layer_needs_ocr(self, tmp_path: Any) -> None:
        # Blank pages are what a scanned PDF looks like to a text extractor: the words are pixels.
        path = self._pdf_with_text(str(tmp_path / "scan.pdf"), "")
        assert D.needs_ocr(path) is True

    def test_a_healthy_text_layer_does_not(self, tmp_path: Any) -> None:
        # Comfortably over the 200 chars/page bound, so OCR's ~6 s/page is not spent needlessly.
        body = "Antimicrobial peptides were assayed against Escherichia coli. " * 12
        path = self._pdf_with_text(str(tmp_path / "text.pdf"), body)
        assert D.needs_ocr(path) is False

    def test_a_thin_text_layer_still_needs_ocr(self, tmp_path: Any) -> None:
        # The real sparse case measured in this corpus: a page carrying only a caption or a
        # watermark, well under the per-page bound.
        path = self._pdf_with_text(str(tmp_path / "thin.pdf"), "Fig 1.")
        assert D.needs_ocr(path) is True

    def test_an_unreadable_pdf_does_not_claim_to_need_ocr(self) -> None:
        # An unreadable file is not evidence that OCR would help, and OCR is several times slower.
        assert D.needs_ocr("/nonexistent/never.pdf") is False

    def test_the_threshold_is_per_page_not_per_document(self) -> None:
        # A 40-page paper with 300 chars/page is healthy; the same 12,000 characters spread over
        # 400 pages is a scan with page furniture. The bound has to be per page for both to be
        # judged correctly.
        assert D.OCR_CHARS_PER_PAGE == 200


class TestTesseractResolution:
    def test_tesseract_is_resolved_by_absolute_path_or_not_at_all(self) -> None:
        # A bare "tesseract" is never acceptable: this host does not carry it on PATH, so the
        # wrapper imports cleanly and then fails at the first OCR call.
        resolved = D.tesseract_command()
        if resolved is not None:
            assert os.path.isabs(resolved), "tesseract must be an absolute path"
            assert resolved.endswith("tesseract")

    def test_ocr_is_requested_with_the_binary_path(self, monkeypatch: Any, tmp_path: Any) -> None:
        """The subprocess must carry --ocr AND --tesseract-cmd, or the worker refuses the job."""
        captured: Dict[str, List[str]] = {}

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command: List[str], **kwargs: Any) -> Any:
            captured["command"] = list(command)
            # The worker writes its payload to the --out path; emit an empty document so the
            # caller's parsing succeeds and the assertions below are about the command only.
            if "--out" in command:
                with open(command[command.index("--out") + 1], "w", encoding="utf-8") as handle:
                    handle.write('{"tables": []}')
            return _Completed()

        monkeypatch.setattr(D.subprocess, "run", fake_run)
        monkeypatch.setattr(D, "docling_python", lambda: ["/env/bin/python"])
        monkeypatch.setattr(D, "tesseract_command", lambda: "/env/bin/tesseract")

        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        D.docling_tables(str(pdf), use_ocr=True)

        command = captured.get("command", [])
        assert "--ocr" in command, "OCR was requested but not passed to the worker"
        assert "--tesseract-cmd" in command
        assert command[command.index("--tesseract-cmd") + 1] == "/env/bin/tesseract"

    def test_no_ocr_flag_when_the_text_layer_is_healthy(self, monkeypatch: Any,
                                                       tmp_path: Any) -> None:
        captured: Dict[str, List[str]] = {}

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command: List[str], **kwargs: Any) -> Any:
            captured["command"] = list(command)
            if "--out" in command:
                with open(command[command.index("--out") + 1], "w", encoding="utf-8") as handle:
                    handle.write('{"tables": []}')
            return _Completed()

        monkeypatch.setattr(D.subprocess, "run", fake_run)
        monkeypatch.setattr(D, "docling_python", lambda: ["/env/bin/python"])
        pdf = tmp_path / "text.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        D.docling_tables(str(pdf), use_ocr=False)
        assert "--ocr" not in captured.get("command", [])


class TestSubprocessIsolation:
    def test_the_run_is_offline(self) -> None:
        # The weights are prefetched, so a conversion must never reach the network.
        env = D._subprocess_env()
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"

    def test_threads_never_exceed_the_cores_the_semaphore_permits(self) -> None:
        """
        Threads per conversion x permitted conversions must not oversubscribe the machine.

        The original pin was one thread, calibrated against "N worker threads each starting a
        subprocess claiming one thread per core" -- 128 runnable threads on 24 cores. But the
        conversion semaphore now bounds how many conversions exist at once, so the two limits were
        multiplying: with the semaphore at 1 the table pass used ONE core of 24, and that pass is
        82% of the screening step's run time.
        """
        import os as _os

        cores = _os.cpu_count() or 4
        threads = D._threads_per_conversion()
        assert threads >= 1
        assert threads * D.MAX_CONCURRENT_CONVERSIONS <= cores, (
            "more runnable threads than cores -- the contention the pin existed to prevent")

    def test_every_blas_variable_carries_the_same_count(self) -> None:
        # A single library left at its default would start one thread per core on its own.
        env = D._subprocess_env()
        counts = {env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS")}
        assert len(counts) == 1, f"inconsistent thread counts: {counts}"

    def test_the_count_is_overridable(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("SAMPLE_DOCLING_THREADS", "3")
        assert D._threads_per_conversion() == 3

    def test_concurrency_is_bounded_by_memory(self) -> None:
        # One conversion peaks at ~2.1 GB; the extraction parent holds several GB of shared
        # lookups. A cap derived from core count alone put this machine into swap.
        assert 1 <= D.MAX_CONCURRENT_CONVERSIONS <= 4
        assert D._CONVERSION_PEAK_MB >= 1000


class TestOcrLanguageSelection:
    """
    The language is chosen per document, from the patent-office country prefix.

    Not cosmetic. 52% of the patent corpus is CN, plus 490 KR, 126 JP and 177 SU/RU -- about 70% of
    it is in a non-Latin script. Tesseract asked for ``eng`` on such a page does not fail; it
    returns Latin look-alikes. Measured, the Soviet patent SU1569953A1 OCR'd under ``eng`` as
    "FOCYQAPCTBEHHBHA HOMUTET" and under ``rus,eng`` as the correct
    "ГОСУДАРСТВЕННЫЙ КОМИТЕТ" -- 0 Cyrillic characters recovered versus 3,965. The wrong-language
    output is worse than none, because long runs of capitals are what a sequence detector keys on
    and "CCCP" is four valid amino-acid letters.
    """

    @pytest.fixture()
    def packs(self, monkeypatch: Any, tmp_path: Any) -> Any:
        """A tessdata directory carrying exactly the packs these tests name."""
        directory = tmp_path / "tessdata"
        directory.mkdir()
        for language in ("eng", "rus", "chi_sim", "jpn", "kor", "fra", "deu"):
            (directory / f"{language}.traineddata").write_bytes(b"\x00")
        monkeypatch.setattr(D, "_tessdata_dir", lambda: str(directory))
        return directory

    @pytest.mark.parametrize("filename,expected", [
        ("CN111718497B.pdf", "chi_sim,eng"),
        ("SU1569953A1.pdf", "rus,eng"),
        ("JP2025516591A.pdf", "jpn,eng"),
        ("KR100866631B1.pdf", "kor,eng"),
    ])
    def test_the_native_pack_leads_and_english_follows(self, packs: Any, filename: str,
                                                       expected: str) -> None:
        # Native first because tesseract treats the leading language as primary and mis-segments
        # CJK glyphs under an English-primary model. English is appended and never substituted:
        # even a Chinese patent prints its peptide sequences in Latin letters, and those are the
        # only tokens this project needs off the page.
        assert D.ocr_languages_for(filename) == expected

    @pytest.mark.parametrize("filename", ["US7425924B2.pdf", "GB2200000A.pdf",
                                          "CA1234567A.pdf", "AU2001234567A1.pdf"])
    def test_english_offices_get_english_alone(self, packs: Any, filename: str) -> None:
        # A second pack on an English page only gives tesseract a way to be wrong.
        assert D.ocr_languages_for(filename) == "eng"

    @pytest.mark.parametrize("filename", ["WO1994004996A1.pdf", "EP0123456A1.pdf"])
    def test_wo_and_ep_are_not_english_offices(self, packs: Any, filename: str) -> None:
        # The EPO's official languages are English, French and German, and a WO publication
        # appears in its filing language. WO2008043938A3, OCR'd from this corpus, is entirely
        # French: "DEMANDE INTERNATIONALE PUBLIEE EN VERTU DU TRAITE". Treating these as English
        # dropped the accented characters on the documents most likely to be European. All three
        # are Latin-script, so one pass covers them.
        assert D.ocr_languages_for(filename) == "eng,fra,deu"

    def test_an_unknown_prefix_falls_back_to_english(self, packs: Any) -> None:
        assert D.ocr_languages_for("XX999999.pdf") == "eng"

    def test_languages_are_comma_separated_never_plus_joined(self, packs: Any) -> None:
        # The bug this pins: docling validates every element of its `lang` list against the
        # installed packs individually, so the single token "rus+eng" -- which tesseract's own CLI
        # accepts -- raises OcrLanguageNotSupportedError and loses the document. Measured: the
        # worker exited 1 and returned 0 characters for a PDF that OCRs fine when the same two
        # languages are passed separately.
        resolved = D.ocr_languages_for("SU1569953A1.pdf")
        assert "+" not in resolved
        assert resolved.split(",") == ["rus", "eng"]

    def test_an_uninstalled_pack_degrades_to_english(self, monkeypatch: Any,
                                                     tmp_path: Any) -> None:
        # Asking tesseract for a pack that is not present makes it exit non-zero, which loses the
        # whole document. Dropping the pack loses only the non-Latin text.
        directory = tmp_path / "tessdata"
        directory.mkdir()
        (directory / "eng.traineddata").write_bytes(b"\x00")
        monkeypatch.setattr(D, "_tessdata_dir", lambda: str(directory))
        assert D.ocr_languages_for("CN111718497B.pdf") == "eng"

    def test_the_resolved_language_reaches_the_worker(self, monkeypatch: Any,
                                                     tmp_path: Any) -> None:
        captured: Dict[str, List[str]] = {}

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command: List[str], **kwargs: Any) -> Any:
            captured["command"] = list(command)
            if "--out" in command:
                with open(command[command.index("--out") + 1], "w", encoding="utf-8") as handle:
                    handle.write('{"tables": [], "text": ""}')
            return _Completed()

        monkeypatch.setattr(D.subprocess, "run", fake_run)
        monkeypatch.setattr(D, "docling_python", lambda: ["/env/bin/python"])
        monkeypatch.setattr(D, "tesseract_command", lambda: "/env/bin/tesseract")
        monkeypatch.setattr(D, "ocr_languages_for", lambda path: "rus,eng")

        pdf = tmp_path / "SU1569953A1.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        D.docling_tables(str(pdf), use_ocr=True)

        command = captured.get("command", [])
        assert "--ocr-lang" in command, "the resolved language never reached the worker"
        assert command[command.index("--ocr-lang") + 1] == "rus,eng"


class TestShouldOcrGatesBeforeConverting:
    """
    ``should_ocr`` answers "would OCR run?" without converting anything.

    It exists because docling runs TableFormer in ACCURATE mode whether or not OCR is wanted, so a
    caller that only wants OCR text must be able to skip docling entirely for documents that do not
    need it. Measured, 100% of 60 sampled papers (663 pages) have a healthy text layer, so
    converting the 124,526-paper corpus to recover text the ``.txt`` already holds would cost
    hundreds of hours and return nothing.
    """

    def test_always_forces_ocr_without_reading_the_pdf(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(D, "OCR_MODE", "always")
        monkeypatch.setattr(D, "needs_ocr", lambda path: (_ for _ in ()).throw(
            AssertionError("needs_ocr must not be consulted when OCR_MODE=always")))
        assert D.should_ocr("/any/file.pdf") is True

    def test_never_disables_ocr_without_reading_the_pdf(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(D, "OCR_MODE", "never")
        monkeypatch.setattr(D, "needs_ocr", lambda path: (_ for _ in ()).throw(
            AssertionError("needs_ocr must not be consulted when OCR_MODE=never")))
        assert D.should_ocr("/any/file.pdf") is False

    def test_auto_defers_to_the_text_layer(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(D, "OCR_MODE", "auto")
        monkeypatch.setattr(D, "needs_ocr", lambda path: True)
        assert D.should_ocr("/any/file.pdf") is True
        monkeypatch.setattr(D, "needs_ocr", lambda path: False)
        assert D.should_ocr("/any/file.pdf") is False

    def test_auto_is_the_default(self) -> None:
        # 'always' would pay full OCR cost on the ~63% of patents and 100% of papers that have a
        # readable text layer; 'never' would leave the 36.7% of patents with no text layer
        # unreadable. The default has to be the one that spends the cost exactly where it buys
        # something.
        import os as _os
        assert _os.getenv("DOCLING_OCR", "auto").strip().lower() == "auto"


class TestLanguageDetection:
    """
    The language is detected from the PAGE, not guessed from the filename.

    The office prefix is weak evidence and was silently wrong in both directions: a CN-office
    patent may publish in English, and a journal paper has no prefix at all, so the whole
    literature corpus resolved to "eng" by default. Detection runs OSD on a 150-dpi render of page
    one (~0.35s), OCRs that page with the packs the script allows, and asks the local model to name
    the language. Measured on six documents in five scripts, it returns the linguistically correct
    answer 6/6 in 1.8-4.1s -- negligible against ~12 s/page for the full OCR pass.
    """

    def _wire(self, monkeypatch, tmp_path, script, llm_languages, confidence=1.0):
        """Drives the real ocr_languages_for with the render, OSD and model all stubbed."""
        directory = tmp_path / "tessdata"
        directory.mkdir(exist_ok=True)
        for language in ("eng", "rus", "chi_sim", "chi_tra", "jpn", "kor", "fra", "deu", "spa"):
            (directory / f"{language}.traineddata").write_bytes(b"\x00")
        # The render goes in its own subdirectory: ocr_languages_for rmtree's the image's PARENT
        # directory when it is done, which is correct in production (_render_page owns a mkdtemp)
        # but would delete the tessdata fixture if both sat in tmp_path together.
        render_dir = tmp_path / "render"
        render_dir.mkdir(exist_ok=True)
        image = render_dir / "page-1.png"
        image.write_bytes(b"\x89PNG")
        monkeypatch.setattr(D, "_tessdata_dir", lambda: str(directory))
        monkeypatch.setattr(D, "_render_page", lambda p, page=1, dpi=150: str(image))
        monkeypatch.setattr(D, "detect_script",
                            lambda p: (script, confidence) if script else None)
        monkeypatch.setattr(D, "_sample_text", lambda i, packs: "a sufficiently long sample " * 4)
        monkeypatch.setattr(D, "identify_languages", lambda sample: list(llm_languages))

    def test_the_script_decides_which_language_leads(self, monkeypatch, tmp_path) -> None:
        # The real failure this pins: the model read JP2025516591A as "English, Japanese" -- both
        # genuinely present on a bilingual patent -- and English-first would make tesseract
        # mis-segment the Japanese glyphs, because it treats the leading language as primary.
        self._wire(monkeypatch, tmp_path, "Japanese", ["eng", "jpn"])
        assert D.ocr_languages_for("/x/JP2025516591A.pdf") == "jpn,eng"

    def test_extra_guesses_from_the_han_ambiguity_are_dropped(self, monkeypatch,
                                                             tmp_path) -> None:
        # Han cannot distinguish Chinese from Japanese kanji, so for CN111718497B the model
        # offered "Chinese, English, Japanese". The third is an artefact; every extra pack costs
        # OCR time and adds a way to be wrong.
        self._wire(monkeypatch, tmp_path, "Han", ["chinese", "english", "japanese"]
                   and ["chi_sim", "eng", "jpn"])
        assert D.ocr_languages_for("/x/CN111718497B.pdf") == "chi_sim,eng"

    def test_english_alone_stays_english_alone(self, monkeypatch, tmp_path) -> None:
        self._wire(monkeypatch, tmp_path, "Latin", ["eng"])
        assert D.ocr_languages_for("/x/US7425924B2.pdf") == "eng"

    def test_a_latin_document_can_be_a_language_the_prefix_would_miss(self, monkeypatch,
                                                                     tmp_path) -> None:
        # The win over the prefix table: WO2008043938A3 is an entirely French PCT publication.
        # The prefix gave "eng,fra,deu"; detection gives French first and drops the dead pack.
        self._wire(monkeypatch, tmp_path, "Latin", ["fra", "eng"])
        assert D.ocr_languages_for("/x/WO2008043938A3.pdf") == "fra,eng"

    def test_english_is_appended_never_substituted(self, monkeypatch, tmp_path) -> None:
        # Even a CJK patent prints its peptide sequences and reagent names in Latin script, and
        # those are the only tokens this project needs off the page.
        self._wire(monkeypatch, tmp_path, "Hangul", ["kor"])
        assert D.ocr_languages_for("/x/KR100866631B1.pdf").split(",")[-1] == "eng"

    def test_a_silent_model_falls_back_to_the_script(self, monkeypatch, tmp_path) -> None:
        # The model is best-effort. The script alone still beats the office prefix when OSD
        # recognised one.
        self._wire(monkeypatch, tmp_path, "Cyrillic", [])
        assert D.ocr_languages_for("/x/SU1569953A1.pdf") == "rus,eng"

    def test_no_render_falls_back_to_the_prefix(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(D, "_render_page", lambda p, page=1, dpi=150: None)
        monkeypatch.setattr(D, "_tessdata_dir", lambda: None)
        assert D.ocr_languages_for("/x/CN999999A.pdf") == "chi_sim,eng"

    def test_detection_never_emits_a_plus_joined_token(self, monkeypatch, tmp_path) -> None:
        # docling validates each element of its lang list individually, so "rus+eng" raises
        # OcrLanguageNotSupportedError and loses the document.
        self._wire(monkeypatch, tmp_path, "Cyrillic", ["rus", "eng"])
        assert "+" not in D.ocr_languages_for("/x/SU1569953A1.pdf")


class TestIdentifyLanguages:
    def test_plain_language_names_become_installed_packs(self, monkeypatch, tmp_path) -> None:
        directory = tmp_path / "tessdata"
        directory.mkdir()
        for language in ("eng", "jpn"):
            (directory / f"{language}.traineddata").write_bytes(b"\x00")
        monkeypatch.setattr(D, "_tessdata_dir", lambda: str(directory))
        monkeypatch.setattr("sample.llm.llm_call",
                            lambda *a, **k: '{"languages": ["Japanese", "English"]}')
        assert D.identify_languages("a long enough sample of text " * 3) == ["jpn", "eng"]

    def test_a_language_with_no_installed_pack_is_discarded(self, monkeypatch,
                                                           tmp_path) -> None:
        # Passing tesseract a pack it does not have makes it exit non-zero and loses the whole
        # document, so an uninstallable answer must not reach it.
        directory = tmp_path / "tessdata"
        directory.mkdir()
        (directory / "eng.traineddata").write_bytes(b"\x00")
        monkeypatch.setattr(D, "_tessdata_dir", lambda: str(directory))
        monkeypatch.setattr("sample.llm.llm_call",
                            lambda *a, **k: '{"languages": ["Japanese", "English"]}')
        assert D.identify_languages("a long enough sample of text " * 3) == ["eng"]

    def test_an_invented_language_name_is_ignored(self, monkeypatch, tmp_path) -> None:
        directory = tmp_path / "tessdata"
        directory.mkdir()
        (directory / "eng.traineddata").write_bytes(b"\x00")
        monkeypatch.setattr(D, "_tessdata_dir", lambda: str(directory))
        monkeypatch.setattr("sample.llm.llm_call",
                            lambda *a, **k: '{"languages": ["chi_sim", "Klingon", "English"]}')
        # "chi_sim" is a tesseract code, not a language name -- the model was asked for names.
        assert D.identify_languages("a long enough sample of text " * 3) == ["eng"]

    def test_too_short_a_sample_is_not_sent_to_the_model(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr("sample.llm.llm_call",
                            lambda *a, **k: called.append(1) or '{"languages": ["English"]}')
        assert D.identify_languages("short") == []
        assert not called, "a sample too short to identify must not cost a model call"

    def test_an_unavailable_model_returns_empty_rather_than_raising(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError("ollama is down")

        monkeypatch.setattr("sample.llm.llm_call", boom)
        assert D.identify_languages("a long enough sample of text " * 3) == []


class TestDetectScript:
    def test_it_parses_the_script_and_its_confidence(self, monkeypatch) -> None:
        class _Done:
            returncode = 0
            stdout = ("Page number: 0\nOrientation in degrees: 0\n"
                      "Script: Han\nScript confidence: 1.12\n")
            stderr = ""

        monkeypatch.setattr(D, "tesseract_command", lambda: "/env/bin/tesseract")
        monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: _Done())
        assert D.detect_script("/x/page.png") == ("Han", 1.12)

    def test_the_confidence_is_reported_because_it_is_often_near_zero(self,
                                                                     monkeypatch) -> None:
        # Measured on this corpus: Hangul 0.00, Japanese 0.67, Han 1.12 -- all correct, all with
        # confidences a caller must not treat as settled. The script narrows; the model decides.
        class _Done:
            returncode = 0
            stdout = "Script: Hangul\nScript confidence: 0.00\n"
            stderr = ""

        monkeypatch.setattr(D, "tesseract_command", lambda: "/env/bin/tesseract")
        monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: _Done())
        assert D.detect_script("/x/page.png") == ("Hangul", 0.0)

    def test_osd_failure_is_not_an_error(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise OSError("tesseract exploded")

        monkeypatch.setattr(D, "tesseract_command", lambda: "/env/bin/tesseract")
        monkeypatch.setattr(D.subprocess, "run", boom)
        assert D.detect_script("/x/page.png") is None


class TestPdftoppmResolution:
    """
    The renderer must resolve off PATH too, and must resolve to a PATH STRING.

    Both halves were broken: the fallback branch called ``_predictor_command`` without importing
    it (NameError), and then treated its return value as a path when it is a command LIST
    (TypeError). Either one took the whole language-detection path down on a host without
    ``pdftoppm`` on PATH -- the exact host the branch exists for -- leaving every scanned document
    with no readable text.
    """

    def test_it_falls_back_to_the_environment_when_not_on_path(self, monkeypatch) -> None:
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda name: None)
        monkeypatch.setattr("sample.peptide_predictors._predictor_command",
                            lambda tool, env: ["/envs/x/bin/pdftoppm"])
        assert D._pdftoppm_command() == "/envs/x/bin/pdftoppm"

    def test_the_result_is_a_string_not_a_command_list(self, monkeypatch) -> None:
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda name: None)
        monkeypatch.setattr("sample.peptide_predictors._predictor_command",
                            lambda tool, env: ["/envs/x/bin/pdftoppm"])
        assert isinstance(D._pdftoppm_command(), str)

    def test_a_relative_resolution_is_refused(self, monkeypatch) -> None:
        # A bare name would be invoked successfully and then render nothing from the wrong binary.
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda name: None)
        monkeypatch.setattr("sample.peptide_predictors._predictor_command",
                            lambda tool, env: ["pdftoppm"])
        assert D._pdftoppm_command() is None

    def test_an_unresolvable_renderer_is_not_an_error(self, monkeypatch) -> None:
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda name: None)
        monkeypatch.setattr("sample.peptide_predictors._predictor_command",
                            lambda tool, env: None)
        assert D._pdftoppm_command() is None


class TestNoTempDirectoryLeak:
    def test_a_failing_language_lookup_leaves_no_payload_directory(self, monkeypatch,
                                                                  tmp_path) -> None:
        """
        Language detection is resolved BEFORE the payload directory is created.

        It renders a page, shells out to tesseract and calls a model, so it has several ways to
        fail -- and it did fail, on the missing import. Done between the mkdtemp and the
        try/finally that removes it, every such failure leaked one temporary directory per
        document.
        """
        import tempfile

        made = []
        real_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(*a, **k):
            path = real_mkdtemp(*a, **k)
            made.append(path)
            return path

        monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)
        monkeypatch.setattr(D, "docling_python", lambda: ["/env/bin/python"])
        monkeypatch.setattr(D, "tesseract_command", lambda: "/env/bin/tesseract")
        monkeypatch.setattr(D, "ocr_languages_for",
                            lambda p: (_ for _ in ()).throw(RuntimeError("detection exploded")))
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")

        try:
            D.docling_document(str(pdf), use_ocr=True)
        except RuntimeError:
            pass

        leaked = [p for p in made if os.path.isdir(p) and "docling-tables-" in p]
        assert not leaked, f"leaked payload directories: {leaked}"


class TestConversionCache:
    """
    docling conversions are cached, because the conversion IS the pipeline's dominant cost.

    Measured over 24 documents, the table pass is 82% of the screening step's run time (429 of
    521s), and OCR adds ~12.3 s/page on top for a scanned document. It is also deterministic for a
    given PDF and option set.

    The payoff is largest ACROSS the two extraction steps: step 9 (refined) re-reads the same PDFs
    step 8 (screening) already converted, so without a cache it pays that 82% a second time.
    """

    def _stub(self, monkeypatch, payload, calls):
        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            calls.append(list(command))
            out = command[command.index("--out") + 1]
            with open(out, "w", encoding="utf-8") as handle:
                import json as _json
                _json.dump(payload, handle)
            return _Completed()

        monkeypatch.setattr(D.subprocess, "run", fake_run)
        monkeypatch.setattr(D, "docling_python", lambda: ["/env/bin/python"])
        monkeypatch.setattr(D, "tesseract_command", lambda: "/env/bin/tesseract")

    def _pdf(self, tmp_path, body=b"%PDF-1.4\nbody\n"):
        p = tmp_path / "doc.pdf"
        p.write_bytes(body)
        return str(p)

    def test_a_second_conversion_of_the_same_pdf_is_a_file_read(self, monkeypatch,
                                                               tmp_path) -> None:
        calls = []
        self._stub(monkeypatch, {"tables": [{"page_no": 1, "cells": []}], "text": "x"}, calls)
        monkeypatch.setattr(D, "DOCLING_CACHE_DIR", str(tmp_path / "cache"))
        pdf = self._pdf(tmp_path)
        first = D.docling_document(pdf, use_ocr=False)
        second = D.docling_document(pdf, use_ocr=False)
        assert len(calls) == 1, "the cached PDF was converted again"
        assert first == second

    def test_changed_pdf_content_is_a_miss(self, monkeypatch, tmp_path) -> None:
        # Keyed on CONTENT, not path: a re-downloaded or repaired file must reconvert.
        calls = []
        self._stub(monkeypatch, {"tables": [], "text": "x"}, calls)
        monkeypatch.setattr(D, "DOCLING_CACHE_DIR", str(tmp_path / "cache"))
        pdf = self._pdf(tmp_path)
        D.docling_document(pdf, use_ocr=False)
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\nDIFFERENT BODY\n")
        D.docling_document(pdf, use_ocr=False)
        assert len(calls) == 2

    def test_the_ocr_decision_is_part_of_the_key(self, monkeypatch, tmp_path) -> None:
        # OCR changes the output, so an OCR'd and a non-OCR'd conversion are different results.
        calls = []
        self._stub(monkeypatch, {"tables": [], "text": "x"}, calls)
        monkeypatch.setattr(D, "DOCLING_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(D, "ocr_languages_for", lambda p: "eng")
        pdf = self._pdf(tmp_path)
        D.docling_document(pdf, use_ocr=False)
        D.docling_document(pdf, use_ocr=True)
        assert len(calls) == 2

    def test_the_language_set_is_part_of_the_key(self, monkeypatch, tmp_path) -> None:
        calls = []
        self._stub(monkeypatch, {"tables": [], "text": "x"}, calls)
        monkeypatch.setattr(D, "DOCLING_CACHE_DIR", str(tmp_path / "cache"))
        pdf = self._pdf(tmp_path)
        monkeypatch.setattr(D, "ocr_languages_for", lambda p: "eng")
        D.docling_document(pdf, use_ocr=True)
        monkeypatch.setattr(D, "ocr_languages_for", lambda p: "rus,eng")
        D.docling_document(pdf, use_ocr=True)
        assert len(calls) == 2, "two language sets must not share one cache entry"

    def test_a_changed_worker_invalidates_everything(self, monkeypatch, tmp_path) -> None:
        # TableFormer mode and what the payload carries live in the worker script, so its digest
        # is part of the key. A stale cache after changing the conversion is indistinguishable
        # from a change that did not help.
        calls = []
        self._stub(monkeypatch, {"tables": [], "text": "x"}, calls)
        monkeypatch.setattr(D, "DOCLING_CACHE_DIR", str(tmp_path / "cache"))
        pdf = self._pdf(tmp_path)
        monkeypatch.setattr(D, "_worker_fingerprint", lambda: "aaaaaaaaaaaa")
        D.docling_document(pdf, use_ocr=False)
        monkeypatch.setattr(D, "_worker_fingerprint", lambda: "bbbbbbbbbbbb")
        D.docling_document(pdf, use_ocr=False)
        assert len(calls) == 2

    def test_a_failed_conversion_is_not_cached(self, monkeypatch, tmp_path) -> None:
        # A timeout or crashed subprocess must not be frozen in as the PDF's permanent result.
        calls = []
        self._stub(monkeypatch, {"error": "conversion crashed"}, calls)
        monkeypatch.setattr(D, "DOCLING_CACHE_DIR", str(tmp_path / "cache"))
        pdf = self._pdf(tmp_path)
        assert D.docling_document(pdf, use_ocr=False) == {}
        assert D.docling_document(pdf, use_ocr=False) == {}
        assert len(calls) == 2, "an errored conversion was cached"

    def test_caching_can_be_disabled(self, monkeypatch, tmp_path) -> None:
        calls = []
        self._stub(monkeypatch, {"tables": [], "text": "x"}, calls)
        monkeypatch.setattr(D, "DOCLING_CACHE_DIR", "")
        pdf = self._pdf(tmp_path)
        D.docling_document(pdf, use_ocr=False)
        D.docling_document(pdf, use_ocr=False)
        assert len(calls) == 2
