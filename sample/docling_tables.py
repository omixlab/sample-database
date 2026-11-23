"""
Table extraction backed by docling's TableFormer, with a Tesseract OCR fallback.

Why this exists alongside :mod:`sample.pdf_tables`, which already reads tables with pdfplumber:

* pdfplumber's ``find_tables``/``extract_tables`` over-detects badly on journal PDFs. Measured on
  one 12-page paper it reported 26 "tables", of which the first several were 2x2 and 2x1 grids
  holding ZERO non-empty cells -- ruled layout blocks and figure borders, not tables.
* It cannot recover a ragged or merged-cell row. ``pdf_tables`` drops any row whose length differs
  from the header's, which is precisely the row TableFormer exists to reconstruct.
* It has no notion of a two-row header, so an assay table whose peptides are its COLUMNS reads as
  though every column header were an organism.

Measured value of the docling path over 40 random PubMed PDFs (437 pages, network blocked):
68% of documents contain at least one table, 25% contain a table carrying peptide-relevant data
(sequence / MIC / strain / activity), for 19 relevant tables and 214 data rows. At 1.14 s/page that
is ~12.5 s per document.

Everything here runs locally. docling cannot be imported into ``sample-extraction`` -- it resolves
numpy 2.x against that environment's pinned numpy<2 -- so it is driven as a subprocess against
``sample-docling``'s own interpreter, resolved by absolute path. ``conda run`` is never used; see
KNOWN_ISSUES.md, which records it misrouting to the wrong environment on this host.
"""
import functools
import json
import logging
import os
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("sample.docling_tables")

#: The environment docling lives in. Isolated on purpose (see the module docstring).
DOCLING_ENV = "sample-docling"
#: The environment carrying the tesseract binary. It is declared in sample-extraction.yml and
#: sample.yml, but is NOT on this host's PATH, so it is always resolved by absolute path.
TESSERACT_ENV = "sample-extraction"

#: Below this many characters per page, a PDF's text layer is too thin to extract from and OCR is
#: worth its cost. Measured over 300 sampled PDFs: 0% have no text layer at all and 0.3% fall below
#: this bound, so this path is a genuine fallback rather than a normal step.
OCR_CHARS_PER_PAGE = 200

#: How the OCR decision is made. ``auto`` runs tesseract only where the text layer is unusable,
#: which is where the entire recall gain is: measured over 60 patents (1,400 pages), 36.7% have a
#: text layer of exactly zero characters and the rest are healthy -- there is almost nothing in
#: between, so the cheap test separates the two cleanly. ``always`` forces the
#: PDF -> PNG -> tesseract path for every page even when a good text layer exists, which is
#: strictly more faithful to the printed page but pays full OCR cost on the ~63% that do not need
#: it. ``never`` disables OCR.
#:
#: DOCLING_OCR=auto (default) | always | never
OCR_MODE = os.getenv("DOCLING_OCR", "auto").strip().lower()

#: Where docling conversion payloads are cached. Set DOCLING_CACHE="" to disable.
#:
#: The conversion is the single most expensive operation in the pipeline: measured over 24
#: documents, the table pass is 82% of the screening step's run time (429 of 521s), and OCR adds
#: ~12.3 s/page on top for a scanned document. It is also deterministic for a given PDF and set of
#: options, which is exactly what a cache needs.
#:
#: The payoff is largest ACROSS the two extraction steps. Step 9 (refined) re-reads the same PDFs
#: step 8 (screening) already converted, so without a cache it pays that 82% twice. With one, step
#: 9's table pass is a file read, which is what makes splitting extraction into two steps nearly
#: free instead of close to doubling the wall clock.
DOCLING_CACHE_DIR = os.getenv("DOCLING_CACHE", os.path.join("data", "docling_cache"))


@functools.lru_cache(maxsize=1)
def _worker_fingerprint() -> str:
    """
    :returns: A digest of the docling worker script, so any change to how a PDF is converted --
        TableFormer mode, cell matching, what the payload carries -- invalidates every entry
        automatically. A hand-maintained version constant would have to be remembered.
    """
    import hashlib

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_docling_worker.py")
    try:
        with open(worker, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except OSError:
        return "unknown"


def _cache_path(pdf_path: str, use_ocr: bool, languages: str) -> Optional[str]:
    """
    Where this conversion's payload is cached, or None when caching is off.

    Keyed on the PDF's CONTENT, not its path: a re-downloaded or repaired file must reconvert, and
    an unchanged one must hit even if it moved. The OCR decision and language set are part of the
    key because they change the output. The accelerator device is NOT: CPU and GPU produce the same
    tables, and excluding it means a cache built on one is still valid on the other.

    :param pdf_path: The PDF.
    :param use_ocr: Whether OCR will run.
    :param languages: The resolved tesseract language string ("" when not OCR-ing).
    :returns: The cache file path, or None.
    """
    if not DOCLING_CACHE_DIR:
        return None
    import hashlib

    digest = hashlib.sha256()
    try:
        with open(pdf_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    key = hashlib.sha256("|".join([
        digest.hexdigest(), "ocr" if use_ocr else "no-ocr", languages, _worker_fingerprint(),
    ]).encode("utf-8")).hexdigest()
    return os.path.join(DOCLING_CACHE_DIR, key[:2], f"{key}.json")

#: Tesseract language packs by patent-office country prefix.
#:
#: A LAST-RESORT fallback only -- see :func:`ocr_languages_for`, which detects the language from the
#: page itself. The office a patent was filed at is weak evidence for the language it is written
#: in: a CN-office patent may be published in English, a US patent may quote Chinese prior art, and
#: a journal paper has no office prefix at all, so this table silently answered "eng" for the
#: entire literature corpus.
OCR_LANGS_BY_PREFIX: Dict[str, str] = {
    "CN": "chi_sim", "TW": "chi_tra", "HK": "chi_tra",
    "JP": "jpn", "KR": "kor",
    "SU": "rus", "RU": "rus", "UA": "ukr", "BY": "bel",
    "DE": "deu", "AT": "deu", "CH": "deu",
    "FR": "fra", "BE": "fra",
    "ES": "spa", "MX": "spa", "AR": "spa", "CL": "spa",
    "BR": "por", "PT": "por",
    "IT": "ita", "NL": "nld", "PL": "pol", "SE": "swe", "DK": "dan",
    "NO": "nor", "FI": "fin", "CZ": "ces", "TR": "tur", "GR": "ell",
    "IL": "heb", "IN": "hin", "TH": "tha", "VN": "vie",
}

#: Candidate packs to OCR a one-page SAMPLE with, keyed by the script tesseract's OSD reports.
#:
#: Narrowing, not deciding. OSD reads the glyph shapes off a rendered page and is reliable about
#: the SCRIPT -- verified correct on all six documents tried (Han, Cyrillic, Hangul, Japanese,
#: Latin, Latin) -- but it cannot name the language, and its confidence is often very low on these
#: scans (Hangul 0.00, Japanese 0.67, Han 1.12). Worse, ``Han`` is genuinely ambiguous: Japanese
#: kanji ARE Han characters, so a Japanese page can be reported as Han and a Chinese one as
#: Japanese. So the script only chooses which packs are worth a cheap sample; the LLM then reads
#: that sample and names the language.
_SCRIPT_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "Han": ("chi_sim", "jpn", "eng"),
    "HanS": ("chi_sim", "eng"),
    "HanT": ("chi_tra", "eng"),
    "HanS_vert": ("chi_sim_vert", "chi_sim", "eng"),
    "HanT_vert": ("chi_tra_vert", "chi_tra", "eng"),
    "Japanese": ("jpn", "chi_sim", "eng"),
    "Japanese_vert": ("jpn_vert", "jpn", "eng"),
    "Hangul": ("kor", "eng"),
    "Hangul_vert": ("kor_vert", "kor", "eng"),
    "Korean": ("kor", "eng"),
    "Cyrillic": ("rus", "eng"),
    "Greek": ("ell", "eng"),
    "Arabic": ("ara", "eng"),
    "Hebrew": ("heb", "eng"),
    "Devanagari": ("hin", "eng"),
    "Thai": ("tha", "eng"),
    "Latin": ("eng", "fra", "deu", "spa"),
    "Fraktur": ("deu", "eng"),
}

#: Language names the model may return, mapped to tesseract packs.
#:
#: Keyed on lower-cased names because the model is asked for plain English language names rather
#: than tesseract codes -- asking a general model for "chi_sim" invites invented codes, while
#: "Simplified Chinese" is something it actually knows. Anything it returns that is not in this
#: table is discarded rather than passed to tesseract, which would exit non-zero and lose the
#: document.
_LANGUAGE_TO_PACK: Dict[str, str] = {
    "english": "eng",
    "chinese": "chi_sim", "simplified chinese": "chi_sim", "chinese (simplified)": "chi_sim",
    "mandarin": "chi_sim", "traditional chinese": "chi_tra", "chinese (traditional)": "chi_tra",
    "japanese": "jpn", "korean": "kor",
    "russian": "rus", "ukrainian": "ukr", "belarusian": "bel",
    "german": "deu", "french": "fra", "spanish": "spa", "portuguese": "por",
    "italian": "ita", "dutch": "nld", "polish": "pol", "swedish": "swe",
    "danish": "dan", "norwegian": "nor", "finnish": "fin", "czech": "ces",
    "turkish": "tur", "greek": "ell", "hebrew": "heb", "hindi": "hin",
    "thai": "tha", "vietnamese": "vie", "arabic": "ara",
}

#: Characters of the sampled page handed to the model. Enough to name a language several times
#: over; a whole page would mostly add OCR noise and tokens.
_LANGUAGE_SAMPLE_CHARS = 1200

_LANGUAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "languages": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["languages"],
}

_LANGUAGE_PROMPT = (
    "You are identifying which natural languages a document is written in, from a short sample of "
    "OCR'd text. The sample may contain OCR errors and layout noise; ignore them.\n\n"
    "Reply with the languages actually present, as plain English language names (for example "
    "\"Simplified Chinese\", \"Japanese\", \"Korean\", \"Russian\", \"English\"), most "
    "prominent first, at most three. Distinguish Chinese from Japanese carefully: Japanese mixes "
    "hiragana and katakana with Han characters, while Chinese uses Han characters only. "
    "Patent and scientific documents are often bilingual with English, so include English only if "
    "English words are genuinely present."
)


def _pdftoppm_command() -> Optional[str]:
    """
    :returns: Absolute path to ``pdftoppm``, or None. Used to render the one page that language
        detection reads; poppler ships it in the extraction environment.
    """
    import shutil

    found = shutil.which("pdftoppm")
    if found:
        return found
    # Imported locally, like the other two users of it in this module: importing
    # sample.peptide_predictors at module scope would pull its own dependency tree into every
    # consumer of this one. Omitting the import entirely raised NameError on exactly the host this
    # branch exists for -- one without pdftoppm on PATH -- which took the whole language-detection
    # path down and left every scanned document yielding nothing.
    from sample.peptide_predictors import _predictor_command

    # _predictor_command returns a COMMAND LIST, not a path -- same as tesseract_command() above,
    # which takes [0]. Treating it as a string raised TypeError in os.path.isabs.
    command = _predictor_command("pdftoppm", TESSERACT_ENV)
    candidate = command[0] if command else None
    return candidate if candidate and os.path.isabs(candidate) else None


def _render_page(pdf_path: str, page: int = 1, dpi: int = 150) -> Optional[str]:
    """
    One page of a PDF as a PNG, for script and language detection.

    :param pdf_path: The PDF.
    :param page: 1-based page number.
    :param dpi: Render resolution. 150 is plenty to recognise a script and read a sample, and
        measured at 0.35s per page -- the full OCR pass runs at docling's scale=3.0 instead.
    :returns: Path to the PNG inside a caller-owned temporary directory, or None on failure. The
        caller is responsible for removing ``os.path.dirname`` of the result.
    """
    command = _pdftoppm_command()
    if not command:
        return None
    import tempfile

    directory = tempfile.mkdtemp(prefix="ocr-lang-")
    prefix = os.path.join(directory, "page")
    try:
        subprocess.run([command, "-f", str(page), "-l", str(page), "-r", str(dpi), "-png",
                        pdf_path, prefix],
                       capture_output=True, timeout=120, check=True)
    except Exception:  # noqa: BLE001 -- detection is best-effort; OCR still has its fallbacks
        import shutil as _shutil

        _shutil.rmtree(directory, ignore_errors=True)
        return None
    for name in sorted(os.listdir(directory)):
        if name.endswith(".png"):
            return os.path.join(directory, name)
    import shutil as _shutil

    _shutil.rmtree(directory, ignore_errors=True)
    return None


def detect_script(image_path: str) -> Optional[Tuple[str, float]]:
    """
    The script tesseract's orientation-and-script detection reports for a page image.

    :param image_path: A rendered page.
    :returns: ``(script, confidence)``, or None when OSD is unavailable or finds nothing. The
        confidence is returned because it is routinely near zero on these scans and callers must
        not treat the script as settled on its own.
    """
    binary = tesseract_command()
    if not binary:
        return None
    try:
        done = subprocess.run([binary, image_path, "-", "--psm", "0"],
                              capture_output=True, text=True, timeout=180, check=True)
    except Exception:  # noqa: BLE001
        return None
    script, confidence = None, 0.0
    for line in (done.stdout or "").splitlines():
        if line.lower().startswith("script:"):
            script = line.split(":", 1)[1].strip()
        elif line.lower().startswith("script confidence:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.0
    return (script, confidence) if script else None


def _sample_text(image_path: str, packs: Sequence[str]) -> str:
    """
    OCR one page image with the candidate packs, to get text for the model to judge.

    :param image_path: A rendered page.
    :param packs: Candidate tesseract packs, already filtered to installed ones.
    :returns: The OCR'd text, or "" on any failure.
    """
    binary = tesseract_command()
    if not binary or not packs:
        return ""
    try:
        done = subprocess.run([binary, image_path, "-", "-l", "+".join(packs)],
                              capture_output=True, text=True, timeout=300, check=True)
    except Exception:  # noqa: BLE001
        return ""
    return (done.stdout or "").strip()


def identify_languages(sample: str) -> List[str]:
    """
    The tesseract packs for the languages a local model reads in a text sample.

    Uses the project's shared :func:`sample.llm.llm_call`, so it runs against the same local Ollama
    model as the rest of the pipeline and reaches no remote provider.

    :param sample: OCR'd text from one page.
    :returns: Installed pack names, most prominent language first; empty when the model is
        unavailable, answers unusably, or names only languages with no pack installed. Empty always
        means "fall back", never an error.
    """
    sample = (sample or "").strip()
    if len(sample) < 40:
        return []
    try:
        from sample.config import DEFAULT_EXTRACT_MODEL
        from sample.llm import clean_json, llm_call

        raw = llm_call(
            [{"role": "system", "content": _LANGUAGE_PROMPT},
             {"role": "user", "content": sample[:_LANGUAGE_SAMPLE_CHARS]}],
            temperature=0.0, response_schema=_LANGUAGE_SCHEMA,
            model=DEFAULT_EXTRACT_MODEL, max_tokens=128, timeout=120,
        )
        data = clean_json(raw)
    except Exception as e:  # noqa: BLE001 -- detection is best-effort
        logger.debug(f"language identification unavailable: {type(e).__name__}: {e}")
        return []

    names = data.get("languages") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return []

    directory = _tessdata_dir()
    packs: List[str] = []
    for name in names[:3]:
        pack = _LANGUAGE_TO_PACK.get(str(name).strip().lower())
        if not pack or pack in packs:
            continue
        if directory and not os.path.exists(os.path.join(directory, f"{pack}.traineddata")):
            continue
        packs.append(pack)
    return packs


def _installed(packs: Sequence[str]) -> List[str]:
    """
    :param packs: Candidate pack names.
    :returns: Those with a ``.traineddata`` present, order preserved. An uninstalled pack makes
        tesseract exit non-zero and loses the whole document, so it is dropped rather than passed.
    """
    directory = _tessdata_dir()
    if not directory:
        return list(packs)
    return [p for p in packs
            if os.path.exists(os.path.join(directory, f"{p}.traineddata"))]


def _languages_from_prefix(pdf_path: str) -> str:
    """
    The office-prefix fallback, used only when detection could not run.

    :param pdf_path: The PDF path; only its basename is inspected.
    :returns: A comma-separated tesseract language string. Always valid.
    """
    name = os.path.basename(pdf_path).upper()
    prefix = "".join(ch for ch in name[:2] if ch.isalpha())
    if prefix in _MULTILINGUAL_PREFIXES:
        languages = list(_MULTILINGUAL_PREFIXES[prefix])
    else:
        languages = ["eng"]
        native = OCR_LANGS_BY_PREFIX.get(prefix) if prefix not in _ENGLISH_PREFIXES else None
        if native:
            languages.insert(0, native)
    usable = _installed(languages)
    return ",".join(usable or ["eng"])


#: Offices that publish in several Latin-script languages rather than one.
_MULTILINGUAL_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "WO": ("eng", "fra", "deu"),
    "EP": ("eng", "fra", "deu"),
}

#: Prefixes already in English, for the fallback path only.
_ENGLISH_PREFIXES = frozenset({"US", "GB", "CA", "AU", "NZ", "IE", "ZA", "SG", "PH"})


@functools.lru_cache(maxsize=4096)
def ocr_languages_for(pdf_path: str) -> str:
    """
    The tesseract language string to OCR one document with, detected from the page itself.

    Three stages, each a fallback for the one before:

    1. **Script**, from tesseract's OSD on a 150-dpi render of page 1 (~0.35s). Reliable about the
       script, but it cannot name a language and its confidence is routinely near zero here, and
       ``Han`` does not distinguish Chinese from Japanese.
    2. **Language**, from the local model reading a one-page OCR sample taken with the packs that
       script allows. This is what actually resolves Chinese vs Japanese, and what gives journal
       papers -- which have no office prefix at all -- a real answer instead of "eng".
    3. **Office prefix**, if neither of the above produced anything.

    ``eng`` is appended to any non-Latin result rather than substituted, because peptide sequences,
    residue names and reagent names are printed in Latin script even in a CJK patent, and those are
    the tokens this project needs off the page.

    Cached per PDF: the result is deterministic and the full OCR pass may ask for it more than once.

    :param pdf_path: The PDF to be OCR'd.
    :returns: A COMMA-separated tesseract language string, e.g. ``"chi_sim,eng"``, matching the
        worker's ``--ocr-lang`` contract. Comma and not ``+``: docling validates every element of
        its ``lang`` list against the installed packs individually, so the single token
        ``"rus+eng"`` -- which tesseract's own CLI accepts -- raises OcrLanguageNotSupportedError
        and loses the document.
    """
    import shutil as _shutil

    image = _render_page(pdf_path)
    if not image:
        return _languages_from_prefix(pdf_path)

    try:
        detected = detect_script(image)
        script = detected[0] if detected else None
        confidence = detected[1] if detected else 0.0
        candidates = _installed(_SCRIPT_CANDIDATES.get(script or "", ()))
        packs = identify_languages(_sample_text(image, candidates)) if candidates else []
    finally:
        _shutil.rmtree(os.path.dirname(image), ignore_errors=True)

    if not packs:
        # The model said nothing usable. The script alone is still better than the office prefix
        # when OSD actually recognised one, so prefer its first candidate.
        if candidates:
            packs = candidates[:1]
        else:
            return _languages_from_prefix(pdf_path)

    # The SCRIPT arbitrates which language leads, because tesseract treats the first language as
    # primary and an English-primary model mis-segments CJK glyphs before any language model
    # applies. Measured: the model read the Japanese patent JP2025516591A as "English, Japanese"
    # -- both genuinely present on a bilingual patent, but ordered that way it would have OCR'd a
    # Japanese page English-first.
    #
    # It also drops the model's extra guesses. Han script cannot distinguish Chinese from Japanese
    # kanji, so for CN111718497B the model offered "Chinese, English, Japanese"; the third is an
    # artefact of that ambiguity and every extra pack costs OCR time and adds a way to be wrong.
    # One language plus English is what this corpus actually needs: the non-Latin text for context
    # and Latin for the sequences and reagent names.
    script_packs = [p for p in candidates if p != "eng"]
    primary = next((p for p in packs if p in script_packs), packs[0])
    packs = [primary] + ([] if primary == "eng" else _installed(["eng"]))
    logger.debug(f"OCR language for {os.path.basename(pdf_path)}: {','.join(packs)} "
                 f"(script={script!r} confidence={confidence})")
    return ",".join(packs)



#: Wall-clock ceiling for one document. The measured mean is ~12.5 s; this bounds the pathological
#: tail without truncating a legitimately long paper.
DEFAULT_TIMEOUT = 900

#: Threads each docling subprocess may use. Pinned to one, deliberately.
#:
#: Torch defaults to one thread per core -- 16 on this 24-core host -- and the extraction step calls
#: this from a pool of worker THREADS, so N workers would start N subprocesses each claiming 16
#: threads. That is 128 runnable threads on 24 cores, and the step would spend its time in the
#: scheduler. Measured on one document: 16.7s at 4 threads against 20.5s at 1, so capping costs
#: ~23% per document and buys back an order of magnitude in contention.
_SUBPROCESS_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

#: Peak resident set of one conversion, measured with `/usr/bin/time -v` on a 12-page paper.
_CONVERSION_PEAK_MB = 2100


def _default_conversion_slots() -> int:
    """
    How many conversions may run at once on this machine.

    Bounded by MEMORY first, not core count. The earlier version of this used ``cores // 4``, which
    on a 24-core host with 15.6 GB of RAM allowed four conversions -- ~8.4 GB of torch processes on
    top of the extraction step's own ~5.2 GB of shared lookups (spaCy, taxonomy automata, the NLI
    model). Observed live: swap-out climbing to 43 MB/s with the CPU 80-95% idle, i.e. the machine
    thrashing rather than working. Cores are not the scarce resource here; resident memory is.

    :returns: A slot count that fits in currently-available memory, capped at 4. Override with
        SAMPLE_DOCLING_SLOTS when the default reads the machine wrongly -- for instance when the
        extraction parent has not yet loaded its lookups at the moment this is first imported.
    """
    override = os.getenv("SAMPLE_DOCLING_SLOTS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)

    by_cores = max(1, (os.cpu_count() or 4) // 4)
    by_memory = 4
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available_mb = int(line.split()[1]) // 1024
                    # Leave a slot's worth of headroom: filling available memory exactly is how the
                    # OOM killer ends a multi-day run.
                    by_memory = max(1, (available_mb - _CONVERSION_PEAK_MB) // _CONVERSION_PEAK_MB)
                    break
    except Exception as exc:                                    # noqa: BLE001
        logger.debug(f"Could not read MemAvailable, keeping the core-based bound: {exc}")
    return max(1, min(4, by_cores, by_memory))


#: How many docling subprocesses may run at once, regardless of how many extraction workers call in.
#: Threads one docling conversion may use. 16 by measurement -- see _threads_per_conversion.
DOCLING_THREADS = 16

#: Conversions permitted at once. Memory gives an upper bound (see _default_conversion_slots), and
#: the thread budget gives another: THREADS x SLOTS must not exceed the cores, or the step spends
#: its time in the scheduler -- the contention the original one-thread pin existed to prevent. The
#: measured configuration is 16 threads on this 24-core host, which means one conversion at a time.
def _thread_budget_slots() -> int:
    """
    How many conversions the CORE budget allows: one per THREADS-sized share of the machine.

    Reads the SAMPLE_DOCLING_THREADS override rather than the DOCLING_THREADS constant, which is
    the bug this replaces. The override was documented as the way to retune threads per
    conversion, but only ``_threads_per_conversion`` consulted it -- this limit used the constant.
    So ``SAMPLE_DOCLING_THREADS=4`` on a 24-core host gave 4 threads per conversion and still only
    ONE conversion (24 // 16), i.e. 4 of 24 cores busy, which is worse than leaving it alone. The
    knob could only ever make things slower, never faster.

    :returns: Cores divided by the threads one conversion may use, at least 1.
    """
    override = os.getenv("SAMPLE_DOCLING_THREADS", "").strip()
    threads = int(override) if override.isdigit() and int(override) > 0 else DOCLING_THREADS
    return max(1, (os.cpu_count() or 4) // max(1, threads))


MAX_CONCURRENT_CONVERSIONS = max(1, min(_default_conversion_slots(), _thread_budget_slots()))


def _threads_per_conversion() -> int:
    """
    How many threads ONE docling conversion may use.

    Derived from the conversion semaphore, which is the piece the original one-thread pin did not
    account for. That pin was calibrated against "N worker threads each starting a subprocess that
    claims one thread per core" -- 128 runnable threads on 24 cores. But MAX_CONCURRENT_CONVERSIONS
    now bounds how many conversions exist at once, so the two limits were multiplying: with the
    semaphore at 1 the table pass was using ONE core of 24, and the table pass is 82% of the
    screening step's run time.

    Splitting the cores across the permitted conversions restores the intent of the pin -- never
    more runnable threads than cores -- without leaving the machine idle.

    The cap is 16, measured on this host over the same 24 documents with a cold cache, four
    extraction workers and TableFormer on the GPU:

        threads   wall clock   loadavg (24 cores)
          8         268.8s       5.25
         16         223.2s       4.23

    So 16 is a real 1.20x over 8, with the same 57 peptides, and the box is nowhere near saturated
    at either -- loadavg stayed around 4-5 of 24, which says the step is bound by the GPU and by
    per-document I/O rather than by CPU. 24 threads was not measured; the cap is a deliberate stop
    short of one-thread-per-core because the extraction step runs its own worker threads on top of
    this, so `threads x slots <= cores` undercounts the real demand, and because each thread adds
    resident memory to a step whose concurrency is already bounded by memory.

    :returns: Threads per conversion. Override with SAMPLE_DOCLING_THREADS.
    """
    override = os.getenv("SAMPLE_DOCLING_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    cores = os.cpu_count() or 4
    return max(1, min(DOCLING_THREADS, cores))


def _subprocess_env() -> Dict[str, str]:
    """
    :returns: The environment one docling subprocess runs with -- the offline pins plus a thread
        count sized to the conversion semaphore.
    """
    threads = str(_threads_per_conversion())
    env = dict(_SUBPROCESS_THREAD_ENV)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[key] = threads
    return env
_CONVERSION_SLOTS = threading.Semaphore(MAX_CONCURRENT_CONVERSIONS)

_AA = set("ACDEFGHIKLMNPQRSTVWY")
_TERMINAL_MODIFIER = re.compile(r"(NH2|AC|COOH)$")
_UNIT_IN_PARENS = re.compile(r"\(([^)]*)\)")
_UNIT_SHAPE = re.compile(r"^[a-zA-Zµμ%]+(/[a-zA-Z]+)?$")
_ASSAY_HEADER = re.compile(r"\bMIC\b|\bMBC\b|\bMFC\b|\bIC ?50\b|\bEC ?50\b|\bLC ?50\b|"
                           r"h[ae]moly|zone of inhibition|\bZOI\b", re.IGNORECASE)
_SEQUENCE_HEADER = re.compile(r"sequence|primary structure", re.IGNORECASE)
#: Any IUPAC three-letter residue name, used only to decide whether a cell is worth handing to the
#: three-letter converter at all.
_THREE_LETTER_RESIDUE = re.compile(
    r"\b(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\b",
    re.IGNORECASE)
_NAME_HEADER = re.compile(r"peptide|compound|analog|derivative|\bname\b|\bcode\b", re.IGNORECASE)

_ACTIVITY_TYPES: Tuple[Tuple[str, str], ...] = (
    ("ic50", "IC50"), ("ec50", "EC50"), ("lc50", "LC50"), ("mbc", "MBC"), ("mfc", "MFC"),
    ("zoi", "Growth Inhibition"), ("zone", "Growth Inhibition"),
    ("hemoly", "Hemolysis"), ("haemoly", "Hemolysis"), ("mic", "MIC"),
)


def _dictionary_words() -> frozenset:
    """
    English words, used only to reject residue-alphabet cells that are ordinary words.

    :returns: Upper-cased words, or an empty set when the system word list is absent (in which case
        the length and alphabet rules below still apply).
    """
    global _WORDS_CACHE
    if _WORDS_CACHE is None:
        try:
            with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as handle:
                _WORDS_CACHE = frozenset(w.strip().upper() for w in handle)
        except OSError:
            _WORDS_CACHE = frozenset()
    return _WORDS_CACHE


_WORDS_CACHE: Optional[frozenset] = None


def sequence_from_cell(cell: Any, min_length: int = 10) -> Optional[str]:
    """
    The one-letter sequence a cell holds, whether written in one- or three-letter code.

    Sequence tables in this literature routinely use three-letter code with residue positions
    interleaved and explicit termini: ``H-Phe 1 -Leu 2 -Pro 3 - ... -Leu 13 -NH 2``. Those cells are
    the ones that actually carry sequences, and a one-letter-only test rejects every one of them.

    The pre-clean is load-bearing rather than cosmetic. Handed the raw cell,
    ``sequence.sequence_to_1_letter`` reads the N-terminal ``H-`` as histidine and returns
    ``HFLPLIGRVLSGIL`` -- a sequence that is wrong rather than absent, which no downstream check
    would catch. Stripping the terminus first yields the correct ``FLPLIGRVLSGIL``.

    :param cell: The raw cell text.
    :param min_length: Shortest accepted sequence.
    :returns: The one-letter sequence, or None when the cell holds no sequence.
    """
    text = str(cell or "").strip()
    if not text or _is_ordinary_prose(text):
        return None

    if looks_like_sequence(text, min_length=min_length):
        candidate = _TERMINAL_MODIFIER.sub("", re.sub(r"[\s\-]", "", text).upper())
        return candidate if _passes_shared_sequence_test(candidate) else None

    # Three-letter spelling. Only attempt it when the cell actually names residues, so ordinary
    # prose cells are not run through a converter that would return a partial match.
    if not _THREE_LETTER_RESIDUE.search(text):
        return None
    cleaned = _clean_three_letter(text)
    if not cleaned:
        return None
    try:
        from sample.sequence import sequence_to_1_letter
        converted = (sequence_to_1_letter(cleaned) or "").strip().upper()
    except Exception as exc:                                    # noqa: BLE001
        logger.debug(f"Three-letter conversion failed for {text[:40]!r}: {exc}")
        return None
    if len(converted) < min_length or any(c not in _AA for c in converted):
        return None
    return converted if _passes_shared_sequence_test(converted) else None


def _passes_shared_sequence_test(sequence: str) -> bool:
    """
    Whether the project's own sequence test also accepts this string.

    Required IN ADDITION to the rules above, because the two catch different mistakes and neither
    is sufficient alone.

    ``nlp.is_valid_sequence`` is too permissive for table cells -- it returns True for
    ``"Magainin 2"``, ``">128"``, ``"ND"`` and ``"Escherichia coli"``, which is why this module
    does not rely on it by itself. But it carries a log-likelihood test (see nlp.py's
    ``_DNA_MIN_LEN`` block) that rejects nucleotide runs spelling valid amino-acid letters, and
    that is the one check here that cannot be hand-rolled: A, C, G and T are all residue codes, so
    a PCR primer like ``GCGAAGGCATCAACAT`` clears every rule in this module -- residue alphabet,
    length floor, dictionary, prose -- and was being recorded as a peptide sequence. Methods
    sections print primer tables routinely; of the ten records from the first two real hits, seven
    were primers and they were the ones that SURVIVED the merge, so they would have reached the
    database as peptides. The text extractor has never had this problem (0 of 3,000 sampled
    outputs contain a DNA-alphabet peptide) because it uses this test.

    :param sequence: A candidate one-letter sequence.
    :returns: True when the shared test accepts it too.
    """
    try:
        from sample.nlp import is_valid_sequence
    except Exception as exc:                                    # noqa: BLE001
        logger.debug(f"Shared sequence test unavailable, accepting {sequence[:16]!r}: {exc}")
        return True
    return bool(is_valid_sequence(sequence))


def _is_ordinary_prose(text: str) -> bool:
    """
    Whether a cell is ordinary English rather than a sequence.

    Checked on the cell BEFORE hyphens are stripped, which is the whole point: ``"Gram-negative"``
    survives every other test here. Its letters are all residue codes, removing the hyphen makes it
    twelve characters -- over the length floor -- and ``GRAMNEGATIVE`` is not in the word list, so
    the dictionary check passes it too. It was accepted as a peptide sequence off a real table's
    clade heading row. Splitting on the hyphen first makes it two ordinary words.

    ``"Gram-positive"`` escaped only by luck (``o`` is not a residue code), which is not a margin
    worth relying on.

    :param text: The raw cell text.
    :returns: True when every whitespace/hyphen-separated part is an English word.
    """
    parts = [p for p in re.split(r"[\s\-]+", str(text or "").strip()) if p]
    if not parts:
        return False
    words = _dictionary_words()
    if not words:
        return False
    alphabetic = [p for p in parts if p.isalpha()]
    if len(alphabetic) != len(parts):
        return False
    return all(part.upper() in words for part in alphabetic)


def _clean_three_letter(text: str) -> str:
    """
    Strip termini and residue numbering from a three-letter sequence cell.

    :param text: The raw cell, e.g. ``"H-Phe 1 -Leu 2 - ... -Leu 13 -NH 2"``.
    :returns: A hyphen-separated three-letter string, e.g. ``"Phe-Leu-..."``.
    """
    cleaned = re.sub(r"^\s*H\s*-", "", str(text))          # N-terminal H-, not histidine
    cleaned = re.sub(r"-?\s*NH\s*2\s*$", "", cleaned)      # C-terminal amide
    cleaned = re.sub(r"-?\s*(OH|COOH|Ac)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=[A-Za-z])\s*\d+\s*", "-", cleaned)   # residue position numbers
    return re.sub(r"[-\s]+", "-", cleaned).strip("-")


def looks_like_sequence(cell: Any, min_length: int = 10) -> bool:
    """
    Whether a table cell holds a peptide sequence.

    This does NOT use ``nlp.is_valid_sequence``. That function is tuned for running prose and, on
    table cells, returns True for ``"Magainin 2"``, ``">128"``, ``"ND"`` and ``"Escherichia coli"``
    -- all of which appear in the cells this module scans, and any of which would be recorded as a
    peptide sequence. ``pdf_tables`` scans every cell of a row with it today.

    There is deliberately no distinct-residue floor: ``KLLKLLKKLLKLLK`` is a real designed AMP and
    uses two residues. The length floor does that work instead, since peptide NAMES that happen to
    be residue-only (``MAGAININ``, ``ALANINE``) are short while published sequences are not.

    :param cell: The raw cell text.
    :param min_length: Shortest accepted sequence, after whitespace and hyphens are removed.
    :returns: True when the cell is written entirely in the one-letter residue alphabet.
    """
    text = re.sub(r"[\s\-]", "", str(cell or "")).upper()
    text = _TERMINAL_MODIFIER.sub("", text)
    if len(text) < min_length or not text.isalpha():
        return False
    if any(character not in _AA for character in text):
        return False
    return text not in _dictionary_words()


def unit_from_header(header: str) -> Optional[str]:
    """
    The concentration unit a column header declares, or None.

    Never guesses. ``pdf_tables`` stamps every activity it builds with ``"ug/mL"`` regardless of
    what the header says, so a ``MIC ( µ M)`` column -- which this corpus really does contain -- is
    recorded as micrograms per millilitre, wrong by roughly a factor of 1000 for a typical peptide.
    Returning None lets the caller omit the field rather than assert something false.

    The token is extracted BEFORE canonicalisation, because ``preprocessing.canonicalize_unit`` is a
    spelling normaliser rather than a parser: handed a whole header it returns the header back
    (``"MIC ( µ M)"`` -> ``"MIC ( u M)"``), and handed an organism abbreviation like ``"MM"`` it
    returns ``"uM"``, inventing a unit from a column that has none.

    :param header: The column header text.
    :returns: A canonical unit such as ``"uM"`` or ``"ug/mL"``, or None when the header declares no
        unit.
    """
    from sample.preprocessing import canonicalize_unit

    match = _UNIT_IN_PARENS.search(str(header or ""))
    if not match:
        return None
    token = re.sub(r"\s+", "", match.group(1))
    if not token or len(token) > 12:
        return None
    canonical = canonicalize_unit(token)
    return canonical if canonical and _UNIT_SHAPE.match(canonical) else None


def activity_type_from_header(header: str) -> Optional[str]:
    """
    The assay a column header names, or None when it names none.

    :param header: The column header text.
    :returns: ``"MIC"``, ``"IC50"``, ``"Hemolysis"`` and so on, or None. ``pdf_tables`` defaults
        this to ``"MIC"`` for any cell holding a digit, which labels hemolysis percentages and
        peptide concentrations as minimum inhibitory concentrations.
    """
    lowered = str(header or "").lower()
    for needle, label in _ACTIVITY_TYPES:
        if needle in lowered:
            return label
    return None


def docling_python() -> Optional[List[str]]:
    """
    The ``sample-docling`` interpreter, as an argv prefix.

    :returns: ``[".../envs/sample-docling/bin/python"]``, or None when that environment has not
        been built -- in which case callers fall back to the pdfplumber path rather than failing.
    """
    from sample.peptide_predictors import _predictor_command
    return _predictor_command("python", DOCLING_ENV)


def tesseract_command() -> Optional[str]:
    """
    The absolute path to the ``tesseract`` binary, or None.

    :returns: The resolved path. A bare ``"tesseract"`` is never returned: this host does not carry
        it on PATH, so pytesseract-style bare invocation fails at the first OCR call even though
        the import succeeds.
    """
    resolved = None
    from sample.peptide_predictors import _predictor_command
    command = _predictor_command("tesseract", TESSERACT_ENV)
    if command:
        resolved = command[0]
    return resolved


@functools.lru_cache(maxsize=8192)
def needs_ocr(pdf_path: str, chars_per_page: int = OCR_CHARS_PER_PAGE) -> bool:
    """
    Whether a PDF's text layer is too thin to read, so OCR is worth its cost.

    Memoized because it is asked twice per document -- once through :func:`should_ocr` for the OCR
    text and once inside :func:`docling_document` for the conversion itself -- and each call opens
    the PDF and extracts text from up to 12 pages. Measured at 24.1 ms per document, so the
    duplicate read costs about 0.31 hours over the 46,189-document corpus: small against the
    model's own cost, but free to remove, and it also stops the two call sites from ever reaching
    different conclusions about the same file.

    :param pdf_path: The PDF to inspect.
    :param chars_per_page: Mean characters per page below which OCR is warranted.
    :returns: True for scanned or image-only PDFs. False on any inspection failure -- an unreadable
        PDF is not evidence that OCR would help, and running it is several times slower.
    """
    try:
        import fitz
    except ImportError:
        return False
    try:
        with fitz.open(pdf_path) as document:
            pages = min(len(document), 12)
            if pages == 0:
                return False
            total = sum(len(document[index].get_text().strip()) for index in range(pages))
    except Exception as exc:                                    # noqa: BLE001
        logger.debug(f"Could not measure the text layer of {pdf_path}: {exc}")
        return False
    return (total / pages) < chars_per_page


def _tessdata_dir() -> Optional[str]:
    """
    :returns: The directory holding ``*.traineddata``, or None when it cannot be located.
    """
    command = tesseract_command()
    if not command:
        return None
    directory = os.path.join(os.path.dirname(os.path.dirname(command)), "share", "tessdata")
    return directory if os.path.isdir(directory) else None


def should_ocr(pdf_path: str) -> bool:
    """
    Whether this document would be OCR'd, decided WITHOUT converting it.

    Exists so a caller that only wants OCR text can skip docling entirely for the documents that do
    not need it. That distinction is the whole cost model: measured, 100% of 60 sampled papers (663
    pages) have a healthy text layer, so running docling over the paper corpus to obtain text it
    already has would cost hundreds of hours and return nothing the ``.txt`` did not already hold.

    :param pdf_path: The PDF to test.
    :returns: True when :data:`OCR_MODE` forces OCR, or when the text layer is unusable.
    """
    if OCR_MODE == "always":
        return True
    if OCR_MODE == "never":
        return False
    return needs_ocr(pdf_path)


def docling_document(pdf_path: str, use_ocr: Optional[bool] = None,
                     timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    One PDF converted by docling: its tables as cell grids AND its text.

    Both come from a SINGLE conversion. Rendering and OCR-ing a scanned document is the expensive
    part by a wide margin, so asking for the text separately would double the cost of the only
    documents that need it.

    :param pdf_path: The PDF to convert.
    :param use_ocr: Force OCR on or off. None consults :data:`OCR_MODE`, which callers should
        normally let decide.
    :param timeout: Seconds before the subprocess is abandoned.
    :returns: ``{"tables": [...], "text": "..."}``, or ``{}`` on any failure -- a missing
        environment, a crashed conversion or a timeout all degrade to "nothing from this backend"
        so the caller can fall back, and each is logged with its reason rather than passing
        silently.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return {}

    interpreter = docling_python()
    if not interpreter:
        logger.debug(f"{DOCLING_ENV} is not built; skipping the docling table backend.")
        return {}

    if use_ocr is None:
        if OCR_MODE == "always":
            use_ocr = True
        elif OCR_MODE == "never":
            use_ocr = False
        else:
            use_ocr = needs_ocr(pdf_path)

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_docling_worker.py")
    # The payload travels in a file, never on stdout. TableFormer's post-processor and the weight
    # loader both print to stdout through channels that `logging`/`warnings` suppression does not
    # reach, so a document with ragged cells -- the kind this exists to recover -- emitted warnings
    # ahead of the JSON and was parsed as zero tables.
    # Resolved BEFORE the payload directory is created. Language detection renders a page, shells
    # out to tesseract and calls a model, so it has several ways to fail; done after the mkdtemp
    # but before the try/finally that removes it, any failure leaked one temporary directory per
    # document -- and it did fail, on the missing import above.
    ocr_arguments: List[str] = []
    if use_ocr:
        binary = tesseract_command()
        if not binary:
            # Asked for OCR and cannot honour it. Saying so beats converting without OCR and
            # returning a confidently empty result for a scanned page.
            logger.warning(f"OCR requested for {pdf_path} but no tesseract binary was found in "
                           f"{TESSERACT_ENV}; converting without OCR.")
        else:
            languages = ocr_languages_for(pdf_path)
            ocr_arguments = ["--ocr", "--tesseract-cmd", binary, "--ocr-lang", languages]
            logger.debug(f"OCR-ing {os.path.basename(pdf_path)} with tesseract langs={languages}")

    # The cache is consulted BEFORE any subprocess is started, so a hit costs one file read
    # instead of a conversion. The language string is part of the key, which is why it is resolved
    # above this point.
    languages_key = ""
    for i, token in enumerate(ocr_arguments):
        if token == "--ocr-lang" and i + 1 < len(ocr_arguments):
            languages_key = ocr_arguments[i + 1]
    cache_path = _cache_path(pdf_path, bool(ocr_arguments), languages_key)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            logger.debug(f"docling cache hit for {os.path.basename(pdf_path)}")
            return cached
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"ignoring unreadable docling cache entry {cache_path}: {e}")

    import tempfile
    payload_dir = tempfile.mkdtemp(prefix="docling-tables-")
    payload_path = os.path.join(payload_dir, "tables.json")
    # The thread count goes to the worker too, not just to the BLAS environment: docling applies
    # it through AcceleratorOptions, which is what actually bounds the table model.
    command = (interpreter + [worker, pdf_path, "--out", payload_path,
                              "--num-threads", str(_threads_per_conversion())] + ocr_arguments)

    import shutil
    environment = dict(os.environ, **_subprocess_env())
    try:
        # The semaphore is held for the conversion itself, not for the whole function: callers are
        # extraction worker THREADS, so without it every worker would start its own torch process.
        with _CONVERSION_SLOTS:
            subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True,
                           env=environment)
        try:
            with open(payload_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            logger.warning(f"docling wrote no payload for {pdf_path}")
            return {}
        except json.JSONDecodeError as exc:
            logger.warning(f"docling returned unparseable output for {pdf_path}: {exc}")
            return {}
    except subprocess.TimeoutExpired:
        logger.warning(f"docling timed out after {timeout}s on {pdf_path}")
        return {}
    except Exception as exc:                                    # noqa: BLE001
        from sample.peptide_predictors import _log_subprocess_failure
        _log_subprocess_failure("docling", exc)
        return {}
    finally:
        shutil.rmtree(payload_dir, ignore_errors=True)

    if "error" in payload:
        logger.warning(f"docling failed on {pdf_path}: {payload['error']}")
        return {}

    # Cached only on a clean conversion. A payload produced by a timeout or a crashed subprocess
    # would otherwise be frozen in as that PDF's permanent result.
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            temporary = f"{cache_path}.partial"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(temporary, cache_path)
        except OSError as e:
            logger.debug(f"could not cache docling payload for {pdf_path}: {e}")
    return payload


def docling_tables(pdf_path: str, use_ocr: Optional[bool] = None,
                   timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """
    The tables docling finds in one PDF, as cell grids.

    :param pdf_path: The PDF to convert.
    :param use_ocr: Force OCR on or off; None consults :data:`OCR_MODE`.
    :param timeout: Seconds before the subprocess is abandoned.
    :returns: One dict per table with ``page_no``, ``num_rows``, ``num_cols`` and ``cells``. Empty
        on any failure.
    """
    return docling_document(pdf_path, use_ocr=use_ocr, timeout=timeout).get("tables", []) or []


def docling_text(pdf_path: str, use_ocr: Optional[bool] = None,
                 timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """
    The document's text as docling reads it -- OCR output when the page had no text layer.

    This is what the LangExtract backend should read in preference to the pdfplumber ``.txt``. For
    a scanned PDF the pdfplumber text is empty and this is the only readable content that exists.

    :param pdf_path: The PDF to convert.
    :param use_ocr: Force OCR on or off; None consults :data:`OCR_MODE`.
    :param timeout: Seconds before the subprocess is abandoned.
    :returns: The text, or None when the conversion failed or produced nothing -- so the caller can
        tell "docling could not read this" from "docling read an empty document".
    """
    text = (docling_document(pdf_path, use_ocr=use_ocr, timeout=timeout).get("text") or "").strip()
    return text or None


def _sequence_visible_in(sequence: Optional[str], window: str) -> bool:
    """
    Whether a sequence is findable in a context window, by the same rule the extraction gate uses.

    Mirrors :func:`sample.document_quality.sequence_supported_by_context`'s test -- a literal match,
    then a match after whitespace and hyphens are removed -- so that a window this module stores is
    one that gate will accept. A record with no sequence has nothing to find and keeps its window.

    :param sequence: The record's one-letter sequence, if it has one.
    :param window: The candidate context window.
    :returns: True when the window genuinely supports the sequence.
    """
    if not sequence:
        return True
    seq, ctx = sequence.strip().upper(), (window or "").strip().upper()
    if not ctx:
        return True
    return seq in ctx or seq in re.sub(r"[\s\-‐-―]+", "", ctx)


def _grid(table: Dict[str, Any]) -> Tuple[List[List[str]], List[List[str]], int]:
    """
    Lay a table's cells out as a dense grid, separating header rows from body rows.

    Spans are expanded rather than collapsed: a header cell spanning four columns names all four,
    which is what makes a two-row header readable ("MIC ( uM)" over the columns, peptide names
    beneath).

    :param table: One entry from :func:`docling_tables`.
    :returns: ``(header_rows, body_rows, header_row_count)``.
    """
    cells = table.get("cells") or []
    if not cells:
        return [], [], 0

    rows = max((c.get("r1") or 0) for c in cells)
    cols = max((c.get("c1") or 0) for c in cells)
    if rows <= 0 or cols <= 0:
        return [], [], 0

    grid = [["" for _ in range(cols)] for _ in range(rows)]
    header_depth = 0
    for cell in cells:
        r0, r1 = cell.get("r0") or 0, cell.get("r1") or 0
        c0, c1 = cell.get("c0") or 0, cell.get("c1") or 0
        text = cell.get("text", "")
        if cell.get("header"):
            header_depth = max(header_depth, r1)
        for r in range(max(0, r0), min(rows, r1)):
            for c in range(max(0, c0), min(cols, c1)):
                if not grid[r][c]:
                    grid[r][c] = text
    return grid[:header_depth], grid[header_depth:], header_depth


def _column_labels(header_rows: Sequence[Sequence[str]], columns: int) -> List[str]:
    """
    One joined label per column, built from every header row.

    :param header_rows: The table's header rows.
    :param columns: Column count.
    :returns: Labels such as ``"MIC ( uM) TL"`` -- the assay from the upper row and the sample from
        the lower, which is the pair the caller needs to tell a unit from a peptide name.
    """
    labels = []
    for index in range(columns):
        parts = [row[index] for row in header_rows if index < len(row) and row[index]]
        seen, unique = set(), []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique.append(part)
        labels.append(" ".join(unique).strip())
    return labels


def is_transposed(header_rows: Sequence[Sequence[str]], body_rows: Sequence[Sequence[str]]) -> bool:
    """
    Whether the table runs peptides across its columns instead of down its rows.

    Assay tables in this literature routinely put one peptide per column and one organism per row,
    under a spanning "MIC (uM)" header. Read as though the columns were organisms -- which is what
    ``pdf_tables`` assumes -- every value is attributed to the wrong entity.

    :param header_rows: The table's header rows.
    :param body_rows: The table's body rows.
    :returns: True when the layout is peptide-per-column.
    """
    if len(header_rows) < 2 or not body_rows:
        return False
    # An assay named once across the top, with per-column sub-labels beneath it.
    top = " ".join(part for part in header_rows[0] if part)
    if not _ASSAY_HEADER.search(top):
        return False
    distinct_sub_labels = {cell for cell in header_rows[1][1:] if cell}
    if len(distinct_sub_labels) < 2:
        return False
    # ...and no column that announces itself as holding sequences, which would mean the ordinary
    # layout with an assay spanning the remaining columns.
    labels = _column_labels(header_rows, len(header_rows[0]))
    return not any(_SEQUENCE_HEADER.search(label) for label in labels)


def _activity(label: str, value: str, default_unit: Optional[str],
              known_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    One activity record from a labelled cell, or None when the cell holds no measurement.

    :param label: The column (or row) label the value sits under.
    :param value: The raw cell text.
    :param default_unit: Unit inherited from a spanning header, used when the label has none.
    :param known_type: The assay, when the caller already knows it from a spanning header. It must
        be passed in rather than assigned to the result afterwards: ``target_group`` is derived
        from the assay, so overwriting ``activity_type`` after this function returns leaves the
        group derived from the wrong assay. That is a real defect this had -- a dose-response
        hemolysis row is labelled with a concentration ("3"), which names no assay, so the type
        defaulted to MIC and the record came out as ``antibacterial`` hemolysis.
    :returns: The activity dict, or None for ``"n.d."``, ``"-"`` and other non-measurements.
    """
    from sample.nlp import infer_activity_group, parse_activity_value_fast

    text = str(value or "").strip()
    if not text:
        return None
    operator, low, high = parse_activity_value_fast(text)
    if low is None and high is None:
        # "n.d.", "ND", "-", "NA" all land here. Recording them as measurements would invent data;
        # pdf_tables keeps any cell merely containing a digit, which admits footnote markers.
        return None

    activity_type = known_type or activity_type_from_header(label) or "MIC"
    unit = unit_from_header(label) or default_unit
    if not unit:
        # No unit is declared anywhere for this column, so there is no measurement to record.
        #
        # The earlier version of this function omitted the field instead. That looked like the
        # honest choice -- better than pdf_tables' hardcoded "ug/mL", which reports micromolar
        # values as micrograms per millilitre -- but it is not what the schema means by a missing
        # unit: strict_schema.normalize_unit(None) returns "", and "" is absent from
        # units_for_activity() for EVERY assay, so each such record was silently dropped by
        # enforce_activity. Measured against 4,000 existing extractions, no surviving activity has
        # an absent or empty unit; they all carry a concrete one.
        #
        # Dropping the cell here is the same outcome the schema would produce, but it happens where
        # the reason is visible rather than three stages downstream.
        return None

    organism = re.sub(r"\s*\([^)]*\)", "", label).strip()
    # Strain designations never resolve against the taxonomy -- "Escherichia coli D21",
    # "Staphylococcus aureus ATCC 25923" are both rejected outright -- and clade headings like
    # "Gram-positive bacteria" or "Yeasts" are not organisms at all. resolve_target_species reduces
    # the first to its binomial and returns "" for the second, which is also what keeps a table's
    # section-heading rows from being recorded as species. Title-casing, which this function did
    # before, breaks the binomial ("Escherichia Coli") and is left to that helper instead.
    try:
        from sample.database_activities import resolve_target_species
        organism = resolve_target_species(organism)
    except Exception as exc:                                    # noqa: BLE001
        logger.debug(f"Could not resolve target species {organism!r}: {exc}")

    # An assay that permits exactly one class implies it, and must not have one inferred from its
    # surroundings. Hemolysis is the case that matters here: in a dose-response table the row label
    # is a concentration ("3"), so inferring from it produced `antibacterial` on a hemolysis
    # measurement. validation_rules.derivable_target_group records that mislabelling hemolysis this
    # way previously caused real data to be rejected as contradictory.
    from sample.validation_rules import derivable_target_group
    group = derivable_target_group(activity_type) or infer_activity_group(organism or label,
                                                                          activity_type)
    record: Dict[str, Any] = {
        # Both names are set, and they are NOT duplicates of each other. `target_group` is what the
        # extraction-time checks read: strict_schema derives it per assay and uses it to catch
        # contradictions such as an antibacterial result reported against a fungus. Left unset --
        # as this function did at first -- that check is silently inert on every record here, which
        # is worse than a wrong value because nothing reports it. `activity_group` is the persisted
        # column (sample/models.py, the activities table in sample/serialize.py, its index, and the
        # `activity_group=` filter the search API and web UI expose), so dropping it in favour of
        # the other name would break serialisation and search.
        "activity_group": group,
        "target_group": group,
        "target_organism": organism or None,
        "activity_type": activity_type,
        "value_raw": text,
        "operator": operator,
        "value_lower": low,
        "value_upper": high,
        "activity_unit": unit,
    }
    return record


def records_from_table(table: Dict[str, Any], page_text: str = "") -> List[Dict[str, Any]]:
    """
    Peptide records from one docling table.

    :param table: One entry from :func:`docling_tables`.
    :param page_text: Text of the page the table sits on, for the context window. A record whose
        context does not contain its own sequence is deleted by the extraction pipeline's third
        gate, so this should be the real page text whenever it is available.
    :returns: Records shaped like :mod:`sample.pdf_tables`' output, so both backends feed the same
        downstream code.
    """
    from sample.nlp import extract_context_window

    header_rows, body_rows, _ = _grid(table)
    if not body_rows:
        return []
    columns = max(len(row) for row in body_rows)
    labels = _column_labels(header_rows, columns) if header_rows else [""] * columns
    page_no = table.get("page_no")

    def _finish(name: Optional[str], sequence: Optional[str],
                activities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # A sequence with no activities is still worth keeping: sequence tables ("peptide" /
        # "sequence") carry no measurements, and they are what supplies the sequence for the assay
        # tables' name-only records during the document-level join below.
        if not (sequence or name):
            return None
        if not activities and not sequence:
            return None
        target = sequence or name
        # Context is stored ONLY when the sequence is actually findable in it.
        #
        # document_quality.sequence_supported_by_context (the extraction pipeline's third gate, and
        # its only irreversible deletion) asks whether a sequence appears in its own recorded
        # window, and deletes it when it does not. That gate is right to do so -- it exists to
        # remove residue runs a regex captured from prose -- but it cannot tell such a hit from a
        # sequence this module converted out of three-letter code. Papers print
        # "H-Phe 1 -Leu 2 - ..."; the one-letter FLPLIGRVLSGIL appears nowhere on the page, not even
        # after whitespace and hyphens are squashed, so a page-derived window fails the check and
        # every table record was deleted (measured: 0 of 4 survived).
        #
        # The gate's own contract resolves this: it returns True for a record with no context at
        # all, naming "a database import, a table parser, or an LLM repair" as sources that are not
        # text-derived. A table cell is not a prose window, so claiming one is a misrepresentation.
        # The alternative -- appending the converted sequence to the stored context so the check
        # passes -- would fabricate the evidence the gate is asking for, which is worse than storing
        # none. Only llm_validator reads this field, and it already falls back to the activity's own
        # context_sentence.
        window = (extract_context_window(page_text, target_term=target, token_window=1000)
                  if page_text else "")
        if window and not _sequence_visible_in(sequence, window):
            window = ""
        return {
            "peptide_name": name or None,
            "peptide_sequence": sequence or None,
            "biological_activities": activities,
            "context": window,
            "source_page": page_no,
        }

    def _adjudicate_units(built: List[Dict[str, Any]]) -> None:
        """
        Ask the local model whether any picogram unit here is really a misread micro sign.

        Runs on the assembled records, where the page's text is in scope -- the judgement depends on
        the surrounding evidence, which the per-cell builder never sees. Only activities whose unit
        is actually ambiguous reach the model, so a document with no picogram units costs nothing.

        :param built: Records to adjudicate in place.
        :returns: ``None``. Records are amended in place, keeping the printed unit alongside the
            verdict so the call can be audited and reversed.
        """
        if not page_text:
            return
        try:
            from sample.llm_validator import adjudicate_picogram_unit, is_ambiguous_mass_unit
        except Exception as exc:                                # noqa: BLE001
            logger.debug(f"Unit adjudication unavailable: {exc}")
            return
        for record in built:
            for activity in record.get("biological_activities") or []:
                printed = activity.get("activity_unit") or ""
                if not is_ambiguous_mass_unit(printed):
                    continue
                decided, reason = adjudicate_picogram_unit(
                    printed, activity.get("activity_type") or "", activity.get("value_raw") or "",
                    page_text)
                if decided != printed:
                    # Both are kept: a consumer can see what the page said and why it was
                    # overridden, which a silent rewrite would destroy.
                    activity["activity_unit"] = decided
                    activity["activity_unit_original"] = printed
                    activity["unit_adjudication"] = reason

    records: List[Dict[str, Any]] = []

    if is_transposed(header_rows, body_rows):
        # Peptides across the columns, organisms (or doses) down the rows.
        top_header = " ".join(p for p in header_rows[0] if p)
        spanning_unit = unit_from_header(top_header)
        spanning_type = activity_type_from_header(top_header)
        # When the row-label column declares a concentration, its labels are DOSES, not organisms:
        # a hemolysis table reads "peptide concn ( uM)" down the side with 3/6/12/24 in the cells.
        # Recording "3" as a target organism is the same category error as stamping every value
        # with a guessed unit, so the organism is left unset and the dose is kept as the label.
        row_labels_are_doses = unit_from_header(labels[0] if labels else "") is not None
        for column in range(1, columns):
            name = header_rows[-1][column] if column < len(header_rows[-1]) else ""
            if not name:
                continue
            activities = []
            for row in body_rows:
                row_label = row[0] if row else ""
                if not row_label or column >= len(row):
                    continue
                # A dose-response column carries no unit of its own; its values are percentages and
                # the unit is supplied below. Passing "%" here keeps the cell from being discarded
                # as unit-less before it reaches that point.
                inherited = "%" if row_labels_are_doses else spanning_unit
                activity = _activity(row_label, row[column], inherited, known_type=spanning_type)
                if not activity:
                    continue
                if row_labels_are_doses:
                    # The row label is the concentration the response was measured AT, not a
                    # species. A dose-response column reports a percentage, which is the unit the
                    # assay is actually in -- the row's own uM belongs to the dose.
                    activity["target_organism"] = None
                    activity["activity_unit"] = "%"
                    activity["dose_raw"] = row_label
                    activity["dose_unit"] = unit_from_header(labels[0])
                activities.append(activity)
            sequence = sequence_from_cell(name)
            record = _finish(None if sequence else name, sequence, activities)
            if record:
                records.append(record)
        # Both return paths adjudicate. This branch produces most of the corpus's assay activities,
        # so wiring only the ordinary layout below would leave the judge running on the minority of
        # tables while appearing to be enabled.
        _adjudicate_units(records)
        return records

    # Ordinary layout: one peptide per row.
    sequence_column = next((i for i, l in enumerate(labels) if _SEQUENCE_HEADER.search(l)), -1)
    name_column = next((i for i, l in enumerate(labels)
                        if _NAME_HEADER.search(l) and i != sequence_column), -1)
    for row in body_rows:
        sequence = None
        if 0 <= sequence_column < len(row):
            sequence = sequence_from_cell(row[sequence_column])
        if not sequence:
            # Fall back to scanning, but only with the strict gate -- this is where organism names
            # and assay values would otherwise be captured as sequences.
            for index, cell in enumerate(row):
                if index == name_column:
                    continue
                sequence = sequence_from_cell(cell)
                if sequence:
                    break
        name = row[name_column].strip() if 0 <= name_column < len(row) else ""

        activities = []
        for index, cell in enumerate(row):
            if index in (sequence_column, name_column) or not cell:
                continue
            label = labels[index] if index < len(labels) else ""
            if not label:
                continue
            activity = _activity(label, cell, None)
            if activity:
                activities.append(activity)
        record = _finish(name, sequence, activities)
        if record:
            records.append(record)
    _adjudicate_units(records)
    return records


def extract_tables_from_pdf(pdf_path: str, use_ocr: Optional[bool] = None,
                            timeout: int = DEFAULT_TIMEOUT,
                            tables: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Peptide activity records from every table in a PDF, via docling.

    Drop-in alternative to ``pdf_tables.extract_tables_from_pdf`` with the same record shape.

    :param pdf_path: The PDF to read.
    :param use_ocr: Force OCR on or off; None consults :data:`OCR_MODE`.
    :param timeout: Seconds before the conversion is abandoned.
    :param tables: Tables from an earlier :func:`docling_document` call on this same PDF. Passing
        them skips the conversion, which is the point: on a scanned document the conversion is the
        expensive part (~12 s/page measured), so a caller that also wants the document's text must
        be able to pay for it once rather than converting the same PDF twice.
    :returns: Records, or an empty list when docling is unavailable or finds nothing.
    """
    if tables is None:
        tables = docling_tables(pdf_path, use_ocr=use_ocr, timeout=timeout)
    if not tables:
        return []

    pages = _page_texts(pdf_path, {t.get("page_no") for t in tables if t.get("page_no")})
    records: List[Dict[str, Any]] = []
    for table in tables:
        records.extend(records_from_table(table, pages.get(table.get("page_no"), "")))
    return merge_document_records(records)


def merge_document_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fold one document's per-table records into one record per peptide.

    Two things happen here, both of which need the whole document rather than a single table:

    * **The name-to-sequence join.** A paper states its sequences once, in a "peptide"/"sequence"
      table, and then refers to those peptides by name in every assay table. Extracted per table,
      the MIC records carry a name and no sequence while the sequence sits unused in another table.
      Joining them is what gives those measurements a sequence at all.
    * **Deduplication.** The same peptide appears in the MIC table and again in the hemolysis
      table, producing two records with disjoint activities where the document describes one
      peptide.

    :param records: Records from every table in one document.
    :returns: One record per peptide, activities concatenated.
    """
    by_name: Dict[str, str] = {}
    for record in records:
        name, sequence = record.get("peptide_name"), record.get("peptide_sequence")
        if name and sequence:
            by_name.setdefault(name.strip().lower(), sequence)

    merged: Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    #: Which merged records have already taken their page from a measurement-bearing table.
    _had_activities: Dict[int, bool] = {}
    for record in records:
        name = record.get("peptide_name")
        if not record.get("peptide_sequence") and name:
            record["peptide_sequence"] = by_name.get(name.strip().lower())
        # The context check belongs HERE, after the join, not where the record was built. An assay
        # table's record has no sequence of its own -- it gains one from the sequence table above --
        # so a check run at construction time sees `sequence=None`, concludes there is nothing to
        # find, and keeps a window the joined sequence does not appear in. Every record then failed
        # the extraction gate (measured: 0 of 4) despite the guard being present.
        if not _sequence_visible_in(record.get("peptide_sequence"), record.get("context") or ""):
            record["context"] = ""
        key = (record.get("peptide_sequence") or None,
               (name or "").strip().lower() or None)
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            ordered.append(record)
            continue
        existing["biological_activities"].extend(record.get("biological_activities") or [])
        if not existing.get("context"):
            existing["context"] = record.get("context", "")
        # Provenance follows the measurements, not the sequence. The sequence table is typically on
        # an early page and the assay tables several pages later, and the first record seen wins the
        # merge -- so without this a record cites page 1 while every value in it was read from page
        # 6. Only overwrite when this record is the one contributing measurements.
        if record.get("biological_activities") and not _had_activities.get(id(existing)):
            existing["source_page"] = record.get("source_page")
            _had_activities[id(existing)] = True

    # A record that never gained an activity and only existed to supply a sequence has done its
    # job; emitting it would add a peptide with no measurement to the extraction.
    return [r for r in ordered if r.get("biological_activities")]


def _page_texts(pdf_path: str, wanted: Sequence[Any]) -> Dict[int, str]:
    """
    Text of the pages that actually carry tables.

    :param pdf_path: The PDF to read.
    :param wanted: 1-based page numbers to fetch.
    :returns: Page number to text. Empty when PyMuPDF is unavailable; the caller then emits records
        with an empty context rather than none at all.
    """
    if not wanted:
        return {}
    try:
        import fitz
    except ImportError:
        return {}
    texts: Dict[int, str] = {}
    try:
        with fitz.open(pdf_path) as document:
            for page_no in wanted:
                if isinstance(page_no, int) and 1 <= page_no <= len(document):
                    texts[page_no] = document[page_no - 1].get_text()
    except Exception as exc:                                    # noqa: BLE001
        logger.debug(f"Could not read page text from {pdf_path}: {exc}")
    return texts
