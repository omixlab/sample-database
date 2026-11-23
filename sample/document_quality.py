"""
Document-level gates that run BEFORE peptide extraction: is this document worth extracting from,
and is it the kind of document whose peptides mean what the pipeline assumes they mean?

**Why this exists.** Both checks here were written against measured failures in the manual-curation
audit (`data/manual_curation`, step 15), not anticipated ones:

* **Anti-bot interstitials.** 2.3% of the literature text corpus -- roughly 940 of 41,061 documents
  -- are Cloudflare "Are you a robot?" / CAPTCHA pages saved as though they were papers. They carry
  primary text, pass every structural check the pipeline has, and yield no peptides. That last part
  is why they are dangerous rather than merely useless: an empty extraction from a junk page is
  indistinguishable from a correct empty extraction from a real paper, so they silently inflate
  apparent recall and pad the negative class of any evaluation drawn from "documents we found
  nothing in".

* **Review articles.** A review's peptides are *other people's* results, summarised. The extractor
  has no way to tell "this paper measured MIC 4 uM" from "this review reports that someone else
  measured MIC 4 uM", so it records the latter as though the review had produced it. In the first
  audit run, a single review article produced 9 of the 10 false-positive peptides -- every one of
  them real, published, and simply not a finding of the document it was extracted from.

Neither check is a classifier in any statistical sense. Both are deliberately conservative keyword
and structure tests over text the pipeline already has, chosen so that a false *positive* (calling a
real paper junk, or a research article a review) is much less likely than a miss. A miss leaves
today's behaviour unchanged; a false positive would silently discard real data, which is the worse
failure and the reason nothing here drops a document on its own -- callers are handed a reason and
decide.
"""
import re
from typing import Any, Optional, Tuple

#: Phrases that appear in bot-check interstitials and essentially never in a research article's
#: text. Matched case-insensitively against the head of the document only: a real paper discussing
#: CAPTCHA research would mention these in its body, not in its first page.
#:
#: MEASURED, not guessed. An earlier version of this list carried eleven markers written from what
#: Cloudflare pages typically say; validated against 4,000 real corpus documents, only these three
#: ever fired (captcha 91, verify-you-are-human 90, are-you-a-robot 84). The other eight --
#: "checking your browser", "just a moment...", "ddos protection by cloudflare", "access denied",
#: "your request has been blocked", "unusual traffic from your computer", "enable javascript and
#: cookies to continue", "please complete the security check" -- matched nothing at all and were
#: removed rather than left in to imply coverage this project has not demonstrated.
#:
#: Re-validate this list against any newly acquired corpus: a different publisher's interstitial
#: will say something different, and a marker list is only as good as the pages it was checked
#: against. The length and watermark gates below catch most novel interstitials anyway, which is
#: why removing the unvalidated eight is safe.
_UNUSABLE_MARKERS: Tuple[str, ...] = (
    "captcha",
    "verify you are human",
    "are you a robot",
)

#: How much of the document the marker scan reads. An interstitial is the WHOLE document, so its
#: markers are at the very top; bounding the scan keeps a passing mention deep inside a real paper
#: from tripping the check.
_MARKER_SCAN_CHARS: int = 4000

#: Below this many characters a document cannot support extraction OR support the claim that there
#: was nothing to extract. Measured: 0.7% of the corpus falls here, and those files are download
#: stubs and error pages rather than short papers.
_MIN_USABLE_CHARS: int = 500


#: A watermark line is a short repeated stamp, not a paragraph. Bounding the length stops a
#: document with one genuinely repeated long passage from being called a watermark.
_MAX_WATERMARK_LINE_CHARS: int = 120

#: Share of a document's non-blank lines that must be the SAME line before it is called
#: watermark-only, and how few distinct lines it may contain. Both must hold: a real paper repeats
#: its running header once per page (a few percent of its lines, among hundreds of distinct ones),
#: which is nowhere near either bound.
_WATERMARK_LINE_SHARE: float = 0.6
_MAX_WATERMARK_DISTINCT_LINES: int = 3


def _is_watermark_only(text: str) -> bool:
    """
    Whether a document's text is nothing but a repeated publisher stamp.

    Found in production rather than anticipated: the first extraction run under these gates skipped
    documents whose entire text was "NII-Electronic Library Service" repeated, or "(c) 1989 Nature
    Publishing Group" repeated, or a DSpace cookie notice. The PDF existed and the converter ran,
    but it recovered only the page watermark -- so this is a signature of a FAILED PDF-to-text
    conversion, not of a short paper, and it is worth its own reason so the class can be counted
    and the affected PDFs re-converted rather than written off.

    :param text: The document's full primary text.
    :returns: Whether the text is dominated by one short repeated line.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    counts: dict = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    top_line, top_count = max(counts.items(), key=lambda kv: kv[1])
    return (len(counts) <= _MAX_WATERMARK_DISTINCT_LINES
            and len(top_line) <= _MAX_WATERMARK_LINE_CHARS
            and top_count / len(lines) >= _WATERMARK_LINE_SHARE)


def is_unusable_document(text: str) -> Tuple[bool, Optional[str]]:
    """
    Decides whether a document is an anti-bot interstitial, a failed conversion, or too short.

    :param text: The document's full primary text.
    :returns: ``(True, reason)`` when the document should not be treated as a paper, else
        ``(False, None)``. The reason is a short stable slug (``captcha_or_bot_check``,
        ``watermark_only``, ``too_short``, ``empty``) suitable for recording in an output record
        and counting later -- a caller that drops these silently loses the ability to distinguish
        "the pipeline found nothing" from "there was nothing to find", and the reasons differ in
        what they imply: a CAPTCHA page needs re-downloading, a watermark-only page needs
        re-converting, and a genuinely short document needs neither.
    """
    if not text or not text.strip():
        return True, "empty"
    head = text[:_MARKER_SCAN_CHARS].lower()
    for marker in _UNUSABLE_MARKERS:
        if marker in head:
            return True, "captcha_or_bot_check"
    # Checked BEFORE the length floor so a long watermark-only document -- which the floor cannot
    # catch, and which currently reaches the extractor -- gets the reason that explains it.
    if _is_watermark_only(text):
        return True, "watermark_only"
    if len(text.strip()) < _MIN_USABLE_CHARS:
        return True, "too_short"
    return False, None


def sequence_supported_by_context(sequence: str, context: str) -> bool:
    """
    Whether an extracted sequence actually appears in the text window it was extracted from.

    Written against a measured failure: a record carrying ``peptide_sequence="RWQWR"`` with an
    EMPTY name, whose stored ``context`` did not contain "RWQWR" anywhere -- while the paper's real
    peptide was the palindrome "RWQWRWQWR", printed in a table the extractor never read. A residue
    run captured from running prose, with no name and no support in its own context window, is the
    signature of a regex hit rather than a reported peptide.

    The check is intentionally weak: it asks only that the sequence be findable in the window the
    extractor itself recorded, which any genuinely text-derived sequence satisfies by construction.
    It therefore cannot penalise a correct extraction, and it says nothing about sequences that
    have no context at all (from a database import, a table parser, or an LLM repair) -- those are
    not text-derived and are reported as supported.

    :param sequence: The extracted one-letter sequence.
    :param context: The stored context window for that peptide.
    :returns: ``False`` only when there IS a context and the sequence is genuinely absent from it.
    """
    seq = (sequence or "").strip().upper()
    ctx = (context or "").strip().upper()
    if not seq or not ctx:
        return True
    if seq in ctx:
        return True
    # Sequences are frequently printed with internal whitespace, hyphens or line breaks
    # ("RWQWR WQWR", "RW-QWR"), which is a formatting artifact rather than a different sequence.
    return seq in re.sub(r"[\s\-‐-―]+", "", ctx)


#: Explicit self-identification, which journals print as a running header. This is the strongest
#: signal available and is checked first: a document that calls itself a review is one.
_REVIEW_HEADERS: Tuple[str, ...] = (
    "review article",
    "a review",
    "systematic review",
    "mini-review",
    "minireview",
    "critical review",
    "narrative review",
    "literature review",
)

#: First-person framing a review uses about its own purpose. Requires more than one hit (see
#: :func:`looks_like_review_article`) because a research paper's introduction can legitimately say
#: "we review the literature on X" in a single sentence.
_REVIEW_PHRASES: Tuple[str, ...] = (
    "in this review",
    "this review summarizes",
    "this review summarises",
    "this review focuses",
    "this review discusses",
    "here we review",
    "we review the",
    "the present review",
    "this review article",
    "the aim of this review",
    "reviewed here",
)

#: Section headings a primary research article has and a review usually does not. Their ABSENCE is
#: corroborating evidence, never proof on its own -- section headings survive PDF text extraction
#: unreliably, so this is only ever used to raise confidence in a document that already looks like
#: a review by its own wording.
_RESEARCH_SECTIONS: Tuple[str, ...] = (
    "materials and methods",
    "experimental section",
    "methods",
    "statistical analysis",
    "results and discussion",
)

#: How far into the document to look for a self-identifying header.
_HEADER_SCAN_CHARS: int = 3000


def looks_like_review_article(text: str) -> Tuple[bool, Optional[str]]:
    """
    Decides whether a document is a review rather than primary research.

    The question this answers is not "is this about peptides" but "does a peptide reported here
    belong to this document?" In a review it does not: the measurements are quoted from the primary
    literature, so extracting them attributes other people's results to this paper and, worse,
    records them as though this paper were their source.

    Deliberately conservative. A self-identifying header is accepted on its own; otherwise two
    independent review phrasings are required, and the absence of research sections only ever
    corroborates. The cost of a miss is today's behaviour; the cost of a false positive is
    discarding a real research paper's peptides.

    :param text: The document's full primary text.
    :returns: ``(True, evidence)`` when the document looks like a review, else ``(False, None)``.
        ``evidence`` quotes the phrase that decided it, so a disputed call can be checked against
        the document rather than argued about.
    """
    if not text or not text.strip():
        return False, None

    header = text[:_HEADER_SCAN_CHARS].lower()
    for marker in _REVIEW_HEADERS:
        # Word-bounded: "a review" must not match "a reviewer", and "review article" should not
        # match inside "peer-review articles".
        if re.search(rf"\b{re.escape(marker)}\b", header):
            return True, f"header: {marker!r} in the first {_HEADER_SCAN_CHARS} characters"

    lowered = text.lower()

    # Count DISTINCT, NON-OVERLAPPING occurrences, not phrase hits. Several entries in
    # _REVIEW_PHRASES are substrings of one another by design ("here we review" and "we review the"
    # both match "here we review the literature"), so counting hits treated ONE occurrence as two
    # independent signals and fired the two-signal rule on a single sentence -- which is how a
    # primary research paper whose introduction says "here we review the literature" got classified
    # as a review. That is the expensive direction of error: it discards a real paper's own
    # measured peptides.
    spans = []
    for phrase in _REVIEW_PHRASES:
        start = lowered.find(phrase)
        while start != -1:
            spans.append((start, start + len(phrase), phrase))
            start = lowered.find(phrase, start + 1)
    spans.sort()
    occurrences = []
    for start, end, phrase in spans:
        if occurrences and start < occurrences[-1][1]:
            # Overlaps the previous occurrence: the same sentence, matched twice.
            occurrences[-1] = (occurrences[-1][0], max(occurrences[-1][1], end),
                               occurrences[-1][2])
            continue
        occurrences.append((start, end, phrase))

    if len(occurrences) >= 2:
        return True, f"phrasing: {', '.join(repr(o[2]) for o in occurrences[:3])}"
    if occurrences and not any(section in lowered for section in _RESEARCH_SECTIONS):
        return True, (f"phrasing: {occurrences[0][2]!r}, and no primary-research section heading "
                      f"({'/'.join(_RESEARCH_SECTIONS[:3])}) is present")
    return False, None


#: Discovery paths whose values were read out of the document TEXT, and can therefore be checked
#: against it. ``table_match`` is deliberately absent: those values come from the PDF via docling,
#: so a number that is genuinely in a table need not appear in the pdfplumber ``.txt`` at all, and
#: grounding them against the text would delete correct records.
_GROUNDABLE_DISCOVERY_PATHS = frozenset({"langextract_match", "sequence_match"})

#: A number as written in a value field: '12.5', '0,5', '164'.
_VALUE_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _number_appears_in(number: str, text: str) -> bool:
    """
    :param number: A numeric token from an extracted value.
    :param text: The document text.
    :returns: Whether the number can be found, allowing for how OCR renders decimals.

    Tolerant on the decimal separator and on spacing around it, because the same value is written
    '12.5', '12,5' and '12 . 5' across this corpus, and a strict match would reject a correct
    extraction because of the typesetting it was read from.
    """
    if number in text:
        return True
    swapped = number.replace(".", ",") if "." in number else number.replace(",", ".")
    if swapped != number and swapped in text:
        return True
    # '12.5' -> r'12\s*[.,]\s*5'
    parts = re.split(r"[.,]", number)
    if len(parts) == 2:
        spaced = re.compile(re.escape(parts[0]) + r"\s*[.,]\s*" + re.escape(parts[1]))
        if spaced.search(text):
            return True
    return False


def activity_supported_by_text(activity: Any, text: str,
                               discovery_path: str = "") -> Tuple[bool, str]:
    """
    Whether an extracted measurement's number can be found in the document it is attributed to.

    :param activity: One activity record.
    :param text: The document text the extraction ran against.
    :param discovery_path: The parent peptide's ``discovery_path``; only text-derived paths are
        checked, see :data:`_GROUNDABLE_DISCOVERY_PATHS`.
    :returns: ``(ok, reason)``; ``reason`` is ``""`` when supported.

    **This exists because the model was caught copying its own prompt into its output.** The
    LangExtract worker's few-shot example is the sentence "Peptide JG436 inhibited MCF-7 cells
    with an IC50 of 12.5 uM." Measured over one run's log, the tuple ``IC50 / 12.5 uM / MCF-7``
    was emitted for **49 different documents** -- and 38 of those 49 contain neither "12.5" nor
    "MCF" anywhere in their text, with exactly one containing both. The next most repeated
    measurement in the same corpus appeared in 3 documents, which is what genuine coincidence on
    a common value looks like. The peptide name from the same example, ``JG436``, also turns up
    as an extracted peptide.

    A fabricated measurement is the most damaging thing this pipeline can produce, because it is
    indistinguishable from a real one by inspection: a real assay type, a round value, a real
    cell line. The only cheap test that separates them is whether the number is in the paper.

    **The target is deliberately NOT checked.** Organism normalisation rewrites it on purpose --
    a paper saying "E. coli" yields ``Escherichia coli`` -- so the extracted target legitimately
    differs from the document's wording, and grounding it would reject correct records. Checking
    the value alone would have caught 40 of the 49 fabrications; adding the target would catch 46
    but at a false-positive cost that is not worth paying.

    Only the FIRST number is required. A range written "0.5 to 4" is anchored by its lower bound,
    and demanding both would reject a legitimate range whose second bound OCR mangled.
    """
    if discovery_path and discovery_path not in _GROUNDABLE_DISCOVERY_PATHS:
        return True, ""
    if not isinstance(activity, dict):
        return True, ""
    raw = str(activity.get("value_raw") or "").strip()
    if not raw:
        lower = activity.get("value_lower")
        raw = "" if lower is None else str(lower)
    if not raw:
        # An activity with no number is not a measurement; there is nothing to ground, and the
        # activity class itself ("Antibacterial") is judged elsewhere.
        return True, ""
    numbers = _VALUE_NUMBER.findall(raw)
    if not numbers:
        return True, ""
    if not _number_appears_in(numbers[0], text or ""):
        return False, "value_not_in_text"
    return True, ""
