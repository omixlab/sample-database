"""
Tests for :mod:`sample.document_quality` -- the pre-extraction document gates.

Both gates were written against measured failures in the step-14 curation audit, so the fixtures
here are the real shapes that failed: a Cloudflare interstitial saved as a paper, and a review
article whose cited peptides were extracted as though it had produced them.

The asymmetry these tests enforce is the important part. A MISS (calling junk usable, or a review
a research paper) leaves today's behaviour unchanged. A FALSE POSITIVE silently discards a real
paper's peptides, which is strictly worse -- so the conservative cases are tested at least as hard
as the positive ones.
"""
from sample import document_quality as Q


class TestIsUnusableDocument:
    def test_cloudflare_bot_check_is_unusable(self) -> None:
        text = ("Just a moment...\nwww.sciencedirect.com\nChecking your browser before accessing.\n"
                "Are you a robot? Please complete the security check to continue." + "x" * 2000)
        unusable, reason = Q.is_unusable_document(text)
        assert unusable
        assert reason == "captcha_or_bot_check"

    def test_captcha_page_is_unusable(self) -> None:
        unusable, reason = Q.is_unusable_document("CAPTCHA required to continue. " + "y" * 1000)
        assert unusable
        assert reason == "captcha_or_bot_check"

    def test_empty_and_whitespace_are_unusable(self) -> None:
        for text in ("", "   \n\t  "):
            unusable, reason = Q.is_unusable_document(text)
            assert unusable
            assert reason == "empty"

    def test_short_stub_is_unusable(self) -> None:
        unusable, reason = Q.is_unusable_document("Download failed.")
        assert unusable
        assert reason == "too_short"

    def test_a_real_paper_is_usable(self) -> None:
        text = ("Antimicrobial peptides from amphibian skin. Materials and Methods. Peptides were "
                "synthesised by Fmoc chemistry and MIC values determined by broth microdilution "
                "against Escherichia coli ATCC 25922. " * 20)
        assert Q.is_unusable_document(text) == (False, None)

    def test_a_paper_that_merely_discusses_captchas_deep_in_the_body_is_usable(self) -> None:
        # The marker scan is bounded to the head for exactly this reason: an interstitial IS the
        # whole document, so its markers are at the top. A passing mention later is not a signal.
        body = ("Antimicrobial peptide discovery. Materials and Methods. " + "z" * 5000
                + " We used a captcha to prevent automated scraping of our web server.")
        assert Q.is_unusable_document(body) == (False, None)


class TestLooksLikeReviewArticle:
    def test_self_identifying_header_is_enough(self) -> None:
        # The real failure: a journal running header reading "REVIEW ARTICLE" on page 2, whose
        # nine cited peptides were all extracted as findings of that document.
        text = "REVIEW ARTICLE\nScorpion venom antimicrobial peptides\n" + ("body text " * 400)
        is_review, evidence = Q.looks_like_review_article(text)
        assert is_review
        assert "review article" in (evidence or "").lower()

    def test_two_review_phrasings_are_enough(self) -> None:
        text = ("Peptides and their uses. In this review we consider scorpion venoms. "
                + ("filler " * 200) + " The present review is organised by peptide family.")
        is_review, evidence = Q.looks_like_review_article(text)
        assert is_review
        assert evidence

    def test_one_phrasing_without_research_sections_is_enough(self) -> None:
        text = "Antimicrobial peptides. In this review we summarise recent work. " + ("f " * 300)
        is_review, _ = Q.looks_like_review_article(text)
        assert is_review

    def test_one_phrasing_WITH_research_sections_is_not_a_review(self) -> None:
        # A primary paper may say "we review the literature" once in its introduction. Calling it
        # a review would discard its own measured peptides -- the expensive failure.
        text = ("Novel peptide X. Introduction: here we review the literature briefly. "
                "Materials and Methods: peptides were synthesised... Results and Discussion: "
                "MIC values are reported in Table 1. " + ("data " * 300))
        assert Q.looks_like_review_article(text) == (False, None)

    def test_a_plain_research_paper_is_not_a_review(self) -> None:
        text = ("Isolation of a novel defensin. Materials and Methods. Statistical analysis was "
                "performed with ANOVA. " + ("results " * 300))
        assert Q.looks_like_review_article(text) == (False, None)

    def test_reviewer_does_not_match_review(self) -> None:
        # Word-bounded matching: "a reviewer" must not trip the "a review" header rule.
        text = ("We thank a reviewer for the suggestion. Materials and Methods followed standard "
                "broth microdilution. " + ("body " * 300))
        assert Q.looks_like_review_article(text) == (False, None)

    def test_empty_text_is_not_a_review(self) -> None:
        assert Q.looks_like_review_article("") == (False, None)


class TestSequenceSupportedByContext:
    def test_the_real_failure_is_caught(self) -> None:
        # Measured in the step-14 audit: peptide_sequence "RWQWR" with an EMPTY name, whose stored
        # context does not contain it anywhere. The paper's actual peptide was the palindrome
        # RWQWRWQWR, printed in a table the extractor never read.
        context = ("s with antimicrobial (AMPs) and anticancer (APCs)activity are of interest due "
                   "to their selectivity for bacterial and cancer cells, without affecting normal "
                   "cells")
        assert not Q.sequence_supported_by_context("RWQWR", context)

    def test_a_sequence_present_in_its_context_is_supported(self) -> None:
        assert Q.sequence_supported_by_context(
            "RWQWRWQWR", "The palindromic peptide RWQWRWQWR (peptide 1) was synthesised.")

    def test_case_insensitive(self) -> None:
        assert Q.sequence_supported_by_context("rwqwr", "we used RWQWR here")

    def test_whitespace_and_hyphen_split_sequences_are_supported(self) -> None:
        # PDF text extraction breaks sequences across lines and inserts hyphens. That is a
        # formatting artifact, not a different sequence -- dropping these would be a false positive.
        assert Q.sequence_supported_by_context("RWQWRWQWR", "peptide RWQWR WQWR was tested")
        assert Q.sequence_supported_by_context("RWQWRWQWR", "peptide RWQWR-WQWR was tested")
        assert Q.sequence_supported_by_context("RWQWRWQWR", "peptide RWQWR\nWQWR was tested")

    def test_no_context_is_treated_as_supported(self) -> None:
        # A record from a database import, a table parser or an LLM repair has no context window.
        # It is not text-derived, so this check has nothing to say about it and must not drop it.
        assert Q.sequence_supported_by_context("GIGKFLHSAK", "")
        assert Q.sequence_supported_by_context("GIGKFLHSAK", None)  # type: ignore[arg-type]

    def test_no_sequence_is_treated_as_supported(self) -> None:
        # Name-only records are legitimate; this check is about sequences specifically.
        assert Q.sequence_supported_by_context("", "some context mentioning magainin")


class TestWatermarkOnlyDocuments:
    """
    A PDF whose text extraction recovered only the page watermark. Found in production during the
    first run under these gates -- the converter ran and produced output, but the output is a
    publisher stamp repeated, not a paper. Distinguished from `too_short` because the remedy
    differs: this PDF needs RE-CONVERTING, a short paper needs nothing.
    """

    def test_repeated_library_stamp_is_watermark_only(self) -> None:
        # The real shape, from data/literature_data_text: one line, repeated, and nothing else.
        text = "NII-Electronic Library Service\n" * 40
        unusable, reason = Q.is_unusable_document(text)
        assert unusable
        assert reason == "watermark_only"

    def test_repeated_copyright_stamp_is_watermark_only(self) -> None:
        text = "© 1989 Nature Publishing Group\n" * 30
        unusable, reason = Q.is_unusable_document(text)
        assert unusable
        assert reason == "watermark_only"

    def test_long_watermark_is_caught_even_above_the_length_floor(self) -> None:
        # The case the length floor CANNOT catch, and the only reason this check earns its place.
        text = "NII-Electronic Library Service\n" * 200
        assert len(text.strip()) > Q._MIN_USABLE_CHARS
        assert Q.is_unusable_document(text) == (True, "watermark_only")

    def test_a_real_paper_with_a_repeated_running_header_is_usable(self) -> None:
        # A journal repeats its running header once per page. That is a few percent of the lines,
        # among hundreds of distinct ones -- nowhere near the share or distinct-line bounds.
        lines = []
        for page in range(20):
            lines.append("J. Med. Chem. 2020, 63, 1-12")
            for i in range(40):
                lines.append(f"Page {page} line {i}: peptide MIC values against E. coli were "
                             f"determined by broth microdilution assay.")
        assert Q.is_unusable_document("\n".join(lines)) == (False, None)

    def test_a_repeated_long_paragraph_is_not_a_watermark(self) -> None:
        # Bounded by _MAX_WATERMARK_LINE_CHARS: a repeated PARAGRAPH is a different problem
        # (duplicated text), not a watermark, and must not be silently discarded as one.
        para = ("This is a long repeated paragraph of genuine scientific prose that considerably "
                "exceeds the watermark line length bound and therefore must not be treated as a "
                "publisher stamp under any circumstances whatsoever.")
        assert len(para) > Q._MAX_WATERMARK_LINE_CHARS
        assert Q.is_unusable_document((para + "\n") * 10) == (False, None)

    def test_too_few_lines_is_not_a_watermark(self) -> None:
        assert not Q._is_watermark_only("Only one line here")


class TestMarkerListIsValidated:
    def test_only_markers_measured_to_fire_are_kept(self) -> None:
        # Eight markers were removed after validating against 4,000 real corpus documents: they
        # matched nothing. Keeping them would imply coverage this project never demonstrated.
        assert set(Q._UNUSABLE_MARKERS) == {"captcha", "verify you are human", "are you a robot"}

    def test_each_surviving_marker_still_fires(self) -> None:
        for marker in Q._UNUSABLE_MARKERS:
            text = f"Access page\n{marker}\nplease continue " + ("x" * 1000)
            assert Q.is_unusable_document(text) == (True, "captcha_or_bot_check"), marker
