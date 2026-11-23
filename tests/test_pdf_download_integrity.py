"""
Tests for the download-integrity rules in :mod:`sample.pdf`.

Each of these pins a bug that recorded a FAILED download as a completed one. That failure mode is
the expensive one here: ``download_paper_pdf`` short-circuits on a file already at the destination,
so a corrupt or absent file recorded as success is never retried, and the extraction step reads
whatever is on disk. There are 43,443 orphaned PDFs in ``data/literature_data/pubmed`` alone
(65,493 files against 22,050 manifest rows), which is what this class of error looks like after the
fact.

Nothing here touches the network or Selenium.
"""
import os

import sample.pdf as P          # before pandas: see the CXXABI note in test_attempts_ledger
import pandas as pd

VALID_PDF = b"%PDF-1.4\n" + b"x" * 2000
HTML_ERROR_PAGE = b"<html><body>Access denied</body></html>" + b"y" * 2000


class TestResumeShortCircuitValidates:
    def test_a_corrupt_file_over_the_size_bound_is_not_treated_as_done(self, tmp_path) -> None:
        # The bug: the short-circuit tested os.path.getsize(...) > 1000, so any partial write or
        # error page that cleared 1000 bytes was accepted as a finished download PERMANENTLY --
        # the DOI was never retried and the corrupt file is what extraction reads.
        path = tmp_path / "paper.pdf"
        path.write_bytes(HTML_ERROR_PAGE)
        assert P.is_valid_pdf(str(path)) is False

    def test_a_real_pdf_is_treated_as_done(self, tmp_path) -> None:
        path = tmp_path / "paper.pdf"
        path.write_bytes(VALID_PDF)
        assert P.is_valid_pdf(str(path)) is True

    def test_a_truncated_pdf_is_rejected(self, tmp_path) -> None:
        # Correct magic bytes but too short to be a document: a transfer cut off early.
        path = tmp_path / "paper.pdf"
        path.write_bytes(b"%PDF-1.4\n")
        assert P.is_valid_pdf(str(path)) is False


class TestMoveIntoPlace:
    def test_it_moves_within_one_filesystem(self, tmp_path) -> None:
        source = tmp_path / "dl.pdf"
        source.write_bytes(VALID_PDF)
        destination = tmp_path / "out" / "final.pdf"
        destination.parent.mkdir()
        assert P._move_into_place(str(source), str(destination)) is True
        assert P.is_valid_pdf(str(destination))

    def test_it_survives_a_filesystem_boundary(self, monkeypatch, tmp_path) -> None:
        """
        The download directory is a TemporaryDirectory and the destination is under data/.

        ``os.replace`` cannot cross a filesystem boundary, so wherever /tmp is a separate mount --
        including the Docker setup this repo ships -- it raised EXDEV and the download was reported
        as a failure, silently, at DEBUG level.
        """
        source = tmp_path / "dl.pdf"
        source.write_bytes(VALID_PDF)
        destination = tmp_path / "out" / "final.pdf"
        destination.parent.mkdir()

        real_replace = os.replace
        state = {"n": 0}

        def replace_refusing_the_first_move(a, b):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError(18, "Invalid cross-device link")
            return real_replace(a, b)

        monkeypatch.setattr(os, "replace", replace_refusing_the_first_move)
        assert P._move_into_place(str(source), str(destination)) is True
        assert P.is_valid_pdf(str(destination))
        assert not os.path.exists(str(source)), "the scratch copy was left behind"
        assert not os.path.exists(str(destination) + ".part")

    def test_a_failure_leaves_no_partial_file_at_the_destination(self, monkeypatch,
                                                                tmp_path) -> None:
        # A partial file at the destination is the worst outcome: the resume short-circuit would
        # see it and never retry.
        source = tmp_path / "dl.pdf"
        source.write_bytes(VALID_PDF)
        destination = tmp_path / "out" / "final.pdf"
        destination.parent.mkdir()
        monkeypatch.setattr(os, "replace",
                            lambda a, b: (_ for _ in ()).throw(OSError(18, "cross-device")))
        monkeypatch.setattr(P.shutil, "copy2",
                            lambda a, b: (_ for _ in ()).throw(OSError("disk full")))
        assert P._move_into_place(str(source), str(destination)) is False
        assert not os.path.exists(str(destination))
        assert not os.path.exists(str(destination) + ".part")


class TestAttemptsLedger:
    """
    A failed DOI must be charged ONE attempt per real try, not one per manifest flush.

    The flush loop rebuilt its frame from the cumulative result list every five results and handed
    that whole frame to the failure ledger, which stamps ``attempts=1`` and then sums per DOI. So a
    failure early in a run collected an extra attempt on every later flush: measured, one failed
    download reached ``attempts=4``, and with the default ``--max-attempts 3``
    :func:`sample.pdf._exhausted_dois` then abandoned that DOI permanently after a single try.
    """

    def _row(self):
        return pd.DataFrame([{"doi": "10.1/fail", "internal_id": "a", "success": False,
                              "reason": "no open-access copy"}])

    def test_one_real_failure_is_one_attempt(self, tmp_path) -> None:
        ledger = str(tmp_path / "failed.csv")
        P._append_failure_ledger(ledger, self._row())
        assert int(pd.read_csv(ledger).set_index("doi").loc["10.1/fail", "attempts"]) == 1

    def test_one_attempt_does_not_exhaust_the_default_budget(self, tmp_path) -> None:
        ledger = str(tmp_path / "failed.csv")
        P._append_failure_ledger(ledger, self._row())
        assert "10.1/fail" not in P._exhausted_dois(ledger, max_attempts=3)

    def test_repeated_appends_still_accumulate_across_real_runs(self, tmp_path) -> None:
        # The summing itself is correct and must stay: three separate runs that each fail once
        # should exhaust a budget of three. Only the per-flush re-append was wrong.
        ledger = str(tmp_path / "failed.csv")
        for _ in range(3):
            P._append_failure_ledger(ledger, self._row())
        assert int(pd.read_csv(ledger).set_index("doi").loc["10.1/fail", "attempts"]) == 3
        assert "10.1/fail" in P._exhausted_dois(ledger, max_attempts=3)
