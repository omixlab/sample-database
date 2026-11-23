"""
Tests for the picogram/microgram judge (:func:`sample.llm_validator.adjudicate_picogram_unit`).

Why a model decides this rather than a rule: ``pg/mL`` is a REAL unit. Measured across 6,000
extraction texts it appears in 4.58% of them, overwhelmingly in genuine measurements -- cytokine
ELISA readouts, limits of quantification, plasma peptide levels. A blanket repair, or a
value-range floor, would corrupt all of those to catch the minority that are OCR artefacts. But
artefacts do occur and are recognisable from context: one real sentence reads "TBSV (0.007-2
ug/ml) and TMVP (0.06-16 pg/ml)" -- same assay, same list, one unit a million-fold out of line.

These tests never call the model. They pin the surrounding contract, which is what makes the
judge safe: it must be cheap to skip, silent about units it has no business touching, and utterly
unwilling to rewrite a measurement when it cannot get an answer.
"""
from typing import Any, Dict

import pytest

from sample import llm_validator as V


class TestAmbiguityGate:
    @pytest.mark.parametrize("unit", ["pg/mL", "pg/ml", "pg mL", "PG/ML", " pg/mL "])
    def test_picogram_spellings_are_ambiguous(self, unit: str) -> None:
        assert V.is_ambiguous_mass_unit(unit) is True

    @pytest.mark.parametrize("unit", ["ug/mL", "uM", "µM", "ng/mL", "mg/mL", "%", "", None])
    def test_everything_else_is_left_alone(self, unit: Any) -> None:
        # The gate runs on every activity, so anything it accepts costs an LLM call. A unit that is
        # not one OCR error away from a micro sign must never reach the model.
        assert V.is_ambiguous_mass_unit(unit) is False

    def test_an_unambiguous_unit_returns_immediately(self, monkeypatch: Any) -> None:
        # Guards the ordering: the gate is checked before the breaker and before any call, so a
        # corpus with no picogram units costs nothing at all.
        def explode(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("the model must not be consulted for ug/mL")

        monkeypatch.setattr(V, "llm_call", explode)
        assert V.adjudicate_picogram_unit("ug/mL", "MIC", "32", "context") == ("ug/mL", None)


class TestFailureIsConservative:
    """A judge that cannot be consulted must not be allowed to rewrite a measurement."""

    def test_an_unreachable_backend_keeps_the_printed_unit(self, monkeypatch: Any) -> None:
        def explode(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("ollama unreachable")

        monkeypatch.setattr(V, "llm_call", explode)
        assert V.adjudicate_picogram_unit("pg/mL", "MIC", "32", "ctx") == ("pg/mL", None)

    def test_an_unparseable_reply_is_not_a_verdict(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(V, "llm_call", lambda *a, **k: "not json at all")
        unit, reason = V.adjudicate_picogram_unit("pg/mL", "MIC", "32", "ctx")
        assert (unit, reason) == ("pg/mL", None)

    def test_a_reply_missing_the_verdict_field_changes_nothing(self, monkeypatch: Any) -> None:
        # A confused model that answers with prose, or with only a reason, has not judged anything.
        monkeypatch.setattr(V, "llm_call", lambda *a, **k: '{"reason":"looks odd"}')
        assert V.adjudicate_picogram_unit("pg/mL", "MIC", "32", "ctx") == ("pg/mL", None)

    def test_an_open_breaker_skips_the_call_entirely(self, monkeypatch: Any) -> None:
        from sample.llm import _UNIT_BREAKER

        def explode(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("no call may be made while the breaker is open")

        monkeypatch.setattr(V, "llm_call", explode)
        monkeypatch.setattr(_UNIT_BREAKER, "is_open", lambda: True)
        assert V.adjudicate_picogram_unit("pg/mL", "MIC", "32", "ctx") == ("pg/mL", None)


class TestVerdicts:
    def test_picogram_confirmed_keeps_the_unit(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            V, "llm_call",
            lambda *a, **k: '{"is_picogram": true, "reason": "ELISA readout, picograms are normal"}')
        unit, reason = V.adjudicate_picogram_unit("pg/mL", "Cytotoxicity", "10", "ctx")
        assert unit == "pg/mL"
        assert reason is None, "no override means nothing to record"

    def test_a_misread_is_overridden_and_explained(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            V, "llm_call",
            lambda *a, **k: '{"is_picogram": false, "reason": "MIC beside ug/ml values"}')
        unit, reason = V.adjudicate_picogram_unit("pg/mL", "MIC", "0.06-16", "ctx")
        assert unit == "ug/mL"
        assert reason and "ug/ml" in reason.lower()

    def test_an_override_without_a_stated_reason_still_records_one(self, monkeypatch: Any) -> None:
        # The override must always be explainable after the fact, even when the model omits its
        # justification -- an unexplained rewrite of a measurement is not auditable.
        monkeypatch.setattr(V, "llm_call", lambda *a, **k: '{"is_picogram": false, "reason": ""}')
        unit, reason = V.adjudicate_picogram_unit("pg/mL", "MIC", "32", "ctx")
        assert unit == "ug/mL"
        assert reason


class TestRecordAmendment:
    def test_the_printed_unit_is_kept_beside_the_verdict(self, monkeypatch: Any) -> None:
        """An override must never destroy what the page actually said."""
        from sample import docling_tables as D

        monkeypatch.setattr(V, "llm_call",
                            lambda *a, **k: '{"is_picogram": false, "reason": "MIC in ug/ml"}')

        def cell(r0: int, r1: int, c0: int, c1: int, text: str,
                 header: bool = False) -> Dict[str, Any]:
            return {"r0": r0, "r1": r1, "c0": c0, "c1": c1, "text": text, "header": header}
        grid: Dict[str, Any] = {"page_no": 2, "cells": [
            cell(0, 1, 0, 1, "peptide", header=True),
            cell(0, 1, 1, 2, "MIC (pg/mL)", header=True),
            cell(1, 2, 0, 1, "TL"),
            cell(1, 2, 1, 2, "32"),
        ]}
        records = D.records_from_table(grid, page_text="MIC values against E. coli in pg/mL.")
        assert records, "the table must still yield a record"
        activity = records[0]["biological_activities"][0]
        assert activity["activity_unit"] == "ug/mL"
        assert activity["activity_unit_original"] == "pg/mL"
        assert activity["unit_adjudication"]

    def test_no_page_text_means_no_adjudication(self, monkeypatch: Any) -> None:
        # The judgement depends on surrounding evidence. With none, the printed unit stands rather
        # than being decided on the assay name alone.
        from sample import docling_tables as D

        def explode(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("must not consult the model without context")

        monkeypatch.setattr(V, "llm_call", explode)

        def cell(r0: int, r1: int, c0: int, c1: int, text: str,
                 header: bool = False) -> Dict[str, Any]:
            return {"r0": r0, "r1": r1, "c0": c0, "c1": c1, "text": text, "header": header}
        grid: Dict[str, Any] = {"page_no": 2, "cells": [
            cell(0, 1, 0, 1, "peptide", header=True),
            cell(0, 1, 1, 2, "MIC (pg/mL)", header=True),
            cell(1, 2, 0, 1, "TL"),
            cell(1, 2, 1, 2, "32"),
        ]}
        records = D.records_from_table(grid, page_text="")
        assert records[0]["biological_activities"][0]["activity_unit"] == "pg/mL"


class TestMicrogramsCounterpart:
    """
    The repair must keep the DENOMINATOR the unit was printed with.

    A verdict of "this was a misread micro sign" says the mass prefix is wrong and says nothing
    about the volume. Returning a hardcoded ``"ug/mL"`` therefore rewrote a ``pg/L`` reading --
    which :data:`sample.llm_validator._AMBIGUOUS_MASS_UNITS` explicitly covers -- as
    per-millilitre: a 1000x denominator error introduced by the very function written to repair a
    1000x numerator error. The per-millilitre spellings were unaffected, which is why the tests
    above never caught it.
    """

    @pytest.mark.parametrize("printed,expected", [
        ("pg/ml", "ug/mL"),
        ("pg/mL", "ug/mL"),
        ("pg ml", "ug/mL"),
        ("pg/l", "ug/L"),
        ("pg/L", "ug/L"),
        ("PG/L", "ug/L"),
        ("  pg / l  ", "ug/L"),
    ])
    def test_the_denominator_is_preserved(self, printed: str, expected: str) -> None:
        assert V.micrograms_counterpart(printed) == expected

    def test_every_ambiguous_unit_repairs_to_its_own_volume(self) -> None:
        # Guards the table and the repair against drifting apart: anything declared ambiguous must
        # repair to the volume it was printed with, never to a default.
        for unit in V._AMBIGUOUS_MASS_UNITS:
            repaired = V.micrograms_counterpart(unit)
            assert repaired.startswith("ug/"), unit
            expected = "ug/L" if unit.strip().lower().endswith("/l") else "ug/mL"
            assert repaired == expected, f"{unit} repaired to {repaired}, expected {expected}"

    def test_a_per_litre_misread_is_overridden_to_per_litre(self, monkeypatch: Any) -> None:
        # The end-to-end case the bug corrupted.
        monkeypatch.setattr(V, "llm_call",
                            lambda *a, **k: '{"is_picogram": false, "reason": "lost micro sign"}')
        unit, reason = V.adjudicate_picogram_unit("pg/L", "MIC", "32", "context text")
        assert unit == "ug/L"
        assert reason
