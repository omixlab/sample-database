"""Tests for :mod:`sample.cell_lines` — Cellosaurus-backed cell line classification."""
from typing import Any, Dict, List, Optional

import pytest

from sample import cell_lines as CL


class TestCleanCellLineName:
    @pytest.mark.parametrize("raw,expected", [
        ("A549", "A549"), ("  MCF-7  ", "MCF-7"), ("", None), ("   ", None),
        ("EC50", None), ("ec50", None), ("CCR5", None), ("ccr5", None),
    ])
    def test_trims_and_filters_known_non_cell_lines(self, raw: str, expected: Optional[str]) -> None:
        assert CL.clean_cell_line_name(raw) == expected


class TestNormalize:
    @pytest.mark.parametrize("a,b", [
        ("A549", "A-549"), ("MDA-MB-231", "MDA_MB_231"), ("thp1", "THP-1"),
        ("Raw 264.7", "RAW264.7"),
    ])
    def test_spelling_variants_normalize_identically(self, a: str, b: str) -> None:
        assert CL._normalize(a) == CL._normalize(b)


class TestClassifyDisease:
    @pytest.mark.parametrize("labels,expected", [
        (["Lung adenocarcinoma"], "lung_cancer"),
        (["Invasive breast carcinoma of no special type"], "breast_cancer"),
        (["Colon adenocarcinoma"], "colorectal_cancer"),
        (["Adult hepatocellular carcinoma"], "liver_cancer"),
        (["Glioblastoma"], "glioma"),
        (["Squamous cell carcinoma of the cervix uteri"], "cervical_cancer"),
        (["Prostate carcinoma"], "prostate_cancer"),
        (["Cutaneous melanoma"], "melanoma"),
        (["Gastric signet ring cell adenocarcinoma"], "gastric_cancer"),
        (["Pancreatic ductal adenocarcinoma"], "pancreatic_cancer"),
        (["Some completely novel malignant condition"], "other_cancer"),
    ])
    def test_keyword_match(self, labels: List[str], expected: str) -> None:
        assert CL._classify_disease(labels) == expected

    def test_empty_disease_list_is_not_a_cancer_signal(self) -> None:
        # Cellosaurus recording no disease is itself the signal that a line is not tumour-derived.
        assert CL._classify_disease([]) is None

    def test_whitespace_only_labels_are_treated_as_empty(self) -> None:
        assert CL._classify_disease(["  "]) is None


class TestTissueCategory:
    def test_slugifies_the_first_recorded_site(self) -> None:
        assert CL._tissue_category(["Fetal kidney"]) == "fetal_kidney"

    def test_not_specified_is_skipped(self) -> None:
        assert CL._tissue_category(["Not specified", "Lung"]) == "lung"

    def test_no_sites_returns_none(self) -> None:
        assert CL._tissue_category([]) is None


class TestClassifyCellLine:
    def test_not_a_cell_line_short_circuits_without_a_lookup(self, monkeypatch: Any) -> None:
        called = []
        monkeypatch.setattr(CL, "lookup_cell_line", lambda *a, **k: called.append(1))
        assert CL.classify_cell_line("EC50") is None
        assert not called

    def test_cancer_line_returns_disease_category(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(CL, "lookup_cell_line", lambda *a, **k: {
            "accession": "CVCL_0023", "name": "A-549", "tissue": ["Lung"],
            "disease": ["Lung adenocarcinoma"]})
        assert CL.classify_cell_line("A549", cache={}) == ("lung_cancer", True)

    def test_normal_line_returns_tissue_category(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(CL, "lookup_cell_line", lambda *a, **k: {
            "accession": "CVCL_0594", "name": "NIH/3T3", "tissue": ["Whole embryo"], "disease": []})
        assert CL.classify_cell_line("NIH/3T3", cache={}) == ("whole_embryo", False)

    def test_unfound_line_returns_none(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(CL, "lookup_cell_line", lambda *a, **k: None)
        assert CL.classify_cell_line("SomeObscureSubclone", cache={}) is None

    def test_result_is_cached_and_not_looked_up_twice(self, monkeypatch: Any) -> None:
        calls = []
        monkeypatch.setattr(CL, "lookup_cell_line", lambda name, **k: calls.append(name) or {
            "accession": "X", "name": name, "tissue": [], "disease": ["Lung adenocarcinoma"]})
        cache: Dict[str, Any] = {}
        CL.classify_cell_line("A549", cache=cache)
        CL.classify_cell_line("A549", cache=cache)
        assert len(calls) == 1


class TestBuildCellLineCategories:
    def test_classifies_and_dedupes_lookups(self, monkeypatch: Any) -> None:
        calls = []

        def fake_lookup(name: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
            calls.append(name)
            fixtures = {
                "A549": {"tissue": ["Lung"], "disease": ["Lung adenocarcinoma"]},
                "MCF-7": {"tissue": ["Pleural effusion"], "disease": ["Breast adenocarcinoma"]},
                "NIH/3T3": {"tissue": ["Whole embryo"], "disease": []},
            }
            return fixtures.get(name)

        monkeypatch.setattr(CL, "lookup_cell_line", fake_lookup)
        monkeypatch.setattr(CL, "_load_cache", lambda: {})
        monkeypatch.setattr(CL, "_save_cache", lambda cache: None)
        monkeypatch.setattr(CL.time, "sleep", lambda s: None)

        # "A549" appears twice -- must only be looked up once.
        result = CL.build_cell_line_categories(["A549", "A549", "MCF-7", "NIH/3T3", "EC50", ""])

        assert result["A549"] == ("lung_cancer", True)
        assert result["MCF-7"] == ("breast_cancer", True)
        assert result["NIH/3T3"] == ("whole_embryo", False)
        assert "EC50" not in result
        assert calls.count("A549") == 1

    def test_unclassifiable_names_are_omitted_not_erroring(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(CL, "lookup_cell_line", lambda *a, **k: None)
        monkeypatch.setattr(CL, "_load_cache", lambda: {})
        monkeypatch.setattr(CL, "_save_cache", lambda cache: None)
        monkeypatch.setattr(CL.time, "sleep", lambda s: None)
        assert CL.build_cell_line_categories(["SomeObscureLine"]) == {}

    def test_network_failure_degrades_to_unclassified(self, monkeypatch: Any) -> None:
        # _query_cellosaurus already swallows network errors into []; this checks the whole path
        # from build_cell_line_categories down does not propagate an exception either.
        monkeypatch.setattr(CL, "_query_cellosaurus", lambda *a, **k: [])
        monkeypatch.setattr(CL, "_load_cache", lambda: {})
        monkeypatch.setattr(CL, "_save_cache", lambda cache: None)
        monkeypatch.setattr(CL.time, "sleep", lambda s: None)
        assert CL.build_cell_line_categories(["A549"]) == {}


class TestFindExactPrefersShortestParentLine:
    def test_engineered_derivatives_are_not_preferred_over_the_parent(self, monkeypatch: Any) -> None:
        candidates = [
            {"name-list": [{"value": "A549 Cas9"}], "accession-list": [{"value": "CVCL_DX70"}]},
            {"name-list": [{"value": "A-549"}], "accession-list": [{"value": "CVCL_0023"}]},
            {"name-list": [{"value": "A549 HIF1A -/-/-"}], "accession-list": [{"value": "CVCL_C3C5"}]},
        ]
        monkeypatch.setattr(CL, "_query_cellosaurus", lambda *a, **k: candidates)
        name, entry = CL._find_exact("A-549", timeout=1.0)
        assert name == "A-549"
        assert entry["accession-list"][0]["value"] == "CVCL_0023"

    def test_no_exact_match_returns_none(self, monkeypatch: Any) -> None:
        candidates = [{"name-list": [{"value": "Something Else"}],
                       "accession-list": [{"value": "CVCL_X"}]}]
        monkeypatch.setattr(CL, "_query_cellosaurus", lambda *a, **k: candidates)
        assert CL._find_exact("A549", timeout=1.0) is None


class TestLookupCellLineHyphenRetry:
    def test_retries_with_a_hyphen_before_the_trailing_digits(self, monkeypatch: Any) -> None:
        # A549 (no exact match) should retry as A-549 (Cellosaurus's own canonical spelling) and
        # succeed on the second attempt -- the real gap this project's data hit.
        queries = []

        def fake_query(name: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
            queries.append(name)
            if name == "A-549":
                return [{"name-list": [{"value": "A-549"}],
                         "accession-list": [{"value": "CVCL_0023"}],
                         "derived-from-site-list": [{"site": {"value": "Lung"}}],
                         "disease-list": [{"label": "Lung adenocarcinoma"}]}]
            return []

        monkeypatch.setattr(CL, "_query_cellosaurus", fake_query)
        result = CL.lookup_cell_line("A549")
        assert queries == ["A549", "A-549"]
        assert result["accession"] == "CVCL_0023"
        assert result["disease"] == ["Lung adenocarcinoma"]

    def test_no_retry_needed_when_the_first_query_succeeds(self, monkeypatch: Any) -> None:
        queries = []

        def fake_query(name: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
            queries.append(name)
            return [{"name-list": [{"value": "MCF-7"}],
                     "accession-list": [{"value": "CVCL_0031"}],
                     "derived-from-site-list": [], "disease-list": []}]

        monkeypatch.setattr(CL, "_query_cellosaurus", fake_query)
        CL.lookup_cell_line("MCF-7")
        assert queries == ["MCF-7"]

    def test_both_attempts_failing_returns_none(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(CL, "_query_cellosaurus", lambda *a, **k: [])
        assert CL.lookup_cell_line("TotallyUnknown123") is None

    def test_empty_name_is_not_queried(self, monkeypatch: Any) -> None:
        called = []
        monkeypatch.setattr(CL, "_query_cellosaurus", lambda *a, **k: called.append(1))
        assert CL.lookup_cell_line("   ") is None
        assert not called
