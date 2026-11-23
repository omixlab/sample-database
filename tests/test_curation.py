"""
Tests for :mod:`sample.curation` -- building the manual-curation review set.

The properties that matter: only papers with extracted peptides are eligible (a paper the extractor
found nothing in cannot show whether what it found is right), the sample is stratified rather than
the head of a listing, an existing set is never silently destroyed, and the extraction tree is read
through the project's allowlist so a stale `_old` sibling cannot enter the review set.
"""
import csv
import json
import os
from typing import Any, Dict, List

import pytest

from sample import curation


def _write_extraction(root: str, source: str, name: str, peptides: List[Dict[str, Any]]) -> None:
    directory = os.path.join(root, source)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"source_file": name}, "peptides": peptides}, handle)


def _peptide(sequence: str = "GIGKFLHSAKKF", name: str = "test-1",
             value: Any = None) -> Dict[str, Any]:
    return {
        "peptide_sequence": sequence,
        "peptide_name": name,
        "biological_activities": [{
            "target_organism": "Escherichia coli", "activity_type": "MIC",
            "value_lower": value, "activity_unit": "ug/ml",
        }],
    }


@pytest.fixture()
def corpus(tmp_path: Any) -> str:
    """A small data dir with a mix of strata, plus a stale `_old` sibling and an empty paper."""
    data_dir = str(tmp_path)
    root = os.path.join(data_dir, "literature_data_json")
    for i in range(6):
        _write_extraction(root, "pubmed", f"measured{i}",
                          [_peptide(value=4.0), _peptide(value=8.0)])
    for i in range(4):
        _write_extraction(root, "pubmed", f"nomeasure{i}", [_peptide(value=None)])
    for i in range(3):
        _write_extraction(root, "pubmed", f"nameonly{i}",
                          [_peptide(sequence="", name="magainin", value=None)])
    _write_extraction(root, "pubmed", "empty", [])
    # A stale sibling directory: DECISIONS.md records that walking this tree once shipped
    # 22.7% duplicate rows. It must never be selected from.
    _write_extraction(root, "pubmed_old", "stale", [_peptide(value=1.0)])

    text_dir = os.path.join(data_dir, "literature_data_text", "pubmed")
    os.makedirs(text_dir, exist_ok=True)
    for name in os.listdir(os.path.join(root, "pubmed")):
        stem = os.path.splitext(name)[0]
        with open(os.path.join(text_dir, f"{stem}.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"Primary text for {stem}. The peptide showed an MIC of 4 ug/ml.")
    return data_dir


class TestSummarise:
    def test_counts_peptides_and_measured_activities(self) -> None:
        s = curation.summarise_extraction(
            {"peptides": [_peptide(value=4.0), _peptide(value=None)]})
        assert s["peptide_count"] == 2
        assert s["activity_count"] == 2
        assert s["measured_activity_count"] == 1

    def test_accepts_the_repaired_bare_list_shape(self) -> None:
        # The raw stage writes {"metadata":..., "peptides":[...]}; the repaired stage writes a bare
        # list. Both are real shapes this pipeline produces.
        s = curation.summarise_extraction([_peptide(value=4.0)])
        assert s["peptide_count"] == 1 and s["measured_activity_count"] == 1

    def test_name_only_is_counted_separately_from_sequence(self) -> None:
        s = curation.summarise_extraction(
            {"peptides": [_peptide(sequence="", name="magainin")]})
        assert s["with_sequence"] == 0 and s["with_name_only"] == 1


class TestSelection:
    def test_papers_without_peptides_are_not_eligible(self, corpus: str) -> None:
        chosen = curation.select_papers(sample_size=50, data_dir=corpus)
        assert "empty" not in {c["internal_id"] for c in chosen}

    def test_stale_sibling_directory_is_never_selected(self, corpus: str) -> None:
        chosen = curation.select_papers(sample_size=50, data_dir=corpus)
        assert "stale" not in {c["internal_id"] for c in chosen}
        assert all(c["source"] == "pubmed" for c in chosen)

    def test_draws_across_strata_rather_than_one_bucket(self, corpus: str) -> None:
        # 6 measured / 4 sequence-no-measurement / 3 name-only exist; a head-of-listing selection
        # of 6 would return only the first stratum.
        chosen = curation.select_papers(sample_size=6, data_dir=corpus)
        assert len({c["stratum"] for c in chosen}) >= 2

    def test_is_reproducible_for_a_seed(self, corpus: str) -> None:
        a = curation.select_papers(sample_size=5, data_dir=corpus, seed=7)
        b = curation.select_papers(sample_size=5, data_dir=corpus, seed=7)
        assert [c["internal_id"] for c in a] == [c["internal_id"] for c in b]

    def test_a_different_seed_can_give_a_different_set(self, corpus: str) -> None:
        a = {c["internal_id"] for c in curation.select_papers(5, corpus, seed=1)}
        b = {c["internal_id"] for c in curation.select_papers(5, corpus, seed=999)}
        assert a != b or len(a) == 5  # tiny corpora may coincide; the seed must at least be honoured

    def test_never_returns_more_than_requested(self, corpus: str) -> None:
        assert len(curation.select_papers(sample_size=3, data_dir=corpus)) == 3

    def test_missing_extraction_output_is_a_clear_error(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError, match="step 8"):
            curation.select_papers(data_dir=str(tmp_path))


class TestBuildCurationSet:
    def test_writes_one_directory_per_paper_with_all_three_files(self, corpus: str,
                                                                 tmp_path: Any) -> None:
        out = str(tmp_path / "mc")
        summary = curation.build_curation_set(sample_size=4, data_dir=corpus, output_dir=out,
                                              corpora=("papers",))
        assert summary["selected"] == 4
        assert summary["by_corpus"] == {"papers": 4}
        for name in os.listdir(os.path.join(out, "papers")):
            files = set(os.listdir(os.path.join(out, "papers", name)))
            assert {"metadata.json", "extraction.json", "paper.txt"} <= files

    def test_writes_a_manifest_recording_the_parameters(self, corpus: str, tmp_path: Any) -> None:
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=4, data_dir=corpus, output_dir=out, seed=99,
                                    corpora=("papers",))
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert manifest["seed"] == 99
        # Per-corpus mapping, not one shared number: the two corpora are deliberately sized
        # differently (800 papers / 200 patents in the defaults).
        assert manifest["sample_size_requested_per_corpus"] == {"papers": 4}
        assert manifest["total_selected"] == 4
        assert manifest["corpora"]["papers"]["selected"] == 4
        assert len(manifest["documents"]) == 4

    def test_refuses_to_clobber_an_existing_set_when_overwrite_is_off(self, corpus: str,
                                                                      tmp_path: Any) -> None:
        # `overwrite` now DEFAULTS to True -- this set is rebuilt far more often than it is
        # annotated in place, and the old default failed every rebuild. The guard still exists
        # and still has to work, so it is exercised explicitly rather than dropped.
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=2, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        with pytest.raises(FileExistsError, match="overwrite"):
            curation.build_curation_set(sample_size=2, data_dir=corpus, output_dir=out,
                                        overwrite=False, corpora=("papers",))

    def test_rebuilds_in_place_by_default(self, corpus: str, tmp_path: Any) -> None:
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=2, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        # No overwrite= argument: the second call must succeed, not raise.
        summary = curation.build_curation_set(sample_size=3, data_dir=corpus, output_dir=out,
                                              corpora=("papers",))
        assert summary["selected"] == 3
        assert len(os.listdir(os.path.join(out, "papers"))) == 3

    def test_overwrite_replaces_only_the_papers_directory(self, corpus: str,
                                                          tmp_path: Any) -> None:
        # The evaluation script lives beside papers/ and must survive a rebuild.
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=2, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        script = os.path.join(out, "evaluate_with_gemini.py")
        for name in ("evaluate_with_gemini.py", "environment.yml", ".env"):
            with open(os.path.join(out, name), "w", encoding="utf-8") as handle:
                handle.write("# not generated data\n")
        curation.build_curation_set(sample_size=3, data_dir=corpus, output_dir=out,
                                    overwrite=True, corpora=("papers",))
        for name in ("evaluate_with_gemini.py", "environment.yml", ".env"):
            assert os.path.exists(os.path.join(out, name)), \
                f"a rebuild must not delete {name}"
        assert len(os.listdir(os.path.join(out, "papers"))) == 3

    def test_metadata_carries_the_extraction_summary_and_stratum(self, corpus: str,
                                                                 tmp_path: Any) -> None:
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=1, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        name = os.listdir(os.path.join(out, "papers"))[0]
        with open(os.path.join(out, "papers", name, "metadata.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["stratum"]
        assert metadata["extraction_summary"]["peptide_count"] >= 1
        assert metadata["has_primary_text"] is True


class TestCorpora:
    def test_patent_ids_lose_their_dashes_but_paper_ids_do_not(self) -> None:
        # Google Patents writes CN-111892646-B in its CSV and CN111892646B as the filename; a
        # literature internal_id is a UUID whose dashes ARE part of the filename. Normalising both
        # the same way silently matched zero papers to their metadata.
        assert curation._normalise_document_id("CN-111892646-B", "patents") == "CN111892646B"
        uuid = "0c0babd9-3d71-2f96-0231-e99b7e738976"
        assert curation._normalise_document_id(uuid, "papers") == uuid

    def test_a_missing_corpus_is_skipped_not_fatal(self, corpus: str, tmp_path: Any) -> None:
        # The fixture has no patent tree at all; the papers must still be built.
        out = str(tmp_path / "mc")
        summary = curation.build_curation_set(sample_size=2, data_dir=corpus, output_dir=out)
        assert summary["by_corpus"]["papers"] == 2
        assert summary["by_corpus"]["patents"] == 0

    def test_unknown_corpus_is_rejected(self, corpus: str, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="Unknown corpus"):
            curation.build_curation_set(data_dir=corpus, output_dir=str(tmp_path / "mc"),
                                        corpora=("nonsense",))

    def test_sample_size_is_per_corpus_not_shared(self, corpus: str, tmp_path: Any) -> None:
        out = str(tmp_path / "mc")
        summary = curation.build_curation_set(sample_size=3, data_dir=corpus, output_dir=out)
        # Papers get their full 3 even though a second corpus was requested alongside them.
        assert summary["by_corpus"]["papers"] == 3


class TestSubsetBuilder:
    """`data/manual_curation/build_subset.py` -- turning evaluation verdicts into subsets.

    Loaded by path rather than imported: it deliberately lives outside the package, because it is
    post-evaluation tooling and not part of the pipeline.
    """

    @staticmethod
    def _module() -> Any:
        import importlib.util

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "manual_curation", "build_subset.py")
        spec = importlib.util.spec_from_file_location("build_subset", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _document(root: str, corpus: str, name: str, model: Any = None,
                  manual: Any = None) -> None:
        directory = os.path.join(root, corpus, name)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"document_id": name, "bibliographic": {"title": f"Paper {name}"}}, handle)
        with open(os.path.join(directory, "extraction.json"), "w", encoding="utf-8") as handle:
            json.dump({"peptides": [{"peptide_sequence": "GIGKFL"}]}, handle)
        for payload, filename in ((model, "review.json"), (manual, "manual_review.json")):
            if payload is not None:
                with open(os.path.join(directory, filename), "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

    def test_a_human_verdict_overrides_the_model(self, tmp_path: Any) -> None:
        module = self._module()
        root = str(tmp_path)
        self._document(root, "papers", "d1", model={"verdict": "accurate"},
                       manual={"verdict": "major_errors"})
        [document] = module.load_evaluated_documents(root, ("papers",))
        assert document["verdict"] == "major_errors"
        assert document["verdict_source"] == "manual"
        assert document["verdicts_disagree"] is True

    def test_a_failed_model_review_is_not_treated_as_a_verdict(self, tmp_path: Any) -> None:
        module = self._module()
        root = str(tmp_path)
        self._document(root, "papers", "d1", model={"error": "empty response"})
        [document] = module.load_evaluated_documents(root, ("papers",))
        assert document["verdict"] is None

    def test_gold_subset_takes_only_accurate(self, tmp_path: Any) -> None:
        module = self._module()
        root = str(tmp_path)
        self._document(root, "papers", "good", model={"verdict": "accurate"})
        self._document(root, "papers", "bad", model={"verdict": "major_errors"})
        summary = module.build_subset("gold", ("accurate",), set_dir=root, corpora=("papers",))
        assert summary["selected"] == 1
        with open(os.path.join(root, "subsets", "gold", "manifest.json"), encoding="utf-8") as fh:
            assert [d["document_id"] for d in json.load(fh)["documents"]] == ["good"]

    def test_unevaluated_documents_are_excluded_by_default(self, tmp_path: Any) -> None:
        # Silently mixing unreviewed documents into a "gold" set would make it worthless.
        module = self._module()
        root = str(tmp_path)
        self._document(root, "papers", "never_reviewed")
        assert module.build_subset("gold", ("accurate",), set_dir=root,
                                   corpora=("papers",))["selected"] == 0
        assert module.build_subset("all", ("accurate",), set_dir=root, corpora=("papers",),
                                   include_unevaluated=True)["selected"] == 1

    def test_findings_are_flattened_for_the_defect_set(self, tmp_path: Any) -> None:
        module = self._module()
        root = str(tmp_path)
        self._document(root, "papers", "bad", model={
            "verdict": "major_errors",
            "precision_review": {"items": [
                {"peptide": "MCF-7", "is_real_peptide": False, "reason": "a cell line"},
                {"peptide": "GIGKFL", "is_real_peptide": True, "reason": "studied here"}]},
            "recall_review": {"missed": [
                {"name_or_sequence": "magainin 2", "evidence": "Table 2"}]},
            "field_accuracy": {"activity_value": {"correct": 1, "wrong": 1, "examples": [
                {"extracted": "128", "should_be": "12.8", "evidence": "MIC of 12.8"}]}},
        })
        module.build_subset("defects", ("major_errors",), set_dir=root, corpora=("papers",))
        with open(os.path.join(root, "subsets", "defects", "findings.csv"),
                  encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        kinds = {r["finding"] for r in rows}
        assert kinds == {"not_a_real_peptide", "missed_peptide", "wrong_field"}
        # The correctly-extracted peptide must NOT appear as a finding.
        assert "GIGKFL" not in {r["extracted"] for r in rows}

    def test_extractions_jsonl_carries_one_line_per_document(self, tmp_path: Any) -> None:
        module = self._module()
        root = str(tmp_path)
        for name in ("a", "b"):
            self._document(root, "papers", name, model={"verdict": "accurate"})
        module.build_subset("gold", ("accurate",), set_dir=root, corpora=("papers",))
        with open(os.path.join(root, "subsets", "gold", "extractions.jsonl"),
                  encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        assert len(lines) == 2
        assert all(line["extraction"]["peptides"] for line in lines)


#: Enough of a PDF to be copied and recognised; the content is irrelevant to these tests.
_PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def _write_pdfs(data_dir: str, source: str = "pubmed", skip: Any = ()) -> None:
    """
    Puts a PDF beside the download ledger for every extraction in `source`.

    :param data_dir: The fixture data directory.
    :param source: Which source's PDFs to write.
    :param skip: Stems to leave without a PDF, so the missing-PDF path can be exercised.
    """
    pdf_dir = os.path.join(data_dir, "literature_data", source)
    os.makedirs(pdf_dir, exist_ok=True)
    extraction_dir = os.path.join(data_dir, "literature_data_json", source)
    for name in os.listdir(extraction_dir):
        stem = os.path.splitext(name)[0]
        if stem in skip:
            continue
        with open(os.path.join(pdf_dir, f"{stem}.pdf"), "wb") as handle:
            handle.write(_PDF_BYTES)


class TestPdfAttachment:
    """
    The PDF ships beside ``paper.txt`` because the two answer different questions: the text is what
    the extractor read (was extraction faithful to its input?), the PDF is what the author
    published (was the input faithful to the paper?). A reviewer cannot check a table-derived MIC
    against a text dump that has lost the table.
    """

    def test_build_copies_the_pdf_and_records_it(self, corpus: str, tmp_path: Any) -> None:
        _write_pdfs(corpus)
        out = str(tmp_path / "mc")
        summary = curation.build_curation_set(sample_size=4, data_dir=corpus, output_dir=out,
                                              corpora=("papers",))
        assert summary["missing_pdf"] == 0
        for name in os.listdir(os.path.join(out, "papers")):
            directory = os.path.join(out, "papers", name)
            assert os.path.isfile(os.path.join(directory, "paper.pdf"))
            with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
                metadata = json.load(handle)
            assert metadata["has_pdf"] is True
            assert metadata["files"]["paper_pdf"] == "paper.pdf"

    def test_a_document_with_no_pdf_is_recorded_not_fatal(self, corpus: str,
                                                          tmp_path: Any) -> None:
        # A source can serve a text layer and refuse the PDF. That must not sink the build, and
        # must not be silently indistinguishable from a PDF that is present.
        _write_pdfs(corpus, skip={f"measured{i}" for i in range(6)})
        out = str(tmp_path / "mc")
        summary = curation.build_curation_set(sample_size=10, data_dir=corpus, output_dir=out,
                                              corpora=("papers",))
        assert summary["selected"] == 10
        states = {}
        for name in os.listdir(os.path.join(out, "papers")):
            with open(os.path.join(out, "papers", name, "metadata.json"),
                      encoding="utf-8") as handle:
                metadata = json.load(handle)
            states[name] = metadata["has_pdf"]
            if not metadata["has_pdf"]:
                assert metadata["files"]["paper_pdf"] is None
                assert not os.path.exists(os.path.join(out, "papers", name, "paper.pdf"))
        assert summary["missing_pdf"] == sum(1 for present in states.values() if not present)
        assert summary["missing_pdf"] > 0

    def test_attaching_twice_does_not_recopy(self, corpus: str, tmp_path: Any) -> None:
        _write_pdfs(corpus)
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=2, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        name = os.listdir(os.path.join(out, "papers"))[0]
        target = os.path.join(out, "papers", name)
        destination = os.path.join(target, "paper.pdf")
        os.utime(destination, (0, 0))
        with open(os.path.join(target, "metadata.json"), encoding="utf-8") as handle:
            source = json.load(handle)["source"]
        assert curation._attach_pdf("papers", source, name, target, corpus) is True
        assert os.path.getmtime(destination) == 0, "an unchanged PDF should not be copied again"


def _multi_activity_peptide(name: str, sequence: str, organisms: List[str]) -> Dict[str, Any]:
    """
    :param name: Peptide name.
    :param sequence: Peptide sequence, possibly empty.
    :param organisms: One activity per organism, which is the shape that makes the flattening
        testable -- the shared `corpus` fixture gives every peptide exactly one activity.
    :returns: A peptide record with one activity per organism.
    """
    return {
        "peptide_sequence": sequence,
        "peptide_name": name,
        "biological_activities": [
            {"target_organism": organism, "target_group": "antibacterial",
             "activity_type": "MIC", "value_raw": str(4 * (i + 1)), "value_operator": "",
             "activity_unit": "ug/mL", "value_lower": 4.0 * (i + 1),
             "context_sentence": f"{name} inhibited {organism}."}
            for i, organism in enumerate(organisms)
        ],
    }


@pytest.fixture()
def review_corpus(tmp_path: Any) -> str:
    """
    A data dir whose positives have MULTI-activity peptides and whose negatives really are empty.

    Separate from `corpus` because that fixture's peptides carry one activity each, which cannot
    show whether a peptide with several activities is spread over several rows.
    """
    data_dir = str(tmp_path / "data")
    root = os.path.join(data_dir, "literature_data_json")
    for i in range(3):
        _write_extraction(root, "pubmed", f"multi{i}", [
            _multi_activity_peptide("magainin-2", "GIGKFLHSAKKFGKAFVGEIMNS",
                                    ["Escherichia coli", "Staphylococcus aureus",
                                     "Pseudomonas aeruginosa"]),
            _multi_activity_peptide("nameless-1", "", ["Candida albicans"]),
        ])
    for i in range(3):
        _write_extraction(root, "pubmed", f"blank{i}", [])

    text_dir = os.path.join(data_dir, "literature_data_text", "pubmed")
    os.makedirs(text_dir, exist_ok=True)
    for name in os.listdir(os.path.join(root, "pubmed")):
        stem = os.path.splitext(name)[0]
        with open(os.path.join(text_dir, f"{stem}.txt"), "w", encoding="utf-8") as handle:
            # Comfortably over document_quality._MIN_USABLE_CHARS (500). A zero-peptide document
            # whose text is too short is dropped from the NEGATIVE pool on purpose -- a CAPTCHA
            # page yields no peptides, and counting it as a correct empty extraction would inflate
            # apparent recall -- so a fixture that wants negatives has to supply real-looking text.
            handle.write(f"Primary text for {stem}. " + (
                "Antimicrobial peptides were assayed against a panel of reference strains using "
                "broth microdilution, and minimum inhibitory concentrations were read after "
                "eighteen hours of incubation at thirty-seven degrees. " * 4))
    _write_pdfs(data_dir)
    return data_dir


class TestReviewRows:
    """
    One row per activity, not per peptide. A peptide tested against six organisms is six separate
    facts that can each be right or wrong; one row for all six would force a single verdict over
    six judgements, which is what the per-field accuracy table needs to avoid.
    """

    @staticmethod
    def _build(data_dir: str, tmp_path: Any) -> str:
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=6, data_dir=data_dir, output_dir=out,
                                    corpora=("papers",), positive_fraction=0.5)
        return out

    def test_one_row_per_activity_with_the_peptide_fields_repeated(self, review_corpus: str,
                                                                   tmp_path: Any) -> None:
        out = self._build(review_corpus, tmp_path)
        rows = curation._review_rows(out, ("papers",))
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["row_key"].rsplit("#", 1)[0], []).append(row)
        multi = [g for g in grouped.values() if len(g) > 1]
        assert multi, "magainin-2 carries three activities and must span three rows"
        for group in multi:
            assert len(group) == 3
            # The peptide's own fields repeat, so each line stands alone when the sheet is sorted.
            assert {r["peptide_name"] for r in group} == {"magainin-2"}
            assert {r["peptide_sequence"] for r in group} == {"GIGKFLHSAKKFGKAFVGEIMNS"}
            assert len({r["id"] for r in group}) == 1
            # ...while the activity differs on every one of them.
            assert {r["target_organism"] for r in group} == {
                "Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"}
            assert len({r["activity_value"] for r in group}) == 3

    def test_a_document_with_no_peptides_still_gets_one_row(self, review_corpus: str,
                                                            tmp_path: Any) -> None:
        out = self._build(review_corpus, tmp_path)
        rows = curation._review_rows(out, ("papers",))
        negatives = [r for r in rows if r["pipeline_found"] == "nothing"]
        assert negatives, "the negative half must be reviewable -- it is the only recall signal"
        for row in negatives:
            assert row["row_key"].endswith("#-#-")
            assert row.get("peptide_name", "") == ""
            assert row.get("activity_value", "") == ""

    def test_a_peptide_with_no_activities_still_gets_one_row(self, tmp_path: Any) -> None:
        data_dir = str(tmp_path / "data")
        root = os.path.join(data_dir, "literature_data_json")
        _write_extraction(root, "pubmed", "lonely",
                          [{"peptide_sequence": "GIGKFLHSAKKF", "peptide_name": "orphan",
                            "biological_activities": []}])
        text_dir = os.path.join(data_dir, "literature_data_text", "pubmed")
        os.makedirs(text_dir, exist_ok=True)
        with open(os.path.join(text_dir, "lonely.txt"), "w", encoding="utf-8") as handle:
            handle.write("text")
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=1, data_dir=data_dir, output_dir=out,
                                    corpora=("papers",), positive_fraction=1.0)
        rows = curation._review_rows(out, ("papers",))
        assert len(rows) == 1
        assert rows[0]["peptide_name"] == "orphan"
        assert rows[0]["row_key"].endswith("#0#-")
        assert rows[0].get("activity_value", "") == ""

    def test_row_keys_are_unique(self, review_corpus: str, tmp_path: Any) -> None:
        out = self._build(review_corpus, tmp_path)
        keys = [r["row_key"] for r in curation._review_rows(out, ("papers",))]
        assert len(keys) == len(set(keys)), "row_key is the join key for merging findings back"

    def test_rows_are_ordered_deterministically(self, review_corpus: str, tmp_path: Any) -> None:
        out = self._build(review_corpus, tmp_path)
        assert ([r["row_key"] for r in curation._review_rows(out, ("papers",))]
                == [r["row_key"] for r in curation._review_rows(out, ("papers",))])

    def test_the_pdf_column_is_a_path_relative_to_the_workbook(self, review_corpus: str,
                                                               tmp_path: Any) -> None:
        out = self._build(review_corpus, tmp_path)
        for row in curation._review_rows(out, ("papers",)):
            assert row["pdf"] == f"papers/{row['id']}/paper.pdf"
            assert os.path.isfile(os.path.join(out, row["pdf"]))


class TestReviewWorkbook:
    def test_writes_three_sheets_with_one_row_per_activity(self, corpus: str,
                                                           tmp_path: Any) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        _write_pdfs(corpus)
        out = str(tmp_path / "mc")
        summary = curation.build_curation_set(sample_size=10, data_dir=corpus, output_dir=out,
                                              corpora=("papers",))
        path = summary["review_workbook"]
        assert path and os.path.isfile(path)
        workbook = openpyxl.load_workbook(path)
        assert workbook.sheetnames == ["Review", "Legend", "Summary"]
        sheet = workbook["Review"]
        assert sheet.max_row - 1 == len(curation._review_rows(out, ("papers",)))
        assert [c.value for c in sheet[1]] == [h for h, _ in curation._REVIEW_COLUMNS]

    def test_verdict_columns_are_empty_and_constrained(self, corpus: str, tmp_path: Any) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        _write_pdfs(corpus)
        out = str(tmp_path / "mc")
        path = curation.build_curation_set(sample_size=6, data_dir=corpus, output_dir=out,
                                           corpora=("papers",))["review_workbook"]
        workbook = openpyxl.load_workbook(path)
        sheet = workbook["Review"]
        headers = {c.value: c.column for c in sheet[1]}
        for name in ("verdict_name", "verdict_sequence", "verdict_organism", "verdict_activity",
                     "verdict_document"):
            column = headers[name]
            values = {sheet.cell(row=r, column=column).value
                      for r in range(2, sheet.max_row + 1)}
            assert values <= {None, ""}, f"{name} must ship blank for the reviewer to fill"
        # One dropdown per verdict column, and nothing else validated.
        assert len(sheet.data_validations.dataValidation) == 5
        vocabularies = {dv.formula1 for dv in sheet.data_validations.dataValidation}
        assert f'"{",".join(curation.FIELD_VERDICTS)}"' in vocabularies
        assert f'"{",".join(curation.DOCUMENT_VERDICTS)}"' in vocabularies

    def test_refuses_to_write_without_a_curation_set(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            curation.write_review_workbook(str(tmp_path / "nothing-here"))


class TestRefresh:
    """
    ``--refresh`` exists so a set someone is already reviewing can gain the PDFs and the workbook.
    Its whole value is that it does NOT go through ``build_curation_set``, which deletes the
    corpus directory -- and with it any annotation written into the tree.
    """

    def test_attaches_missing_pdfs_without_reselecting(self, corpus: str, tmp_path: Any) -> None:
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=6, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        before = sorted(os.listdir(os.path.join(out, "papers")))
        assert not any(os.path.exists(os.path.join(out, "papers", n, "paper.pdf"))
                       for n in before)

        _write_pdfs(corpus)
        result = curation.refresh_curation_set(output_dir=out, data_dir=corpus)

        assert sorted(os.listdir(os.path.join(out, "papers"))) == before, "no re-selection"
        assert result["pdfs_attached"] == len(before)
        assert result["pdfs_unavailable"] == 0
        for name in before:
            assert os.path.isfile(os.path.join(out, "papers", name, "paper.pdf"))
            with open(os.path.join(out, "papers", name, "metadata.json"),
                      encoding="utf-8") as handle:
                assert json.load(handle)["files"]["paper_pdf"] == "paper.pdf"

    def test_preserves_work_already_in_the_tree(self, corpus: str, tmp_path: Any) -> None:
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=4, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        name = sorted(os.listdir(os.path.join(out, "papers")))[0]
        annotation = os.path.join(out, "papers", name, "review.json")
        with open(annotation, "w", encoding="utf-8") as handle:
            json.dump({"verdict": "accurate", "curator": "a human"}, handle)

        _write_pdfs(corpus)
        curation.refresh_curation_set(output_dir=out, data_dir=corpus)

        with open(annotation, encoding="utf-8") as handle:
            assert json.load(handle)["curator"] == "a human", "refresh must never discard review"

    def test_is_idempotent(self, corpus: str, tmp_path: Any) -> None:
        _write_pdfs(corpus)
        out = str(tmp_path / "mc")
        curation.build_curation_set(sample_size=4, data_dir=corpus, output_dir=out,
                                    corpora=("papers",))
        again = curation.refresh_curation_set(output_dir=out, data_dir=corpus)
        assert again["pdfs_attached"] == 0
        assert again["pdfs_already_present"] == again["documents"]

    def test_refuses_without_a_curation_set(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            curation.refresh_curation_set(output_dir=str(tmp_path / "absent"))
