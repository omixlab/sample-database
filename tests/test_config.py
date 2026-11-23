"""Tests for :mod:`sample.config` — paths and extraction-file discovery."""
import json
import os
from typing import Any

import pytest

from sample import config as C


class TestPaths:
    @pytest.mark.parametrize("name", [
        "BASE_DIR", "DATA_DIR", "PROMPTS_DIR", "LITERATURE_DIR", "PATENTS_DIR",
        "LOGS_DIR", "RESOURCES_DIR",
    ])
    def test_directory_constants_are_absolute(self, name: str) -> None:
        assert os.path.isabs(str(getattr(C, name)))

    def test_data_dir_sits_under_base_dir(self) -> None:
        assert str(C.DATA_DIR).startswith(str(C.BASE_DIR))

    @pytest.mark.parametrize("name", [
        "DEFAULT_SCREEN_MODEL", "DEFAULT_EXTRACT_MODEL", "DEFAULT_TRIAGE_MODEL",
    ])
    def test_model_names_are_non_empty_strings(self, name: str) -> None:
        value = getattr(C, name)
        assert isinstance(value, str) and value.strip()


class TestExtractionSourceDirs:
    def test_covers_every_pipeline_source(self) -> None:
        assert set(C.EXTRACTION_SOURCE_DIRS) >= {"pubmed", "scopus", "google_patents"}

    def test_is_a_tuple(self) -> None:
        # Iterated in several stages; a mutable default would be a shared-state hazard.
        assert isinstance(C.EXTRACTION_SOURCE_DIRS, tuple)


class TestIterExtractionFiles:
    def test_finds_files_in_source_subdirectories(self, tmp_path: Any) -> None:
        (tmp_path / "pubmed").mkdir()
        (tmp_path / "pubmed" / "a.json").write_text("{}", encoding="utf-8")
        (tmp_path / "scopus").mkdir()
        (tmp_path / "scopus" / "b.json").write_text("{}", encoding="utf-8")

        found = C.iter_extraction_files(str(tmp_path))
        assert len(found) == 2
        assert {os.path.basename(f) for f in found} == {"a.json", "b.json"}

    def test_finds_files_sitting_directly_in_the_directory(self, tmp_path: Any) -> None:
        # How a stage is addressed when pointed straight at .../pubmed.
        (tmp_path / "c.json").write_text("{}", encoding="utf-8")
        assert len(C.iter_extraction_files(str(tmp_path))) == 1

    def test_ignores_unrecognized_subdirectories(self, tmp_path: Any) -> None:
        # This is what protects a run from picking up stale `pubmed.bak` / `pubmed_old` trees.
        (tmp_path / "pubmed.bak").mkdir()
        (tmp_path / "pubmed.bak" / "old.json").write_text("{}", encoding="utf-8")
        (tmp_path / "pubmed_old").mkdir()
        (tmp_path / "pubmed_old" / "older.json").write_text("{}", encoding="utf-8")
        assert C.iter_extraction_files(str(tmp_path)) == []

    def test_ignores_non_json_files(self, tmp_path: Any) -> None:
        (tmp_path / "pubmed").mkdir()
        (tmp_path / "pubmed" / "notes.txt").write_text("x", encoding="utf-8")
        assert C.iter_extraction_files(str(tmp_path)) == []

    @pytest.mark.parametrize("path", ["", None, "/definitely/not/here"])
    def test_missing_directory_yields_empty_list(self, path: Any) -> None:
        # A stage can be pointed at an output that does not exist yet; that must not raise.
        assert C.iter_extraction_files(path) == []

    def test_a_file_path_is_not_a_directory(self, tmp_path: Any) -> None:
        target = tmp_path / "f.json"
        target.write_text("{}", encoding="utf-8")
        assert C.iter_extraction_files(str(target)) == []

    def test_result_is_sorted(self, tmp_path: Any) -> None:
        (tmp_path / "pubmed").mkdir()
        for name in ("c.json", "a.json", "b.json"):
            (tmp_path / "pubmed" / name).write_text("{}", encoding="utf-8")
        found = [os.path.basename(f) for f in C.iter_extraction_files(str(tmp_path))]
        assert found == sorted(found)

    def test_returned_paths_are_readable(self, extraction_dir: str) -> None:
        for path in C.iter_extraction_files(extraction_dir):
            with open(path, encoding="utf-8") as fh:
                json.load(fh)


class TestBackupDirectoriesExcluded:
    """
    The allowlist exists to keep working copies out of the pipeline.

    `data/literature_data_json_repaired/` holds `pubmed` and `scopus` beside `pubmed.bak`,
    `pubmed_old`, `scopus.bak` and `scopus_old` -- 128,049 files where the live set is 46,189, and
    the backups duplicate each other. Step 19 scanned that tree with `os.walk` and shipped a
    database whose activity table was 22.7 % duplicate rows. Anything reading extraction output
    goes through `iter_extraction_files`; these tests pin the behaviour that makes that safe.
    """

    @pytest.mark.parametrize("stale", ["pubmed.bak", "pubmed_old", "scopus.bak", "scopus_old"])
    def test_stale_sibling_is_not_read(self, tmp_path: Any, stale: str) -> None:
        (tmp_path / "pubmed").mkdir()
        (tmp_path / "pubmed" / "live.json").write_text("{}", encoding="utf-8")
        (tmp_path / stale).mkdir()
        (tmp_path / stale / "stale.json").write_text("{}", encoding="utf-8")

        found = [os.path.basename(f) for f in C.iter_extraction_files(str(tmp_path))]
        assert found == ["live.json"]

    def test_only_recognized_sources_are_read(self, tmp_path: Any) -> None:
        for name in C.EXTRACTION_SOURCE_DIRS:
            (tmp_path / name).mkdir()
            (tmp_path / name / f"{name}.json").write_text("{}", encoding="utf-8")
        (tmp_path / "not_a_source").mkdir()
        (tmp_path / "not_a_source" / "x.json").write_text("{}", encoding="utf-8")

        found = {os.path.basename(f) for f in C.iter_extraction_files(str(tmp_path))}
        assert found == {f"{n}.json" for n in C.EXTRACTION_SOURCE_DIRS}


class TestEnvironmentDependencies:
    """
    Environments that run a step must be able to import what that step imports.

    `sample-chem` runs step 20, which reaches into `sample.nlp` for `infer_activity_group`, and
    `sample/nlp.py` imports `ahocorasick` at module level. The package was declared in every other
    environment but not that one, so the import raised inside a per-file `except` and the DBAASP
    layer contributed nothing -- 152,308 activities, 60 % of the table, lost with no error in the
    log. `make check-envs` compares versions of packages that are present; it cannot catch one that
    is absent everywhere it matters.
    """

    def _pip_entries(self, name: str) -> str:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "environments", name)
        if not os.path.exists(path):
            pytest.skip(f"{name} not present")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    @pytest.mark.parametrize("env", ["sample-chem.yml", "sample-extraction.yml"])
    def test_environment_declares_ahocorasick(self, env: str) -> None:
        assert "pyahocorasick" in self._pip_entries(env), (
            f"{env} runs sample-cli steps that import sample.nlp, which imports ahocorasick "
            f"at module level")
