"""
Tests for :mod:`sample.zenodo_dump` -- building and versioning the Zenodo archive.

Weighted towards the three things the old ``cp -r`` chain got wrong while still exiting 0, because
those are the failures an archive's reader cannot detect:

* a file removed from the project stayed in the archive forever (``cp -r`` merges, never deletes),
* ``__pycache__`` / ``.pyc`` were shipped,
* nothing was versioned, so two releases shared a filename.

Every test builds against a synthetic repository root in ``tmp_path``, so nothing touches the real
13 GB tree and no test needs the release data to exist.
"""
import json
import os
from typing import Any

import pytest

from sample import zenodo_dump as zd


def _make_root(tmp_path: Any, version: str = "1.0.0") -> str:
    """Builds a minimal repository + data tree that :func:`zd.build` accepts."""
    root = tmp_path / "repo"
    (root / "sample").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    (root / "environments").mkdir()

    major, minor, _ = zd.parse_version(version)
    (root / "pyproject.toml").write_text(f'[project]\nname = "sample"\nversion = "{version}"\n',
                                         encoding="utf-8")
    (root / "sample" / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (root / "docs" / "conf.py").write_text(f"version = '{major}.{minor}'\n", encoding="utf-8")
    (root / "README.md").write_text("# sAMPle\n", encoding="utf-8")
    (root / "Makefile").write_text("all:\n\t@echo hi\n", encoding="utf-8")
    (root / "environments" / "sample.yml").write_text("name: sample\n", encoding="utf-8")
    # environment.yml is a SYMLINK in the real repository; keep that shape here.
    os.symlink(os.path.join("environments", "sample.yml"), str(root / "environment.yml"))
    (root / "sample" / "core.py").write_text("x = 1\n", encoding="utf-8")
    (root / "scripts" / "helper.py").write_text("y = 2\n", encoding="utf-8")

    data = root / "data"
    for tree in zd.RELEASE_TREES:
        (data / tree).mkdir(parents=True)
        (data / tree / "payload.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (data / "sample.db").write_text("sqlite", encoding="utf-8")
    (data / "sample_final_release" / "models_retired").mkdir()
    (data / "sample_final_release" / "models_retired" / "synergy.pkl").write_text(
        "retired", encoding="utf-8")
    (data / "sample_final_release" / "sample_records_json_pdb_sdf.zip").write_text(
        "zip", encoding="utf-8")
    return str(root)


class TestParseAndBump:
    def test_three_component_version(self) -> None:
        assert zd.parse_version("1.2.3") == (1, 2, 3)

    def test_two_component_version_implies_patch_zero(self) -> None:
        assert zd.parse_version("1.2") == (1, 2, 0)

    @pytest.mark.parametrize("bad", ["", "1", "1.2.3.4", "v1.2.3", "1.x.3", "a.b.c"])
    def test_non_numeric_versions_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            zd.parse_version(bad)

    def test_major_bump_resets_minor_and_patch(self) -> None:
        assert zd.bump_version("1.4.7", "major") == "2.0.0"

    def test_minor_bump_resets_patch_but_not_major(self) -> None:
        assert zd.bump_version("1.4.7", "minor") == "1.5.0"

    def test_patch_bump_touches_only_patch(self) -> None:
        assert zd.bump_version("1.4.7", "patch") == "1.4.8"

    def test_bump_from_zero(self) -> None:
        assert zd.bump_version("0.0.0", "patch") == "0.0.1"

    def test_double_digit_components_do_not_sort_as_strings(self) -> None:
        assert zd.bump_version("1.9.9", "minor") == "1.10.0"
        assert zd.bump_version("1.10.0", "patch") == "1.10.1"

    def test_an_unknown_part_is_refused(self) -> None:
        with pytest.raises(ValueError, match="major, minor, patch"):
            zd.bump_version("1.0.0", "revision")


class TestVersionSites:
    def test_reads_the_version_when_all_sites_agree(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "2.3.4")
        assert zd.read_version(root) == "2.3.4"

    def test_docs_conf_is_compared_on_major_minor_only(self, tmp_path: Any) -> None:
        # Sphinx convention: docs/conf.py carries 1.0, not 1.0.5. That is not drift.
        root = _make_root(tmp_path, "1.0.5")
        (tmp_path / "repo" / "docs" / "conf.py").write_text("version = '1.0'\n", encoding="utf-8")
        assert zd.read_version(root) == "1.0.5"

    def test_disagreeing_sites_raise_rather_than_picking_one(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        (tmp_path / "repo" / "sample" / "__init__.py").write_text(
            '__version__ = "9.9.9"\n', encoding="utf-8")
        with pytest.raises(zd.VersionDrift, match="disagree"):
            zd.read_version(root)

    def test_a_site_whose_format_changed_raises_instead_of_being_skipped(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        (tmp_path / "repo" / "pyproject.toml").write_text(
            "[project]\nversion = 'single-quoted'\n", encoding="utf-8")
        with pytest.raises(zd.VersionDrift, match="no version matching"):
            zd.version_sites(root)

    def test_apply_version_writes_every_site_together(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        updated = zd.apply_version(root, "1.1.0")
        assert set(updated) == {"pyproject.toml", "sample/__init__.py", "docs/conf.py"}
        assert 'version = "1.1.0"' in (tmp_path / "repo" / "pyproject.toml").read_text()
        assert '__version__ = "1.1.0"' in (tmp_path / "repo" / "sample" / "__init__.py").read_text()
        # docs/conf.py gets major.minor only.
        assert "version = '1.1'" in (tmp_path / "repo" / "docs" / "conf.py").read_text()

    def test_apply_version_dry_run_writes_nothing(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        zd.apply_version(root, "5.0.0", dry_run=True)
        assert 'version = "1.0.0"' in (tmp_path / "repo" / "pyproject.toml").read_text()


class TestBuild:
    def test_builds_the_archive_and_reports_the_version(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False)
        assert out["version"] == "1.0.0"
        dump = out["dump_dir"]
        for tree in zd.RELEASE_TREES:
            assert os.path.exists(os.path.join(dump, tree, "payload.csv"))
        assert os.path.exists(os.path.join(dump, "sample.db"))
        assert os.path.exists(os.path.join(dump, "sample", "core.py"))
        assert os.path.exists(os.path.join(dump, "scripts", "helper.py"))

    def test_a_file_removed_from_the_project_is_removed_from_the_archive(self, tmp_path: Any) -> None:
        # THE defect of the old `cp -r` chain: it merged and never deleted, so six modules that no
        # longer existed anywhere in the repository were still being published.
        root = _make_root(tmp_path, "1.0.0")
        zd.build(root=root, archive=False)
        stale = os.path.join(root, "data", "zenodo_dump", "sample", "core.py")
        assert os.path.exists(stale)

        os.remove(os.path.join(root, "sample", "core.py"))
        out = zd.build(root=root, archive=False)
        assert not os.path.exists(stale), "a module deleted from the project must leave the archive"
        assert "sample/core.py" in out["stale_removed"]

    def test_pycache_and_pyc_are_never_archived(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        cache = os.path.join(root, "sample", "__pycache__")
        os.makedirs(cache)
        with open(os.path.join(cache, "core.cpython-311.pyc"), "w", encoding="utf-8") as handle:
            handle.write("bytecode")
        out = zd.build(root=root, archive=False)
        dump = out["dump_dir"]
        assert not os.path.exists(os.path.join(dump, "sample", "__pycache__"))
        found = [n for _, _, names in os.walk(dump) for n in names if n.endswith(".pyc")]
        assert found == []

    def test_retired_models_are_excluded_by_default(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False)
        retired = os.path.join(out["dump_dir"], "sample_final_release", "models_retired")
        assert not os.path.exists(retired), "withdrawn models must not be published by default"

    def test_retired_models_are_archived_when_asked_for(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False, include_retired=True)
        assert os.path.exists(os.path.join(
            out["dump_dir"], "sample_final_release", "models_retired", "synergy.pkl"))

    def test_the_symlinked_environment_yml_is_archived_as_a_real_file(self, tmp_path: Any) -> None:
        # In the real repository environment.yml -> environments/sample.yml. Preserving the link
        # would ship a pointer; rsync -aL resolves it.
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False)
        target = os.path.join(out["dump_dir"], "environment.yml")
        assert os.path.isfile(target)
        assert not os.path.islink(target)
        assert "name: sample" in open(target, encoding="utf-8").read()

    def test_the_records_zip_survives_a_delete_rebuild(self, tmp_path: Any) -> None:
        # serialize_archive.py drops this at the archive root; a naive --delete would wipe it.
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False)
        assert os.path.exists(os.path.join(out["dump_dir"], "sample_records_json_pdb_sdf.zip"))

    def test_version_and_manifest_are_written(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.2.3")
        out = zd.build(root=root, archive=False)
        dump = out["dump_dir"]
        assert open(os.path.join(dump, "VERSION"), encoding="utf-8").read().strip() == "1.2.3"
        manifest = json.load(open(os.path.join(dump, "MANIFEST.json"), encoding="utf-8"))
        assert manifest["version"] == "1.2.3"
        assert manifest["includes_retired_models"] is False
        assert manifest["contents"]["sample"]["files"] >= 1

    def test_bump_raises_the_version_and_writes_it_back(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False, bump="minor")
        assert out["previous_version"] == "1.0.0"
        assert out["version"] == "1.1.0"
        assert zd.read_version(root) == "1.1.0"
        assert open(os.path.join(out["dump_dir"], "VERSION"), encoding="utf-8").read().strip() \
            == "1.1.0"

    def test_bump_and_explicit_version_together_are_refused(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        with pytest.raises(ValueError, match="not both"):
            zd.build(root=root, archive=False, bump="minor", set_version="3.0.0")

    def test_the_tarball_name_carries_the_version(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, bump="major")
        assert out["version"] == "2.0.0"
        assert os.path.basename(out["tarball"]) == "zenodo_dump-v2.0.0.tar.gz"
        assert os.path.exists(out["tarball"])

    def test_the_tarball_gets_a_sha256_sidecar_matching_its_contents(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root)
        sidecar = out["tarball"] + ".sha256"
        assert os.path.exists(sidecar)
        assert out["tarball_sha256"] == zd.sha256(out["tarball"])
        assert out["tarball_sha256"] in open(sidecar, encoding="utf-8").read()

    def test_a_missing_release_tree_fails_the_build(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        import shutil
        shutil.rmtree(os.path.join(root, "data", "sample_raw_release"))
        with pytest.raises(FileNotFoundError, match="sample_raw_release"):
            zd.build(root=root, archive=False)

    def test_dry_run_changes_nothing(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False, bump="major", dry_run=True)
        assert out["version"] == "2.0.0", "the summary reports what WOULD happen"
        assert zd.read_version(root) == "1.0.0", "but nothing is written"
        assert not os.path.exists(os.path.join(root, "data", "zenodo_dump", "VERSION"))

    def test_dry_run_reports_what_it_would_delete(self, tmp_path: Any) -> None:
        # The point of a dry run on a PUBLISHED archive is previewing what leaves it, so the
        # deletion list must be populated even though nothing is written.
        root = _make_root(tmp_path, "1.0.0")
        zd.build(root=root, archive=False)
        os.remove(os.path.join(root, "sample", "core.py"))

        out = zd.build(root=root, archive=False, dry_run=True)
        assert "sample/core.py" in out["stale_removed"], \
            "a dry run must name the files it would remove from the archive"
        assert os.path.exists(os.path.join(root, "data", "zenodo_dump", "sample", "core.py")), \
            "and must not actually remove them"

    def test_dry_run_against_a_nonexistent_archive_is_not_an_error(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False, dry_run=True)
        assert out["stale_removed"] == []
        assert not os.path.exists(os.path.join(root, "data", "zenodo_dump"))

    def test_pyc_already_in_the_archive_is_purged_not_merely_skipped(self, tmp_path: Any) -> None:
        # rsync --delete PROTECTS destination files matching an --exclude, so excluding
        # __pycache__ from the copy is not enough to get it out of an archive that already has it.
        # --delete-excluded is what makes the exclude a statement about the destination.
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False)
        planted = os.path.join(out["dump_dir"], "sample", "__pycache__")
        os.makedirs(planted)
        with open(os.path.join(planted, "core.cpython-311.pyc"), "w", encoding="utf-8") as handle:
            handle.write("bytecode")

        zd.build(root=root, archive=False)
        assert not os.path.exists(planted), "stale bytecode already in the archive must be purged"

    def test_retired_models_already_in_the_archive_are_purged(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False, include_retired=True)
        retired = os.path.join(out["dump_dir"], "sample_final_release", "models_retired")
        assert os.path.exists(retired)
        zd.build(root=root, archive=False)
        assert not os.path.exists(retired), \
            "switching the flag off must take the withdrawn models back out of the archive"

    def test_an_undeclared_top_level_entry_is_pruned(self, tmp_path: Any) -> None:
        # The archive root is assembled from many rsync calls, none of which prunes it, so an
        # entry no longer in the recipe would sit there forever.
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False)
        junk = os.path.join(out["dump_dir"], "leftover_from_an_old_recipe")
        os.makedirs(junk)
        with open(os.path.join(junk, "x.txt"), "w", encoding="utf-8") as handle:
            handle.write("stale")

        second = zd.build(root=root, archive=False)
        assert not os.path.exists(junk)
        # A real run reports the before/after file diff, so the contained file is what is named --
        # more useful than the directory, since it says exactly what left the archive.
        assert "leftover_from_an_old_recipe/x.txt" in second["stale_removed"]

    def test_declared_entries_are_never_pruned(self, tmp_path: Any) -> None:
        root = _make_root(tmp_path, "1.0.0")
        out = zd.build(root=root, archive=False)
        zd.build(root=root, archive=False)
        for name in ("sample", "scripts", "sample.db", "README.md", "Makefile",
                     "pyproject.toml", "environment.yml", "environments", "VERSION",
                     "MANIFEST.json", "sample_records_json_pdb_sdf.zip", *zd.RELEASE_TREES):
            assert os.path.exists(os.path.join(out["dump_dir"], name)), f"{name} was pruned"
