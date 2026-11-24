"""
Build the Zenodo archive for a release, and version it.

This replaced a ``cp -r`` chain in the Makefile (step 54). The chain worked in the sense that it
exited 0, which is exactly why its defects survived so long:

* **``cp -r`` merges into an existing directory and never deletes.** Every file ever removed from
  the project stayed in the dump forever. At the time this module was written the dump still
  carried six modules that no longer existed anywhere in the repository
  (``genomic_context.py``, ``genomic_context_store.py``, ``pdb_homology.py``, ``expression.py``,
  ``result_cache.py``, ``templates/expression.html``) plus 126 ``.pyc`` files in two
  ``__pycache__`` directories. An archive is a claim about what a release contained, so shipping
  a module the release does not have is not untidiness -- it is a false claim, and the kind a
  reader cannot detect.
* **Nothing was versioned.** The tarball was always ``zenodo_dump.tar.gz``, so two different
  releases were indistinguishable by name or by content inspection; there was no ``VERSION`` file
  and no manifest.
* **``environment.yml`` is a symlink** into ``environments/``, and that directory was never
  copied -- so the archive's one environment specification pointed at a file the archive did not
  contain. Copied with symlinks resolved, and ``environments/`` is now included.
* **The retired models were shipped.** 348 MB of withdrawn models, 216 MB of it the synergy
  pickles retired because only 2 of 18 passed their own gate. Publishing a model that was pulled
  for being unreliable invites someone to use it, so ``models_retired/`` is excluded unless asked
  for explicitly.

The version lives in three places that nothing kept aligned (``pyproject.toml``,
``sample/__init__.py``, ``docs/conf.py``). :func:`read_version` reads all three and **refuses to
proceed when they disagree**, rather than picking one and letting the drift widen; ``--bump``
writes all three together.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sample.zenodo_dump")

#: Release directories under ``data/`` copied into the archive whole.
RELEASE_TREES: Tuple[str, ...] = (
    "sample_raw_release",
    "sample_enriched_release",
    "sample_final_release",
    "sample_decoy_databases",
)

#: Single files under ``data/`` copied to the archive root.
DATA_FILES: Tuple[str, ...] = (
    "sample.db",
    # Written by sample/serialize_archive.py, which also drops its own copy at the archive root.
    # Named here so a --delete rebuild re-places it instead of wiping it: that is precisely the
    # file the old `cp -r` chain never copied and only kept by accident, because it never deleted.
    "sample_final_release/sample_records_json_pdb_sdf.zip",
)

#: Repository paths copied for reproducibility. ``environments/`` is here because
#: ``environment.yml`` is a symlink into it; without it the archive's env spec dangles.
REPO_PATHS: Tuple[str, ...] = (
    "README.md",
    "Makefile",
    "pyproject.toml",
    "environment.yml",
    "environments",
    "sample",
    "scripts",
)

#: Never archived, whatever it is copied from.
EXCLUDES: Tuple[str, ...] = (
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    ".mypy_cache/",
    ".DS_Store",
    "*.swp",
)

#: Excluded unless ``include_retired``. A model withdrawn from service should not be published as
#: though it were part of the release.
RETIRED_SUBPATH = "models_retired"

#: ``(path, regex capturing the version, replacement template)``. ``docs/conf.py`` carries
#: ``major.minor`` only, which is Sphinx's convention, so its template takes ``{mm}``.
VERSION_SITES: Tuple[Tuple[str, str, str], ...] = (
    ("pyproject.toml", r'(?m)^version = "(\d+\.\d+\.\d+)"$', 'version = "{full}"'),
    ("sample/__init__.py", r'(?m)^__version__ = "(\d+\.\d+\.\d+)"$', '__version__ = "{full}"'),
    ("docs/conf.py", r"(?m)^version = '(\d+\.\d+)'$", "version = '{mm}'"),
)

VALID_BUMPS = ("major", "minor", "patch")


class VersionDrift(RuntimeError):
    """Raised when the version strings in the repository disagree with each other."""


def parse_version(text: str) -> Tuple[int, int, int]:
    """
    :param text: A version string, ``major.minor.patch`` or ``major.minor``.
    :returns: Its three components, a missing patch counting as 0.
    :raises ValueError: When ``text`` is not a dotted numeric version.
    """
    parts = str(text).strip().split(".")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"Not a numeric version: {text!r}")
    major, minor = int(parts[0]), int(parts[1])
    patch = int(parts[2]) if len(parts) == 3 else 0
    return major, minor, patch


def bump_version(current: str, part: str) -> str:
    """
    :param current: The version being bumped.
    :param part: ``major``, ``minor`` or ``patch``.
    :returns: The bumped version, with every component below the bumped one reset to 0 --
        ``1.4.7`` bumped ``minor`` is ``1.5.0``, not ``1.5.7``.
    :raises ValueError: When ``part`` is not one of :data:`VALID_BUMPS`.
    """
    if part not in VALID_BUMPS:
        raise ValueError(f"Version part must be one of {', '.join(VALID_BUMPS)}; got {part!r}")
    major, minor, patch = parse_version(current)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def version_sites(root: str) -> Dict[str, str]:
    """
    :param root: Repository root.
    :returns: ``path -> the version found there``, for each of :data:`VERSION_SITES` that exists.
    :raises VersionDrift: When a site exists but its version pattern does not match -- a silently
        unmatched site is a site that would never get bumped.
    """
    found: Dict[str, str] = {}
    for relative, pattern, _ in VERSION_SITES:
        path = os.path.join(root, relative)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            match = re.search(pattern, handle.read())
        if not match:
            raise VersionDrift(
                f"{relative} exists but carries no version matching {pattern!r}. Its format "
                f"changed, so a bump would silently skip it; fix the pattern in VERSION_SITES.")
        found[relative] = match.group(1)
    if not found:
        raise VersionDrift("No version site found at all; cannot determine the release version.")
    return found


def read_version(root: str) -> str:
    """
    :param root: Repository root.
    :returns: The release version, taken from ``pyproject.toml`` when present.
    :raises VersionDrift: When the sites disagree. ``docs/conf.py`` holds ``major.minor`` only, so
        it is compared on those two components; the others must match in full.
    """
    found = version_sites(root)
    authority = found.get("pyproject.toml") or sorted(found.values())[0]
    expected_mm = ".".join(str(p) for p in parse_version(authority)[:2])
    for relative, value in found.items():
        same = (value == authority if len(value.split(".")) == 3
                else value == expected_mm)
        if not same:
            raise VersionDrift(
                "Version strings disagree across the repository: "
                + ", ".join(f"{k} = {v}" for k, v in sorted(found.items()))
                + ". Set them to one value (or run with --bump, which writes all of them) before "
                  "building an archive, so the archive's version means something.")
    return authority


def apply_version(root: str, new_version: str, dry_run: bool = False) -> List[str]:
    """
    Writes ``new_version`` to every site in :data:`VERSION_SITES`.

    :param root: Repository root.
    :param new_version: The version to write.
    :param dry_run: Report what would change without writing.
    :returns: The relative paths updated (or that would be).
    :raises VersionDrift: When a site's replacement does not apply exactly once.
    """
    major, minor, _ = parse_version(new_version)
    updated: List[str] = []
    for relative, pattern, template in VERSION_SITES:
        path = os.path.join(root, relative)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        replacement = template.format(full=new_version, mm=f"{major}.{minor}")
        new_text, count = re.subn(pattern, replacement.replace("\\", "\\\\"), text)
        if count != 1:
            raise VersionDrift(
                f"{relative}: expected exactly one version line to rewrite, matched {count}.")
        if not dry_run:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(new_text)
        updated.append(relative)
    logger.info("Version %s written to %d file(s)%s: %s", new_version, len(updated),
                " (dry run)" if dry_run else "", ", ".join(updated))
    return updated


def _rsync(source: str, destination: str, excludes: Tuple[str, ...],
           dry_run: bool, prefix: str = "") -> List[str]:
    """
    Copies ``source`` to ``destination`` with ``--delete``, so the destination *mirrors* the source.

    ``-L`` resolves symlinks into real files: ``environment.yml`` is a symlink into
    ``environments/``, and an archive that preserved the link would ship a dangling pointer.

    :param source: File or directory to copy. A trailing slash is added for directories so the
        contents, not the directory itself, land inside ``destination``.
    :param destination: Where it goes.
    :param excludes: rsync ``--exclude`` patterns.
    :param dry_run: Pass ``--dry-run`` to rsync, changing nothing.
    :param prefix: Archive-relative prefix, prepended to the deletion paths reported back.
    :returns: The archive-relative paths rsync deleted (or, under ``dry_run``, would delete). This
        is the point of running a dry run at all: for a *published* archive the question worth
        previewing is what is about to be taken out of it, not what is going in.
    :raises subprocess.CalledProcessError: When rsync fails.
    """
    # --delete-excluded, not just --delete: a plain --delete PROTECTS destination files that match
    # an --exclude pattern, so the 126 .pyc files the old chain had accumulated would have survived
    # a mirror that merely excluded them from the copy. The excludes here mean "must not be in the
    # archive", which is a statement about the destination, so the destination copies must go too.
    command = ["rsync", "-aL", "--delete", "--delete-excluded", "--out-format=%o %n"]
    for pattern in excludes:
        command += ["--exclude", pattern]
    if dry_run:
        command.append("--dry-run")
    if os.path.isdir(source):
        command += [source.rstrip("/") + "/", destination.rstrip("/") + "/"]
    else:
        command += [source, destination]
    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(destination)) if not os.path.isdir(source)
                    else destination, exist_ok=True)
    elif not os.path.isdir(os.path.dirname(os.path.abspath(destination))):
        # Nothing to compare against yet, and rsync would fail on the missing parent.
        return []
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    deleted = []
    for line in result.stdout.splitlines():
        operation, _, name = line.partition(" ")
        if operation == "del." and name.strip() and not name.endswith("/"):
            deleted.append(os.path.join(prefix, name.strip()) if prefix else name.strip())
    return deleted


def _tree_stats(path: str) -> Dict[str, Any]:
    """:returns: ``{"files": n, "bytes": total}`` for a file or directory tree."""
    if os.path.isfile(path):
        return {"files": 1, "bytes": os.path.getsize(path)}
    files = 0
    total = 0
    for current, _, names in os.walk(path):
        for name in names:
            full = os.path.join(current, name)
            if os.path.isfile(full):
                files += 1
                total += os.path.getsize(full)
    return {"files": files, "bytes": total}


def sha256(path: str, chunk: int = 1 << 20) -> str:
    """:returns: The file's SHA-256, read in ``chunk``-sized pieces so a 7 GB tarball fits in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build(root: str = ".", data_dir: Optional[str] = None, dump_dir: Optional[str] = None,
          tarball: Optional[str] = None, bump: Optional[str] = None,
          set_version: Optional[str] = None, include_retired: bool = False,
          archive: bool = True, dry_run: bool = False) -> Dict[str, Any]:
    """
    Builds the Zenodo archive directory, and optionally the tarball.

    :param root: Repository root.
    :param data_dir: ``data/`` location; ``<root>/data`` when omitted.
    :param dump_dir: Archive directory; ``<data_dir>/zenodo_dump`` when omitted.
    :param tarball: Tarball path; ``<data_dir>/zenodo_dump-v<version>.tar.gz`` when omitted, so two
        releases cannot share a filename.
    :param bump: ``major``, ``minor`` or ``patch`` -- bump the version before building. Mutually
        exclusive with ``set_version``.
    :param set_version: An explicit version to write before building.
    :param include_retired: Archive ``models_retired/`` too. Off by default: those models were
        withdrawn from service, and publishing one invites its use.
    :param archive: Create the tarball. ``False`` builds the directory only.
    :param dry_run: Change nothing; report what would happen.
    :returns: A summary -- ``version``, ``dump_dir``, ``tarball``, ``tarball_sha256``,
        ``contents`` (per-entry file counts and sizes), ``version_files`` and ``stale_removed``.
    :raises ValueError: When both ``bump`` and ``set_version`` are given.
    :raises FileNotFoundError: When a declared release tree is missing -- an archive silently
        short a release directory is worse than a failed build.
    """
    if bump and set_version:
        raise ValueError("Pass --bump or --version, not both.")

    root = os.path.abspath(root)
    data_dir = os.path.abspath(data_dir or os.path.join(root, "data"))
    dump_dir = os.path.abspath(dump_dir or os.path.join(data_dir, "zenodo_dump"))

    current = read_version(root)
    version = current
    version_files: List[str] = []
    if bump:
        version = bump_version(current, bump)
        version_files = apply_version(root, version, dry_run=dry_run)
        logger.info("Version bumped (%s): %s -> %s", bump, current, version)
    elif set_version:
        parse_version(set_version)
        version = set_version
        version_files = apply_version(root, version, dry_run=dry_run)
        logger.info("Version set explicitly: %s -> %s", current, version)
    else:
        logger.info("Version unchanged at %s (pass --bump major|minor|patch to raise it).", version)

    missing = [t for t in RELEASE_TREES if not os.path.exists(os.path.join(data_dir, t))]
    if missing:
        raise FileNotFoundError(
            "Release directories missing from "
            f"{data_dir}: {', '.join(missing)}. Run the pipeline steps that produce them before "
            "archiving; an archive short a release tree would look complete.")

    excludes = tuple(EXCLUDES)
    if not include_retired:
        excludes += (RETIRED_SUBPATH + "/",)

    before = set()
    if os.path.isdir(dump_dir):
        for current_dir, _, names in os.walk(dump_dir):
            for name in names:
                before.add(os.path.relpath(os.path.join(current_dir, name), dump_dir))

    if not dry_run:
        os.makedirs(dump_dir, exist_ok=True)

    contents: Dict[str, Dict[str, Any]] = {}
    deleted: List[str] = []
    for tree in RELEASE_TREES:
        source = os.path.join(data_dir, tree)
        deleted += _rsync(source, os.path.join(dump_dir, tree), excludes, dry_run, prefix=tree)
        if not dry_run:
            contents[tree] = _tree_stats(os.path.join(dump_dir, tree))

    for relative in DATA_FILES:
        source = os.path.join(data_dir, relative)
        if not os.path.exists(source):
            logger.warning("Declared data file absent, not archived: %s", relative)
            continue
        target = os.path.join(dump_dir, os.path.basename(relative))
        deleted += _rsync(source, target, excludes, dry_run)
        if not dry_run:
            contents[os.path.basename(relative)] = _tree_stats(target)

    for relative in REPO_PATHS:
        source = os.path.join(root, relative)
        if not os.path.exists(source):
            logger.warning("Declared repository path absent, not archived: %s", relative)
            continue
        deleted += _rsync(source, os.path.join(dump_dir, relative), excludes, dry_run,
                          prefix=relative if os.path.isdir(source) else "")
        if not dry_run:
            contents[relative] = _tree_stats(os.path.join(dump_dir, relative))

    # rsync --delete only prunes inside the directories it syncs; the archive ROOT is assembled
    # from many separate calls, so an entry nobody declares any more would sit there untouched.
    # The archive must be exactly what the recipe declares, so anything else is removed.
    expected_top = (set(RELEASE_TREES)
                    | {os.path.basename(p) for p in DATA_FILES}
                    | {p.split("/")[0] for p in REPO_PATHS}
                    | {"VERSION", "MANIFEST.json"})
    pruned: List[str] = []
    if os.path.isdir(dump_dir):
        for name in sorted(os.listdir(dump_dir)):
            if name in expected_top:
                continue
            target = os.path.join(dump_dir, name)
            pruned.append(name)
            logger.warning("Pruning undeclared archive entry: %s", name)
            if not dry_run:
                shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)
    deleted += pruned

    after = set()
    if os.path.isdir(dump_dir) and not dry_run:
        for current_dir, _, names in os.walk(dump_dir):
            for name in names:
                after.add(os.path.relpath(os.path.join(current_dir, name), dump_dir))
    stale_removed = sorted(before - after) if not dry_run else sorted(set(deleted))

    manifest: Dict[str, Any] = {
        "version": version,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "includes_retired_models": bool(include_retired),
        "excluded_patterns": list(excludes),
        "contents": contents,
        "files_removed_as_stale": stale_removed,
    }

    if not dry_run:
        with open(os.path.join(dump_dir, "VERSION"), "w", encoding="utf-8") as handle:
            handle.write(version + "\n")
        with open(os.path.join(dump_dir, "MANIFEST.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    tarball_path = tarball or os.path.join(data_dir, f"zenodo_dump-v{version}.tar.gz")
    tarball_hash = None
    if archive and not dry_run:
        logger.info("Creating %s ...", tarball_path)
        subprocess.run(
            ["tar", "-czf", tarball_path, "-C", os.path.dirname(dump_dir),
             os.path.basename(dump_dir)], check=True)
        tarball_hash = sha256(tarball_path)
        with open(tarball_path + ".sha256", "w", encoding="utf-8") as handle:
            handle.write(f"{tarball_hash}  {os.path.basename(tarball_path)}\n")
        logger.info("Archive %s (%.2f GB), sha256 %s", tarball_path,
                    os.path.getsize(tarball_path) / 1e9, tarball_hash)

    total_files = sum(entry["files"] for entry in contents.values())
    total_bytes = sum(entry["bytes"] for entry in contents.values())
    logger.info("Zenodo dump v%s: %d file(s), %.2f GB across %d entries; %d stale file(s) removed.",
                version, total_files, total_bytes / 1e9, len(contents), len(stale_removed))

    return {
        "version": version, "previous_version": current, "dump_dir": dump_dir,
        "tarball": tarball_path if archive else None, "tarball_sha256": tarball_hash,
        "contents": contents, "version_files": version_files, "stale_removed": stale_removed,
        "total_files": total_files, "total_bytes": total_bytes, "dry_run": bool(dry_run),
    }
