from sample.config import DATA_DIR
import os
import re
import sys
import json
import time
import glob
import logging
import requests
import pandas as pd
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.databases")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MAX_RETRIES = 10
SLEEP_TIME = 2

def request_or_retry(method: str, url: str, data: Optional[Any] = None,
                     json_data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None,
                     timeout: int = 30, verify: bool = False) -> Optional[Any]:
    """
    Issues an HTTP request, retrying on transient failure.

    Every scraper in this module goes through here so that retry policy, timeouts and TLS
    verification are decided in one place rather than per call site.

    :param method: HTTP verb, e.g. ``"get"`` or ``"post"``.
    :param url: Target URL.
    :param data: Form-encoded body, passed straight to ``requests``.
    :param json_data: JSON body, passed straight to ``requests``.
    :param params: Query-string parameters.
    :param timeout: Per-attempt timeout in seconds.
    :param verify: Whether to verify the TLS certificate. Several of these sources present
        expired or mismatched certificates, hence the permissive default.
    :returns: The ``requests.Response`` on success, or ``None`` once retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            if method.upper() == 'POST':
                response = requests.post(url, data=data, json=json_data, params=params, allow_redirects=True, timeout=timeout, verify=verify)
            else:
                response = requests.get(url, data=data, json=json_data, params=params, allow_redirects=True, timeout=timeout, verify=verify)
            if response.status_code == 200:
                return response
        except Exception as e:
            logger.debug(f"Request failed for {url} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
        time.sleep(SLEEP_TIME)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}")

# ==========================================
# ADP DATABASE
# ==========================================

def fetch_adp_peptide(peptide_id: str, output_dir: str) -> bool:
    """
    Downloads one ADP (APD) peptide record.

    :param peptide_id: APD identifier, e.g. ``"AP00001"``.
    :param output_dir: Directory to write the record into.
    :returns: ``True`` if a record was written, ``False`` on failure or an empty response.
    """
    csv_path = os.path.join(output_dir, f"{peptide_id}.csv")
    if os.path.exists(csv_path):
        return True
    try:
        pid_num = peptide_id[2:] if peptide_id.startswith("AP") else peptide_id
        response = request_or_retry('post', 'https://aps.unmc.edu/database/peptide', data={'ID': pid_num}, verify=False)
        tables = pd.read_html(response.text)
        if tables:
            df_data = tables[0].T
            df_data.to_csv(csv_path, index=False)
            return True
    except Exception as e:
        logger.debug(f"Failed to fetch ADP peptide {peptide_id}: {e}")
    return False

def download_adp_data(output_dir: str, max_workers: int = 10) -> None:
    """
    Scrapes the full ADP (APD) database.

    :param output_dir: Directory for the downloaded records.
    :param max_workers: Concurrent download threads.
    :returns: ``None``; progress and failures are logged.
    """
    logger.info("Downloading ADP data...")
    peptides_dir = os.path.join(output_dir, "peptides")
    os.makedirs(peptides_dir, exist_ok=True)

    fasta_path = os.path.join(output_dir, "peptides.fasta")
    if not os.path.exists(fasta_path):
        response = request_or_retry('get', 'https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta', verify=False)
        # APD fasta starts with extra headers, remove them to keep it clean
        fasta_lines = response.text.split("\n")
        # Find first line starting with '>'
        start_idx = 0
        for idx, line in enumerate(fasta_lines):
            if line.startswith(">"):
                start_idx = idx
                break
        with open(fasta_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fasta_lines[start_idx:]))

    peptides = [record.id for record in SeqIO.parse(fasta_path, 'fasta')]
    logger.info(f"Parsed {len(peptides)} APD IDs from fasta. Downloading metadata concurrently...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_adp_peptide, pid, peptides_dir): pid for pid in peptides}
        for future in tqdm(as_completed(futures), total=len(futures), desc="ADP Metadata"):
            pid = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"Error downloading ADP ID {pid}: {e}")

    # Build consensus adb.csv from downloaded csvs
    all_adp_dfs = []
    unreadable_csvs = 0
    for csv_file in glob.glob(os.path.join(peptides_dir, "*.csv")):
        try:
            df = pd.read_csv(csv_file)
            all_adp_dfs.append(df)
        except Exception:
            unreadable_csvs += 1
    if unreadable_csvs:
        logger.warning(f"ADP consensus build: {unreadable_csvs} downloaded CSV(s) unreadable, "
                       f"of {len(all_adp_dfs) + unreadable_csvs} found.")
    if all_adp_dfs:
        df_consensus = pd.concat(all_adp_dfs, ignore_index=True)
        df_consensus.to_csv(os.path.join(output_dir, "adb.csv"), index=False)
        logger.info(f"Saved ADP consensus db to {output_dir}/adb.csv")

    logger.info(f"Completed ADP download. Downloaded {success_count}/{len(peptides)} records.")

# ==========================================
# DBAASP DATABASE
# ==========================================

def fetch_dbaasp_peptide(dbaasp_id: str, output_dir: str) -> bool:
    """
    Downloads one DBAASP peptide record as JSON.

    :param dbaasp_id: DBAASP identifier, e.g. ``"DBAASPN_10400"``.
    :param output_dir: Directory to write the JSON into.
    :returns: ``True`` if a record was written, ``False`` otherwise.
    """
    json_path = os.path.join(output_dir, f"{dbaasp_id}.json")
    if os.path.exists(json_path):
        return True
    try:
        response = request_or_retry("get", f"https://dbaasp.org/peptides/{dbaasp_id}", verify=False)
        data = response.json()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.debug(f"Failed to fetch DBAASP peptide {dbaasp_id}: {e}")
    return False

def download_dbaasp_data(output_dir: str, limit: int = 100, max_workers: int = 10) -> None:
    """
    Scrapes the full DBAASP database.

    DBAASP is the only source that publishes per-target measured values, so these records are what
    :mod:`sample.database_activities` later mines for structured activities.

    :param output_dir: Directory for the downloaded JSON records.
    :param limit: Page size for the id-listing endpoint.
    :param max_workers: Concurrent download threads.
    :returns: ``None``; progress and failures are logged.
    """
    logger.info("Downloading DBAASP data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total count
    try:
        response = request_or_retry('get', "https://dbaasp.org/peptides", params={'limit': 10, 'offset': 0})
        total_peptides = response.json().get('totalCount', 25000)
    except Exception as e:
        logger.error(f"Failed to query total count for DBAASP: {e}")
        total_peptides = 25000 # fallback
        
    logger.info(f"Total DBAASP peptides to query: {total_peptides}")
    
    # First, collect all peptide IDs via pagination
    logger.info("Fetching DBAASP ID lists...")
    dbaasp_ids = []
    
    def fetch_ids(offset: int) -> List[str]:
        """
        Fetches one page of DBAASP peptide ids.

        :param offset: Zero-based index of the first record in the page.
        :returns: The ids on that page; empty on failure, which ends pagination.
        """
        try:
            res = request_or_retry('get', "https://dbaasp.org/peptides", params={'limit': limit, 'offset': offset})
            data = res.json().get('data', [])
            return [row.get('dbaaspId') for row in data if row.get('dbaaspId')]
        except Exception as e:
            logger.warning(f"Error fetching IDs at offset {offset}: {e}")
            return []

    offsets = list(range(0, total_peptides, limit))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_ids, offset) for offset in offsets]
        for future in tqdm(as_completed(futures), total=len(futures), desc="DBAASP ID Retrieval"):
            dbaasp_ids.extend(future.result())
            
    logger.info(f"Collected {len(dbaasp_ids)} DBAASP IDs. Fetching individual details...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_dbaasp_peptide, pid, output_dir): pid for pid in dbaasp_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="DBAASP Metadata"):
            pid = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"Error fetching DBAASP peptide {pid}: {e}")
                
    logger.info(f"Completed DBAASP download. Fetched {success_count}/{len(dbaasp_ids)} records.")

# ==========================================
# DRAMP DATABASE
# ==========================================

#: DRAMP serves its CURRENT release from a path still named after 3.0, updating the files in
#: place. Verified 2026-10-16: ``DRAMP3.0_new/general_amps.txt`` answers 200 with 13,221,293
#: bytes and a 28-column header, while ``DRAMP5.0/`` and ``DRAMP4.0_new/`` both 404. The version
#: is therefore NOT in the URL and cannot be pinned there -- which is why this module detects the
#: release from the data and records it (see :func:`detect_dramp_release`, :func:`write_dramp_manifest`).
DRAMP_DOWNLOAD_BASE = ("https://dramp.cpu-bioinfor.org/downloads/download.php"
                       "?filename=download_data/DRAMP3.0_new/")

#: The file inventory, grouped by the subdirectory each file is read from.
#:
#: All three trees are listed because all three are READ. ``database_activities._dramp_rows``
#: globs ``Datasets/``, ``Activity/`` and ``Subclass/``, and ``sample.merge`` shares that same
#: iterator -- but only ``Datasets/`` was ever downloaded. The Activity and Subclass files present
#: on this host arrived by hand in March 2026, so a fresh clone built a database from a third of
#: the dump and nothing said so.
#:
#: The ``*_smiles.*`` downloads offered by DRAMP 5.0 are deliberately NOT here. The SMILES they
#: carry is already the 28th column of the Datasets files, so fetching them would add ~100 MB of
#: duplicate data -- and, worse, land extra ``*.txt`` files inside ``Datasets/`` where
#: ``_dramp_rows`` would read them as additional rows. That iterator de-duplicates by DRAMP_ID
#: keeping the FIRST row seen, so a two-column SMILES row could shadow the full record for the
#: same peptide.
DRAMP_FILES: Dict[str, List[str]] = {
    "Datasets": [
        "general_amps.txt",
        "patent_amps.txt",
        "clinical_amps.xlsx",      # xlsx upstream; there is no .txt for this category
        "specific_amps.txt",
        "stability_amps.txt",
        "expanded_amps.txt",
    ],
    "Activity": [
        "Antibacterial_amps.txt",
        "Anti-Gram-positive_amps.txt",
        # Upstream filename, misspelled at the source: the "negative" is missing. Kept verbatim
        # because it is what the server serves; renaming it locally would break the next refresh.
        "Anti-Gram-_amps.txt",
        "Antifungal_amps.txt",
        "Antiviral_amps.txt",
        "Anticancer_amps.txt",
        "Antiprotozoal_amps.txt",
        "Antiparasitic_amps.txt",
        "Antiinfective_amps.txt",
        "Antiinflammatory_amps.txt",
    ],
    "Subclass": [
        "natural_amps.txt",
        "synthetic_amps.txt",
        "stapled_amps.txt",
        "candidate_amps.txt",
    ],
}

#: Entry counts as PUBLISHED in each DRAMP paper, used as evidence when identifying which release
#: is on disk. Not a validation target -- a mismatch is a question, not a failure.
#:
#: 5.0: Ma, T., Huang, Y., Liu, H., Wu, W., Ren, L., Akhtar, F., Nawaz, M., Hua, Z., Zheng, H. and
#:      Lao, X. "DRAMP 5.0: an update integrating sequence and structure in a novel FASTS format
#:      for AI-driven design of antimicrobial peptides." Nucleic Acids Research,
#:      doi:10.1093/nar/gkag989 (advance article, 15 Oct 2026). Verified against the publisher
#:      record; this was previously attributed here to "Shi et al.", who are the DRAMP 3.0
#:      authors, and the manuscript's bibliography inherited that error.
#:      32,296 total (2,036 newly added): 12,784 general, 18,715 patent, 96 clinical, 110
#:      stability, 179 expanded, 412 specific. 22,298 SMILES annotations; 16,016 predicted
#:      structures (1,570 in 4.0).
#: 4.0: 30,260 total: 11,612 general, 17,886 patent, 377 stapled.
#: 3.0: 22,259 total. 2.0: 19,899 total.
#:
#: Only `general` and `patent` are used for matching. The published "specific" row counts the
#: STAPLED dataset (Table 1 labels it so), which is not what ``specific_amps.txt`` contains --
#: that file held 6,286 rows while 4.0 published 377 -- so keying on it would mis-identify every
#: release.
DRAMP_PUBLISHED_COUNTS: Dict[str, Dict[str, int]] = {
    "5.0": {"general": 12784, "patent": 18715, "total": 32296},
    "4.0": {"general": 11612, "patent": 17886, "total": 30260},
    "3.0": {"total": 22259},
    "2.0": {"total": 19899},
}

#: The column DRAMP 5.0 appended to every Datasets export. Its presence is the single clearest
#: signal of a 5.0-or-later dump: 5.0 is the release that introduced SMILES annotation, and the
#: schema change is append-only, so nothing else in the parsing path has to move.
DRAMP_SMILES_COLUMN = "SMILES"

#: Where the provenance record lands, inside the DRAMP tree it describes.
DRAMP_MANIFEST_NAME = "MANIFEST.json"


def _dramp_file_report(path: str) -> Dict[str, Any]:
    """
    Size, digest and shape of one downloaded DRAMP file.

    :param path: The local file.
    :returns: A dict with ``bytes``, ``sha256``, and -- for a readable TSV -- ``rows``,
        ``unique_ids`` and ``has_smiles``. Unreadable files report ``error`` rather than being
        omitted: a file that cannot be parsed is the single most useful thing a manifest can say.
    """
    import hashlib

    report: Dict[str, Any] = {}
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
        report["bytes"] = len(blob)
        report["sha256"] = hashlib.sha256(blob).hexdigest()
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if not path.endswith(".txt"):
        return report
    try:
        # engine="python" with on_bad_lines="skip": these exports contain embedded newlines and
        # unbalanced quotes inside free-text Comments fields, which the C parser rejects outright.
        frame = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip", engine="python")
        report["rows"] = int(len(frame))
        if "DRAMP_ID" in frame.columns:
            report["unique_ids"] = int(frame["DRAMP_ID"].nunique())
        report["has_smiles"] = DRAMP_SMILES_COLUMN in frame.columns
        report["columns"] = int(len(frame.columns))
    except Exception as e:                                 # noqa: BLE001 -- recorded, not fatal
        report["error"] = f"{type(e).__name__}: {e}"
    return report


def detect_dramp_release(dramp_dir: str) -> Dict[str, Any]:
    """
    Works out which DRAMP release is on disk, from the data rather than from the URL.

    :param dramp_dir: The local ``.../antimicrobial_databases/dramp`` tree.
    :returns: ``{"release", "confidence", "evidence"}``. ``release`` is ``None`` when the dump
        cannot be identified; it is never guessed.

    This exists because the download path is version-less and is updated in place. The
    consequence was already live on this host: the URLs say ``DRAMP3.0_new``, the files
    downloaded through them on 2026-03-23 are DRAMP **4.0** (general 11,612 and patent 17,886
    match 4.0's published table exactly), and the same URLs now serve 5.0. Nothing recorded any
    of that, so "which release is this database built from?" -- a question any reader of a
    published resource may ask -- had no answer short of counting rows by hand.

    Two signals, in order of strength:

    1. the presence of the ``SMILES`` column, which 5.0 introduced;
    2. the ``general`` and ``patent`` row counts against :data:`DRAMP_PUBLISHED_COUNTS`.
    """
    datasets = os.path.join(dramp_dir, "Datasets")
    evidence: Dict[str, Any] = {}

    counts: Dict[str, int] = {}
    has_smiles = False
    for category, filename in (("general", "general_amps.txt"), ("patent", "patent_amps.txt")):
        path = os.path.join(datasets, filename)
        if not os.path.exists(path):
            evidence[category] = "missing"
            continue
        report = _dramp_file_report(path)
        evidence[category] = report
        if isinstance(report.get("rows"), int):
            counts[category] = report["rows"]
        has_smiles = has_smiles or bool(report.get("has_smiles"))

    evidence["has_smiles_column"] = has_smiles

    # A release only matches on keys it actually publishes AND that were actually counted.
    # Requiring a non-empty overlap is load-bearing: DRAMP 2.0 and 3.0 publish only a total, so
    # an `all()` over an empty set of comparisons returns True and they matched every dump
    # vacuously -- which made a dump whose counts identify 4.0 exactly come back
    # "unidentified" because three releases tied.
    matched = []
    for release, published in DRAMP_PUBLISHED_COUNTS.items():
        shared = [key for key in counts if key in published]
        if not shared:
            continue
        if all(published[key] == counts[key] for key in shared):
            matched.append(release)
    if len(matched) == 1:
        release = matched[0]
        # The two signals must agree. A dump whose counts say 4.0 but which carries a SMILES
        # column is neither release, and saying so is more useful than picking one.
        if has_smiles and release != "5.0":
            return {"release": None, "confidence": "conflicting",
                    "evidence": {**evidence,
                                 "note": f"row counts match DRAMP {release} but the SMILES "
                                         f"column that 5.0 introduced is present"}}
        if not has_smiles and release == "5.0":
            return {"release": None, "confidence": "conflicting",
                    "evidence": {**evidence,
                                 "note": "row counts match DRAMP 5.0 but the SMILES column is "
                                         "absent"}}
        return {"release": release, "confidence": "counts+schema", "evidence": evidence}

    if has_smiles:
        # Counts did not match any published table -- expected for a dump taken between
        # releases, or once 5.x point updates land in place. The schema still bounds it.
        return {"release": "5.0+", "confidence": "schema-only", "evidence": evidence}
    return {"release": None, "confidence": "unidentified", "evidence": evidence}


def write_dramp_manifest(dramp_dir: str, fetched: Optional[Dict[str, str]] = None) -> str:
    """
    Records what is in the local DRAMP tree, so the release a database was built from is a fact
    rather than an inference.

    :param dramp_dir: The local ``.../antimicrobial_databases/dramp`` tree.
    :param fetched: Optional ``{relative path: source URL}`` for the files this run downloaded,
        distinguishing them from files that were already present.
    :returns: The manifest path.
    """
    manifest: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "download_base": DRAMP_DOWNLOAD_BASE,
        # Stated explicitly because it is the trap: the path names 3.0 and serves whatever is
        # current. Anyone reading this manifest later must not read "3.0" out of the URL.
        "download_base_note": ("DRAMP serves its current release from this version-less path and "
                               "updates it in place; the '3.0' in the URL does NOT indicate the "
                               "release. See the 'release' key."),
        "release": detect_dramp_release(dramp_dir),
        "files": {},
    }
    for subdir, filenames in DRAMP_FILES.items():
        for filename in filenames:
            relative = f"{subdir}/{filename}"
            path = os.path.join(dramp_dir, subdir, filename)
            if not os.path.exists(path):
                manifest["files"][relative] = {"present": False}
                continue
            entry = {"present": True, **_dramp_file_report(path)}
            entry["modified"] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                              time.localtime(os.path.getmtime(path)))
            if fetched and relative in fetched:
                entry["fetched_from"] = fetched[relative]
            manifest["files"][relative] = entry

    os.makedirs(dramp_dir, exist_ok=True)
    manifest_path = os.path.join(dramp_dir, DRAMP_MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path


def read_dramp_manifest(dramp_dir: str) -> Optional[Dict[str, Any]]:
    """
    :param dramp_dir: The local DRAMP tree.
    :returns: The manifest, or ``None`` when it has not been written or cannot be read -- which
        is the normal state for a tree downloaded before manifests existed.
    """
    path = os.path.join(dramp_dir, DRAMP_MANIFEST_NAME)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def download_dramp_data(output_dir: str, refresh: bool = False,
                        archive: bool = True) -> Dict[str, Any]:
    """
    Downloads the DRAMP exports that this pipeline reads, and records what it got.

    :param output_dir: The ``.../antimicrobial_databases/dramp`` directory.
    :param refresh: Re-download files that already exist. **Off by default**, and that default is
        why this host sat on DRAMP 4.0 for seven months: the old loop did
        ``if os.path.exists(dest): continue``, so once a file was on disk no later release could
        ever replace it, and a new upstream release was indistinguishable from an up-to-date
        local copy. Skipping is still the right default for resuming an interrupted step-5 run;
        picking up a release is now an explicit request.
    :param archive: When refreshing, move the existing tree aside to
        ``dramp_<release>_<date>/`` instead of overwriting it. The current database was built
        from the files being replaced, so discarding them would make that database
        irreproducible -- and a source dump is small next to the corpus it annotates.
    :returns: A summary dict: ``downloaded``, ``skipped``, ``failed``, ``archived``,
        ``manifest`` and ``release``.

    Every category this project reads is fetched, not only ``Datasets/`` -- see
    :data:`DRAMP_FILES`.
    """
    logger.info("Downloading DRAMP data%s...", " (refresh)" if refresh else "")

    archived = None
    if refresh and archive and os.path.isdir(os.path.join(output_dir, "Datasets")):
        before = detect_dramp_release(output_dir)
        tag = (before.get("release") or "unidentified").replace(".", "_").replace("+", "plus")
        archived = f"{output_dir.rstrip('/')}_{tag}_{time.strftime('%Y%m%d')}"
        if os.path.exists(archived):
            archived = f"{archived}_{time.strftime('%H%M%S')}"
        os.rename(output_dir, archived)
        logger.info(f"Archived the existing DRAMP tree (detected release: "
                    f"{before.get('release') or 'unidentified'}) to {archived}")

    fetched: Dict[str, str] = {}
    downloaded = skipped = 0
    failed: Dict[str, str] = {}
    targets = [(subdir, filename) for subdir, names in DRAMP_FILES.items() for filename in names]

    for subdir, filename in tqdm(targets, desc="DRAMP"):
        target_dir = os.path.join(output_dir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path) and not refresh:
            skipped += 1
            continue
        url = DRAMP_DOWNLOAD_BASE + filename
        try:
            res = request_or_retry('get', url, timeout=300)
            if res is None:
                raise RuntimeError("no response")
            # Written via a temporary file and renamed: a refresh that dies mid-write would
            # otherwise leave a truncated export in place of a complete one, and a short TSV
            # parses fine -- it just silently loses its tail.
            temporary = f"{dest_path}.part"
            with open(temporary, 'wb') as handle:
                handle.write(res.content)
            os.replace(temporary, dest_path)
            fetched[f"{subdir}/{filename}"] = url
            downloaded += 1
        except Exception as e:                            # noqa: BLE001 -- counted, not fatal
            failed[f"{subdir}/{filename}"] = f"{type(e).__name__}: {e}"
            logger.error(f"Failed downloading DRAMP file {url}: {e}")

    manifest_path = write_dramp_manifest(output_dir, fetched=fetched)
    release = detect_dramp_release(output_dir)
    logger.info(f"DRAMP: {downloaded} downloaded, {skipped} already present, {len(failed)} failed. "
                f"Detected release: {release.get('release') or 'unidentified'} "
                f"({release.get('confidence')}). Manifest: {manifest_path}")
    if failed:
        logger.warning(f"DRAMP files that did not download: {failed}")
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed,
            "archived": archived, "manifest": manifest_path, "release": release}

# ==========================================
# DBAMP DATABASE
# ==========================================

def fetch_dbamp_peptide(dbamp_id: str, output_dir: str) -> bool:
    """
    Downloads one dbAMP peptide record.

    :param dbamp_id: dbAMP identifier, e.g. ``"dbAMP_00013"``.
    :param output_dir: Directory to write the record into.
    :returns: ``True`` if a record was written, ``False`` otherwise.
    """
    extracted_dir = os.path.join(output_dir, "extracted_metadata", dbamp_id)
    if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) >= 2:
        return True
        
    os.makedirs(extracted_dir, exist_ok=True)
    try:
        response = request_or_retry('get', f'https://ycclab.cuhk.edu.cn/dbAMP/information.php?db={dbamp_id}', verify=False)
        tables = pd.read_html(response.text)
        for idx, table in enumerate(tables):
            table.to_csv(os.path.join(extracted_dir, f"{idx}.csv"), index=False)
        return True
    except Exception as e:
        logger.debug(f"Failed to fetch DBAMP peptide {dbamp_id}: {e}")
    return False

def download_dbamp_data(output_dir: str, max_workers: int = 10) -> None:
    """
    Scrapes the full dbAMP database.

    :param output_dir: Directory for the downloaded records.
    :param max_workers: Concurrent download threads.
    :returns: ``None``; progress and failures are logged.
    """
    logger.info("Downloading DBAMP data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Download zip file and extract it
    zip_path = os.path.join(output_dir, "dbAMP2024.zip")
    if not os.path.exists(zip_path):
        res = request_or_retry('get', 'https://ycclab.cuhk.edu.cn/dbAMP/download/3.0/dbAMP2024.zip', verify=False)
        with open(zip_path, 'wb') as f:
            f.write(res.content)
            
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            
    fasta_path = os.path.join(output_dir, "dbAMP2024", "dbAMP3.fasta")
    if not os.path.exists(fasta_path):
        fasta_path = os.path.join(output_dir, "dbAMP3.fasta") # fallback
        
    if not os.path.exists(fasta_path):
        logger.error(f"dbAMP3.fasta not found at {fasta_path}")
        return

    records = list(SeqIO.parse(fasta_path, 'fasta'))
    logger.info(f"Loaded {len(records)} DBAMP IDs. Fetching metadata concurrently...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_dbamp_peptide, record.id, output_dir): record.id for record in records}
        for future in tqdm(as_completed(futures), total=len(futures), desc="DBAMP Metadata"):
            pid = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"Error fetching DBAMP peptide {pid}: {e}")
                
    logger.info(f"Completed DBAMP download. Fetched {success_count}/{len(records)} records.")

# ==========================================
# CAMP DATABASE
# ==========================================

def fetch_camp_peptide(camp_id: str, seq_href: str, output_dir: str) -> bool:
    """
    Downloads one CAMP peptide record.

    :param camp_id: CAMP identifier, e.g. ``"CAMPSQ10000"``.
    :param seq_href: Relative URL of the entry's detail page, taken from the listing.
    :param output_dir: Directory to write the record into.
    :returns: ``True`` if a record was written, ``False`` otherwise.
    """
    csv_path = os.path.join(output_dir, f"{camp_id}.csv")
    if os.path.exists(csv_path):
        return True
    try:
        url = f"https://camp.bicnirrh.res.in/{seq_href}"
        res = request_or_retry('get', url, verify=False)
        tables = pd.read_html(res.text)
        if len(tables) >= 3:
            df_metadata = tables[2][1::]
            df_metadata = df_metadata.rename(columns={0: 'key', 1:'value'})
            df_metadata['key'] = df_metadata['key'].map(lambda key: str(key).strip(":").strip(' '))
            df_metadata.to_csv(csv_path, index=False)
            return True
    except Exception as e:
        logger.debug(f"Failed to fetch CAMP peptide {camp_id}: {e}")
    return False

def download_camp_data(output_dir: str, num_pages: int = 1000, max_workers: int = 10) -> None:
    """
    Downloads CAMP database using direct HTTP requests (WITHOUT selenium!).
    This is extremely fast, headless-friendly, and lightweight.
    """
    logger.info(f"Downloading CAMP data for {num_pages} pages...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fetch sequence display links page by page
    logger.info("Extracting CAMP sequence links...")
    seq_links = []
    
    def fetch_page_links(page_num: int) -> List[tuple]:
        """
        Fetches one page of the CAMP sequence listing.

        :param page_num: 1-based page number.
        :returns: ``(camp_id, detail_href)`` pairs found on that page.
        """
        url = f'https://camp.bicnirrh.res.in/seqDb.php?page={page_num}'
        try:
            res = request_or_retry('get', url, verify=False)
            links = re.findall(r'href="(seqDisp\.php\?id=[^"]+)"', res.text)
            return links
        except Exception as e:
            logger.warning(f"Failed to fetch CAMP page {page_num}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_page_links, page) for page in range(num_pages)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="CAMP Link Extraction"):
            seq_links.extend(future.result())
            
    # Remove duplicates
    seq_links = list(set(seq_links))
    logger.info(f"Extracted {len(seq_links)} unique CAMP peptide links. Fetching metadata concurrently...")
    
    # 2. Fetch individual peptide pages concurrently
    success_count = 0
    unparseable_links = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for seq_href in seq_links:
            try:
                camp_id = seq_href.split('id=')[1]
                futures[executor.submit(fetch_camp_peptide, camp_id, seq_href, output_dir)] = camp_id
            except Exception:
                unparseable_links += 1
        if unparseable_links:
            logger.warning(f"CAMP link parsing: {unparseable_links} of {len(seq_links)} links had "
                           f"no 'id=' segment and were skipped.")

        for future in tqdm(as_completed(futures), total=len(futures), desc="CAMP Metadata"):
            pid = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"Error fetching CAMP peptide {pid}: {e}")
                
    logger.info(f"Completed CAMP download. Fetched {success_count}/{len(seq_links)} records.")

# ==========================================
# DATABASE PAPERS DOWNLOAD
# ==========================================

def get_doi_from_pmid(pmid: str, email: str) -> str:
    """
    Resolves a PubMed ID to a DOI via NCBI E-utilities.

    :param pmid: PubMed identifier.
    :param email: Address registered with NCBI, required for rate-limit accounting.
    :returns: The DOI, or ``""`` when the record has none or the lookup fails.
    """
    from Bio import Entrez
    Entrez.email = email
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        for article in records.get('PubmedArticle', []):
            article_ids = article.get('PubmedData', {}).get('ArticleIdList', [])
            for article_id in article_ids:
                if article_id.attributes.get('IdType') == 'doi':
                    return str(article_id)
    except Exception as e:
        logger.debug(f"NCBI Entrez DOI lookup failed for PMID {pmid}: {e}")
    return None

def get_doi_from_title(title: str) -> str:
    """
    Resolves a publication title to a DOI via the Crossref API.

    The fallback for records whose source database gives a citation but no identifier.

    :param title: Publication title to search for.
    :returns: The DOI of the best match, or ``""`` when nothing matched confidently.
    """
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 1, "select": "DOI"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI")
    except Exception as e:
        logger.debug(f"Crossref DOI lookup failed for title {title}: {e}")
    return None

def download_database_papers(database: str, email: str, routers: Optional[list] = None,
                             max_workers: int = 4) -> None:
    """
    Downloads full-text paper PDFs linked inside database entries.
    """
    from sample.pdf import download_paper_pdf
    logger.info(f"Downloading papers linked in database '{database}'...")
    
    papers_dir = f"{DATA_DIR}/antimicrobial_databases/papers/{database}"
    os.makedirs(papers_dir, exist_ok=True)
    
    tasks = [] # list of (identifier, title/pmid, paper_dir)
    
    if database == 'adp':
        adb_path = f"{DATA_DIR}/antimicrobial_databases/adp/adb.csv"
        if not os.path.exists(adb_path):
            logger.error(f"ADP consensus file not found at {adb_path}")
            return
        df_adp = pd.read_csv(adb_path)
        for _, row in df_adp.iterrows():
            title = row.get('Title:')
            apd_id = row.get('APD ID:')
            if pd.notna(title) and title and pd.notna(apd_id) and apd_id:
                if not str(title).startswith("Title:"):
                    tasks.append((apd_id, "title", str(title), os.path.join(papers_dir, str(apd_id))))
                    
    elif database == 'dbamp':
        dbamp_errors: Dict[str, int] = {}
        dbamp_files = glob.glob(f"{DATA_DIR}/antimicrobial_databases/dbamp/extracted_metadata/*/1.csv")
        for csv_file in dbamp_files:
            try:
                df = pd.read_csv(csv_file, header=None)
                dbamp_id = None
                pubmed_id = None
                for _, row in df.iterrows():
                    # Format: first row usually has ID, PubMed is key
                    if pd.isna(row.values[0]):
                        continue
                    if row.values[0] == 'dbAMP ID' or (row.values[0] == 0 and not dbamp_id):
                        dbamp_id = row.values[1]
                    elif str(row.values[0]).lower() == 'pubmed':
                        pubmed_id = row.values[1]
                if dbamp_id and pubmed_id:
                    tasks.append((dbamp_id, "pmid", str(pubmed_id), os.path.join(papers_dir, str(dbamp_id))))
            except Exception as e:
                dbamp_errors[type(e).__name__] = dbamp_errors.get(type(e).__name__, 0) + 1
        if dbamp_errors:
            logger.warning(f"dbAMP paper queueing: {sum(dbamp_errors.values())} of "
                           f"{len(dbamp_files)} files failed: {dbamp_errors}")

    elif database == 'dramp':
        dramp_errors: Dict[str, int] = {}
        dramp_files = glob.glob(f"{DATA_DIR}/antimicrobial_databases/dramp/Datasets/*.txt")
        for txt_file in dramp_files:
            try:
                df_dramp = pd.read_csv(txt_file, sep='\t')
                for _, row in df_dramp.iterrows():
                    dramp_id = row.get('DRAMP_ID')
                    title = row.get('Title')
                    if dramp_id and title:
                        tasks.append((dramp_id, "title", str(title), os.path.join(papers_dir, str(dramp_id))))
            except Exception as e:
                dramp_errors[type(e).__name__] = dramp_errors.get(type(e).__name__, 0) + 1
        if dramp_errors:
            logger.warning(f"DRAMP paper queueing: {sum(dramp_errors.values())} of "
                           f"{len(dramp_files)} files failed: {dramp_errors}")

    elif database == 'dbaasp':
        dbaasp_errors: Dict[str, int] = {}
        dbaasp_files = glob.glob(f"{DATA_DIR}/antimicrobial_databases/dbaasp/*.json")
        for json_file in dbaasp_files:
            try:
                dbaasp_id = os.path.basename(json_file).split('.')[0]
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                articles = json_data.get('articles', [])
                for idx, article in enumerate(articles):
                    title = article.get('title')
                    if title:
                        tasks.append((f"{dbaasp_id}_{idx}", "title", str(title), os.path.join(papers_dir, dbaasp_id)))
            except Exception as e:
                dbaasp_errors[type(e).__name__] = dbaasp_errors.get(type(e).__name__, 0) + 1
        if dbaasp_errors:
            logger.warning(f"DBAASP paper queueing: {sum(dbaasp_errors.values())} of "
                           f"{len(dbaasp_files)} files failed: {dbaasp_errors}")

    logger.info(f"Queued {len(tasks)} papers for {database}.")
    
    # Process tasks concurrently
    def worker(task: tuple) -> bool:
        """
        Downloads the paper PDF for one database entry.

        :param task: ``(entry_id, source_type, identifier, paper_dir)``.
        :returns: ``True`` if a PDF is present afterwards, including when it already existed.
        """
        entry_id, src_type, identifier, paper_dir = task
        pdf_path = os.path.join(paper_dir, f"paper_{entry_id}.pdf" if "_" in entry_id else "paper.pdf")
        if os.path.exists(pdf_path):
            return True
            
        doi_number = None
        if src_type == "pmid":
            doi_number = get_doi_from_pmid(identifier, email)
        elif src_type == "title":
            doi_number = get_doi_from_title(identifier)
            
        if doi_number:
            return download_paper_pdf(doi_number, pdf_path, email, routers)
        return False

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{database} Papers"):
            task = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to fetch paper for {task[0]}: {e}")
                
    logger.info(f"Finished. Successfully downloaded {success_count}/{len(tasks)} papers for {database}.")
