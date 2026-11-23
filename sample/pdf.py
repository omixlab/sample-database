import os
import sys
import time
import glob
import socket
import requests
import re
import base64
import shutil
import tempfile
import gc
import logging
import multiprocessing as mp
from functools import partial
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import uuid

from typing import Any, List, Optional

logger = logging.getLogger("sample.pdf")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_valid_pdf(path: str, min_bytes: int = 1000) -> bool:
    """
    Whether a file on disk is a usable PDF rather than a truncated or error-page download.

    Checked by magic bytes, not size alone. ``download_paper_pdf`` short-circuits to success for any
    existing file over 1000 bytes, so a partial write that clears that bound is accepted as a
    completed download forever after -- the paper is never retried and the corrupt file is what the
    extraction step reads.

    :param path: File to inspect.
    :param min_bytes: Smallest plausible PDF; anything shorter is a stub or an error page.
    :returns: True when the file exists, clears ``min_bytes`` and starts with ``%PDF``.
    """
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as handle:
            return handle.read(4) == b"%PDF"
    except OSError:
        return False


def write_pdf_atomically(data: bytes, final_output_path: str) -> bool:
    """
    Writes PDF bytes through a temporary file and renames them into place.

    A direct ``open(dest, "wb")`` truncates the destination before the first byte arrives, so an
    interrupted write leaves a corrupt file at the path a later run will treat as already
    downloaded. ``os.replace`` is atomic on POSIX: a reader sees the whole file or nothing.

    :param data: Complete PDF bytes.
    :param final_output_path: Destination path.
    :returns: True when a valid PDF was put in place; the temporary file is removed otherwise.
    """
    os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)
    tmp_path = f"{final_output_path}.part"
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(data)
        if not is_valid_pdf(tmp_path):
            os.remove(tmp_path)
            return False
        os.replace(tmp_path, final_output_path)
        return True
    except OSError as exc:
        logger.debug(f"Atomic PDF write failed for {final_output_path}: {exc}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def _move_into_place(source_path: str, final_output_path: str) -> bool:
    """
    Moves a finished download to its destination, atomically and across filesystems.

    ``os.replace`` alone is atomic but cannot cross a filesystem boundary, and the download
    directory here is a :class:`tempfile.TemporaryDirectory` while the destination is under
    ``data/`` -- so wherever ``/tmp`` is a separate mount, which includes the Docker setup this
    repo ships, it raises ``OSError: [Errno 18] Invalid cross-device link`` and the download is
    silently reported as a failure. ``shutil.move`` alone crosses boundaries but copies straight to
    the destination, leaving a partial file at the path a later run reads as "already downloaded".

    Copying beside the destination and renaming gets both properties: the copy may cross
    filesystems, and the rename that publishes it cannot be seen half-done.

    :param source_path: The downloaded file.
    :param final_output_path: Where it belongs.
    :returns: True when the file is in place; the temporary copy is removed otherwise.
    """
    tmp_path = f"{final_output_path}.part"
    try:
        try:
            os.replace(source_path, final_output_path)
            return True
        except OSError:
            # Same-filesystem rename was not possible; fall back to copy-then-rename.
            shutil.copy2(source_path, tmp_path)
            os.replace(tmp_path, final_output_path)
            try:
                os.remove(source_path)
            except OSError:
                pass
            return True
    except OSError as exc:
        logger.debug(f"Could not move {source_path} to {final_output_path}: {exc}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def verify_pdf_link(url: str) -> bool:
    """
    Checks whether a URL actually serves a PDF.

    Publisher links routinely resolve to an HTML landing page or a paywall notice, so the
    content type is checked before the download is attempted.

    :param url: Candidate PDF URL.
    :returns: ``True`` if the response looks like a PDF.
    """
    try:
        with requests.head(url, timeout=10, allow_redirects=True) as response:
            content_type = response.headers.get('Content-Type', '').lower()
            return 'application/pdf' in content_type
    except Exception:
        return False

def get_all_unpaywall_urls(doi: str, email: str) -> List[str]:
    """
    Collects every open-access PDF location Unpaywall knows for a DOI.

    All locations are returned rather than only the "best" one, because the best-scored link is
    frequently dead while a repository mirror still works.

    :param doi: DOI to look up.
    :param email: Address required by the Unpaywall API.
    :returns: Candidate PDF URLs, best-scored first. Empty when nothing open-access exists.
    """
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    found_urls = []
    try:
        with requests.Session() as session:
            with session.get(api_url, timeout=15) as response:
                if response.status_code == 200:
                    data = response.json()
                    locations = data.get("oa_locations", [])
                    for loc in locations:
                        url = loc.get("url_for_pdf") or loc.get("url")
                        if url and url not in found_urls:
                            found_urls.append(url)
                    
                    best_loc = data.get("best_oa_location")
                    if best_loc:
                        best_url = best_loc.get("url_for_pdf") or best_loc.get("url")
                        if best_url and best_url not in found_urls:
                            found_urls.insert(0, best_url)
    except Exception as e:
        logger.warning(f"Unpaywall API failure for DOI {doi}: {e}")
        
    return found_urls

def download_direct_pdf(url: str, output_path: str) -> bool:
    """
    Attempts to download a PDF directly via standard HTTP requests.
    Extremely fast and avoids Selenium overhead.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        with requests.get(url, headers=headers, timeout=20, stream=True, allow_redirects=True) as response:
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                # Also accept generic stream if the URL ends with .pdf
                if 'application/pdf' in content_type or url.lower().split('?')[0].endswith('.pdf'):
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    # Streamed to a temporary file beside the destination and renamed in, rather
                    # than written over the destination directly. Writing straight to output_path
                    # truncates it before the first byte arrives, so an interrupted transfer left a
                    # corrupt file exactly where a later run looks to decide the DOI is already
                    # done. The rename is atomic because .part is in the destination directory.
                    tmp_path = f"{output_path}.part"
                    try:
                        with open(tmp_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        if is_valid_pdf(tmp_path):
                            os.replace(tmp_path, output_path)
                            return True
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
    except Exception as e:
        logger.debug(f"Direct download failed for {url}: {e}")
    return False

def get_stealth_driver(download_dir: str, driver_path: Optional[str] = None) -> Any:
    """
    Builds a Selenium Chrome driver configured to look like a normal browser.

    Several publishers block obvious automation, so ``selenium-stealth`` is applied and the
    download directory is pre-set to avoid the native save dialog.

    :param download_dir: Directory Chrome should write downloads into.
    :param driver_path: Explicit chromedriver path; ``None`` lets webdriver-manager resolve it.
    :returns: A configured ``selenium.webdriver.Chrome`` instance.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium_stealth import stealth

    options = Options()
    options.add_argument("--headless=new") 
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-extensions")
    options.add_argument("--disk-cache-size=1")
    options.add_argument("--media-cache-size=1")
    options.add_argument("--js-flags=--max-old-space-size=256")
    options.add_argument("--disable-logging")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
        "profile.default_content_settings.popups": 0,
    }
    options.add_experimental_option("prefs", prefs)

    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True)
            
    return driver

def save_as_pdf(driver: Any, final_output_path: str) -> bool:
    """
    Prints the driver's current page to PDF via Chrome DevTools.

    The fallback for articles served as HTML with no downloadable PDF.

    :param driver: Selenium driver positioned on the page to capture.
    :param final_output_path: Destination path.
    :returns: ``True`` if a PDF was written.
    """
    print_options = {
        "printBackground": True,
        "paperWidth": 8.27,
        "paperHeight": 11.69,
        "scale": 1.0
    }
    pdf_data = driver.execute_cdp_cmd("Page.printToPDF", print_options)
    # Written atomically and validated, and the result is REPORTED. This wrote straight to the
    # destination and returned None, so the caller could not tell a successful capture from a
    # failed one -- and because download_paper_pdf accepts any existing file over 1000 bytes as a
    # completed download, a truncated print-to-PDF became a permanent success that no retry would
    # ever revisit.
    return write_pdf_atomically(base64.b64decode(pdf_data['data']), final_output_path)

def wait_for_download(download_dir: str, final_output_path: str, timeout: int = 60) -> bool:
    """
    Waits for Chrome to finish writing a download, then moves it into place.

    Chrome writes ``.crdownload`` while in flight, so completion is detected by that suffix
    disappearing rather than by the file merely existing.

    :param download_dir: Directory Chrome is downloading into.
    :param final_output_path: Where the completed file should end up.
    :param timeout: Seconds to wait before giving up.
    :returns: ``True`` if a completed download was moved into place.
    """
    end_time = time.time() + timeout
    while time.time() < end_time:
        crdownloads = glob.glob(os.path.join(download_dir, "*.crdownload"))
        if crdownloads:
            time.sleep(1)
            continue
            
        pdfs = glob.glob(os.path.join(download_dir, "*.pdf"))
        if pdfs:
            # Newest, not pdfs[0]. glob order is arbitrary, so with more than one file in the
            # directory -- a leftover from an earlier attempt on the same driver, say -- this could
            # move ANOTHER paper's PDF to this DOI's destination and record it as a success. That
            # misattribution is silent and survives into the database.
            downloaded_file = max(pdfs, key=os.path.getmtime)
            if not is_valid_pdf(downloaded_file):
                # A partial or error-page download is not a result. Removed so the next poll can
                # see a genuine one rather than returning this as success.
                try:
                    os.remove(downloaded_file)
                except OSError:
                    pass
                time.sleep(1)
                continue
            os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)
            if not _move_into_place(downloaded_file, final_output_path):
                time.sleep(1)
                continue
            return True
        time.sleep(1)
    return False

def process_url_with_selenium(driver: Any, target_url: str, output_path: str,
                              temp_dir: str) -> bool:
    """
    Attempts to obtain a PDF from one URL: direct download first, print-to-PDF as fallback.

    :param driver: Selenium driver to navigate with.
    :param target_url: URL to fetch.
    :param output_path: Destination path for the PDF.
    :param temp_dir: Scratch directory for the in-flight download.
    :returns: ``True`` if a PDF was obtained.
    """
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By

    try:
        driver.get(target_url)
        
        if wait_for_download(temp_dir, output_path, timeout=10):
            return True

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        potential_pdf_urls = []
        
        for tag in soup.find_all(['a', 'iframe'], src=True) + soup.find_all('a', href=True):
            link = tag.get('href') or tag.get('src')
            if not link:
                continue
            full_url = urljoin(driver.current_url, link)
            if full_url not in potential_pdf_urls:
                potential_pdf_urls.append(full_url)

        for pdf_url in potential_pdf_urls:
            # Try direct download first for any extracted potential PDF links
            if download_direct_pdf(pdf_url, output_path):
                soup.decompose()
                return True
            
            # Fallback to driver loading
            if verify_pdf_link(pdf_url) or re.search(r'\.pdf(\?.*)?$', pdf_url, re.I):
                driver.get(pdf_url)
                if wait_for_download(temp_dir, output_path, timeout=30):
                    soup.decompose()
                    return True

        is_html_article = soup.find(['article', 'main']) or "abstract" in driver.page_source.lower()
        if is_html_article and not driver.current_url.lower().endswith(".pdf"):
            # The RESULT is returned. save_as_pdf already validates the capture and reports
            # whether it wrote a file; discarding that and returning True unconditionally recorded
            # a failed print-to-PDF as a completed download with nothing on disk -- and because
            # the caller's short-circuit treats a present file as done, the paper was never
            # retried. That is the exact hole write_pdf_atomically was added to close.
            captured = save_as_pdf(driver, output_path)
            soup.decompose()
            return captured

        soup.decompose()
    except Exception as e:
        logger.debug(f"Selenium process failure for URL {target_url}: {e}")
        
    return False

def download_paper_pdf(doi: str, output_path: str, email: str, routers: list = None) -> bool:
    """
    Downloads paper PDF by first checking direct HTTP links (via Unpaywall/routers),
    falling back to Selenium if required.
    """
    # is_valid_pdf, not a size test. A size-only check accepts any partial write over 1000 bytes
    # as a finished download, permanently: the DOI is never retried and the corrupt file is what
    # the extraction step reads. This is the short-circuit is_valid_pdf was written for.
    if is_valid_pdf(output_path):
        return True
        
    routers = routers or []
    target_urls = get_all_unpaywall_urls(doi, email)
    
    for router in routers:
        router_url = f"{router.rstrip('/')}/{doi}"
        if router_url not in target_urls:
            target_urls.append(router_url)
            
    if not target_urls:
        return False
        
    # Phase 1: Try direct HTTP download on all URLs (Fast & light!)
    for url in target_urls:
        if download_direct_pdf(url, output_path):
            logger.info(f"Direct download success: {doi} -> {output_path}")
            return True
            
    # Phase 2: Fallback to Selenium (Heavy)
    from webdriver_manager.chrome import ChromeDriverManager
    try:
        driver_path = ChromeDriverManager().install()
    except Exception as e:
        logger.error(f"ChromeDriverManager failed to install: {e}")
        return False
        
    with tempfile.TemporaryDirectory() as temp_dir:
        for url in target_urls:
            driver = None
            try:
                driver = get_stealth_driver(temp_dir, driver_path)
                if process_url_with_selenium(driver, url, output_path, temp_dir):
                    logger.info(f"Selenium download success: {doi} -> {output_path}")
                    return True
            except Exception as e:
                logger.debug(f"Selenium attempt failed: {e}")
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception as e:
                        logger.debug(f"Selenium driver.quit() failed for {url}: {e}")
                gc.collect()
                
    return False

#: Hosts probed to decide whether the network is up. Both are public DNS resolvers, addressed by
#: IP so the check does not itself depend on name resolution -- a broken resolver is one of the
#: failure modes this is meant to detect, not be defeated by.
_CONNECTIVITY_PROBES: tuple = (("1.1.1.1", 53), ("8.8.8.8", 53))


def internet_available(timeout: float = 3.0) -> bool:
    """
    Reports whether the network is reachable.

    Opens a TCP connection rather than issuing an HTTP request: it is cheaper, it does not depend
    on any particular site being up, and it does not pollute the logs of a service that has no
    part in this pipeline. Two probes are tried because a single blocked resolver is common enough
    on institutional networks to cause a false negative on its own.

    :param timeout: Seconds to wait per probe.
    :returns: ``True`` when any probe connects.
    """
    for host, port in _CONNECTIVITY_PROBES:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def wait_for_connection(max_wait: float = 60.0, interval: float = 5.0) -> bool:
    """
    Blocks until the network comes back, giving up after `max_wait`.

    Bounded deliberately. This runs inside every worker of the download pool, so an unbounded
    "wait until the network returns" loop -- which is what this replaced -- can wedge the entire
    step indefinitely on a machine that is simply offline, with no output explaining why. On
    expiry the caller proceeds and lets the download fail normally, where the existing retry loop
    already handles it.

    Measured against a wall-clock deadline rather than by summing the sleeps, because each probe
    itself blocks for up to its own timeout when the network is down. Counting only the sleeps
    made the real wait roughly 1.6x the requested `max_wait`.

    :param max_wait: Wall-clock seconds to wait before giving up. The call may overshoot by at
        most the duration of the one probe straddling the deadline.
    :param interval: Seconds between probes.
    :returns: ``True`` if the network came back within `max_wait`.
    """
    deadline = time.monotonic() + max_wait
    while True:
        if internet_available():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        logger.warning(f"No internet connection. Retrying in {min(interval, remaining):.0f}s "
                       f"({remaining:.0f}s left before giving up)...")
        time.sleep(min(interval, remaining))


def process_paper_worker(row_tuple: tuple, source: str, email: str,
                         routers: Optional[Any] = None) -> Optional[dict]:
    """
    Downloads the full text for one publication record. The unit of work for the download pool.

    :param row_tuple: ``(index, row)`` from the metadata DataFrame; the row supplies the DOI.
    :param source: Source name (``"pubmed"`` or ``"scopus"``), selecting the output directory.
    :param email: Address used for Unpaywall and NCBI lookups.
    :param routers: Optional proxy/router pool for rotating egress addresses.
    :returns: A record describing the download, or ``None`` when the row had no usable DOI.
    """
    idx, row = row_tuple
    row_dict = row.to_dict()
    
    if pd.isna(row_dict.get('doi')) or not row_dict.get('doi'):
        return None

    # Use UUID for internal ID if not present
    if 'internal_id' not in row_dict or pd.isna(row_dict['internal_id']):
        row_dict['internal_id'] = str(uuid.uuid4())
        
    row_dict['success'] = False
    output_dir = os.path.join('data', 'literature_data', source)
    os.makedirs(output_dir, exist_ok=True)
    pdf_dest = os.path.join(output_dir, f"{row_dict['internal_id']}.pdf")

    for attempt in range(5):
        if not wait_for_connection():
            # Attempt the download anyway. The probe can be wrong -- a network that blocks
            # outbound DNS still reaches the publishers over HTTPS -- and a failed download is
            # retried by this same loop, so proceeding is strictly safer than skipping the row.
            logger.warning(f"Network still unreachable; attempting {row_dict['doi']} regardless.")

        success = download_paper_pdf(
            row_dict['doi'], 
            output_path=pdf_dest, 
            email=email, 
            routers=routers
        )
        
        if success:
            row_dict['success'] = True
            row_dict['pdf_path'] = pdf_dest
            break
        
        time.sleep(2)
        
    return row_dict

def _append_failure_ledger(ledger_path: str, df_failed: "pd.DataFrame") -> None:
    """
    Records download failures so a later run can tell "not yet tried" from "tried and failed".

    The manifest deliberately keeps only successes -- it means "papers whose PDF exists", and
    pipeline_runner queues every row in it. Discarding the failures entirely, which is what happened
    before this, leaves no record that a DOI was ever attempted: each re-run repeats the full pending
    set, spending five attempts and ten seconds of sleep on papers that have no open-access copy at
    all. With ~97k pending DOIs that is the difference between a bounded retry and a run that never
    finishes.

    Written atomically, for the same reason the manifest is: a truncated ledger is worse than none,
    because the next run would silently treat its lost rows as never attempted.

    :param ledger_path: Destination CSV.
    :param df_failed: Rows whose ``success`` is not True.
    :returns: ``None``.
    """
    columns = ["doi", "internal_id", "attempts", "last_attempt", "reason"]
    record = df_failed.copy()
    record["last_attempt"] = pd.Timestamp.utcnow().isoformat()
    if "reason" not in record.columns:
        # download_paper_pdf returns a bare bool today, so the reason is recorded as unclassified
        # rather than invented. Classifying it (no open-access location vs a failed lookup vs a
        # blocked fetch) is what would let a retry skip the permanently dead ones.
        record["reason"] = "unclassified"
    record["attempts"] = 1
    record = record[[c for c in columns if c in record.columns]]

    if os.path.exists(ledger_path):
        try:
            previous = pd.read_csv(ledger_path)
            record = pd.concat([previous, record], ignore_index=True)
            if "attempts" in record.columns and "doi" in record.columns:
                # Keep a running count per DOI rather than one row per attempt, so the ledger stays
                # the size of the pending set instead of growing without bound across runs.
                totals = record.groupby("doi", dropna=False)["attempts"].transform("sum")
                record["attempts"] = totals
        except Exception as exc:                                # noqa: BLE001 - ledger is advisory
            logger.warning(f"Could not read the existing failure ledger {ledger_path}: {exc}")
    if "doi" in record.columns:
        record.drop_duplicates(subset=["doi"], keep="last", inplace=True)

    tmp_ledger = f"{ledger_path}.tmp"
    record.to_csv(tmp_ledger, index=False)
    os.replace(tmp_ledger, ledger_path)


def _exhausted_dois(ledger_path: str, max_attempts: int) -> set:
    """
    DOIs that have already failed ``max_attempts`` times or more, across all previous runs.

    :param ledger_path: The failure ledger written by :func:`_append_failure_ledger`.
    :param max_attempts: Attempts after which a DOI stops being retried.
    :returns: DOIs to skip. Empty when the ledger is absent or unreadable -- a ledger that cannot be
        read must not silently cancel work, so the failure mode is "retry everything", which is the
        behaviour this had before the ledger existed.
    """
    if max_attempts <= 0 or not os.path.exists(ledger_path):
        return set()
    try:
        ledger = pd.read_csv(ledger_path)
    except Exception as exc:                                    # noqa: BLE001 - advisory only
        logger.warning(f"Could not read the failure ledger {ledger_path}: {exc}")
        return set()
    if "doi" not in ledger.columns:
        return set()
    if "attempts" not in ledger.columns:
        ledger["attempts"] = 1
    spent = ledger[pd.to_numeric(ledger["attempts"], errors="coerce").fillna(1) >= max_attempts]
    return set(spent["doi"].dropna().astype(str))


def download_papers_multiprocess(source: str, email: str, routers: Optional[list] = None,
                                num_workers: int = 4, max_attempts: int = 3,
                                limit: int = 0) -> None:
    """
    Reads [source]_metadata.csv and downloads missing papers.

    :param source: ``"pubmed"`` or ``"scopus"``.
    :param email: Address for the Unpaywall lookups.
    :param routers: Optional proxy pool.
    :param num_workers: Download processes.
    :param max_attempts: Give up on a DOI after this many recorded failures across runs. 0 retries
        everything, which is the pre-ledger behaviour.
    :param limit: Attempt at most this many pending papers; 0 means all. Applied AFTER the manifest
        and ledger exclusions, so a bounded trial run samples from the work that would actually be
        done rather than from rows that were going to be skipped anyway.
    """
    downloaded_csv_path = f"data/literature_data/{source}_downloaded.csv"
    #: Failures live beside the manifest, not in it. See _append_failure_ledger.
    failed_csv_path = f"data/literature_data/{source}_failed.csv"
    metadata_csv = f'data/literature_data/{source}_metadata.csv'
    
    if not os.path.exists(metadata_csv):
        logger.error(f"Metadata file {metadata_csv} does not exist.")
        return
        
    df_metadata = pd.read_csv(metadata_csv)
    
    # Load already downloaded entries
    if os.path.exists(downloaded_csv_path):
        df_downloaded = pd.read_csv(downloaded_csv_path)
        # Purge entries where file is physically missing
        df_downloaded['file_exists'] = df_downloaded['pdf_path'].apply(
            lambda x: os.path.isfile(str(x)) if pd.notna(x) else False
        )
        df_downloaded = df_downloaded[df_downloaded['file_exists']].drop(columns=['file_exists'])
    else:
        df_downloaded = pd.DataFrame(columns=df_metadata.columns.tolist() + ['internal_id', 'success', 'pdf_path'])
        
    valid_dois = set(df_downloaded['doi'].dropna().unique())
    df_metadata_pending = df_metadata[~df_metadata['doi'].isin(valid_dois)]

    # Skip DOIs this has already failed on repeatedly. Without this the ledger is write-only and a
    # re-run repeats the entire pending set: ~97k DOIs, each costing five attempts and ten seconds
    # of sleep before it gives up, which is a 269-hour floor from the sleeps alone. The exhausted
    # set is usually papers with no open-access copy at all, and no number of retries changes that.
    #
    # Keyed on ATTEMPT COUNT, not on a failure reason. download_paper_pdf returns a bare bool, so
    # the ledger honestly records "unclassified" -- distinguishing "no OA location exists"
    # (permanent) from "the lookup timed out" (transient) needs that function to report which, and
    # until it does, pretending to know would be worse than counting. A missing or unreadable
    # ledger simply means nothing is skipped, so this can only ever narrow the work, never widen it.
    exhausted = _exhausted_dois(failed_csv_path, max_attempts)
    if exhausted:
        before = len(df_metadata_pending)
        df_metadata_pending = df_metadata_pending[~df_metadata_pending['doi'].isin(exhausted)]
        logger.info(
            f"Skipping {before - len(df_metadata_pending)} DOI(s) that already failed "
            f"{max_attempts}+ times (from {failed_csv_path}); {len(df_metadata_pending)} remain. "
            f"Delete that file or raise --max-attempts to retry them."
        )

    if limit and limit > 0 and len(df_metadata_pending) > limit:
        logger.info(
            f"Limiting this run to {limit} of {len(df_metadata_pending)} pending papers. "
            f"Failures are recorded in {failed_csv_path}, so a later unbounded run resumes rather "
            f"than repeating this work."
        )
        df_metadata_pending = df_metadata_pending.head(limit)

    already_processed = len(df_metadata) - len(df_metadata_pending)
    
    if df_metadata_pending.empty:
        logger.info(f"Papers already processed (downloaded): {already_processed} | Papers remaining: 0")
        logger.info(f"All papers for source '{source}' are already downloaded.")
        return

    logger.info(f"Papers already processed (downloaded): {already_processed} | Papers remaining: {len(df_metadata_pending)}")
    logger.info(f"Found {len(df_metadata_pending)} pending papers to download for source '{source}'. Running with {num_workers} workers.")
    
    worker_func = partial(process_paper_worker, source=source, email=email, routers=routers)
    
    processed_rows = []
    # How much of processed_rows has already been written to the FAILURE ledger. The manifest below
    # is rebuilt wholesale each flush and deduplicated, so re-including a row there is harmless;
    # the ledger is append-only and counts attempts, so re-including a row there is not.
    ledgered_upto = 0
    with mp.Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker_func, df_metadata_pending.iterrows()), total=len(df_metadata_pending)):
            if result is not None:
                processed_rows.append(result)
                
            # Periodic flush to disk to avoid data loss
            if len(processed_rows) % 5 == 0 or len(processed_rows) == len(df_metadata_pending):
                df_new = pd.DataFrame([r for r in processed_rows if r is not None])
                if not df_new.empty:
                    df_combined = pd.concat([df_downloaded, df_new], ignore_index=True)
                    if 'success' in df_combined.columns:
                        df_combined = df_combined[df_combined['success'] == True]
                    df_combined.drop_duplicates(subset=['doi'], keep='last', inplace=True)
                    # Written through a temporary file and renamed into place. `to_csv` straight to
                    # the destination TRUNCATES it first, and this loop rewrites the entire manifest
                    # every five downloads -- 19,389 times over a full pending run. A crash, kill or
                    # full disk during any one of those rewrites leaves a truncated CSV, and since
                    # this file is the ONLY record of which PDFs were fetched, every row lost that
                    # way strands its PDF on disk unreferenced. There are 43,443 such orphaned PDFs
                    # in data/literature_data/pubmed alone (65,493 files against 22,050 rows), which
                    # is what this failure mode looks like after the fact. Same .tmp + os.replace
                    # pattern as sample/revision.py and sample/gene_ontology.py.
                    tmp_manifest = f"{downloaded_csv_path}.tmp"
                    df_combined.to_csv(tmp_manifest, index=False)
                    os.replace(tmp_manifest, downloaded_csv_path)

                    # Failures are recorded too, in their own file. The line above keeps only
                    # successes, which is correct for a manifest that means "papers that exist" --
                    # pipeline_runner queues every row it finds. But discarding the failures
                    # entirely leaves no record of what was attempted, so each re-run repeats all
                    # ~97k pending DOIs from scratch, spending 5 attempts and 10s of sleep on papers
                    # that have no open-access copy at all.
                    # `'success' in df_new` tests COLUMN membership and df_new.get('success')
                    # returns a Series, so the previous form compared a column to True rather than
                    # producing the mask intended. Written explicitly.
                    # Only the rows added SINCE THE LAST FLUSH reach the ledger. df_new is the
                    # cumulative list, and _append_failure_ledger stamps attempts=1 and then sums
                    # per DOI -- so appending the cumulative failures every flush charged each
                    # failed DOI one extra attempt per remaining flush. A single failure early in a
                    # run reached attempts=4, and with the default --max-attempts 3 _exhausted_dois
                    # then abandoned that DOI permanently after one real try.
                    #
                    # This slice also absorbs a second cause of double-counting: the flush
                    # condition tests len(processed_rows) % 5, and a None result leaves the length
                    # unchanged, so consecutive flushes can fire at the same length. The second one
                    # now finds nothing new.
                    fresh_rows = processed_rows[ledgered_upto:]
                    ledgered_upto = len(processed_rows)
                    df_fresh = pd.DataFrame(fresh_rows)
                    if 'success' in df_fresh.columns:
                        df_failed = df_fresh[df_fresh['success'] != True]
                        if not df_failed.empty:
                            _append_failure_ledger(failed_csv_path, df_failed)
                    
    logger.info(f"Completed downloading papers for '{source}'. Processed {len(processed_rows)} papers. State saved to {downloaded_csv_path}")
