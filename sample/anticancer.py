"""
Import anticancer peptide datasets and normalise them into one table the pipeline can merge.

The release began as an antimicrobial-peptide database; adding experimentally validated *anticancer*
peptides is what gives the cancer/cell-line models (see :mod:`sample.ml_models`,
:mod:`sample.cell_lines`) enough positives to clear their quality floor. Four sources were named for
this: AntiCP2, CancerPPD, ApInAPDB and DCTPep.

**What this module actually does today.** Only **AntiCP2** (Agrawal et al., *Brief. Bioinform.* 2021)
is wired, because it is the one whose host currently answers and whose format was inspected and
verified rather than assumed -- its download files are plain text, one peptide sequence per line, in
labelled positive/negative train/test splits. The other three are left as registry entries with no
parser: at the time of writing their download URLs 404 (the host answers, the path does not), so
writing a parser for them would be guessing at a format this project's rules forbid shipping. When a
reachable URL is found, each becomes a `_Source` with a verified parser alongside AntiCP2.

The output is deliberately narrow and honest: a positive is an anticancer peptide (``label=1``); an
AntiCP2 *negative* is an AMP with no reported anticancer activity (``label=0``) -- a labelled hard
negative for training an anticancer classifier, **not** evidence the peptide is inactive against
everything. Sequences are validated and de-duplicated; nothing is invented for a row that does not
parse.
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.anticancer")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Where downloaded raw files and the normalised table are written.
ANTICANCER_DIR = os.path.join(DATA_DIR, "anticancer_databases")

#: AntiCP 2.0 download base (verified reachable and format-inspected 2026-08-13).
ANTICP2_BASE = "https://webs.iiitd.edu.in/raghava/anticp2/add_data"

#: The AntiCP2 files, each with the split and label a line in it carries. Positives are anticancer
#: peptides; negatives are AMPs without reported anticancer activity (a labelled hard negative).
ANTICP2_FILES: Dict[str, Dict[str, Any]] = {
    "pos_train_main": {"dataset": "main", "split": "train", "label": 1},
    "neg_train_main": {"dataset": "main", "split": "train", "label": 0},
    "pos_test_main": {"dataset": "main", "split": "test", "label": 1},
    "neg_test_main": {"dataset": "main", "split": "test", "label": 0},
    "pos_train_alternate": {"dataset": "alternate", "split": "train", "label": 1},
    "neg_train_alternate": {"dataset": "alternate", "split": "train", "label": 0},
    "pos_test_alternate": {"dataset": "alternate", "split": "test", "label": 1},
    "neg_test_alternate": {"dataset": "alternate", "split": "test", "label": 0},
}

#: Sources named for import. Only those with a verified parser are importable; the rest record why
#: not, so the gap is explicit rather than a silent omission.
SOURCES: Dict[str, Dict[str, Any]] = {
    "anticp2": {"importable": True,
                "note": "Plain text, one sequence per line; format verified 2026-08-13."},
    "cancerppd": {"importable": True,
                  "note": ("Bulk id/sequence lists at /cancerppd/{natural,non-natural}_seq_dwn/ "
                           "(verified 2026-08-17: 3,141 sequences across six files). Per-entry "
                           "detail pages carry the measured activity and are scraped separately "
                           "-- see `scrape_cancerppd_details`.")},
    "apinapdb": {"importable": False,
                 "note": ("bioinf.modares.ac.ir does not respond (connection times out, rechecked "
                          "2026-08-17); no reachable bulk download to inspect.")},
    "dctpep": {"importable": True,
               "note": ("Bulk .xlsx at dctpep.cpu-bioinfor.org/downloads/ -- this became "
                        "available since the earlier 404 and was verified 2026-08-17: 6,107 rows, "
                        "51 columns, sequences plus cross-references and cytotoxicity.")},
}

#: CancerPPD's bulk id/sequence lists, by file stem -> (URL path, chirality, natural/non-natural).
#: Six files rather than one download: the database splits its peptides by residue chirality
#: (all-L, all-D, mixed) and by whether they contain non-natural residues, and each split is a real
#: property of the peptide worth keeping rather than flattening away.
CANCERPPD_BASE: str = "https://webs.iiitd.edu.in/raghava/cancerppd"
CANCERPPD_FILES: Dict[str, Dict[str, str]] = {
    "l_natural": {"path": "natural_seq_dwn/l_natural.txt", "chirality": "L", "residues": "natural"},
    "d_natural": {"path": "natural_seq_dwn/d_natural.txt", "chirality": "D", "residues": "natural"},
    "mix_natural": {"path": "natural_seq_dwn/mix_natural.txt", "chirality": "mixed",
                    "residues": "natural"},
    "l_non-natural": {"path": "non-natural_seq_dwn/l_non-natural.txt", "chirality": "L",
                      "residues": "non-natural"},
    "d_non-natural": {"path": "non-natural_seq_dwn/d_non-natural.txt", "chirality": "D",
                      "residues": "non-natural"},
    "mix_non-natural": {"path": "non-natural_seq_dwn/mix_non-natural.txt", "chirality": "mixed",
                        "residues": "non-natural"},
}
CANCERPPD_DIR: str = os.path.join(ANTICANCER_DIR, "cancerppd")

#: One CancerPPD entry's detail page. The bulk files give only id + sequence; everything that makes
#: an entry useful here -- the measured IC50, its cell line, tissue, assay and PMID -- lives on this
#: page and nowhere else, which is why scraping it is worth the requests.
CANCERPPD_DETAIL_URL: str = CANCERPPD_BASE + "/display_sub.php?details={entry_id}"

#: Detail-page labels -> this project's field names. Taken from a real rendered page (id 1001,
#: inspected 2026-08-17), not guessed from the database's paper.
CANCERPPD_DETAIL_FIELDS: Dict[str, str] = {
    "PMID": "pmid",
    "YEAR": "year",
    "SEQUENCE": "sequence",
    "NAME": "name",
    "C-ter Modification": "c_terminal_modification",
    "N-ter Modification": "n_terminal_modification",
    "Linear/Cyclic": "topology_raw",
    "L/D/Mix": "chirality_raw",
    "Non-Natural": "non_natural",
    "NATURE": "nature",
    "ACTIVITY": "activity_raw",
    "SOURCE": "source_organism",
    "Cancer Type": "cancer_type",
    "ASSAY": "assay",
    "Tissue": "tissue",
    "CELL LINE": "cell_line",
    "TEST TIME": "test_time",
    "SMILES": "smiles",
}

#: DCTPep's bulk peptide library. The site also publishes a drug library and a FASTA of the same
#: peptides; the .xlsx is the one imported because it is the only one carrying the metadata (the
#: FASTA is sequences alone, which the .xlsx already contains).
DCTPEP_URL: str = "http://dctpep.cpu-bioinfor.org/downloads/peptide_library_all.xlsx"
DCTPEP_DIR: str = os.path.join(ANTICANCER_DIR, "dctpep")

#: The DCTPep columns actually read, mapped to this project's own field names. Deliberately a
#: subset of the 51 available: everything here is either an identity, a cross-reference, or a
#: property this database already models. The rest (computed descriptors like pI, Boman index,
#: aliphatic index) are recomputed by `sample.enrichment` from the sequence itself, so importing
#: another source's numbers for them would introduce a second, conflicting answer to the same
#: question.
DCTPEP_COLUMNS: Dict[str, str] = {
    "DCTPep_ID": "source_id",
    "Peptide_Name": "name",
    "Sequence": "sequence",
    "Origin": "source_organism",
    "Nature": "activities_raw",
    "Cytotoxicity": "cytotoxicity_raw",
    "Linear_Cyclic": "topology_raw",
    "Disulfide_Bond": "disulfide_raw",
    "N-terminal_Modification": "n_terminal_modification",
    "C-terminal_Modification": "c_terminal_modification",
    "PDB_ID": "pdb_id",
    "UniProt_ID": "uniprot_id",
    "PubChem_CID": "pubchem_cid",
    "DRAMP_ID": "dramp_id",
    "CancerPPD_ID": "cancerppd_id",
    "DBAASP_ID": "dbaasp_id",
    "Literature": "reference",
}

#: Values DCTPep uses for "no value". Checked against the real file rather than assumed: a bare
#: `pd.isna` misses these, and importing the literal string "Not available" as a peptide name or a
#: PDB id would put junk into the release.
DCTPEP_NULLS: frozenset = frozenset({"", "-", "na", "n/a", "none", "not available",
                                     "not applicable", "nan", "null", "unknown"})

#: One-letter residues (with the usual ambiguity codes) a peptide line may contain.
_RESIDUES = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZ]+$")


def parse_sequence_per_line(path: str, min_length: int = 2) -> List[str]:
    """
    Reads a plain-text file of one peptide sequence per line.

    Blank lines, FASTA headers (a stray ``>`` line) and sequences with non-residue characters are
    skipped rather than guessed at. The result is de-duplicated but keeps first-seen order.

    :param path: The file to read.
    :param min_length: Shortest sequence to keep.
    :returns: The validated, de-duplicated sequences.
    """
    seen: Dict[str, None] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            sequence = line.strip().upper()
            if not sequence or sequence.startswith(">"):
                continue
            if len(sequence) < min_length or not _RESIDUES.match(sequence):
                continue
            seen.setdefault(sequence, None)
    return list(seen)


def _clean_accession(value: Any) -> str:
    """
    Reads an identifier cell, undoing pandas' numeric coercion.

    A numeric accession ("1003") survives a CSV round trip as a float, so a bare ``str()`` yields
    ``"1003.0"`` -- an accession that no longer resolves anywhere. Integral floats are rendered
    without the decimal point; everything else falls through to :func:`_dctpep_value`'s null
    handling.

    :param value: A raw cell.
    :returns: The cleaned identifier, or ``""`` when there is none.
    """
    text = _dctpep_value(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".")[0]
    return text


def download_cancerppd(dest_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Fetches CancerPPD's six bulk id/sequence lists.

    :param dest_dir: Where to write them; :data:`CANCERPPD_DIR` when omitted.
    :returns: ``{file stem: path}`` for each file that downloaded.
    """
    import requests

    dest_dir = dest_dir or CANCERPPD_DIR
    os.makedirs(dest_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for stem, meta in CANCERPPD_FILES.items():
        url = f"{CANCERPPD_BASE}/{meta['path']}"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except Exception as e:  # noqa: BLE001 -- one unavailable split must not lose the others
            logger.warning("CancerPPD %s not downloaded (%s): %s", stem, url, e)
            continue
        path = os.path.join(dest_dir, f"{stem}.txt")
        with open(path, "wb") as handle:
            handle.write(response.content)
        written[stem] = path
    logger.info("Downloaded %d/%d CancerPPD files to %s.",
                len(written), len(CANCERPPD_FILES), dest_dir)
    return written


def parse_cancerppd_file(path: str) -> List[Dict[str, str]]:
    """
    Reads one CancerPPD bulk list: a tab-separated ``id``/``Sequence`` table.

    The real files carry ~1M trailing blank lines (l_natural.txt is 1,047,981 lines for 2,679
    records), so rows are validated rather than counted -- a line only becomes a record when it has
    both fields and a plausible sequence.

    :param path: The file to read.
    :returns: ``[{"entry_id", "sequence"}]``, de-duplicated on the pair, first occurrence kept.
    """
    records: List[Dict[str, str]] = []
    seen = set()
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            entry_id, sequence = parts[0], parts[1].upper().replace(" ", "")
            if entry_id.lower() == "id" or not entry_id.isdigit():
                continue  # the header row, or a malformed line
            if not _RESIDUES.match(sequence):
                continue  # modified/non-natural spellings that no longer denote a plain peptide
            key = (entry_id, sequence)
            if key in seen:
                continue
            seen.add(key)
            records.append({"entry_id": entry_id, "sequence": sequence})
    return records


def load_cancerppd(data_dir: Optional[str] = None) -> "Any":
    """
    Parses CancerPPD's bulk lists into the shared normalised table.

    CancerPPD catalogues *experimentally validated* anticancer peptides, so every entry is a
    positive; like DCTPep it ships no negatives and none are invented. Each row keeps the chirality
    and natural/non-natural split it came from, since an all-D or non-natural peptide is a
    materially different molecule from its L-natural counterpart even at the same sequence.

    Detail-page fields (the measured IC50, cell line, assay, PMID) are merged in when
    :func:`scrape_cancerppd_details` has been run; without it this is sequences and splits only.

    :param data_dir: Directory holding the downloaded files; :data:`CANCERPPD_DIR` when omitted.
    :returns: A DataFrame in the shared schema.
    :raises FileNotFoundError: When no bulk file has been downloaded.
    """
    import pandas as pd

    data_dir = data_dir or CANCERPPD_DIR
    details = _load_cancerppd_details(data_dir)
    rows: List[Dict[str, Any]] = []
    found_any = False
    for stem, meta in CANCERPPD_FILES.items():
        path = os.path.join(data_dir, f"{stem}.txt")
        if not os.path.exists(path):
            continue
        found_any = True
        for record in parse_cancerppd_file(path):
            row: Dict[str, Any] = {
                "sequence": record["sequence"],
                "source_db": "CancerPPD",
                "activity": "anticancer",
                "label": 1,
                "dataset": "cancerppd",
                "split": "",
                "cancerppd_source_id": record["entry_id"],
                "cancerppd_chirality": meta["chirality"],
                "cancerppd_residues": meta["residues"],
            }
            for field, value in (details.get(record["entry_id"]) or {}).items():
                # The detail page repeats the sequence; the bulk file's is authoritative here since
                # it is what the split was defined on.
                if field != "sequence":
                    row[f"cancerppd_{field}"] = value
            rows.append(row)

    if not found_any:
        raise FileNotFoundError(
            f"No CancerPPD files in {data_dir}. Run `sample-cli import-anticancer --download`.")
    logger.info("CancerPPD: %d sequences (%d with scraped detail data).",
                len(rows), sum(1 for r in rows if r.get("cancerppd_activity_raw")))
    return pd.DataFrame(rows)


def _load_cancerppd_details(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    :param data_dir: Directory holding ``details.json``, if it has been scraped.
    :returns: ``{entry_id: fields}``; empty when the scrape has not been run.
    """
    import json

    path = os.path.join(data_dir, "details.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as e:  # noqa: BLE001 -- a bad cache is "not scraped yet", never fatal
        logger.warning("Ignoring unreadable CancerPPD details at %s: %s", path, e)
        return {}


def parse_cancerppd_detail(html: str) -> Dict[str, str]:
    """
    Extracts one detail page's fields.

    The page is a table of label/value cells, so the parse walks the stripped text stream and takes
    the cell after each known label rather than relying on positional indexes, which would break the
    moment the database adds a row. Labels it does not know are ignored, so a new field is a no-op
    rather than a crash.

    :param html: The rendered detail page.
    :returns: ``{field: value}`` for the labels in :data:`CANCERPPD_DETAIL_FIELDS` that appeared.
    """
    text = re.sub(r"<[^>]*>", "|", html)
    cells = [c.strip() for c in text.split("|")]
    cells = [c for c in cells if c]

    found: Dict[str, str] = {}
    for index, cell in enumerate(cells[:-1]):
        field = CANCERPPD_DETAIL_FIELDS.get(cell)
        if field is None or field in found:
            continue
        value = cells[index + 1].strip()
        # "ACTIVITY" renders as "IC", "50", "=6.49 µg/ml" across three cells (the subscript is its
        # own tag), so the value is rejoined until it carries a number and a unit.
        if field == "activity_raw":
            value = " ".join(cells[index + 1:index + 4]).strip()
        if value and value.lower() not in {"na", "none", "nil", "-"}:
            found[field] = value
    return found


def scrape_cancerppd_details(entry_ids: Optional[List[str]] = None,
                             data_dir: Optional[str] = None,
                             delay_seconds: float = 0.5,
                             limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Scrapes CancerPPD's per-entry detail pages, resumably.

    **This makes one HTTP request per entry** (~3,100 for the full database) against an academic
    server, so it is rate-limited by default, resumable, and never run implicitly -- the same
    treatment the RefSeq sweep gets. Already-scraped ids are skipped, and results are written after
    every page, so an interrupted run loses at most one request's work.

    :param entry_ids: Which entries to fetch; every id in the downloaded bulk files when omitted.
    :param data_dir: Where the bulk files and ``details.json`` live; :data:`CANCERPPD_DIR` when
        omitted.
    :param delay_seconds: Pause between requests, to stay polite to the host.
    :param limit: Stop after this many NEW entries; ``None`` fetches all outstanding.
    :returns: A summary (``scraped``, ``skipped``, ``failed``, ``total``, ``path``).
    """
    import json
    import time

    import requests

    data_dir = data_dir or CANCERPPD_DIR
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "details.json")
    details = _load_cancerppd_details(data_dir)

    if entry_ids is None:
        entry_ids = []
        seen = set()
        for stem in CANCERPPD_FILES:
            file_path = os.path.join(data_dir, f"{stem}.txt")
            if not os.path.exists(file_path):
                continue
            for record in parse_cancerppd_file(file_path):
                if record["entry_id"] not in seen:
                    seen.add(record["entry_id"])
                    entry_ids.append(record["entry_id"])

    # "Already scraped" and "not attempted because of --limit" are different facts and are counted
    # separately: reporting them as one number made a first, limited run look like it had skipped
    # thousands of entries it had simply not reached yet.
    already_scraped = sum(1 for e in entry_ids if e in details)
    outstanding = [e for e in entry_ids if e not in details]
    remaining_after_limit = 0
    if limit:
        remaining_after_limit = max(0, len(outstanding) - limit)
        outstanding = outstanding[:limit]

    scraped = failed = 0
    for entry_id in outstanding:
        try:
            response = requests.get(CANCERPPD_DETAIL_URL.format(entry_id=entry_id), timeout=30)
            response.raise_for_status()
            fields = parse_cancerppd_detail(response.text)
        except Exception as e:  # noqa: BLE001 -- one bad page must not end the run
            logger.warning("CancerPPD %s not scraped: %s", entry_id, e)
            failed += 1
            continue
        if fields:
            details[entry_id] = fields
            scraped += 1
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(details, handle)
        if delay_seconds:
            time.sleep(delay_seconds)

    logger.info("CancerPPD detail scrape: %d new, %d already had, %d failed, %d still outstanding "
                "(%d known in total).",
                scraped, already_scraped, failed, remaining_after_limit, len(details))
    return {"scraped": scraped, "already_scraped": already_scraped, "failed": failed,
            "remaining": remaining_after_limit, "total": len(details), "path": path}


def _dctpep_value(value: Any) -> str:
    """
    Reads one DCTPep cell, treating its several spellings of "no value" as empty.

    :param value: A raw cell.
    :returns: The cleaned string, or ``""`` when the cell carries no real value. Never the literal
        ``"nan"`` -- ``str(float('nan'))`` is a truthy string, the same trap already fixed in
        :mod:`sample.cycpeptmpdb`.
    """
    import pandas as pd

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in DCTPEP_NULLS else text


def download_dctpep(dest_dir: Optional[str] = None) -> str:
    """
    Fetches DCTPep's bulk peptide library.

    :param dest_dir: Where to write it; :data:`DCTPEP_DIR` when omitted.
    :returns: Path to the downloaded ``.xlsx``.
    :raises RuntimeError: When the download fails, naming the URL so the failure is diagnosable.
    """
    import requests

    dest_dir = dest_dir or DCTPEP_DIR
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "peptide_library_all.xlsx")
    try:
        response = requests.get(DCTPEP_URL, timeout=120)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not download DCTPep from {DCTPEP_URL}: {e}") from e
    with open(path, "wb") as handle:
        handle.write(response.content)
    logger.info("Downloaded DCTPep to %s (%d bytes).", path, len(response.content))
    return path


def load_dctpep(data_dir: Optional[str] = None) -> "Any":
    """
    Parses DCTPep's bulk library into the same normalised table :func:`load_anticp2` produces.

    DCTPep is a curated database of anticancer/tumour-active peptides, so every row with a usable
    sequence is a positive (``label=1``). Unlike AntiCP2 it ships NO negatives -- and none are
    invented here: a database of anticancer peptides simply does not report non-anticancer ones,
    which is a different fact from a peptide being tested and found inactive.

    Only rows whose sequence is a real one-letter peptide are kept. DCTPep includes stapled and
    otherwise heavily modified entries whose "sequence" carries non-residue characters; those are
    dropped rather than stripped into something that no longer denotes the real molecule.

    :param data_dir: Directory holding the downloaded file; :data:`DCTPEP_DIR` when omitted.
    :returns: A DataFrame with the shared columns (``sequence``, ``source_db``, ``activity``,
        ``label``, ``dataset``, ``split``) plus DCTPep's own identity/cross-reference fields, so
        the extra metadata is not thrown away just because AntiCP2 has no equivalent.
    :raises FileNotFoundError: When the file has not been downloaded.
    """
    import pandas as pd

    data_dir = data_dir or DCTPEP_DIR
    path = os.path.join(data_dir, "peptide_library_all.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DCTPep library not found at {path}. Run `sample-cli import-anticancer --download` "
            f"or fetch {DCTPEP_URL}.")

    frame = pd.read_excel(path)
    rows: List[Dict[str, Any]] = []
    skipped_no_sequence = 0
    for _, record in frame.iterrows():
        sequence = _dctpep_value(record.get("Sequence")).upper().replace(" ", "")
        if not sequence or not _RESIDUES.match(sequence):
            skipped_no_sequence += 1
            continue
        row: Dict[str, Any] = {
            "sequence": sequence,
            "source_db": "DCTPep",
            "activity": "anticancer",
            "label": 1,
            # DCTPep is a curated database, not a train/test benchmark like AntiCP2; saying so
            # explicitly keeps a model-training consumer from reading a split that is not there.
            "dataset": "dctpep",
            "split": "",
        }
        for source_column, field in DCTPEP_COLUMNS.items():
            if field == "sequence":
                continue
            row[f"dctpep_{field}"] = _dctpep_value(record.get(source_column))
        rows.append(row)

    logger.info("DCTPep: %d usable sequences (%d rows skipped for a missing or non-standard "
                "sequence).", len(rows), skipped_no_sequence)
    return pd.DataFrame(rows)


def download_anticp2(dest_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Downloads the AntiCP2 dataset files.

    :param dest_dir: Directory to write into; ``data/anticancer_databases/anticp2`` when omitted.
    :returns: Mapping from file name to the local path written.
    :raises RuntimeError: When a file cannot be fetched, naming which -- so a partial download is a
        loud failure, not a silently short dataset.
    """
    import urllib.request

    dest_dir = dest_dir or os.path.join(ANTICANCER_DIR, "anticp2")
    os.makedirs(dest_dir, exist_ok=True)

    paths: Dict[str, str] = {}
    for name in ANTICP2_FILES:
        url = f"{ANTICP2_BASE}/{name}"
        target = os.path.join(dest_dir, name)
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # noqa: BLE001 -- re-raised with context below
            raise RuntimeError(f"Failed to download AntiCP2 file {name} from {url}: {exc}") from exc
        paths[name] = target
    logger.info("Downloaded %d AntiCP2 files to %s.", len(paths), dest_dir)
    return paths


def load_anticp2(data_dir: Optional[str] = None) -> "Any":
    """
    Parses the downloaded AntiCP2 files into a normalised table.

    :param data_dir: Directory holding the AntiCP2 files; ``data/anticancer_databases/anticp2`` when
        omitted.
    :returns: A DataFrame with ``sequence``, ``source_db`` (``AntiCP2``), ``activity``
        (``anticancer`` for positives, ``non-anticancer-amp`` for negatives), ``label`` (1/0),
        ``dataset`` (main/alternate) and ``split`` (train/test). A sequence appearing in more than
        one file keeps its first, positive-preferring occurrence (see :func:`build_anticancer_table`).
    :raises FileNotFoundError: When the files have not been downloaded.
    """
    import pandas as pd

    data_dir = data_dir or os.path.join(ANTICANCER_DIR, "anticp2")
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name, meta in ANTICP2_FILES.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        for sequence in parse_sequence_per_line(path):
            rows.append({
                "sequence": sequence,
                "source_db": "AntiCP2",
                "activity": "anticancer" if meta["label"] == 1 else "non-anticancer-amp",
                "label": meta["label"],
                "dataset": meta["dataset"],
                "split": meta["split"],
            })
    if missing and not rows:
        raise FileNotFoundError(
            f"No AntiCP2 files found in {data_dir} (missing {len(missing)}). Run "
            f"`sample-cli import-anticancer --download` first.")
    if missing:
        logger.warning("AntiCP2: %d of %d files missing from %s.",
                       len(missing), len(ANTICP2_FILES), data_dir)
    return pd.DataFrame(rows)


def positive_sequences(data_dir: Optional[str] = None,
                       output_csv: Optional[str] = None) -> List[str]:
    """
    The anticancer-positive sequences to integrate into the release.

    Prefers the combined, de-duplicated table (:func:`build_anticancer_table`'s output) when it
    exists, since that is the canonical, positive-preferring set; otherwise falls back to parsing the
    raw AntiCP2 positive files directly. Returns an empty list -- not an error -- when nothing has
    been imported, so the pipeline steps that call this degrade gracefully for a checkout that never
    ran ``import-anticancer``.

    :param data_dir: Directory holding the downloaded raw files; the default when omitted.
    :param output_csv: The combined table to read first; the default when omitted.
    :returns: Unique anticancer-positive sequences.
    """
    output_csv = output_csv or os.path.join(ANTICANCER_DIR, "anticancer_peptides.csv")
    if os.path.exists(output_csv):
        import pandas as pd
        table = pd.read_csv(output_csv)
        if "label" in table.columns and "sequence" in table.columns:
            return [str(s) for s in table.loc[table["label"] == 1, "sequence"].tolist()]

    # No combined table yet: try the raw AntiCP2 positive files. The fallback location tracks the
    # table's -- an explicit ``data_dir``, else the directory the (absent) ``output_csv`` names --
    # so a build reading a non-default table does not silently fall back to the packaged raw files.
    if data_dir:
        anticp2_dir = os.path.join(data_dir, "anticp2")
    else:
        anticp2_dir = os.path.join(os.path.dirname(os.path.abspath(output_csv)), "anticp2")
    seen: Dict[str, None] = {}
    for name, meta in ANTICP2_FILES.items():
        if meta["label"] != 1:
            continue
        path = os.path.join(anticp2_dir, name)
        if os.path.exists(path):
            for sequence in parse_sequence_per_line(path):
                seen.setdefault(sequence, None)
    return list(seen)


def iter_activity_records(data_dir: Optional[str] = None) -> "Any":
    """
    Yields anticancer positives as source-database records for :func:`sample.database_activities.load_all`.

    Each record carries a single qualitative activity with ``target_group="anticancer"`` and no
    measured value -- AntiCP2 reports activity/inactivity, not a potency. That one activity row is
    what places the sequence in the ``anticancer`` group task in :func:`sample.ml_models.build_tasks`;
    ``activity_type`` is left empty so no spurious per-assay task is created. The record shape matches
    the other loaders (see :func:`sample.database_activities.load_adp`) so the activity-table builder
    consumes it unchanged.

    Negatives are deliberately **not** emitted: an AntiCP2 negative is an AMP with no *reported*
    anticancer activity, which is not a labelled measurement of anything and must not be written into
    the activity table as if it were.

    :param data_dir: Directory holding the downloaded raw files; the default when omitted.
    :yields: One record per anticancer-positive sequence.
    """
    import pandas as pd

    table_path = os.path.join(
        os.path.dirname(os.path.abspath(
            os.path.join(data_dir, "x") if data_dir else os.path.join(ANTICANCER_DIR, "x"))),
        "anticancer_peptides.csv")

    # The combined table carries each source's own columns, so a row can be turned into a record
    # that keeps what that source actually measured. Without the table (nothing imported yet, or a
    # raw-files-only checkout) this falls back to the qualitative behaviour below.
    detailed: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(table_path):
        try:
            table = pd.read_csv(table_path, low_memory=False)
            for _, row in table.iterrows():
                if int(row.get("label", 0) or 0) != 1:
                    continue
                sequence = str(row.get("sequence") or "").strip().upper()
                if sequence and sequence not in detailed:
                    detailed[sequence] = row.to_dict()
        except Exception as e:  # noqa: BLE001 -- an unreadable table means "no detail", never fatal
            logger.warning("Could not read %s for activity detail: %s", table_path, e)

    for index, sequence in enumerate(positive_sequences(data_dir), start=1):
        row = detailed.get(sequence, {})
        source = _dctpep_value(row.get("source_db")) or "AntiCP2"
        # Numeric ids survive a CSV round trip as floats ("1003.0"); an accession must not gain a
        # decimal point on the way through pandas.
        record_id = (_clean_accession(row.get("cancerppd_source_id"))
                     or _clean_accession(row.get("dctpep_source_id"))
                     or f"ANTICP2_{index:05d}")

        activity = _anticancer_activity(row)
        publications = _anticancer_publications(row)
        cross_references = _anticancer_cross_references(row)

        yield {
            "database": source,
            "record_id": record_id,
            "sequence": sequence,
            "name": _dctpep_value(row.get("cancerppd_name")) or _dctpep_value(row.get("dctpep_name")),
            "source_organism": (_dctpep_value(row.get("cancerppd_source_organism"))
                                or _dctpep_value(row.get("dctpep_source_organism"))),
            "activity_labels": ["anticancer"],
            "cross_references": cross_references,
            "publications": publications,
            "activities": [activity],
        }


#: A measured potency as CancerPPD/DCTPep render it, e.g. "IC 50 =6.49 µg/ml", "IC50 = >100 µM".
#: Captures the assay type, an optional operator, the number and its unit. Deliberately strict:
#: anything that does not match stays a qualitative record rather than being coerced into a number.
_ACTIVITY_MEASUREMENT = re.compile(
    r"(?P<type>IC\s*50|EC\s*50|LC\s*50|LD\s*50|GI\s*50|CC\s*50|MIC)\s*"
    r"[=:]?\s*(?P<operator>[<>~]?=?)\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[µumn]?[gGmM]/?[mM]?[lL]?|[µumn]M|%)",
    re.IGNORECASE)


def _anticancer_activity(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds one activity row from an imported record, keeping a real measurement when there is one.

    CancerPPD and DCTPep report genuine potencies against named cell lines; AntiCP2 reports only
    membership. Parsing the former into ``activity_type``/``value_lower``/``activity_unit`` is what
    lets those rows become per-assay tasks in :func:`sample.ml_models.build_tasks` rather than only
    a group label -- and the cell line and assay are what make the number interpretable at all.

    A value that does not parse cleanly is left as ``value_raw`` with no numeric fields set: an
    unparsed potency must never become a fabricated number (the project's standing rule), and the
    raw string is still worth keeping for a human reader.

    :param row: One row of the combined table, or ``{}`` for a source with no detail.
    :returns: An activity dict in the loader schema.
    """
    activity: Dict[str, Any] = {
        "activity_type": "", "value_lower": None, "activity_unit": "", "value_operator": "",
        "value_raw": "", "value_raw_unit": "", "target_organism": "", "target_cell_line": "",
        "target_group": "anticancer", "assay_medium": "", "assay_ph": "",
    }
    if not row:
        return activity

    raw = (_dctpep_value(row.get("cancerppd_activity_raw"))
           or _dctpep_value(row.get("dctpep_cytotoxicity_raw")))
    cell_line = _dctpep_value(row.get("cancerppd_cell_line"))
    assay = _dctpep_value(row.get("cancerppd_assay"))

    # DCTPep writes cytotoxicity as "NIH 3T3: IC50=16 µM" -- the cell line precedes the colon.
    if not cell_line and ":" in raw:
        head, _, tail = raw.partition(":")
        if len(head) <= 40 and _ACTIVITY_MEASUREMENT.search(tail):
            cell_line, raw = head.strip(), tail.strip()

    activity["target_cell_line"] = cell_line
    activity["assay_medium"] = assay
    if raw:
        activity["value_raw"] = raw[:200]

    match = _ACTIVITY_MEASUREMENT.search(raw)
    if match:
        activity["activity_type"] = re.sub(r"\s+", "", match.group("type")).upper()
        activity["value_operator"] = (match.group("operator") or "").replace("=", "").strip()
        try:
            activity["value_lower"] = float(match.group("value"))
        except ValueError:
            activity["value_lower"] = None
        unit = match.group("unit").replace("u", "µ") if match.group("unit") else ""
        activity["activity_unit"] = unit
        activity["value_raw_unit"] = unit
    return activity


def _anticancer_publications(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    :param row: One row of the combined table.
    :returns: Publication records for the activity table. CancerPPD gives a PMID per entry, which
        is what links a measurement back to the paper that reported it; DCTPep's ``Literature``
        column is free text, so it is passed through as a citation string rather than parsed into
        an identifier this project would then have to trust.
    """
    publications: List[Dict[str, str]] = []
    pmid = _clean_accession(row.get("cancerppd_pmid"))
    if pmid.isdigit():
        publications.append({"pmid": pmid, "doi": "", "citation": ""})
    literature = _dctpep_value(row.get("dctpep_reference"))
    if literature and not publications:
        # DCTPep packs its reference as "PMID++Title++DOI++Year" (verified against the real file).
        # Splitting it out means the DOI becomes a resolvable identifier rather than being buried
        # in a citation string; the fields are matched by shape, not position, since "Not
        # available" appears in any of them.
        parts = [p.strip() for p in literature.split("++") if p.strip()]
        pmid_part = next((p for p in parts if p.isdigit()), "")
        doi_part = next((p for p in parts if p.lower().startswith("10.")), "")
        title = next((p for p in parts
                      if p != pmid_part and p != doi_part and not p.isdigit()
                      and p.lower() not in DCTPEP_NULLS), "")
        if pmid_part or doi_part or title:
            publications.append({"pmid": pmid_part, "doi": doi_part, "citation": title[:300]})
    return publications


def _anticancer_cross_references(row: Dict[str, Any]) -> Dict[str, str]:
    """
    :param row: One row of the combined table.
    :returns: The source databases' own accessions, so an imported peptide stays traceable to every
        database that catalogued it. Only non-empty values are kept.
    """
    candidates = {
        "cancerppd": row.get("cancerppd_source_id"),
        "dctpep": row.get("dctpep_source_id"),
        "dramp": row.get("dctpep_dramp_id"),
        "dbaasp": row.get("dctpep_dbaasp_id"),
        "uniprot": row.get("dctpep_uniprot_id"),
        "pubchem": row.get("dctpep_pubchem_cid"),
        "pdb": row.get("dctpep_pdb_id") or row.get("cancerppd_pdb_id"),
    }
    return {k: _clean_accession(v) for k, v in candidates.items() if _clean_accession(v)}


def consensus_frame(data_dir: Optional[str] = None,
                    output_csv: Optional[str] = None) -> "Any":
    """
    Anticancer positives as a ``{internal_id, source, sequence}`` frame for :func:`sample.merge.merge_databases`.

    This is the features side of the integration: including these sequences in the consensus is what
    gives each anticancer peptide an ``entry_id``, a structure and (downstream) descriptor/embedding
    features, so it reaches ``final_dataset.csv`` rather than existing only as a label. A sequence
    already present from another database merges into its existing consensus row; a genuinely novel
    one becomes a new entry.

    :param data_dir: Directory holding the downloaded raw files; the default when omitted.
    :param output_csv: The combined table to read; the packaged location when omitted. ``merge``
        passes a path derived from its own ``base_dir`` so a build against a different data root (a
        test, a custom checkout) reads *that* root's anticancer import rather than the packaged one.
    :returns: A DataFrame of AntiCP2 anticancer positives (empty when none imported).
    """
    import pandas as pd

    rows = [{"internal_id": f"ANTICP2_{index:05d}", "source": "AntiCP2", "sequence": sequence,
             "reference": "", "biological_activity": "anticancer"}
            for index, sequence in enumerate(
                positive_sequences(data_dir, output_csv=output_csv), start=1)]
    return pd.DataFrame(rows, columns=["internal_id", "source", "sequence",
                                       "reference", "biological_activity"])


def build_anticancer_table(data_dir: Optional[str] = None,
                           output_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds the combined, de-duplicated anticancer peptide table from every importable source.

    De-duplication is positive-preferring: if the same sequence is labelled anticancer in one place
    and a hard negative in another, the anticancer label wins, since a reported activity outweighs an
    unreported one.

    :param data_dir: Directory holding the downloaded raw files; the default when omitted.
    :param output_csv: Where to write the table; ``data/anticancer_databases/anticancer_peptides.csv``
        when omitted.
    :returns: A summary (``path``, ``rows``, ``positives``, ``by_source``).
    """
    import pandas as pd

    output_csv = output_csv or os.path.join(ANTICANCER_DIR, "anticancer_peptides.csv")

    frames = []
    # Each source is loaded independently and a missing one is skipped with a warning rather than
    # aborting the build: a checkout that has only downloaded AntiCP2 must still produce a table,
    # the same way the pipeline tolerates any one external database being unavailable.
    for name, loader in (("AntiCP2", lambda: load_anticp2(
                              data_dir and os.path.join(data_dir, "anticp2"))),
                         ("DCTPep", lambda: load_dctpep(
                              data_dir and os.path.join(data_dir, "dctpep"))),
                         ("CancerPPD", lambda: load_cancerppd(
                              data_dir and os.path.join(data_dir, "cancerppd")))):
        try:
            frames.append(loader())
        except FileNotFoundError as e:
            logger.warning("%s not imported: %s", name, e)

    if not frames or not any(len(f) for f in frames):
        raise FileNotFoundError(
            "No anticancer source could be loaded. Run `sample-cli import-anticancer --download`.")
    combined = pd.concat([f for f in frames if len(f)], ignore_index=True)

    # Positive-preferring de-dup: sort label descending so the kept row for a sequence is a positive
    # when any occurrence is positive.
    combined = (combined.sort_values("label", ascending=False)
                        .drop_duplicates(subset=["sequence"], keep="first")
                        .reset_index(drop=True))

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    combined.to_csv(output_csv, index=False)

    positives = int((combined["label"] == 1).sum())
    by_source = combined["source_db"].value_counts().to_dict()
    logger.info("Anticancer table: %d unique sequences (%d anticancer positives) from %s -> %s.",
                len(combined), positives, by_source, output_csv)
    return {"path": output_csv, "rows": len(combined), "positives": positives,
            "by_source": by_source}
