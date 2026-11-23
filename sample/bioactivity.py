"""
Links peptides to the bioactivity archives: ChEMBL, PubChem BioAssay and BindingDB.

The release records activities as its source databases stated them -- an MIC, a target organism, a
unit -- with no route to the assay archives where the same molecule may have been screened against
hundreds of other targets. This adds that route, joining on identifiers the pipeline already
resolved: 2,860 ChEMBL ids and 606 PubChem CIDs from step 12's cache, and 1,598 UniProt accessions
recovered when ``cross_references`` was normalised.

**Expect uneven coverage, and read it as a finding rather than a failure.** These three archives
are built around small molecules and their protein targets, which is not what most of this dataset
is:

* **ChEMBL** is the productive join. It holds antimicrobial peptides as molecules in their own
  right, with assay descriptions and BAO ontology terms attached.
* **PubChem BioAssay** indexes screening campaigns, which are overwhelmingly small-molecule. Spot
  checks are stark: two ordinary small molecules returned 6,585 and 648 assays, while nisin and
  indolicidin -- both well studied, both present in PubChem -- returned none at all. A peptide
  having no PubChem assays says something about high-throughput screening practice, not about the
  peptide.
* **BindingDB** is *target*-centric: its UniProt key names the protein a ligand binds. The UniProt
  accessions held here identify the peptides themselves, so a hit only occurs where a peptide is
  also catalogued as a binding target. Empty results are the norm and are recorded as such.

Coverage is therefore measured and reported by :func:`link_all` rather than assumed, so a reader
can tell "not screened" from "not looked up".
"""
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("sample.bioactivity")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
BINDINGDB_API = "https://bindingdb.org/rest"

#: Minimum seconds between requests per host. PubChem documents five requests per second and
#: enforces it with 503s; the others are unpublished, so these are deliberately conservative.
MIN_INTERVAL = {
    "pubchem.ncbi.nlm.nih.gov": 0.25,
    "www.ebi.ac.uk": 0.15,
    "bindingdb.org": 0.5,
}

#: Activities pulled per ChEMBL molecule. A few molecules have thousands; the cap keeps one
#: promiscuous compound from dominating both the runtime and the stored result.
CHEMBL_ACTIVITY_LIMIT = 200

MAX_RETRIES = 3

_CACHE_CONN: Optional[sqlite3.Connection] = None
_CACHE_LOCK = threading.Lock()
_LAST_REQUEST: Dict[str, float] = {}
_THROTTLE_LOCK = threading.Lock()


def _get_cache() -> sqlite3.Connection:
    """
    :returns: The persistent bioactivity cache, created on first use.
    """
    global _CACHE_CONN
    if _CACHE_CONN is None:
        with _CACHE_LOCK:
            if _CACHE_CONN is None:
                os.makedirs(DATA_DIR, exist_ok=True)
                conn = sqlite3.connect(
                    os.path.join(DATA_DIR, "bioactivity_cache.sqlite"), check_same_thread=False)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS lookup ("
                    "  source TEXT NOT NULL,"
                    "  query TEXT NOT NULL,"
                    "  payload TEXT,"
                    "  fetched_at REAL,"
                    "  PRIMARY KEY (source, query))")
                conn.commit()
                _CACHE_CONN = conn
    return _CACHE_CONN


def cache_get(source: str, query: str) -> Tuple[bool, Any]:
    """
    :param source: Archive name.
    :param query: Identifier looked up.
    :returns: ``(hit, payload)``; ``hit`` separates a cached empty result from never having asked.
    """
    row = _get_cache().execute(
        "SELECT payload FROM lookup WHERE source = ? AND query = ?", (source, query)).fetchone()
    if row is None:
        return False, None
    return True, (json.loads(row[0]) if row[0] else None)


def cache_put(source: str, query: str, payload: Any) -> None:
    """
    :param source: Archive name.
    :param query: Identifier looked up.
    :param payload: Result to store; ``None`` records a definitive absence.
    :returns: ``None``.
    """
    conn = _get_cache()
    with _CACHE_LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO lookup (source, query, payload, fetched_at) VALUES (?,?,?,?)",
            (source, query, json.dumps(payload) if payload is not None else None, time.time()))
        conn.commit()


def _throttle(host: str) -> None:
    """
    :param host: Hostname being called.
    :returns: ``None``.
    """
    interval = MIN_INTERVAL.get(host, 0.3)
    with _THROTTLE_LOCK:
        elapsed = time.monotonic() - _LAST_REQUEST.get(host, 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _LAST_REQUEST[host] = time.monotonic()


def _request(url: str, params: Optional[Dict[str, Any]] = None,
             timeout: float = 45.0) -> Optional[Any]:
    """
    Performs one throttled GET and decodes JSON.

    :param url: Absolute URL.
    :param params: Query parameters.
    :param timeout: Per-attempt timeout.
    :returns: The decoded body, or ``None`` when absent or repeatedly failing.
    """
    import requests

    host = url.split("/")[2]
    for attempt in range(MAX_RETRIES):
        _throttle(host)
        try:
            response = requests.get(
                url, params=params, timeout=timeout,
                headers={"User-Agent": "sAMPle/1.0", "Accept": "application/json"})
            if response.status_code == 404:
                return None
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(float(response.headers.get("Retry-After", 2 ** attempt)), 30.0))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Request to {url} failed ({attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt)
    return None


def chembl_activities(chembl_id: str, limit: int = CHEMBL_ACTIVITY_LIMIT) -> List[Dict[str, Any]]:
    """
    Fetches measured activities for one ChEMBL molecule.

    :param chembl_id: A ChEMBL molecule id, e.g. ``CHEMBL415216``.
    :param limit: Maximum activities to keep.
    :returns: One record per activity, reduced to the fields worth storing. ``[]`` both when the
        molecule has no activities and when ChEMBL does not hold it -- the two are distinguished
        in the cache, not here.
    """
    hit, payload = cache_get("chembl", chembl_id)
    if hit:
        return payload or []

    body = _request(f"{CHEMBL_API}/activity", params={
        "molecule_chembl_id": chembl_id, "format": "json", "limit": limit})
    activities = (body or {}).get("activities") or []

    reduced = [{
        "activity_id": a.get("activity_id"),
        "assay_chembl_id": a.get("assay_chembl_id"),
        "assay_description": a.get("assay_description"),
        "assay_type": a.get("assay_type"),
        "target_chembl_id": a.get("target_chembl_id"),
        "target_organism": a.get("target_organism"),
        "target_pref_name": a.get("target_pref_name"),
        "standard_type": a.get("standard_type"),
        "standard_relation": a.get("standard_relation"),
        "standard_value": a.get("standard_value"),
        "standard_units": a.get("standard_units"),
        "bao_label": a.get("bao_label"),
        "document_chembl_id": a.get("document_chembl_id"),
    } for a in activities]

    if body is not None:
        cache_put("chembl", chembl_id, reduced)
    return reduced


def pubchem_assays(cid: str) -> List[str]:
    """
    Fetches the assay identifiers a PubChem compound appears in.

    :param cid: A PubChem compound id.
    :returns: The AIDs as strings; ``[]`` when the compound has been screened in none, which is
        the usual answer for a peptide (see the module docstring).
    """
    hit, payload = cache_get("pubchem_aids", str(cid))
    if hit:
        return payload or []

    body = _request(f"{PUBCHEM_API}/compound/cid/{cid}/aids/JSON")
    aids: List[str] = []
    if isinstance(body, dict) and "InformationList" in body:
        for info in body["InformationList"].get("Information") or []:
            aids.extend(str(a) for a in info.get("AID") or [])

    # A PUGREST.NotFound fault comes back as a decoded body, not an exception, and means "screened
    # in nothing" rather than "request failed" -- so it is cached.
    if body is not None:
        cache_put("pubchem_aids", str(cid), aids)
    return aids


def bindingdb_affinities(uniprot: str) -> List[Dict[str, Any]]:
    """
    Fetches BindingDB affinities recorded against one UniProt accession.

    :param uniprot: A UniProt accession.
    :returns: Affinity records, each with the measurement type, value and source publication.
        Usually ``[]`` here, because this key names a binding *target* and these accessions name
        the peptides themselves.
    """
    hit, payload = cache_get("bindingdb", uniprot)
    if hit:
        return payload or []

    body = _request(f"{BINDINGDB_API}/getLigandsByUniprots",
                    params={"uniprot": uniprot, "response": "application/json"})
    # The service misspells its own response key; matching on a prefix survives a correction.
    affinities: List[Dict[str, Any]] = []
    if isinstance(body, dict):
        for key, value in body.items():
            if key.lower().startswith("getlinds") or key.lower().startswith("getligands"):
                affinities = (value or {}).get("affinities") or []
                break

    reduced = [{
        "target": a.get("query"),
        "monomer_id": a.get("monomerid"),
        "smiles": a.get("smile"),
        "affinity_type": a.get("affinity_type"),
        "affinity": a.get("affinity"),
        "pmid": a.get("pmid"),
        "doi": a.get("doi"),
    } for a in affinities]

    if body is not None:
        cache_put("bindingdb", uniprot, reduced)
    return reduced


def _identifiers_from_cache() -> Dict[str, List[str]]:
    """
    Reads the ChEMBL and PubChem identifiers step 12 already resolved.

    :returns: ``{"chembl": [...], "pubchem": [...]}`` keyed by archive.
    """
    path = os.path.join(DATA_DIR, "external_db_cache.sqlite")
    found: Dict[str, List[str]] = {"chembl": [], "pubchem": []}
    if not os.path.exists(path):
        logger.warning(f"No external database cache at {path}; run step 14 first.")
        return found

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for source, ext_id in conn.execute(
                "SELECT source, ext_id FROM lookup_cache "
                "WHERE ext_id IS NOT NULL AND ext_id != ''"):
            if source in found:
                found[source].append(str(ext_id))
    finally:
        conn.close()

    for source in found:
        found[source] = sorted(set(found[source]))
    return found


def _uniprot_accessions(activities_csv: Optional[str] = None) -> List[str]:
    """
    Reads UniProt accessions out of the activity table's ``cross_references`` column.

    Not from ``papers``: that table's ``source_type`` holds source-*database* names, so it has no
    UniProt rows at all and querying it silently yields nothing to look up. The accessions live in
    the free-text ``cross_references`` cells, which is why they go back through
    :func:`~sample.external_links.normalize_cross_references` -- roughly three in five tokens in
    that column are not identifiers.

    :param activities_csv: Activity table; the packaged release when omitted.
    :returns: Distinct, validated accessions.
    """
    from sample.external_links import normalize_cross_references

    activities_csv = activities_csv or os.path.join(
        DATA_DIR, "sample_raw_release", "activities.csv")
    if not os.path.exists(activities_csv):
        logger.warning(f"No activity table at {activities_csv}; no UniProt accessions to link.")
        return []

    import pandas as pd

    frame = pd.read_csv(activities_csv, low_memory=False, usecols=["cross_references"])
    accessions: set = set()
    for blob in frame["cross_references"].dropna().unique():
        accessions.update(normalize_cross_references(blob).get("uniprot", []))
    return sorted(accessions)


def link_all(output_path: Optional[str] = None, limit: Optional[int] = None,
             sources: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Resolves every available identifier against the three archives and writes the result.

    :param output_path: Where to write the linkage; ``data/bioactivity.json`` when omitted.
    :param limit: Process at most this many identifiers per archive, for a smoke test.
    :param sources: Restrict to some of ``chembl``, ``pubchem``, ``bindingdb``.
    :returns: A coverage summary -- how many identifiers were looked up and how many returned
        anything, per archive.
    """
    chosen = set(sources) if sources else {"chembl", "pubchem", "bindingdb"}
    identifiers = _identifiers_from_cache()
    uniprots = _uniprot_accessions() if "bindingdb" in chosen else []

    result: Dict[str, Dict[str, Any]] = {"chembl": {}, "pubchem": {}, "bindingdb": {}}
    summary: Dict[str, Any] = {}

    if "chembl" in chosen:
        ids = identifiers["chembl"][:limit] if limit else identifiers["chembl"]
        logger.info(f"ChEMBL: resolving {len(ids)} molecules...")
        for index, chembl_id in enumerate(ids, start=1):
            activities = chembl_activities(chembl_id)
            if activities:
                result["chembl"][chembl_id] = activities
            if index % 250 == 0:
                logger.info(f"  ...{index}/{len(ids)} ChEMBL molecules "
                            f"({len(result['chembl'])} with activities).")
        summary["chembl"] = {
            "looked_up": len(ids), "with_data": len(result["chembl"]),
            "activities": sum(len(v) for v in result["chembl"].values())}

    if "pubchem" in chosen:
        ids = identifiers["pubchem"][:limit] if limit else identifiers["pubchem"]
        logger.info(f"PubChem BioAssay: resolving {len(ids)} compounds...")
        for index, cid in enumerate(ids, start=1):
            aids = pubchem_assays(cid)
            if aids:
                result["pubchem"][cid] = aids
            if index % 250 == 0:
                logger.info(f"  ...{index}/{len(ids)} PubChem compounds "
                            f"({len(result['pubchem'])} with assays).")
        summary["pubchem"] = {
            "looked_up": len(ids), "with_data": len(result["pubchem"]),
            "assays": sum(len(v) for v in result["pubchem"].values())}

    if "bindingdb" in chosen:
        ids = uniprots[:limit] if limit else uniprots
        logger.info(f"BindingDB: resolving {len(ids)} UniProt accessions...")
        for index, accession in enumerate(ids, start=1):
            affinities = bindingdb_affinities(accession)
            if affinities:
                result["bindingdb"][accession] = affinities
            if index % 100 == 0:
                logger.info(f"  ...{index}/{len(ids)} accessions "
                            f"({len(result['bindingdb'])} with affinities).")
        summary["bindingdb"] = {
            "looked_up": len(ids), "with_data": len(result["bindingdb"]),
            "affinities": sum(len(v) for v in result["bindingdb"].values())}

    output_path = output_path or os.path.join(DATA_DIR, "bioactivity.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)

    summary["path"] = output_path
    logger.info(f"Bioactivity linkage written to {output_path}: "
                + "; ".join(f"{k}={v['with_data']}/{v['looked_up']}"
                            for k, v in summary.items() if isinstance(v, dict)))
    return summary
