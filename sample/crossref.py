"""
Reference resolution against CrossRef and the NCBI ID Converter.

The source databases split cleanly by identifier: DBAASP, DRAMP and CAMP know PubMed ids and no
DOIs, while ADP and SAMPLE's own literature harvest know DOIs and no PubMed ids. Neither half can
be linked to the other locally, so a reader holding a DBAASP citation cannot reach the DOI and a
reader holding a SAMPLE DOI cannot reach PubMed. This module closes that gap in both directions
and fills in the bibliographic detail an APA citation needs.

Two services, chosen for different jobs:

* **NCBI E-utilities** does the identifier mapping. ``esummary`` returns a PubMed record's
  ``articleids``, DOI included, for 200 ids per request, so the roughly 7,000 unresolved PMIDs
  cost about 35 requests. CrossRef can only *search* for a PMID, which means guessing from
  bibliographic similarity -- the wrong tool for a mapping that has an exact answer.

  The obvious candidate, the PMC ID Converter, is not used: it now answers **403 Forbidden**.
  Worse for a pipeline, it fails silently rather than loudly -- every PMID simply resolves to "no
  DOI", which is indistinguishable from an old paper that genuinely has none. ``esummary`` also
  covers all of PubMed, whereas the converter only ever covered PMC deposits.
* **CrossRef** supplies the metadata: authors with given/family split, container title, volume,
  issue, pages and the published date. That is what turns a bare identifier into a citation, and
  it is the service the request named.

Everything is cached in ``data/crossref_cache.sqlite``, including negative results. A DOI that
CrossRef does not hold will not be found on a later run either, and re-asking on every pipeline
run would waste the politeness budget that keeps this workload welcome.

**Set ``SAMPLE_CROSSREF_MAILTO``.** CrossRef routes requests carrying a contact address into a
faster, more reliable pool and the address is how they warn you before blocking a misbehaving
client. Without it this module still works, on the shared anonymous pool.
"""
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sample.references import Reference, extract_doi, extract_pmid

logger = logging.getLogger("sample.crossref")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CROSSREF_API = "https://api.crossref.org/works"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

#: Ids per E-utilities request. 200 is the largest NCBI accepts on a GET.
EUTILS_BATCH = 200

#: DOIs per ``esearch`` when going the other way. Kept well below the id batch because each DOI
#: becomes its own ``[AID]`` clause and the query has to stay inside URL length limits.
DOI_SEARCH_BATCH = 50

#: Minimum seconds between requests to one host. CrossRef publishes no hard limit for the polite
#: pool but asks for restraint; 20/s is well inside what they describe as reasonable and finishes
#: 9,333 lookups in about eight minutes. NCBI's documented ceiling is 3/s anonymous and 10/s with
#: an API key, and it enforces them -- :func:`_ncbi_interval` picks the right one.
MIN_INTERVAL = {"api.crossref.org": 0.05, "eutils.ncbi.nlm.nih.gov": 0.34}

#: Attempts per request before giving up on one identifier.
MAX_RETRIES = 3

_CACHE_CONN: Optional[sqlite3.Connection] = None
_CACHE_LOCK = threading.Lock()
_LAST_REQUEST: Dict[str, float] = {}
_THROTTLE_LOCK = threading.Lock()


def mailto() -> str:
    """
    :returns: The contact address sent to CrossRef, from ``SAMPLE_CROSSREF_MAILTO``. Empty when
        unset, which drops the client into the anonymous pool rather than failing -- an
        unconfigured environment should still be able to run the pipeline.
    """
    return os.environ.get("SAMPLE_CROSSREF_MAILTO", "").strip()


def user_agent() -> str:
    """
    :returns: A User-Agent identifying this project, with the contact address appended when set.
        CrossRef asks clients to identify themselves; an anonymous agent is rate-limited harder.
    """
    base = "sAMPle/1.0 (https://github.com/fredericokremer/SAMPLE)"
    address = mailto()
    return f"{base} mailto:{address}" if address else base


def _get_cache() -> sqlite3.Connection:
    """
    :returns: The persistent lookup cache, created on first use. Survives across runs, unlike an
        in-process memo, because these lookups are expensive and their answers are stable.
    """
    global _CACHE_CONN
    if _CACHE_CONN is None:
        with _CACHE_LOCK:
            if _CACHE_CONN is None:
                os.makedirs(DATA_DIR, exist_ok=True)
                conn = sqlite3.connect(
                    os.path.join(DATA_DIR, "crossref_cache.sqlite"), check_same_thread=False)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS lookup ("
                    "  kind TEXT NOT NULL,"
                    "  query TEXT NOT NULL,"
                    "  payload TEXT,"          # JSON, or NULL for a cached miss
                    "  fetched_at REAL,"
                    "  PRIMARY KEY (kind, query))")
                conn.commit()
                _CACHE_CONN = conn
    return _CACHE_CONN


def cache_get(kind: str, query: str) -> Tuple[bool, Any]:
    """
    :param kind: Lookup family, e.g. ``"doi"`` or ``"pmid2doi"``.
    :param query: The identifier looked up.
    :returns: ``(hit, payload)``. ``hit`` distinguishes "cached as absent" from "never asked",
        which matters because a negative result is worth remembering.
    """
    row = _get_cache().execute(
        "SELECT payload FROM lookup WHERE kind = ? AND query = ?", (kind, query)).fetchone()
    if row is None:
        return False, None
    return True, (json.loads(row[0]) if row[0] else None)


def cache_put(kind: str, query: str, payload: Any) -> None:
    """
    :param kind: Lookup family.
    :param query: The identifier looked up.
    :param payload: Result to store; ``None`` records a miss.
    :returns: ``None``.
    """
    conn = _get_cache()
    with _CACHE_LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO lookup (kind, query, payload, fetched_at) VALUES (?,?,?,?)",
            (kind, query, json.dumps(payload) if payload is not None else None, time.time()))
        conn.commit()


def ncbi_params() -> Dict[str, str]:
    """
    :returns: The identification parameters NCBI asks every client to send. ``NCBI_API_KEY``
        raises the rate ceiling from 3 to 10 requests per second when set.
    """
    params = {"tool": "sample"}
    address = mailto()
    if address:
        params["email"] = address
    key = os.environ.get("NCBI_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    return params


def _ncbi_interval() -> float:
    """
    :returns: Seconds to leave between E-utilities calls -- 0.11 with an API key, 0.34 without,
        each just inside the documented ceiling.
    """
    return 0.11 if os.environ.get("NCBI_API_KEY", "").strip() else 0.34


def _throttle(host: str) -> None:
    """
    Sleeps until this host's minimum interval has elapsed.

    :param host: Hostname being called.
    :returns: ``None``.
    """
    interval = (_ncbi_interval() if host == "eutils.ncbi.nlm.nih.gov"
                else MIN_INTERVAL.get(host, 0.2))
    with _THROTTLE_LOCK:
        elapsed = time.monotonic() - _LAST_REQUEST.get(host, 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _LAST_REQUEST[host] = time.monotonic()


def _request(url: str, params: Optional[Dict[str, Any]] = None,
             timeout: float = 30.0) -> Optional[Any]:
    """
    Performs one throttled GET and decodes JSON.

    :param url: Absolute URL.
    :param params: Query parameters.
    :param timeout: Per-attempt timeout in seconds.
    :returns: The decoded body, or ``None`` when the resource is absent or the request failed
        after :data:`MAX_RETRIES`. A 404 is a definitive answer and is not retried.
    """
    import requests

    host = url.split("/")[2]
    for attempt in range(MAX_RETRIES):
        _throttle(host)
        try:
            response = requests.get(
                url, params=params, timeout=timeout,
                headers={"User-Agent": user_agent(), "Accept": "application/json"})
            if response.status_code == 404:
                return None
            if response.status_code in (429, 500, 502, 503, 504):
                wait = float(response.headers.get("Retry-After", 2 ** attempt))
                logger.debug(f"{host} returned {response.status_code}; waiting {wait}s.")
                time.sleep(min(wait, 30.0))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:  # network error, timeout, or malformed JSON
            logger.debug(f"Request to {url} failed ({attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt)
    return None


def _esummary(pmids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches PubMed summary records.

    :param pmids: Up to :data:`EUTILS_BATCH` PubMed identifiers.
    :returns: A mapping from PMID to its summary record; empty when the request failed. An empty
        result is deliberately indistinguishable from a failure *to this function* -- the caller
        checks for failure separately, because "the service did not answer" must never be cached
        as "this identifier has no DOI".
    """
    body = _request(f"{EUTILS}/esummary.fcgi", params={
        **ncbi_params(), "db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    result = (body or {}).get("result") or {}
    return {uid: result[uid] for uid in result.get("uids", []) if uid in result}


def _articleids(record: Dict[str, Any]) -> Dict[str, str]:
    """
    :param record: One ESummary record.
    :returns: Its ``articleids`` flattened to ``{idtype: value}``, e.g. ``{"doi": "10.1021/..."}``.
    """
    return {str(a.get("idtype", "")).lower(): str(a.get("value", ""))
            for a in record.get("articleids") or []}


def resolve_pmids_to_dois(pmids: Sequence[str]) -> Dict[str, Optional[str]]:
    """
    Maps PubMed ids to DOIs through E-utilities.

    :param pmids: PubMed identifiers.
    :returns: A mapping from PMID to DOI, with ``None`` where no DOI exists. Absence is a real
        answer here -- plenty of older PubMed records genuinely predate DOIs -- so it is cached
        like any other, but only once the service has actually replied.
    """
    resolved: Dict[str, Optional[str]] = {}
    pending: List[str] = []

    for pmid in {str(p) for p in pmids if p}:
        hit, payload = cache_get("pmid2doi", pmid)
        if hit:
            resolved[pmid] = payload
        else:
            pending.append(pmid)

    logger.info(f"PMID->DOI: {len(resolved)} cached, {len(pending)} to fetch.")
    for start in range(0, len(pending), EUTILS_BATCH):
        batch = pending[start:start + EUTILS_BATCH]
        records = _esummary(batch)
        if not records:
            logger.warning(f"No ESummary response for {len(batch)} PMIDs; leaving them unresolved.")
            continue
        for pmid in batch:
            record = records.get(pmid)
            if record is None:
                continue  # PubMed did not return this id; ask again on a later run.
            doi = extract_doi(_articleids(record).get("doi"))
            resolved[pmid] = doi
            cache_put("pmid2doi", pmid, doi)
        if start and start % (EUTILS_BATCH * 10) == 0:
            logger.info(f"  ...{start}/{len(pending)} PMIDs resolved.")

    return resolved


def fetch_work(doi: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves one work's metadata from CrossRef.

    :param doi: The DOI to look up.
    :returns: CrossRef's ``message`` object, or ``None`` when CrossRef does not hold the DOI.
    """
    key = doi.lower()
    hit, payload = cache_get("work", key)
    if hit:
        return payload

    body = _request(f"{CROSSREF_API}/{key}")
    message = (body or {}).get("message") if isinstance(body, dict) else None
    cache_put("work", key, message)
    return message


def prefetch_works(dois: Sequence[str], workers: int = 8) -> int:
    """
    Warms the cache with many CrossRef records at once.

    One work per request is unavoidable -- CrossRef has no bulk endpoint keyed by a DOI list --
    so 7,600 lookups run serially take most of an hour, dominated by round-trip latency rather
    than by the rate limit. Issuing them concurrently keeps the *rate* identical, because
    :func:`_throttle` holds a process-wide lock and still admits only one request per interval;
    what overlaps is the waiting.

    :param dois: DOIs to fetch. Already-cached entries are skipped without a request.
    :param workers: Concurrent requests in flight.
    :returns: The number of DOIs fetched from the network.
    """
    import concurrent.futures

    pending = []
    for doi in {d.lower() for d in dois if d}:
        hit, _ = cache_get("work", doi)
        if not hit:
            pending.append(doi)

    if not pending:
        return 0

    logger.info(f"Fetching {len(pending)} CrossRef records with {workers} workers...")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch_work, pending):
            done += 1
            if done % 500 == 0:
                logger.info(f"  ...{done}/{len(pending)} CrossRef records fetched.")
    return done


def work_to_reference(message: Dict[str, Any]) -> Reference:
    """
    Converts a CrossRef work into a :class:`~sample.references.Reference`.

    CrossRef gives authors as separate ``given``/``family`` fields, which is strictly better than
    the source databases' run-together ``Surname XY`` -- so these names are recomposed into
    ``Surname XY`` form for :func:`~sample.references.format_author` to render, keeping one code
    path for author formatting rather than two that could disagree.

    :param message: A CrossRef ``message`` object.
    :returns: The equivalent reference record.
    """
    authors: List[str] = []
    for author in message.get("author") or []:
        family = (author.get("family") or "").strip()
        given = (author.get("given") or "").strip()
        if family and given:
            initials = "".join(part[0].upper() for part in given.replace(".", " ").split() if part)
            authors.append(f"{family} {initials}")
        elif family:
            authors.append(family)
        elif author.get("name"):
            authors.append(str(author["name"]).strip())

    # `published` covers print and online; `issued` is the fallback CrossRef always populates.
    date = (message.get("published") or message.get("issued") or {})
    parts = (date.get("date-parts") or [[None]])[0]
    year = parts[0] if parts and isinstance(parts[0], int) else None

    titles = message.get("title") or []
    containers = message.get("container-title") or []

    return Reference(
        doi=extract_doi(message.get("DOI")),
        title=str(titles[0]).strip() if titles else None,
        authors=authors,
        journal=str(containers[0]).strip() if containers else None,
        year=year,
        volume=str(message["volume"]).strip() if message.get("volume") else None,
        issue=str(message["issue"]).strip() if message.get("issue") else None,
        pages=str(message["page"]).strip() if message.get("page") else None,
        source_db="CrossRef",
    )


def search_by_title(title: str, author_hint: str = "") -> Optional[Dict[str, Any]]:
    """
    Finds a work from its title, for references carrying no identifier at all.

    About 1,600 harvested references have a title and nothing else. A bibliographic search can
    recover those, but it is a guess rather than a lookup, so a result is accepted only when
    CrossRef's own relevance score clears :data:`TITLE_MATCH_MIN_SCORE` -- a wrong DOI attached to
    a real measurement is worse than no DOI, because it sends a reader to a different paper and
    looks authoritative doing it.

    :param title: The article title.
    :param author_hint: First author surname, which sharpens the query when supplied.
    :returns: The matched ``message`` object, or ``None``.
    """
    query = " ".join(x for x in (title, author_hint) if x).strip()
    if len(query) < 20:
        # Too little to identify a work; a short query returns confident nonsense.
        return None

    key = query.lower()[:300]
    hit, payload = cache_get("title", key)
    if hit:
        return payload

    body = _request(CROSSREF_API, params={
        "query.bibliographic": query, "rows": 1, "select":
            "DOI,title,author,container-title,volume,issue,page,published,issued,score"})
    items = ((body or {}).get("message") or {}).get("items") or []
    match = items[0] if items else None

    if match is not None and float(match.get("score", 0.0)) < TITLE_MATCH_MIN_SCORE:
        logger.debug(f"Rejected weak title match ({match.get('score')}) for {title[:60]!r}.")
        match = None

    cache_put("title", key, match)
    return match


#: CrossRef relevance score below which a title search result is discarded. CrossRef scores are
#: unnormalised, but scores under about 60 in this corpus were consistently different papers that
#: merely shared vocabulary ("antimicrobial peptide", "activity").
TITLE_MATCH_MIN_SCORE: float = 60.0


def enrich(references: Dict[str, Reference], resolve_titles: bool = True,
           limit: Optional[int] = None,
           keys: Optional[Set[str]] = None) -> Dict[str, Reference]:
    """
    Fills in DOIs and bibliographic detail for a harvested reference set.

    Three passes, cheapest first: map known PMIDs to DOIs in bulk, look up every resulting DOI in
    CrossRef for full metadata, then -- only for references still carrying no identifier -- try a
    bibliographic title search.

    Harvested values win over CrossRef's on conflict. The source databases were curated against
    the paper itself, whereas CrossRef reflects what the publisher deposited, which is
    occasionally truncated or missing an author list. CrossRef fills gaps rather than overwriting.

    :param references: Harvested references, keyed as by
        :func:`~sample.references.harvest_all`.
    :param resolve_titles: Whether to run the third pass. Off makes the run substantially shorter
        and gives up only on identifier-less records.
    :param limit: Process at most this many references, for a smoke test against the live API.
    :param keys: Restrict network resolution to these keys. References outside the set are
        returned exactly as harvested -- they keep their DOI, title and source, and stay in the
        bibliography, they simply are not looked up.

        This matters because the corpus harvester dwarfs the rest.
        :func:`~sample.references.harvest_sample_literature` yields every paper SAMPLE downloaded,
        keyed by its own DOI, which took the bibliography from 9,333 references to 94,564. Those
        DOIs match no ``papers.source_id``, so none of them reaches a peptide. Resolving them all
        meant 35,778 CrossRef fetches at roughly 4/s -- about 2.5 hours -- for records no page can
        display. Passing the linked keys spends the network on the references someone will read.
    :returns: A new mapping with the same keys and enriched values.
    """
    items = list(references.items())
    if limit is not None:
        items = items[:limit]
    # Split rather than filter: unresolved references still belong in the returned mapping, or the
    # bibliography would lose every corpus paper the moment this is used.
    if keys is not None:
        deferred = [(k, r) for k, r in items if k not in keys]
        items = [(k, r) for k, r in items if k in keys]
        logger.info(f"CrossRef: resolving {len(items)} peptide-linked references, "
                    f"passing through {len(deferred)} unlinked.")
    else:
        deferred = []

    # Pass 1 -- PMID to DOI, in batches.
    pmids = [ref.pmid for _, ref in items if ref.pmid and not ref.doi]
    mapping = resolve_pmids_to_dois(pmids) if pmids else {}

    # Pass 2 preparation -- warm the CrossRef cache concurrently, so the loop below is a series of
    # cache hits rather than a series of round trips.
    prefetch_works([ref.doi for _, ref in items if ref.doi]
                   + [doi for doi in mapping.values() if doi])

    enriched: Dict[str, Reference] = {}
    n_doi_from_pmid = n_crossref = n_title = 0

    for key, ref in items:
        if not ref.doi and ref.pmid:
            doi = mapping.get(str(ref.pmid))
            if doi:
                ref = Reference(**{**ref.__dict__, "doi": doi})
                n_doi_from_pmid += 1

        # Pass 2 -- CrossRef metadata for anything with a DOI.
        if ref.doi:
            message = fetch_work(ref.doi)
            if message:
                ref = ref.merge(work_to_reference(message))
                n_crossref += 1

        # Pass 3 -- last resort for records with no identifier at all.
        elif resolve_titles and ref.title and not ref.pmid:
            message = search_by_title(
                ref.title, ref.authors[0].split()[0] if ref.authors else "")
            if message:
                ref = ref.merge(work_to_reference(message))
                n_title += 1

        enriched[key] = ref

    # Unlinked references pass through untouched, so restricting the network work never costs the
    # bibliography an entry.
    for key, ref in deferred:
        enriched[key] = ref

    logger.info(f"Enrichment: {n_doi_from_pmid} DOIs recovered from PMIDs, "
                f"{n_crossref} works fetched from CrossRef, {n_title} matched by title.")
    return enriched


def backfill_pmids(references: Dict[str, Reference],
                   keys: Optional[Set[str]] = None) -> Dict[str, Reference]:
    """
    Recovers PubMed ids for references known only by DOI.

    This is the reverse of :func:`resolve_pmids_to_dois`, and it is what gives SAMPLE's own
    literature -- 152,534 DOIs and not one PMID, because step 1 never recorded them -- a PubMed
    link.

    :param references: References to complete, modified only where a PMID is missing.
    :param keys: Restrict the lookups to these keys, for the reason given in :func:`enrich`. Every
        reference is still returned; the rest simply keep whatever PMID they were harvested with.
        This pass is the more expensive of the two -- it queries NCBI twice per batch, since no
        endpoint maps DOI to PMID directly -- so leaving it unscoped undoes the saving.
    :returns: A new mapping with PMIDs filled in where the converter knew one.
    """
    dois = [ref.doi for key, ref in references.items()
            if ref.doi and not ref.pmid and (keys is None or key in keys)]
    resolved: Dict[str, str] = {}
    pending: List[str] = []

    for doi in {d.lower() for d in dois}:
        hit, payload = cache_get("doi2pmid", doi)
        if hit:
            if payload:
                resolved[doi] = payload
        else:
            pending.append(doi)

    logger.info(f"DOI->PMID: {len(resolved)} cached, {len(pending)} to fetch.")
    for start in range(0, len(pending), DOI_SEARCH_BATCH):
        batch = pending[start:start + DOI_SEARCH_BATCH]

        # Two steps, because no NCBI endpoint maps DOI to PMID directly. `esearch` over OR'd
        # `[AID]` clauses returns the matching PMIDs but not which DOI each answers, so the
        # PMIDs are then summarised and the mapping read back off their own DOI field. That also
        # makes the match self-verifying: a PMID is only accepted for a DOI it reports itself.
        search = _request(f"{EUTILS}/esearch.fcgi", params={
            **ncbi_params(), "db": "pubmed", "retmode": "json", "retmax": len(batch) * 2,
            "term": " OR ".join(f'"{doi}"[AID]' for doi in batch)})
        found_ids = ((search or {}).get("esearchresult") or {}).get("idlist") or []

        records = _esummary(found_ids) if found_ids else {}
        back: Dict[str, str] = {}
        for pmid, record in records.items():
            doi = extract_doi(_articleids(record).get("doi"))
            if doi:
                back[doi.lower()] = pmid

        if search is None:
            continue  # Service failure: leave the batch unresolved rather than caching a miss.
        for doi in batch:
            pmid = back.get(doi)
            if pmid:
                resolved[doi] = pmid
            cache_put("doi2pmid", doi, pmid)

    out: Dict[str, Reference] = {}
    for key, ref in references.items():
        pmid = resolved.get((ref.doi or "").lower())
        out[key] = Reference(**{**ref.__dict__, "pmid": pmid}) if (pmid and not ref.pmid) else ref
    return out
