"""
Bibliographic references: harvesting, normalisation and APA formatting.

The release had no bibliography. ``papers`` looked like one, but its ``source_type`` values are
source-database names (``CAMP``, ``DBAASP``, ``ADP``, ``DBAMP``, ``DRAMP``) and its ``source_id``
values are that database's own accessions -- it records *where a peptide was copied from*, not
*which paper measured it*. Across 40,723 rows there was not one PMID, DOI or citation, so a reader
could not check any reported activity against its source.

Every source database ships the missing material, in a different shape:

===========  ==================================================================  =============
Source       Reference material                                                  Primary key
===========  ==================================================================  =============
DBAASP       28,918 article records: journal, year, volume, pages, title,        PMID
             authors, PubMed id -- and ``targetActivities[i].reference`` is a
             1-based index into that list, so references attach per activity
DRAMP        ``Pubmed_ID`` (~100 % populated), ``Reference``, ``Author``,        PMID
             ``Title``
CAMP         a ``PubMed`` key in ~96 % of records                                PMID
ADP          ``Title``, ``Author`` (with trailing year) and ``Reference``, an    DOI
             NLM citation string carrying an embedded DOI
dbAMP        none -- ``dbAMP3_pepinfo`` carries only UniProt/PDB/taxonomy        --
SAMPLE       ``pubmed_metadata.csv`` (99,105 rows, 96,736 DOIs) and              DOI
             ``scopus_metadata.csv`` (57,656 rows, 55,804 DOIs)
===========  ==================================================================  =============

Two consequences shape this module. First, the sources disagree about which identifier they
carry -- DBAASP/DRAMP/CAMP know PMIDs and no DOIs, while SAMPLE's own literature and ADP know DOIs
and no PMIDs (step 1 wrote a ``pmid`` column and left it entirely NaN). So references have to be
reconciled in both directions, which is what :mod:`sample.crossref` does. Second, only DBAASP
states which paper supports which measurement; for the other sources a reference is attached to
the peptide, and every activity from that peptide inherits it. That is weaker provenance and is
recorded as such in :attr:`Reference.scope`, so a reader is never told a citation is
activity-specific when it is not.
"""
import csv
import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger("sample.references")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: A DOI as it appears inside free text. Trailing sentence punctuation is excluded so that
#: ``doi: 10.1006/bbrc.1993.1314. PubMed.`` does not capture the full stop.
DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]*[^\s\"<>,;.])", re.IGNORECASE)

#: A PubMed identifier: 1--8 digits, not part of a longer number.
PMID_PATTERN = re.compile(r"\b(\d{1,8})\b")

#: The NLM citation format used by both ADP's and DRAMP's ``Reference`` field, e.g.
#: ``Appl Environ Microbiol. 1998 Sep;64(9):3140-3146.``
_NLM_PATTERN = re.compile(
    r"^\s*(?P<journal>[^.]+?)\.\s*"
    r"(?P<year>(?:19|20)\d{2})[^;]*?"
    r";\s*(?P<volume>\d+)"
    r"(?:\s*\((?P<issue>[^)]+)\))?"
    r"\s*:\s*(?P<pages>[\dA-Za-z]+(?:\s*[-–]\s*[\dA-Za-z]+)?)")

#: An ADP accession, e.g. ``AP00001``. Used to tell data rows from the file's two header lines.
_ADP_ACCESSION = re.compile(r"AP\d{3,6}", re.IGNORECASE)

#: Values that occupy an identifier field without identifying anything.
_NULL_TOKENS = frozenset({
    "", "-", "na", "n/a", "none", "null", "nan", "not available", "unknown", "no entry found",
})


def _is_null(value: Any) -> bool:
    """
    :param value: A candidate field value.
    :returns: Whether the value is missing or one of the sources' "no data" spellings.
    """
    return str(value).strip().lower() in _NULL_TOKENS if value is not None else True


def extract_doi(text: Any) -> Optional[str]:
    """
    Pulls a DOI out of free text.

    :param text: A citation string, URL or bare DOI.
    :returns: The lowercased DOI, or ``None``. Lowercased because DOIs are case-insensitive and
        the same article arrives capitalised differently from different sources; without
        normalising, deduplication would keep both.
    """
    if _is_null(text):
        return None
    match = DOI_PATTERN.search(str(text))
    return match.group(1).lower().rstrip(".") if match else None


def extract_pmid(text: Any) -> Optional[str]:
    """
    :param text: A PubMed id, a PubMed URL, or a field that may hold one.
    :returns: The PMID as a string, or ``None``. PMIDs are kept as strings, not ints, because
        they are identifiers rather than quantities and leading-zero forms appear in the wild.
    """
    if _is_null(text):
        return None
    raw = str(text).strip()
    # A PubMed URL ends with the id; a bare field is the id itself.
    tail = raw.rstrip("/").split("/")[-1]
    match = PMID_PATTERN.fullmatch(tail) or PMID_PATTERN.search(raw)
    if not match:
        return None
    pmid = match.group(1).lstrip("0") or "0"
    return pmid if pmid != "0" else None


@dataclass
class Reference:
    """
    One bibliographic record, however partially known.

    Fields are optional throughout because no source supplies all of them and a half-known
    reference still lets a reader find the paper. :meth:`key` decides what counts as the same
    work when merging across sources.
    """

    pmid: Optional[str] = None
    doi: Optional[str] = None
    scopus_eid: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    #: Which database this record came from, for provenance.
    source_db: Optional[str] = None
    #: ``"activity"`` when the source ties this reference to a specific measurement,
    #: ``"peptide"`` when it only ties it to the peptide as a whole.
    scope: str = "peptide"

    def key(self) -> str:
        """
        :returns: The identity used to merge duplicates. DOI first, then PMID, then a normalised
            title -- the title fallback matters because ADP and SAMPLE literature often describe
            the same paper with no shared identifier.
        """
        if self.doi:
            return f"doi:{self.doi}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        if self.title:
            return "title:" + re.sub(r"[^a-z0-9]+", "", self.title.lower())[:120]
        return f"anon:{id(self)}"

    def merge(self, other: "Reference") -> "Reference":
        """
        Combines two records of the same work, preferring values already present.

        :param other: Another record for the same work.
        :returns: A new record filling this one's gaps from ``other``. ``scope`` is upgraded to
            ``"activity"`` if either source knew the measurement-level link, since that is the
            stronger claim and it is justified as soon as one source supports it.
        """
        merged = replace(self)
        for name in ("pmid", "doi", "scopus_eid", "title", "journal", "year", "volume",
                     "issue", "pages"):
            if getattr(merged, name) in (None, "") and getattr(other, name) not in (None, ""):
                setattr(merged, name, getattr(other, name))
        if not merged.authors and other.authors:
            merged.authors = list(other.authors)
        if merged.scope != "activity" and other.scope == "activity":
            merged.scope = "activity"
        if merged.source_db and other.source_db and other.source_db not in merged.source_db:
            merged.source_db = f"{merged.source_db},{other.source_db}"
        return merged

    def url(self) -> Optional[str]:
        """
        :returns: The best link for this reference -- DOI resolver, else PubMed, else ``None``.
        """
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return None


def format_author(raw: str) -> Optional[str]:
    """
    Converts a source's author spelling into APA form.

    Sources write ``Mitova M``, ``Boulain JC`` or ``De Rosa S`` -- surname first, initials run
    together, no punctuation. APA wants ``Mitova, M.``, ``Boulain, J. C.`` and ``De Rosa, S.``.
    Multi-word surnames are preserved by treating only a short trailing all-caps token as
    initials.

    :param raw: One author as the source wrote it.
    :returns: The APA rendering, or ``None`` when the input is empty.
    """
    name = re.sub(r"\s+", " ", str(raw or "")).strip().strip(",")
    if not name:
        return None

    # Already APA-shaped ("Mitova, M.") -- trust it.
    if "," in name:
        return name if name.endswith(".") or len(name.split(",")[-1].strip()) > 3 else name

    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0]

    tail = parts[-1]
    if tail.isupper() and len(tail) <= 4:
        surname = " ".join(parts[:-1])
        initials = " ".join(f"{ch}." for ch in tail)
        return f"{surname}, {initials}"

    # Otherwise assume Western order ("Maria Mitova") and invert.
    surname = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
    return f"{surname}, {initials}"


def format_author_list(authors: Iterable[str]) -> str:
    """
    Joins authors following APA 7 rules.

    :param authors: Authors in source order.
    :returns: The joined list: ``&`` before the last of up to 20; for 21 or more, the first 19,
        an ellipsis, then the final author, which is what APA 7 prescribes.
    """
    formatted = [a for a in (format_author(x) for x in authors) if a]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def format_apa(ref: Reference, html: bool = False) -> str:
    """
    Renders a reference as an APA 7 journal citation.

    Every element is optional, because the sources are uneven: a record with only a title and a
    DOI still produces a usable citation rather than a string of placeholders. Titles are emitted
    as the source wrote them -- APA asks for sentence case, but recasing real titles corrupts
    proper nouns, gene symbols and species names, which is a worse error than a capital letter.

    :param ref: The reference to render.
    :param html: When ``True``, italicise the journal and volume with ``<em>`` as APA requires.
        The plain form is for CSV and JSON, where markup would be noise.
    :returns: The formatted citation.
    """
    def _em(text: str) -> str:
        return f"<em>{text}</em>" if html else text

    parts: List[str] = []
    authors = format_author_list(ref.authors)
    if authors:
        parts.append(authors if authors.endswith(".") else authors + ".")

    parts.append(f"({ref.year})." if ref.year else "(n.d.).")

    if ref.title:
        title = str(ref.title).strip()
        parts.append(title if title.endswith((".", "?", "!")) else title + ".")

    if ref.journal:
        locator = _em(str(ref.journal).strip().rstrip("."))
        if ref.volume:
            locator += f", {_em(str(ref.volume))}"
            if ref.issue:
                locator += f"({ref.issue})"
        if ref.pages:
            locator += f", {ref.pages}"
        parts.append(locator + ".")

    link = ref.url()
    if link:
        parts.append(f'<a href="{link}" target="_blank" rel="noopener">{link}</a>' if html
                     else link)

    return " ".join(p for p in parts if p).strip()


def parse_nlm_citation(text: Any) -> Dict[str, Any]:
    """
    Parses the NLM citation string used by ADP and DRAMP.

    ``Biochem Biophys Res Commun. 1993 Mar 31;191(3):983-90. doi: 10.1006/bbrc.1993.1314.``
    yields journal, year, volume, issue, pages and DOI. Both databases use the same format, so
    one parser serves both.

    :param text: The citation string.
    :returns: The fields recovered; an empty mapping when nothing matched.
    """
    if _is_null(text):
        return {}
    raw = str(text).strip()
    out: Dict[str, Any] = {}

    doi = extract_doi(raw)
    if doi:
        out["doi"] = doi

    match = _NLM_PATTERN.match(raw)
    if match:
        out["journal"] = match.group("journal").strip()
        out["year"] = int(match.group("year"))
        out["volume"] = match.group("volume")
        if match.group("issue"):
            out["issue"] = match.group("issue")
        out["pages"] = re.sub(r"\s*[-–]\s*", "-", match.group("pages"))
    else:
        # No structured match, but a bare year is still worth having.
        year = re.search(r"\b((?:19|20)\d{2})\b", raw)
        if year:
            out["year"] = int(year.group(1))
        journal = re.match(r"^([A-Za-z][A-Za-z .&'\-]{3,60}?)\.", raw)
        if journal:
            out["journal"] = journal.group(1).strip()
    return out


def split_author_string(text: Any) -> List[str]:
    """
    Splits the comma-separated author strings used by ADP and DRAMP.

    ``Heidrich C, Pag U, Josten M, Sahl HG.`` becomes four authors. ADP appends the publication
    year to the same field, which is stripped here so it does not become a fictitious author. The
    year is sometimes separated (``... Nicolas P. 1993``) and sometimes run straight onto the last
    author's initials (``... Broekaert WF1995``), so no word boundary is required before it --
    with one, the second form yielded an author named ``WF1995``.

    :param text: The raw author field.
    :returns: One entry per author, in source order.
    """
    if _is_null(text):
        return []
    raw = re.sub(r"[\s.]*(?:19|20)\d{2}\s*\.?\s*$", "", str(text).strip()).rstrip(".")
    return [part.strip() for part in raw.split(",") if part.strip()]


# --------------------------------------------------------------------------------------------
# Harvesters. Each yields ``(source_accession, Reference, activity_index)`` where
# ``activity_index`` is the source's own 1-based reference number when it links references to
# individual measurements, and ``None`` when the reference belongs to the peptide as a whole.
# --------------------------------------------------------------------------------------------

HarvestRow = Tuple[str, Reference, Optional[str]]


def harvest_dbaasp(directory: Optional[str] = None) -> Iterator[HarvestRow]:
    """
    Reads DBAASP peptide cards.

    DBAASP is the only source that ties a reference to a specific measurement:
    ``targetActivities[i].reference`` and ``synergies[i].reference`` hold a 1-based index into the
    record's ``articles`` list. That index is emitted so the caller can attach the citation to the
    activity rather than to the peptide.

    :param directory: DBAASP JSON directory; the packaged location when omitted.
    :returns: An iterator of harvested rows.
    """
    directory = directory or os.path.join(DATA_DIR, "antimicrobial_databases", "dbaasp")
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError) as e:
            logger.debug(f"Skipping unreadable DBAASP card {path}: {e}")
            continue

        accession = record.get("dbaaspId") or os.path.splitext(os.path.basename(path))[0]
        articles = record.get("articles") or []
        for index, article in enumerate(articles, start=1):
            journal = article.get("journal") or {}
            pubmed = article.get("pubmed") or {}
            yield accession, Reference(
                pmid=extract_pmid(pubmed.get("pubmedId")),
                title=(article.get("title") or "").strip() or None,
                authors=[a.get("name") for a in (article.get("authors") or []) if a.get("name")],
                journal=(journal.get("description") or journal.get("name") or "").strip() or None,
                year=int(article["year"]) if str(article.get("year", "")).isdigit() else None,
                volume=(article.get("volume") or "").strip() or None,
                pages=(article.get("pages") or "").strip() or None,
                source_db="DBAASP",
                scope="activity",
            ), str(index)


def harvest_dramp(path: Optional[str] = None) -> Iterator[HarvestRow]:
    """
    Reads DRAMP's ``general_amps`` table, which carries a PMID for essentially every entry.

    :param path: The DRAMP dataset file; the packaged location when omitted.
    :returns: An iterator of harvested rows.
    """
    path = path or os.path.join(
        DATA_DIR, "antimicrobial_databases", "dramp", "Datasets", "general_amps.txt")
    if not os.path.exists(path):
        logger.warning(f"DRAMP dataset not found at {path}.")
        return

    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession = (row.get("DRAMP_ID") or "").strip()
            if not accession:
                continue
            citation = parse_nlm_citation(row.get("Reference"))
            reference = Reference(
                pmid=extract_pmid(row.get("Pubmed_ID")),
                doi=citation.get("doi"),
                title=(row.get("Title") or "").strip() or None,
                authors=split_author_string(row.get("Author")),
                journal=citation.get("journal"),
                year=citation.get("year"),
                volume=citation.get("volume"),
                issue=citation.get("issue"),
                pages=citation.get("pages"),
                source_db="DRAMP",
            )
            if reference.pmid or reference.doi or reference.title:
                yield accession, reference, None


def harvest_camp(directory: Optional[str] = None) -> Iterator[HarvestRow]:
    """
    Reads CAMP's per-peptide key/value CSVs.

    CAMP stores one file per peptide with a ``key``/``value`` layout; the ``PubMed`` key holds the
    identifier and is present in roughly 96 % of records. A record may list several PMIDs.

    :param directory: CAMP directory; the packaged location when omitted.
    :returns: An iterator of harvested rows.
    """
    directory = directory or os.path.join(DATA_DIR, "antimicrobial_databases", "camp")
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        accession = os.path.splitext(os.path.basename(path))[0]
        fields: Dict[str, str] = {}
        try:
            with open(path, encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("key") or "").strip()
                    if key and not _is_null(row.get("value")):
                        fields.setdefault(key, str(row.get("value")).strip())
        except OSError as e:
            logger.debug(f"Skipping unreadable CAMP record {path}: {e}")
            continue

        title = fields.get("Title")
        for token in re.split(r"[,;\s]+", fields.get("PubMed", "")):
            pmid = extract_pmid(token)
            if pmid:
                yield accession, Reference(
                    pmid=pmid, title=title, source_db="CAMP"), None


def harvest_adp(path: Optional[str] = None) -> Iterator[HarvestRow]:
    """
    Reads ADP's ``adb.csv``.

    The file carries two non-data lines before the records: a numeric header (``0,1,...,16``) and
    then the human labels (``APD ID:``, ``Title:``, ...). Neither is skippable by
    :class:`csv.DictReader`, so columns are addressed positionally and rows are admitted only when
    the first field looks like an ADP accession -- otherwise the numeric header parses as a
    reference whose title is ``14`` and whose author is ``15``.

    ADP is the one source database carrying a DOI, embedded in its NLM-format ``Reference``
    string, though only about a quarter of its records include one.

    :param path: ADP CSV; the packaged location when omitted.
    :returns: An iterator of harvested rows.
    """
    path = path or os.path.join(DATA_DIR, "antimicrobial_databases", "adp", "adb.csv")
    if not os.path.exists(path):
        logger.warning(f"ADP dataset not found at {path}.")
        return

    # Positional, because neither leading line is a usable header.
    id_col, title_col, author_col, reference_col = 0, 14, 15, 16
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) <= reference_col:
                continue
            accession = row[id_col].strip()
            if not _ADP_ACCESSION.fullmatch(accession):
                continue
            citation = parse_nlm_citation(row[reference_col])
            reference = Reference(
                doi=citation.get("doi"),
                title=(row[title_col] or "").strip() or None,
                authors=split_author_string(row[author_col]),
                journal=citation.get("journal"),
                year=citation.get("year"),
                volume=citation.get("volume"),
                issue=citation.get("issue"),
                pages=citation.get("pages"),
                source_db="ADP",
            )
            if reference.doi or reference.title:
                yield accession, reference, None


def harvest_sample_literature(directory: Optional[str] = None) -> Iterator[HarvestRow]:
    """
    Reads SAMPLE's own PubMed and Scopus harvests.

    These are the papers SAMPLE searched and downloaded itself, keyed by DOI. The ``pmid`` column
    in the PubMed files is empty for every one of the 99,105 rows -- step 1 recorded the DOI and
    the link but never the identifier the search returned -- so PMIDs for this corpus have to come
    back from :mod:`sample.crossref`.

    :param directory: Literature directory; the packaged location when omitted.
    :returns: An iterator of harvested rows keyed by DOI rather than a database accession, since
        these references belong to the corpus rather than to one source database's peptide.
    """
    directory = directory or os.path.join(DATA_DIR, "literature_data")
    for name, source in (("pubmed_metadata.csv", "PubMed"),
                         ("scopus_metadata.csv", "Scopus")):
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            logger.warning(f"Literature metadata not found at {path}.")
            continue
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                doi = extract_doi(row.get("doi")) or extract_doi(row.get("link"))
                pmid = extract_pmid(row.get("pmid"))
                eid = None
                link = str(row.get("link") or "")
                if "scopus_id/" in link:
                    eid = link.rsplit("scopus_id/", 1)[-1].strip("/ ")
                if not (doi or pmid or eid):
                    continue
                yield (doi or pmid or f"scopus:{eid}"), Reference(
                    doi=doi, pmid=pmid, scopus_eid=eid,
                    title=(row.get("title") or "").strip() or None,
                    source_db=source,
                ), None


#: The harvesters, in the order their records are merged. Earlier sources win on conflicting
#: scalar fields, so DBAASP -- which supplies structured authors, journal, volume and pages --
#: leads, and the DOI-only corpora fill in behind it.
HARVESTERS = {
    "DBAASP": harvest_dbaasp,
    "DRAMP": harvest_dramp,
    "ADP": harvest_adp,
    "CAMP": harvest_camp,
    "SAMPLE": harvest_sample_literature,
}

#: Harvesters that describe the corpus rather than any peptide's citations.
#:
#: The four database harvesters key each row by a source accession that reaches a peptide through
#: the ``papers`` table. :func:`harvest_sample_literature` instead yields every paper SAMPLE
#: downloaded, keyed by its own DOI -- 154,386 rows against 63,806 from all four databases
#: combined -- and none of those DOIs match a ``papers.source_id``, so they attach to nothing.
#: They are still worth having in the bibliography, but resolving them against CrossRef costs
#: hours and buys nothing a reader can see; see the ``linked_only`` argument of
#: :func:`build_bibliography`.
CORPUS_HARVESTERS = frozenset({"SAMPLE"})


def harvest_all(sources: Optional[Iterable[str]] = None,
                ) -> Tuple[Dict[str, Reference], List[Tuple[str, str, Optional[str]]]]:
    """
    Runs every harvester and merges references that describe the same work.

    :param sources: Restrict to these harvester names; all of :data:`HARVESTERS` when omitted.
    :returns: ``(references, links)`` where ``references`` maps :meth:`Reference.key` to the
        merged record, and ``links`` holds ``(source_accession, reference_key, activity_index)``
        so a caller can attach each reference to the peptide -- or, for DBAASP, to the individual
        activity that cites it.
    """
    chosen = list(sources) if sources else list(HARVESTERS)
    references: Dict[str, Reference] = {}
    links: List[Tuple[str, str, Optional[str]]] = []

    for name in chosen:
        harvester = HARVESTERS.get(name)
        if harvester is None:
            logger.warning(f"No harvester named {name}; skipping.")
            continue
        count = 0
        for accession, reference, activity_index in harvester():
            key = reference.key()
            references[key] = (references[key].merge(reference)
                               if key in references else reference)
            links.append((accession, key, activity_index))
            count += 1
        logger.info(f"{name}: harvested {count} reference links.")

    logger.info(f"Merged into {len(references)} distinct references "
                f"from {len(links)} links.")
    return references, links


def build_bibliography(output_path: Optional[str] = None, sources: Optional[Iterable[str]] = None,
                       resolve: bool = True, resolve_titles: bool = True,
                       linked_only: bool = True) -> Dict[str, Any]:
    """
    Harvests, resolves and writes the bibliography consumed by step 44.

    Kept separate from the SQLite load because it is the only part that touches the network: the
    load can then be re-run offline any number of times against a bibliography that was resolved
    once.

    :param output_path: Where to write the JSON; ``data/references.json`` when omitted.
    :param sources: Harvesters to run; all when omitted.
    :param resolve: Whether to call out to CrossRef and E-utilities. ``False`` harvests local
        metadata only, which is what the tests and offline runs use.
    :param resolve_titles: Whether to attempt a bibliographic title search for references with no
        identifier at all.
    :param linked_only: Resolve only references that some peptide or activity links to. On by
        default: :func:`harvest_sample_literature` contributes the whole downloaded corpus with no
        peptide attached, which is 90 % of the reference set and none of what a page can show.
        Set ``False`` to resolve everything, which is hours of CrossRef traffic.
    :returns: A summary of what was written.
    """
    references, links = harvest_all(sources)

    if resolve:
        from sample import crossref

        # Only references something actually cites are worth a network round trip. The corpus
        # harvester attaches no peptide to its rows by design, so without this the run spends
        # hours resolving papers that no page can display -- see `crossref.enrich`.
        # Which harvester produced a reference decides whether a page can show it, and the
        # accession does not: `harvest_sample_literature` uses the DOI itself as its accession, so
        # testing for a non-empty accession marks all 94,564 references as linked and saves
        # nothing. The four database harvesters emit a source accession (`DBAASPN_10055`,
        # `AP00001`, `CAMPSQ1`, `DRAMP00005`) that reaches a peptide through the `papers` table;
        # the corpus harvester's DOIs match nothing there.
        linked = None
        if linked_only:
            linked = set()
            for name in (list(sources) if sources else list(HARVESTERS)):
                harvester = HARVESTERS.get(name)
                if harvester is None or name in CORPUS_HARVESTERS:
                    continue
                linked |= {reference.key() for _, reference, _ in harvester()}
        references = crossref.enrich(references, resolve_titles=resolve_titles, keys=linked)
        references = crossref.backfill_pmids(references, keys=linked)

    payload = {
        "references": {
            key: {**ref.__dict__, "apa": format_apa(ref), "url": ref.url()}
            for key, ref in references.items()
        },
        "links": [
            {"accession": accession, "reference": key, "activity_index": index}
            for accession, key, index in links
        ],
    }

    output_path = output_path or os.path.join(DATA_DIR, "references.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    summary = {
        "path": output_path,
        "references": len(references),
        "links": len(links),
        "with_doi": sum(1 for r in references.values() if r.doi),
        "with_pmid": sum(1 for r in references.values() if r.pmid),
        "activity_scoped": sum(1 for _, _, i in links if i is not None),
    }
    logger.info(f"Wrote {summary['references']} references to {output_path} "
                f"({summary['with_doi']} with DOI, {summary['with_pmid']} with PMID).")
    return summary
