"""
Read access to the released SQLite database.

Every read of ``data/sample.db`` goes through this module. That is not organisational tidiness --
it exists to enforce one invariant that cannot be enforced by convention:

**Decoys share a table with real peptides.** ``peptides`` holds 35,411 real entries and 35,411
generated decoys (shuffled sequences and UniProt fragments) distinguished only by ``is_decoy``.
They are training negatives, not peptides; publishing one as an antimicrobial peptide would be
presenting a fabricated sequence as a measured result. The previous Flask query layer had no
``is_decoy`` filter at all, and searching for a decoy's own sequence returned it as a normal hit.

So there is exactly one gate: every query here filters ``is_decoy = 0`` unless the caller passes
``include_decoys=True`` explicitly. No route may issue raw SQL against ``peptides``.

Opened read-only. This process never writes the release database -- step 44 builds it -- and a
read-only handle makes that structural rather than a matter of care.
"""
import functools
import json
import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger("sample.db")

#: Columns returned for a peptide in list and detail views. Explicit rather than ``SELECT *`` so a
#: new column added to the table cannot silently appear in the public API.
PEPTIDE_COLUMNS: Tuple[str, ...] = (
    # `Reference` is selected because it is the only column holding an actual citation. It was
    # left out while `_with_reference` synthesised the field from LITERATURE and PATENT, so the
    # stored value could never reach a page even once the merge began resolving each extraction
    # to its title, DOI and PMID.
    "entry_id", "sequence", "SMILES", "source", "is_decoy", "LITERATURE", "PATENT", "Reference",
    "source_organism", "cross_references",
)

_MEMORY_CONNECTION: Optional[sqlite3.Connection] = None

#: Columns a regular expression may be matched against. An allow-list rather than a free column
#: name, because the field is interpolated into SQL: a caller-supplied column name would be an
#: injection point that no amount of parameter binding could close.
REGEX_FIELDS: Dict[str, str] = {
    "sequence": "p.sequence",
    "entry_id": "p.entry_id",
    "smiles": "p.SMILES",
}

#: Longest accepted pattern. Long patterns are not more expressive here -- a motif over a 20-residue
#: median sequence is short -- but they are the raw material for a pathological one.
REGEX_MAX_PATTERN: int = 200

#: Wall-clock budget for one regex scan, in seconds.
REGEX_TIME_BUDGET: float = 5.0

#: Ceiling on matches collected in a single scan, so a pattern like ``.`` cannot materialise every
#: row. Beyond this the result is reported as truncated rather than silently cut.
REGEX_MAX_MATCHES: int = 10_000


class InvalidPattern(ValueError):
    """A regular expression the caller wrote incorrectly."""


class SearchTimeout(RuntimeError):
    """A regex scan that exceeded :data:`REGEX_TIME_BUDGET`."""


def default_db_path() -> str:
    """
    :returns: Location of the released database. ``SAMPLE_DB`` overrides it, which is how the
        tests point at a fixture database.
    """
    override = os.environ.get("SAMPLE_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "sample.db")


def _in_memory_requested() -> bool:
    """
    :returns: Whether ``SAMPLE_DB_IN_MEMORY`` asks for the database to be held in RAM.
    """
    return os.environ.get("SAMPLE_DB_IN_MEMORY", "").lower() in ("1", "true", "yes")


def get_connection(db_path: Optional[str] = None,
                   shared: bool = True) -> sqlite3.Connection:
    """
    Opens a read-only connection, or returns the shared in-memory copy.

    In-memory mode is opt-in because each process that enables it holds a private copy of the
    whole database (~111 MB). That is affordable for one web process and wasteful across a pool of
    Celery workers, so it is a deployment decision rather than a default.

    :param db_path: Database file; :func:`default_db_path` when omitted.
    :param shared: When ``False``, always open a private connection even in in-memory mode. Used
        by :func:`regex_search`, which registers a per-query function on the handle and would
        otherwise race with a concurrent request on the shared one.
    :returns: A connection with ``row_factory`` set to :class:`sqlite3.Row`.
    :raises FileNotFoundError: When the database has not been built. Step 44 creates it.
    """
    global _MEMORY_CONNECTION

    path = db_path or default_db_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Released database not found at {path}. Run step 46 (`sample-cli load-sql`).")

    if shared and _in_memory_requested() and db_path is None:
        if _MEMORY_CONNECTION is None:
            logger.info(f"Loading {path} into memory (SAMPLE_DB_IN_MEMORY is set)...")
            source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            _MEMORY_CONNECTION = sqlite3.connect(":memory:", check_same_thread=False)
            source.backup(_MEMORY_CONNECTION)
            source.close()
            _MEMORY_CONNECTION.row_factory = sqlite3.Row
            logger.info("Database resident in memory.")
        return _MEMORY_CONNECTION

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _close(conn: sqlite3.Connection) -> None:
    """
    Closes a per-request connection, leaving the shared in-memory one open.

    :param conn: Connection returned by :func:`get_connection`.
    :returns: ``None``.
    """
    if conn is not _MEMORY_CONNECTION:
        conn.close()


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    """
    :param conn: Open connection.
    :param name: Table name.
    :returns: Whether the table exists. Tables added by later pipeline steps -- names, families,
        bibliography, bioactivity -- are optional, and a database built before one of them must
        still serve every query that does not need it.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1",
        (name,)).fetchone()
    return row is not None


def _decoy_clause(include_decoys: bool, alias: str = "p") -> str:
    """
    :param include_decoys: When ``False``, restrict to real peptides.
    :param alias: Table alias used in the query.
    :returns: A SQL fragment to append to a ``WHERE`` clause. ``COALESCE`` because the column is
        nullable and a NULL must read as "not a decoy", not as "unknown, so exclude".
    """
    return "" if include_decoys else f" AND COALESCE({alias}.is_decoy, 0) = 0"


def search_peptides(query_str: str = "", activity_group: str = "", target: str = "",
                    mechanism: str = "", metric: str = "", unit: str = "",
                    limit: int = 100, offset: int = 0, include_decoys: bool = False,
                    db_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Structured search over peptides and their activities.

    Ported from the Flask ``run_db_search``, with two changes: decoys are excluded (see the module
    docstring), and a total count is returned so a client can page without guessing.

    :param query_str: Free text matched against entry id, sequence, SMILES and paper identifiers.
    :param activity_group: Restrict to one activity class, e.g. ``"antibacterial"``.
    :param target: Substring match on target organism.
    :param mechanism: Substring match on mechanism of action.
    :param metric: Restrict to one assay type, e.g. ``"MIC"``.
    :param unit: Restrict to one canonical unit.
    :param limit: Maximum rows returned.
    :param offset: Rows to skip, for pagination.
    :param include_decoys: Opt in to training negatives. Never set this from a public route.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``(rows, total_matching)``.
    """
    where = "WHERE 1=1" + _decoy_clause(include_decoys)
    params: List[Any] = []
    conn = get_connection(db_path)

    if query_str:
        needle = f"%{query_str.lower()}%"
        # Names and synonyms are matched through a subquery rather than another join, so a peptide
        # with twelve recorded names does not multiply into twelve rows of the result. Before this
        # clause existed, searching "magainin" returned nothing while magainin's sequence sat in
        # the table -- the release stored no names at all.
        #
        # Conditional because a database built before the names step has no such table, and search
        # is too central to fail wholesale over an optional one: without the check, every query
        # raised "no such table: peptide_names".
        name_clause = ""
        if _has_table(conn, "peptide_names"):
            name_clause = (" OR EXISTS (SELECT 1 FROM peptide_names pn"
                           "            WHERE pn.entry_id = p.entry_id AND lower(pn.name) LIKE ?)")
        where += (" AND (lower(p.entry_id) LIKE ? OR lower(p.sequence) LIKE ?"
                  " OR lower(p.SMILES) LIKE ? OR lower(pap.source_id) LIKE ?"
                  + name_clause + ")")
        params.extend([needle] * (5 if name_clause else 4))
    if activity_group:
        ag_norm = activity_group.strip().lower()
        if ag_norm in ("anti-inflammatory", "antinflammatory"):
            where += " AND lower(a.activity_group) IN ('anti-inflammatory', 'antinflammatory')"
        else:
            where += " AND lower(a.activity_group) = ?"
            params.append(ag_norm)
    if target:
        where += " AND lower(a.target_organism) LIKE ?"
        params.append(f"%{target.lower()}%")
    if mechanism:
        where += " AND lower(a.mechanism_of_action) LIKE ?"
        params.append(f"%{mechanism.lower()}%")
    if metric:
        where += " AND lower(a.activity_type) = ?"
        params.append(metric.lower())
    if unit:
        u_norm = unit.strip().lower()
        if u_norm in ("µm", "um", "umol/l", "μm"):
            where += " AND (lower(a.unit) IN ('um', 'umol/l') OR a.unit IN ('µM', 'μM', 'uM', 'umol/L'))"
        elif u_norm in ("ug/ml", "µg/ml", "μg/ml", "mcg/ml"):
            where += " AND (lower(a.unit) IN ('ug/ml', 'mcg/ml') OR a.unit IN ('µg/ml', 'μg/ml', 'ug/mL', 'µg/mL'))"
        elif u_norm in ("%", "percentage", "percent"):
            where += " AND (lower(a.unit) IN ('%', 'percentage', 'percent') OR a.unit = '%')"
        else:
            where += " AND lower(a.unit) = ?"
            params.append(u_norm)

    joins = """
        FROM peptides p
        LEFT JOIN activities a ON p.entry_id = a.entry_id
        LEFT JOIN peptide_references pr ON p.entry_id = pr.peptide_id
        LEFT JOIN papers pap ON pr.paper_id = pap.id
    """
    selected = _peptide_select(conn)

    try:
        total = conn.execute(
            f"SELECT COUNT(DISTINCT p.entry_id) {joins} {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT DISTINCT {selected} {joins} {where} ORDER BY p.entry_id ASC LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)]).fetchall()
    finally:
        _close(conn)

    return [_with_reference(dict(r)) for r in rows], int(total)


def full_text_search(query_str: str, limit: int = 100, offset: int = 0,
                     db_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Full-text search over the FTS5 index built by step 44.

    Supports the FTS5 query grammar directly, which is what gives the search box boolean operators
    (``AND``/``OR``/``NOT``), prefix wildcards (``magain*``) and phrase queries for free. The index
    contains only real peptides, so no decoy filter is needed here.

    :param query_str: An FTS5 MATCH expression.
    :param limit: Maximum rows returned.
    :param offset: Rows to skip.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``(rows, total_matching)``; an empty result for a malformed query rather than a
        raised error, because the query comes from a user-facing search box.
    """
    conn = get_connection(db_path)
    selected = _peptide_select(conn)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM peptides_fts WHERE peptides_fts MATCH ?",
            (query_str,)).fetchone()[0]
        rows = conn.execute(
            f"SELECT {selected} FROM peptides_fts f JOIN peptides p ON p.entry_id = f.entry_id "
            "WHERE peptides_fts MATCH ? ORDER BY rank LIMIT ? OFFSET ?",
            (query_str, int(limit), int(offset))).fetchall()
    except sqlite3.OperationalError as e:
        logger.info(f"Rejected full-text query {query_str!r}: {e}")
        return [], 0
    finally:
        _close(conn)

    return [_with_reference(dict(r)) for r in rows], int(total)


def _compile_motif(pattern: str) -> Pattern[str]:
    """
    Compiles a user-supplied motif pattern.

    :param pattern: A Python regular expression, e.g. ``^GIG.{2}[FL]`` for a magainin-like motif.
    :returns: The compiled pattern, case-insensitive so a lowercase motif matches stored sequences.
    :raises InvalidPattern: When the pattern is empty, over :data:`REGEX_MAX_PATTERN`, or does not
        compile. Unlike :func:`full_text_search`, this reports the error instead of returning an
        empty page: an FTS5 expression is usually malformed by accident (a bare parenthesis in a
        peptide name is FTS syntax), whereas a regex is written deliberately, so ``missing ),
        unterminated subpattern`` tells the author what to fix.
    """
    text = (pattern or "").strip()
    if not text:
        raise InvalidPattern("Empty pattern.")
    if len(text) > REGEX_MAX_PATTERN:
        raise InvalidPattern(
            f"Pattern longer than {REGEX_MAX_PATTERN} characters.")
    try:
        return re.compile(text, re.IGNORECASE)
    except re.error as e:
        raise InvalidPattern(f"Invalid regular expression: {e}") from e


def regex_search(pattern: str, field: str = "sequence", limit: int = 100, offset: int = 0,
                 include_decoys: bool = False, time_budget: Optional[float] = None,
                 db_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Motif search: matches a regular expression against one peptide column.

    This is the one search mode FTS5 cannot serve. FTS5 tokenises, so it can find whole terms and
    prefixes but not ``^GIG.{2}[FL]`` -- and a sequence motif is exactly the query this dataset
    exists to answer. SQLite has no built-in ``REGEXP``, so one is registered on the connection for
    the duration of the scan.

    A user-supplied regex over 35,411 rows is a denial-of-service surface: a pattern with nested
    quantifiers can backtrack for an unbounded time. Three bounds contain it -- pattern length,
    a wall-clock budget checked between rows, and a cap on matches collected.

    :param pattern: The regular expression.
    :param field: Column to match; one of :data:`REGEX_FIELDS`.
    :param limit: Maximum rows returned.
    :param offset: Rows to skip.
    :param include_decoys: Opt in to training negatives. Never set this from a public route.
    :param time_budget: Seconds before the scan gives up; :data:`REGEX_TIME_BUDGET` when omitted.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``(rows, total_matching)``. The total is exact up to :data:`REGEX_MAX_MATCHES`.
    :raises InvalidPattern: When the pattern does not compile.
    :raises SearchTimeout: When the scan exceeds its budget. Reported rather than returned as an
        empty page, so a caller can tell "no such motif" from "gave up looking".
    """
    compiled = _compile_motif(pattern)
    column = REGEX_FIELDS.get(field.lower())
    if column is None:
        raise InvalidPattern(
            f"Cannot match a pattern against {field!r}. Choose one of "
            f"{', '.join(sorted(REGEX_FIELDS))}.")

    deadline = time.monotonic() + (
        REGEX_TIME_BUDGET if time_budget is None else float(time_budget))

    # sqlite3 replaces an exception raised inside a user function with a generic
    # ``OperationalError: user-defined function raised exception``, discarding the type and
    # message. So the timeout records itself here on the way out.
    expired: List[bool] = []

    def _matches(value: Optional[str]) -> int:
        """Registered as the SQL function ``REGEXP_MATCH``."""
        if time.monotonic() > deadline:
            expired.append(True)
            raise SearchTimeout(
                f"Motif search exceeded its time budget. Pattern {pattern!r} is too expensive.")
        return 1 if value and compiled.search(value) else 0

    # A private handle: the function below closes over this call's pattern, and registering it on
    # the process-wide in-memory connection would let two concurrent searches overwrite each
    # other's pattern mid-scan.
    conn = get_connection(db_path, shared=False)
    conn.create_function("REGEXP_MATCH", 1, _matches)
    selected = _peptide_select(conn)
    try:
        rows = conn.execute(
            f"SELECT {selected} FROM peptides p WHERE REGEXP_MATCH({column}) = 1"
            + _decoy_clause(include_decoys)
            + " ORDER BY p.entry_id ASC LIMIT ?", (REGEX_MAX_MATCHES,)).fetchall()
    except sqlite3.OperationalError as e:
        if expired:
            raise SearchTimeout(
                f"Motif search for {pattern!r} exceeded its time budget of "
                f"{REGEX_TIME_BUDGET if time_budget is None else time_budget:g}s.") from e
        raise
    finally:
        conn.close()

    page = [_with_reference(dict(r)) for r in rows[offset:offset + int(limit)]]
    return page, len(rows)


def _with_taxonomy_link(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds the NCBI Taxonomy and ATCC product URLs for an activity's target organism.

    The ATCC link is not from a stored accession -- nothing in the schema has one -- but read back
    out of ``target_organism`` itself: DBAASP's raw ``targetSpecies.name`` ("Staphylococcus aureus
    ATCC 29213") is stored as-is in that column (see ``sample/serialize.py``'s DBAASP activity
    block), and 34.4% of released activities carry a catalog number that way. Computed at read
    time rather than as a separate stored column, since the source string is already there and a
    second copy of the same fact would just be one more place for the two to disagree.

    :param row: An activity row carrying ``target_taxid`` and ``target_organism``.
    :returns: The same row with ``target_taxonomy_url`` and ``target_atcc_url`` set (either may be
        ``None``).
    """
    from sample.external_links import build_url
    from sample.nlp import extract_atcc_id

    taxid = row.get("target_taxid")
    row["target_taxonomy_url"] = build_url("taxonomy", taxid) if taxid else None

    atcc_id = extract_atcc_id(row.get("target_organism"))
    row["target_atcc_url"] = build_url("atcc", atcc_id) if atcc_id else None
    return row


#: An entry that names an internal file rather than a paper: either the merge's ``File: <uuid>``
#: prefix, or a ``source:filename.json`` extraction token such as
#: ``pubmed_old:1c273b8f-….json``. Both were being shown to readers under the heading "Literature
#: & Patent References"; neither is a citation, and the second names a directory the extraction
#: browser does not even serve.
_INTERNAL_REFERENCE = re.compile(r"^(?:File:\s|\S+:\S+\.json\s*$)")


def is_internal_reference(text: Any) -> bool:
    """
    Whether a reference entry names an internal file rather than a publication.

    :param text: One entry from a pipe-delimited reference cell.
    :returns: ``True`` when it should not be shown as a citation.
    """
    return bool(_INTERNAL_REFERENCE.match(str(text or "").strip()))


def _peptide_select(conn: sqlite3.Connection, alias: str = "p") -> str:
    """
    The peptide column list, restricted to columns the database actually has.

    A release built by an older loader has no ``Reference`` column, and naming it unconditionally
    made every search fail with ``no such column: p.Reference`` -- the same way the name subquery
    once broke search against a database built before the names step. Columns are optional in
    practice, so the select list is built from what is there.

    :param conn: Open connection.
    :param alias: Table alias to qualify each column with. Empty for an unqualified list, which
        :func:`get_peptide` needs because it reads the row back by bare column name.
    :returns: A comma-separated select list.
    """
    try:
        present = {row[1] for row in conn.execute("PRAGMA table_info(peptides)")}
    except sqlite3.OperationalError:
        present = set(PEPTIDE_COLUMNS)
    columns = [c for c in PEPTIDE_COLUMNS if c in present] or list(PEPTIDE_COLUMNS)
    prefix = f"{alias}." if alias else ""
    return ", ".join(f"{prefix}{c}" for c in columns)


def _with_reference(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalises the ``Reference`` field the templates display.

    This used to *overwrite* the stored column with ``LITERATURE`` and ``PATENT`` joined together,
    which is why the peptide page listed ``pubmed_old:<uuid>.json`` under "Literature & Patent
    References" and the search results showed the same under "Reference Source". Those columns
    hold extraction filenames -- provenance for the Source Documents links, never a citation.

    The stored column is kept instead, minus any entry that is itself just a filename. Callers
    that need the extraction tokens read ``LITERATURE``/``PATENT`` directly.

    :param row: A peptide row.
    :returns: The same row, with ``Reference`` holding only real citations, or ``None``.
    """
    stored = str(row.get("Reference") or "")
    citations = [part.strip() for part in stored.split(" | ")
                 if part.strip() and not is_internal_reference(part)]
    row["Reference"] = " | ".join(citations) if citations else None
    return row


def get_peptide(entry_id: str, include_decoys: bool = False,
                db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Loads one peptide with its descriptors, activities and papers.

    :param entry_id: Peptide identifier.
    :param include_decoys: Opt in to fetching a training negative by id.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The peptide with ``descriptors``, ``activities`` and ``papers`` attached, or ``None``
        when no such (real) peptide exists.
    """
    conn = get_connection(db_path)
    try:
        # Unqualified (no `p.` prefix) because the row is read back by bare column name below.
        # Was built straight from PEPTIDE_COLUMNS rather than through _peptide_select, which is
        # the one site that missed the degrade-on-an-older-schema fix and broke with
        # `no such column: source_organism` against any database predating this column.
        selected = _peptide_select(conn, alias="")
        row = conn.execute(
            f"SELECT {selected} FROM peptides p WHERE entry_id = ?"
            + _decoy_clause(include_decoys), (entry_id,)).fetchone()
        if row is None:
            return None

        from sample.external_links import annotate_papers, links_from_reference_tokens

        peptide = _with_reference(dict(row))
        papers = [dict(r) for r in conn.execute(
            "SELECT pap.source_type, pap.source_id FROM papers pap "
            "JOIN peptide_references pr ON pap.id = pr.paper_id WHERE pr.peptide_id = ?",
            (entry_id,)).fetchall()]
        # Each reference gains a resolvable URL. Patent numbers additionally live inside the
        # PATENT filename tokens, which the papers table does not always carry.
        peptide["papers"] = annotate_papers(papers)
        known = {(p.get("source_type"), p.get("source_id")) for p in papers}
        peptide["papers"] += [link for link in links_from_reference_tokens(peptide.get("PATENT"))
                              if (link["source_type"], link["source_id"]) not in known]

        descriptor_row = conn.execute(
            "SELECT * FROM descriptors WHERE entry_id = ?", (entry_id,)).fetchone()
        peptide["descriptors"] = dict(descriptor_row) if descriptor_row else {}

        peptide["activities"] = [_with_taxonomy_link(dict(r)) for r in conn.execute(
            "SELECT * FROM activities WHERE entry_id = ?", (entry_id,)).fetchall()]

        peptide["bibliography"] = _bibliography_for(conn, entry_id)
        _attach_activity_citations(conn, peptide)
        peptide["family"] = _family_for(conn, entry_id)
        peptide["bioactivity"] = _bioactivity_for(conn, entry_id)
        peptide["names"] = _names_for(conn, entry_id)
        peptide["topology"] = _topology_for(conn, entry_id)
        peptide["modifications"] = _modifications_for(conn, entry_id)
        peptide["go_terms"] = _go_terms_for(conn, entry_id)
    finally:
        _close(conn)
    return peptide


def _names_for(conn: sqlite3.Connection, entry_id: str) -> List[Dict[str, Any]]:
    """
    Loads a peptide's names and synonyms.

    :param conn: Open connection.
    :param entry_id: Peptide identifier.
    :returns: Names with primaries first, then synonyms, each alphabetical. Empty when the name
        table has not been built.
    """
    try:
        rows = conn.execute(
            "SELECT name, name_kind, source_db FROM peptide_names WHERE entry_id = ? "
            "ORDER BY CASE name_kind WHEN 'primary' THEN 0 ELSE 1 END, name ASC",
            (entry_id,)).fetchall()
    except sqlite3.OperationalError:
        return []

    # Filtered on the way out as well as during the harvest, so a database built before the
    # harvest was fixed does not show its artifacts. dbAMP files the source organism, UniProt
    # entry names and fixed-width-truncated strings in the same column as the peptide name; one
    # peptide listed `South American frog Phyllomedusa bicolor`, `DMS2_PHYBI`,
    # `Adenoregulin (Dermasepti` and `Adenoregulindata about MIC or HC50 is present` as names.
    try:
        from sample.names import looks_like_junk_name
    except Exception:
        return [dict(r) for r in rows]

    seen = set()
    names = []
    for row in rows:
        name = str(row["name"] or "").strip()
        key = name.lower()
        if not name or key in seen or looks_like_junk_name(name):
            continue
        seen.add(key)
        names.append(dict(row))
    return names


def _modifications_for(conn: sqlite3.Connection, entry_id: str) -> List[Dict[str, Any]]:
    """
    Loads a peptide's chemical modifications.

    :param conn: Open connection.
    :param entry_id: Peptide identifier.
    :returns: Terminal modifications first (N before C), then residue modifications in position
        order. Empty when the table has not been built.
    """
    if not _has_table(conn, "peptide_modifications"):
        return []
    try:
        rows = conn.execute(
            "SELECT terminus, position, code, description, kind, source_db "
            "FROM peptide_modifications WHERE entry_id = ? "
            "ORDER BY CASE WHEN terminus IS NULL THEN 1 ELSE 0 END, terminus, position",
            (entry_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


def _go_terms_for(conn: sqlite3.Connection, entry_id: str) -> List[Dict[str, Any]]:
    """
    Loads a peptide's Gene Ontology annotations, harvested from CAMP.

    :param conn: Open connection.
    :param entry_id: Peptide identifier.
    :returns: Terms ordered by category (biological process, molecular function, cellular
        component, then uncategorised), then alphabetically. Some peptides carry over a hundred
        terms; grouping and capping for display is the template's job, not this query's, since
        the count itself is worth exposing.
    """
    if not _has_table(conn, "peptide_go_terms"):
        return []
    try:
        rows = conn.execute(
            "SELECT go_id, category, term, evidence FROM peptide_go_terms WHERE entry_id = ? "
            "ORDER BY CASE category "
            "  WHEN 'Biological process' THEN 0 WHEN 'Molecular function' THEN 1 "
            "  WHEN 'Cellular component' THEN 2 ELSE 3 END, term",
            (entry_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


def _topology_for(conn: sqlite3.Connection, entry_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads a peptide's topology: whether its backbone is closed and how it is braced.

    :param conn: Open connection.
    :param entry_id: Peptide identifier.
    :returns: The topology record with its bond list decoded, or ``None`` when unknown. Unknown
        means no source stated it -- not that the peptide is linear.
    """
    try:
        row = conn.execute(
            "SELECT kind, is_cyclic, n_disulfides, mass_correction, bonds, source_db "
            "FROM peptide_topology WHERE entry_id = ?", (entry_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None

    record = dict(row)
    record["is_cyclic"] = bool(record.get("is_cyclic"))
    try:
        record["bonds"] = json.loads(record.get("bonds") or "[]")
    except (TypeError, ValueError):
        record["bonds"] = []
    return record


def _bioactivity_for(conn: sqlite3.Connection, entry_id: str) -> List[Dict[str, Any]]:
    """
    Loads assay-archive records linked to a peptide.

    :param conn: Open connection.
    :param entry_id: Peptide identifier.
    :returns: One entry per external identifier, with a record count and a resolving URL. Empty
        means this peptide was never resolved to an archive identifier -- not that it has no
        assays. See ``_link_peptides_to_bioactivity`` for why coverage is narrow.
    """
    try:
        rows = conn.execute(
            "SELECT pb.archive, pb.identifier, COALESCE(b.n_records, 0) AS n_records "
            "FROM peptide_bioactivity pb "
            "LEFT JOIN bioactivity b ON b.archive = pb.archive AND b.identifier = pb.identifier "
            "WHERE pb.entry_id = ? ORDER BY n_records DESC", (entry_id,)).fetchall()
    except sqlite3.OperationalError:
        return []

    from sample.external_links import build_url

    #: The archive key names a different namespace per source, so each resolves differently.
    namespace = {"chembl": "chembl", "pubchem": "pubchem", "bindingdb": "uniprot"}
    out = []
    for row in rows:
        record = dict(row)
        record["url"] = build_url(namespace.get(record["archive"], ""), record["identifier"])
        out.append(record)
    return out


def _family_for(conn: sqlite3.Connection, entry_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads a peptide's family, with a link out when it is a curated Pfam one.

    :param conn: Open connection.
    :param entry_id: Peptide identifier.
    :returns: The family record, or ``None`` when unassigned or the table has not been built.
    """
    try:
        row = conn.execute(
            "SELECT family_id, family_name, source, evalue, all_domains "
            "FROM peptide_families WHERE entry_id = ?", (entry_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None

    from sample.external_links import build_url

    family = dict(row)
    # Only a Pfam accession resolves anywhere; a cluster id is local to this release, and linking
    # it to Pfam would point at an unrelated entry.
    family["url"] = build_url("pfam", family["family_id"]) if family["source"] == "pfam" else None
    family["n_members"] = int(conn.execute(
        "SELECT COUNT(*) FROM peptide_families WHERE family_id = ?",
        (family["family_id"],)).fetchone()[0])
    return family


def list_families(source: str = "", limit: int = 100, offset: int = 0,
                  db_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Lists families with their member counts.

    :param source: Restrict to ``pfam`` (curated) or ``cluster`` (derived here).
    :param limit: Maximum rows returned.
    :param offset: Rows to skip.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``(rows, total_families)``.
    """
    where = "WHERE family_id IS NOT NULL"
    params: List[Any] = []
    if source:
        where += " AND source = ?"
        params.append(source.lower())

    conn = get_connection(db_path)
    try:
        total = conn.execute(
            f"SELECT COUNT(DISTINCT family_id) FROM peptide_families {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT family_id, family_name, source, COUNT(*) AS n_members "
            f"FROM peptide_families {where} GROUP BY family_id, family_name, source "
            f"ORDER BY n_members DESC, family_id ASC LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)]).fetchall()
    except sqlite3.OperationalError:
        return [], 0
    finally:
        _close(conn)

    from sample.external_links import build_url

    return [{**dict(r),
             "url": build_url("pfam", r["family_id"]) if r["source"] == "pfam" else None}
            for r in rows], int(total)


def family_members(family_id: str, limit: int = 100, offset: int = 0,
                   db_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Lists the peptides in one family.

    :param family_id: Pfam accession or SAMPLE cluster id.
    :param limit: Maximum rows returned.
    :param offset: Rows to skip.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``(rows, total_members)``. Decoys cannot appear: they were never assigned a family.
    """
    conn = get_connection(db_path)
    selected = _peptide_select(conn)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM peptide_families WHERE family_id = ?", (family_id,)).fetchone()[0]
        rows = conn.execute(
            f"SELECT {selected} FROM peptide_families f "
            f"JOIN peptides p ON p.entry_id = f.entry_id "
            f"WHERE f.family_id = ?" + _decoy_clause(False) +
            " ORDER BY p.entry_id ASC LIMIT ? OFFSET ?",
            (family_id, int(limit), int(offset))).fetchall()
    except sqlite3.OperationalError:
        return [], 0
    finally:
        _close(conn)
    return [_with_reference(dict(r)) for r in rows], int(total)


def _bibliography_for(conn: sqlite3.Connection, entry_id: str) -> List[Dict[str, Any]]:
    """
    Loads the formatted citations for one peptide.

    :param conn: Open connection.
    :param entry_id: Peptide identifier.
    :returns: Reference records ordered by year then first author, or ``[]`` when the bibliography
        has not been built. Missing tables are treated as "no citations" rather than an error, so
        a database built before ``sample-cli build-references`` still serves peptide pages.
    """
    try:
        rows = conn.execute(
            "SELECT b.id, b.pmid, b.doi, b.title, b.authors, b.journal, b.year, b.volume, "
            "       b.issue, b.pages, b.apa, b.url "
            "FROM bibliography b JOIN peptide_bibliography pb ON pb.reference_id = b.id "
            "WHERE pb.peptide_id = ? ORDER BY b.year DESC, b.authors ASC", (entry_id,)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(r) for r in rows]


def _attach_activity_citations(conn: sqlite3.Connection, peptide: Dict[str, Any]) -> None:
    """
    Attaches per-measurement citations to each activity.

    Only DBAASP states which paper supports which measurement, so most activities carry none and
    fall back to the peptide-level bibliography in the template. Presenting a peptide-level
    citation as though it backed a specific MIC would claim more than the source does.

    :param conn: Open connection.
    :param peptide: The peptide being assembled; its ``activities`` are modified in place.
    :returns: ``None``.
    """
    activities = peptide.get("activities") or []
    if not activities:
        return
    try:
        rows = conn.execute(
            "SELECT ar.activity_id, b.apa, b.url, b.pmid, b.doi "
            "FROM activity_references ar JOIN bibliography b ON b.id = ar.reference_id "
            "WHERE ar.activity_id IN (SELECT id FROM activities WHERE entry_id = ?)",
            (peptide.get("entry_id"),)).fetchall()
    except sqlite3.OperationalError:
        rows = []

    by_activity: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        by_activity.setdefault(row["activity_id"], []).append(
            {k: row[k] for k in ("apa", "url", "pmid", "doi")})

    for activity in activities:
        activity["references"] = by_activity.get(activity.get("id"), [])
        activity["origin"] = _activity_origin(activity)
        activity["origin_url"] = _origin_url(activity["origin"], activity.get("source_accession"))


#: Maps the marker `serialize` writes into `mechanism_of_action` to the database it came from.
#: That column doubles as a provenance flag for third-party rows, which is why the lookup is by
#: prefix rather than an exact value -- a real mechanism of action never starts this way.
_ORIGIN_MARKERS = (
    ("DBAASP database record", "DBAASP"),
    ("CAMP database record", "CAMP"),
    ("DRAMP database record", "DRAMP"),
    ("ADP database record", "ADP"),
)


def _activity_origin(activity: Dict[str, Any]) -> Optional[str]:
    """
    Names where a measurement came from, for rows with no paper of their own.

    Most activities carry no measurement-level citation: only DBAASP records which article backs
    which measurement. Showing those rows an em-dash said nothing, when the database they were
    copied from is known and is exactly the provenance a reader wants.

    :param activity: One activity row.
    :returns: The source database, ``"SAMPLE extraction"`` for rows this project read out of a
        paper itself, or ``None`` when neither can be determined.
    """
    marker = str(activity.get("mechanism_of_action") or "")
    for prefix, name in _ORIGIN_MARKERS:
        if marker.startswith(prefix):
            return name
    # A row with no marker and no source accession was extracted here rather than copied.
    if not activity.get("source_accession"):
        return "SAMPLE extraction"
    return None


def _origin_url(origin: Optional[str], source_accession: Any) -> Optional[str]:
    """
    Resolves an activity's ``origin`` (e.g. ``"DBAASP"``) to a clickable link.

    ``source_accession`` is only populated for DBAASP today (the id its own ``dbaaspId`` field
    carries -- see ``sample/serialize.py``'s DBAASP activity block), which resolves to the exact
    peptide card via :func:`sample.external_links.build_url`. A row with no accession (CAMP, or
    ``"SAMPLE extraction"``) still gets the source database's home page rather than an inert badge
    -- reaching the right database is more useful than reaching nothing, the same tradeoff
    :data:`sample.external_links.HOME_PAGES` already makes for an unresolvable citation.

    :param origin: :func:`_activity_origin`'s result for this row.
    :param source_accession: The row's ``source_accession`` column, if any.
    :returns: A URL, or ``None`` when ``origin`` names no known external database (e.g. ``"SAMPLE
        extraction"``, which is this project's own pipeline, not an external source).
    """
    if not origin:
        return None
    from sample.external_links import HOME_PAGES, build_url

    key = origin.strip().lower()
    if source_accession:
        url = build_url(key, source_accession)
        if url:
            return url
    return HOME_PAGES.get(key)


def list_activities(target: str = "", activity_group: str = "", metric: str = "",
                    limit: int = 100, offset: int = 0,
                    db_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Paginated access to the activity table, joined to real peptides only.

    :param target: Substring match on target organism.
    :param activity_group: Restrict to one activity class.
    :param metric: Restrict to one assay type.
    :param limit: Maximum rows returned.
    :param offset: Rows to skip.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``(rows, total_matching)``.
    """
    where = "WHERE COALESCE(p.is_decoy, 0) = 0"
    params: List[Any] = []
    if target:
        where += " AND lower(a.target_organism) LIKE ?"
        params.append(f"%{target.lower()}%")
    if activity_group:
        where += " AND lower(a.activity_group) = ?"
        params.append(activity_group.lower())
    if metric:
        where += " AND lower(a.activity_type) = ?"
        params.append(metric.lower())

    joins = "FROM activities a JOIN peptides p ON p.entry_id = a.entry_id"
    conn = get_connection(db_path)
    try:
        total = conn.execute(f"SELECT COUNT(*) {joins} {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT a.* {joins} {where} ORDER BY a.id ASC LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)]).fetchall()
    finally:
        _close(conn)
    return [_with_taxonomy_link(dict(r)) for r in rows], int(total)


@functools.lru_cache(maxsize=1)
def get_stats(db_path: Optional[str] = None) -> Dict[str, int]:
    """
    Dataset counts for the dashboard.

    Read from SQLite and cached for the process. The Flask version re-parsed three CSVs with
    pandas on every request to ``/``, which cost hundreds of milliseconds per page view for
    numbers that only change when the pipeline re-runs.

    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: Counts keyed for the dashboard template.
    """
    conn = get_connection(db_path)
    try:
        def _count(sql: str) -> int:
            try:
                return int(conn.execute(sql).fetchone()[0])
            except sqlite3.Error:
                return 0

        return {
            "consensus_count": _count(
                "SELECT COUNT(*) FROM peptides WHERE COALESCE(is_decoy, 0) = 0"),
            "decoys_count": _count("SELECT COUNT(*) FROM peptides WHERE is_decoy = 1"),
            "final_count": _count("SELECT COUNT(*) FROM peptides"),
            # Reports what the database actually holds. This used to be
            # max(COUNT(*), 225858) -- a hardcoded floor equal to the row count of
            # data/sample_raw_release/activities.csv, which is a DIFFERENT quantity: that CSV is
            # step 21's flat export of every merged activity row, while this table is built from
            # the activity lists carried on the peptides that survived into final_dataset.csv.
            # The two legitimately differ, but the floor made the dashboard publish the CSV's
            # number while serving the database's rows -- 225,858 claimed against 204,370 present,
            # overstating by 21,488 (10.5%) and, being a max(), guaranteed to keep showing the
            # stale figure for as long as the real count stayed below it.
            "activities_count": _count("SELECT COUNT(*) FROM activities"),
            "papers_count": _count("SELECT COUNT(*) FROM papers"),
        }
    finally:
        _close(conn)
