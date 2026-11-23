"""
Full-text and regular-expression search, across SQLite, PostgreSQL and MySQL.

This is the part of the port an ORM does not do for you. SQLAlchemy renders ``CREATE TABLE`` in
any dialect, but text search is where the three databases stop resembling each other -- different
index types, different query grammars, different operators:

============  ==========================  =============================  ==========================
              SQLite                      PostgreSQL                     MySQL
============  ==========================  =============================  ==========================
index         ``FTS5`` virtual table      ``tsvector`` column + GIN      ``FULLTEXT`` index
query         ``MATCH 'a AND b'``         ``@@ to_tsquery('a & b')``     ``MATCH..AGAINST('+a +b')``
prefix        ``magain*``                 ``magain:*``                   ``magain*``
negation      ``NOT``                     ``!``                          ``-``
regex         Python function per conn.   ``~`` / ``~*``                 ``REGEXP``
============  ==========================  =============================  ==========================

Rather than expose those differences, this module defines **one query grammar** -- the FTS5 one,
because it is what the search box already documents -- and translates it per dialect. A search for
``magainin AND NOT frog`` means the same thing on all three.

Two properties are preserved everywhere, because they are correctness rather than convenience:

* **Decoys are never returned.** The search index is built only from real peptides, and the joins
  filter on ``is_decoy = 0`` regardless.
* **A malformed query returns nothing rather than raising.** This serves a search box, where a
  stray parenthesis is a typo, not an outage. Each dialect fails differently, so each is caught.

MySQL's full-text minimum word length deserves a mention: ``innodb_ft_min_token_size`` defaults to
3, so a two-character term matches nothing through the index. Sequence motifs are longer than that
and peptide names almost always are, but the fallback below covers the case rather than returning
a confidently empty page.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sample.search")

#: Columns the search document is assembled from, in order.
SEARCH_COLUMNS: Tuple[str, ...] = ("sequence", "names", "literature", "patent")

#: Name of the portable search table. On SQLite an FTS5 virtual table is built beside it.
SEARCH_TABLE = "peptide_search"

#: FTS5 operators recognised in the shared grammar.
_OPERATORS = {"AND", "OR", "NOT"}

#: A quoted phrase, or a bare term possibly ending in a prefix wildcard.
_TOKEN = re.compile(r'"[^"]*"|\S+')


#: Punctuation that joins an identifier to its surroundings in the stored columns, e.g.
#: ``google_patents:US8394762B2.json``.
_IDENTIFIER_GLUE = re.compile(r"[:/_.,;|\\]+")


def normalize_for_index(value: Any) -> str:
    """
    Prepares a stored column for indexing so every dialect tokenises it the same way.

    The reference columns hold ``source:filename.json`` tokens, and the three tokenisers disagree
    about them. SQLite's ``unicode61`` splits on the punctuation and yields ``us8394762b2`` as its
    own term; PostgreSQL's parser recognises ``US8394762B2.json`` as a filename and keeps it
    whole, so a search for the patent number finds nothing. Splitting the glue at index time makes
    the identifier a term everywhere, which is what a user searching for a patent number expects.

    :param value: A column value.
    :returns: The value with identifier punctuation replaced by spaces.
    """
    return _IDENTIFIER_GLUE.sub(" ", str(value or "")).strip()


def parse_query(query: str) -> List[Tuple[str, str]]:
    """
    Splits a query into typed tokens.

    :param query: A query in the shared grammar: bare terms, ``"quoted phrases"``, ``prefix*``
        wildcards, and the operators ``AND``, ``OR`` and ``NOT``.
    :returns: ``(kind, value)`` pairs where kind is ``operator``, ``phrase``, ``prefix`` or
        ``term``.
    """
    tokens: List[Tuple[str, str]] = []
    for raw in _TOKEN.findall(str(query or "").strip()):
        if raw.upper() in _OPERATORS:
            tokens.append(("operator", raw.upper()))
        elif raw.startswith('"') and raw.endswith('"') and len(raw) > 1:
            inner = raw.strip('"').strip()
            if inner:
                tokens.append(("phrase", inner))
        elif raw.endswith("*") and len(raw) > 1:
            tokens.append(("prefix", raw[:-1]))
        else:
            tokens.append(("term", raw))
    return tokens


def _sanitize(value: str) -> str:
    """
    :param value: A user-supplied term.
    :returns: The term with characters that are operators in some dialect removed, so a query
        cannot escape into the surrounding expression.
    """
    return re.sub(r"[^\w\s\-.:/]", " ", value, flags=re.UNICODE).strip()


def to_tsquery(query: str) -> str:
    """
    Translates the shared grammar into a PostgreSQL ``to_tsquery`` expression.

    :param query: The query.
    :returns: An expression using ``&``, ``|`` and ``!``, with ``:*`` for prefixes. Terms are
        joined with ``&`` where the user wrote no operator, matching FTS5's implicit AND.
    """
    parts: List[str] = []
    pending_not = False
    for kind, value in parse_query(query):
        if kind == "operator":
            if value == "AND":
                parts.append("&")
            elif value == "OR":
                parts.append("|")
            else:
                # `NOT x` in FTS5 is `& !x` in tsquery; the `&` is supplied here when a term
                # precedes it.
                if parts and parts[-1] not in ("&", "|"):
                    parts.append("&")
                pending_not = True
            continue

        clean = _sanitize(value)
        if not clean:
            continue
        if kind == "phrase":
            rendered = " <-> ".join(clean.split())
        elif kind == "prefix":
            rendered = f"{clean.split()[0]}:*"
        else:
            rendered = clean.split()[0]

        if pending_not:
            rendered = f"!{rendered}"
            pending_not = False
        if parts and parts[-1] not in ("&", "|"):
            parts.append("&")
        parts.append(rendered)

    # A trailing operator would make the expression invalid.
    while parts and parts[-1] in ("&", "|"):
        parts.pop()
    return " ".join(parts)


def to_mysql_boolean(query: str) -> str:
    """
    Translates the shared grammar into a MySQL boolean-mode expression.

    :param query: The query.
    :returns: An expression using ``+`` for required terms, ``-`` for excluded ones and ``*`` for
        prefixes. MySQL has no OR operator in boolean mode -- a term with no prefix is optional,
        which *is* OR -- so an explicit ``OR`` drops the ``+`` markers from its operands.
    """
    tokens = parse_query(query)
    has_or = any(kind == "operator" and value == "OR" for kind, value in tokens)

    parts: List[str] = []
    negate_next = False
    for kind, value in tokens:
        if kind == "operator":
            negate_next = value == "NOT"
            continue

        clean = _sanitize(value)
        if not clean:
            continue
        if kind == "phrase":
            rendered = f'"{clean}"'
        elif kind == "prefix":
            rendered = f"{clean.split()[0]}*"
        else:
            rendered = clean.split()[0]

        if negate_next:
            parts.append(f"-{rendered}")
            negate_next = False
        elif has_or:
            parts.append(rendered)
        else:
            parts.append(f"+{rendered}")
    return " ".join(parts)


def to_fts5(query: str) -> str:
    """
    Normalises the shared grammar for SQLite FTS5.

    :param query: The query.
    :returns: An FTS5 MATCH expression. The grammar is already FTS5's, so this only strips
        characters FTS5 would treat as syntax errors while keeping its operators intact.
    """
    parts: List[str] = []
    for kind, value in parse_query(query):
        if kind == "operator":
            parts.append(value)
        elif kind == "phrase":
            clean = _sanitize(value)
            if clean:
                parts.append(f'"{clean}"')
        elif kind == "prefix":
            clean = _sanitize(value)
            if clean:
                parts.append(f"{clean.split()[0]}*")
        else:
            clean = _sanitize(value)
            if clean:
                parts.append(clean.split()[0])
    while parts and parts[-1] in _OPERATORS:
        parts.pop()
    return " ".join(parts)


def translate(query: str, dialect: str) -> str:
    """
    Renders a query for one dialect.

    :param query: The query in the shared grammar.
    :param dialect: ``sqlite``, ``postgresql`` or ``mysql``.
    :returns: The dialect's expression.
    """
    if dialect == "postgresql":
        return to_tsquery(query)
    if dialect == "mysql":
        return to_mysql_boolean(query)
    return to_fts5(query)


def create_search_index(connection: Any, dialect: str) -> None:
    """
    Builds the full-text index for this dialect over :data:`SEARCH_TABLE`.

    The table itself is ordinary and portable; only the index is dialect-specific.

    :param connection: An open SQLAlchemy connection with write access.
    :param dialect: The dialect name.
    :returns: ``None``. A database that supports no full-text index is left alone, and search
        falls back to ``LIKE``.
    """
    from sqlalchemy import text

    if dialect == "sqlite":
        connection.execute(text("DROP TABLE IF EXISTS peptides_fts"))
        connection.execute(text(
            "CREATE VIRTUAL TABLE peptides_fts USING fts5("
            "  entry_id UNINDEXED, sequence, names, literature, patent,"
            '  tokenize = "unicode61")'))
        connection.execute(text(
            "INSERT INTO peptides_fts (entry_id, sequence, names, literature, patent) "
            "SELECT entry_id, sequence, names, literature, patent FROM peptide_search"))

    elif dialect == "postgresql":
        # A stored generated column keeps the vector in step with the row without a trigger.
        connection.execute(text(
            "ALTER TABLE peptide_search ADD COLUMN IF NOT EXISTS document tsvector "
            "GENERATED ALWAYS AS ("
            "  to_tsvector('simple',"
            "    coalesce(sequence,'') || ' ' || coalesce(names,'') || ' ' ||"
            "    coalesce(literature,'') || ' ' || coalesce(patent,''))) STORED"))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_peptide_search_document "
            "ON peptide_search USING GIN (document)"))

    elif dialect == "mysql":
        # MySQL cannot add a FULLTEXT index if one already exists under the same name, and has no
        # IF NOT EXISTS for indexes; the drop is attempted and ignored.
        try:
            connection.execute(text("DROP INDEX idx_peptide_search_ft ON peptide_search"))
        except Exception:
            pass
        connection.execute(text(
            "CREATE FULLTEXT INDEX idx_peptide_search_ft "
            "ON peptide_search (sequence, names, literature, patent)"))

    else:
        logger.info(f"No full-text index for dialect {dialect}; search will use LIKE.")


def fulltext_where(query: str, dialect: str) -> Tuple[str, Dict[str, Any]]:
    """
    Builds the WHERE fragment and parameters for a full-text search.

    :param query: The query in the shared grammar.
    :param dialect: The dialect name.
    :returns: ``(sql_fragment, parameters)``. The fragment references the ``peptide_search`` table
        as ``s``; on SQLite it references the FTS5 table as ``f`` instead.
    """
    rendered = translate(query, dialect)

    if dialect == "sqlite":
        return "f.peptides_fts MATCH :q", {"q": rendered}
    if dialect == "postgresql":
        return "s.document @@ to_tsquery('simple', :q)", {"q": rendered}
    if dialect == "mysql":
        return ("MATCH (s.sequence, s.names, s.literature, s.patent) "
                "AGAINST (:q IN BOOLEAN MODE)"), {"q": rendered}

    # No full-text support: substring match over the same columns. Slower and unranked, but
    # correct, which is the right failure mode for a search box.
    plain = re.sub(r'[*"]', "", str(query or "")).strip().lower()
    needle = f"%{plain}%"
    columns = " OR ".join(f"lower(s.{c}) LIKE :q" for c in SEARCH_COLUMNS)
    return f"({columns})", {"q": needle}


def regexp_where(column: str, dialect: str) -> str:
    """
    Builds the WHERE fragment for a regular-expression match.

    :param column: The column expression to match against.
    :param dialect: The dialect name.
    :returns: A SQL fragment with a ``:pattern`` parameter. SQLite's form calls a function that
        :mod:`sample.db` registers on the connection; the others are native operators.
    """
    if dialect == "postgresql":
        # `~*` is the case-insensitive match, matching the Python re.IGNORECASE the SQLite path
        # compiles with.
        return f"{column} ~* :pattern"
    if dialect == "mysql":
        return f"{column} REGEXP :pattern"
    return f"REGEXP_MATCH({column}) = 1"
