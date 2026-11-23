"""
Database engine and the dialect differences that an ORM does not erase.

The schema is declared once in :mod:`sample.models` and rendered by SQLAlchemy into whichever
dialect the engine speaks, so moving to PostgreSQL or MySQL is a connection-string change::

    export SAMPLE_DATABASE_URL=postgresql+psycopg://user:pass@host/sample
    export SAMPLE_DATABASE_URL=mysql+pymysql://user:pass@host/sample

Three things do not translate themselves, and are the real content of a port:

**Full-text search.** FTS5 is a SQLite virtual table. PostgreSQL does the same job with
``tsvector`` and a GIN index; MySQL with a ``FULLTEXT`` index. The query syntax differs too --
FTS5 has its own ``MATCH`` grammar, PostgreSQL has ``to_tsquery``, MySQL has ``MATCH ... AGAINST``.
:func:`fulltext_support` reports which is available so the search layer can pick, and degrade to
``LIKE`` rather than fail.

**Regular expressions.** SQLite ships no ``REGEXP``; it must be registered per connection as a
Python function, which is why :mod:`sample.db` opens a private connection for motif search.
PostgreSQL has the ``~`` operator and MySQL ``REGEXP``, both server-side.

**Upserts.** ``INSERT OR IGNORE`` and ``INSERT OR REPLACE`` are SQLite spellings. PostgreSQL wants
``ON CONFLICT``, MySQL ``INSERT IGNORE``. :func:`insert_ignore` builds the right one.

Read-only by default. This process never writes the release database -- the pipeline builds it --
and on SQLite that is enforced at the connection level rather than by convention.
"""
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("sample.engine")

#: Environment variable holding a SQLAlchemy URL. Unset means the packaged SQLite file.
URL_VARIABLE = "SAMPLE_DATABASE_URL"

_ENGINES: Dict[str, Any] = {}


def _configured_url() -> str:
    """
    Reads the configured URL, loading ``.env`` if it has not been already.

    The URL carries the driver as well as the server -- ``postgresql+psycopg://``,
    ``mysql+pymysql://`` -- so choosing a database and choosing how to talk to it are one setting
    in one place. ``.env`` is loaded here rather than relied upon, because the web and worker
    processes do not all go through the CLI, which is the only entry point that loaded it before.

    :returns: The configured URL, or ``""`` when unset.
    """
    value = os.environ.get(URL_VARIABLE, "").strip()
    if value:
        return value

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return ""
    return os.environ.get(URL_VARIABLE, "").strip()


def default_sqlite_path() -> str:
    """
    :returns: Location of the released SQLite database. ``SAMPLE_DB`` overrides it, which is how
        the tests point at a fixture.
    """
    override = os.environ.get("SAMPLE_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "sample.db")


def database_url(read_only: bool = True, path: Optional[str] = None) -> str:
    """
    Builds the connection URL.

    :param read_only: Open SQLite in read-only mode. Ignored for server databases, where access is
        a grant on the account rather than a property of the connection.
    :param path: SQLite file to use; :func:`default_sqlite_path` when omitted.
    :returns: A SQLAlchemy URL.
    """
    configured = _configured_url()
    if configured:
        return configured

    location = path or default_sqlite_path()
    if read_only:
        return f"sqlite+pysqlite:///file:{location}?mode=ro&uri=true"
    return f"sqlite+pysqlite:///{location}"


def get_engine(read_only: bool = True, path: Optional[str] = None, echo: bool = False) -> Any:
    """
    Returns a cached engine for the configured database.

    :param read_only: See :func:`database_url`.
    :param path: SQLite file to use.
    :param echo: Log every statement, for debugging.
    :returns: A SQLAlchemy :class:`~sqlalchemy.engine.Engine`.
    """
    from sqlalchemy import create_engine

    url = database_url(read_only=read_only, path=path)
    key = f"{url}|{echo}"
    if key not in _ENGINES:
        kwargs: Dict[str, Any] = {"echo": echo, "future": True}
        if url.startswith("sqlite"):
            # SQLite's default thread check rejects a connection shared across the worker threads
            # a web server uses; the pool below serialises access instead.
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs["pool_pre_ping"] = True
        _ENGINES[key] = create_engine(url, **kwargs)
    return _ENGINES[key]


def dispose_engines() -> None:
    """
    Closes and forgets every cached engine. Used by tests that swap databases.

    A disposal failure does not stop the others, or leave a broken engine cached under its old
    key -- ``_ENGINES.clear()`` runs regardless -- but it is logged rather than swallowed, since it
    would otherwise be indistinguishable from a clean shutdown.

    :returns: ``None``.
    """
    for url, engine in _ENGINES.items():
        try:
            engine.dispose()
        except Exception as e:
            logger.warning(f"Failed to dispose engine for {url!r}: {e}")
    _ENGINES.clear()


def dialect_name(engine: Optional[Any] = None) -> str:
    """
    :param engine: Engine to inspect; the default one when omitted.
    :returns: ``sqlite``, ``postgresql``, ``mysql`` or whatever the driver reports.
    """
    return (engine or get_engine()).dialect.name


def fulltext_support(engine: Optional[Any] = None) -> str:
    """
    Reports how full-text search is available on this database.

    :param engine: Engine to inspect; the default one when omitted.
    :returns: ``fts5``, ``tsvector``, ``mysql_fulltext`` or ``like``. ``like`` means no index-backed
        full text is available and the caller should fall back to substring matching -- slower and
        without ranking, but correct, which is the right failure mode for a search box.
    """
    name = dialect_name(engine)
    if name == "sqlite":
        return "fts5"
    if name == "postgresql":
        return "tsvector"
    if name == "mysql":
        return "mysql_fulltext"
    return "like"


def regexp_support(engine: Optional[Any] = None) -> str:
    """
    Reports how regular-expression matching is available.

    :param engine: Engine to inspect; the default one when omitted.
    :returns: ``python_udf`` on SQLite, where the function must be registered per connection;
        ``native`` on PostgreSQL and MySQL, which evaluate it server-side.
    """
    return "python_udf" if dialect_name(engine) == "sqlite" else "native"


def insert_ignore(table: Any, rows: Any, engine: Optional[Any] = None) -> Any:
    """
    Builds an insert that skips rows conflicting with an existing key.

    ``INSERT OR IGNORE`` is SQLite-only; this renders the dialect's own spelling so the loaders do
    not have to know which database they are writing to.

    :param table: The target table or mapped class.
    :param rows: Rows to insert.
    :param engine: Engine whose dialect decides the spelling; the default one when omitted.
    :returns: An executable insert construct.
    """
    name = dialect_name(engine)
    target = getattr(table, "__table__", table)

    if name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        return sqlite_insert(target).values(rows).on_conflict_do_nothing()
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        return pg_insert(target).values(rows).on_conflict_do_nothing()
    if name == "mysql":
        from sqlalchemy.dialects.mysql import insert as mysql_insert
        return mysql_insert(target).values(rows).prefix_with("IGNORE")

    from sqlalchemy import insert
    return insert(target).values(rows)


def create_schema(engine: Optional[Any] = None) -> None:
    """
    Creates every table declared in :mod:`sample.models`.

    Does **not** create the FTS5 index: that is a SQLite virtual table with no cross-dialect
    equivalent, and it is built by the loader where the dialect is known.

    :param engine: Engine to build in; the default one when omitted.
    :returns: ``None``.
    """
    from sample.models import Base

    Base.metadata.create_all(engine or get_engine(read_only=False))
    logger.info(f"Schema created on {dialect_name(engine)}.")


def create_table(model: Any, engine: Optional[Any] = None) -> None:
    """
    Creates one table if it is not already there, leaving every other table alone.

    Needed because the pipeline's tables and the admin table have opposite lifecycles. Step 44
    rebuilds the released data by dropping and repopulating, which is right for data derived from
    the CSVs; the ``admins`` table is not derived from anything and must survive every rebuild, so
    it is created on demand rather than as part of a schema reset.

    :param model: The declarative class whose table to create.
    :param engine: Engine to build in; the default writable one when omitted.
    :returns: ``None``.
    """
    model.__table__.create(bind=engine or get_engine(read_only=False), checkfirst=True)


def session_scope(engine: Optional[Any] = None, read_only: bool = False) -> Any:
    """
    A transactional session context manager.

    :param engine: Engine to bind; the default one when omitted.
    :param read_only: Whether to open the connection read-only.
    :returns: A context manager yielding a :class:`~sqlalchemy.orm.Session` that commits on a
        clean exit and rolls back on an exception, so a failed admin edit cannot leave the
        database half-changed.
    """
    from contextlib import contextmanager

    from sqlalchemy.orm import Session

    @contextmanager
    def _scope() -> Any:
        session = Session(engine or get_engine(read_only=read_only))
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return _scope()


def describe() -> Dict[str, str]:
    """
    :returns: What this database is and what it supports, for logging and diagnostics.
    """
    engine = get_engine()
    return {
        "url": str(engine.url.render_as_string(hide_password=True)),
        "dialect": dialect_name(engine),
        "fulltext": fulltext_support(engine),
        "regexp": regexp_support(engine),
    }
