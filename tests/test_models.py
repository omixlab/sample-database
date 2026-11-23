"""
Tests for the entity model and the engine layer.

Two questions matter here, and they are different from each other:

* **Does the model describe the database that exists?** A schema declared in Python and a schema
  built by ``CREATE TABLE`` strings can drift apart silently, and the first symptom is a query
  that fails in production. Writing these tests immediately caught one: ``peptide_names`` was
  declared with a surrogate ``id`` the real table does not have, so every name lookup raised
  ``no such column``.
* **Does it render for PostgreSQL and MySQL?** That is the point of the exercise. An ORM does not
  make a schema portable by itself -- autoincrement, upserts and full-text search all differ --
  so the dialect-specific parts are asserted rather than assumed.
"""
import os
from typing import Any

import pytest


class TestDocstringHygiene:
    """
    Docstrings must not contain control characters.

    A docstring that documents a regular expression -- ``\\b``, ``\\s``, ``\\x1b`` -- and is not a
    raw string has those escapes interpreted at parse time, so the module ships an actual
    backspace where the text should read ``\\b``. It is invisible in an editor, harmless at
    runtime, and stops the LaTeX build dead: ``Text line contains an invalid character``.
    """

    def test_no_control_characters_in_docstrings(self) -> None:
        import ast
        import pathlib

        offenders = []
        root = pathlib.Path(__file__).resolve().parent.parent / "sample"
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                    continue
                text = ast.get_docstring(node)
                if not text:
                    continue
                bad = {hex(ord(c)) for c in text if ord(c) < 32 and c not in "\n\t"}
                if bad:
                    name = getattr(node, "name", "<module>")
                    offenders.append(f"{path.name}:{getattr(node, 'lineno', 0)} {name} {bad}")

        assert not offenders, (
            "Docstrings contain control characters -- mark them r\"\"\"raw\"\"\": "
            + "; ".join(offenders))


class TestNoShadowedModuleImports:
    """
    A function that re-imports a name the module already imports at the top makes that name local
    to the *whole* function, so every use of it before the inner ``import`` raises
    ``UnboundLocalError`` -- regardless of the module-level import being right there.

    This shipped. `sample/cli.py` imported `pandas as pd` at module level and again inside
    `main()`, in a branch near the end of a 1,500-line `args.command` dispatch. That made `pd`
    local to `main()` and broke all thirteen earlier uses of it: `enrich-base` (step 23),
    `enrich-frequency-embeddings` (step 28), `enrich-structures-fallback`, `enrich`, the predictor
    commands, `assembly`, `update-dataset`, `fold` and `update-families`. Step 23 died with
    ``UnboundLocalError: cannot access local variable 'pd'`` on a release run, after steps 19-22
    had completed. Nothing before the dispatch reached that branch, so no import error, no
    linter warning at the call site, and no test covered it -- the failure needs the pipeline to
    get as far as step 23.

    `KNOWN_ISSUES.md` already lists latent NameErrors as a class worth re-scanning for. This is
    the scan, run on every commit instead.
    """

    def test_no_function_shadows_a_module_level_import_it_uses_earlier(self) -> None:
        import ast
        import pathlib

        offenders = []
        root = pathlib.Path(__file__).resolve().parent.parent / "sample"
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            module_names = set()
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        module_names.add(alias.asname or alias.name.split(".")[0])
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                local_imports: dict = {}
                for node in ast.walk(func):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            name = alias.asname or alias.name.split(".")[0]
                            local_imports.setdefault(name, []).append(node.lineno)
                for name, lines in local_imports.items():
                    if name not in module_names:
                        continue
                    first = min(lines)
                    earlier = sorted({
                        node.lineno for node in ast.walk(func)
                        if isinstance(node, ast.Name) and node.id == name
                        and isinstance(node.ctx, ast.Load) and node.lineno < first})
                    if earlier:
                        offenders.append(
                            f"{path.name}:{first} {func.name}() re-imports '{name}', "
                            f"which it already uses at line(s) {earlier[:5]}")

        assert not offenders, (
            "A local import shadows a module-level one that the same function uses earlier, so "
            "those earlier uses raise UnboundLocalError. Delete the inner import: "
            + "; ".join(offenders))


class TestEntityModel:
    def test_every_entity_is_registered(self) -> None:
        from sample.models import ALL_MODELS, Base

        declared = {model.__tablename__ for model in ALL_MODELS}
        assert declared == set(Base.metadata.tables)

    def test_peptide_relationships_are_wired(self) -> None:
        from sample.models import Peptide

        mapper = Peptide.__mapper__
        assert {"descriptors", "activities", "names", "family", "topology"} <= set(
            mapper.relationships.keys())

    def test_peptide_names_uses_its_natural_key(self) -> None:
        # A peptide has a name once. Declaring a surrogate `id` here produced a model that could
        # not read the shipped database, whose table has no such column.
        from sample.models import PeptideName

        assert {c.name for c in PeptideName.__table__.primary_key} == {"entry_id", "name"}

    def test_decoy_flag_is_indexed(self) -> None:
        # Every read filters on it; an unindexed flag makes the filter a full scan of 70,822 rows.
        from sample.models import Peptide

        assert Peptide.__table__.c.is_decoy.index

    def test_activity_carries_its_source_pointer(self) -> None:
        # Without both columns a paper can only be attached to a peptide, which would credit it
        # with every measurement of that peptide.
        from sample.models import Activity

        assert {"source_accession", "source_reference_index"} <= set(
            Activity.__table__.c.keys())


class TestDialectPortability:
    @pytest.mark.parametrize("dialect_module,expected", [
        ("sqlite", "INTEGER"),
        ("postgresql", "SERIAL"),
        ("mysql", "AUTO_INCREMENT"),
    ])
    def test_autoincrement_renders_per_dialect(self, dialect_module: str, expected: str) -> None:
        # The point of the migration: one declaration, three databases.
        import importlib

        from sqlalchemy.schema import CreateTable

        from sample.models import Activity

        dialect = importlib.import_module(f"sqlalchemy.dialects.{dialect_module}").dialect()
        ddl = str(CreateTable(Activity.__table__).compile(dialect=dialect))
        assert expected in ddl

    def test_every_table_renders_for_postgresql_and_mysql(self) -> None:
        from sqlalchemy.dialects import mysql, postgresql
        from sqlalchemy.schema import CreateTable

        from sample.models import Base

        for table in Base.metadata.tables.values():
            for dialect in (postgresql.dialect(), mysql.dialect()):
                assert str(CreateTable(table).compile(dialect=dialect)).strip()

    def test_insert_ignore_uses_the_dialect_spelling(self, tmp_path: Any) -> None:
        # `INSERT OR IGNORE` is SQLite-only; the loaders must not have to know which database
        # they are writing to.
        from sqlalchemy import create_engine

        from sample.engine import insert_ignore
        from sample.models import PeptideName

        engine = create_engine(f"sqlite:///{tmp_path/'x.db'}")
        statement = insert_ignore(PeptideName, [{"entry_id": "P1", "name": "n"}], engine)
        assert "ON CONFLICT" in str(statement.compile(dialect=engine.dialect)).upper()


class TestEngine:
    def test_defaults_to_the_packaged_sqlite_read_only(self, monkeypatch: pytest.MonkeyPatch
                                                       ) -> None:
        from sample.engine import URL_VARIABLE, database_url

        monkeypatch.delenv(URL_VARIABLE, raising=False)
        url = database_url(read_only=True, path="/tmp/x.db")
        assert url.startswith("sqlite+pysqlite:///file:")
        assert "mode=ro" in url

    def test_environment_url_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Moving to PostgreSQL is a connection-string change, not a code change.
        from sample.engine import URL_VARIABLE, database_url

        monkeypatch.setenv(URL_VARIABLE, "postgresql+psycopg://u:p@h/sample")
        assert database_url() == "postgresql+psycopg://u:p@h/sample"

    @pytest.mark.parametrize("url,fulltext,regexp", [
        ("sqlite://", "fts5", "python_udf"),
        ("postgresql+psycopg://u:p@h/d", "tsvector", "native"),
        ("mysql+pymysql://u:p@h/d", "mysql_fulltext", "native"),
    ])
    def test_capabilities_are_reported_per_dialect(self, url: str, fulltext: str, regexp: str,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
        # These are the three things an ORM does not make portable by itself, so the search layer
        # asks rather than assumes.
        from sqlalchemy.engine import make_url

        from sample import engine as engine_module

        class _FakeEngine:
            dialect = make_url(url).get_dialect()()

        fake = _FakeEngine()
        assert engine_module.fulltext_support(fake) == fulltext
        assert engine_module.regexp_support(fake) == regexp

    def test_schema_round_trips_through_a_fresh_database(self, tmp_path: Any) -> None:
        from sqlalchemy import create_engine, func, select
        from sqlalchemy.orm import Session

        from sample.engine import create_schema
        from sample.models import Peptide, PeptideName

        engine = create_engine(f"sqlite:///{tmp_path/'fresh.db'}")
        create_schema(engine)

        with Session(engine) as session:
            session.add(Peptide(entry_id="P1", sequence="GIGKFLHSA", is_decoy=0))
            session.add(PeptideName(entry_id="P1", name="magainin", name_kind="primary"))
            session.commit()

        with Session(engine) as session:
            peptide = session.get(Peptide, "P1")
            assert peptide.sequence == "GIGKFLHSA"
            assert [n.name for n in peptide.names] == ["magainin"]
            assert session.scalar(select(func.count()).select_from(Peptide)) == 1


class TestAgainstTheReleasedDatabase:
    """
    The model must read the database that step 44's real loader actually produces, not an
    idealised version of it.

    This used to open ``data/sample.db`` directly, skipping when the file was absent. Two
    problems came from that. It was not hermetic: the suite raced a concurrent pipeline rebuild
    and failed spuriously mid-``DROP``/``CREATE`` -- seen twice in one session. And a clean
    checkout with no release built skipped the test entirely, so a column the ORM declares and
    the loader does not produce could go unnoticed indefinitely.

    Instead this builds a throwaway database through :func:`sample.serialize.load_sql_database`
    itself -- the actual function step 44 runs -- from a two-row fixture CSV, so it still tests
    "does the ORM match what the loader really writes" without depending on any file the pipeline
    might be rewriting.
    """

    def test_reads_peptides_and_relationships(self, tmp_path: Any) -> None:
        import pandas as pd
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        from sample.engine import get_engine
        from sample.models import Peptide
        from sample.serialize import load_sql_database

        csv_path = str(tmp_path / "final_dataset.csv")
        db_path = str(tmp_path / "release.db")
        pd.DataFrame([
            {"entry_id": "SAMPLE0000000001", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
             "SMILES": "", "source": "merged", "is_decoy": 0},
            {"entry_id": "SAMPLE0000000002", "sequence": "SHUFFLEDDECOYXX",
             "SMILES": "", "source": "decoy", "is_decoy": 1},
        ]).to_csv(csv_path, index=False)

        load_sql_database(csv_path, db_path=db_path)

        with Session(get_engine(path=db_path)) as session:
            total = session.scalar(select(func.count()).select_from(Peptide))
            real = session.scalar(
                select(func.count()).select_from(Peptide).where(Peptide.is_decoy == 0))
            assert total == 2 and real == 1

            peptide = session.scalars(
                select(Peptide).where(Peptide.is_decoy == 0).limit(1)).first()
            # Relationship access must not raise: that is what catches a column the model
            # declares and the loader does not produce.
            assert peptide.sequence
            _ = peptide.activities, peptide.names, peptide.family, peptide.descriptors
