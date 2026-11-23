"""
Tests for administrator accounts, sessions and the admin interface.

The interesting cases here are the ones where a mistake is not visible from the outside: a session
that survives the password reset meant to kill it, a delete that leaves nobody able to sign in, or
a login form that reveals which usernames exist. Each of those looks like a working feature.
"""
import os
from typing import Any

import pytest


@pytest.fixture
def admin_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    A throwaway database with the full schema and a fixed signing key.

    :returns: The database URL.
    """
    url = f"sqlite:///{tmp_path}/admin.db"
    monkeypatch.setenv("SAMPLE_DATABASE_URL", url)
    monkeypatch.setenv("SAMPLE_SECRET_KEY", "fixed-key-so-tokens-are-reproducible")

    from sample.engine import get_engine
    from sample.models import Base

    get_engine.cache_clear() if hasattr(get_engine, "cache_clear") else None
    Base.metadata.create_all(get_engine(read_only=False))
    return url


@pytest.fixture
def client(admin_db: str) -> Any:
    """:returns: A ``TestClient`` that does not follow redirects, so they can be asserted on."""
    from fastapi.testclient import TestClient

    from sample.api.app import app

    return TestClient(app, follow_redirects=False)


@pytest.fixture
def alice(admin_db: str) -> str:
    """:returns: The username of a created administrator."""
    from sample import admin as admin_mod
    from sample.engine import session_scope

    with session_scope() as session:
        admin_mod.create_admin(session, "alice", "a-very-long-password")
    return "alice"


def _sign_in(client: Any, username: str = "alice", password: str = "a-very-long-password") -> Any:
    return client.post("/admin/login", data={"username": username, "password": password})


class TestPasswordHashing:
    """Storage format and verification."""

    def test_hash_is_salted_and_verifies(self) -> None:
        from sample.admin import hash_password, verify_password

        first = hash_password("a-very-long-password")
        second = hash_password("a-very-long-password")

        assert first != second, "a per-password salt must make identical passwords hash differently"
        assert first.startswith("scrypt$")
        assert verify_password("a-very-long-password", first)
        assert verify_password("a-very-long-password", second)
        assert not verify_password("a-very-long-passwore", first)

    def test_short_passwords_are_refused(self) -> None:
        from sample.admin import MIN_PASSWORD_LENGTH, AdminError, hash_password

        with pytest.raises(AdminError):
            hash_password("x" * (MIN_PASSWORD_LENGTH - 1))

    @pytest.mark.parametrize("stored", ["", "garbage", "scrypt$bad", "bcrypt$1$2$3$4$5", None])
    def test_corrupt_stored_hash_fails_closed(self, stored: Any) -> None:
        """A row that cannot be parsed must deny access, never grant it."""
        from sample.admin import verify_password

        assert verify_password("anything-at-all", stored) is False


class TestTokens:
    """Signed session tokens."""

    def test_round_trip(self, admin_db: str) -> None:
        from sample.admin import issue_token, verify_token

        assert verify_token(issue_token("alice", 3)) == ("alice", 3)

    def test_expired_token_is_rejected(self, admin_db: str) -> None:
        from sample.admin import issue_token, verify_token

        assert verify_token(issue_token("alice", 1, ttl=-1)) is None

    @pytest.mark.parametrize("forged", [
        "alice:1:99999999999:deadbeef",          # wrong signature
        "alice:1:99999999999",                   # no signature
        "",                                      # nothing at all
    ])
    def test_forged_tokens_are_rejected(self, admin_db: str, forged: str) -> None:
        from sample.admin import verify_token

        assert verify_token(forged) is None

    def test_payload_cannot_be_edited_without_breaking_the_signature(self, admin_db: str) -> None:
        """The signature covers the username and expiry, not just the token as a whole."""
        from sample.admin import issue_token, verify_token

        username, version, expiry, signature = issue_token("alice", 1).split(":")
        assert verify_token(f"bob:{version}:{expiry}:{signature}") is None
        assert verify_token(f"{username}:{version}:9999999999:{signature}") is None


class TestAccountRules:
    """The invariants that keep an installation reachable."""

    def test_usernames_are_case_insensitive(self, admin_db: str, alice: str) -> None:
        from sample import admin as admin_mod
        from sample.engine import session_scope

        with session_scope() as session:
            assert admin_mod.get_admin(session, "ALICE") is not None
            assert admin_mod.authenticate(session, "Alice", "a-very-long-password") is not None
            with pytest.raises(admin_mod.AdminError):
                admin_mod.create_admin(session, "ALICE", "another-long-password")

    def test_cannot_delete_the_last_active_admin(self, admin_db: str, alice: str) -> None:
        from sample import admin as admin_mod
        from sample.engine import session_scope

        with session_scope() as session:
            with pytest.raises(admin_mod.AdminError):
                admin_mod.delete_admin(session, "alice")

    def test_cannot_disable_the_last_active_admin(self, admin_db: str, alice: str) -> None:
        from sample import admin as admin_mod
        from sample.engine import session_scope

        with session_scope() as session:
            with pytest.raises(admin_mod.AdminError):
                admin_mod.set_active(session, "alice", False)

    def test_password_change_revokes_existing_tokens(self, admin_db: str, alice: str) -> None:
        from sample import admin as admin_mod
        from sample.engine import session_scope

        with session_scope() as session:
            admin = admin_mod.get_admin(session, "alice")
            token = admin_mod.issue_token(admin.username, admin.token_version)
            assert admin_mod.admin_for_token(session, token) is not None
            admin_mod.set_password(session, "alice", "a-brand-new-password")

        with session_scope() as session:
            assert admin_mod.admin_for_token(session, token) is None

    def test_suspended_admin_cannot_authenticate(self, admin_db: str, alice: str) -> None:
        from sample import admin as admin_mod
        from sample.engine import session_scope

        with session_scope() as session:
            admin_mod.create_admin(session, "bob", "bobs-long-password")
            admin_mod.set_active(session, "bob", False)
            assert admin_mod.authenticate(session, "bob", "bobs-long-password") is None


class TestAdminPages:
    """The HTTP surface."""

    def test_dashboard_requires_sign_in(self, client: Any) -> None:
        response = client.get("/admin")
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/login"

    @pytest.mark.parametrize("path,data", [
        ("/admin/password", {"current_password": "x", "new_password": "y", "confirm_password": "y"}),
        ("/admin/admins/create", {"new_username": "mallory", "new_user_password": "long-enough-pw"}),
        ("/admin/admins/delete", {"target": "alice"}),
        ("/admin/admins/toggle", {"target": "alice", "active": "0"}),
        ("/admin/entry", {"entry_id": "SAMPLE0000000001"}),
    ])
    def test_every_mutating_route_requires_sign_in(self, client: Any, alice: str,
                                                   path: str, data: dict) -> None:
        """An unauthenticated POST must change nothing, whichever endpoint it reaches."""
        response = client.post(path, data=data)
        assert response.status_code == 303
        assert response.headers["location"] == "/admin/login"

    def test_sign_in_and_out(self, client: Any, alice: str) -> None:
        from sample.admin import SESSION_COOKIE

        assert _sign_in(client).status_code == 303
        assert client.get("/admin").status_code == 200
        assert client.post("/admin/logout").status_code == 303
        assert not client.cookies.get(SESSION_COOKIE)

    def test_wrong_password_is_rejected(self, client: Any, alice: str) -> None:
        assert _sign_in(client, password="wrong-but-long-enough").status_code == 401

    def test_unknown_and_wrong_are_indistinguishable(self, client: Any, alice: str) -> None:
        """The form must not become a way to enumerate usernames."""
        unknown = _sign_in(client, username="nobody", password="wrong-but-long-enough")
        wrong = _sign_in(client, username="alice", password="wrong-but-long-enough")

        assert unknown.status_code == wrong.status_code == 401
        assert unknown.text == wrong.text

    def test_password_change_reissues_the_cookie(self, client: Any, alice: str) -> None:
        """The change revokes the caller's own token, so it must hand back a fresh one."""
        from sample.admin import SESSION_COOKIE

        _sign_in(client)
        old = client.cookies.get(SESSION_COOKIE)
        response = client.post("/admin/password", data={
            "current_password": "a-very-long-password",
            "new_password": "a-replacement-password",
            "confirm_password": "a-replacement-password"})

        assert response.status_code == 303
        assert client.cookies.get(SESSION_COOKIE) != old
        assert client.get("/admin").status_code == 200

    def test_password_change_requires_the_current_one(self, client: Any, alice: str) -> None:
        _sign_in(client)
        response = client.post("/admin/password", data={
            "current_password": "not-the-right-one", "new_password": "a-replacement-password",
            "confirm_password": "a-replacement-password"})

        assert "current password is incorrect" in response.text

    def test_mismatched_confirmation_is_refused(self, client: Any, alice: str) -> None:
        _sign_in(client)
        response = client.post("/admin/password", data={
            "current_password": "a-very-long-password", "new_password": "a-replacement-password",
            "confirm_password": "a-different-password"})

        assert "did not match" in response.text

    def test_cannot_delete_own_account_from_the_page(self, client: Any, alice: str) -> None:
        _sign_in(client)
        response = client.post("/admin/admins/delete", data={"target": "alice"})
        assert "cannot delete the account" in response.text

    def test_create_and_delete_another_admin(self, client: Any, alice: str) -> None:
        """Asserted against the database rather than the rendered page, which HTML-escapes the
        quotes in the confirmation message."""
        from sample import admin as admin_mod
        from sample.engine import session_scope

        _sign_in(client)
        client.post("/admin/admins/create",
                    data={"new_username": "bob", "new_user_password": "bobs-long-password"})
        with session_scope() as session:
            assert admin_mod.get_admin(session, "bob") is not None

        client.post("/admin/admins/delete", data={"target": "bob"})
        with session_scope() as session:
            assert admin_mod.get_admin(session, "bob") is None


class TestEntryEditing:
    """Manual record correction."""

    @pytest.fixture
    def peptide(self, admin_db: str) -> str:
        from sample.engine import session_scope
        from sample.models import Peptide

        with session_scope() as session:
            session.add(Peptide(entry_id="SAMPLE0000000001", sequence="GIGKFLK",
                                source="OriginalDB", is_decoy=0))
        return "SAMPLE0000000001"

    def test_edit_is_persisted(self, client: Any, alice: str, peptide: str) -> None:
        from sqlalchemy import select

        from sample.engine import session_scope
        from sample.models import Peptide

        _sign_in(client)
        response = client.post("/admin/entry", data={
            "entry_id": peptide, "sequence": "GIGKFLK", "source": "CorrectedDB"})
        assert response.status_code == 200

        with session_scope() as session:
            stored = session.scalars(
                select(Peptide).where(Peptide.entry_id == peptide)).first()
            assert stored.source == "CorrectedDB"

    def test_empty_sequence_is_refused(self, client: Any, alice: str, peptide: str) -> None:
        _sign_in(client)
        response = client.post("/admin/entry", data={"entry_id": peptide, "sequence": ""})
        assert "may not be empty" in response.text

    def test_unknown_entry_reports_rather_than_crashing(self, client: Any, alice: str,
                                                        peptide: str) -> None:
        _sign_in(client)
        response = client.post("/admin/entry", data={"entry_id": "NOPE", "sequence": "AAAA"})
        assert "No peptide with id" in response.text

    def test_fields_outside_the_allow_list_are_ignored(self, client: Any, alice: str,
                                                       peptide: str) -> None:
        """`is_decoy` decides whether a fabricated sequence is served as real; a form may not set it."""
        from sqlalchemy import select

        from sample.engine import session_scope
        from sample.models import Peptide

        _sign_in(client)
        client.post("/admin/entry", data={
            "entry_id": peptide, "sequence": "GIGKFLK", "is_decoy": "1", "entry_id_new": "HACK"})

        with session_scope() as session:
            stored = session.scalars(
                select(Peptide).where(Peptide.entry_id == peptide)).first()
            assert stored.is_decoy == 0
            assert stored.entry_id == peptide

    def test_allow_list_names_real_columns(self) -> None:
        """A typo here would silently make a field uneditable rather than fail."""
        from sample.api.admin import EDITABLE_PEPTIDE_FIELDS
        from sample.models import Peptide

        columns = {c.name for c in Peptide.__table__.columns}
        assert set(EDITABLE_PEPTIDE_FIELDS) <= columns
        assert "entry_id" not in EDITABLE_PEPTIDE_FIELDS
        assert "is_decoy" not in EDITABLE_PEPTIDE_FIELDS

    def test_edit_is_recorded_as_a_durable_override(self, client: Any, alice: str,
                                                     peptide: str) -> None:
        """Every changed field must leave a `PeptideOverride` row, not just a changed column --
        that row is what makes the correction survive the next pipeline rebuild."""
        from sqlalchemy import select

        from sample.engine import session_scope
        from sample.models import PeptideOverride

        _sign_in(client)
        client.post("/admin/entry", data={
            "entry_id": peptide, "sequence": "GIGKFLK", "source": "CorrectedDB"})

        with session_scope() as session:
            override = session.scalars(
                select(PeptideOverride).where(PeptideOverride.entry_id == peptide,
                                              PeptideOverride.field == "source")).first()
            assert override is not None
            assert override.value == "CorrectedDB"
            assert override.admin_username == "alice"

    def test_editing_a_field_twice_updates_the_override_in_place(self, client: Any, alice: str,
                                                                  peptide: str) -> None:
        from sqlalchemy import select

        from sample.engine import session_scope
        from sample.models import PeptideOverride

        _sign_in(client)
        client.post("/admin/entry", data={
            "entry_id": peptide, "sequence": "GIGKFLK", "source": "FirstFix"})
        client.post("/admin/entry", data={
            "entry_id": peptide, "sequence": "GIGKFLK", "source": "SecondFix"})

        with session_scope() as session:
            overrides = list(session.scalars(
                select(PeptideOverride).where(PeptideOverride.entry_id == peptide,
                                              PeptideOverride.field == "source")))
            assert len(overrides) == 1, "a second edit must update the row, not add another"
            assert overrides[0].value == "SecondFix"

    def test_success_message_says_the_edit_is_durable(self, client: Any, alice: str,
                                                       peptide: str) -> None:
        _sign_in(client)
        response = client.post("/admin/entry", data={
            "entry_id": peptide, "sequence": "GIGKFLK", "source": "CorrectedDB"})
        assert "survive the next pipeline rebuild" in response.text

    def test_record_override_rejects_a_field_outside_the_allow_list(self, admin_db: str) -> None:
        from sample.admin import AdminError, record_override
        from sample.engine import session_scope

        with pytest.raises(AdminError):
            with session_scope() as session:
                record_override(session, "SAMPLE0000000001", "entry_id", "HACK", "alice")


class TestOverrideReplay:
    """
    `sample.serialize._apply_peptide_overrides` -- reapplying durable corrections after a pipeline
    rebuild repopulates `peptides` from the CSVs, which knows nothing about them.

    Uses a bare sqlite3 connection rather than the SQLAlchemy fixtures above: this is the function
    the pipeline itself calls, in the `sample-chem` environment, which does not carry SQLAlchemy --
    the whole reason `EDITABLE_PEPTIDE_FIELDS` lives in `sample.config` and not `sample.admin`.
    """

    @pytest.fixture
    def conn(self, tmp_path: Any) -> Any:
        import sqlite3

        connection = sqlite3.connect(str(tmp_path / "rebuild.db"))
        connection.execute("""
            CREATE TABLE peptides (
                entry_id TEXT PRIMARY KEY, sequence TEXT, SMILES TEXT, source TEXT,
                Biological_Activity TEXT, Reference TEXT
            )
        """)
        connection.execute(
            "INSERT INTO peptides (entry_id, sequence, SMILES, source) "
            "VALUES ('SAMPLE0000000001', 'GIGKFLK', 'CCO', 'OriginalDB')")
        connection.commit()
        yield connection
        connection.close()

    def test_override_replaces_the_freshly_loaded_value(self, conn: Any) -> None:
        from sample.serialize import _apply_peptide_overrides

        conn.execute("""
            CREATE TABLE peptide_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT, field TEXT, value TEXT,
                admin_username TEXT, changed_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO peptide_overrides (entry_id, field, value, admin_username, changed_at) "
            "VALUES ('SAMPLE0000000001', 'source', 'CorrectedDB', 'alice', '2026-08-12T00:00:00')")
        conn.commit()

        applied = _apply_peptide_overrides(conn.cursor())
        conn.commit()

        assert applied == 1
        row = conn.execute(
            "SELECT source FROM peptides WHERE entry_id = 'SAMPLE0000000001'").fetchone()
        assert row[0] == "CorrectedDB"

    def test_creates_its_own_table_when_absent(self, conn: Any) -> None:
        """A fresh clone that has never run the admin app must still load cleanly."""
        from sample.serialize import _apply_peptide_overrides

        assert _apply_peptide_overrides(conn.cursor()) == 0

    def test_override_for_a_peptide_no_longer_in_the_release_is_skipped_not_raised(
            self, conn: Any) -> None:
        from sample.serialize import _apply_peptide_overrides

        conn.execute("""
            CREATE TABLE peptide_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT, field TEXT, value TEXT,
                admin_username TEXT, changed_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO peptide_overrides (entry_id, field, value, admin_username, changed_at) "
            "VALUES ('SAMPLE_LONG_GONE', 'source', 'X', 'alice', '2026-08-12T00:00:00')")
        conn.commit()

        assert _apply_peptide_overrides(conn.cursor()) == 0

    def test_field_name_is_validated_against_the_allow_list_before_reaching_sql(
            self, conn: Any) -> None:
        """Rows in `peptide_overrides` are replayed as a raw SQL column name; a field outside the
        allow-list -- `is_decoy` decides whether a fabricated sequence is served as real -- must
        never reach the `UPDATE` statement, however it got into the table."""
        from sample.serialize import _apply_peptide_overrides

        conn.execute("""
            CREATE TABLE peptide_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT, field TEXT, value TEXT,
                admin_username TEXT, changed_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO peptide_overrides (entry_id, field, value, admin_username, changed_at) "
            "VALUES ('SAMPLE0000000001', 'is_decoy', '1', 'alice', '2026-08-12T00:00:00')")
        conn.commit()

        assert _apply_peptide_overrides(conn.cursor()) == 0

    def test_survives_a_real_load_sql_database_rebuild(self, tmp_path: Any) -> None:
        """
        End-to-end through the actual function step 44 runs, not just the helper in isolation --
        the same hermetic pattern as `TestAgainstTheReleasedDatabase` in `tests/test_models.py`,
        which exists for exactly this reason: a unit test of the helper alone would not have
        caught a wrong call order or a table-name typo in `load_sql_database` itself.
        """
        import sqlite3

        import pandas as pd

        from sample.serialize import load_sql_database

        csv_path = str(tmp_path / "final_dataset.csv")
        db_path = str(tmp_path / "release.db")
        pd.DataFrame([
            {"entry_id": "SAMPLE0000000001", "sequence": "GIGKFLK",
             "SMILES": "", "source": "OriginalDB", "is_decoy": 0},
        ]).to_csv(csv_path, index=False)

        # Simulates a correction made through /admin before this rebuild ran.
        seed = sqlite3.connect(db_path)
        seed.execute("""
            CREATE TABLE peptide_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT, field TEXT, value TEXT,
                admin_username TEXT, changed_at TEXT
            )
        """)
        seed.execute(
            "INSERT INTO peptide_overrides (entry_id, field, value, admin_username, changed_at) "
            "VALUES ('SAMPLE0000000001', 'source', 'CorrectedDB', 'alice', '2026-08-12T00:00:00')")
        seed.commit()
        seed.close()

        load_sql_database(csv_path, db_path=db_path)

        result = sqlite3.connect(db_path).execute(
            "SELECT source FROM peptides WHERE entry_id = 'SAMPLE0000000001'").fetchone()
        assert result[0] == "CorrectedDB"


class TestNoNewDependencies:
    """
    The admin layer must import in the environment that actually serves it.

    `sample-web` carries a narrower dependency set than the development environment: `bcrypt`,
    `passlib` and `itsdangerous` are all absent from it. This session already lost 60 % of the
    activity table to exactly that gap in `sample-chem`, so authentication uses the standard
    library only.
    """

    @pytest.mark.parametrize("forbidden", ["bcrypt", "passlib", "itsdangerous", "argon2"])
    def test_admin_modules_import_nothing_extra(self, forbidden: str) -> None:
        import sample.admin
        import sample.api.admin

        for module in (sample.admin, sample.api.admin):
            source = open(module.__file__, encoding="utf-8").read()
            assert f"import {forbidden}" not in source, (
                f"{module.__name__} must not depend on {forbidden}; it is missing from sample-web")
