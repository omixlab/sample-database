"""
Administrator accounts, password hashing and signed session tokens.

Everything here is deliberately built on the standard library. The web layer runs in the
``sample-web`` environment, which carries a narrower dependency set than the development
environment -- ``bcrypt``, ``passlib`` and ``itsdangerous`` are all absent from it. A missing
dependency in that environment is not a loud failure either: this session found step 19 silently
dropping 60 % of the activity table because ``pyahocorasick`` was missing from ``sample-chem`` and
the import error landed inside a per-file ``except``. Authentication is the last place to inherit
that failure mode, so it depends on nothing that is not in Python itself.

**Passwords** use :func:`hashlib.scrypt`, which is memory-hard and in the standard library. The
parameters below (N=2^14, r=8, p=1) are the interactive-login settings from the scrypt paper,
costing roughly 16 MB and ~100 ms per verification -- slow enough to make offline guessing
expensive, fast enough for a login form.

**Sessions** are HMAC-signed cookies rather than server-side state, so the app stays stateless and
a restart does not sign everyone out. The token is ``username:version:expiry:signature``; it is
signed, not encrypted, which is correct because it carries no secret -- a reader learns only a
username they already typed. What matters is that it cannot be *forged*, and the signature covers
every field, so an attacker cannot extend their own expiry or rename themselves to another user.

The one thing a stateless token cannot do by itself is revocation, which is why
:class:`~sample.models.Admin` carries ``token_version``: changing a password bumps it and every
cookie signed with the old value stops verifying.
"""
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from sample.config import EDITABLE_PEPTIDE_FIELDS
from sample.models import Admin, PeptideOverride

logger = logging.getLogger("sample.admin")

#: scrypt cost parameters. Raising N later is safe: the digest records the parameters it was
#: created with, so old hashes keep verifying and only new ones get the higher cost.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16

#: Rejected outright. Long passphrases beat short complex ones, so the rule is a length floor
#: rather than a character-class rule that pushes people toward "P@ssw0rd!".
MIN_PASSWORD_LENGTH = 12

#: How long a signed session stays valid. Short enough that a leaked cookie expires on its own,
#: long enough not to interrupt a working session.
SESSION_TTL_SECONDS = 12 * 60 * 60

#: Cookie the browser sends back.
SESSION_COOKIE = "sample_admin"

#: Where the signing key is kept when the environment does not supply one.
SECRET_FILENAME = ".admin_secret"


class AdminError(Exception):
    """A refusal the caller should show the operator verbatim, not a bug."""


def _now() -> int:
    """:returns: Current Unix time, as an int. Indirected so tests can freeze it."""
    return int(time.time())


def secret_key() -> bytes:
    """
    The key session tokens are signed with.

    Read from ``SAMPLE_SECRET_KEY`` when set. Otherwise a key is generated once and stored beside
    the database with owner-only permissions, because the alternative -- a hardcoded default --
    would mean every deployment of this code shares a signing key and anyone holding the source
    can mint a valid admin cookie for any installation.

    Losing the key signs everyone out; it does not lose any data.

    :returns: The signing key.
    :raises AdminError: If the stored key cannot be created or read.
    """
    configured = os.environ.get("SAMPLE_SECRET_KEY", "").strip()
    if configured:
        return configured.encode("utf-8")

    from sample.config import DATA_DIR

    path = os.path.join(str(DATA_DIR), SECRET_FILENAME)
    try:
        if os.path.exists(path):
            with open(path, "rb") as handle:
                stored = handle.read().strip()
            if stored:
                return stored
        generated = secrets.token_hex(32).encode("ascii")
        # 0600 before anything is written: a key that exists world-readable for even an instant is
        # a key that may have been read.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(generated)
        logger.warning(
            f"No SAMPLE_SECRET_KEY set; generated one at {path}. Set the variable in production "
            f"so sessions survive a redeploy and the key is not on disk beside the database.")
        return generated
    except OSError as e:
        raise AdminError(f"Could not establish a session signing key: {e}") from e


def hash_password(password: str) -> str:
    """
    Hashes a password for storage.

    :param password: The plaintext, which is never stored or logged.
    :returns: ``scrypt$N$r$p$salt$digest``, all base64. Self-describing so the cost parameters can
        change without invalidating existing accounts.
    :raises AdminError: If the password is shorter than :data:`MIN_PASSWORD_LENGTH`.
    """
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise AdminError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R,
                            p=SCRYPT_P, dklen=SCRYPT_DKLEN)
    return "$".join([
        "scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    ])


def verify_password(password: str, stored: str) -> bool:
    """
    Checks a password against a stored digest.

    :param password: Plaintext offered at the login form.
    :param stored: The value from :attr:`~sample.models.Admin.password_hash`.
    :returns: Whether they match. A malformed or empty stored value returns ``False`` rather than
        raising -- a corrupt row must fail closed, never grant access.
    """
    try:
        scheme, n, r, p, salt_b64, digest_b64 = str(stored).split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        computed = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt_b64),
            n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except Exception:
        return False
    # Constant-time: a byte-by-byte comparison leaks how much of the digest matched.
    return hmac.compare_digest(computed, expected)


def issue_token(username: str, token_version: int, ttl: int = SESSION_TTL_SECONDS) -> str:
    """
    Mints a signed session token.

    :param username: Account the token authenticates.
    :param token_version: The account's current version, so a later password change revokes it.
    :param ttl: Lifetime in seconds.
    :returns: ``username:version:expiry:signature``.
    """
    expiry = _now() + int(ttl)
    payload = f"{username}:{token_version}:{expiry}"
    signature = hmac.new(secret_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def verify_token(token: str) -> Optional[Tuple[str, int]]:
    """
    Validates a session token.

    :param token: Cookie value presented by the browser.
    :returns: ``(username, token_version)`` when the signature is valid and the token has not
        expired, otherwise ``None``. The caller must still confirm the version matches the stored
        one -- this function proves the token was issued by us, not that it is still current.
    """
    if not token:
        return None
    parts = str(token).rsplit(":", 1)
    if len(parts) != 2:
        return None
    payload, signature = parts
    expected = hmac.new(secret_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    # Verify before parsing: everything in the payload is attacker-controlled until this passes.
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        username, version, expiry = payload.split(":")
        if _now() >= int(expiry):
            return None
        return username, int(version)
    except (ValueError, TypeError):
        return None


def _normalise(username: str) -> str:
    """:returns: The canonical spelling of a username -- trimmed and lowercased."""
    return str(username or "").strip().lower()


def get_admin(session: Session, username: str) -> Optional[Admin]:
    """
    :param session: Open session.
    :param username: Any spelling; matched case-insensitively.
    :returns: The account, or ``None``.
    """
    name = _normalise(username)
    if not name:
        return None
    return session.scalars(select(Admin).where(Admin.username == name)).first()


def list_admins(session: Session) -> List[Admin]:
    """:returns: Every account, including disabled ones, ordered by username."""
    return list(session.scalars(select(Admin).order_by(Admin.username)))


def count_active(session: Session) -> int:
    """:returns: How many accounts can currently sign in."""
    return len([a for a in list_admins(session) if a.is_active])


def create_admin(session: Session, username: str, password: str) -> Admin:
    """
    Creates an account.

    :param session: Open session; the caller commits.
    :param username: Desired name, normalised before storage.
    :param password: Plaintext, hashed here.
    :returns: The new account.
    :raises AdminError: If the name is empty, already taken, or the password is too short.
    """
    name = _normalise(username)
    if not name:
        raise AdminError("Username may not be empty.")
    if len(name) > 64:
        raise AdminError("Username may be at most 64 characters.")
    if get_admin(session, name) is not None:
        raise AdminError(f"An admin named {name!r} already exists.")

    admin = Admin(username=name, password_hash=hash_password(password),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S"), token_version=1, is_active=1)
    session.add(admin)
    session.flush()
    logger.info(f"Created admin {name!r}.")
    return admin


def set_password(session: Session, username: str, password: str) -> Admin:
    """
    Changes an account's password and revokes its existing sessions.

    :param session: Open session; the caller commits.
    :param username: Whose password to change.
    :param password: New plaintext.
    :returns: The updated account.
    :raises AdminError: If there is no such account, or the password is too short.
    """
    admin = get_admin(session, username)
    if admin is None:
        raise AdminError(f"No admin named {_normalise(username)!r}.")
    admin.password_hash = hash_password(password)
    # Anything signed with the old version stops verifying, which is the point of a reset.
    admin.token_version = int(admin.token_version or 1) + 1
    session.flush()
    logger.info(f"Password changed for admin {admin.username!r}; existing sessions revoked.")
    return admin


def set_active(session: Session, username: str, active: bool) -> Admin:
    """
    Enables or suspends an account.

    :param session: Open session; the caller commits.
    :param username: Account to change.
    :param active: Whether it may sign in.
    :returns: The updated account.
    :raises AdminError: If there is no such account, or disabling it would leave none active.
    """
    admin = get_admin(session, username)
    if admin is None:
        raise AdminError(f"No admin named {_normalise(username)!r}.")
    if not active and admin.is_active and count_active(session) <= 1:
        raise AdminError(
            "This is the only active admin; disabling it would lock everyone out. Create another "
            "first.")
    admin.is_active = 1 if active else 0
    if not active:
        admin.token_version = int(admin.token_version or 1) + 1
    session.flush()
    return admin


def delete_admin(session: Session, username: str) -> None:
    """
    Removes an account.

    :param session: Open session; the caller commits.
    :param username: Account to remove.
    :raises AdminError: If there is no such account, or it is the last one that can sign in.

        Refusing the last admin is the difference between an inconvenience and an unrecoverable
        installation: with no account left, nobody can create one through the web interface. The
        command-line utility can still rescue such a database, but the web path must not be able
        to create that situation in the first place.
    """
    admin = get_admin(session, username)
    if admin is None:
        raise AdminError(f"No admin named {_normalise(username)!r}.")
    if admin.is_active and count_active(session) <= 1:
        raise AdminError(
            "Refusing to delete the only active admin -- nobody would be able to sign in. Create "
            "a replacement first.")
    session.delete(admin)
    session.flush()
    logger.info(f"Deleted admin {admin.username!r}.")


def authenticate(session: Session, username: str, password: str) -> Optional[Admin]:
    """
    Checks a username and password.

    :param session: Open session.
    :param username: Offered name.
    :param password: Offered plaintext.
    :returns: The account on success, otherwise ``None``.

        A missing account and a wrong password are both ``None`` on purpose: distinguishing them
        tells an attacker which usernames exist. The unknown-user path still runs a scrypt hash so
        the two answers take comparable time and cannot be told apart by a stopwatch either.
    """
    admin = get_admin(session, username)
    if admin is None or not admin.is_active:
        hash_password("x" * MIN_PASSWORD_LENGTH)
        return None
    if not verify_password(password, admin.password_hash):
        return None
    return admin


def record_override(session: Session, entry_id: str, field: str, value: str,
                    admin_username: str) -> None:
    """
    Upserts a durable manual correction, so it survives the next pipeline rebuild.

    :param session: Open session; the caller commits.
    :param entry_id: Peptide being corrected.
    :param field: Column name; must be in :data:`~sample.config.EDITABLE_PEPTIDE_FIELDS`.
    :param value: New value.
    :param admin_username: Who made the change.
    :raises AdminError: If `field` is not editable. `sample.serialize.load_sql_database` replays
        this table's rows as raw SQL, so a caller writing an unchecked column name here would be
        opening a SQL-injection path there, not just a data-quality one.
    """
    if field not in EDITABLE_PEPTIDE_FIELDS:
        raise AdminError(f"{field!r} is not an editable field.")

    existing = session.scalars(
        select(PeptideOverride).where(PeptideOverride.entry_id == entry_id,
                                      PeptideOverride.field == field)).first()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if existing is not None:
        existing.value = value
        existing.admin_username = admin_username
        existing.changed_at = now
    else:
        session.add(PeptideOverride(entry_id=entry_id, field=field, value=value,
                                    admin_username=admin_username, changed_at=now))
    session.flush()


def admin_for_token(session: Session, token: str) -> Optional[Admin]:
    """
    Resolves a session cookie to the account it authenticates.

    :param session: Open session.
    :param token: Cookie value.
    :returns: The account when the token verifies, has not expired, matches the account's current
        ``token_version`` and the account is still active; otherwise ``None``.
    """
    verified = verify_token(token)
    if verified is None:
        return None
    username, version = verified
    admin = get_admin(session, username)
    if admin is None or not admin.is_active:
        return None
    if int(admin.token_version or 1) != int(version):
        return None
    return admin
