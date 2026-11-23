"""
The administrator interface: sign-in, account management and manual record edits.

Mounted under ``/admin`` and deliberately kept out of the OpenAPI schema -- it is an operator
tool, not part of the public API, and listing its routes in the published documentation would
advertise them to everyone reading the API docs.

**What "manual edit" means here, and what it cannot mean.** Most tables in this database are
*derived*: step 44 rebuilds peptides, descriptors, activities, names, families and the
bibliography from the CSVs, dropping and repopulating as they go. A value written directly to one
of those tables is erased by the next pipeline run -- so a change made here is instead recorded in
:class:`~sample.models.PeptideOverride`, which step 44 does not touch, and
:func:`sample.serialize.load_sql_database` reapplies every row there onto the freshly rebuilt
``peptides`` table before it hands the database back. The correction survives as many rebuilds as
it takes for the underlying source data to actually fix itself.

Editing is still restricted to :class:`~sample.models.Peptide` fields a curator would plausibly
correct (see :data:`~sample.config.EDITABLE_PEPTIDE_FIELDS`), and every save is recorded in the log
with the account that made it, in addition to the durable override row.

Names are not editable here. A peptide's names live in :class:`~sample.models.PeptideName`, one row
per synonym, so editing them is a different shape of operation than overwriting a column -- it is
add/remove against a set. Pretending otherwise with a single text box would silently drop the other
60,648 harvested synonyms.
"""
import logging
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from sample import admin as admin_mod
from sample.api.common import templates
from sample.config import EDITABLE_PEPTIDE_FIELDS

logger = logging.getLogger("sample.api.admin")

router = APIRouter(prefix="/admin", tags=["admin"], include_in_schema=False)


async def _field(request: Request, name: str, default: str = "") -> str:
    """
    Reads one value from a submitted form.

    Read this way rather than through FastAPI's ``Form(...)`` parameters, which is the more
    idiomatic spelling but fails *at import* when ``python-multipart`` is absent -- one such
    parameter anywhere stops the whole application from starting, rather than breaking only the
    page that uses it. Reading the body here fails at request time instead, which is the failure
    the page routes already have.

    ``python-multipart`` is required either way: Starlette 1.4 asserts on it for **all** form
    parsing, not only multipart uploads, so it is now declared in the web environment. That was
    not true of the 1.3 series, where urlencoded bodies parsed without it -- which is exactly why
    it went undeclared for so long.

    :param request: Incoming request.
    :param name: Field to read.
    :param default: Value when the field is absent.
    :returns: The submitted value, stripped.
    """
    form = await request.form()
    value = form.get(name)
    return str(value).strip() if value is not None else default


def _session() -> Any:
    """
    :returns: A writable transactional session, with the admin and overrides tables guaranteed to
        exist. Both have the opposite lifecycle from the pipeline-derived tables (see
        :func:`sample.engine.create_table`), so neither is created by a schema reset -- only here,
        on demand.
    """
    from sample.engine import create_table, session_scope
    from sample.models import Admin, PeptideOverride

    create_table(Admin)
    create_table(PeptideOverride)
    return session_scope()


def current_admin(request: Request) -> Optional[str]:
    """
    Identifies the signed-in administrator.

    :param request: Incoming request, whose session cookie is checked.
    :returns: The username, or ``None`` when nobody is signed in. The account is re-read on every
        request rather than trusted from the cookie, so disabling an account or resetting its
        password takes effect immediately instead of at the next expiry.
    """
    token = request.cookies.get(admin_mod.SESSION_COOKIE, "")
    if not token:
        return None
    try:
        with _session() as session:
            admin = admin_mod.admin_for_token(session, token)
            return admin.username if admin else None
    except Exception as e:
        logger.error(f"Could not verify admin session: {e}")
        return None


def _redirect_to_login() -> RedirectResponse:
    """:returns: A redirect to the sign-in page, as a GET."""
    return RedirectResponse(url="/admin/login", status_code=303)


def _render(request: Request, username: str, **extra: Any) -> Any:
    """
    Renders the dashboard with the account list.

    :param request: Incoming request.
    :param username: The signed-in administrator.
    :param extra: Message or error to display.
    :returns: The rendered ``admin.html``.
    """
    with _session() as session:
        admins = [{"username": a.username, "is_active": bool(a.is_active),
                   "created_at": a.created_at} for a in admin_mod.list_admins(session)]
    return templates.TemplateResponse(request, "admin.html", {
        "active_tab": "admin", "title": "Administration",
        "username": username, "admins": admins,
        "editable_fields": EDITABLE_PEPTIDE_FIELDS,
        "min_password_length": admin_mod.MIN_PASSWORD_LENGTH,
        **extra,
    })


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> Any:
    """
    The sign-in page.

    :param request: Incoming request.
    :returns: The form, or a redirect to the dashboard when already signed in.
    """
    if current_admin(request):
        return RedirectResponse(url="/admin", status_code=303)
    with _session() as session:
        no_accounts = not admin_mod.list_admins(session)
    return templates.TemplateResponse(request, "admin_login.html", {
        "title": "Sign in", "active_tab": "admin", "no_accounts": no_accounts,
    })


@router.post("/login")
async def login(request: Request) -> Any:
    """
    Checks credentials and starts a session.

    :param request: Incoming request, carrying ``username`` and ``password``.
    :returns: A redirect to the dashboard on success, or the form with an error.
    """
    form = await request.form()
    username = str(form.get("username") or "").strip()
    password = str(form.get("password") or "")

    with _session() as session:
        admin = admin_mod.authenticate(session, username, password)
        if admin is None:
            logger.warning(f"Failed admin sign-in for {str(username).strip().lower()!r}.")
            return templates.TemplateResponse(request, "admin_login.html", {
                "title": "Sign in", "active_tab": "admin",
                # One message for both causes, so the form cannot be used to discover which
                # usernames exist.
                "error": "Incorrect username or password.",
            }, status_code=401)
        token = admin_mod.issue_token(admin.username, int(admin.token_version or 1))
        name = admin.username

    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        admin_mod.SESSION_COOKIE, token,
        max_age=admin_mod.SESSION_TTL_SECONDS,
        httponly=True,      # not readable from JavaScript, so an XSS bug cannot steal the session
        samesite="lax",     # not sent on cross-site POSTs, which blocks the basic CSRF shape
        secure=request.url.scheme == "https",
    )
    logger.info(f"Admin {name!r} signed in.")
    return response


@router.post("/logout")
def logout(request: Request) -> Any:
    """
    Ends the session.

    :param request: Incoming request.
    :returns: A redirect to the sign-in page with the cookie cleared.
    """
    response = _redirect_to_login()
    response.delete_cookie(admin_mod.SESSION_COOKIE)
    return response


@router.get("", response_class=HTMLResponse)
def dashboard(request: Request) -> Any:
    """
    The administration dashboard.

    :param request: Incoming request.
    :returns: The dashboard, or a redirect to sign in.
    """
    username = current_admin(request)
    if not username:
        return _redirect_to_login()
    return _render(request, username)


@router.post("/password")
async def change_password(request: Request) -> Any:
    """
    Changes the signed-in administrator's own password.

    Requires the current password even though the session already proves identity: without it,
    anyone reaching an unattended browser could take the account over permanently.

    :param request: Incoming request, carrying ``current_password``, ``new_password`` and
        ``confirm_password``.
    :returns: The dashboard with the outcome. On success the session cookie is reissued, because
        changing the password revokes every token signed with the previous version -- including
        the one this request arrived with.
    """
    username = current_admin(request)
    if not username:
        return _redirect_to_login()

    form = await request.form()
    current_password = str(form.get("current_password") or "")
    new_password = str(form.get("new_password") or "")
    confirm_password = str(form.get("confirm_password") or "")

    if new_password != confirm_password:
        return _render(request, username, error="The new passwords did not match.")
    try:
        with _session() as session:
            if admin_mod.authenticate(session, username, current_password) is None:
                logger.warning(f"Admin {username!r} gave the wrong current password.")
                return _render(request, username, error="Your current password is incorrect.")
            admin = admin_mod.set_password(session, username, new_password)
            token = admin_mod.issue_token(admin.username, int(admin.token_version))
    except admin_mod.AdminError as e:
        return _render(request, username, error=str(e))

    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        admin_mod.SESSION_COOKIE, token, max_age=admin_mod.SESSION_TTL_SECONDS,
        httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return response


@router.post("/admins/create")
async def add_admin(request: Request) -> Any:
    """
    Creates another administrator.

    :param request: Incoming request, carrying ``new_username`` and ``new_user_password``.
    :returns: The dashboard with the outcome.
    """
    username = current_admin(request)
    if not username:
        return _redirect_to_login()

    form = await request.form()
    new_username = str(form.get("new_username") or "").strip()
    new_user_password = str(form.get("new_user_password") or "")

    try:
        with _session() as session:
            admin_mod.create_admin(session, new_username, new_user_password)
    except admin_mod.AdminError as e:
        return _render(request, username, error=str(e))
    logger.info(f"Admin {username!r} created account "
                f"{str(new_username).strip().lower()!r}.")
    return _render(request, username,
                   message=f"Created {str(new_username).strip().lower()!r}.")


@router.post("/admins/delete")
async def remove_admin(request: Request) -> Any:
    """
    Deletes another administrator.

    :param request: Incoming request, carrying ``target``.
    :returns: The dashboard with the outcome. Deleting your own account is refused, as is deleting
        the last active one -- see :func:`sample.admin.delete_admin`.
    """
    username = current_admin(request)
    if not username:
        return _redirect_to_login()

    target = await _field(request, "target")
    if str(target).strip().lower() == username:
        return _render(request, username,
                       error="You cannot delete the account you are signed in with.")
    try:
        with _session() as session:
            admin_mod.delete_admin(session, target)
    except admin_mod.AdminError as e:
        return _render(request, username, error=str(e))
    logger.info(f"Admin {username!r} deleted account {str(target).strip().lower()!r}.")
    return _render(request, username, message=f"Deleted {str(target).strip().lower()!r}.")


@router.post("/admins/toggle")
async def toggle_admin(request: Request) -> Any:
    """
    Suspends or re-enables another administrator.

    :param request: Incoming request, carrying ``target`` and ``active`` (``"1"`` to enable,
        anything else to suspend).
    :returns: The dashboard with the outcome.
    """
    username = current_admin(request)
    if not username:
        return _redirect_to_login()

    form = await request.form()
    target = str(form.get("target") or "").strip()
    active = str(form.get("active") or "")

    if str(target).strip().lower() == username:
        return _render(request, username,
                       error="You cannot suspend the account you are signed in with.")
    try:
        with _session() as session:
            admin_mod.set_active(session, target, active == "1")
    except admin_mod.AdminError as e:
        return _render(request, username, error=str(e))
    return _render(request, username,
                   message=f"{str(target).strip().lower()!r} "
                           f"{'enabled' if active == '1' else 'suspended'}.")


@router.get("/entry", response_class=HTMLResponse)
def find_entry(request: Request, entry_id: str = "") -> Any:
    """
    Loads one peptide for editing.

    :param request: Incoming request.
    :param entry_id: The peptide to load.
    :returns: The dashboard with the record populated, or an error when it does not exist.
    """
    username = current_admin(request)
    if not username:
        return _redirect_to_login()
    if not entry_id.strip():
        return _render(request, username)

    from sqlalchemy import select

    from sample.models import Peptide

    with _session() as session:
        peptide = session.scalars(
            select(Peptide).where(Peptide.entry_id == entry_id.strip())).first()
        if peptide is None:
            return _render(request, username, error=f"No peptide with id {entry_id.strip()!r}.")
        record = {"entry_id": peptide.entry_id, "is_decoy": bool(peptide.is_decoy),
                  **{f: getattr(peptide, f, None) for f in EDITABLE_PEPTIDE_FIELDS}}
    return _render(request, username, record=record)


@router.post("/entry")
async def save_entry(request: Request) -> Any:
    """
    Writes hand-edited values back to a peptide.

    :param request: Incoming request; ``entry_id`` names the record and the edited fields are read
        from the same body, so the allow-list is applied in one place rather than repeated as
        parameters.
    :returns: The dashboard with the outcome.
    """
    username = current_admin(request)
    if not username:
        return _redirect_to_login()

    form = await request.form()
    entry_id = str(form.get("entry_id") or "").strip()
    if not entry_id:
        return _render(request, username, error="No entry id was submitted.")

    from sqlalchemy import select

    from sample.models import Peptide

    with _session() as session:
        peptide = session.scalars(
            select(Peptide).where(Peptide.entry_id == entry_id.strip())).first()
        if peptide is None:
            return _render(request, username, error=f"No peptide with id {entry_id.strip()!r}.")

        changed = []
        for field in EDITABLE_PEPTIDE_FIELDS:
            if field not in form:
                continue
            new = str(form.get(field) or "").strip()
            old = str(getattr(peptide, field, "") or "")
            if new == old:
                continue
            # A peptide with no sequence is unusable: it is the key every other table joins on and
            # the input to every descriptor.
            if field == "sequence" and not new:
                return _render(request, username, error="A peptide's sequence may not be empty.")
            setattr(peptide, field, new)
            admin_mod.record_override(session, entry_id.strip(), field, new, username)
            changed.append(f"{field}: {old!r} -> {new!r}")

        record = {"entry_id": peptide.entry_id, "is_decoy": bool(peptide.is_decoy),
                  **{f: getattr(peptide, f, None) for f in EDITABLE_PEPTIDE_FIELDS}}

    if not changed:
        return _render(request, username, record=record, message="No changes to save.")

    # The audit trail: who changed what, since the edit itself leaves no other trace.
    logger.info(f"Admin {username!r} edited {entry_id.strip()!r}: {'; '.join(changed)}")
    return _render(request, username, record=record,
                   message=f"Saved {len(changed)} change(s) to {entry_id.strip()}. "
                           f"This will survive the next pipeline rebuild (step 44).")
