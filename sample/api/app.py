"""
The FastAPI application.

Replaces the Flask app in ``sample/webserver.py``. Two routers are mounted:

* :mod:`sample.api.pages` — the HTML dashboard, at the URL paths it has always used. The
  templates hard-code those paths and the compose healthcheck hits ``/``, so they are the
  contract and did not change during the port.
* :mod:`sample.api.v1` — the JSON API under ``/api/v1``, with OpenAPI generated from its response
  models rather than maintained as a separate document that would drift.
"""
import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sample.api import admin, pages, v1
from sample.api.common import STATIC_DIR

logger = logging.getLogger("sample.api.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DESCRIPTION = """
Programmatic access to the sAMPle antimicrobial peptide database.

**Decoys are never returned.** The underlying table stores generated training negatives
(shuffled sequences and UniProt fragments) alongside real peptides; every endpoint here filters
them out. A sequence returned by this API is a real, recorded peptide.

Prediction is asynchronous: submit to `/api/v1/predict`, then poll `/api/v1/jobs/{job_id}`.
"""

app = FastAPI(
    title="sAMPle API",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

SESSION_COOKIE_NAME = "sample_session_id"


@app.middleware("http")
async def session_middleware(request: Request, call_next: Any) -> Any:
    cookie_session = request.cookies.get(SESSION_COOKIE_NAME)
    header_session = request.headers.get("x-session-id")
    session_id = cookie_session or header_session
    had_cookie = bool(cookie_session)
    if not session_id:
        session_id = str(uuid.uuid4())
    request.state.session_id = session_id

    response = await call_next(request)
    if not had_cookie:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=30 * 86400,
            httponly=False,
            samesite="lax",
            path="/"
        )
    return response

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(v1.router)
# Before the page router, whose catch-all detail routes would otherwise shadow /admin/*.
app.include_router(admin.router)
app.include_router(pages.router)


@app.exception_handler(FileNotFoundError)
def handle_missing_database(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """
    Turns a missing release database into an explanatory 503 rather than a stack trace.

    The database is built by pipeline step 44, so this is a deployment state a fresh checkout is
    legitimately in, and the response should say which step produces it.

    :param request: The request that failed.
    :param exc: The underlying error.
    :returns: A 503 naming the remedy.
    """
    logger.error(f"{request.url.path}: {exc}")
    return JSONResponse(status_code=503, detail=None, content={"detail": str(exc)})


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """
    Starts the application under uvicorn. Also launches the background ML worker
    to process queued predictions unless SAMPLE_NO_WORKER=1 is set.

    :param host: Interface to bind. Defaults to all interfaces so the dashboard is reachable from
        other machines on the network.
    :param port: TCP port to listen on.
    :param reload: Restart on source changes; for development only.
    :returns: ``None``; blocks until the server stops.
    """
    import subprocess
    import sys
    import uvicorn

    worker_proc = None
    if os.environ.get("SAMPLE_NO_WORKER", "").lower() not in ("1", "true", "yes"):
        try:
            logger.info("Spawning background ML model worker process (sample.worker)...")
            worker_proc = subprocess.Popen([sys.executable, "-m", "sample.worker"])
        except Exception as e:
            logger.warning(f"Could not spawn ML worker process: {e}")

    logger.info(f"Starting sAMPle on http://{host}:{port} (docs at /api/v1/docs)")
    try:
        uvicorn.run("sample.api.app:app" if reload else app, host=host, port=port, reload=reload)
    finally:
        if worker_proc and worker_proc.poll() is None:
            logger.info("Stopping background ML model worker process...")
            try:
                worker_proc.terminate()
                worker_proc.wait(timeout=5)
            except Exception:
                try:
                    worker_proc.kill()
                except Exception:
                    pass

