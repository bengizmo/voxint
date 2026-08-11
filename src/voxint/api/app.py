"""FastAPI application: ingest, runs, review UI, health."""

from fastapi import FastAPI

from voxint import __version__

app = FastAPI(title="Voxint", version=__version__)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
