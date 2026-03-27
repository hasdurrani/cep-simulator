"""
CEP Simulator — FastAPI application.

Run from the project root:
    uvicorn frontend.cep_sim.api.app:app --reload --port 8000
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from frontend.cep_sim.api.routes import setup, simulate, baseline

app = FastAPI(title="CEP Simulator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(setup.router,    prefix="/api")
app.include_router(simulate.router, prefix="/api")
app.include_router(baseline.router, prefix="/api")

# Serve the React UI as static files
_ui_dir = Path(__file__).parent.parent / "ui"
app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
