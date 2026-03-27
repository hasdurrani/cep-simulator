"""GET /api/export/{session_id} — download simulation results as a ZIP."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _session_out_dir(session_id: str) -> Path:
    return PROJECT_ROOT / "outputs" / "cep_sim" / "ui_sessions" / session_id


@router.get("/export/{session_id}")
def export_session(session_id: str):
    """
    Stream a ZIP containing CSVs and JSON for the session.

    Always included:
      market_baseline.csv     — population-average recall probs per brand × scenario
      model_params.json       — fitted parameters and calibration metrics

    Included if a simulation has been run:
      flight_summary.csv      — pre/post recall table for the focal scenario
      ad_impact_summary.csv   — per-scenario mean recall delta per brand
      scenario_diagnostics.csv — per-scenario MAE and Spearman ρ
    """
    from frontend.cep_sim.api import session as session_store

    sess = session_store.get(session_id)
    if sess is None:
        raise HTTPException(404, "Session not found. Please run /api/setup first.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

        # ── market_baseline.csv ──────────────────────────────────────────
        try:
            baseline = (
                sess.scenario_recall_df
                .groupby(["scenario_name", "brand_name"])["recall_prob"]
                .mean()
                .mul(100).round(2)
                .reset_index()
                .rename(columns={"recall_prob": "recall_pct"})
                .pivot(index="brand_name", columns="scenario_name", values="recall_pct")
                .reset_index()
            )
            zf.writestr("market_baseline.csv", baseline.to_csv(index=False))
        except Exception as exc:
            zf.writestr("market_baseline_error.txt", str(exc))

        # ── model_params.json ────────────────────────────────────────────
        params = {
            "country":      sess.country,
            "mae":          sess.mae,
            "holdout_mae":  sess.holdout_mae,
            "holdout_rho":  sess.holdout_rho,
            "fitted_params": {
                k: v for k, v in (sess.fitted_params or {}).items()
                if k != "grid_results"
            } if sess.fitted_params else None,
            "respondent_count": len(sess.respondent_ids),
            "brand_count":  len(sess.brand_name_map),
        }
        zf.writestr("model_params.json", json.dumps(params, indent=2))

        # ── simulation artifacts from disk ───────────────────────────────
        out_dir = _session_out_dir(session_id)
        csv_files = {
            "flight_summary.csv":        "flight_simulator_summary.csv",
            "ad_impact_summary.csv":     "ad_impact_output.csv",
            "scenario_diagnostics.csv":  "scenario_diagnostics.csv",
            "scenario_recall.csv":       "scenario_recall_output.csv",
        }
        for zip_name, disk_name in csv_files.items():
            p = out_dir / disk_name
            if p.exists():
                zf.writestr(zip_name, p.read_bytes())

    buf.seek(0)
    country = (sess.country or "market").lower().replace(" ", "_")
    filename = f"cep_sim_{country}_export.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
