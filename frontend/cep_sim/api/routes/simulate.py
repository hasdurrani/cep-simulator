"""POST /api/simulate — apply ad and generate outputs for a session."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]

CHANNELS = ["digital_video", "social_media", "tv", "ooh", "radio", "cinema"]
EMOTIONS  = ["social_warmth", "confidence", "pride", "fun", "depth", "nostalgia"]


class SimulateRequest(BaseModel):
    session_id: str
    brand_id: str
    focal_cep_ids: list[str]
    secondary_cep_ids: list[str] = []
    focal_scenario_label: str = "custom"
    branding_clarity: float = 0.9
    attention_weight: float = 1.0
    channel: str = "digital_video"
    emotion: str = "social_warmth"


class MetricsSummary(BaseModel):
    mae: float | None
    median_spearman: float | None
    focal_brand_lift: float | None
    focal_brand_rank_pre: int | None
    focal_brand_rank_post: int | None


class SimulateResponse(BaseModel):
    run_id: str
    session_id: str
    brand_name: str
    focal_cep_labels: list[str]
    secondary_cep_labels: list[str] = []
    artifact_base_url: str           # prefix for all artifact files
    flight_table: list[dict]         # inline data for the results table
    metrics: MetricsSummary
    plotly_charts: dict[str, str] = {}  # chart_name -> Plotly JSON string


@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest):
    from backend.service.ad_engine import Ad, apply_ad_to_population, save_episodic_events
    from backend.service.output_builder import (
        flight_simulator_summary, generate_standard_outputs,
    )
    from backend.service.validator import (
        build_segment_summary, run_ad_impact, run_calibration_check, run_spearman_validity,
    )
    from frontend.cep_sim.api import session as session_store

    sess = session_store.get(request.session_id)
    if sess is None:
        raise HTTPException(404, "Session not found. Please run /api/setup first.")

    if not request.focal_cep_ids:
        raise HTTPException(400, "At least one focal CEP is required.")

    if request.brand_id not in sess.brand_name_map:
        raise HTTPException(400, f"Unknown brand_id: {request.brand_id!r}")

    brand_name = sess.brand_name_map[request.brand_id]
    run_id     = str(uuid.uuid4())

    # Per-session output directory (overwritten on each simulate call)
    out_dir = PROJECT_ROOT / "outputs" / "cep_sim" / "ui_sessions" / request.session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Apply ad ──────────────────────────────────────────────────────
    ad = Ad(
        ad_id=f"ui_{run_id[:8]}",
        brand_id=request.brand_id,
        brand_name=brand_name,
        focal_ceps=request.focal_cep_ids,
        secondary_ceps=request.secondary_cep_ids,
        branding_clarity=request.branding_clarity,
        attention_weight=request.attention_weight,
        channel=request.channel,
        emotion=request.emotion,
    )

    try:
        rbc_post, events = apply_ad_to_population(
            sess.respondent_ids, ad, sess.rbc_df, sess.config,
            responsiveness_map=sess.responsiveness_map,
        )
        # Design decision: episodic events are archived for analysis only.
        # Scoring uses rbc_post (updated semantic weights) as the sole source
        # of ad effect — episodic events are NOT passed to run_scenario_recall
        # or run_ad_impact, avoiding double-counting of the ad exposure.
        save_episodic_events(events, sess.config)
    except Exception as exc:
        raise HTTPException(500, f"Ad application failed: {exc}")

    # ── Ad impact ────────────────────────────────────────────────────
    from backend.service.recall_engine import get_scenarios
    scenarios = get_scenarios(sess.country)

    try:
        ad_impact_df = run_ad_impact(
            sess.respondent_ids, scenarios,
            sess.rbc_df, rbc_post,
            sess.cep_master_df, sess.brand_name_map, sess.config,
            brand_priors=sess.brand_priors,
            brand_similarity=sess.brand_similarity,
        )
        segment_summary_df = build_segment_summary(ad_impact_df, sess.respondents_df)
    except Exception as exc:
        raise HTTPException(500, f"Ad impact computation failed: {exc}")

    # ── Calibration metrics ───────────────────────────────────────────
    try:
        cal_df      = run_calibration_check(sess.scenario_recall_df, sess.long_df)
        spearman_df = run_spearman_validity(sess.scenario_recall_df, sess.long_df)
        mae         = round(float(cal_df["abs_error"].mean()), 4) if not cal_df.empty and "abs_error" in cal_df.columns else None
        med_rho     = round(float(spearman_df["spearman_rho"].median()), 4) if not spearman_df.empty else None
    except Exception:
        cal_df = spearman_df = None
        mae = med_rho = None

    # ── Resolve focal_scenario to a scenario_name key in impact_df ───
    # request.focal_scenario_label is a human-readable CEP label from the UI,
    # not a scenario_name key. Find the first scenario whose active_ceps resolve
    # to overlap with the requested focal CEP IDs.
    from backend.service.recall_engine import _resolve_cep_ids
    focal_scenario = scenarios[0]["scenario_name"] if scenarios else "custom"
    for scenario in scenarios:
        active_ids = _resolve_cep_ids(scenario["active_ceps"], sess.cep_master_df)
        if any(c in request.focal_cep_ids for c in active_ids):
            focal_scenario = scenario["scenario_name"]
            break

    # ── Generate outputs ─────────────────────────────────────────────
    try:
        generate_standard_outputs(
            rbc_pre=sess.rbc_df,
            rbc_post=rbc_post,
            impact_df=ad_impact_df,
            cep_master_df=sess.cep_master_df,
            long_df=sess.long_df,
            scenario_recall_df=sess.scenario_recall_df,
            focal_brand_id=request.brand_id,
            focal_brand_name=brand_name,
            focal_scenario=focal_scenario,
            config=sess.config,
            segment_summary_df=segment_summary_df,
            run_id=run_id,
            out_dir=out_dir,
            holdout_mae=mae,
            median_spearman=med_rho,
        )
    except Exception as exc:
        raise HTTPException(500, f"Output generation failed: {exc}")

    # ── Flight simulator table ────────────────────────────────────────
    flight_table = []
    try:
        summary = flight_simulator_summary(ad_impact_df, focal_scenario)
        flight_table = summary.to_dict(orient="records")
    except Exception:
        pass

    # ── Focal brand metrics ───────────────────────────────────────────
    focal_lift = focal_rank_pre = focal_rank_post = None
    for row in flight_table:
        if row.get("brand_name") == brand_name:
            focal_lift      = round(float(row.get("delta", 0)), 4)
            focal_rank_pre  = int(row.get("rank_pre", 0))
            focal_rank_post = int(row.get("rank_post", 0))
            break

    # ── CEP labels for response ───────────────────────────────────────
    cep_desc_map = sess.cep_master_df.set_index("cep_id")["cep_description"].to_dict() \
        if "cep_description" in sess.cep_master_df.columns else \
        sess.cep_master_df.set_index("cep_id")["cep_label"].to_dict() \
        if "cep_label" in sess.cep_master_df.columns else {}
    focal_cep_labels = [
        cep_desc_map.get(c, c).rstrip(" .") for c in request.focal_cep_ids
    ]
    secondary_cep_labels = [
        cep_desc_map.get(c, c).rstrip(" .") for c in request.secondary_cep_ids
    ]

    artifact_base_url = f"/api/artifacts/{request.session_id}"

    # ── Plotly interactive charts ─────────────────────────────────────
    plotly_charts: dict[str, str] = {}
    try:
        from frontend.cep_sim.api.plotly_charts import (
            flight_chart, memory_map_chart, calibration_chart,
        )
        plotly_charts["flight"] = flight_chart(
            ad_impact_df, focal_scenario, brand_name,
        )
        plotly_charts["memory_map"] = memory_map_chart(
            sess.rbc_df, sess.cep_master_df, request.brand_id,
        )
        if cal_df is not None and not cal_df.empty:
            plotly_charts["calibration"] = calibration_chart(cal_df, spearman_df)
    except Exception as _plotly_exc:
        import traceback
        print(f"[CEP Sim] Plotly chart generation failed: {_plotly_exc}")
        traceback.print_exc()

    return SimulateResponse(
        run_id=run_id,
        session_id=request.session_id,
        brand_name=brand_name,
        focal_cep_labels=focal_cep_labels,
        secondary_cep_labels=secondary_cep_labels,
        artifact_base_url=artifact_base_url,
        flight_table=flight_table,
        metrics=MetricsSummary(
            mae=mae,
            median_spearman=med_rho,
            focal_brand_lift=focal_lift,
            focal_brand_rank_pre=focal_rank_pre,
            focal_brand_rank_post=focal_rank_post,
        ),
        plotly_charts=plotly_charts,
    )


@router.get("/artifacts/{session_id}/{filename}")
def get_artifact(session_id: str, filename: str):
    """Serve a generated output file for a session."""
    path = PROJECT_ROOT / "outputs" / "cep_sim" / "ui_sessions" / session_id / filename
    if not path.exists():
        raise HTTPException(404, f"Artifact not found: {filename}")
    # Prevent path traversal
    try:
        path.resolve().relative_to(
            (PROJECT_ROOT / "outputs" / "cep_sim" / "ui_sessions").resolve()
        )
    except ValueError:
        raise HTTPException(403, "Forbidden")
    return FileResponse(str(path))
