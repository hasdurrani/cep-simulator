"""POST /api/compare — run two ads against the same baseline and compare displacement."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class AdSpec(BaseModel):
    brand_id: str
    focal_cep_ids: list[str]
    secondary_cep_ids: list[str] = []
    branding_clarity: float = 0.9
    attention_weight: float = 1.0
    channel: str = "digital_video"
    emotion: str = "social_warmth"


class CompareRequest(BaseModel):
    session_id: str
    ad_a: AdSpec
    ad_b: AdSpec


class CompareResponse(BaseModel):
    session_id: str
    run_id: str
    focal_scenario: str
    ad_a_brand_name: str
    ad_b_brand_name: str
    ad_a_table: list[dict]
    ad_b_table: list[dict]
    compare_chart: str   # Plotly JSON


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    from backend.service.ad_engine import Ad, apply_ad_to_population
    from backend.service.validator import run_ad_impact
    from backend.service.output_builder import flight_simulator_summary
    from backend.service.recall_engine import get_scenarios, _resolve_cep_ids
    from frontend.cep_sim.api import session as session_store
    from frontend.cep_sim.api.plotly_charts import compare_chart

    sess = session_store.get(request.session_id)
    if sess is None:
        raise HTTPException(404, "Session not found. Please run /api/setup first.")

    for label, spec in [("Ad A", request.ad_a), ("Ad B", request.ad_b)]:
        if not spec.focal_cep_ids:
            raise HTTPException(400, f"{label}: at least one focal CEP is required.")
        if spec.brand_id not in sess.brand_name_map:
            raise HTTPException(400, f"{label}: unknown brand_id {spec.brand_id!r}")

    brand_a_name = sess.brand_name_map[request.ad_a.brand_id]
    brand_b_name = sess.brand_name_map[request.ad_b.brand_id]
    run_id = str(uuid.uuid4())

    scenarios = get_scenarios(sess.country)

    # ── Find focal scenario from Ad A's CEPs ─────────────────────────────
    focal_scenario = scenarios[0]["scenario_name"] if scenarios else "custom"
    for scenario in scenarios:
        active_ids = _resolve_cep_ids(scenario["active_ceps"], sess.cep_master_df)
        if any(c in request.ad_a.focal_cep_ids for c in active_ids):
            focal_scenario = scenario["scenario_name"]
            break

    # ── Apply each ad against the same unmodified baseline ───────────────
    def _apply(spec: AdSpec, brand_name: str):
        ad = Ad(
            ad_id=f"compare_{run_id[:8]}",
            brand_id=spec.brand_id,
            brand_name=brand_name,
            focal_ceps=spec.focal_cep_ids,
            secondary_ceps=spec.secondary_cep_ids,
            branding_clarity=spec.branding_clarity,
            attention_weight=spec.attention_weight,
            channel=spec.channel,
            emotion=spec.emotion,
        )
        rbc_post, _ = apply_ad_to_population(
            sess.respondent_ids, ad, sess.rbc_df, sess.config,
            responsiveness_map=sess.responsiveness_map,
        )
        impact_df = run_ad_impact(
            sess.respondent_ids, scenarios,
            sess.rbc_df, rbc_post,
            sess.cep_master_df, sess.brand_name_map, sess.config,
            brand_priors=sess.brand_priors,
            brand_similarity=sess.brand_similarity,
        )
        summary = flight_simulator_summary(impact_df, focal_scenario)
        return impact_df, summary.to_dict(orient="records")

    try:
        impact_a, table_a = _apply(request.ad_a, brand_a_name)
        impact_b, table_b = _apply(request.ad_b, brand_b_name)
    except Exception as exc:
        raise HTTPException(500, f"Simulation failed: {exc}")

    # ── Comparison chart ─────────────────────────────────────────────────
    try:
        chart_json = compare_chart(
            impact_a, impact_b,
            focal_scenario,
            brand_a_name, brand_b_name,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        chart_json = "{}"

    return CompareResponse(
        session_id=request.session_id,
        run_id=run_id,
        focal_scenario=focal_scenario,
        ad_a_brand_name=brand_a_name,
        ad_b_brand_name=brand_b_name,
        ad_a_table=table_a,
        ad_b_table=table_b,
        compare_chart=chart_json,
    )
