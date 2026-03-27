"""POST /api/setup — run pipeline steps 1-5 and store session."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Available configs relative to project root
AVAILABLE_CONFIGS = {
    "uk": {
        "label": "UK — Heineken Beer Study",
        "path":  "backend/configs/cep_sim_config_uk.toml",
        "country": "UK",
    },
    "brazil": {
        "label": "Brazil — Heineken Study",
        "path":  "backend/configs/cep_sim_config_brazil.toml",
        "country": "Brazil",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # repo root


class SetupRequest(BaseModel):
    config_key: str  # "uk" or "brazil"


class CepInfo(BaseModel):
    cep_id: str
    cep_description: str
    cep_label: str
    cep_family: str


class BrandInfo(BaseModel):
    brand_id: str
    brand_name: str


class SetupResponse(BaseModel):
    session_id: str
    country: str
    respondent_count: int
    brand_count: int
    cep_count: int
    brands: list[BrandInfo]
    cep_families: dict[str, list[CepInfo]]   # family → list of CEPs
    default_brand_id: str                     # from config [ad] section
    default_focal_cep_ids: list[str]
    default_secondary_cep_ids: list[str]


@router.get("/configs")
def list_configs():
    return [{"key": k, **v} for k, v in AVAILABLE_CONFIGS.items()]


@router.post("/setup", response_model=SetupResponse)
def setup(request: SetupRequest):
    if request.config_key not in AVAILABLE_CONFIGS:
        raise HTTPException(400, f"Unknown config key: {request.config_key!r}. Choose from {list(AVAILABLE_CONFIGS)}")

    cfg_meta    = AVAILABLE_CONFIGS[request.config_key]
    config_path = PROJECT_ROOT / cfg_meta["path"]

    # ── Imports (deferred so startup is fast) ────────────────────────
    from backend.analysis.cep_sim.schemas.config import load_cep_sim_config
    from backend.analysis.cep_sim.service.load_data import load_survey
    from backend.analysis.cep_sim.service.ontology_builder import build_ontology
    from backend.analysis.cep_sim.service.recall_engine import _resolve_cep_ids, get_scenarios
    from backend.analysis.cep_sim.service.reshape_survey import reshape_wide_to_long
    from backend.analysis.cep_sim.service.respondent_builder import (
        build_respondent_brand_cep, build_respondents,
    )
    from backend.analysis.cep_sim.service.validator import run_scenario_recall
    from frontend.cep_sim.api import session as session_store

    try:
        config = load_cep_sim_config(str(config_path))
    except Exception as exc:
        raise HTTPException(500, f"Failed to load config: {exc}")

    # Resolve relative paths
    config.survey.zip_path      = str(PROJECT_ROOT / config.survey.zip_path)
    config.output.outputs_dir   = str(PROJECT_ROOT / config.output.outputs_dir)
    config.output.processed_dir = str(PROJECT_ROOT / config.output.processed_dir)

    try:
        df       = load_survey(config)
        long_df  = reshape_wide_to_long(df, config)
        cep_master_df, raw_map_df = build_ontology(long_df, config)
        respondents_df = build_respondents(df, config)
        rbc_df         = build_respondent_brand_cep(long_df, raw_map_df, config)
    except Exception as exc:
        raise HTTPException(500, f"Pipeline setup failed: {exc}")

    brand_name_map = (
        rbc_df[["brand_id", "brand_name"]]
        .drop_duplicates()
        .set_index("brand_id")["brand_name"]
        .to_dict()
    )
    respondent_ids = respondents_df["respondent_id"].astype(str).tolist()

    scenarios = get_scenarios(config.survey.country)
    try:
        scenario_recall_df = run_scenario_recall(
            respondent_ids, scenarios, rbc_df, cep_master_df, brand_name_map, config,
        )
    except Exception as exc:
        raise HTTPException(500, f"Baseline recall failed: {exc}")

    # ── Store session ─────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    sess = session_store.Session(
        session_id=session_id,
        country=config.survey.country,
        config=config,
        rbc_df=rbc_df,
        cep_master_df=cep_master_df,
        raw_map_df=raw_map_df,
        long_df=long_df,
        scenario_recall_df=scenario_recall_df,
        brand_name_map=brand_name_map,
        respondent_ids=respondent_ids,
        respondents_df=respondents_df,
    )
    session_store.put(sess)

    # ── Build response ────────────────────────────────────────────────
    brands = [
        BrandInfo(brand_id=bid, brand_name=bname)
        for bid, bname in sorted(brand_name_map.items(), key=lambda x: x[1])
    ]

    cep_families: dict[str, list[CepInfo]] = {}
    for _, row in cep_master_df.iterrows():
        family = row.get("cep_family", "Other")
        cep_families.setdefault(family, []).append(CepInfo(
            cep_id=row["cep_id"],
            cep_description=row["cep_description"],
            cep_label=row.get("cep_label", row["cep_id"]),
            cep_family=family,
        ))

    # Defaults from config [ad] section
    default_focal     = []
    default_secondary = []
    default_brand_id  = brands[0].brand_id if brands else ""
    if config.ad:
        default_brand_id  = config.ad.brand_id
        default_focal     = _resolve_cep_ids(config.ad.focal_cep_keywords, cep_master_df)
        default_secondary = _resolve_cep_ids(config.ad.secondary_cep_keywords, cep_master_df)

    return SetupResponse(
        session_id=session_id,
        country=config.survey.country,
        respondent_count=len(respondents_df),
        brand_count=len(brand_name_map),
        cep_count=len(cep_master_df),
        brands=brands,
        cep_families=cep_families,
        default_brand_id=default_brand_id,
        default_focal_cep_ids=default_focal,
        default_secondary_cep_ids=default_secondary,
    )
