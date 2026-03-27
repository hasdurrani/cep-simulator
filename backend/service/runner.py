"""
CEP Simulator — CLI runner.

Usage
-----
    python -m backend.service.runner backend/configs/cep_sim_config_uk.toml
    python -m backend.service.runner backend/configs/cep_sim_config_brazil.toml

The config TOML must contain an [ad] section (AdConfig).  All outputs are
written to config.output.outputs_dir and a typed run_manifest.json is produced.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def run(config_path: str) -> dict:
    from backend.schemas.config import load_cep_sim_config
    from backend.service.ad_engine import Ad, apply_ad_to_population, save_episodic_events
    from backend.service.ontology_builder import build_ontology, save_ontology
    from backend.service.load_data import load_survey
    from backend.service.output_builder import generate_standard_outputs
    from backend.service.recall_engine import get_scenarios, _resolve_cep_ids
    from backend.service.reshape_survey import reshape_wide_to_long, save_long_survey
    from backend.service.respondent_builder import (
        build_respondents, build_respondent_brand_cep, save_respondent_tables,
    )
    from backend.service.validator import (
        build_segment_summary, run_ad_impact, run_scenario_recall,
    )

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[4]  # repo root

    config = load_cep_sim_config(str(project_root / config_path)
                                 if not Path(config_path).is_absolute()
                                 else config_path)

    # Resolve relative paths from project root
    config.survey.zip_path      = str(project_root / config.survey.zip_path)
    config.output.outputs_dir   = str(project_root / config.output.outputs_dir)
    config.output.processed_dir = str(project_root / config.output.processed_dir)

    if config.ad is None:
        raise ValueError(
            "Config is missing an [ad] section. "
            "Add brand_id, brand_name, focal_scenario, and focal_cep_keywords."
        )

    country = config.survey.country
    logger.info("CEP sim run | country=%s | config=%s", country, config_path)

    # ------------------------------------------------------------------
    # 2. Load & reshape survey
    # ------------------------------------------------------------------
    logger.info("Loading survey data")
    df = load_survey(config)

    logger.info("Reshaping wide → long")
    long_df = reshape_wide_to_long(df, config)
    save_long_survey(long_df, config)
    logger.info("Long table: %d rows, %d unique CEPs",
                len(long_df), long_df["cep_description"].nunique())

    # ------------------------------------------------------------------
    # 3. CEP ontology
    # ------------------------------------------------------------------
    logger.info("Building CEP ontology")
    cep_master_df, raw_map_df = build_ontology(long_df, config)
    save_ontology(cep_master_df, raw_map_df, config)
    logger.info("Ontology: %d CEPs", len(cep_master_df))

    # ------------------------------------------------------------------
    # 4. Respondent memory tables
    # ------------------------------------------------------------------
    logger.info("Building respondent memory tables")
    respondents_df  = build_respondents(df, config)
    rbc_df          = build_respondent_brand_cep(long_df, raw_map_df, config)
    save_respondent_tables(respondents_df, rbc_df, config)

    brand_name_map  = (
        rbc_df[["brand_id", "brand_name"]]
        .drop_duplicates()
        .set_index("brand_id")["brand_name"]
        .to_dict()
    )
    respondent_ids  = respondents_df["respondent_id"].astype(str).tolist()
    logger.info("Respondents: %d | Memory edges: %d", len(respondents_df), len(rbc_df))

    # ------------------------------------------------------------------
    # 5. Baseline recall — all respondents × all scenarios
    # ------------------------------------------------------------------
    logger.info("Running scenario recall")
    scenarios = get_scenarios(country)
    scenario_recall_df = run_scenario_recall(
        respondent_ids, scenarios, rbc_df, cep_master_df, brand_name_map, config,
    )
    logger.info("Scenario recall: %d rows", len(scenario_recall_df))

    # ------------------------------------------------------------------
    # 6. Build and apply ad
    # ------------------------------------------------------------------
    ad_cfg = config.ad

    focal_cep_ids     = _resolve_cep_ids(ad_cfg.focal_cep_keywords, cep_master_df)
    secondary_cep_ids = _resolve_cep_ids(ad_cfg.secondary_cep_keywords, cep_master_df)

    if not focal_cep_ids:
        raise ValueError(
            f"No CEPs matched focal_cep_keywords {ad_cfg.focal_cep_keywords!r}. "
            "Check the keywords against cep_description values in cep_master.csv."
        )

    ad = Ad(
        ad_id=ad_cfg.ad_id,
        brand_id=ad_cfg.brand_id,
        brand_name=ad_cfg.brand_name,
        focal_ceps=focal_cep_ids,
        secondary_ceps=secondary_cep_ids,
        branding_clarity=ad_cfg.branding_clarity,
        attention_weight=ad_cfg.attention_weight,
        channel=ad_cfg.channel,
        emotion=ad_cfg.emotion,
    )

    logger.info(
        "Applying ad | brand=%s | focal_ceps=%s | secondary_ceps=%s",
        ad.brand_name, focal_cep_ids, secondary_cep_ids,
    )
    rbc_post, events = apply_ad_to_population(respondent_ids, ad, rbc_df, config)
    save_episodic_events(events, config)
    logger.info("Ad applied to %d respondents", len(events))

    # ------------------------------------------------------------------
    # 7. Ad impact + segment breakdown
    # ------------------------------------------------------------------
    logger.info("Running ad impact")
    ad_impact_df = run_ad_impact(
        respondent_ids, scenarios, rbc_df, rbc_post, cep_master_df, brand_name_map, config,
    )

    segment_summary_df = build_segment_summary(ad_impact_df, respondents_df)
    logger.info("Ad impact: %d rows | Segment summary: %d rows",
                len(ad_impact_df), len(segment_summary_df))

    # ------------------------------------------------------------------
    # 8. Standard outputs + manifest
    # ------------------------------------------------------------------
    logger.info("Generating standard outputs")
    saved = generate_standard_outputs(
        rbc_pre=rbc_df,
        rbc_post=rbc_post,
        impact_df=ad_impact_df,
        cep_master_df=cep_master_df,
        long_df=long_df,
        scenario_recall_df=scenario_recall_df,
        focal_brand_id=ad_cfg.brand_id,
        focal_brand_name=ad_cfg.brand_name,
        focal_scenario=ad_cfg.focal_scenario,
        config=config,
        segment_summary_df=segment_summary_df,
    )

    logger.info(
        "Run complete | %d files written to %s",
        len(saved), config.output.outputs_dir,
    )
    return {
        "status":       "success",
        "country":      country,
        "outputs_dir":  config.output.outputs_dir,
        "files_written": {k: str(v) for k, v in saved.items()},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.service.runner <config_path>")
        print("  e.g. python -m backend.service.runner backend/configs/cep_sim_config_uk.toml")
        sys.exit(1)

    config_path = sys.argv[1]

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_stem = Path(config_path).stem
    log_file    = log_dir / f"cep_sim_{config_stem}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger.info("Writing logs to %s", log_file)

    result = run(config_path)
    logger.info("Final result: %s", result)
