"""
Reshape Dynata survey from wide coded format to long respondent-brand-CEP format.

Uses the codebook column map to identify which columns are CEP recall variables,
then melts into rows: respondent_id | cep_block | cep_description | brand_name | cep_raw | mentioned
"""
import logging
from pathlib import Path

import pandas as pd

from backend.analysis.cep_sim.schemas.config import CepSimConfig
from backend.analysis.cep_sim.service.codebook_parser import parse_codebook, build_column_map

logger = logging.getLogger(__name__)


def reshape_wide_to_long(df: pd.DataFrame, config: CepSimConfig) -> pd.DataFrame:
    """
    Reshape wide coded survey to long format.

    Returns DataFrame with columns:
        respondent_id, cep_block, cep_description, cep_raw, brand_name, mentioned
    """
    codebook_df = parse_codebook(config.survey.zip_path, config.survey.codebook_file)
    col_map = build_column_map(
        codebook_df,
        config.survey.recall.cep_blocks,
        config.survey.recall.exclude_brands,
    )

    rid_col = config.survey.respondent_id_column
    records = []
    for col_name, meta in col_map.items():
        if col_name not in df.columns:
            logger.warning("Column %s not found in survey data", col_name)
            continue
        sub = df[[rid_col, col_name]].copy()
        sub = sub.rename(columns={col_name: "mentioned"})
        sub["cep_block"] = meta["cep_block"]
        sub["cep_description"] = meta["cep_description"]
        sub["cep_raw"] = meta["cep_description"]   # raw = description for Dynata format
        sub["brand_name"] = meta["brand_name"]
        sub["respondent_id"] = sub[rid_col]
        records.append(sub[["respondent_id", "cep_block", "cep_description", "cep_raw", "brand_name", "mentioned"]])

    long_df = pd.concat(records, ignore_index=True)
    long_df["mentioned"] = pd.to_numeric(long_df["mentioned"], errors="coerce").fillna(0).astype(int)
    logger.info(
        "Reshaped to long: %d rows, %d unique CEPs, mention rate %.1f%%",
        len(long_df),
        long_df["cep_description"].nunique(),
        long_df["mentioned"].mean() * 100,
    )
    return long_df


def save_long_survey(long_df: pd.DataFrame, config: CepSimConfig) -> Path:
    out_dir = Path(config.output.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "long_survey.csv"
    long_df.to_csv(path, index=False)
    logger.info("Saved long survey: %s (%d rows)", path, len(long_df))
    return path
