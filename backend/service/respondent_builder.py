import logging
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

from backend.schemas.config import CepSimConfig
from backend.service.utils import brand_to_id

logger = logging.getLogger(__name__)

_GENDER_MAP = {1: "male", 2: "female", 3: "non_binary"}

def _decode_age(code) -> int | None:
    """Q3 coded 1=18, 2=19, ..., 43=60, 44=61+. Returns midpoint or None."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if 1 <= c <= 43:
        return 17 + c
    if c == 44:
        return 65  # representative for 61+
    return None


def _age_band(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def build_respondents(df: pd.DataFrame, config: CepSimConfig) -> pd.DataFrame:
    """Build respondents table from raw survey DataFrame."""
    demo = config.survey.demographic_columns
    rid_col = config.survey.respondent_id_column

    base = df[[rid_col]].copy()
    base = base.rename(columns={rid_col: "respondent_id"})
    base["respondent_id"] = base["respondent_id"].astype(str)

    # Gender
    if demo.gender and demo.gender in df.columns:
        base["gender"] = df[demo.gender].map(_GENDER_MAP).fillna("unknown")
    else:
        base["gender"] = "unknown"

    # Age → age_band
    if demo.age and demo.age in df.columns:
        base["age_numeric"] = df[demo.age].apply(_decode_age)
        base["age_band"] = base["age_numeric"].apply(_age_band)
    else:
        base["age_band"] = "unknown"

    # Segment = gender × age_band
    base["segment"] = base["gender"] + "_" + base["age_band"]

    base["region"] = config.survey.country
    base["category_usage_freq"] = None
    base["weight"] = 1.0

    keep = ["respondent_id", "age_band", "gender", "region", "segment",
            "category_usage_freq", "weight"]
    result = base[keep].reset_index(drop=True)
    logger.info("Built respondents table: %d respondents", len(result))
    return result


def build_respondent_brand_cep(
    long_df: pd.DataFrame,
    raw_map_df: pd.DataFrame,
    config: CepSimConfig,
) -> pd.DataFrame:
    """
    Build the respondent-brand-CEP memory edge table.
    Only creates edges where mentioned = 1.
    """
    merged = long_df.merge(
        raw_map_df[["raw_response", "cep_id"]],
        left_on="cep_raw",
        right_on="raw_response",
        how="left",
    )

    positive = merged[merged["mentioned"] == 1].copy()

    positive["brand_id"] = positive["brand_name"].apply(brand_to_id)

    # Scale strength by 1/n_brands_mentioned at that CEP for each respondent.
    # A respondent who mentioned only one brand at a CEP encodes a stronger, more
    # discriminating link (strength=1.0) than one who mentioned every brand (strength≈0.05).
    n_mentions_at_cep = positive.groupby(["respondent_id", "cep_id"])["brand_id"].transform("count")
    positive["assoc_strength"] = config.defaults.assoc_strength_if_mentioned / n_mentions_at_cep

    positive["source"] = "survey"
    positive["confidence"] = 1.0
    positive["last_updated"] = datetime.now(timezone.utc).isoformat()

    keep = ["respondent_id", "brand_id", "brand_name", "cep_id",
            "assoc_strength", "source", "confidence", "last_updated"]

    result = positive[keep].reset_index(drop=True)
    logger.info(
        "Built respondent_brand_cep: %d edges (%d respondents, %d unique brands)",
        len(result),
        result["respondent_id"].nunique(),
        result["brand_id"].nunique(),
    )
    return result


def save_respondent_tables(
    respondents_df: pd.DataFrame,
    rbc_df: pd.DataFrame,
    config: CepSimConfig,
) -> tuple[Path, Path]:
    out_dir = Path(config.output.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r_path = out_dir / "respondents.csv"
    rbc_path = out_dir / "respondent_brand_cep.csv"

    respondents_df.to_csv(r_path, index=False)
    rbc_df.to_csv(rbc_path, index=False)

    logger.info("Saved respondents: %s", r_path)
    logger.info("Saved respondent_brand_cep: %s", rbc_path)
    return r_path, rbc_path
