"""Tests for backend/service/respondent_builder.py using synthetic fixtures."""
import pytest
import pandas as pd
from backend.schemas.config import (
    CepSimConfig, SurveyConfig, OutputConfig, DemographicColumns, RecallConfig,
)
from backend.service.respondent_builder import build_respondents, build_respondent_brand_cep


@pytest.fixture
def config():
    return CepSimConfig(
        survey=SurveyConfig(
            zip_path="", data_file="", codebook_file="",
            respondent_id_column="CID", country="UK",
            demographic_columns=DemographicColumns(gender="Q2", age="Q3"),
            recall=RecallConfig(cep_blocks=["Q10"]),
        ),
        output=OutputConfig(),
    )


@pytest.fixture
def raw_df():
    """Minimal raw survey DataFrame."""
    return pd.DataFrame([
        {"CID": "r1", "Q2": 1, "Q3": 7},   # male, age=24 → "18-24"
        {"CID": "r2", "Q2": 2, "Q3": 15},  # female, age=32 → "25-34"
        {"CID": "r3", "Q2": 1, "Q3": 25},  # male, age=42 → "35-44"
    ])


def test_build_respondents_basic(config, raw_df):
    """Returns a DataFrame with required columns."""
    result = build_respondents(raw_df, config)
    expected_cols = {"respondent_id", "age_band", "gender", "region", "segment"}
    assert expected_cols.issubset(set(result.columns))


def test_build_respondents_count(config, raw_df):
    """Output has same number of rows as input."""
    result = build_respondents(raw_df, config)
    assert len(result) == len(raw_df)


def test_build_respondent_brand_cep_only_positive():
    """Only rows where mentioned=1 become edges."""
    long_df = pd.DataFrame([
        {"respondent_id": "r1", "brand_name": "Brand A", "cep_raw": "sport", "mentioned": 1},
        {"respondent_id": "r1", "brand_name": "Brand B", "cep_raw": "sport", "mentioned": 0},
        {"respondent_id": "r2", "brand_name": "Brand A", "cep_raw": "sport", "mentioned": 0},
    ])
    raw_map_df = pd.DataFrame([
        {"raw_response": "sport", "cep_id": "cep_001"},
    ])
    config = CepSimConfig(
        survey=SurveyConfig(
            zip_path="", data_file="", codebook_file="",
            respondent_id_column="CID", country="UK",
            recall=RecallConfig(cep_blocks=["Q10"]),
        ),
        output=OutputConfig(),
    )
    result = build_respondent_brand_cep(long_df, raw_map_df, config)
    # Only Brand A by r1 should be in result
    assert len(result) == 1
    assert result.iloc[0]["brand_name"] == "Brand A"
    assert result.iloc[0]["respondent_id"] == "r1"


def test_build_respondent_brand_cep_breadth_weighting():
    """
    Breadth weighting: 1 brand at CEP → assoc_strength=1.0,
    2 brands at CEP → assoc_strength=0.5 each.
    """
    long_df = pd.DataFrame([
        # r1: only brand_a at cep_001 (n=1)
        {"respondent_id": "r1", "brand_name": "Brand A", "cep_raw": "sport", "mentioned": 1},
        # r2: brand_a AND brand_b at cep_001 (n=2)
        {"respondent_id": "r2", "brand_name": "Brand A", "cep_raw": "sport", "mentioned": 1},
        {"respondent_id": "r2", "brand_name": "Brand B", "cep_raw": "sport", "mentioned": 1},
    ])
    raw_map_df = pd.DataFrame([
        {"raw_response": "sport", "cep_id": "cep_001"},
    ])
    config = CepSimConfig(
        survey=SurveyConfig(
            zip_path="", data_file="", codebook_file="",
            respondent_id_column="CID", country="UK",
            recall=RecallConfig(cep_blocks=["Q10"]),
        ),
        output=OutputConfig(),
    )
    result = build_respondent_brand_cep(long_df, raw_map_df, config)

    r1_row = result[result["respondent_id"] == "r1"]
    assert len(r1_row) == 1
    assert abs(r1_row.iloc[0]["assoc_strength"] - 1.0) < 1e-9

    r2_rows = result[result["respondent_id"] == "r2"]
    assert len(r2_rows) == 2
    for _, row in r2_rows.iterrows():
        assert abs(row["assoc_strength"] - 0.5) < 1e-9


def test_build_respondent_brand_cep_columns():
    """Result has required columns."""
    long_df = pd.DataFrame([
        {"respondent_id": "r1", "brand_name": "Brand A", "cep_raw": "sport", "mentioned": 1},
    ])
    raw_map_df = pd.DataFrame([
        {"raw_response": "sport", "cep_id": "cep_001"},
    ])
    config = CepSimConfig(
        survey=SurveyConfig(
            zip_path="", data_file="", codebook_file="",
            respondent_id_column="CID", country="UK",
            recall=RecallConfig(cep_blocks=["Q10"]),
        ),
        output=OutputConfig(),
    )
    result = build_respondent_brand_cep(long_df, raw_map_df, config)
    expected_cols = {
        "respondent_id", "brand_id", "brand_name", "cep_id",
        "assoc_strength", "source", "confidence", "last_updated",
    }
    assert expected_cols.issubset(set(result.columns))
