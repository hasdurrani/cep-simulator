"""Tests for backend/service/validator.py using synthetic fixtures."""
import pytest
import pandas as pd
import numpy as np
from backend.schemas.config import (
    CepSimConfig, DefaultsConfig, SurveyConfig, OutputConfig, RecallConfig,
)
from backend.service.validator import (
    run_scenario_recall, run_ad_impact, run_calibration_check,
)


@pytest.fixture
def config():
    return CepSimConfig(
        survey=SurveyConfig(
            zip_path="", data_file="", codebook_file="",
            respondent_id_column="CID", country="UK",
            recall=RecallConfig(cep_blocks=["Q10"]),
        ),
        output=OutputConfig(),
    )


@pytest.fixture
def cep_master_df():
    return pd.DataFrame([
        {"cep_id": "cep_001", "cep_label": "watching_sport", "cep_description": "Watching sport at home", "cep_family": "Sport"},
        {"cep_id": "cep_002", "cep_label": "trendy_bar",     "cep_description": "At a trendy bar",        "cep_family": "Social"},
    ])


@pytest.fixture
def rbc_df():
    rows = []
    for rid in ["r1", "r2", "r3"]:
        rows += [
            {"respondent_id": rid, "brand_id": "brand_a", "brand_name": "Brand A", "cep_id": "cep_001", "assoc_strength": 1.0, "source": "survey"},
            {"respondent_id": rid, "brand_id": "brand_b", "brand_name": "Brand B", "cep_id": "cep_001", "assoc_strength": 0.5, "source": "survey"},
            {"respondent_id": rid, "brand_id": "brand_a", "brand_name": "Brand A", "cep_id": "cep_002", "assoc_strength": 0.5, "source": "survey"},
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def brand_name_map():
    return {"brand_a": "Brand A", "brand_b": "Brand B"}


@pytest.fixture
def scenarios():
    return [
        {"scenario_name": "sport", "active_ceps": ["watching_sport"], "context": {}},
        {"scenario_name": "bar",   "active_ceps": ["trendy_bar"],     "context": {}},
    ]


def test_run_scenario_recall_probs_sum_to_one(config, cep_master_df, rbc_df, brand_name_map, scenarios):
    """Probabilities per respondent per scenario sum to 1.0 (within 1e-6)."""
    result = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
    )
    sums = result.groupby(["respondent_id", "scenario_name"])["recall_prob"].sum()
    for val in sums:
        assert abs(val - 1.0) < 1e-6, f"Probabilities sum to {val}, expected 1.0"


def test_run_scenario_recall_all_positive(config, cep_master_df, rbc_df, brand_name_map, scenarios):
    """All recall_prob values are >= 0."""
    result = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
    )
    assert (result["recall_prob"] >= 0).all()


def test_run_scenario_recall_correct_scenarios(config, cep_master_df, rbc_df, brand_name_map, scenarios):
    """Result has the expected scenario names."""
    result = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
    )
    scenario_names = set(result["scenario_name"].unique())
    assert "sport" in scenario_names
    assert "bar" in scenario_names


def test_run_scenario_recall_brand_priors_shift_probs(config, cep_master_df, rbc_df, brand_name_map, scenarios):
    """
    Passing brand_priors={"brand_a": 0.8, "brand_b": 0.1} makes brand_a have higher
    mean recall than with reversed priors.
    """
    result_a = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
        brand_priors={"brand_a": 0.8, "brand_b": 0.1},
    )
    result_b = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
        brand_priors={"brand_a": 0.1, "brand_b": 0.8},
    )
    mean_a_high = result_a[result_a["brand_id"] == "brand_a"]["recall_prob"].mean()
    mean_a_low  = result_b[result_b["brand_id"] == "brand_a"]["recall_prob"].mean()
    mean_b_high = result_b[result_b["brand_id"] == "brand_b"]["recall_prob"].mean()
    mean_b_low  = result_a[result_a["brand_id"] == "brand_b"]["recall_prob"].mean()

    assert mean_a_high > mean_a_low, "High prior for brand_a should increase its recall"
    assert mean_b_high > mean_b_low, "High prior for brand_b should increase its recall"


def test_run_scenario_recall_brand_similarity_affects_competition(
    config, cep_master_df, rbc_df, brand_name_map, scenarios
):
    """
    When brand_similarity has max similarity between brand_a and brand_b,
    competition column should be non-zero and different from the zero-similarity case.
    """
    result_sim = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
        brand_similarity={
            ("brand_a", "brand_b"): 1.0,
            ("brand_b", "brand_a"): 1.0,
            ("brand_a", "brand_a"): 0.0,
            ("brand_b", "brand_b"): 0.0,
        },
    )
    result_nosim = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
        brand_similarity={
            ("brand_a", "brand_b"): 0.0,
            ("brand_b", "brand_a"): 0.0,
        },
    )
    # With high similarity, recall_scores differ from zero-similarity case
    # (probabilities differ when competition weight changes)
    sport_sim   = result_sim[result_sim["scenario_name"] == "sport"].sort_values(["respondent_id", "brand_id"])["recall_prob"].values
    sport_nosim = result_nosim[result_nosim["scenario_name"] == "sport"].sort_values(["respondent_id", "brand_id"])["recall_prob"].values
    # They should differ (competition penalty is non-zero vs zero)
    assert not np.allclose(sport_sim, sport_nosim), "Similarity should affect recall probabilities"


def test_run_ad_impact_focal_brand_gains(config, cep_master_df, rbc_df, brand_name_map, scenarios):
    """
    Create rbc_post by adding 0.5 to brand_a's assoc_strength for cep_001,
    verify brand_a's mean delta > 0.
    """
    rbc_post = rbc_df.copy()
    mask = (rbc_post["brand_id"] == "brand_a") & (rbc_post["cep_id"] == "cep_001")
    rbc_post.loc[mask, "assoc_strength"] += 0.5

    result = run_ad_impact(
        ["r1", "r2", "r3"], scenarios, rbc_df, rbc_post,
        cep_master_df, brand_name_map, config,
    )
    brand_a_delta = result[result["brand_id"] == "brand_a"]["delta"].mean()
    assert brand_a_delta > 0, f"Expected brand_a mean delta > 0, got {brand_a_delta}"


def test_run_ad_impact_delta_shape(config, cep_master_df, rbc_df, brand_name_map, scenarios):
    """Result has expected columns."""
    rbc_post = rbc_df.copy()
    result = run_ad_impact(
        ["r1", "r2", "r3"], scenarios, rbc_df, rbc_post,
        cep_master_df, brand_name_map, config,
    )
    expected_cols = {
        "scenario_name", "brand_id", "brand_name",
        "recall_pre", "recall_post", "delta", "rank_pre", "rank_post",
    }
    assert expected_cols.issubset(set(result.columns)), (
        f"Missing columns: {expected_cols - set(result.columns)}"
    )


def test_run_calibration_check_mae_positive(config, cep_master_df, rbc_df, brand_name_map, scenarios):
    """
    Create a minimal scenario_recall_df and long_df,
    run run_calibration_check, verify cal_df.attrs['mae'] >= 0.
    """
    # Build scenario_recall_df via run_scenario_recall
    scenario_recall_df = run_scenario_recall(
        ["r1", "r2", "r3"], scenarios, rbc_df, cep_master_df, brand_name_map, config,
    )

    # Build minimal long_df: respondent_id, brand_name, cep_id, mentioned, cep_description
    rows = []
    for rid in ["r1", "r2", "r3"]:
        rows += [
            {"respondent_id": rid, "brand_name": "Brand A", "cep_id": "cep_001",
             "cep_description": "Watching sport at home", "mentioned": 1},
            {"respondent_id": rid, "brand_name": "Brand B", "cep_id": "cep_001",
             "cep_description": "Watching sport at home", "mentioned": 0},
        ]
    long_df = pd.DataFrame(rows)

    cal_df = run_calibration_check(scenario_recall_df, long_df)
    assert "mae" in cal_df.attrs
    assert cal_df.attrs["mae"] >= 0
