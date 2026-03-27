"""
Tests for calibration.py — run_ablation and supporting functions.

Uses the same small synthetic fixture as test_recall_engine.py
(3 respondents, 2 brands, 2 CEPs) so tests run without real survey data.
"""
import math

import pandas as pd
import pytest

from backend.analysis.cep_sim.service.calibration import (
    make_holdout_split,
    compute_brand_priors,
    compute_cep_brand_priors,
    compute_respondent_responsiveness,
    run_ablation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cep_master_df():
    return pd.DataFrame([
        {"cep_id": "cep_001", "cep_label": "Trendy Bar",   "cep_description": "bar da moda",              "cep_family": "on_premise", "active": True},
        {"cep_id": "cep_002", "cep_label": "Outdoor",      "cep_description": "ao ar livre em um dia quente", "cep_family": "outdoor", "active": True},
    ])


@pytest.fixture
def rbc_df():
    return pd.DataFrame([
        {"respondent_id": "R1", "brand_id": "brand_heineken", "brand_name": "Heineken", "cep_id": "cep_001", "assoc_strength": 1.0},
        {"respondent_id": "R1", "brand_id": "brand_brahma",   "brand_name": "Brahma",   "cep_id": "cep_001", "assoc_strength": 1.0},
        {"respondent_id": "R1", "brand_id": "brand_heineken", "brand_name": "Heineken", "cep_id": "cep_002", "assoc_strength": 1.0},
        {"respondent_id": "R2", "brand_id": "brand_brahma",   "brand_name": "Brahma",   "cep_id": "cep_001", "assoc_strength": 1.0},
        {"respondent_id": "R3", "brand_id": "brand_heineken", "brand_name": "Heineken", "cep_id": "cep_001", "assoc_strength": 0.5},
    ])


_ALL_RIDS = [f"R{i}" for i in range(10)]  # 10 respondents; seed=1 gives 2 holdout, 8 train


@pytest.fixture
def long_df():
    """Long-format survey data for 10 respondents."""
    heavy_heineken = {"R0", "R2", "R4", "R6"}
    rows = []
    for rid in _ALL_RIDS:
        for brand_name in ("Heineken", "Brahma"):
            for cep_id in ("cep_001", "cep_002"):
                rows.append({
                    "respondent_id": rid,
                    "brand_name":    brand_name,
                    "cep_id":        cep_id,
                    "cep_description": "bar da moda" if cep_id == "cep_001" else "ao ar livre em um dia quente",
                    "mentioned":     1 if (rid in heavy_heineken and brand_name == "Heineken") else 0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def rbc_df_full():
    """RBC edges for all 10 respondents."""
    heavy_heineken = {"R0", "R2", "R4", "R6"}
    rows = []
    for rid in _ALL_RIDS:
        rows.append({
            "respondent_id": rid, "brand_id": "brand_heineken", "brand_name": "Heineken",
            "cep_id": "cep_001", "assoc_strength": 1.0 if rid in heavy_heineken else 0.2,
        })
        rows.append({
            "respondent_id": rid, "brand_id": "brand_brahma", "brand_name": "Brahma",
            "cep_id": "cep_001", "assoc_strength": 0.8 if rid in {"R1", "R3", "R5"} else 0.2,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def scenarios():
    return [
        {"scenario_name": "trendy_bar",    "active_ceps": ["bar da moda"],              "context": {}},
        {"scenario_name": "outdoor_hot",   "active_ceps": ["ao ar livre em um dia quente"], "context": {}},
    ]


@pytest.fixture
def config():
    from backend.analysis.cep_sim.schemas.config import (
        CepSimConfig, SurveyConfig, RecallConfig, DefaultsConfig, OutputConfig,
    )
    defaults = DefaultsConfig(
        assoc_strength_if_mentioned=1.0,
        base_usage_default=0.2,
        learning_rate=0.1,
        competition_penalty_weight=0.05,
        softmax_temperature=1.0,
    )
    survey = SurveyConfig(
        zip_path="dummy.zip",
        data_file="dummy.csv",
        codebook_file="dummy.txt",
        recall=RecallConfig(cep_blocks=["Q10"]),
    )
    return CepSimConfig(survey=survey, defaults=defaults, output=OutputConfig())


# ---------------------------------------------------------------------------
# make_holdout_split
# ---------------------------------------------------------------------------

def test_holdout_split_sizes():
    ids = [f"R{i}" for i in range(20)]
    train, holdout = make_holdout_split(ids, holdout_fraction=0.2, seed=42)
    assert len(train) + len(holdout) == 20
    # Expect roughly 20% holdout
    assert 2 <= len(holdout) <= 6


def test_holdout_split_deterministic():
    ids = [f"R{i}" for i in range(50)]
    train1, holdout1 = make_holdout_split(ids, seed=42)
    train2, holdout2 = make_holdout_split(ids, seed=42)
    assert train1 == train2
    assert holdout1 == holdout2


def test_holdout_split_no_overlap():
    ids = [f"R{i}" for i in range(30)]
    train, holdout = make_holdout_split(ids, seed=99)
    assert set(train) & set(holdout) == set()


# ---------------------------------------------------------------------------
# compute_brand_priors / compute_cep_brand_priors
# ---------------------------------------------------------------------------

def test_brand_priors_keys_are_brand_ids(long_df):
    priors = compute_brand_priors(long_df)
    assert all(k.startswith("brand_") for k in priors)


def test_brand_priors_values_in_unit_interval(long_df):
    priors = compute_brand_priors(long_df)
    assert all(0.0 <= v <= 1.0 for v in priors.values())


def test_cep_brand_priors_keys_are_tuples(long_df):
    cbp = compute_cep_brand_priors(long_df)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in cbp)


def test_cep_brand_priors_values_in_unit_interval(long_df):
    cbp = compute_cep_brand_priors(long_df)
    assert all(0.0 <= v <= 1.0 for v in cbp.values())


# ---------------------------------------------------------------------------
# compute_respondent_responsiveness
# ---------------------------------------------------------------------------

def test_responsiveness_keys_match_respondent_ids(rbc_df_full):
    resp = compute_respondent_responsiveness(rbc_df_full)
    assert set(resp.keys()) == set(rbc_df_full["respondent_id"].unique())


def test_responsiveness_clipped_to_bounds(rbc_df_full):
    resp = compute_respondent_responsiveness(rbc_df_full)
    for v in resp.values():
        assert 0.5 <= v <= 3.0


# ---------------------------------------------------------------------------
# run_ablation
# ---------------------------------------------------------------------------

_ABLATION_SEED = 1  # seed=1 gives 2 holdout, 8 train for 10 respondents


def test_ablation_returns_five_rows(long_df, rbc_df_full, cep_master_df, scenarios, config):
    result = run_ablation(long_df, rbc_df_full, cep_master_df, scenarios, config, seed=_ABLATION_SEED)
    assert len(result) == 5


def test_ablation_variant_names(long_df, rbc_df_full, cep_master_df, scenarios, config):
    result = run_ablation(long_df, rbc_df_full, cep_master_df, scenarios, config, seed=_ABLATION_SEED)
    expected = [
        "global_priors_only",
        "fitted_params",
        "saturation_friction",
        "cep_priors",
        "responsiveness",
    ]
    assert result["variant"].tolist() == expected


def test_ablation_columns(long_df, rbc_df_full, cep_master_df, scenarios, config):
    result = run_ablation(long_df, rbc_df_full, cep_master_df, scenarios, config, seed=_ABLATION_SEED)
    assert set(result.columns) >= {"variant", "holdout_mae", "median_rho", "focal_lift", "worst_3_scenarios"}


def test_ablation_mae_positive_finite(long_df, rbc_df_full, cep_master_df, scenarios, config):
    result = run_ablation(long_df, rbc_df_full, cep_master_df, scenarios, config, seed=_ABLATION_SEED)
    for mae in result["holdout_mae"]:
        assert mae > 0
        assert math.isfinite(mae)


def test_ablation_focal_lift_nan_when_no_ads(long_df, rbc_df_full, cep_master_df, scenarios, config):
    result = run_ablation(long_df, rbc_df_full, cep_master_df, scenarios, config, seed=_ABLATION_SEED)
    assert all(math.isnan(v) for v in result["focal_lift"])


def test_ablation_restores_config(long_df, rbc_df_full, cep_master_df, scenarios, config):
    orig_tau   = config.defaults.softmax_temperature
    orig_gamma = config.defaults.competition_penalty_weight
    orig_pw    = config.defaults.base_prior_weight
    orig_wmax  = config.defaults.w_max
    orig_ne    = config.defaults.new_edge_weight

    run_ablation(long_df, rbc_df_full, cep_master_df, scenarios, config, seed=_ABLATION_SEED)

    assert config.defaults.softmax_temperature        == orig_tau
    assert config.defaults.competition_penalty_weight == orig_gamma
    assert config.defaults.base_prior_weight          == orig_pw
    assert config.defaults.w_max                      == orig_wmax
    assert config.defaults.new_edge_weight            == orig_ne


def test_ablation_with_focal_lift(long_df, rbc_df_full, cep_master_df, scenarios, config):
    from backend.analysis.cep_sim.service.ad_engine import Ad
    ad = Ad(
        ad_id="test_ad",
        brand_id="brand_heineken",
        brand_name="Heineken",
        focal_ceps=["cep_001"],
        secondary_ceps=[],
        branding_clarity=0.8,
    )
    result = run_ablation(
        long_df, rbc_df_full, cep_master_df, scenarios, config,
        focal_brand_id="brand_heineken",
        focal_scenario="trendy_bar",
        ads=[ad],
        seed=_ABLATION_SEED,
    )
    # Variants with ads defined should have a finite focal_lift
    for val in result["focal_lift"]:
        assert math.isfinite(val)
    # Lift should be non-negative (ad can only reinforce or create edges)
    assert all(v >= 0 for v in result["focal_lift"])
