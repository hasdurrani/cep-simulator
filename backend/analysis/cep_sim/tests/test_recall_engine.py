"""
Tests for recall_engine.py

Uses a small synthetic fixture (3 respondents, 3 brands, 2 CEPs) so tests
run without loading the real survey data.
"""
import pytest
import pandas as pd

from backend.analysis.cep_sim.service.recall_engine import (
    get_recall_scores,
    get_recall_probs,
    rank_brands,
    _resolve_cep_ids,
)
from backend.analysis.cep_sim.service.utils import softmax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cep_master_df():
    return pd.DataFrame([
        {"cep_id": "cep_001", "cep_label": "Trendy Bar", "cep_description": "bar da moda", "cep_family": "on_premise", "active": True},
        {"cep_id": "cep_002", "cep_label": "Outdoor", "cep_description": "ao ar livre em um dia quente", "cep_family": "outdoor", "active": True},
    ])


@pytest.fixture
def rbc_df():
    """
    Respondent-brand-CEP edges.
    R1 mentions Heineken and Brahma at cep_001; only Heineken at cep_002.
    R2 mentions only Brahma at cep_001.
    R3 mentions nothing.
    """
    return pd.DataFrame([
        {"respondent_id": "R1", "brand_id": "brand_heineken", "brand_name": "Heineken",  "cep_id": "cep_001", "assoc_strength": 1.0},
        {"respondent_id": "R1", "brand_id": "brand_brahma",   "brand_name": "Brahma",    "cep_id": "cep_001", "assoc_strength": 1.0},
        {"respondent_id": "R1", "brand_id": "brand_heineken", "brand_name": "Heineken",  "cep_id": "cep_002", "assoc_strength": 1.0},
        {"respondent_id": "R2", "brand_id": "brand_brahma",   "brand_name": "Brahma",    "cep_id": "cep_001", "assoc_strength": 1.0},
    ])


@pytest.fixture
def config(tmp_path):
    from backend.analysis.cep_sim.schemas.config import DefaultsConfig, CepSimConfig, SurveyConfig, RecallConfig
    # Minimal config with real defaults
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
    from backend.analysis.cep_sim.schemas.config import OutputConfig
    return CepSimConfig(survey=survey, defaults=defaults, output=OutputConfig())


# ---------------------------------------------------------------------------
# _resolve_cep_ids
# ---------------------------------------------------------------------------

def test_resolve_by_exact_id(cep_master_df):
    result = _resolve_cep_ids(["cep_001"], cep_master_df)
    assert result == ["cep_001"]


def test_resolve_by_description_substring(cep_master_df):
    result = _resolve_cep_ids(["bar da moda"], cep_master_df)
    assert "cep_001" in result


def test_resolve_by_label_substring(cep_master_df):
    result = _resolve_cep_ids(["outdoor"], cep_master_df)
    assert "cep_002" in result


def test_resolve_unknown_returns_empty(cep_master_df):
    result = _resolve_cep_ids(["completely_unknown_xyz"], cep_master_df)
    assert result == []


def test_resolve_deduplicates(cep_master_df):
    # "bar" appears in cep_001 description; passing it twice should still give one result
    result = _resolve_cep_ids(["bar da moda", "bar da moda"], cep_master_df)
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# get_recall_scores
# ---------------------------------------------------------------------------

def test_scores_return_dict_keyed_by_brand(rbc_df, cep_master_df, config):
    scores = get_recall_scores("R1", ["bar da moda"], rbc_df, cep_master_df, config)
    assert isinstance(scores, dict)
    assert "brand_heineken" in scores
    assert "brand_brahma" in scores


def test_scores_nonzero_for_mentioner(rbc_df, cep_master_df, config):
    scores = get_recall_scores("R1", ["bar da moda"], rbc_df, cep_master_df, config)
    # R1 mentioned both brands at cep_001 — both should have positive semantic
    assert scores["brand_heineken"] > 0
    assert scores["brand_brahma"] > 0


def test_scores_base_prior_applies_to_all(rbc_df, cep_master_df, config):
    # R3 mentioned nothing — all semantics = 0, so other_semantic_sum = 0,
    # competition = 0, and every brand scores exactly base.
    scores = get_recall_scores("R3", ["bar da moda"], rbc_df, cep_master_df, config)
    beta = config.defaults.base_usage_default
    for brand_id, score in scores.items():
        assert abs(score - beta) < 1e-9, f"{brand_id}: {score} != {beta}"


def test_scores_unknown_cep_returns_empty(rbc_df, cep_master_df, config):
    scores = get_recall_scores("R1", ["unknown_cep_xyz"], rbc_df, cep_master_df, config)
    assert scores == {}


def test_scores_mentioner_outscores_non_mentioner(rbc_df, cep_master_df, config):
    scores_r1 = get_recall_scores("R1", ["bar da moda"], rbc_df, cep_master_df, config)
    scores_r3 = get_recall_scores("R3", ["bar da moda"], rbc_df, cep_master_df, config)
    assert scores_r1["brand_heineken"] > scores_r3["brand_heineken"]


# ---------------------------------------------------------------------------
# get_recall_probs
# ---------------------------------------------------------------------------

def test_probs_sum_to_one(rbc_df, cep_master_df, config):
    probs = get_recall_probs("R1", ["bar da moda"], rbc_df, cep_master_df, config)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_probs_all_positive(rbc_df, cep_master_df, config):
    probs = get_recall_probs("R1", ["bar da moda"], rbc_df, cep_master_df, config)
    assert all(p > 0 for p in probs.values())


def test_probs_mentioner_leads(rbc_df, cep_master_df, config):
    # R1 mentioned Heineken at cep_001 AND cep_002 — should lead for outdoor scenario
    probs = get_recall_probs("R1", ["ao ar livre em um dia quente"], rbc_df, cep_master_df, config)
    ranked = rank_brands(probs)
    assert ranked[0][0] == "brand_heineken"


def test_probs_empty_for_unknown_cep(rbc_df, cep_master_df, config):
    probs = get_recall_probs("R1", ["unknown_xyz"], rbc_df, cep_master_df, config)
    assert probs == {}


# ---------------------------------------------------------------------------
# rank_brands
# ---------------------------------------------------------------------------

def test_rank_brands_descending():
    probs = {"a": 0.5, "b": 0.3, "c": 0.2}
    ranked = rank_brands(probs)
    assert [b for b, _ in ranked] == ["a", "b", "c"]


def test_rank_brands_empty():
    assert rank_brands({}) == []


# ---------------------------------------------------------------------------
# softmax utility
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    result = softmax({"a": 1.0, "b": 2.0, "c": 0.5})
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_softmax_preserves_ordering():
    result = softmax({"a": 3.0, "b": 1.0, "c": 2.0})
    assert result["a"] > result["c"] > result["b"]


def test_softmax_empty():
    assert softmax({}) == {}


def test_softmax_numerically_stable_large_values():
    # Should not overflow with large scores
    result = softmax({"a": 1000.0, "b": 999.0})
    assert abs(sum(result.values()) - 1.0) < 1e-9
