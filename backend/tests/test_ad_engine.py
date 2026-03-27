"""
Tests for ad_engine.py

Uses the same synthetic fixture pattern as test_recall_engine.py.
"""
import pytest
import pandas as pd

from backend.service.ad_engine import Ad, apply_ad, apply_ad_to_population
from backend.schemas.events import EpisodicEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rbc_df():
    return pd.DataFrame([
        {"respondent_id": "R1", "brand_id": "brand_heineken", "brand_name": "Heineken",
         "cep_id": "cep_001", "assoc_strength": 1.0, "source": "survey",
         "confidence": 1.0, "last_updated": "2025-01-01"},
        {"respondent_id": "R1", "brand_id": "brand_brahma", "brand_name": "Brahma",
         "cep_id": "cep_001", "assoc_strength": 0.5, "source": "survey",
         "confidence": 1.0, "last_updated": "2025-01-01"},
        {"respondent_id": "R2", "brand_id": "brand_brahma", "brand_name": "Brahma",
         "cep_id": "cep_001", "assoc_strength": 1.0, "source": "survey",
         "confidence": 1.0, "last_updated": "2025-01-01"},
    ])


@pytest.fixture
def config():
    from backend.schemas.config import (
        CepSimConfig, SurveyConfig, RecallConfig, DefaultsConfig, OutputConfig
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


@pytest.fixture
def heineken_ad():
    return Ad(
        ad_id="test_ad_001",
        brand_id="brand_heineken",
        brand_name="Heineken",
        focal_ceps=["cep_001"],
        secondary_ceps=["cep_002"],
        branding_clarity=0.9,
        attention_weight=1.0,
        channel="social_media",
        emotion="confidence",
        exposure_strength=1.0,
    )


# ---------------------------------------------------------------------------
# apply_ad — weight updates
# ---------------------------------------------------------------------------

def test_apply_ad_returns_updated_df_and_event(rbc_df, heineken_ad, config):
    updated, event = apply_ad("R1", heineken_ad, rbc_df, config)
    assert isinstance(updated, pd.DataFrame)
    assert isinstance(event, EpisodicEvent)


def test_apply_ad_increases_focal_cep_strength(rbc_df, heineken_ad, config):
    original_strength = rbc_df.loc[
        (rbc_df["respondent_id"] == "R1") & (rbc_df["brand_id"] == "brand_heineken") & (rbc_df["cep_id"] == "cep_001"),
        "assoc_strength"
    ].values[0]

    updated, _ = apply_ad("R1", heineken_ad, rbc_df, config)

    new_strength = updated.loc[
        (updated["respondent_id"] == "R1") & (updated["brand_id"] == "brand_heineken") & (updated["cep_id"] == "cep_001"),
        "assoc_strength"
    ].values[0]

    assert new_strength > original_strength


def test_apply_ad_focal_lift_equals_formula(rbc_df, heineken_ad, config):
    lr = config.defaults.learning_rate  # 0.1
    base_lift = lr * heineken_ad.exposure_strength * heineken_ad.branding_clarity * 1.0  # cep_fit=1.0

    original = rbc_df.loc[
        (rbc_df["respondent_id"] == "R1") & (rbc_df["brand_id"] == "brand_heineken") & (rbc_df["cep_id"] == "cep_001"),
        "assoc_strength"
    ].values[0]

    # Saturation: effective_lift = base_lift * (1 - current_w / w_max)
    w_max = config.defaults.w_max
    saturation_factor = max(0.0, 1.0 - original / w_max)
    expected_lift = base_lift * saturation_factor

    updated, _ = apply_ad("R1", heineken_ad, rbc_df, config)

    new_strength = updated.loc[
        (updated["respondent_id"] == "R1") & (updated["brand_id"] == "brand_heineken") & (updated["cep_id"] == "cep_001"),
        "assoc_strength"
    ].values[0]

    assert abs((new_strength - original) - expected_lift) < 1e-9


def test_apply_ad_secondary_lift_is_half_focal(rbc_df, heineken_ad, config):
    lr = config.defaults.learning_rate
    # New edge — friction applies: effective = base_lift * new_edge_weight
    new_edge_weight = config.defaults.new_edge_weight
    focal_base_lift = lr * heineken_ad.exposure_strength * heineken_ad.branding_clarity * 1.0
    secondary_base_lift = lr * heineken_ad.exposure_strength * heineken_ad.branding_clarity * 0.5
    # Both are new edges → multiply by new_edge_weight
    secondary_effective = secondary_base_lift * new_edge_weight

    updated, _ = apply_ad("R1", heineken_ad, rbc_df, config)

    # Secondary cep_002 didn't exist for R1 before — new row created
    new_row = updated.loc[
        (updated["respondent_id"] == "R1") & (updated["brand_id"] == "brand_heineken") & (updated["cep_id"] == "cep_002"),
        "assoc_strength"
    ]
    assert len(new_row) == 1
    assert abs(new_row.values[0] - secondary_effective) < 1e-9
    # Secondary base lift is still half of focal base lift
    assert abs(secondary_base_lift - focal_base_lift / 2) < 1e-9


def test_apply_ad_creates_new_edge_when_none_exists(rbc_df, heineken_ad, config):
    # R1 has no edge for cep_002 — should be created
    before = rbc_df[
        (rbc_df["respondent_id"] == "R1") & (rbc_df["cep_id"] == "cep_002")
    ]
    assert len(before) == 0

    updated, _ = apply_ad("R1", heineken_ad, rbc_df, config)

    after = updated[
        (updated["respondent_id"] == "R1") & (updated["brand_id"] == "brand_heineken") & (updated["cep_id"] == "cep_002")
    ]
    assert len(after) == 1


def test_apply_ad_does_not_modify_other_respondent(rbc_df, heineken_ad, config):
    updated, _ = apply_ad("R1", heineken_ad, rbc_df, config)

    r2_before = rbc_df[rbc_df["respondent_id"] == "R2"]["assoc_strength"].values
    r2_after = updated[updated["respondent_id"] == "R2"]["assoc_strength"].values

    assert list(r2_before) == list(r2_after)


def test_apply_ad_does_not_affect_other_brand(rbc_df, heineken_ad, config):
    brahma_before = rbc_df.loc[
        (rbc_df["respondent_id"] == "R1") & (rbc_df["brand_id"] == "brand_brahma"),
        "assoc_strength"
    ].values[0]

    updated, _ = apply_ad("R1", heineken_ad, rbc_df, config)

    brahma_after = updated.loc[
        (updated["respondent_id"] == "R1") & (updated["brand_id"] == "brand_brahma"),
        "assoc_strength"
    ].values[0]

    assert brahma_before == brahma_after


def test_apply_ad_original_df_not_mutated(rbc_df, heineken_ad, config):
    original_copy = rbc_df.copy()
    apply_ad("R1", heineken_ad, rbc_df, config)
    pd.testing.assert_frame_equal(rbc_df, original_copy)


# ---------------------------------------------------------------------------
# apply_ad — episodic event
# ---------------------------------------------------------------------------

def test_event_brand_and_respondent_correct(rbc_df, heineken_ad, config):
    _, event = apply_ad("R1", heineken_ad, rbc_df, config)
    assert event.respondent_id == "R1"
    assert event.brand_id == "brand_heineken"


def test_event_type_is_ad_exposure(rbc_df, heineken_ad, config):
    _, event = apply_ad("R1", heineken_ad, rbc_df, config)
    assert event.event_type == "ad_exposure"


def test_event_strength_formula(rbc_df, heineken_ad, config):
    _, event = apply_ad("R1", heineken_ad, rbc_df, config)
    expected = heineken_ad.exposure_strength * heineken_ad.branding_clarity * heineken_ad.attention_weight
    assert abs(event.strength - expected) < 1e-9


def test_event_focal_cep_recorded(rbc_df, heineken_ad, config):
    _, event = apply_ad("R1", heineken_ad, rbc_df, config)
    assert event.cep_id == heineken_ad.focal_ceps[0]


def test_event_has_unique_id(rbc_df, heineken_ad, config):
    _, event1 = apply_ad("R1", heineken_ad, rbc_df, config)
    _, event2 = apply_ad("R1", heineken_ad, rbc_df, config)
    assert event1.event_id != event2.event_id


# ---------------------------------------------------------------------------
# apply_ad_to_population
# ---------------------------------------------------------------------------

def test_population_returns_one_event_per_respondent(rbc_df, heineken_ad, config):
    respondent_ids = ["R1", "R2", "R3"]
    _, events = apply_ad_to_population(respondent_ids, heineken_ad, rbc_df, config)
    assert len(events) == len(respondent_ids)


def test_population_updated_df_has_new_edges(rbc_df, heineken_ad, config):
    respondent_ids = ["R1", "R2"]
    updated, _ = apply_ad_to_population(respondent_ids, heineken_ad, rbc_df, config)
    # Both R1 and R2 should now have a Heineken × cep_002 edge
    for rid in respondent_ids:
        row = updated[
            (updated["respondent_id"] == rid)
            & (updated["brand_id"] == "brand_heineken")
            & (updated["cep_id"] == "cep_002")
        ]
        assert len(row) == 1, f"R{rid} missing secondary CEP edge"


def test_population_cumulative_updates(rbc_df, heineken_ad, config):
    # Applying to R1 and R2 in sequence — R1's focal edge should be updated once
    respondent_ids = ["R1", "R2"]
    updated, _ = apply_ad_to_population(respondent_ids, heineken_ad, rbc_df, config)

    lr = config.defaults.learning_rate
    base_lift = lr * heineken_ad.exposure_strength * heineken_ad.branding_clarity * 1.0

    original = rbc_df.loc[
        (rbc_df["respondent_id"] == "R1") & (rbc_df["brand_id"] == "brand_heineken") & (rbc_df["cep_id"] == "cep_001"),
        "assoc_strength"
    ].values[0]

    # Saturation: effective_lift = base_lift * (1 - current_w / w_max)
    w_max = config.defaults.w_max
    saturation_factor = max(0.0, 1.0 - original / w_max)
    expected_lift = base_lift * saturation_factor

    new_strength = updated.loc[
        (updated["respondent_id"] == "R1") & (updated["brand_id"] == "brand_heineken") & (updated["cep_id"] == "cep_001"),
        "assoc_strength"
    ].values[0]

    assert abs((new_strength - original) - expected_lift) < 1e-9
