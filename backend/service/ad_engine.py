"""
Ad engine — applies an ad exposure to a respondent's memory and returns
updated association strengths + an episodic event record.

Update rule:
    new_strength = current_strength + (
        learning_rate * exposure_strength * branding_clarity * cep_fit
    )

Where:
    cep_fit = 1.0  for focal CEPs
    cep_fit = 0.5  for secondary CEPs
    cep_fit = 0.0  otherwise
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from backend.schemas.config import CepSimConfig
from backend.schemas.events import EpisodicEvent

logger = logging.getLogger(__name__)


@dataclass
class Ad:
    ad_id: str
    brand_id: str
    brand_name: str
    focal_ceps: list[str]       # cep_ids
    secondary_ceps: list[str]   # cep_ids
    branding_clarity: float = 0.8
    attention_weight: float = 1.0
    channel: str = "unknown"
    emotion: str = "neutral"
    exposure_strength: float = 1.0


def apply_ad(
    respondent_id: str,
    ad: Ad,
    rbc_df: pd.DataFrame,
    config: CepSimConfig,
    responsiveness_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, EpisodicEvent]:
    """
    Apply an ad exposure to a respondent.

    Returns:
        updated_rbc_df : copy of rbc_df with updated/new association strengths
        event          : EpisodicEvent recording the exposure
    """
    lr = config.defaults.learning_rate
    if responsiveness_map is not None:
        lr = lr * responsiveness_map.get(respondent_id, 1.0)
    updated = rbc_df.copy()

    all_ceps = (
        [(cep_id, 1.0) for cep_id in ad.focal_ceps]
        + [(cep_id, 0.5) for cep_id in ad.secondary_ceps]
    )

    for cep_id, cep_fit in all_ceps:
        base_lift = lr * ad.exposure_strength * ad.branding_clarity * cep_fit
        w_max = config.defaults.w_max

        mask = (
            (updated["respondent_id"] == respondent_id)
            & (updated["brand_id"] == ad.brand_id)
            & (updated["cep_id"] == cep_id)
        )

        if updated[mask].empty:
            # New edge — apply new-edge friction (harder to form than reinforce)
            effective_lift = base_lift * config.defaults.new_edge_weight
            new_row = pd.DataFrame([{
                "respondent_id": respondent_id,
                "brand_id":      ad.brand_id,
                "brand_name":    ad.brand_name,
                "cep_id":        cep_id,
                "assoc_strength": effective_lift,
                "source":        "ad_exposure",
                "confidence":    ad.branding_clarity,
                "last_updated":  datetime.now(timezone.utc).isoformat(),
            }])
            updated = pd.concat([updated, new_row], ignore_index=True)
        else:
            current_w = float(updated.loc[mask, "assoc_strength"].iloc[0])
            # Saturation: diminishing returns as strength approaches w_max
            saturation_factor = max(0.0, 1.0 - current_w / w_max)
            effective_lift = base_lift * saturation_factor
            updated.loc[mask, "assoc_strength"] = current_w + effective_lift
            updated.loc[mask, "last_updated"] = datetime.now(timezone.utc).isoformat()

    event = EpisodicEvent(
        event_id=str(uuid.uuid4()),
        respondent_id=respondent_id,
        event_type="ad_exposure",
        brand_id=ad.brand_id,
        cep_id=ad.focal_ceps[0] if ad.focal_ceps else None,
        event_time=datetime.now(timezone.utc).isoformat(),
        context_json={
            "ad_id":    ad.ad_id,
            "channel":  ad.channel,
            "emotion":  ad.emotion,
            "focal_ceps":     ad.focal_ceps,
            "secondary_ceps": ad.secondary_ceps,
        },
        strength=ad.exposure_strength * ad.branding_clarity * ad.attention_weight,
        source="ad_engine",
    )

    logger.debug(
        "Applied ad %s to respondent %s (brand=%s, lift=%.4f per focal CEP)",
        ad.ad_id, respondent_id, ad.brand_id,
        lr * ad.exposure_strength * ad.branding_clarity,
    )
    return updated, event


def apply_ad_to_population(
    respondent_ids: list[str],
    ad: Ad,
    rbc_df: pd.DataFrame,
    config: CepSimConfig,
    responsiveness_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[EpisodicEvent]]:
    """Apply an ad to the full population — vectorized (single DataFrame pass)."""
    from datetime import datetime, timezone

    lr_base    = config.defaults.learning_rate
    w_max      = config.defaults.w_max
    new_edge_w = config.defaults.new_edge_weight
    now        = datetime.now(timezone.utc).isoformat()
    rid_set    = {str(r) for r in respondent_ids}

    cep_fit_map = {c: 1.0 for c in ad.focal_ceps}
    cep_fit_map.update({c: 0.5 for c in ad.secondary_ceps})
    all_cep_ids = list(cep_fit_map)

    updated = rbc_df.copy()
    updated["respondent_id"] = updated["respondent_id"].astype(str)

    # ── 1. Update existing edges ──────────────────────────────────────
    mask = (
        updated["respondent_id"].isin(rid_set)
        & (updated["brand_id"] == ad.brand_id)
        & (updated["cep_id"].isin(all_cep_ids))
    )
    if mask.any():
        idx = updated.index[mask]
        cep_fit = updated.loc[idx, "cep_id"].map(cep_fit_map)
        if responsiveness_map:
            lr = updated.loc[idx, "respondent_id"].map(
                lambda r: lr_base * responsiveness_map.get(r, 1.0)
            )
        else:
            lr = lr_base
        base_lift  = lr * ad.exposure_strength * ad.branding_clarity * cep_fit
        saturation = (1.0 - updated.loc[idx, "assoc_strength"] / w_max).clip(lower=0.0)
        updated.loc[idx, "assoc_strength"] += (base_lift * saturation).values
        updated.loc[idx, "last_updated"]    = now

    # ── 2. Create missing edges ───────────────────────────────────────
    existing_keys = (
        set(zip(updated.loc[mask, "respondent_id"], updated.loc[mask, "cep_id"]))
        if mask.any() else set()
    )
    missing_rows = [
        (rid, cep_id)
        for rid in rid_set
        for cep_id in all_cep_ids
        if (rid, cep_id) not in existing_keys
    ]
    if missing_rows:
        miss = pd.DataFrame(missing_rows, columns=["respondent_id", "cep_id"])
        cep_fit_s = miss["cep_id"].map(cep_fit_map)
        lr_s = (
            miss["respondent_id"].map(lambda r: lr_base * responsiveness_map.get(r, 1.0))
            if responsiveness_map else lr_base
        )
        miss["assoc_strength"] = (
            lr_s * ad.exposure_strength * ad.branding_clarity * cep_fit_s * new_edge_w
        )
        miss["brand_id"]     = ad.brand_id
        miss["brand_name"]   = ad.brand_name
        miss["source"]       = "ad_exposure"
        miss["confidence"]   = ad.branding_clarity
        miss["last_updated"] = now
        updated = pd.concat([updated, miss], ignore_index=True)

    # ── 3. Episodic events (one per respondent) ───────────────────────
    events = [
        EpisodicEvent(
            event_id=str(uuid.uuid4()),
            respondent_id=rid,
            event_type="ad_exposure",
            brand_id=ad.brand_id,
            cep_id=ad.focal_ceps[0] if ad.focal_ceps else None,
            event_time=now,
            context_json={
                "ad_id":          ad.ad_id,
                "channel":        ad.channel,
                "emotion":        ad.emotion,
                "focal_ceps":     ad.focal_ceps,
                "secondary_ceps": ad.secondary_ceps,
            },
            strength=ad.exposure_strength * ad.branding_clarity * ad.attention_weight,
            source="ad_engine",
        )
        for rid in respondent_ids
    ]

    logger.info(
        "Applied ad %s to %d respondents (vectorized)%s",
        ad.ad_id, len(respondent_ids),
        " (with responsiveness)" if responsiveness_map else "",
    )
    return updated, events


def save_episodic_events(events: list[EpisodicEvent], config: CepSimConfig) -> None:
    if not events:
        return
    out_dir = __import__("pathlib").Path(config.output.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "episodic_events.csv"
    rows = [e.model_dump() for e in events]
    # Flatten context_json to a string
    for r in rows:
        r["context_json"] = str(r["context_json"])
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved episodic_events: %s (%d rows)", path, len(rows))
