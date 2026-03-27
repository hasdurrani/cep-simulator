"""
Validator — runs scenarios across respondents and produces output CSVs.
Also runs basic sanity checks on the results.
"""
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backend.schemas.config import CepSimConfig
from backend.service.recall_engine import (
    get_recall_scores,
    get_recall_probs,
    rank_brands,
    _resolve_cep_ids,
)

logger = logging.getLogger(__name__)


def _softmax_col(s: pd.Series, temperature: float) -> pd.Series:
    """Numerically stable softmax over a pandas Series."""
    shifted = (s - s.max()) / temperature
    e = np.exp(shifted)
    return e / e.sum()


def run_scenario_recall(
    respondent_ids: list[str],
    scenarios: list[dict],
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    brand_name_map: dict[str, str],
    config: CepSimConfig,
    episodic_events: pd.DataFrame | None = None,
    brand_priors: dict[str, float] | None = None,
    cep_brand_priors: dict[tuple, float] | None = None,
) -> pd.DataFrame:
    """
    Run each scenario for all respondents using vectorised pandas operations.

    For each scenario:
      1. Resolve active CEP IDs.
      2. Aggregate assoc_strength per (respondent, brand) → semantic score.
      3. Cross-join with all respondent_ids so zero-semantic respondents are included.
      4. Add base prior, subtract competition (sum of other brands' semantics × penalty).
      5. Apply per-respondent softmax with configured temperature.
    """
    temperature = config.defaults.softmax_temperature
    base = config.defaults.base_usage_default
    penalty = config.defaults.competition_penalty_weight

    # Double-count guard (once per call, not per respondent)
    if episodic_events is not None and len(episodic_events) > 0 and "source" in rbc_df.columns:
        if (rbc_df["source"] == "ad_exposure").any():
            logger.warning(
                "run_scenario_recall: rbc_df contains ad_exposure edges AND episodic_events "
                "are also provided. This may double-count ad effects."
            )

    r_frame = pd.DataFrame({"respondent_id": respondent_ids})
    chunks: list[pd.DataFrame] = []

    for scenario in scenarios:
        name = scenario["scenario_name"]
        active_cep_ids = _resolve_cep_ids(scenario["active_ceps"], cep_master_df)
        if not active_cep_ids:
            logger.warning("Scenario %r: no CEP IDs resolved, skipping.", name)
            continue

        # Global brands for this scenario (all respondents pooled)
        active = rbc_df[rbc_df["cep_id"].isin(active_cep_ids)]
        global_brands = active["brand_id"].unique()
        if len(global_brands) == 0:
            continue

        # Semantic: sum assoc_strength per (respondent, brand) for active CEPs
        semantic = (
            active.groupby(["respondent_id", "brand_id"])["assoc_strength"]
            .sum()
            .reset_index()
            .rename(columns={"assoc_strength": "semantic"})
        )

        # Cross-join: every respondent × every global brand
        b_frame = pd.DataFrame({"brand_id": global_brands})
        cross = r_frame.merge(b_frame, how="cross")
        cross = cross.merge(semantic, on=["respondent_id", "brand_id"], how="left")
        cross["semantic"] = cross["semantic"].fillna(0.0)

        # Episodic boost
        if episodic_events is not None and len(episodic_events) > 0:
            ep = (
                episodic_events[episodic_events["cep_id"].isin(active_cep_ids)]
                .groupby(["respondent_id", "brand_id"])["strength"]
                .sum()
                .reset_index()
            )
            cross = cross.merge(ep, on=["respondent_id", "brand_id"], how="left")
            cross["strength"] = cross["strength"].fillna(0.0)
            cross["semantic"] += cross["strength"]
            cross = cross.drop(columns=["strength"])

        # Competition: penalty × sum of other brands' semantic for this respondent
        total_sem = cross.groupby("respondent_id")["semantic"].transform("sum")
        cross["competition"] = penalty * (total_sem - cross["semantic"]).clip(lower=0.0)

        # Base priors — CEP-conditioned preferred, global brand prior as fallback
        if cep_brand_priors is not None:
            # Average prior across active CEPs for each brand
            active_entries = [
                {"brand_id": bid, "prior_val": val}
                for (bid, cid), val in cep_brand_priors.items()
                if cid in active_cep_ids
            ]
            if active_entries:
                ap_df = pd.DataFrame(active_entries)
                mean_cep_prior = (
                    ap_df.groupby("brand_id")["prior_val"]
                    .mean()
                    .mul(config.defaults.base_prior_weight)
                    .rename("base_prior")
                    .reset_index()
                )
                cross = cross.merge(mean_cep_prior, on="brand_id", how="left")
                # Fallback for brands not covered by cep_brand_priors
                fallback_prior = config.defaults.base_usage_default
                if brand_priors is not None:
                    fallback_series = cross["brand_id"].map(brand_priors).fillna(config.defaults.base_usage_default) * config.defaults.base_prior_weight
                    cross["base_prior"] = cross["base_prior"].fillna(fallback_series)
                else:
                    cross["base_prior"] = cross["base_prior"].fillna(fallback_prior)
            else:
                cross["base_prior"] = base
        elif brand_priors is not None:
            prior_frame = pd.DataFrame([
                {"brand_id": bid, "base_prior": v * config.defaults.base_prior_weight}
                for bid, v in brand_priors.items()
            ])
            cross = cross.merge(prior_frame, on="brand_id", how="left")
            cross["base_prior"] = cross["base_prior"].fillna(config.defaults.base_usage_default)
        else:
            cross["base_prior"] = base

        cross["recall_score"] = cross["semantic"] + cross['base_prior'] - cross["competition"]
        cross = cross.drop(columns=['base_prior'], errors='ignore')

        # Softmax per respondent
        cross["recall_prob"] = cross.groupby("respondent_id")["recall_score"].transform(
            lambda x: _softmax_col(x, temperature)
        )

        # Rank within respondent
        cross["rank"] = cross.groupby("respondent_id")["recall_prob"].rank(
            ascending=False, method="min"
        ).astype(int)

        cross["brand_name"] = cross["brand_id"].map(brand_name_map).fillna(cross["brand_id"])
        cross["scenario_name"] = name
        cross["recall_score"] = cross["recall_score"].round(6)
        cross["recall_prob"] = cross["recall_prob"].round(6)

        chunks.append(cross[[
            "respondent_id", "scenario_name", "brand_id", "brand_name",
            "recall_score", "recall_prob", "rank",
        ]])

    if not chunks:
        return pd.DataFrame(columns=[
            "respondent_id", "scenario_name", "brand_id", "brand_name",
            "recall_score", "recall_prob", "rank",
        ])
    return pd.concat(chunks, ignore_index=True)


def run_ad_impact(
    respondent_ids: list[str],
    scenarios: list[dict],
    rbc_pre: pd.DataFrame,
    rbc_post: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    brand_name_map: dict[str, str],
    config: CepSimConfig,
    brand_priors: dict[str, float] | None = None,
    cep_brand_priors: dict[tuple, float] | None = None,
) -> pd.DataFrame:
    """
    Compare recall before and after ad exposure for each respondent × scenario.
    """
    pre_df = run_scenario_recall(
        respondent_ids, scenarios, rbc_pre, cep_master_df, brand_name_map, config,
        brand_priors=brand_priors, cep_brand_priors=cep_brand_priors,
    )
    post_df = run_scenario_recall(
        respondent_ids, scenarios, rbc_post, cep_master_df, brand_name_map, config,
        brand_priors=brand_priors, cep_brand_priors=cep_brand_priors,
    )

    pre_df = pre_df.rename(columns={"recall_score": "recall_pre", "rank": "rank_pre"})
    post_df = post_df.rename(columns={"recall_score": "recall_post", "rank": "rank_post"})

    merged = pre_df.merge(
        post_df[["respondent_id", "scenario_name", "brand_id", "recall_post", "rank_post"]],
        on=["respondent_id", "scenario_name", "brand_id"],
        how="outer",
    ).fillna(0)

    merged["delta"] = (merged["recall_post"] - merged["recall_pre"]).round(6)
    merged["recall_pre"] = merged["recall_pre"].round(6)
    merged["recall_post"] = merged["recall_post"].round(6)

    return merged[[
        "respondent_id", "scenario_name", "brand_id", "brand_name",
        "recall_pre", "recall_post", "delta", "rank_pre", "rank_post",
    ]]


def build_segment_summary(
    ad_impact_df: pd.DataFrame,
    respondents_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate ad impact by segment × scenario × brand."""
    merged = ad_impact_df.merge(
        respondents_df[["respondent_id", "segment"]],
        on="respondent_id",
        how="left",
    )
    merged["segment"] = merged["segment"].fillna("unknown")

    summary = (
        merged.groupby(["segment", "scenario_name", "brand_id", "brand_name"])
        .agg(
            avg_recall_pre=("recall_pre", "mean"),
            avg_recall_post=("recall_post", "mean"),
            avg_delta=("delta", "mean"),
        )
        .round(6)
        .reset_index()
    )
    return summary


def run_sanity_checks(
    scenario_recall_df: pd.DataFrame,
    ad_impact_df: pd.DataFrame,
) -> dict:
    """
    Run basic sanity checks and return a dict of results.
    Raises a warning (not an exception) for each failing check.
    """
    results = {}

    # 1. Cue relevance: at least one brand has recall_prob > 0 for every scenario
    zero_recall = (
        scenario_recall_df.groupby("scenario_name")["recall_prob"]
        .max()
        .lt(0.001)
    )
    results["cue_relevance"] = "PASS" if not zero_recall.any() else f"WARN: {zero_recall[zero_recall].index.tolist()}"

    # 2. Respondent heterogeneity: recall prob std > 0 across respondents for at least one scenario
    std_by_scenario = (
        scenario_recall_df.groupby(["scenario_name", "brand_id"])["recall_prob"].std()
    )
    results["respondent_heterogeneity"] = (
        "PASS" if (std_by_scenario > 0).any() else "WARN: all respondents identical"
    )

    # 3. Ad lift: mean delta > 0 for advertised brand
    mean_delta = ad_impact_df.groupby("brand_id")["delta"].mean()
    positive_lift = (mean_delta > 0).any()
    results["ad_lift"] = "PASS" if positive_lift else "WARN: no positive lift detected"

    # 4. Competition realism: more than one brand appears in rankings
    n_brands = scenario_recall_df["brand_id"].nunique()
    results["competition_realism"] = (
        "PASS" if n_brands > 1 else f"WARN: only {n_brands} brand in rankings"
    )

    for check, outcome in results.items():
        if outcome.startswith("WARN"):
            logger.warning("Sanity check [%s]: %s", check, outcome)
        else:
            logger.info("Sanity check [%s]: %s", check, outcome)

    return results


def run_calibration_check(
    scenario_recall_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare model-predicted recall probabilities against observed survey mention rates.

    For each (scenario_name, brand_name) cell:
      - observed_mention_rate : fraction of respondents who mentioned that brand at that CEP
      - predicted_recall_prob : mean recall_prob from the model across all respondents
      - abs_error             : |predicted - observed|

    The scenario_name in scenario_recall_df must match the cep_description in long_df.
    Matching is done by joining on brand_name and the CEP description substring used
    to build each scenario's active_ceps.

    Returns a DataFrame sorted by abs_error descending, plus summary MAE.
    """
    # Observed mention rates: mean of binary mention per (cep_description, brand_name)
    observed = (
        long_df.groupby(["cep_description", "brand_name"])["mentioned"]
        .mean()
        .reset_index()
        .rename(columns={"mentioned": "observed_mention_rate"})
    )

    # Predicted: mean recall_prob per (scenario_name, brand_name)
    predicted = (
        scenario_recall_df.groupby(["scenario_name", "brand_name"])["recall_prob"]
        .mean()
        .reset_index()
        .rename(columns={"recall_prob": "predicted_recall_prob"})
    )

    # Join on brand_name; align scenario to CEP description by substring match.
    # Build a mapping: scenario_name → cep_description (first description containing
    # the scenario's active_cep keyword, using the scenario_name as a loose key).
    # Since scenario_recall_df doesn't carry cep_description, we match by brand_name
    # and let the caller interpret the per-scenario MAE.
    #
    # For a direct cell-level join we merge on brand_name only and keep all combos,
    # then report MAE at the brand level (averaged across scenarios/CEPs).
    brand_observed = (
        long_df.groupby("brand_name")["mentioned"]
        .mean()
        .reset_index()
        .rename(columns={"mentioned": "observed_mention_rate"})
    )
    brand_predicted = (
        scenario_recall_df.groupby("brand_name")["recall_prob"]
        .mean()
        .reset_index()
        .rename(columns={"recall_prob": "predicted_recall_prob"})
    )

    cal = brand_predicted.merge(brand_observed, on="brand_name", how="left")
    cal["abs_error"] = (cal["predicted_recall_prob"] - cal["observed_mention_rate"]).abs().round(4)
    cal = cal.sort_values("abs_error", ascending=False).reset_index(drop=True)

    mae = cal["abs_error"].mean()
    logger.info(
        "Calibration check: MAE = %.4f (mean abs error, predicted recall_prob vs observed mention rate)",
        mae,
    )
    for _, row in cal.iterrows():
        level = logging.WARNING if row["abs_error"] > 0.05 else logging.INFO
        logger.log(
            level,
            "  %-22s  predicted=%.3f  observed=%.3f  err=%.3f",
            row["brand_name"], row["predicted_recall_prob"],
            row["observed_mention_rate"], row["abs_error"],
        )

    cal.attrs["mae"] = mae
    return cal


def run_spearman_validity(
    scenario_recall_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct-validity check: Spearman rank correlation between predicted recall
    probabilities and observed survey mention rates, per scenario.

    A positive rho indicates the model correctly orders brands by accessibility.
    Interpretation:
        rho > 0.6  strong construct validity
        rho 0.3–0.6  moderate
        rho < 0.3  weak — model ranking diverges from observed data

    Returns a DataFrame with columns:
        scenario_name, spearman_rho, n_brands
    plus attrs['mean_rho'] for the overall mean.
    """
    observed = (
        long_df.groupby("brand_name")["mentioned"]
        .mean()
        .reset_index()
        .rename(columns={"mentioned": "observed_rate"})
    )

    results = []
    for scenario_name, group in scenario_recall_df.groupby("scenario_name"):
        predicted = (
            group.groupby("brand_name")["recall_prob"]
            .mean()
            .reset_index()
            .rename(columns={"recall_prob": "predicted_prob"})
        )
        merged = predicted.merge(observed, on="brand_name", how="inner")
        if len(merged) < 3:
            continue

        rho = merged["predicted_prob"].corr(merged["observed_rate"], method="spearman")
        results.append({
            "scenario_name": scenario_name,
            "spearman_rho":  round(rho, 4),
            "n_brands":      len(merged),
        })

    results_df = pd.DataFrame(
        results,
        columns=["scenario_name", "spearman_rho", "n_brands"],
    ).sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    mean_rho = results_df["spearman_rho"].mean() if len(results_df) else float("nan")
    results_df.attrs["mean_rho"] = mean_rho
    logger.info("Construct validity: mean Spearman rho = %.4f across %d scenarios",
                mean_rho, len(results_df))
    return results_df


def check_brand_deduplication(rbc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for near-duplicate brand names that may indicate the same brand was
    encoded under two slightly different strings.

    Uses simple character-overlap similarity (SequenceMatcher ratio > 0.85).
    Returns a DataFrame of suspect pairs; empty DataFrame = no issues.
    """
    from difflib import SequenceMatcher

    brand_names = sorted(rbc_df["brand_name"].unique())
    issues = []
    for i, a in enumerate(brand_names):
        for b in brand_names[i + 1:]:
            ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if ratio > 0.85:
                issues.append({
                    "brand_a":    a,
                    "brand_b":    b,
                    "similarity": round(ratio, 3),
                })

    issues_df = pd.DataFrame(issues)
    if len(issues_df) > 0:
        logger.warning("Brand deduplication: %d suspect pair(s) found — %s",
                       len(issues_df), issues_df[["brand_a", "brand_b"]].values.tolist())
    else:
        logger.info("Brand deduplication: no near-duplicate brand names found.")
    return issues_df


def run_scenario_diagnostics(
    scenario_recall_df: pd.DataFrame,
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-scenario calibration diagnostics.

    For each scenario, computes:
      mae             : mean absolute error between predicted recall_prob and observed mention rate
      spearman_rho    : rank correlation between predicted and observed brand ordering
      n_brands        : number of brands present in both predicted and observed
      coverage        : fraction of observed brands also in predicted set
      top_over_brand  : brand most over-predicted (predicted >> observed)
      top_under_brand : brand most under-predicted (observed >> predicted)

    Returns DataFrame sorted by mae descending.
    Scenarios with spearman_rho < 0.2 are candidates for ontology review.
    """
    obs = (
        long_df.groupby("brand_name")["mentioned"]
        .mean()
        .rename("observed_rate")
    )

    rows = []
    for name, grp in scenario_recall_df.groupby("scenario_name"):
        pred = (
            grp.groupby("brand_name")["recall_prob"]
            .mean()
            .rename("predicted_prob")
        )
        merged = pd.concat([pred, obs], axis=1).dropna()
        if len(merged) < 3:
            continue

        merged["error"] = merged["predicted_prob"] - merged["observed_rate"]
        mae = merged["error"].abs().mean()
        rho = merged["predicted_prob"].corr(merged["observed_rate"], method="spearman")
        coverage = len(merged) / max(len(obs), 1)

        rows.append({
            "scenario_name":   name,
            "mae":             round(float(mae), 4),
            "spearman_rho":    round(float(rho), 4),
            "n_brands":        len(merged),
            "coverage":        round(float(coverage), 3),
            "top_over_brand":  str(merged["error"].idxmax()),
            "top_under_brand": str(merged["error"].idxmin()),
        })

    result = pd.DataFrame(
        rows,
        columns=["scenario_name", "mae", "spearman_rho", "n_brands", "coverage",
                 "top_over_brand", "top_under_brand"],
    ).sort_values("mae", ascending=False).reset_index(drop=True)
    n_weak = (result["spearman_rho"] < 0.2).sum() if len(result) else 0
    if n_weak > 0:
        logger.warning(
            "Scenario diagnostics: %d scenario(s) with Spearman ρ < 0.2 — check ontology/CEP mapping.",
            n_weak,
        )
    return result


def save_outputs(
    scenario_recall_df: pd.DataFrame,
    ad_impact_df: pd.DataFrame,
    segment_summary_df: pd.DataFrame,
    config: CepSimConfig,
) -> dict[str, Path]:
    out_dir = Path(config.output.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in [
        ("scenario_recall_output", scenario_recall_df),
        ("ad_impact_output",       ad_impact_df),
        ("segment_summary",        segment_summary_df),
    ]:
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
        logger.info("Saved %s: %s (%d rows)", name, p, len(df))

    return paths
