"""
Phase 2A calibration tooling.

Provides:
  make_holdout_split()      — deterministic 80/20 respondent split
  compute_brand_priors()    — brand-level accessibility priors from training data
  fit_parameters()          — joint grid search over γ, τ, β_scale
  run_holdout_validation()  — full eval on held-out respondents
  build_calibration_report()— writes markdown + CSV summary
  tune_softmax_temperature() — kept for quick single-parameter tuning (legacy)
"""
import hashlib
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backend.schemas.config import CepSimConfig
from backend.service.utils import brand_to_id

logger = logging.getLogger(__name__)

_DEFAULT_TEMP_GRID    = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]
_DEFAULT_GAMMA_GRID   = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
_DEFAULT_PRIOR_GRID   = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def make_holdout_split(
    respondent_ids: list[str],
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Deterministic hash-based train/holdout split.

    Each respondent ID is hashed with the seed to get a stable bucket assignment.
    Reproducible across runs; independent of insertion order.

    Returns (train_ids, holdout_ids).
    """
    def _bucket(rid: str) -> int:
        h = hashlib.md5(f"{seed}:{rid}".encode()).hexdigest()
        return int(h, 16) % 100

    holdout_pct = int(holdout_fraction * 100)
    train, holdout = [], []
    for rid in respondent_ids:
        (holdout if _bucket(rid) < holdout_pct else train).append(rid)

    logger.info(
        "Holdout split: %d train / %d holdout (%.0f%% holdout)",
        len(train), len(holdout), 100 * len(holdout) / len(respondent_ids),
    )
    return train, holdout


def compute_brand_priors(long_df: pd.DataFrame) -> dict[str, float]:
    """
    Derive brand-level accessibility priors from survey mention rates.

    Prior = mean(mentioned) across all CEPs for that brand in the provided DataFrame.
    Pass training-only long_df to avoid leaking holdout data.

    Returns {brand_id: mention_rate} where mention_rate ∈ [0, 1].
    """
    rates = (
        long_df.groupby("brand_name")["mentioned"]
        .mean()
        .reset_index()
    )
    rates["brand_id"] = rates["brand_name"].apply(brand_to_id)
    priors = rates.set_index("brand_id")["mentioned"].to_dict()
    logger.info(
        "Brand priors: %d brands, mean prior=%.3f, range=[%.3f, %.3f]",
        len(priors), np.mean(list(priors.values())),
        min(priors.values()), max(priors.values()),
    )
    return priors


def _precompute_scenario_frames(
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    scenarios: list[dict],
    respondent_ids: list[str],
    brand_priors: dict[str, float],
    long_df: pd.DataFrame,
    brand_similarity: dict[tuple[str, str], float] | None = None,
) -> tuple[list[dict], np.ndarray]:
    """
    Precompute per-scenario numpy matrices for fast grid search.

    Returns (frames, obs_by_brand) where:
      frames      — list of dicts with keys: semantic, comp, prior, brand_ids
      obs_by_brand — ordered array of observed mention rates (matches a global brand list)

    Each frame has:
      semantic [n_resp, n_brands_s] — assoc_strength sums for active CEPs
      comp     [n_resp, n_brands_s] — pre-computed competition matrix
      prior    [n_brands_s]         — brand prior values (raw, before pw scaling)
      brand_ids list[str]           — ordered brand_id list for this scenario
    """
    from backend.service.recall_engine import _resolve_cep_ids

    r_frame = pd.DataFrame({"respondent_id": respondent_ids})
    resp_order = {r: i for i, r in enumerate(respondent_ids)}

    # Observed mention rates by brand_name (for MAE computation)
    obs_by_name = long_df.groupby("brand_name")["mentioned"].mean().to_dict()

    # brand_id → brand_name from rbc_df
    bid_to_name = (
        rbc_df[["brand_id", "brand_name"]].drop_duplicates()
        .set_index("brand_id")["brand_name"].to_dict()
    )

    frames: list[dict] = []

    for scenario in scenarios:
        active_cep_ids = _resolve_cep_ids(scenario["active_ceps"], cep_master_df)
        if not active_cep_ids:
            continue

        active = rbc_df[rbc_df["cep_id"].isin(active_cep_ids)]
        brand_ids = list(active["brand_id"].unique())
        if not brand_ids:
            continue

        # Semantic: sum assoc_strength per (respondent, brand) for active CEPs
        sem_agg = (
            active.groupby(["respondent_id", "brand_id"])["assoc_strength"]
            .sum().reset_index()
        )

        # Pivot to [n_resp × n_brands] numpy matrix (fast path via reindex)
        b_frame = pd.DataFrame({"brand_id": brand_ids})
        cross = r_frame.merge(b_frame, how="cross")
        cross = cross.merge(sem_agg, on=["respondent_id", "brand_id"], how="left")
        cross["assoc_strength"] = cross["assoc_strength"].fillna(0.0)

        sem_wide = (
            cross.pivot(index="respondent_id", columns="brand_id", values="assoc_strength")
            .fillna(0.0)
            .reindex(index=respondent_ids, columns=brand_ids, fill_value=0.0)
        )
        sem_matrix = sem_wide.values.astype(np.float32)   # [n_resp, n_brands]

        # Competition matrix
        if brand_similarity is not None:
            sim_vals = np.array(
                [[brand_similarity.get((a, b), 0.0) for b in brand_ids] for a in brand_ids],
                dtype=np.float32,
            )
            comp_matrix = sem_matrix @ sim_vals.T          # [n_resp, n_brands]
        else:
            # Flat competition (cancels in softmax — γ will have no effect)
            total_sem = sem_matrix.sum(axis=1, keepdims=True)
            comp_matrix = (total_sem - sem_matrix).astype(np.float32)

        # Per-brand prior and observed rate arrays (aligned to brand_ids order)
        prior_arr = np.array([brand_priors.get(bid, 0.0) for bid in brand_ids], dtype=np.float32)
        obs_arr   = np.array(
            [obs_by_name.get(bid_to_name.get(bid, ""), 0.0) for bid in brand_ids],
            dtype=np.float32,
        )

        frames.append({
            "semantic":  sem_matrix,
            "comp":      comp_matrix,
            "prior":     prior_arr,
            "obs":       obs_arr,
            "brand_ids": brand_ids,
        })

    return frames


def _fast_grid_mae(
    frames: list[dict],
    base: float,
    tau: float,
    gamma: float,
    prior_weight: float,
) -> float:
    """
    Compute calibration MAE for one (τ, γ, prior_weight) combination.

    Uses precomputed per-scenario numpy matrices — no pandas in the hot loop.
    MAE is averaged across all (scenario, brand) pairs.
    """
    total_ae = 0.0
    count = 0
    for sf in frames:
        sem   = sf["semantic"]   # [n_resp, n_brands]
        comp  = sf["comp"]       # [n_resp, n_brands]
        prior = sf["prior"]      # [n_brands]
        obs   = sf["obs"]        # [n_brands]

        score = sem + prior_weight * prior[np.newaxis, :] + base - gamma * comp
        # Numerically stable softmax per respondent
        s = score / tau
        s -= s.max(axis=1, keepdims=True)
        exp_s = np.exp(s)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)   # [n_resp, n_brands]

        pred_mean = probs.mean(axis=0)                      # [n_brands]
        total_ae += float(np.abs(pred_mean - obs).mean())
        count += 1

    return total_ae / count if count else float("nan")


def fit_parameters(
    long_df_train: pd.DataFrame,
    rbc_df_train: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    scenarios: list[dict],
    config: CepSimConfig,
    brand_priors: dict[str, float],
    tau_grid: list[float] | None = None,
    gamma_grid: list[float] | None = None,
    prior_weight_grid: list[float] | None = None,
    cep_brand_priors: dict[tuple, float] | None = None,
    brand_similarity: dict[tuple[str, str], float] | None = None,
) -> dict:
    """
    Joint grid search over softmax temperature (τ), competition weight (γ),
    and brand-prior scalar (β_scale) to minimise calibration MAE on training data.

    Uses a fast numpy path: semantics and competition matrices are precomputed once
    per scenario, then the grid loop only does softmax + MAE (no pandas per iteration).

    Two-pass: coarse grid → fine grid around best coarse result.
    Restores original config values after search.

    Returns a dict with keys: tau, gamma, prior_weight, mae, grid_results (DataFrame).
    """
    orig_tau   = config.defaults.softmax_temperature
    orig_gamma = config.defaults.competition_penalty_weight
    orig_pw    = config.defaults.base_prior_weight

    tau_grid = tau_grid   or [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    gamma_grid = gamma_grid or [0.0, 0.1, 0.3, 1.0]
    pw_grid  = prior_weight_grid or [0.0, 1.0, 3.0, 5.0]

    train_ids = rbc_df_train["respondent_id"].astype(str).unique().tolist()
    base      = config.defaults.base_usage_default

    n_total = len(tau_grid) * len(gamma_grid) * len(pw_grid)
    logger.info(
        "Fitting parameters: %d combinations (fast numpy path, %s brand_similarity) ...",
        n_total, "with" if brand_similarity else "without",
    )

    # ── Precompute per-scenario matrices once ─────────────────────────────
    frames = _precompute_scenario_frames(
        rbc_df_train, cep_master_df, scenarios, train_ids,
        brand_priors, long_df_train, brand_similarity=brand_similarity,
    )

    # ── Coarse grid search ────────────────────────────────────────────────
    rows = []
    for tau in tau_grid:
        for gamma in gamma_grid:
            for pw in pw_grid:
                mae = _fast_grid_mae(frames, base, tau, gamma, pw)
                rows.append({"tau": tau, "gamma": gamma, "prior_weight": pw, "mae": round(mae, 6)})

    grid_df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = grid_df.iloc[0]

    # ── Fine grid around best coarse result ───────────────────────────────
    def _neighbours(v, grid):
        idx = grid.index(v) if v in grid else 0
        lo = grid[max(0, idx - 1)]
        hi = grid[min(len(grid) - 1, idx + 1)]
        return sorted(set([lo, v, hi, (lo + v) / 2, (v + hi) / 2]))

    tau_fine   = _neighbours(float(best["tau"]),          tau_grid)
    gamma_fine = _neighbours(float(best["gamma"]),        gamma_grid)
    pw_fine    = _neighbours(float(best["prior_weight"]), pw_grid)

    fine_rows = []
    for tau in tau_fine:
        for gamma in gamma_fine:
            for pw in pw_fine:
                mae = _fast_grid_mae(frames, base, tau, gamma, pw)
                fine_rows.append({"tau": tau, "gamma": gamma, "prior_weight": pw, "mae": round(mae, 6)})

    all_df = pd.concat([grid_df, pd.DataFrame(fine_rows)], ignore_index=True)
    all_df = (
        all_df.sort_values("mae")
        .drop_duplicates(subset=["tau", "gamma", "prior_weight"])
        .reset_index(drop=True)
    )

    best_row  = all_df.iloc[0]
    best_tau   = float(best_row["tau"])
    best_gamma = float(best_row["gamma"])
    best_pw    = float(best_row["prior_weight"])
    best_mae   = float(best_row["mae"])

    # Restore original config
    config.defaults.softmax_temperature        = orig_tau
    config.defaults.competition_penalty_weight = orig_gamma
    config.defaults.base_prior_weight          = orig_pw

    logger.info(
        "Best params: τ=%.3f  γ=%.3f  β_scale=%.2f  →  MAE=%.4f",
        best_tau, best_gamma, best_pw, best_mae,
    )
    return {
        "tau":          best_tau,
        "gamma":        best_gamma,
        "prior_weight": best_pw,
        "mae":          best_mae,
        "grid_results": all_df,
    }


def run_holdout_validation(
    holdout_ids: list[str],
    long_df_holdout: pd.DataFrame,
    rbc_df_holdout: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    scenarios: list[dict],
    config: CepSimConfig,
    brand_priors: dict[str, float],
    cep_brand_priors: dict[tuple, float] | None = None,
) -> dict:
    """
    Evaluate calibration and construct validity on held-out respondents.

    Uses brand_priors derived from training data (no leakage).
    Config should already have tuned parameters applied before calling.

    Returns dict with keys: cal_df, spearman_df, mae, mean_rho.
    """
    from backend.service.validator import (
        run_scenario_recall,
        run_calibration_check,
        run_spearman_validity,
    )

    brand_name_map = (
        rbc_df_holdout[["brand_id", "brand_name"]].drop_duplicates()
        .set_index("brand_id")["brand_name"].to_dict()
    )

    recall_df = run_scenario_recall(
        holdout_ids, scenarios, rbc_df_holdout, cep_master_df,
        brand_name_map, config, brand_priors=brand_priors,
        cep_brand_priors=cep_brand_priors,
    )

    cal_df      = run_calibration_check(recall_df, long_df_holdout)
    spearman_df = run_spearman_validity(recall_df, long_df_holdout)

    result = {
        "recall_df":   recall_df,
        "cal_df":      cal_df,
        "spearman_df": spearman_df,
        "mae":         float(cal_df.attrs["mae"]),
        "mean_rho":    float(spearman_df.attrs.get("mean_rho", float("nan"))),
    }
    logger.info(
        "Holdout validation: MAE=%.4f  mean_rho=%.4f  (%d respondents)",
        result["mae"], result["mean_rho"], len(holdout_ids),
    )
    return result


def build_calibration_report(
    train_results: dict,
    holdout_results: dict,
    fitted_params: dict,
    config: CepSimConfig,
    n_train: int,
    n_holdout: int,
) -> str:
    """
    Build a markdown calibration/validity report string.

    Parameters
    ----------
    train_results   : output of run_holdout_validation() on training set
    holdout_results : output of run_holdout_validation() on holdout set
    fitted_params   : output of fit_parameters()
    config          : CepSimConfig with tuned parameters applied
    n_train / n_holdout : respondent counts

    Returns the markdown string (caller can print or save to file).
    """
    def _tier(rate):
        if rate >= 0.25:  return "high (>=25%)"
        if rate >= 0.10:  return "mid (10-25%)"
        return "low (<10%)"

    cal_train   = train_results["cal_df"].copy()
    cal_holdout = holdout_results["cal_df"].copy()
    sp_train    = train_results["spearman_df"]
    sp_holdout  = holdout_results["spearman_df"]

    cal_train["tier"]   = cal_train["observed_mention_rate"].apply(_tier)
    cal_holdout["tier"] = cal_holdout["observed_mention_rate"].apply(_tier)

    mae_train   = train_results["mae"]
    mae_holdout = holdout_results["mae"]
    rho_train   = train_results["mean_rho"]
    rho_holdout = holdout_results["mean_rho"]

    # Brand-tier MAE
    tier_train   = cal_train.groupby("tier")["abs_error"].mean().round(4).to_dict()
    tier_holdout = cal_holdout.groupby("tier")["abs_error"].mean().round(4).to_dict()

    # Pass/fail
    def _flag(mae): return "PASS" if mae < 0.05 else ("CLOSE" if mae < 0.10 else "FAIL")
    def _rho_flag(rho): return "strong" if rho > 0.6 else ("moderate" if rho > 0.3 else "weak")

    lines = [
        "# CEP Simulator — Calibration & Validity Report",
        "",
        f"**Country:** {config.survey.country}  ",
        f"**Train N:** {n_train}  |  **Holdout N:** {n_holdout}",
        "",
        "## Fitted parameters",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Temperature (tau) | {fitted_params['tau']:.3f} |",
        f"| Competition weight (gamma) | {fitted_params['gamma']:.3f} |",
        f"| Prior weight (beta_scale) | {fitted_params['prior_weight']:.2f} |",
        f"| Train MAE at best params | {fitted_params['mae']:.4f} |",
        "",
        "## Calibration summary",
        "",
        f"| Set | MAE | Target (<5 pp) | Strong (<2 pp) |",
        f"|-----|-----|----------------|----------------|",
        f"| Train   | {mae_train:.4f} | {_flag(mae_train)} | {'PASS' if mae_train < 0.02 else '-'} |",
        f"| Holdout | {mae_holdout:.4f} | {_flag(mae_holdout)} | {'PASS' if mae_holdout < 0.02 else '-'} |",
        "",
        "### MAE by brand tier (holdout)",
        "",
        f"| Tier | MAE |",
        f"|------|-----|",
    ]
    for tier, mae in sorted(tier_holdout.items()):
        lines.append(f"| {tier} | {mae:.4f} |")

    lines += [
        "",
        "## Construct validity (Spearman rho)",
        "",
        f"| Set | Mean rho | Interpretation |",
        f"|-----|--------|----------------|",
        f"| Train   | {rho_train:.4f} | {_rho_flag(rho_train)} |",
        f"| Holdout | {rho_holdout:.4f} | {_rho_flag(rho_holdout)} |",
        "",
        "### Per-scenario rho (holdout)",
        "",
        f"| Scenario | rho | n_brands |",
        f"|----------|---|----------|",
    ]
    for _, row in sp_holdout.iterrows():
        flag = " [!]" if row["spearman_rho"] < 0.2 else ""
        lines.append(f"| {row['scenario_name']} | {row['spearman_rho']:.4f}{flag} | {row['n_brands']} |")

    lines += [
        "",
        "## Verdict",
        "",
        f"- Holdout MAE: **{mae_holdout:.4f}** ({_flag(mae_holdout)})",
        f"- Holdout Spearman rho: **{rho_holdout:.4f}** ({_rho_flag(rho_holdout)})",
        "",
        "> Scenarios with rho < 0.2 flagged [!] above are candidates for ontology review.",
    ]

    return "\n".join(lines)


def compute_cep_brand_priors(long_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """
    Compute CEP-conditioned brand accessibility priors from training mention rates.

    Returns {(brand_id, cep_id): mention_rate} where mention_rate is the fraction
    of respondents who mentioned that brand at that specific CEP.

    More informative than global brand priors because brands have different baseline
    accessibility depending on the consumption context (e.g. Guinness is more salient
    at 'pub with friends' than at 'hosting at home').

    Pass training-only long_df to avoid leaking holdout data.
    """
    rates = (
        long_df.groupby(["brand_name", "cep_id"])["mentioned"]
        .mean()
        .reset_index()
    )
    rates["brand_id"] = rates["brand_name"].apply(brand_to_id)
    result = {
        (row["brand_id"], row["cep_id"]): float(row["mentioned"])
        for _, row in rates.iterrows()
    }
    logger.info(
        "CEP-brand priors: %d (brand, cep) pairs across %d brands and %d CEPs",
        len(result),
        rates["brand_id"].nunique(),
        rates["cep_id"].nunique(),
    )
    return result


def compute_respondent_responsiveness(
    rbc_df: pd.DataFrame,
    method: str = "repertoire",
) -> dict[str, float]:
    """
    Compute per-respondent ad responsiveness multipliers.

    method='repertoire':
        Respondents with a narrower brand repertoire (fewer distinct brands
        mentioned across all CEPs) are more responsive to ad exposure —
        they are less entrenched and more open to forming new associations.

        responsiveness(r) = clip(mean_repertoire / n_brands(r), 0.5, 3.0)

        A respondent who mentioned 3 brands (vs a mean of 10) gets ~3.3×
        amplification (capped at 3.0). One who mentioned 20 gets ~0.5×.

    Returns {respondent_id: multiplier}.
    """
    n_brands = rbc_df.groupby("respondent_id")["brand_id"].nunique()
    mean_n = float(n_brands.mean())
    responsiveness = (mean_n / n_brands).clip(lower=0.5, upper=3.0)
    logger.info(
        "Respondent responsiveness: mean=%.3f  min=%.3f  max=%.3f  (%d respondents)",
        responsiveness.mean(), responsiveness.min(), responsiveness.max(), len(responsiveness),
    )
    return responsiveness.to_dict()


def run_ablation(
    long_df: pd.DataFrame,
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    scenarios: list[dict],
    config: "CepSimConfig",
    focal_brand_id: str | None = None,
    focal_scenario: str | None = None,
    ads: list | None = None,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    5-variant ablation table to isolate the contribution of each modelling decision.

    Variants (additive — each builds on the previous):
      1. global_priors_only    — brand priors, default (unfitted) params
      2. fitted_params         — + joint tau/gamma/beta_scale grid search
      3. saturation_friction   — + saturation ceiling + new-edge friction (ad engine)
      4. cep_priors            — + CEP-conditioned brand priors
      5. responsiveness        — + respondent-level ad responsiveness

    Metrics per variant:
      holdout_mae          mean absolute error on held-out respondents
      median_rho           median Spearman rank correlation across scenarios
      focal_lift           mean recall delta for focal_brand_id in focal_scenario
                           (NaN when ads / focal_brand_id / focal_scenario not supplied)
      worst_3_scenarios    three scenario names with highest MAE (comma-separated)

    All priors are derived from training respondents only — no holdout leakage.
    Config is restored to its original values after the function returns.

    Parameters
    ----------
    long_df           : full long-format survey DataFrame (all respondents)
    rbc_df            : full respondent-brand-CEP edge DataFrame (all respondents)
    cep_master_df     : CEP ontology
    scenarios         : list of scenario dicts (scenario_name, active_ceps, context)
    config            : CepSimConfig — modified temporarily, always restored
    focal_brand_id    : brand_id to track for ad-lift metric (e.g. "brand_heineken")
    focal_scenario    : scenario_name to track for ad-lift metric
    ads               : list of Ad objects to apply; if None, focal_lift is NaN
    holdout_fraction  : fraction of respondents to hold out (default 0.2)
    seed              : deterministic split seed (default 42)
    """
    from backend.service.validator import (
        run_scenario_recall,
        run_calibration_check,
        run_spearman_validity,
        run_scenario_diagnostics,
    )

    # --- save original config --------------------------------------------------
    orig_tau      = config.defaults.softmax_temperature
    orig_gamma    = config.defaults.competition_penalty_weight
    orig_pw       = config.defaults.base_prior_weight
    orig_wmax     = config.defaults.w_max
    orig_new_edge = config.defaults.new_edge_weight

    try:
        # --- split -------------------------------------------------------------
        all_ids = long_df["respondent_id"].unique().tolist()
        train_ids, holdout_ids = make_holdout_split(all_ids, holdout_fraction, seed)

        train_long   = long_df[long_df["respondent_id"].isin(train_ids)]
        holdout_long = long_df[long_df["respondent_id"].isin(holdout_ids)]
        train_rbc    = rbc_df[rbc_df["respondent_id"].isin(train_ids)]
        holdout_rbc  = rbc_df[rbc_df["respondent_id"].isin(holdout_ids)]

        # --- priors from train only (leakage fix) ------------------------------
        brand_priors  = compute_brand_priors(train_long)
        cep_bp        = compute_cep_brand_priors(train_long)
        # responsiveness from each respondent's own data — not an outcome variable
        resp_map      = compute_respondent_responsiveness(rbc_df)

        brand_name_map = (
            rbc_df[["brand_id", "brand_name"]].drop_duplicates()
            .set_index("brand_id")["brand_name"].to_dict()
        )

        # --- fit parameters on train (shared by variants 2–5) -----------------
        logger.info("Ablation: fitting parameters on training set ...")
        fitted = fit_parameters(
            train_long, train_rbc, cep_master_df, scenarios, config, brand_priors,
            cep_brand_priors=None,
        )
        best_tau   = fitted["tau"]
        best_gamma = fitted["gamma"]
        best_pw    = fitted["prior_weight"]

        # --- inner eval --------------------------------------------------------
        def _eval_variant(tau, gamma, pw, bp, cbp, resp, w_max, new_edge_w):
            config.defaults.softmax_temperature       = tau
            config.defaults.competition_penalty_weight = gamma
            config.defaults.base_prior_weight          = pw
            config.defaults.w_max                      = w_max
            config.defaults.new_edge_weight            = new_edge_w

            recall_df = run_scenario_recall(
                holdout_ids, scenarios, holdout_rbc, cep_master_df,
                brand_name_map, config, brand_priors=bp, cep_brand_priors=cbp,
            )
            cal_df  = run_calibration_check(recall_df, holdout_long)
            sp_df   = run_spearman_validity(recall_df, holdout_long)
            diag_df = run_scenario_diagnostics(recall_df, holdout_long)

            mae        = float(cal_df.attrs["mae"])
            median_rho = float(sp_df["spearman_rho"].median()) if len(sp_df) else float("nan")
            worst_3    = ", ".join(diag_df["scenario_name"].head(3).tolist())

            focal_lift = float("nan")
            if ads is not None and focal_brand_id and focal_scenario:
                from backend.service.ad_engine import apply_ad_to_population
                from backend.service.validator import run_ad_impact
                rbc_post = holdout_rbc.copy()
                for ad in ads:
                    rbc_post, _ = apply_ad_to_population(
                        holdout_ids, ad, rbc_post, config,
                        responsiveness_map=resp,
                    )
                impact_df = run_ad_impact(
                    holdout_ids, scenarios, holdout_rbc, rbc_post,
                    cep_master_df, brand_name_map, config,
                    brand_priors=bp, cep_brand_priors=cbp,
                )
                mask = (
                    (impact_df["brand_id"] == focal_brand_id)
                    & (impact_df["scenario_name"] == focal_scenario)
                )
                focal_lift = float(impact_df.loc[mask, "delta"].mean()) if mask.any() else float("nan")

            return mae, median_rho, focal_lift, worst_3

        # --- variant definitions -----------------------------------------------
        # Saturation/friction "off" = effectively infinite ceiling + no friction
        NO_SAT   = 1e9
        NO_FRICT = 1.0

        variants = [
            # name, tau, gamma, pw, brand_priors, cep_brand_priors, resp_map, w_max, new_edge_w
            ("global_priors_only",  orig_tau,   orig_gamma, orig_pw,   brand_priors, None,  None,     NO_SAT,      NO_FRICT),
            ("fitted_params",       best_tau,   best_gamma, best_pw,   brand_priors, None,  None,     NO_SAT,      NO_FRICT),
            ("saturation_friction", best_tau,   best_gamma, best_pw,   brand_priors, None,  None,     orig_wmax,   orig_new_edge),
            ("cep_priors",          best_tau,   best_gamma, best_pw,   brand_priors, cep_bp, None,    orig_wmax,   orig_new_edge),
            ("responsiveness",      best_tau,   best_gamma, best_pw,   brand_priors, cep_bp, resp_map, orig_wmax,  orig_new_edge),
        ]

        rows = []
        for (name, tau, gamma, pw, bp, cbp, resp, w_max, new_edge_w) in variants:
            logger.info("Ablation: running variant %r ...", name)
            mae, median_rho, focal_lift, worst_3 = _eval_variant(
                tau, gamma, pw, bp, cbp, resp, w_max, new_edge_w,
            )
            lift_str = f"{focal_lift:.4f}" if not math.isnan(focal_lift) else "N/A"
            logger.info(
                "  %-25s  MAE=%.4f  median_rho=%.4f  focal_lift=%s",
                name, mae, median_rho, lift_str,
            )
            rows.append({
                "variant":           name,
                "holdout_mae":       round(mae, 4),
                "median_rho":        round(median_rho, 4),
                "focal_lift":        round(focal_lift, 4) if not math.isnan(focal_lift) else float("nan"),
                "worst_3_scenarios": worst_3,
            })

        return pd.DataFrame(rows)

    finally:
        config.defaults.softmax_temperature       = orig_tau
        config.defaults.competition_penalty_weight = orig_gamma
        config.defaults.base_prior_weight          = orig_pw
        config.defaults.w_max                      = orig_wmax
        config.defaults.new_edge_weight            = orig_new_edge


def tune_softmax_temperature(
    long_df: pd.DataFrame,
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    scenarios: list[dict],
    config: CepSimConfig,
    temperatures: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Quick single-parameter temperature grid search (kept for backwards compatibility).
    For joint beta/gamma/tau fitting use fit_parameters() instead.
    """
    from backend.service.validator import (
        run_scenario_recall,
        run_calibration_check,
    )
    if temperatures is None:
        temperatures = _DEFAULT_TEMP_GRID

    orig = config.defaults.softmax_temperature
    respondent_ids = rbc_df["respondent_id"].unique().tolist()
    brand_name_map = (
        rbc_df[["brand_id", "brand_name"]].drop_duplicates()
        .set_index("brand_id")["brand_name"].to_dict()
    )

    rows = []
    for temp in temperatures:
        config.defaults.softmax_temperature = temp
        recall_df = run_scenario_recall(
            respondent_ids, scenarios, rbc_df, cep_master_df, brand_name_map, config
        )
        cal  = run_calibration_check(recall_df, long_df)
        mae  = float(cal.attrs["mae"])
        rows.append({"temperature": temp, "mae": round(mae, 6)})
        logger.info("  temperature=%.3f  MAE=%.4f", temp, mae)

    config.defaults.softmax_temperature = orig
    results_df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best_temp = float(results_df.loc[0, "temperature"])
    results_df.attrs["best_temperature"] = best_temp
    logger.info("Best temperature: %.3f (MAE=%.4f)", best_temp, results_df.loc[0, "mae"])
    return best_temp, results_df
