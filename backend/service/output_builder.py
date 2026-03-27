"""
Standard output generator for a CEP simulator run.

One call — generate_standard_outputs() — produces all client-facing artefacts,
saves them to the configured outputs directory, and writes a typed RunManifest:

  memory_graph.parquet              — full RBC edge table (machine-readable)
  scenario_recall_output.csv        — recall probs by respondent × scenario × brand
  ad_impact_output.csv              — pre/post deltas by respondent × scenario × brand
  flight_simulator_summary.csv      — aggregated pre/post for the focal scenario
  campaign_flight_simulator.png     — before/after + displacement chart
  memory_map_comparison.png         — loyalist vs competitor heatmaps
  calibration_dashboard.png         — scatter + Spearman + per-scenario MAE
  scenario_diagnostics.csv          — per-scenario MAE, Spearman, over/under brands
  calibration_report.md             — human-readable calibration summary
  analysis_summary_bundle.json      — curated LLM-ready bundle
  run_manifest.json                 — typed NodeArtifact index for the frontend
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.schemas.config import CepSimConfig
from backend.service.plotting import (
    plot_calibration_dashboard,
    plot_flight_simulator,
    plot_memory_map_comparison,
)
from backend.service.validator import (
    run_calibration_check,
    run_scenario_diagnostics,
    run_spearman_validity,
)
from backend.framework.artifacts.manifest import (
    generate_run_id,
    make_artifact,
    utc_now,
    write_manifest,
)
from backend.framework.schemas.artifact import RunManifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_standard_outputs(
    rbc_pre: pd.DataFrame,
    rbc_post: pd.DataFrame,
    impact_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    long_df: pd.DataFrame,
    scenario_recall_df: pd.DataFrame,
    focal_brand_id: str,
    focal_brand_name: str,
    focal_scenario: str,
    config: CepSimConfig,
    segment_summary_df: pd.DataFrame | None = None,
    run_id: str | None = None,
    node_id: str | None = None,
    out_dir: str | Path | None = None,
    dpi: int = 150,
    holdout_mae: float | None = None,
    median_spearman: float | None = None,
    skip_charts: bool = False,
) -> dict[str, Path]:
    """
    Generate all standard outputs for a CEP simulator run and write a RunManifest.

    Parameters
    ----------
    rbc_pre / rbc_post     : RBC edge tables before / after ad exposure
    impact_df              : output of run_ad_impact()
    cep_master_df          : CEP ontology
    long_df                : long-format survey DataFrame (must have cep_id column)
    scenario_recall_df     : output of run_scenario_recall()
    focal_brand_id         : e.g. "brand_heineken"
    focal_brand_name       : display name, e.g. "Heineken"
    focal_scenario         : scenario_name for the flight simulator centrepiece
    config                 : CepSimConfig — output directory from config.output.outputs_dir
    run_id                 : stable identifier for this run (generated if not supplied)
    node_id                : node label for the manifest (defaults to "cep_sim_{country}")
    out_dir                : override output directory
    dpi                    : PNG resolution
    holdout_mae / median_spearman : optional pre-computed calibration metrics for bundle

    Returns
    -------
    dict mapping output name → saved Path  (run_manifest.json always included)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_id  = run_id  or generate_run_id()
    country = getattr(config.survey, "country", "unknown")
    node_id = node_id or f"cep_sim_{country.lower()}"
    started = utc_now()

    out_path = Path(out_dir) if out_dir else Path(config.output.outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved:     dict[str, Path]  = {}
    artifacts: list             = []

    def _art(artifact_type, title, description, path, preview_type, row_count=None, metadata=None):
        artifacts.append(make_artifact(
            artifact_type=artifact_type,
            run_id=run_id,
            node_id=node_id,
            node_type="cep_sim",
            title=title,
            description=description,
            storage_path=path,
            preview_type=preview_type,
            row_count=row_count,
            metadata=metadata or {},
        ))

    # ------------------------------------------------------------------
    # 1. Memory graph (parquet)
    # ------------------------------------------------------------------
    try:
        p = out_path / "memory_graph.parquet"
        rbc_pre.to_parquet(p, index=False)
        saved["memory_graph"] = p
        _art("memory_graph", "Memory Graph",
             "Respondent-brand-CEP association strength edges (pre-ad)",
             p, "heatmap", row_count=len(rbc_pre),
             metadata={"respondents": rbc_pre["respondent_id"].nunique(),
                       "brands": rbc_pre["brand_id"].nunique(),
                       "ceps": rbc_pre["cep_id"].nunique()})
        logger.info("Saved memory graph: %s (%d rows)", p, len(rbc_pre))
    except Exception as exc:
        logger.warning("memory_graph failed: %s", exc)

    # ------------------------------------------------------------------
    # 2. Scenario recall CSV
    # ------------------------------------------------------------------
    try:
        p = out_path / "scenario_recall_output.csv"
        scenario_recall_df.to_csv(p, index=False)
        saved["scenario_recall"] = p
        _art("scenario_recall", "Scenario Recall Output",
             "Brand recall probability by respondent × scenario × brand",
             p, "leaderboard", row_count=len(scenario_recall_df),
             metadata={"scenarios": scenario_recall_df["scenario_name"].nunique(),
                       "brands": scenario_recall_df["brand_name"].nunique()})
        logger.info("Saved scenario recall: %s", p)
    except Exception as exc:
        logger.warning("scenario_recall failed: %s", exc)

    # ------------------------------------------------------------------
    # 3. Ad impact CSV
    # ------------------------------------------------------------------
    try:
        p = out_path / "ad_impact_output.csv"
        impact_df.to_csv(p, index=False)
        saved["ad_impact"] = p
        _art("ad_impact", "Ad Impact Output",
             "Pre/post recall delta by respondent × scenario × brand",
             p, "table", row_count=len(impact_df),
             metadata={"focal_brand": focal_brand_name,
                       "focal_scenario": focal_scenario})
        logger.info("Saved ad impact: %s", p)
    except Exception as exc:
        logger.warning("ad_impact failed: %s", exc)

    # ------------------------------------------------------------------
    # 4. Campaign flight simulator (summary CSV; chart only if not skip_charts)
    # ------------------------------------------------------------------
    try:
        if not skip_charts:
            fig, _ = plot_flight_simulator(
                impact_df, scenario_name=focal_scenario,
                focal_brand_name=focal_brand_name, top_n=8,
                title=f"Campaign flight simulator — {focal_brand_name} | {focal_scenario} | {country}",
            )
            p = out_path / "campaign_flight_simulator.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved["flight_simulator"] = p
            _art("chart", "Campaign Flight Simulator",
                 "Pre/post recall and competitive displacement chart",
                 p, "image")
            logger.info("Saved flight simulator: %s", p)

        summary = flight_simulator_summary(impact_df, focal_scenario)
        csv_p = out_path / "flight_simulator_summary.csv"
        summary.to_csv(csv_p, index=False)
        saved["flight_simulator_summary"] = csv_p
        _art("flight_simulator", "Flight Simulator Summary",
             "Aggregated pre/post recall table for the focal scenario",
             csv_p, "table", row_count=len(summary),
             metadata={"focal_brand": focal_brand_name, "focal_scenario": focal_scenario})
        logger.info("Saved flight simulator summary: %s", csv_p)
    except Exception as exc:
        logger.warning("flight_simulator failed: %s", exc)

    # ------------------------------------------------------------------
    # 5. Memory map comparison (chart only if not skip_charts)
    # ------------------------------------------------------------------
    if not skip_charts:
        try:
            fig, _ = plot_memory_map_comparison(
                rbc_pre, cep_master_df, focal_brand_id=focal_brand_id,
                top_n_brands=8, top_n_ceps=10,
                title=f"Respondent memory maps — {focal_brand_name} loyalist vs competitor | {country}",
            )
            p = out_path / "memory_map_comparison.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved["memory_map_comparison"] = p
            _art("chart", "Memory Map Comparison",
                 "Side-by-side brand loyalist memory heatmaps",
                 p, "heatmap")
            logger.info("Saved memory map comparison: %s", p)
        except Exception as exc:
            logger.warning("memory_map_comparison failed: %s", exc)

    # ------------------------------------------------------------------
    # 6. Calibration dashboard (chart + diagnostics CSV + markdown report)
    # ------------------------------------------------------------------
    try:
        cal_df      = run_calibration_check(scenario_recall_df, long_df)
        spearman_df = run_spearman_validity(scenario_recall_df, long_df)
        diag_df     = run_scenario_diagnostics(scenario_recall_df, long_df)

        if not skip_charts:
            fig, _ = plot_calibration_dashboard(
                cal_df, spearman_df, diag_df,
                title=f"Calibration & trust dashboard | {country}",
            )
            p = out_path / "calibration_dashboard.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved["calibration_dashboard"] = p
            _art("chart", "Calibration Dashboard",
                 "Calibration scatter, per-scenario Spearman ρ, and per-scenario MAE",
                 p, "image")
            logger.info("Saved calibration dashboard: %s", p)

        diag_p = out_path / "scenario_diagnostics.csv"
        diag_df.to_csv(diag_p, index=False)
        saved["scenario_diagnostics"] = diag_p
        _art("model_diagnostics", "Scenario Diagnostics",
             "Per-scenario MAE, Spearman ρ, coverage, top over/under-predicted brands",
             diag_p, "table", row_count=len(diag_df))
        logger.info("Saved scenario diagnostics: %s", diag_p)

        # Markdown calibration report
        report_md = _build_calibration_md(
            cal_df, spearman_df, diag_df,
            holdout_mae=holdout_mae, median_spearman=median_spearman,
            country=country,
        )
        md_p = out_path / "calibration_report.md"
        md_p.write_text(report_md)
        saved["calibration_report"] = md_p
        _art("report", "Calibration Report",
             "Human-readable calibration and construct validity summary",
             md_p, "markdown")
        logger.info("Saved calibration report: %s", md_p)

    except Exception as exc:
        logger.warning("calibration outputs failed: %s", exc)
        cal_df      = pd.DataFrame()
        spearman_df = pd.DataFrame()
        diag_df     = pd.DataFrame()

    # ------------------------------------------------------------------
    # 7. Segment summary CSV (optional — passed in from caller)
    # ------------------------------------------------------------------
    if segment_summary_df is not None and not segment_summary_df.empty:
        try:
            p = out_path / "segment_summary.csv"
            segment_summary_df.to_csv(p, index=False)
            saved["segment_summary"] = p
            _art("segment_summary", "Segment Summary",
                 "Ad impact breakdown by respondent segment × scenario × brand",
                 p, "table", row_count=len(segment_summary_df))
            logger.info("Saved segment summary: %s", p)
        except Exception as exc:
            logger.warning("segment_summary failed: %s", exc)

    # ------------------------------------------------------------------
    # 8. Analysis summary bundle (LLM-ready JSON)
    # ------------------------------------------------------------------
    try:
        bundle = build_analysis_summary_bundle(
            run_id=run_id,
            node_id=node_id,
            scenario_recall_df=scenario_recall_df,
            impact_df=impact_df,
            cal_df=cal_df,
            spearman_df=spearman_df,
            diag_df=diag_df,
            focal_brand_id=focal_brand_id,
            focal_brand_name=focal_brand_name,
            focal_scenario=focal_scenario,
            config=config,
            holdout_mae=holdout_mae,
            median_spearman=median_spearman,
            chart_paths={k: str(v) for k, v in saved.items()
                         if str(v).endswith(".png")},
        )
        bundle_p = out_path / "analysis_summary_bundle.json"
        bundle_p.write_text(json.dumps(bundle, indent=2))
        saved["analysis_summary_bundle"] = bundle_p
        _art("summary_bundle", "Analysis Summary Bundle",
             "Curated LLM-ready JSON with key findings, diagnostics, and chart references",
             bundle_p, "json")
        logger.info("Saved analysis summary bundle: %s", bundle_p)
    except Exception as exc:
        logger.warning("analysis_summary_bundle failed: %s", exc)

    # ------------------------------------------------------------------
    # 9. Run manifest
    # ------------------------------------------------------------------
    manifest = RunManifest(
        run_id=run_id,
        node_id=node_id,
        node_type="cep_sim",
        status="success",
        started_at=started,
        config_summary={
            "market": country,
            "respondents": rbc_pre["respondent_id"].nunique(),
            "brands": rbc_pre["brand_id"].nunique(),
            "ceps": rbc_pre["cep_id"].nunique(),
            "scenarios": scenario_recall_df["scenario_name"].nunique()
            if not scenario_recall_df.empty else 0,
            "focal_brand": focal_brand_name,
            "focal_scenario": focal_scenario,
        },
        artifacts=artifacts,
    )
    manifest_p = write_manifest(manifest, out_path)
    saved["run_manifest"] = manifest_p
    logger.info("Saved run manifest: %s (%d artifacts)", manifest_p, len(artifacts))

    logger.info("Standard outputs complete — %d files in %s", len(saved), out_path)
    return saved


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def flight_simulator_summary(
    impact_df: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """
    Aggregated pre/post recall table for a focal scenario.

    Columns: brand_name, recall_pre, recall_post, delta, rank_pre, rank_post, rank_delta.
    Sorted by recall_post descending.
    """
    df = impact_df[impact_df["scenario_name"] == scenario_name].copy()
    if df.empty:
        raise ValueError(f"No data for scenario {scenario_name!r}")

    summary = (
        df.groupby("brand_name")[["recall_pre", "recall_post", "delta"]]
        .mean()
        .round(4)
    )
    summary["rank_pre"]   = summary["recall_pre"].rank(ascending=False, method="min").astype(int)
    summary["rank_post"]  = summary["recall_post"].rank(ascending=False, method="min").astype(int)
    summary["rank_delta"] = summary["rank_pre"] - summary["rank_post"]
    return summary.sort_values("recall_post", ascending=False).reset_index()


def build_analysis_summary_bundle(
    run_id: str,
    node_id: str,
    scenario_recall_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    focal_brand_id: str,
    focal_brand_name: str,
    focal_scenario: str,
    config: CepSimConfig,
    holdout_mae: float | None = None,
    median_spearman: float | None = None,
    chart_paths: dict[str, str] | None = None,
) -> dict:
    """
    Build the LLM-ready analysis summary bundle for a CEP simulator run.

    The bundle is a self-contained JSON object containing run metadata,
    key findings, diagnostics, and chart references. It is designed to be
    passed directly to an LLM / agent node for summarisation, interpretation,
    or slide generation.
    """
    country = getattr(config.survey, "country", "unknown")

    # --- run metadata ---
    run_meta = {
        "run_id":       run_id,
        "node_id":      node_id,
        "market":       country,
        "generated_at": utc_now(),
        "respondents":  int(scenario_recall_df["respondent_id"].nunique())
                        if not scenario_recall_df.empty else 0,
        "brands":       int(scenario_recall_df["brand_name"].nunique())
                        if not scenario_recall_df.empty else 0,
        "scenarios":    int(scenario_recall_df["scenario_name"].nunique())
                        if not scenario_recall_df.empty else 0,
        "focal_brand":  focal_brand_name,
        "focal_scenario": focal_scenario,
    }

    # --- baseline brand leaderboard (across all scenarios) ---
    baseline_top = []
    if not scenario_recall_df.empty:
        overall = (
            scenario_recall_df.groupby("brand_name")["recall_prob"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        for rank, (brand, prob) in enumerate(overall.items(), 1):
            baseline_top.append({"rank": rank, "brand": brand, "mean_recall_prob": round(float(prob), 4)})

    # --- focal scenario leaderboard ---
    focal_leaderboard = []
    if not scenario_recall_df.empty:
        scene_df = scenario_recall_df[scenario_recall_df["scenario_name"] == focal_scenario]
        if not scene_df.empty:
            focal_top = (
                scene_df.groupby("brand_name")["recall_prob"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            for rank, (brand, prob) in enumerate(focal_top.items(), 1):
                focal_leaderboard.append({
                    "rank": rank, "brand": brand,
                    "mean_recall_prob": round(float(prob), 4),
                })

    # --- ad impact for focal brand + competitor displacement ---
    focal_lift: dict = {}
    displaced: list = []
    if not impact_df.empty:
        scene_impact = impact_df[impact_df["scenario_name"] == focal_scenario]
        if not scene_impact.empty:
            agg = scene_impact.groupby("brand_name")[["recall_pre", "recall_post", "delta"]].mean().round(4)
            if focal_brand_name in agg.index:
                row = agg.loc[focal_brand_name]
                focal_lift = {
                    "brand":       focal_brand_name,
                    "recall_pre":  round(float(row["recall_pre"]),  4),
                    "recall_post": round(float(row["recall_post"]), 4),
                    "delta":       round(float(row["delta"]),        4),
                    "rank_pre":    int(agg["recall_pre"].rank(ascending=False, method="min")[focal_brand_name]),
                    "rank_post":   int(agg["recall_post"].rank(ascending=False, method="min")[focal_brand_name]),
                }
            # competitors sorted by delta ascending (biggest losers first)
            others = agg[agg.index != focal_brand_name].sort_values("delta")
            for brand, row in others.head(5).iterrows():
                displaced.append({
                    "brand": brand,
                    "delta": round(float(row["delta"]), 4),
                    "recall_pre":  round(float(row["recall_pre"]),  4),
                    "recall_post": round(float(row["recall_post"]), 4),
                })

    # --- diagnostics ---
    diag_summary: dict = {}
    if holdout_mae is not None:
        diag_summary["holdout_mae"] = round(holdout_mae, 4)
    elif not cal_df.empty and "abs_error" in cal_df.columns:
        diag_summary["train_mae"] = round(float(cal_df["abs_error"].mean()), 4)

    if median_spearman is not None:
        diag_summary["median_spearman_rho"] = round(median_spearman, 4)
    elif not spearman_df.empty and "spearman_rho" in spearman_df.columns:
        diag_summary["median_spearman_rho"] = round(float(spearman_df["spearman_rho"].median()), 4)

    strong_scenarios, weak_scenarios = [], []
    if not diag_df.empty and "spearman_rho" in diag_df.columns:
        strong_scenarios = diag_df[diag_df["spearman_rho"] >= 0.6]["scenario_name"].tolist()
        weak_scenarios   = diag_df[diag_df["spearman_rho"] <  0.2]["scenario_name"].tolist()
        diag_summary["strong_scenarios"] = strong_scenarios
        diag_summary["weak_scenarios"]   = weak_scenarios
        if not diag_df.empty and "mae" in diag_df.columns:
            worst = diag_df.nlargest(3, "mae")[["scenario_name", "mae"]].round(4)
            diag_summary["worst_3_scenarios"] = worst.to_dict(orient="records")

    caveats = []
    if weak_scenarios:
        caveats.append(
            f"{len(weak_scenarios)} scenario(s) have Spearman ρ < 0.2 "
            f"({', '.join(weak_scenarios[:3])}{'...' if len(weak_scenarios) > 3 else ''}) "
            "— likely a CEP mapping issue, not a model failure."
        )
    if holdout_mae and holdout_mae > 0.05:
        caveats.append(
            f"Holdout MAE is {holdout_mae:.3f} (above the 5pp target). "
            "Interpret rankings rather than absolute probabilities."
        )
    diag_summary["caveats"] = caveats

    return {
        "artifact_type":  "summary_bundle",
        "schema_version": "1.0",
        "run_id":         run_id,
        "node_id":        node_id,
        "run_metadata":   run_meta,
        "key_findings": {
            "baseline_brand_leaderboard":   baseline_top,
            "focal_scenario_leaderboard":   focal_leaderboard,
            "ad_lift":                      focal_lift,
            "biggest_displaced_competitors": displaced,
        },
        "diagnostics":    diag_summary,
        "chart_references": chart_paths or {},
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_calibration_md(
    cal_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    holdout_mae: float | None,
    median_spearman: float | None,
    country: str,
) -> str:
    lines = [
        f"# CEP Simulator — Calibration Report",
        f"",
        f"**Market:** {country}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    mae = holdout_mae if holdout_mae is not None else (
        float(cal_df["abs_error"].mean()) if not cal_df.empty and "abs_error" in cal_df.columns else None
    )
    rho = median_spearman if median_spearman is not None else (
        float(spearman_df["spearman_rho"].median())
        if not spearman_df.empty and "spearman_rho" in spearman_df.columns else None
    )
    mae_label = "Holdout MAE" if holdout_mae is not None else "Train MAE"
    rho_label = "Holdout median Spearman ρ" if median_spearman is not None else "Median Spearman ρ"

    def _flag(v, t5, t2): return "✅ PASS" if v < t2 else ("⚠️ CLOSE" if v < t5 else "❌ FAIL")
    def _rho_label(v): return "strong" if v > 0.6 else ("moderate" if v > 0.3 else "weak")

    lines += ["## Summary", ""]
    if mae is not None:
        lines.append(f"| {mae_label} | {mae:.4f} | {_flag(mae, 0.05, 0.02)} |")
    if rho is not None:
        lines.append(f"| {rho_label} | {rho:.4f} | {_rho_label(rho)} |")
    lines.append("")

    if not diag_df.empty and "spearman_rho" in diag_df.columns:
        lines += ["## Per-scenario diagnostics", ""]
        lines.append("| Scenario | MAE | Spearman ρ | Note |")
        lines.append("|---|---|---|---|")
        for _, row in diag_df.sort_values("mae", ascending=False).iterrows():
            note = " [weak]" if row["spearman_rho"] < 0.2 else ""
            lines.append(
                f"| {row['scenario_name']} | {row['mae']:.4f} | {row['spearman_rho']:.4f} |{note} |"
            )
        lines.append("")

    if not spearman_df.empty and "spearman_rho" in spearman_df.columns:
        weak = spearman_df[spearman_df["spearman_rho"] < 0.2]["scenario_name"].tolist()
        if weak:
            lines += [
                "## Scenarios flagged for review",
                "",
                "> Spearman ρ < 0.2 indicates the model's brand ranking diverges from "
                "observed survey data for this scenario. Likely cause: imprecise CEP "
                "keyword matching or low respondent coverage.",
                "",
            ]
            for s in weak:
                lines.append(f"- `{s}`")
            lines.append("")

    lines += [
        "## What is working",
        "",
        "- Brand rank ordering (Spearman ρ) is the primary validity signal.",
        "- Absolute recall probabilities are calibrated indicators, not purchase probabilities.",
        "- Ad lift direction is reliable; magnitude is sensitive to temperature (τ).",
        "",
        "## What still needs hardening",
        "",
        "- Scenarios with ρ < 0.2 need CEP mapping review.",
        "- MAE > 5pp indicates the model over- or under-predicts brand accessibility "
        "in aggregate — interpret rankings, not absolute values.",
    ]

    return "\n".join(lines)
