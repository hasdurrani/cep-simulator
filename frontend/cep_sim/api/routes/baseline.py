"""GET /api/baseline/{session_id} — market-level baseline charts and stats."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]


@router.get("/baseline/{session_id}")
def get_baseline(session_id: str):
    from frontend.cep_sim.api import session as session_store
    from backend.analysis.cep_sim.service.plotting import (
        plot_brand_situation_heatmap,
        plot_brand_leaderboard,
    )
    import matplotlib.pyplot as plt

    sess = session_store.get(session_id)
    if sess is None:
        raise HTTPException(404, "Session not found. Please run /api/setup first.")

    out_dir = PROJECT_ROOT / "outputs" / "cep_sim" / "ui_sessions" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path    = out_dir / "brand_situation_heatmap.png"
    leaderboard_path = out_dir / "brand_leaderboard.png"

    # Only regenerate if the files don't already exist
    if not heatmap_path.exists() or not leaderboard_path.exists():
        try:
            fig, _ = plot_brand_situation_heatmap(sess.scenario_recall_df)
            fig.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            raise HTTPException(500, f"Failed to generate brand situation heatmap: {exc}")

        try:
            fig, _ = plot_brand_leaderboard(sess.scenario_recall_df)
            fig.savefig(str(leaderboard_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            raise HTTPException(500, f"Failed to generate brand leaderboard: {exc}")

    # Build brand_leaderboard response list
    import pandas as pd
    ranked = (
        sess.scenario_recall_df.groupby("brand_name")["recall_prob"]
        .mean()
        .nlargest(15)
        .sort_values(ascending=False)
        .reset_index()
    )
    brand_leaderboard = [
        {
            "rank": i + 1,
            "brand_name": row["brand_name"],
            "mean_recall_pct": round(float(row["recall_prob"]) * 100, 2),
        }
        for i, row in ranked.iterrows()
    ]
    # Reset rank to be sequential
    for i, entry in enumerate(brand_leaderboard):
        entry["rank"] = i + 1

    # Build scenario_summary response list
    scenario_summary_rows = (
        sess.scenario_recall_df
        .groupby("scenario_name")
        .agg(
            mean_recall=("recall_prob", "mean"),
            n_brands=("brand_name", "nunique"),
        )
        .reset_index()
        .sort_values("mean_recall", ascending=False)
    )

    def _top_brand(scenario_name: str) -> str:
        sub = sess.scenario_recall_df[sess.scenario_recall_df["scenario_name"] == scenario_name]
        if sub.empty:
            return "—"
        return sub.groupby("brand_name")["recall_prob"].mean().idxmax()

    scenario_summary = [
        {
            "scenario_name": row["scenario_name"],
            "top_brand": _top_brand(row["scenario_name"]),
            "mean_recall_pct": round(float(row["mean_recall"]) * 100, 2),
            "n_brands": int(row["n_brands"]),
        }
        for _, row in scenario_summary_rows.iterrows()
    ]

    respondent_count = int(len(sess.respondent_ids))
    brand_count      = int(sess.scenario_recall_df["brand_name"].nunique())
    scenario_count   = int(sess.scenario_recall_df["scenario_name"].nunique())

    # Build brand × scenario matrix for inline heatmap rendering
    # Scenarios ordered by mean recall desc; brands ordered by mean recall desc
    pivot = (
        sess.scenario_recall_df
        .groupby(["brand_name", "scenario_name"])["recall_prob"]
        .mean()
        .unstack(fill_value=0.0)
    )
    scenario_order = (
        sess.scenario_recall_df.groupby("scenario_name")["recall_prob"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    brand_order = (
        sess.scenario_recall_df.groupby("brand_name")["recall_prob"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Only keep scenarios that appear in pivot columns
    scenario_order = [s for s in scenario_order if s in pivot.columns]
    brand_order    = [b for b in brand_order    if b in pivot.index]
    pivot = pivot.loc[brand_order, scenario_order]

    brand_scenario_matrix = {
        "scenarios": scenario_order,
        "rows": [
            {
                "brand": brand,
                "values": [round(float(pivot.loc[brand, s]) * 100, 1) for s in scenario_order],
            }
            for brand in brand_order
        ],
    }

    artifact_base_url = f"/api/artifacts/{session_id}"

    return {
        "session_id": session_id,
        "artifact_base_url": artifact_base_url,
        "brand_leaderboard": brand_leaderboard,
        "scenario_summary": scenario_summary,
        "brand_scenario_matrix": brand_scenario_matrix,
        "respondent_count": respondent_count,
        "brand_count": brand_count,
        "scenario_count": scenario_count,
    }
