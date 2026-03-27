"""Plotly interactive chart builders for the CEP simulator UI.

Each function returns a Plotly figure as a JSON string that can be embedded
directly in the API response and rendered client-side with Plotly.js.
"""
from __future__ import annotations
import pandas as pd


_LAYOUT = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#94a3b8", size=11),
)


def flight_chart(
    impact_df: pd.DataFrame,
    focal_scenario: str,
    focal_brand_name: str,
    top_n: int = 12,
) -> str:
    """Competitive displacement bar chart (delta only, horizontal)."""
    import plotly.graph_objects as go

    df = impact_df[impact_df["scenario_name"] == focal_scenario].copy()
    if df.empty:
        return "{}"

    agg = (
        df.groupby("brand_name")[["recall_pre", "recall_post", "delta"]]
        .mean()
        .nlargest(top_n, "recall_post")
        .sort_values("delta", ascending=True)
    )

    colors = []
    for name, row in agg.iterrows():
        if name == focal_brand_name:
            colors.append("#22c55e" if row["delta"] >= 0 else "#ef4444")
        else:
            colors.append("#f97316" if row["delta"] < 0 else "#86efac")

    fig = go.Figure(go.Bar(
        x=agg["delta"].values,
        y=agg.index.tolist(),
        orientation="h",
        marker_color=colors,
        customdata=list(zip(
            (agg["recall_pre"] * 100).round(2),
            (agg["recall_post"] * 100).round(2),
            (agg["delta"] * 100).round(3),
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pre:  %{customdata[0]:.1f}%<br>"
            "Post: %{customdata[1]:.1f}%<br>"
            "Δ:    %{customdata[2]:+.3f} pp"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color="#475569", line_width=1)

    fig.update_layout(
        **_LAYOUT,
        margin=dict(l=180, r=90, t=50, b=50),
        title=dict(
            text=f"Competitive displacement — {focal_scenario.replace('_', ' ')}",
            font=dict(size=13, color="#e2e8f0"),
        ),
        xaxis=dict(
            title="Mean recall delta (post − pre)",
            gridcolor="#334155",
            tickformat="+.3f",
            zerolinecolor="#475569",
        ),
        yaxis=dict(gridcolor="#334155", ticklabelstandoff=10),
        height=max(320, len(agg) * 40 + 100),
        showlegend=False,
    )
    return fig.to_json()


def memory_map_chart(
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    focal_brand_id: str,
    top_n_brands: int = 8,
    top_n_ceps: int = 10,
) -> str:
    """Side-by-side heatmaps: brand loyalist vs competitor loyalist."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    label_col = "cep_description" if "cep_description" in cep_master_df.columns else "cep_label"
    label_map = cep_master_df.set_index("cep_id")[label_col].to_dict()

    focal_mask = rbc_df["brand_id"] == focal_brand_id
    focal_strength = rbc_df[focal_mask].groupby("respondent_id")["assoc_strength"].sum()
    rival_strength = rbc_df[~focal_mask].groupby("respondent_id")["assoc_strength"].sum()
    if focal_strength.empty or rival_strength.empty:
        return "{}"

    resp_a = focal_strength.idxmax()
    resp_b = rival_strength.idxmax()

    focal_rows = rbc_df[focal_mask]
    focal_brand_name = focal_rows["brand_name"].iloc[0] if len(focal_rows) else focal_brand_id

    top_brands = (
        rbc_df[rbc_df["respondent_id"].isin([resp_a, resp_b])]
        .groupby("brand_name")["assoc_strength"].sum()
        .nlargest(top_n_brands).index.tolist()
    )
    top_ceps = (
        rbc_df.groupby("cep_id")["assoc_strength"].sum()
        .nlargest(top_n_ceps).index.tolist()
    )
    # Full labels used as y values (shown in hover via %{y})
    top_cep_labels = [label_map.get(c, c) for c in top_ceps]
    # Truncated labels used as axis tick display text
    _MAX = 32
    tick_display = [
        (lbl if len(lbl) <= _MAX else lbl[:_MAX - 1] + "…")
        for lbl in top_cep_labels
    ]

    def _pivot(resp_id):
        sub = rbc_df[
            (rbc_df["respondent_id"] == resp_id)
            & (rbc_df["brand_name"].isin(top_brands))
            & (rbc_df["cep_id"].isin(top_ceps))
        ]
        p = sub.pivot_table(
            index="cep_id", columns="brand_name",
            values="assoc_strength", aggfunc="sum", fill_value=0,
        )
        p = p.reindex(index=top_ceps, columns=top_brands, fill_value=0)
        p.index = top_cep_labels
        return p

    pivot_a = _pivot(resp_a)
    pivot_b = _pivot(resp_b)
    vmax = max(float(pivot_a.values.max()), float(pivot_b.values.max()), 0.01)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f"{focal_brand_name} loyalist", "Competitor loyalist"],
        vertical_spacing=0.10,
    )

    # customdata: 2-D array (n_ceps × n_brands) carrying the full CEP label per cell
    full_label_grid = [[lbl for _ in top_brands] for lbl in top_cep_labels]

    for row_idx, pivot in [(1, pivot_a), (2, pivot_b)]:
        text_vals = [
            [f"{v:.2f}" if v > 0 else "" for v in row]
            for row in pivot.values
        ]
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=top_brands,
                y=tick_display,          # truncated labels on axis
                customdata=full_label_grid,  # full labels in hover
                colorscale="YlOrRd",
                zmin=0,
                zmax=vmax,
                showscale=(row_idx == 2),
                colorbar=dict(
                    title=dict(text="Assoc.", side="right"),
                    thickness=12,
                ) if row_idx == 2 else None,
                hovertemplate="<b>%{x}</b><br>%{customdata}<br>Strength: %{z:.3f}<extra></extra>",
                text=text_vals,
                texttemplate="<b>%{text}</b>",
                textfont=dict(size=9, color="#1e293b", family="Arial Black, sans-serif"),
            ),
            row=row_idx, col=1,
        )

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text="Memory maps — brand loyalists comparison",
            font=dict(size=13, color="#e2e8f0"),
        ),
        margin=dict(l=220, r=80, t=80, b=60),
        height=max(600, top_n_ceps * 38 * 2 + 200),
    )
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10, family="Arial Black, sans-serif"))
    fig.update_yaxes(tickfont=dict(size=9, family="Arial Black, sans-serif"), ticklabelstandoff=10)
    return fig.to_json()


def calibration_chart(
    cal_df: pd.DataFrame,
    spearman_df: pd.DataFrame | None,
) -> str:
    """Calibration scatter + per-scenario Spearman ρ."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if cal_df is None or cal_df.empty:
        return "{}"

    has_spearman = spearman_df is not None and not spearman_df.empty
    mae = (
        cal_df.attrs.get("mae", cal_df["abs_error"].mean())
        if "abs_error" in cal_df.columns else None
    )
    mae_str = f"MAE = {mae:.3f}" if mae is not None else ""

    titles = [f"Calibration  ({mae_str})"]
    if has_spearman:
        med = spearman_df["spearman_rho"].median()
        titles.append(f"Spearman ρ by scenario  (median = {med:.3f})")

    row_heights = [0.52, 0.48] if has_spearman else [1.0]
    fig = make_subplots(
        rows=2 if has_spearman else 1,
        cols=1,
        subplot_titles=titles,
        row_heights=row_heights,
        vertical_spacing=0.14,
    )

    # ── Scatter ──────────────────────────────────────────────────────
    lim = max(
        float(cal_df["observed_mention_rate"].max()),
        float(cal_df["predicted_recall_prob"].max()),
    ) * 1.12

    fig.add_trace(
        go.Scatter(
            x=[0, lim], y=[0, lim],
            mode="lines",
            line=dict(dash="dash", color="#64748b", width=1),
            name="y = x",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )

    hover_cols = ["observed_mention_rate", "predicted_recall_prob"]
    if "abs_error" in cal_df.columns:
        hover_cols.append("abs_error")
    customdata = cal_df[hover_cols].values

    extra_row = "<br>Abs error: %{customdata[2]:.3f}" if "abs_error" in cal_df.columns else ""
    fig.add_trace(
        go.Scatter(
            x=cal_df["observed_mention_rate"],
            y=cal_df["predicted_recall_prob"],
            mode="markers",
            marker=dict(color="#3b82f6", size=9, opacity=0.85),
            text=cal_df["brand_name"],
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Observed:  %{customdata[0]:.3f}<br>"
                "Predicted: %{customdata[1]:.3f}"
                + extra_row +
                "<extra></extra>"
            ),
            name="Brands",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Observed mention rate", range=[0, lim], gridcolor="#334155", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Predicted recall prob", range=[0, lim], gridcolor="#334155", zeroline=False, row=1, col=1)

    # ── Spearman ρ ───────────────────────────────────────────────────
    if has_spearman:
        sp = spearman_df.sort_values("spearman_rho", ascending=True)
        bar_colors = [
            "#22c55e" if r >= 0.6 else "#f97316" if r >= 0.3 else "#ef4444"
            for r in sp["spearman_rho"]
        ]
        fig.add_trace(
            go.Bar(
                x=sp["spearman_rho"],
                y=sp["scenario_name"].str.replace("_", " "),
                orientation="h",
                marker_color=bar_colors,
                hovertemplate="<b>%{y}</b><br>ρ = %{x:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=2, col=1,
        )
        for xval, clr in [(0.6, "#22c55e"), (0.3, "#f97316")]:
            fig.add_vline(x=xval, line_color=clr, line_dash="dot", line_width=1, row=2, col=1)
        fig.update_xaxes(title_text="Spearman ρ", gridcolor="#334155", row=2, col=1)
        fig.update_yaxes(tickfont=dict(size=8), gridcolor="#334155", ticklabelstandoff=10, row=2, col=1)

    # Compute left margin from longest scenario label (approx 7px per char)
    if has_spearman:
        max_label_len = spearman_df["scenario_name"].str.replace("_", " ").str.len().max()
        left_margin = max(160, int(max_label_len * 7))
    else:
        left_margin = 60

    fig.update_layout(
        **_LAYOUT,
        margin=dict(l=left_margin, r=40, t=80, b=50),
        height=720 if has_spearman else 460,
    )
    return fig.to_json()
