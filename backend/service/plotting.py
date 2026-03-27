"""
Plotting helpers for the CEP simulator demo.

All functions return a (fig, ax) or (fig, axes) tuple so callers can
save, display, or further annotate as needed.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def plot_memory_map(
    rbc_df: pd.DataFrame,
    respondent_id: str,
    cep_master_df: pd.DataFrame,
    top_n_brands: int = 10,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of brand × CEP association strengths for a single respondent.

    Parameters
    ----------
    rbc_df        : respondent-brand-CEP edge table
    respondent_id : the respondent to visualise
    cep_master_df : used to map cep_id → cep_label
    top_n_brands  : only show the top-N brands by total strength
    """
    subset = rbc_df[rbc_df["respondent_id"] == respondent_id].copy()
    if subset.empty:
        raise ValueError(f"No data for respondent {respondent_id!r}")

    # Attach readable CEP labels
    label_col = "cep_description" if "cep_description" in cep_master_df.columns else "cep_label"
    label_map = cep_master_df.set_index("cep_id")[label_col].to_dict()
    subset["cep_label"] = subset["cep_id"].map(label_map).fillna(subset["cep_id"])

    # Keep only top-N brands by total strength
    top_brands = (
        subset.groupby("brand_name")["assoc_strength"]
        .sum()
        .nlargest(top_n_brands)
        .index
    )
    subset = subset[subset["brand_name"].isin(top_brands)]

    pivot = subset.pivot_table(
        index="brand_name",
        columns="cep_label",
        values="assoc_strength",
        aggfunc="sum",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.9), max(4, pivot.shape[0] * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Association strength")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Category Entry Point")
    ax.set_ylabel("Brand")
    ax.set_title(title or f"Memory map — respondent {respondent_id}")
    fig.tight_layout()
    return fig, ax


def plot_pre_post_recall(
    impact_df: pd.DataFrame,
    scenario_name: str,
    top_n: int = 10,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grouped bar chart of mean recall score before and after ad exposure,
    for a single scenario, aggregated across respondents.

    Parameters
    ----------
    impact_df     : output of run_ad_impact(), with recall_pre / recall_post
    scenario_name : which scenario to plot
    top_n         : show only top-N brands by post-exposure recall
    """
    df = impact_df[impact_df["scenario_name"] == scenario_name].copy()
    if df.empty:
        raise ValueError(f"No data for scenario {scenario_name!r}")

    agg = (
        df.groupby("brand_name")[["recall_pre", "recall_post"]]
        .mean()
        .nlargest(top_n, "recall_post")
        .sort_values("recall_post", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.5)))
    y = np.arange(len(agg))
    h = 0.35

    ax.barh(y - h / 2, agg["recall_pre"],  height=h, label="Pre-exposure",  color="#6baed6")
    ax.barh(y + h / 2, agg["recall_post"], height=h, label="Post-exposure", color="#fd8d3c")

    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=9)
    ax.set_xlabel("Mean recall score")
    ax.set_title(title or f"Pre/post recall — {scenario_name}")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    return fig, ax


def plot_recall_ranking(
    scenario_recall_df: pd.DataFrame,
    scenario_name: str,
    top_n: int = 10,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart of mean recall probability, ranked highest → lowest,
    for a single scenario.

    Parameters
    ----------
    scenario_recall_df : output of run_scenario_recall()
    scenario_name      : which scenario to plot
    top_n              : how many brands to include
    """
    df = scenario_recall_df[scenario_recall_df["scenario_name"] == scenario_name].copy()
    if df.empty:
        raise ValueError(f"No data for scenario {scenario_name!r}")

    ranked = (
        df.groupby("brand_name")["recall_prob"]
        .mean()
        .nlargest(top_n)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(3, len(ranked) * 0.45)))
    bars = ax.barh(ranked.index, ranked.values, color="#4292c6")

    # Value labels at end of each bar
    for bar, val in zip(bars, ranked.values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Mean recall probability")
    ax.set_title(title or f"Brand recall ranking — {scenario_name}")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0, ranked.values.max() * 1.18)
    fig.tight_layout()
    return fig, ax


def plot_calibration(
    cal_df: pd.DataFrame,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of predicted recall probability vs observed mention rate,
    with a 45° identity line and brand labels.

    Parameters
    ----------
    cal_df : output of run_calibration_check(), with columns:
             brand_name, predicted_recall_prob, observed_mention_rate, abs_error
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        cal_df["observed_mention_rate"],
        cal_df["predicted_recall_prob"],
        s=60,
        color="#2171b5",
        zorder=3,
    )

    for _, row in cal_df.iterrows():
        ax.annotate(
            row["brand_name"],
            (row["observed_mention_rate"], row["predicted_recall_prob"]),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
        )

    # 45° identity line
    lim_max = max(
        cal_df["observed_mention_rate"].max(),
        cal_df["predicted_recall_prob"].max(),
    ) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "--", color="#aaa", linewidth=1, label="y = x")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    mae = cal_df.attrs.get("mae", cal_df["abs_error"].mean())
    ax.set_xlabel("Observed mention rate (survey)")
    ax.set_ylabel("Predicted recall probability (model)")
    ax.set_title(title or f"Calibration check  (MAE = {mae:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_flight_simulator(
    impact_df: pd.DataFrame,
    scenario_name: str,
    focal_brand_name: str,
    top_n: int = 8,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Before/after campaign flight simulator for a single scenario.

    Left panel  : pre/post horizontal bars for top-N brands (focal brand highlighted).
    Right panel : delta bars sorted descending; focal brand highlighted; green/red fill.

    Parameters
    ----------
    impact_df        : output of run_ad_impact()
    scenario_name    : which scenario to plot
    focal_brand_name : brand to highlight (the advertised brand)
    top_n            : how many brands to show
    """
    df = impact_df[impact_df["scenario_name"] == scenario_name].copy()
    if df.empty:
        raise ValueError(f"No data for scenario {scenario_name!r}")

    agg = (
        df.groupby("brand_name")[["recall_pre", "recall_post", "delta"]]
        .mean()
        .nlargest(top_n, "recall_post")
        .sort_values("recall_post", ascending=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(agg) * 0.55)))

    y     = np.arange(len(agg))
    h     = 0.38
    names = agg.index.tolist()

    focal_color_pre  = "#1a6b2a"
    focal_color_post = "#2ca644"
    other_color_pre  = "#9ecae1"
    other_color_post = "#3182bd"

    pre_colors  = [focal_color_pre  if n == focal_brand_name else other_color_pre  for n in names]
    post_colors = [focal_color_post if n == focal_brand_name else other_color_post for n in names]

    # --- left: pre/post bars ---
    axes[0].barh(y - h / 2, agg["recall_pre"],  height=h, color=pre_colors,  label="Pre-campaign",  zorder=2)
    axes[0].barh(y + h / 2, agg["recall_post"], height=h, color=post_colors, label="Post-campaign", zorder=2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_xlabel("Mean recall probability")
    axes[0].set_title("Recall before / after campaign")
    axes[0].legend(fontsize=8)
    axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    axes[0].grid(axis="x", alpha=0.3, zorder=0)

    # --- right: delta bars ---
    delta_sorted = agg["delta"].sort_values(ascending=True)
    d_names = delta_sorted.index.tolist()
    d_y     = np.arange(len(delta_sorted))
    d_colors = []
    for n, v in zip(d_names, delta_sorted.values):
        if n == focal_brand_name:
            d_colors.append("#1a6b2a" if v >= 0 else "#a63220")
        else:
            d_colors.append("#74c476" if v >= 0 else "#fc8d59")

    axes[1].barh(d_y, delta_sorted.values, color=d_colors, zorder=2)
    axes[1].axvline(0, color="#444", linewidth=0.8)
    for i, (n, v) in enumerate(zip(d_names, delta_sorted.values)):
        axes[1].text(
            v + (0.0003 if v >= 0 else -0.0003),
            i,
            f"{v:+.4f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=7.5,
        )
    axes[1].set_yticks(d_y)
    axes[1].set_yticklabels(d_names, fontsize=9)
    axes[1].set_xlabel("Mean recall delta (post − pre)")
    axes[1].set_title(f"Competitive displacement\n(↑ gains  ↓ loses  |  focal: {focal_brand_name})")
    axes[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    axes[1].grid(axis="x", alpha=0.3, zorder=0)

    fig.suptitle(
        title or f"Campaign flight simulator — {scenario_name}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig, axes


def plot_memory_map_comparison(
    rbc_df: pd.DataFrame,
    cep_master_df: pd.DataFrame,
    focal_brand_id: str,
    top_n_brands: int = 8,
    top_n_ceps: int = 10,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side memory map heatmaps for two representative respondents.

    Left  : respondent with the strongest focal-brand memory (brand loyalist).
    Right : respondent with the strongest competing-brand memory.

    Rows = top CEPs by total population weight, columns = top brands.
    Values = assoc_strength (white = 0, dark = strong).

    Parameters
    ----------
    rbc_df          : respondent-brand-CEP edge table (pre-ad)
    cep_master_df   : used to map cep_id → cep_label
    focal_brand_id  : e.g. "brand_heineken" — used to identify the two respondents
    top_n_brands    : columns shown in each heatmap
    top_n_ceps      : rows shown in each heatmap
    """
    label_col = "cep_description" if "cep_description" in cep_master_df.columns else "cep_label"
    label_map = cep_master_df.set_index("cep_id")[label_col].to_dict()

    # Pick respondent A: strongest focal-brand total weight
    focal_strength = (
        rbc_df[rbc_df["brand_id"] == focal_brand_id]
        .groupby("respondent_id")["assoc_strength"]
        .sum()
    )
    resp_a = focal_strength.idxmax() if len(focal_strength) else None

    # Pick respondent B: strongest competing-brand weight (excluding focal brand)
    rival_strength = (
        rbc_df[rbc_df["brand_id"] != focal_brand_id]
        .groupby("respondent_id")["assoc_strength"]
        .sum()
    )
    resp_b = rival_strength.idxmax() if len(rival_strength) else None

    if resp_a is None or resp_b is None:
        raise ValueError("Could not identify representative respondents.")

    # Top brands and CEPs across both respondents
    top_brands = (
        rbc_df[rbc_df["respondent_id"].isin([resp_a, resp_b])]
        .groupby("brand_name")["assoc_strength"]
        .sum()
        .nlargest(top_n_brands)
        .index.tolist()
    )
    top_ceps = (
        rbc_df.groupby("cep_id")["assoc_strength"]
        .sum()
        .nlargest(top_n_ceps)
        .index.tolist()
    )
    top_cep_labels = [label_map.get(c, c) for c in top_ceps]

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
        # Align to consistent ordering
        p = p.reindex(index=top_ceps, columns=top_brands, fill_value=0)
        p.index = top_cep_labels
        return p

    pivot_a = _pivot(resp_a)
    pivot_b = _pivot(resp_b)
    vmax = max(pivot_a.values.max(), pivot_b.values.max(), 0.01)

    focal_brand_name = (
        rbc_df[rbc_df["brand_id"] == focal_brand_id]["brand_name"].iloc[0]
        if len(rbc_df[rbc_df["brand_id"] == focal_brand_id]) else focal_brand_id
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(12, top_n_brands * 1.2), max(5, top_n_ceps * 0.55)),
        sharey=True,
    )

    for ax, pivot, resp_id, label in [
        (axes[0], pivot_a, resp_a, f"{focal_brand_name} loyalist"),
        (axes[1], pivot_b, resp_b, "Competitor loyalist"),
    ]:
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(top_brands)))
        ax.set_xticklabels(top_brands, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(top_cep_labels)))
        ax.set_yticklabels(top_cep_labels, fontsize=8)
        ax.set_title(f"{label}\n(respondent {resp_id})", fontsize=9)

        # Annotate cells with non-zero values
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                if v > 0:
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black" if v < vmax * 0.6 else "white")

    plt.colorbar(im, ax=axes, label="Association strength", shrink=0.6)
    fig.suptitle(
        title or "Respondent memory maps — brand loyalists comparison",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig, axes


def plot_calibration_dashboard(
    cal_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Three-panel calibration trust dashboard.

    Top-left  : Predicted vs observed scatter (calibration plot).
    Top-right : Per-scenario Spearman ρ (bar chart, threshold lines).
    Bottom    : Per-scenario MAE (bar chart, target lines).

    Parameters
    ----------
    cal_df       : output of run_calibration_check()
    spearman_df  : output of run_spearman_validity()
    diag_df      : output of run_scenario_diagnostics()
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_rho     = fig.add_subplot(gs[0, 1])
    ax_mae     = fig.add_subplot(gs[1, :])

    # --- calibration scatter ---
    ax_scatter.scatter(
        cal_df["observed_mention_rate"],
        cal_df["predicted_recall_prob"],
        s=55, color="#2171b5", zorder=3,
    )
    for _, row in cal_df.iterrows():
        ax_scatter.annotate(
            row["brand_name"],
            (row["observed_mention_rate"], row["predicted_recall_prob"]),
            fontsize=6.5, xytext=(3, 2), textcoords="offset points",
        )
    lim = max(
        cal_df["observed_mention_rate"].max(),
        cal_df["predicted_recall_prob"].max(),
    ) * 1.12
    ax_scatter.plot([0, lim], [0, lim], "--", color="#aaa", linewidth=1, label="y = x")
    ax_scatter.set_xlim(0, lim)
    ax_scatter.set_ylim(0, lim)
    mae = cal_df.attrs.get("mae", cal_df["abs_error"].mean())
    ax_scatter.set_xlabel("Observed mention rate")
    ax_scatter.set_ylabel("Predicted recall prob")
    ax_scatter.set_title(f"Calibration  (MAE = {mae:.3f})", fontsize=9)
    ax_scatter.legend(fontsize=7)

    # --- per-scenario Spearman ---
    if not spearman_df.empty:
        sp = spearman_df.sort_values("spearman_rho", ascending=True)
        colors = ["#74c476" if r >= 0.6 else ("#fdae6b" if r >= 0.3 else "#fc8d59")
                  for r in sp["spearman_rho"]]
        ax_rho.barh(sp["scenario_name"], sp["spearman_rho"], color=colors, zorder=2)
        ax_rho.axvline(0.6, color="green",  linestyle="--", linewidth=1, label="strong (0.6)")
        ax_rho.axvline(0.3, color="orange", linestyle="--", linewidth=1, label="moderate (0.3)")
        ax_rho.axvline(0.2, color="red",    linestyle=":",  linewidth=1, label="flag (0.2)")
        ax_rho.set_xlabel("Spearman ρ")
        ax_rho.set_title(f"Construct validity — median ρ = {spearman_df['spearman_rho'].median():.3f}", fontsize=9)
        ax_rho.legend(fontsize=6.5)
        ax_rho.tick_params(axis="y", labelsize=7.5)
        ax_rho.grid(axis="x", alpha=0.3, zorder=0)

    # --- per-scenario MAE ---
    if not diag_df.empty:
        d = diag_df.sort_values("mae", ascending=False)
        bar_colors = ["#fc8d59" if m > 0.05 else "#74c476" for m in d["mae"]]
        bars = ax_mae.bar(range(len(d)), d["mae"], color=bar_colors, width=0.6, zorder=2)
        ax_mae.axhline(0.05, color="red",   linestyle="--", linewidth=1, label="5 pp target")
        ax_mae.axhline(0.02, color="green", linestyle="--", linewidth=1, label="2 pp target")
        for bar, val in zip(bars, d["mae"]):
            ax_mae.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )
        ax_mae.set_xticks(range(len(d)))
        ax_mae.set_xticklabels(d["scenario_name"], rotation=38, ha="right", fontsize=8)
        ax_mae.set_ylabel("MAE")
        ax_mae.set_title("Per-scenario MAE", fontsize=9)
        ax_mae.legend(fontsize=7.5)
        ax_mae.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle(title or "Calibration & trust dashboard", fontsize=12, fontweight="bold")
    return fig, np.array([ax_scatter, ax_rho, ax_mae])


def plot_brand_situation_heatmap(
    scenario_recall_df: pd.DataFrame,
    top_n_brands: int = 12,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of mean recall probability per brand × scenario.

    Brands are the top_n_brands by mean recall across all scenarios, sorted
    ascending (highest brand at the top of the heatmap).  Scenarios are sorted
    descending by mean recall (most-recalled scenario on the left).

    Parameters
    ----------
    scenario_recall_df : output of run_scenario_recall()
    top_n_brands       : number of brands to include (default 12)
    title              : optional override for the figure title
    """
    plt.style.use("dark_background")

    pivot = scenario_recall_df.pivot_table(
        index="brand_name",
        columns="scenario_name",
        values="recall_prob",
        aggfunc="mean",
    )

    # Select top-N brands by mean recall across all scenarios
    brand_means = pivot.mean(axis=1)
    top_brands  = brand_means.nlargest(top_n_brands).index
    pivot = pivot.loc[top_brands]

    # Sort brands ascending (highest at top)
    brand_order = pivot.mean(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[brand_order]

    # Sort scenarios descending (most recalled on left)
    scenario_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot[scenario_order]

    n_brands    = len(pivot)
    n_scenarios = len(pivot.columns)
    figsize = (max(10, n_scenarios * 0.9), max(5, n_brands * 0.45))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)

    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n_brands))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Annotate each cell
    for r in range(n_brands):
        for c in range(n_scenarios):
            v = pivot.values[r, c]
            ax.text(
                c, r,
                f"{v:.0%}",
                ha="center", va="center",
                fontsize=7,
                color="black" if v > pivot.values.max() * 0.55 else "white",
            )

    plt.colorbar(im, ax=ax, label="Mean recall probability", format=mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Brand")
    ax.set_title(title or "Brand situation heatmap — recall by scenario")
    fig.tight_layout()
    return fig, ax


def plot_brand_leaderboard(
    scenario_recall_df: pd.DataFrame,
    top_n: int = 15,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart of mean recall probability across all scenarios,
    top-N brands ranked highest → lowest (highest bar at the top).

    Parameters
    ----------
    scenario_recall_df : output of run_scenario_recall()
    top_n              : number of brands to show (default 15)
    title              : optional override for the figure title
    """
    plt.style.use("dark_background")

    ranked = (
        scenario_recall_df.groupby("brand_name")["recall_prob"]
        .mean()
        .nlargest(top_n)
        .sort_values(ascending=True)   # ascending so highest is at the top
    )

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    bars = ax.barh(ranked.index, ranked.values * 100, color="#22c55e")

    for bar, val in zip(bars, ranked.values):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=8,
            color="#e2e8f0",
        )

    ax.set_xlabel("Mean recall probability (%)")
    ax.set_xlim(0, ranked.values.max() * 100 * 1.22)
    ax.set_title(title or "Brand recall leaderboard")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    return fig, ax


def plot_scenario_diagnostics(
    diag_df: pd.DataFrame,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Dual-panel bar chart for scenario-level diagnostics.

    Top panel   : MAE per scenario (with 5pp and 2pp target lines)
    Bottom panel: Spearman ρ per scenario (with moderate/strong threshold lines)

    Parameters
    ----------
    diag_df : output of run_scenario_diagnostics()
    """
    sorted_df = diag_df.sort_values("mae", ascending=False)
    x = np.arange(len(sorted_df))
    labels = sorted_df["scenario_name"].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # MAE panel
    bars = axes[0].bar(x, sorted_df["mae"], color="#2171b5", width=0.6)
    axes[0].axhline(0.05, color="red",   linestyle="--", linewidth=1, label="5 pp target")
    axes[0].axhline(0.02, color="green", linestyle="--", linewidth=1, label="2 pp target")
    for bar, val in zip(bars, sorted_df["mae"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("MAE")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title or "Scenario diagnostics")

    # Spearman ρ panel
    colors = ["#74c476" if r >= 0.3 else "#fd8d3c" for r in sorted_df["spearman_rho"]]
    axes[1].bar(x, sorted_df["spearman_rho"], color=colors, width=0.6)
    axes[1].axhline(0.6, color="green",  linestyle="--", linewidth=1, label="strong (0.6)")
    axes[1].axhline(0.3, color="orange", linestyle="--", linewidth=1, label="moderate (0.3)")
    axes[1].axhline(0.2, color="red",    linestyle=":",  linewidth=1, label="flag threshold (0.2)")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_xlabel("Scenario")
    axes[1].legend(fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    fig.tight_layout()
    return fig, axes
