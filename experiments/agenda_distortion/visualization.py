"""
H1 Experiment — Visualization Module
======================================
Publication-quality figures for the H1 (Agenda Distortion) analysis.
All figures use consistent thesis styling and are PDF-exportable.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter

# Import thesis design constants from the shared library
import sys

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT / "1a_BERTopic") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "1a_BERTopic"))

from merged_outlets_analysis import (
    THESIS_COLORS,
    THESIS_RC,
    _thesis_axis_style,
    _thesis_bar_labels,
)

ALT_OUTLET_CATEGORY_MAP = {
    "rt": "Pro-Russian / right-extremist",
    "antispiegel": "Pro-Russian / right-extremist",
    "compact": "Pro-Russian / right-extremist",
    "deutschlandkurier": "Pro-Russian / right-extremist",
    "nius": "Populist-alternative",
    "tichys": "Populist-alternative",
}

AGENDA_SCATTER_COLORS = {
    "Pro-Russian / right-extremist": "#D4603A",
    "Populist-alternative": "#4A90D9",
    "Convergence": "#B0B0B0",
}

TOPIC_LABEL_EXACT = {
    "ukraine_putin_selenskyj_trump": "Ukraine / Putin",
    "russland_ukraine_russischen_russische": "Russian Ukraine framing",
    "ukraine_russland_russischen_ukrainische": "Russia-Ukraine war",
    "spiegel_bellingcat_journalisten_occrp": "Media criticism",
    "afd_bsw_partei_spd": "AfD / BSW parties",
    "merz_kanzler_afd_stadtbild": "Merz / chancellor race",
    "dobrindt_asylbewerber_grenzkontrollen_geflüchtete": "Migration / border controls",
    "social media_media_jugendliche_social": "Social media / youth",
    "bundesanwaltschaft_gericht_haft_prozess": "Courts / criminal cases",
    "buchmesse_messe_kirk_halle": "Book fair debate",
    "moderator_studio_radio_abend": "TV / radio commentary",
    "venezuela_maduro_trump_venezuelas": "Venezuela / Maduro",
    "babis_türkei_chp_partei": "Turkey / CHP politics",
    "co2_strom_reiche_energiewende": "Energy / climate policy",
    "trump_epstein_trumps_demokraten": "Trump / Epstein",
    "grönland_trump_nato_dänemark": "Greenland / NATO",
    "wirtschaft_ifo_deutsche wirtschaft_wachstum": "German economy",
    "china_xi_indien_trump": "China / India",
}

TOPIC_TOKEN_MAP = {
    "afd": "AfD",
    "bsw": "BSW",
    "spd": "SPD",
    "cdu": "CDU",
    "csu": "CSU",
    "ukraine": "Ukraine",
    "russland": "Russia",
    "russischen": "Russia",
    "russische": "Russia",
    "ukrainische": "Ukraine",
    "putin": "Putin",
    "selenskyj": "Zelenskyy",
    "trump": "Trump",
    "merz": "Merz",
    "kanzler": "Chancellor",
    "israel": "Israel",
    "hamas": "Hamas",
    "gazastreifen": "Gaza",
    "gaza": "Gaza",
    "bundesanwaltschaft": "Federal prosecutor",
    "gericht": "Courts",
    "prozess": "Trials",
    "haft": "Detention",
    "social": "Social media",
    "media": "Media",
    "jugendliche": "Youth",
    "asylbewerber": "Asylum",
    "grenzkontrollen": "Borders",
    "geflüchtete": "Refugees",
    "migration": "Migration",
    "bellingcat": "Bellingcat",
    "journalisten": "Journalists",
    "strom": "Electricity",
    "energiewende": "Energy transition",
    "wirtschaft": "Economy",
    "wachstum": "Growth",
    "ifo": "IFO",
    "venezuela": "Venezuela",
    "maduro": "Maduro",
    "buchmesse": "Book fair",
    "messe": "Fair",
    "radio": "Radio",
    "moderator": "Host",
    "studio": "Studio",
    "abend": "Evening show",
    "türkei": "Turkey",
    "chp": "CHP",
    "partei": "Party politics",
    "co2": "CO2",
    "grönland": "Greenland",
    "nato": "NATO",
    "dänemark": "Denmark",
}

TOPIC_TOKEN_STOPWORDS = {
    "und",
    "der",
    "die",
    "das",
    "des",
    "den",
    "dem",
    "eu",
    "euro",
    "deutsche",
    "deutschen",
    "deutschland",
    "deutschlands",
}


def _save_fig(fig, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Saved: {path}")


def _clean_topic_source_label(value: object) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"^Topic\s+\d+\s+[—-]\s+", "", text)
    return text.strip()


def suggest_short_topic_label(raw_label: object) -> str:
    """Turn BERTopic keyword labels into short, reviewable plot labels."""

    cleaned = _clean_topic_source_label(raw_label).lower()
    if not cleaned:
        return "Unlabeled topic"
    if cleaned in TOPIC_LABEL_EXACT:
        return TOPIC_LABEL_EXACT[cleaned]

    tokens = [token for token in re.split(r"[_\s/]+", cleaned) if token]
    seen: set[str] = set()
    words: list[str] = []
    for token in tokens:
        if token in TOPIC_TOKEN_STOPWORDS:
            continue
        mapped = TOPIC_TOKEN_MAP.get(token, token.replace("-", " ").strip())
        key = mapped.lower()
        if key in seen or not mapped:
            continue
        seen.add(key)
        if mapped.islower():
            mapped = mapped.title()
        words.append(mapped)
        if len(words) >= 3:
            break

    if not words:
        fallback = cleaned.replace("_", " ").strip()
        return fallback.title() if fallback else "Unlabeled topic"

    return " / ".join(words)


# ---------------------------------------------------------------------------
# 1. H1 Metrics Panel (4-panel horizontal bar chart)
# ---------------------------------------------------------------------------


def plot_h1_metrics_panel(
    metrics_df: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Four-panel bar chart showing all H1 metrics per outlet.

    Panels: Entropy | JSD vs Tagesschau | Spearman ρ | Top-K Overlap
    Sorted by JSD descending (most divergent at top).
    """
    colors = colors or THESIS_COLORS
    plot_df = (
        metrics_df.copy()
        .sort_values("jsd_vs_tagesschau", ascending=True)
        .reset_index(drop=True)
    )
    y_pos = np.arange(len(plot_df))
    bar_colors = [colors.get(label, "#999999") for label in plot_df["outlet_label"]]

    panels = [
        ("entropy_normalized", "Topic Entropy", "Entropy (0–1)", "{:.2f}", None),
        ("jsd_vs_tagesschau", "JSD vs Tagesschau", "JSD (0–1)", "{:.3f}", None),
        ("spearman_rho", "Spearman ρ", "Rank correlation", "{:+.2f}", 0.0),
        ("topk_overlap", "Top-K Overlap", "Overlap share", "{:.2f}", None),
    ]

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=150, sharey=True)
        fig.patch.set_facecolor("white")

        for ax, (col, title, xlabel, fmt, refline) in zip(axes, panels):
            _thesis_axis_style(ax)
            vals = plot_df[col].astype(float).to_numpy()

            # Flag small outlets on JSD panel
            labels = plot_df["outlet_label"].tolist()
            if col == "jsd_vs_tagesschau":
                labels = [
                    f"{l} ⚠" if int(n) < 1000 else l
                    for l, n in zip(
                        plot_df["outlet_label"], plot_df["n_articles"]
                    )
                ]

            ax.barh(y_pos, vals, height=0.55, color=bar_colors, alpha=0.92)
            if refline is not None:
                ax.axvline(refline, color="#AAAAAA", ls="--", lw=1)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels if col in ("jsd_vs_tagesschau",) else plot_df["outlet_label"])
            ax.set_title(title, fontsize=12, fontweight="semibold", pad=10)
            ax.set_xlabel(xlabel, fontsize=10)
            _thesis_bar_labels(ax, vals, y_pos, fmt=fmt)
            ax.invert_yaxis()

        fig.suptitle(
            "H1 Evidence: Agenda Distortion Metrics per Outlet",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Coverage Breadth Panel
# ---------------------------------------------------------------------------


def plot_coverage_breadth(
    metrics_df: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of coverage breadth (relative) per outlet.

    Reference line at 1.0 = "as broad as corpus size predicts".
    """
    colors = colors or THESIS_COLORS
    plot_df = (
        metrics_df.copy()
        .sort_values("coverage_breadth_relative", ascending=True)
        .reset_index(drop=True)
    )
    y_pos = np.arange(len(plot_df))
    bar_colors = [colors.get(l, "#999999") for l in plot_df["outlet_label"]]

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        fig.patch.set_facecolor("white")
        _thesis_axis_style(ax)

        vals = plot_df["coverage_breadth_relative"].astype(float).to_numpy()
        ax.barh(y_pos, vals, height=0.55, color=bar_colors, alpha=0.92)
        ax.axvline(1.0, color="#AAAAAA", ls="--", lw=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["outlet_label"])
        ax.set_title("Coverage Breadth (Relative)", fontsize=13, fontweight="semibold", pad=12)
        ax.set_xlabel("Actual / Expected breadth (1.0 = neutral)")
        _thesis_bar_labels(ax, vals, y_pos, fmt="{:.2f}")
        ax.invert_yaxis()
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Bootstrap CI plot
# ---------------------------------------------------------------------------


def plot_bootstrap_cis(
    bootstrap_df: pd.DataFrame,
    metric: str = "jsd_vs_tagesschau",
    *,
    reference_label: str = "Tagesschau",
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Forest plot of bootstrap confidence intervals for a single metric."""
    colors = colors or THESIS_COLORS
    df = bootstrap_df[bootstrap_df["metric"] == metric].copy()
    df = df[df["outlet_label"] != reference_label].sort_values(
        "point_estimate", ascending=True
    ).reset_index(drop=True)

    metric_labels = {
        "jsd_vs_tagesschau": "Jensen-Shannon Divergence vs Tagesschau",
        "spearman_rho": "Spearman ρ (rank correlation with Tagesschau)",
        "topk_overlap": "Top-K Topic Overlap with Tagesschau",
    }

    y_pos = np.arange(len(df))

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        fig.patch.set_facecolor("white")
        _thesis_axis_style(ax)

        for i, row in df.iterrows():
            color = colors.get(row["outlet_label"], "#999999")
            ax.plot(
                [row["ci_lower"], row["ci_upper"]], [i, i],
                color=color, lw=2.5, solid_capstyle="round",
            )
            ax.plot(
                row["point_estimate"], i,
                "o", color=color, ms=8, zorder=5,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["outlet_label"])
        ax.set_title(
            f"Bootstrap 95% CI — {metric_labels.get(metric, metric)}",
            fontsize=12, fontweight="semibold", pad=12,
        )
        ax.set_xlabel(metric_labels.get(metric, metric))
        ax.invert_yaxis()
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Topic heatmap (re-export from merged_outlets_analysis)
# ---------------------------------------------------------------------------


def plot_topic_heatmap(
    merged_articles: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
    *,
    top_n_topics: int = 25,
    normalize: str = "outlet",
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Outlet × Topic heatmap — delegates to the shared implementation."""
    from merged_outlets_analysis import plot_outlet_topic_heatmap

    fig = plot_outlet_topic_heatmap(
        merged_articles,
        top_n_topics=top_n_topics,
        normalize=normalize,
        source_colors=colors or THESIS_COLORS,
        measures_df=metrics_df,
    )
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Semantic footprint maps (per-outlet UMAP)
# ---------------------------------------------------------------------------


def plot_all_outlet_umaps(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    save_dir: Path | None = None,
) -> list[plt.Figure]:
    """Generate semantic footprint UMAP for each outlet."""
    from merged_outlets_analysis import OUTLET_SPECS, plot_outlet_highlight_umap

    figs = []
    for key in OUTLET_SPECS:
        fig, _ = plot_outlet_highlight_umap(
            merged_articles,
            key,
            merged_topic_info=merged_topic_info,
        )
        if save_dir is not None:
            _save_fig(fig, save_dir / f"umap_{key}.pdf")
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# 6. Outlet comparison radar
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Outlet clustering dendrogram + pairwise JSD heatmap
# ---------------------------------------------------------------------------


def plot_pairwise_jsd_heatmap(
    jsd_matrix: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of pairwise JSD between all outlets.

    Annotated with JSD values. Outlets ordered by hierarchical clustering.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    colors = colors or THESIS_COLORS

    # Cluster ordering
    condensed = squareform(jsd_matrix.values, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)
    ordered = jsd_matrix.iloc[order, order]

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
        fig.patch.set_facecolor("white")

        im = ax.imshow(ordered.values, cmap="YlOrRd", aspect="auto", vmin=0)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Jensen-Shannon Divergence")

        outlets = ordered.index.tolist()
        ax.set_xticks(range(len(outlets)))
        ax.set_xticklabels(outlets, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(outlets)))
        ax.set_yticklabels(outlets, fontsize=10)

        # Annotate cells
        for i in range(len(outlets)):
            for j in range(len(outlets)):
                val = ordered.values[i, j]
                text_color = "white" if val > 0.25 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="semibold")

        ax.set_title("Pairwise Agenda Distance (JSD)", fontsize=13,
                      fontweight="semibold", pad=12)
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


def plot_outlet_dendrogram(
    jsd_matrix: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Hierarchical clustering dendrogram of outlets by topic distribution.

    Uses Ward's method on pairwise JSD distances.

    Literature: Muller & Freudenthaler (2022) used hierarchical clustering
    on topic proportions to identify media landscape clusters.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    colors = colors or THESIS_COLORS
    condensed = squareform(jsd_matrix.values, checks=False)
    Z = linkage(condensed, method="ward")

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor("white")
        _thesis_axis_style(ax)

        dend = dendrogram(
            Z,
            labels=jsd_matrix.index.tolist(),
            ax=ax,
            leaf_font_size=11,
            color_threshold=0,
            above_threshold_color="#555555",
        )

        # Color the x-tick labels by outlet
        xlabels = ax.get_xticklabels()
        for lbl in xlabels:
            outlet = lbl.get_text()
            lbl.set_color(colors.get(outlet, "#333333"))
            lbl.set_fontweight("semibold")

        ax.set_ylabel("Ward linkage distance (JSD)", fontsize=11)
        ax.set_title(
            "Outlet Clustering by Topic Distribution",
            fontsize=13, fontweight="semibold", pad=12,
        )

        # Add category annotations
        ax.annotate(
            "", xy=(0, 0), fontsize=8, color="#888888",
            xytext=(0, 0), annotation_clip=False,
        )

        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Chi-squared volcano plot
# ---------------------------------------------------------------------------


def plot_chi_squared_volcano(
    chi_df: pd.DataFrame,
    outlet: str | None = None,
    *,
    alpha: float = 0.01,
    min_diff_pp: float = 1.0,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Volcano plot of topic over/under-representation significance.

    X-axis: difference in percentage points (outlet - Tagesschau).
    Y-axis: -log10(corrected p-value).
    Labeled points: significant topics above thresholds.
    """
    colors = colors or THESIS_COLORS

    if outlet is not None:
        df = chi_df[chi_df["outlet_label"] == outlet].copy()
        title = f"Topic Significance — {outlet} vs Tagesschau"
    else:
        df = chi_df.copy()
        title = "Topic Significance — All Outlets vs Tagesschau"

    df["neg_log_p"] = -np.log10(df["p_corrected"].clip(lower=1e-50))

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        fig.patch.set_facecolor("white")
        _thesis_axis_style(ax)

        sig_mask = (df["p_corrected"] < alpha) & (df["diff_pp"].abs() >= min_diff_pp)

        # Non-significant points
        ns = df[~sig_mask]
        ax.scatter(ns["diff_pp"], ns["neg_log_p"], c="#CCCCCC", s=20, alpha=0.5, zorder=1)

        # Significant points
        sig = df[sig_mask]
        if outlet is not None:
            point_colors = [colors.get(outlet, "#E74C3C")] * len(sig)
        else:
            point_colors = [colors.get(o, "#E74C3C") for o in sig["outlet_label"]]
        ax.scatter(sig["diff_pp"], sig["neg_log_p"], c=point_colors, s=40, alpha=0.8, zorder=2)

        # Label top significant topics
        top_n = sig.nlargest(15, "neg_log_p")
        for _, row in top_n.iterrows():
            label = row["topic_label"]
            if len(label) > 30:
                label = label[:28] + "..."
            ax.annotate(
                label,
                (row["diff_pp"], row["neg_log_p"]),
                fontsize=7, alpha=0.85,
                xytext=(5, 3), textcoords="offset points",
            )

        # Threshold lines
        ax.axhline(-np.log10(alpha), color="#AAAAAA", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(-min_diff_pp, color="#AAAAAA", ls=":", lw=0.8, alpha=0.6)
        ax.axvline(min_diff_pp, color="#AAAAAA", ls=":", lw=0.8, alpha=0.6)

        ax.set_xlabel("Difference (pp) vs Tagesschau", fontsize=11)
        ax.set_ylabel("-log₁₀(corrected p-value)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="semibold", pad=12)
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Temporal analysis plots
# ---------------------------------------------------------------------------


def plot_topic_time_series(
    weekly_props: pd.DataFrame,
    topic: int,
    topic_label: str = "",
    *,
    reference_label: str = "Tagesschau",
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Weekly topic proportion time series for all outlets.

    Shows how a specific topic's salience evolves over time for each outlet,
    with Tagesschau as dashed reference line.
    """
    colors = colors or THESIS_COLORS
    topic_data = weekly_props[weekly_props["merged_topic"] == topic].copy()

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        fig.patch.set_facecolor("white")
        _thesis_axis_style(ax)

        for outlet in sorted(topic_data["outlet_label"].unique()):
            odata = topic_data[topic_data["outlet_label"] == outlet].sort_values("week")
            color = colors.get(outlet, "#999999")
            style = "--" if outlet == reference_label else "-"
            lw = 1.5 if outlet == reference_label else 2
            alpha = 0.6 if outlet == reference_label else 0.85
            ax.plot(odata["week"], odata["proportion"] * 100, style,
                    color=color, lw=lw, alpha=alpha, label=outlet, marker="o", ms=3)

        ax.set_xlabel("Week", fontsize=11)
        ax.set_ylabel("Topic share (%)", fontsize=11)
        title = f"Topic Salience Over Time — {topic_label}" if topic_label else f"Topic {topic} Over Time"
        ax.set_title(title, fontsize=13, fontweight="semibold", pad=12)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


def plot_rolling_jsd(
    rolling_df: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Rolling JSD over time for each outlet vs Tagesschau.

    Shows whether agenda distance is stable, increasing, or decreasing
    over the study period.
    """
    colors = colors or THESIS_COLORS

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        fig.patch.set_facecolor("white")
        _thesis_axis_style(ax)

        for outlet in sorted(rolling_df["outlet_label"].unique()):
            odata = rolling_df[rolling_df["outlet_label"] == outlet].sort_values("week_end")
            color = colors.get(outlet, "#999999")
            ax.plot(odata["week_end"], odata["rolling_jsd"],
                    "-", color=color, lw=2, alpha=0.85, label=outlet, marker="o", ms=3)

        ax.set_xlabel("Week", fontsize=11)
        ax.set_ylabel("Rolling JSD vs Tagesschau (4-week window)", fontsize=11)
        ax.set_title("Agenda Distance Over Time", fontsize=13, fontweight="semibold", pad=12)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


def plot_lag_correlation(
    lag_df: pd.DataFrame,
    topic_label: str = "",
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of time-lagged correlations for a specific topic.

    X-axis: lag in weeks (negative = outlet follows Tagesschau).
    Y-axis: Pearson correlation.
    One subplot per outlet.
    """
    colors = colors or THESIS_COLORS
    outlets = sorted(lag_df["outlet_label"].unique())
    n_outlets = len(outlets)

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(1, n_outlets, figsize=(3.5 * n_outlets, 4),
                                  dpi=150, sharey=True)
        fig.patch.set_facecolor("white")
        if n_outlets == 1:
            axes = [axes]

        for ax, outlet in zip(axes, outlets):
            _thesis_axis_style(ax)
            odata = lag_df[lag_df["outlet_label"] == outlet].sort_values("lag")
            color = colors.get(outlet, "#999999")

            bars = ax.bar(odata["lag"], odata["correlation"], color=color, alpha=0.8, width=0.7)

            # Highlight significant bars
            for bar, (_, row) in zip(bars, odata.iterrows()):
                if row["p_value"] < 0.05:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.5)

            ax.axhline(0, color="#AAAAAA", lw=0.8)
            ax.axvline(0, color="#CCCCCC", ls=":", lw=0.8)
            ax.set_xlabel("Lag (weeks)", fontsize=9)
            ax.set_title(outlet, fontsize=10, fontweight="semibold")

        axes[0].set_ylabel("Pearson r", fontsize=10)
        title = f"Lagged Correlation — {topic_label}" if topic_label else "Lagged Correlation"
        fig.suptitle(title, fontsize=13, fontweight="semibold", y=1.02)
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 9. Outlet comparison radar
# ---------------------------------------------------------------------------


def plot_distortion_radar(
    metrics_df: pd.DataFrame,
    reference_label: str = "Tagesschau",
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Radar chart comparing all outlets across H1 dimensions.

    Each axis is one metric, min-max normalized across outlets.
    Reference outlet shown as dashed baseline.
    """
    colors = colors or THESIS_COLORS
    dims = ["entropy_normalized", "jsd_vs_tagesschau", "topk_overlap", "coverage_breadth_relative"]
    dim_labels = ["Entropy", "JSD", "Top-K\nOverlap", "Breadth\n(relative)"]

    df = metrics_df.copy()
    # Normalize each dimension to 0-1 across outlets
    for col in dims:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.5

    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150, subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("white")

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), dim_labels)

        for _, row in df.iterrows():
            label = row["outlet_label"]
            values = [row[f"{d}_norm"] for d in dims]
            values += values[:1]
            color = colors.get(label, "#999999")

            if label == reference_label:
                ax.plot(angles, values, "o--", color=color, lw=1.5, ms=4, alpha=0.6, label=label)
            else:
                ax.plot(angles, values, "o-", color=color, lw=2, ms=5, label=label)

        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.set_title("H1 Distortion Profile — All Outlets", fontsize=13, fontweight="semibold", pad=20)
        fig.tight_layout()

    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 10. Substitution-amplification scatter for H1
# ---------------------------------------------------------------------------


def _canonical_topic_info(topic_info_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize topic-info columns from either BERTopic or article-export sources."""

    renamed = topic_info_df.rename(
        columns={
            "Topic": "merged_topic",
            "DisplayTopic": "merged_display_topic",
            "KeywordLabel": "topic_keyword_label",
            "TopicLabel": "topic_label",
            "DisplayLabel": "topic_display_label",
            "ManualTopicName": "manual_topic_name",
        }
    ).copy()

    for column in ("merged_topic", "merged_display_topic"):
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce").astype("Int64")

    return renamed


def build_topic_label_review(
    topic_info_df: pd.DataFrame,
    *,
    override_path: Path | None = None,
) -> pd.DataFrame:
    """Create a reviewable topic-label mapping and optional override template."""

    info = _canonical_topic_info(topic_info_df)
    info = info.loc[info["merged_topic"] != -1].copy()

    raw_label = (
        info.get("manual_topic_name")
        .fillna("")
        .astype(str)
        .where(lambda s: s.str.strip() != "", other=np.nan)
        .fillna(info.get("topic_label"))
        .fillna(info.get("topic_keyword_label"))
    )

    review = info.copy()
    review["raw_label"] = raw_label
    review["suggested_plot_label"] = review["raw_label"].map(suggest_short_topic_label)
    review["manual_plot_label"] = pd.Series(pd.NA, index=review.index, dtype="object")

    if override_path is not None:
        override_path.parent.mkdir(parents=True, exist_ok=True)
        if override_path.exists():
            overrides = pd.read_csv(override_path)
            if "merged_topic" in overrides.columns:
                overrides["merged_topic"] = pd.to_numeric(
                    overrides["merged_topic"], errors="coerce"
                ).astype("Int64")
                manual_cols = [col for col in overrides.columns if col == "manual_plot_label"]
                merge_cols = ["merged_topic"] + manual_cols
                review = review.merge(overrides[merge_cols], on="merged_topic", how="left", suffixes=("", "_override"))
                if "manual_plot_label_override" in review.columns:
                    review["manual_plot_label"] = review["manual_plot_label_override"].combine_first(
                        review["manual_plot_label"]
                    )
                    review = review.drop(columns=["manual_plot_label_override"])
        else:
            template = review[
                [
                    "merged_topic",
                    "merged_display_topic",
                    "topic_keyword_label",
                    "raw_label",
                    "suggested_plot_label",
                ]
            ].copy()
            template["manual_plot_label"] = ""
            template.to_csv(override_path, index=False)

    review["final_plot_label"] = review["manual_plot_label"].fillna("").astype(str)
    review["final_plot_label"] = review["final_plot_label"].where(
        review["final_plot_label"].str.strip() != "",
        review["suggested_plot_label"],
    )

    ordered = [
        "merged_topic",
        "merged_display_topic",
        "topic_keyword_label",
        "raw_label",
        "suggested_plot_label",
        "manual_plot_label",
        "final_plot_label",
    ]
    return review.loc[:, [col for col in ordered if col in review.columns]].sort_values(
        ["merged_display_topic", "merged_topic"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def _zone_from_shares(
    tagesschau_share: float,
    outlet_share: float,
    *,
    substitution_cutoff: float = 0.02,
    amplification_multiplier: float = 1.5,
) -> str:
    """Classify a topic point into the requested conceptual zones."""

    if tagesschau_share < substitution_cutoff:
        return "Substitution"
    if outlet_share > amplification_multiplier * tagesschau_share:
        return "Amplification"
    return "Near parity"


def select_signature_topics(
    all_comparisons_df: pd.DataFrame,
    topic_label_review_df: pd.DataFrame,
    *,
    outlet_keys: tuple[str, ...] = ("rt", "antispiegel", "compact", "deutschlandkurier", "nius", "tichys"),
    min_signature_share: float = 0.03,
    substitution_cutoff: float = 0.02,
    amplification_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Pick one signature topic per alternative outlet."""

    label_map = (
        topic_label_review_df[["merged_topic", "final_plot_label"]]
        .drop_duplicates(subset=["merged_topic"])
        .set_index("merged_topic")["final_plot_label"]
        .to_dict()
    )

    top5_map = {}
    for outlet_key in outlet_keys:
        outlet_rows = all_comparisons_df.loc[
            (all_comparisons_df["outlet_key"] == outlet_key)
            & (all_comparisons_df["merged_topic"] != -1)
        ].copy()
        top5_map[outlet_key] = (
            outlet_rows.sort_values(["article_share", "share_diff"], ascending=[False, False])
            .head(5)["merged_topic"]
            .astype("Int64")
            .tolist()
        )

    rows: list[dict[str, object]] = []
    for outlet_key in outlet_keys:
        outlet_rows = all_comparisons_df.loc[
            (all_comparisons_df["outlet_key"] == outlet_key)
            & (all_comparisons_df["merged_topic"] != -1)
        ].copy()
        outlet_rows = outlet_rows.sort_values(
            ["share_diff", "article_share", "baseline_article_share"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        visible_rows = outlet_rows.loc[outlet_rows["article_share"] >= min_signature_share].copy()
        used_visibility_fallback = False
        if not visible_rows.empty:
            chosen = visible_rows.iloc[0]
            if int(chosen["merged_topic"]) != int(outlet_rows.iloc[0]["merged_topic"]):
                used_visibility_fallback = True
        else:
            chosen = outlet_rows.iloc[0]

        merged_topic = int(chosen["merged_topic"])
        overlapping_outlets = [
            other_key
            for other_key, top_topics in top5_map.items()
            if other_key != outlet_key and merged_topic in top_topics
        ]
        rows.append(
            {
                "outlet_key": outlet_key,
                "outlet_label": chosen["outlet_label"],
                "category": ALT_OUTLET_CATEGORY_MAP.get(outlet_key, "Other"),
                "merged_topic": merged_topic,
                "merged_display_topic": int(chosen["merged_display_topic"]),
                "raw_label": chosen["topic_keyword_label"],
                "plot_label": label_map.get(merged_topic, suggest_short_topic_label(chosen["topic_keyword_label"])),
                "outlet_share": float(chosen["article_share"]),
                "tagesschau_share": float(chosen["baseline_article_share"]),
                "share_difference": float(chosen["share_diff"]),
                "zone": _zone_from_shares(
                    float(chosen["baseline_article_share"]),
                    float(chosen["article_share"]),
                    substitution_cutoff=substitution_cutoff,
                    amplification_multiplier=amplification_multiplier,
                ),
                "signature_rank_by_difference": int(outlet_rows.index[outlet_rows["merged_topic"] == merged_topic][0]) + 1,
                "used_visibility_fallback": used_visibility_fallback,
                "overlap_with_other_top5": ", ".join(overlapping_outlets),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["category", "share_difference"],
        ascending=[True, False],
    ).reset_index(drop=True)


def select_convergence_topics(
    prevalence_df: pd.DataFrame,
    topic_label_review_df: pd.DataFrame,
    *,
    baseline_key: str = "tagesschau",
    alt_outlet_keys: tuple[str, ...] = ("rt", "antispiegel", "compact", "deutschlandkurier", "nius", "tichys"),
    n_topics: int = 3,
    min_baseline_share: float = 0.015,
    min_mean_share: float = 0.012,
    min_all_share: float = 0.003,
) -> pd.DataFrame:
    """Identify topics where all outlets sit relatively close together."""

    label_map = (
        topic_label_review_df[["merged_topic", "final_plot_label"]]
        .drop_duplicates(subset=["merged_topic"])
        .set_index("merged_topic")["final_plot_label"]
        .to_dict()
    )

    all_keys = [baseline_key, *alt_outlet_keys]
    wide = (
        prevalence_df.loc[prevalence_df["merged_topic"] != -1]
        .pivot_table(index="merged_topic", columns="outlet_key", values="article_share", fill_value=0)
        .reindex(columns=all_keys, fill_value=0)
    )
    wide["alt_mean_share"] = wide[list(alt_outlet_keys)].mean(axis=1)
    wide["alt_std_share"] = wide[list(alt_outlet_keys)].std(axis=1)
    wide["alt_range_share"] = wide[list(alt_outlet_keys)].max(axis=1) - wide[list(alt_outlet_keys)].min(axis=1)
    wide["diagonal_gap"] = (wide["alt_mean_share"] - wide[baseline_key]).abs()
    wide["mean_all_share"] = wide[all_keys].mean(axis=1)
    wide["min_all_share"] = wide[all_keys].min(axis=1)

    candidates = wide.loc[
        (wide[baseline_key] >= min_baseline_share)
        & (wide["mean_all_share"] >= min_mean_share)
        & (wide["min_all_share"] >= min_all_share)
    ].copy()

    info_cols = (
        prevalence_df[["merged_topic", "merged_display_topic", "topic_keyword_label"]]
        .drop_duplicates(subset=["merged_topic"])
    )
    candidates = (
        candidates.reset_index()
        .merge(info_cols, on="merged_topic", how="left")
        .sort_values(["diagonal_gap", "alt_std_share", "mean_all_share"], ascending=[True, True, False])
        .head(n_topics)
        .reset_index(drop=True)
    )

    candidates["plot_label"] = candidates["merged_topic"].map(label_map).fillna(
        candidates["topic_keyword_label"].map(suggest_short_topic_label)
    )
    candidates["outlet_label"] = "All outlets"
    candidates["category"] = "Convergence"
    candidates["tagesschau_share"] = candidates[baseline_key].astype(float)
    candidates["outlet_share"] = candidates["alt_mean_share"].astype(float)
    return candidates


def build_substitution_amplification_points(
    signature_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine signature and convergence topics into one plotting dataframe."""

    signature_points = signature_df.copy()
    signature_points["point_type"] = "signature"
    signature_points["point_color"] = signature_points["category"].map(AGENDA_SCATTER_COLORS).fillna("#888888")
    signature_points["point_size"] = 120 + (signature_points["outlet_share"] * 1200)

    convergence_points = convergence_df.copy()
    convergence_points["point_type"] = "convergence"
    convergence_points["point_color"] = AGENDA_SCATTER_COLORS["Convergence"]
    convergence_points["point_size"] = 115 + (convergence_points["outlet_share"] * 1000)
    convergence_points["zone"] = "Convergence"

    points = pd.concat([signature_points, convergence_points], ignore_index=True, sort=False)
    points["x_share"] = points["tagesschau_share"].astype(float)
    points["y_share"] = points["outlet_share"].astype(float)
    return points


def plot_substitution_amplification_scatter(
    points_df: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
    png_preview_path: Path | None = None,
    figsize: tuple[float, float] = (11.5, 8.5),
) -> plt.Figure:
    """Render the publication-style substitution-amplification scatter."""

    colors = colors or AGENDA_SCATTER_COLORS
    plot_df = points_df.copy()
    if plot_df.empty:
        raise ValueError("No points available for substitution-amplification scatter.")

    axis_limit = float(max(plot_df["x_share"].max(), plot_df["y_share"].max()))
    axis_limit = max(axis_limit * 1.12, 0.08)

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        fig.patch.set_facecolor("white")
        _thesis_axis_style(ax)
        ax.grid(False)

        zone_rectangles = [
            (
                Rectangle(
                    (0.0, axis_limit * 0.56),
                    axis_limit * 0.34,
                    axis_limit * 0.40,
                    facecolor="#F4EDE7",
                    edgecolor="none",
                    alpha=0.85,
                    zorder=0,
                ),
                "Substitution zone\nNot in mainstream",
                (axis_limit * 0.03, axis_limit * 0.93),
            ),
            (
                Rectangle(
                    (axis_limit * 0.36, axis_limit * 0.56),
                    axis_limit * 0.56,
                    axis_limit * 0.40,
                    facecolor="#F5F1E8",
                    edgecolor="none",
                    alpha=0.85,
                    zorder=0,
                ),
                "Amplification zone\nSame topic, much higher weight",
                (axis_limit * 0.53, axis_limit * 0.93),
            ),
            (
                Rectangle(
                    (axis_limit * 0.50, 0.0),
                    axis_limit * 0.42,
                    axis_limit * 0.28,
                    facecolor="#EEF1EB",
                    edgecolor="none",
                    alpha=0.90,
                    zorder=0,
                ),
                "Mainstream only",
                (axis_limit * 0.63, axis_limit * 0.17),
            ),
        ]
        for rectangle, label, (x_text, y_text) in zone_rectangles:
            ax.add_patch(rectangle)
            ax.text(
                x_text,
                y_text,
                label,
                ha="left",
                va="top",
                fontsize=11,
                color="#6B6B6B",
                zorder=1,
            )

        ax.plot([0, axis_limit], [0, axis_limit], ls=(0, (4, 4)), lw=1.2, color="#C9C9C9", zorder=1)
        ax.text(axis_limit * 0.98, axis_limit * 0.98, "parity", ha="left", va="bottom", fontsize=10, color="#8D8D8D")

        ax.scatter(
            plot_df["x_share"],
            plot_df["y_share"],
            s=plot_df["point_size"],
            c=plot_df["point_color"],
            edgecolors="white",
            linewidths=1.2,
            alpha=0.95,
            zorder=3,
        )

        for _, row in plot_df.iterrows():
            dx = 8
            dy = 8 if row["y_share"] >= row["x_share"] else -10
            ha = "left"
            va = "bottom" if row["y_share"] >= row["x_share"] else "top"
            ax.annotate(
                str(row["plot_label"]),
                xy=(float(row["x_share"]), float(row["y_share"])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=11,
                color="#2F2F2F",
                zorder=4,
            )
            ax.annotate(
                str(row["outlet_label"]),
                xy=(float(row["x_share"]), float(row["y_share"])),
                xytext=(dx, dy - 14 if dy > 0 else dy + 14),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=9,
                color="#7A7A7A",
                zorder=4,
            )

        ax.set_xlim(0, axis_limit)
        ax.set_ylim(0, axis_limit)
        tick_step = 0.05 if axis_limit >= 0.15 else 0.02
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Tagesschau topic prevalence →", fontsize=12)
        ax.set_ylabel("Outlet topic prevalence →", fontsize=12)
        ax.set_title("Substitution and amplification of topic prevalence vs Tagesschau", pad=16)

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Pro-Russian / right-extremist", markerfacecolor=colors["Pro-Russian / right-extremist"], markersize=12),
            Line2D([0], [0], marker="o", color="w", label="Populist-alternative", markerfacecolor=colors["Populist-alternative"], markersize=12),
            Line2D([0], [0], marker="o", color="w", label="Near parity", markerfacecolor=colors["Convergence"], markersize=12),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=3,
            frameon=False,
            fontsize=10,
        )
        fig.tight_layout()

    _save_fig(fig, save_path)
    _save_fig(fig, png_preview_path)
    return fig
