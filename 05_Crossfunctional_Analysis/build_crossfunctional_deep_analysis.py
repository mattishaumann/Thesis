from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.spatial.distance import jensenshannon


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"
NOTEBOOK_PATH = BASE_DIR / "01_crossfunctional_deep_analysis.ipynb"

TOPIC_CSV = OUTPUT_DIR / "03_new_topic_assignments.csv"
FULLTEXT_EMOTION_CSV = OUTPUT_DIR / "emotion_fulltext_with_topiclabels.csv"
MAINSTREAM_EMOTION_CSV = OUTPUT_DIR / "emotion_mainstream_resultsv3.csv"
FRAMING_CSV = OUTPUT_DIR / "media_framing_final_run_5_4mini_classifications.csv"

INTEGRATED_OUTPUT_CSV = OUTPUT_DIR / "crossfunctional_integrated_dataset.csv"

DELEGIT_CATEGORIES = {
    "VERZERRUNG/MANIPULATION",
    "POSITIONS-/PARTEILICHKEITS-BIAS",
    "DISINFORMATION/FALSCHDARSTELLUNG",
    "VERSAGEN/INKOMPETENZ",
}

OUTLET_GROUPS = {
    "Tagesschau": "mainstream",
    "RT_de": "pro_russian_alt",
    "Antispiegel": "pro_russian_alt",
    "Compact": "right_extremist_alt",
    "Deutschlandkurier": "right_extremist_alt",
    "Nius": "right_conservative_alt",
    "Tichys_Einblick": "right_conservative_alt",
}

ALT_ONLY_OUTLETS = [
    "RT_de",
    "Antispiegel",
    "Compact",
    "Deutschlandkurier",
    "Nius",
    "Tichys_Einblick",
]

EMOTION_COLS = [
    "emotion_anger",
    "emotion_fear",
    "emotion_disgust",
    "emotion_sadness",
    "emotion_joy",
    "emotion_enthusiasm",
    "emotion_pride",
    "emotion_hope",
]


@dataclass
class BootstrapResult:
    mean: float
    low: float
    high: float


def normalize_source(source: pd.Series) -> pd.Series:
    mapping = {
        "RT": "RT_de",
        "RT_de": "RT_de",
        "Tichys Einblick": "Tichys_Einblick",
        "Tichys_Einblick": "Tichys_Einblick",
        "Tagesschau": "Tagesschau",
        "Compact": "Compact",
        "Nius": "Nius",
        "Antispiegel": "Antispiegel",
        "Deutschlandkurier": "Deutschlandkurier",
    }
    return source.astype(str).str.strip().map(mapping).fillna(source.astype(str).str.strip())


def bootstrap_mean(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> BootstrapResult:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return BootstrapResult(np.nan, np.nan, np.nan)
    if len(values) == 1:
        return BootstrapResult(float(values[0]), float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        draws.append(float(np.mean(sample)))
    return BootstrapResult(float(np.mean(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def ensure_dirs() -> None:
    for path in (FIG_DIR, TABLE_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topics = pd.read_csv(TOPIC_CSV, low_memory=False)
    fulltext = pd.read_csv(FULLTEXT_EMOTION_CSV, low_memory=False)
    mainstream_emotion = pd.read_csv(MAINSTREAM_EMOTION_CSV, low_memory=False)
    framing = pd.read_csv(FRAMING_CSV, low_memory=False)
    return topics, fulltext, mainstream_emotion, framing


def prepare_base_frames(
    topics: pd.DataFrame,
    fulltext: pd.DataFrame,
    mainstream_emotion: pd.DataFrame,
    framing: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (topics, fulltext, mainstream_emotion, framing):
        df["source_norm"] = normalize_source(df["source"])
        df["row_id"] = df["row_id"].astype(str)

    topics["Date"] = pd.to_datetime(topics["Date"], errors="coerce")
    fulltext["Date"] = pd.to_datetime(fulltext["Date"], errors="coerce")

    for col in EMOTION_COLS:
        if col in fulltext.columns:
            fulltext[col] = pd.to_numeric(fulltext[col], errors="coerce")
        if col in mainstream_emotion.columns:
            mainstream_emotion[col] = pd.to_numeric(mainstream_emotion[col], errors="coerce")

    article = topics.merge(
        fulltext[
            [
                "source_norm",
                "row_id",
                "Date",
                "emotion_dominant",
                *EMOTION_COLS,
            ]
        ],
        on=["source_norm", "row_id"],
        how="left",
        suffixes=("", "_fulltext"),
    )
    article["outlet_group"] = article["source_norm"].map(OUTLET_GROUPS).fillna("unknown")
    article["article_month"] = article["Date"].dt.to_period("M").dt.to_timestamp()

    mainstream_hits = mainstream_emotion.copy()
    mainstream_hits["is_delegit"] = mainstream_hits["category"].isin(DELEGIT_CATEGORIES).astype(int)
    mainstream_hits["outlet_group"] = mainstream_hits["source_norm"].map(OUTLET_GROUPS).fillna("unknown")

    return article, mainstream_hits


def build_article_framing_agg(mainstream_hits: pd.DataFrame) -> pd.DataFrame:
    category_share = (
        mainstream_hits.groupby(["source_norm", "row_id", "category"]).size().rename("n").reset_index()
    )
    totals = category_share.groupby(["source_norm", "row_id"])["n"].sum().rename("n_hits").reset_index()
    category_share = category_share.merge(totals, on=["source_norm", "row_id"], how="left")
    category_share["share"] = category_share["n"] / category_share["n_hits"]
    category_wide = category_share.pivot_table(
        index=["source_norm", "row_id"],
        columns="category",
        values="share",
        fill_value=0.0,
    ).reset_index()
    category_wide.columns = [
        c if isinstance(c, str) else c[1] for c in category_wide.columns.to_flat_index()
    ]

    agg_dict: dict[str, str] = {"is_delegit": "mean", "hit_id": "count"}
    for col in EMOTION_COLS:
        if col in mainstream_hits.columns:
            agg_dict[col] = "mean"
    framing_agg = (
        mainstream_hits.groupby(["source_norm", "row_id"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"is_delegit": "delegit_hit_share", "hit_id": "mainstream_hit_count"})
    )
    rename_emotions = {
        col: f"mainstream_hit_{col}"
        for col in EMOTION_COLS
        if col in framing_agg.columns
    }
    if rename_emotions:
        framing_agg = framing_agg.rename(columns=rename_emotions)
    framing_agg = framing_agg.merge(category_wide, on=["source_norm", "row_id"], how="left")
    return framing_agg


def compute_rq1(article: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_col = "Manual Topic Label"
    topic_dist = (
        article.dropna(subset=[topic_col])
        .groupby(["source_norm", topic_col])
        .size()
        .rename("n")
        .reset_index()
    )
    topic_dist["share"] = topic_dist["n"] / topic_dist.groupby("source_norm")["n"].transform("sum")

    baseline = (
        topic_dist[topic_dist["source_norm"] == "Tagesschau"][[topic_col, "share"]]
        .rename(columns={"share": "share_tagesschau"})
    )
    topic_vs_main = topic_dist.merge(baseline, on=topic_col, how="left")
    topic_vs_main["share_tagesschau"] = topic_vs_main["share_tagesschau"].fillna(0.0)
    eps = 1e-6
    topic_vs_main["log2_amplification_vs_tagesschau"] = np.log2(
        (topic_vs_main["share"] + eps) / (topic_vs_main["share_tagesschau"] + eps)
    )

    pivot = (
        topic_vs_main.pivot_table(
            index=topic_col,
            columns="source_norm",
            values="log2_amplification_vs_tagesschau",
            fill_value=0.0,
        )
        .reindex(columns=ALT_ONLY_OUTLETS)
        .fillna(0.0)
    )
    top_topics = (
        topic_dist.groupby(topic_col)["n"].sum().sort_values(ascending=False).head(25).index.tolist()
    )
    pivot = pivot.loc[pivot.index.intersection(top_topics)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot.sort_values(by=ALT_ONLY_OUTLETS[0], ascending=False),
        cmap="coolwarm",
        center=0.0,
        linewidths=0.3,
        cbar_kws={"label": "log2 amplification vs Tagesschau"},
    )
    plt.title("RQ1: Topic Amplification/Omission vs Tagesschau")
    plt.xlabel("Outlet")
    plt.ylabel("Manual Topic Label")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq1_topic_amplification_heatmap.png", dpi=250)
    plt.close()

    # Outlet-level scorecard
    score_rows: list[dict[str, float | str]] = []
    all_topics = sorted(topic_dist[topic_col].dropna().unique().tolist())
    outlet_share_map = {
        outlet: topic_dist[topic_dist["source_norm"] == outlet].set_index(topic_col)["share"].reindex(all_topics, fill_value=0.0)
        for outlet in topic_dist["source_norm"].unique()
    }
    main_share = outlet_share_map["Tagesschau"]
    for outlet, shares in outlet_share_map.items():
        entropy = -(shares[shares > 0] * np.log(shares[shares > 0])).sum() / np.log(len(all_topics))
        jsd = float(jensenshannon(shares, main_share, base=2.0) ** 2)
        spear = float(pd.Series(shares.values).corr(pd.Series(main_share.values), method="spearman"))
        score_rows.append(
            {
                "source_norm": outlet,
                "entropy_norm": entropy,
                "jsd_vs_tagesschau": jsd,
                "spearman_vs_tagesschau": spear,
            }
        )
    scorecard = pd.DataFrame(score_rows).sort_values("jsd_vs_tagesschau", ascending=False)
    scorecard["outlet_group"] = scorecard["source_norm"].map(OUTLET_GROUPS)
    scorecard.to_csv(TABLE_DIR / "rq1_outlet_divergence_scorecard.csv", index=False)

    plt.figure(figsize=(8, 5))
    score_plot = scorecard.sort_values("jsd_vs_tagesschau", ascending=False)
    sns.barplot(data=score_plot, x="jsd_vs_tagesschau", y="source_norm", hue="outlet_group", dodge=False)
    plt.title("RQ1: Jensen-Shannon Divergence vs Tagesschau")
    plt.xlabel("JSD")
    plt.ylabel("Outlet")
    plt.legend(loc="lower right", fontsize=8, title="Outlet group")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq1_jsd_bar.png", dpi=250)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=scorecard,
        x="entropy_norm",
        y="jsd_vs_tagesschau",
        hue="outlet_group",
        style="source_norm",
        s=120,
    )
    for _, row in scorecard.iterrows():
        plt.text(row["entropy_norm"] + 0.002, row["jsd_vs_tagesschau"] + 0.001, row["source_norm"], fontsize=8)
    plt.title("RQ1: Diversity Breadth vs Agenda Divergence")
    plt.xlabel("Normalized topic entropy")
    plt.ylabel("JSD vs Tagesschau")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq1_jsd_entropy_scatter.png", dpi=250)
    plt.close()

    ampl_rows = []
    for outlet in ALT_ONLY_OUTLETS:
        sub = topic_vs_main[topic_vs_main["source_norm"] == outlet].copy()
        sub = sub.sort_values("log2_amplification_vs_tagesschau", ascending=False)
        top = sub.head(10).assign(direction="amplified")
        bottom = sub.tail(10).assign(direction="omitted")
        ampl_rows.append(pd.concat([top, bottom], ignore_index=True))
    amplification_table = pd.concat(ampl_rows, ignore_index=True)
    amplification_table.to_csv(TABLE_DIR / "rq1_topic_amplification_top_bottom.csv", index=False)

    return topic_vs_main, scorecard


def compute_rq2(
    mainstream_hits: pd.DataFrame, article_with_topics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_outlet_cat = (
        mainstream_hits.groupby(["source_norm", "category"]).size().rename("n").reset_index()
    )
    by_outlet_total = by_outlet_cat.groupby("source_norm")["n"].sum().rename("n_total").reset_index()
    by_outlet_cat = by_outlet_cat.merge(by_outlet_total, on="source_norm", how="left")
    by_outlet_cat["share"] = by_outlet_cat["n"] / by_outlet_cat["n_total"]
    by_outlet_cat.to_csv(TABLE_DIR / "rq2_framing_category_profiles.csv", index=False)

    stacked = by_outlet_cat.pivot_table(index="source_norm", columns="category", values="share", fill_value=0.0)
    stacked = stacked.reindex(index=["Tagesschau", *ALT_ONLY_OUTLETS]).fillna(0.0)
    stacked.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="tab20")
    plt.title("RQ2: Framing Category Composition by Outlet")
    plt.ylabel("Share of mainstream-reference hits")
    plt.xlabel("Outlet")
    plt.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq2_category_stacked.png", dpi=250)
    plt.close()

    tagesschau_ref = stacked.loc["Tagesschau"]
    slope_df = stacked.loc[[*ALT_ONLY_OUTLETS]].copy().stack().rename("share").reset_index()
    slope_df.columns = ["source_norm", "category", "share"]
    slope_df["share_tagesschau"] = slope_df["category"].map(tagesschau_ref.to_dict())
    slope_df["delta_vs_tagesschau"] = slope_df["share"] - slope_df["share_tagesschau"]
    slope_df.to_csv(TABLE_DIR / "rq2_category_delta_vs_tagesschau.csv", index=False)

    plt.figure(figsize=(10, 8))
    categories = slope_df["category"].unique().tolist()
    outlets = ALT_ONLY_OUTLETS
    for i, cat in enumerate(categories):
        cat_sub = slope_df[slope_df["category"] == cat]
        x = np.arange(len(outlets))
        y = cat_sub.set_index("source_norm").reindex(outlets)["delta_vs_tagesschau"].values
        plt.plot(x, y, marker="o", label=cat)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xticks(np.arange(len(outlets)), outlets, rotation=25, ha="right")
    plt.ylabel("Share difference vs Tagesschau")
    plt.title("RQ2: Framing Category Shifts vs Tagesschau")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq2_slope_vs_tagesschau.png", dpi=250)
    plt.close()

    topic_hits = mainstream_hits.merge(
        article_with_topics[
            ["source_norm", "row_id", "Manual Topic Label", "Manual Cluster Label", "outlet_group"]
        ],
        on=["source_norm", "row_id"],
        how="left",
    )
    topic_hits["is_delegit"] = topic_hits["category"].isin(DELEGIT_CATEGORIES).astype(int)
    topic_delegit = (
        topic_hits.dropna(subset=["Manual Topic Label"])
        .groupby(["source_norm", "Manual Topic Label"], as_index=False)
        .agg(hit_n=("hit_id", "count"), delegit_share=("is_delegit", "mean"))
    )
    topic_delegit = topic_delegit[topic_delegit["hit_n"] >= 20].copy()
    topic_delegit.to_csv(TABLE_DIR / "rq2_topic_delegitimization_table.csv", index=False)

    heat = topic_delegit.pivot_table(
        index="Manual Topic Label", columns="source_norm", values="delegit_share", fill_value=np.nan
    )
    top_rows = topic_delegit.groupby("Manual Topic Label")["hit_n"].sum().sort_values(ascending=False).head(20).index
    heat = heat.reindex(index=top_rows, columns=ALT_ONLY_OUTLETS)
    plt.figure(figsize=(11, 9))
    sns.heatmap(heat, cmap="mako", vmin=0, vmax=1, linewidths=0.2, cbar_kws={"label": "Delegitimization share"})
    plt.title("RQ2: Topic-Level Delegitimization Intensity")
    plt.xlabel("Outlet")
    plt.ylabel("Manual Topic Label")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq2_topic_delegit_heatmap.png", dpi=250)
    plt.close()

    return by_outlet_cat, topic_delegit


def compute_rq3(article_with_topics: pd.DataFrame) -> pd.DataFrame:
    df = article_with_topics.copy()
    for col in ("emotion_anger", "emotion_fear"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["z_anger"] = (df["emotion_anger"] - df["emotion_anger"].mean()) / df["emotion_anger"].std(ddof=0)
    df["z_fear"] = (df["emotion_fear"] - df["emotion_fear"].mean()) / df["emotion_fear"].std(ddof=0)
    df["ami"] = (df["z_anger"] + df["z_fear"]) / 2
    df["ami_sensitivity_raw_mean"] = (df["emotion_anger"] + df["emotion_fear"]) / 2

    outlet_rows = []
    for outlet, sub in df.groupby("source_norm"):
        b_ami = bootstrap_mean(sub["ami"].to_numpy())
        b_anger = bootstrap_mean(sub["emotion_anger"].to_numpy())
        b_fear = bootstrap_mean(sub["emotion_fear"].to_numpy())
        outlet_rows.append(
            {
                "source_norm": outlet,
                "n_articles": len(sub),
                "ami_mean": b_ami.mean,
                "ami_ci_low": b_ami.low,
                "ami_ci_high": b_ami.high,
                "anger_mean": b_anger.mean,
                "fear_mean": b_fear.mean,
                "outlet_group": OUTLET_GROUPS.get(outlet, "unknown"),
            }
        )
    outlet_ami = pd.DataFrame(outlet_rows).sort_values("ami_mean", ascending=False)
    outlet_ami.to_csv(TABLE_DIR / "rq3_ami_outlet_bootstrap_ci.csv", index=False)

    plt.figure(figsize=(10, 6))
    plot_df = df[df["source_norm"].isin(["Tagesschau", *ALT_ONLY_OUTLETS])].copy()
    sns.boxplot(
        data=plot_df,
        x="source_norm",
        y="ami",
        hue="outlet_group",
        showfliers=False,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("RQ3: Affective Mobilization Index (AMI) by Outlet (Boxplots)")
    plt.xlabel("Outlet")
    plt.ylabel("AMI")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq3_ami_violin.png", dpi=250)
    plt.close()

    monthly = (
        df.dropna(subset=["article_month"])
        .groupby(["article_month", "outlet_group"], as_index=False)
        .agg(ami_mean=("ami", "mean"), n=("ami", "size"))
    )
    monthly = monthly[monthly["n"] >= 20].copy()
    monthly.to_csv(TABLE_DIR / "rq3_monthly_ami_trends.csv", index=False)
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=monthly, x="article_month", y="ami_mean", hue="outlet_group", marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("RQ3: Monthly AMI Trends by Outlet Group")
    plt.xlabel("Month")
    plt.ylabel("Mean AMI")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq3_monthly_trends.png", dpi=250)
    plt.close()

    topic_ami = (
        df.dropna(subset=["Manual Topic Label"])
        .groupby(["source_norm", "Manual Topic Label"], as_index=False)
        .agg(n=("ami", "size"), ami_mean=("ami", "mean"))
    )
    topic_ami = topic_ami[topic_ami["n"] >= 20]
    topic_ami.to_csv(TABLE_DIR / "rq3_topic_ami_table.csv", index=False)
    pivot = topic_ami.pivot_table(
        index="Manual Topic Label", columns="source_norm", values="ami_mean", fill_value=np.nan
    ).reindex(columns=ALT_ONLY_OUTLETS)
    top_topics = topic_ami.groupby("Manual Topic Label")["n"].sum().sort_values(ascending=False).head(20).index
    pivot = pivot.reindex(index=top_topics)
    plt.figure(figsize=(11, 9))
    sns.heatmap(pivot, cmap="rocket", center=0.0, linewidths=0.2, cbar_kws={"label": "Mean AMI"})
    plt.title("RQ3: Topic-Outlet AMI Heatmap")
    plt.xlabel("Outlet")
    plt.ylabel("Manual Topic Label")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rq3_topic_ami_heatmap.png", dpi=250)
    plt.close()

    return df


def _zscore_col(df: pd.DataFrame, col: str) -> pd.Series:
    std = df[col].std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (df[col] - df[col].mean()) / std


def compute_synthesis(
    article_enriched: pd.DataFrame,
    topic_vs_main: pd.DataFrame,
    topic_delegit: pd.DataFrame,
) -> pd.DataFrame:
    topic_amp = topic_vs_main[
        ["source_norm", "Manual Topic Label", "log2_amplification_vs_tagesschau", "share", "share_tagesschau"]
    ].copy()
    topic_ami = (
        article_enriched.dropna(subset=["Manual Topic Label"])
        .groupby(["source_norm", "Manual Topic Label"], as_index=False)
        .agg(ami_mean=("ami", "mean"), article_n=("row_id", "size"))
    )
    synth = topic_amp.merge(topic_delegit, on=["source_norm", "Manual Topic Label"], how="left")
    synth = synth.merge(topic_ami, on=["source_norm", "Manual Topic Label"], how="left")
    synth["outlet_group"] = synth["source_norm"].map(OUTLET_GROUPS).fillna("unknown")
    synth["delegit_share"] = synth["delegit_share"].fillna(0.0)
    synth["z_amp"] = _zscore_col(synth.fillna(0), "log2_amplification_vs_tagesschau")
    synth["z_delegit"] = _zscore_col(synth.fillna(0), "delegit_share")
    synth["z_ami"] = _zscore_col(synth.fillna(0), "ami_mean")
    synth["convergent_score"] = synth["z_amp"] + synth["z_delegit"] + synth["z_ami"]
    synth.to_csv(TABLE_DIR / "synthesis_topic_outlet_matrix.csv", index=False)

    shortlist = (
        synth[(synth["source_norm"] != "Tagesschau") & (synth["article_n"] >= 20)]
        .sort_values("convergent_score", ascending=False)
        .head(30)
    )
    shortlist.to_csv(TABLE_DIR / "synthesis_convergent_mechanism_shortlist.csv", index=False)

    bubble = shortlist.copy()
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=bubble,
        x="log2_amplification_vs_tagesschau",
        y="delegit_share",
        hue="ami_mean",
        size="article_n",
        sizes=(40, 350),
        palette="viridis",
        alpha=0.8,
    )
    for _, row in bubble.head(12).iterrows():
        label = f"{row['source_norm']} | {str(row['Manual Topic Label'])[:24]}"
        plt.text(row["log2_amplification_vs_tagesschau"], row["delegit_share"], label, fontsize=7)
    plt.title("Cross-Mechanism Synthesis: Amplification vs Delegitimization vs AMI")
    plt.xlabel("Topic amplification log2 vs Tagesschau")
    plt.ylabel("Delegitimization share")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "synthesis_bubble_plot.png", dpi=250)
    plt.close()

    model_df = article_enriched.dropna(subset=["ami", "delegit_hit_share", "Manual Cluster Label"]).copy()
    if not model_df.empty:
        mod1 = smf.ols("ami ~ delegit_hit_share + C(outlet_group) + C(Q('Manual Cluster Label'))", data=model_df).fit()
        (TABLE_DIR / "synthesis_model_ami_vs_delegit.txt").write_text(mod1.summary().as_text(), encoding="utf-8")

    topic_model_df = synth.dropna(
        subset=["log2_amplification_vs_tagesschau", "delegit_share", "ami_mean", "article_n"]
    ).copy()
    topic_model_df = topic_model_df[topic_model_df["article_n"] >= 20]
    if not topic_model_df.empty:
        mod2 = smf.wls(
            "log2_amplification_vs_tagesschau ~ delegit_share + ami_mean + C(outlet_group)",
            data=topic_model_df,
            weights=topic_model_df["article_n"],
        ).fit()
        (TABLE_DIR / "synthesis_model_amplification_vs_mechanisms.txt").write_text(
            mod2.summary().as_text(), encoding="utf-8"
        )
    return synth


def export_integrated_dataset(article: pd.DataFrame, framing_agg: pd.DataFrame) -> pd.DataFrame:
    integrated = article.merge(framing_agg, on=["source_norm", "row_id"], how="left")
    integrated["delegit_hit_share"] = integrated["delegit_hit_share"].fillna(0.0)
    integrated["mainstream_hit_count"] = integrated["mainstream_hit_count"].fillna(0.0)
    integrated.to_csv(INTEGRATED_OUTPUT_CSV, index=False)
    return integrated


def render_key_findings(
    scorecard: pd.DataFrame, outlet_ami: pd.DataFrame, shortlist: pd.DataFrame
) -> str:
    top_jsd = scorecard.sort_values("jsd_vs_tagesschau", ascending=False).head(3)
    top_ami = outlet_ami.sort_values("ami_mean", ascending=False).head(3)
    top_conv = shortlist.head(5)

    jsd_lines = "\n".join(
        f"- {row.source_norm}: JSD={row.jsd_vs_tagesschau:.3f}, entropy={row.entropy_norm:.3f}"
        for row in top_jsd.itertuples(index=False)
    )
    ami_lines = "\n".join(
        f"- {row.source_norm}: AMI={row.ami_mean:.3f} (95% CI {row.ami_ci_low:.3f} to {row.ami_ci_high:.3f})"
        for row in top_ami.itertuples(index=False)
    )
    conv_lines = "\n".join(
        (
            f"- {row['source_norm']} | {row['Manual Topic Label']}: "
            f"score={row['convergent_score']:.2f}, "
            f"amp={row['log2_amplification_vs_tagesschau']:.2f}, "
            f"delegit={row['delegit_share']:.2f}, ami={row['ami_mean']:.2f}"
        )
        for _, row in top_conv.iterrows()
    )
    return (
        "## Key Findings Snapshot\n\n"
        "### Strongest agenda divergence vs Tagesschau\n"
        f"{jsd_lines}\n\n"
        "### Highest affective mobilization (AMI)\n"
        f"{ami_lines}\n\n"
        "### Top convergent mechanism topic-outlet cells\n"
        f"{conv_lines}\n"
    )


def build_notebook(
    scorecard: pd.DataFrame, outlet_ami: pd.DataFrame, shortlist: pd.DataFrame
) -> None:
    nb = nbf.v4.new_notebook()
    findings_md = render_key_findings(scorecard, outlet_ami, shortlist)
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Crossfunctional Deep Analysis\n"
            "This notebook documents the integrated RQ1-RQ3 analysis run built from the four constrained CSV inputs."
        ),
        nbf.v4.new_markdown_cell(
            "## Inputs and Artifacts\n"
            "- `outputs/03_new_topic_assignments.csv`\n"
            "- `outputs/emotion_fulltext_with_topiclabels.csv`\n"
            "- `outputs/emotion_mainstream_resultsv3.csv`\n"
            "- `outputs/media_framing_final_run_5_4mini_classifications.csv`\n\n"
            "Generated artifacts are saved into `figures/`, `tables/`, and `outputs/crossfunctional_integrated_dataset.csv`."
        ),
        nbf.v4.new_markdown_cell(findings_md),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "base = Path('05_Crossfunctional_Analysis')\n"
            "scorecard = pd.read_csv(base / 'tables/rq1_outlet_divergence_scorecard.csv')\n"
            "ami = pd.read_csv(base / 'tables/rq3_ami_outlet_bootstrap_ci.csv')\n"
            "shortlist = pd.read_csv(base / 'tables/synthesis_convergent_mechanism_shortlist.csv')\n"
            "display(scorecard)\n"
            "display(ami)\n"
            "display(shortlist.head(15))"
        ),
        nbf.v4.new_markdown_cell(
            "## Figure Index\n"
            "- `figures/rq1_topic_amplification_heatmap.png`\n"
            "- `figures/rq1_jsd_bar.png`\n"
            "- `figures/rq1_jsd_entropy_scatter.png`\n"
            "- `figures/rq2_category_stacked.png`\n"
            "- `figures/rq2_slope_vs_tagesschau.png`\n"
            "- `figures/rq2_topic_delegit_heatmap.png`\n"
            "- `figures/rq3_ami_violin.png`\n"
            "- `figures/rq3_monthly_trends.png`\n"
            "- `figures/rq3_topic_ami_heatmap.png`\n"
            "- `figures/synthesis_bubble_plot.png`"
        ),
        nbf.v4.new_markdown_cell(
            "## Method Notes\n"
            "- Source-name harmonization is applied before joins.\n"
            "- Hit-level framing is aggregated to article level for synthesis.\n"
            "- AMI is `(z(anger)+z(fear))/2`, with a raw-score sensitivity column exported.\n"
            "- Uncertainty is represented with bootstrap confidence intervals on outlet-level means."
        ),
    ]
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    topics, fulltext, mainstream_emotion, framing = load_inputs()
    article_base, mainstream_hits = prepare_base_frames(topics, fulltext, mainstream_emotion, framing)
    framing_agg = build_article_framing_agg(mainstream_hits)
    article = export_integrated_dataset(article_base, framing_agg)

    topic_vs_main, scorecard = compute_rq1(article)
    _, topic_delegit = compute_rq2(mainstream_hits, article)
    article = compute_rq3(article)
    synth = compute_synthesis(article, topic_vs_main, topic_delegit)
    shortlist = pd.read_csv(TABLE_DIR / "synthesis_convergent_mechanism_shortlist.csv")
    outlet_ami = pd.read_csv(TABLE_DIR / "rq3_ami_outlet_bootstrap_ci.csv")
    build_notebook(scorecard, outlet_ami, shortlist)

    # Refresh integrated output with AMI columns from RQ3 run.
    article.to_csv(INTEGRATED_OUTPUT_CSV, index=False)
    synth.to_csv(TABLE_DIR / "synthesis_topic_outlet_matrix.csv", index=False)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
