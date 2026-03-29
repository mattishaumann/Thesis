"""
H1 Experiment — Metrics Module
================================
Size-controlled KPIs for measuring agenda distortion per outlet,
each grounded in media communication literature.

All metrics compare individual outlets against Tagesschau (mainstream
reference), not "alt media vs mainstream" as a binary — each outlet
gets its own distortion profile.

Metrics:
    1. Normalized Shannon entropy — topic concentration (Boydstun 2014)
    2. JSD per outlet vs Tagesschau — agenda divergence (DiMaggio et al. 2013)
    3. Spearman rank correlation — topic priority agreement (McCombs & Shaw 1972)
    4. Top-K overlap — shared top topics (Heidenreich et al. 2019)
    5. Coverage breadth (relative) — topic range vs expected (size-controlled)

All metrics are normalized / rank-based to handle class imbalance
(corpus sizes range from 565 to 6,320 articles).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def normalized_entropy(topic_counts: pd.Series) -> float:
    """Shannon entropy normalized by log(K) where K = number of topics.

    Range: 0 (all articles in one topic) to 1 (uniform distribution).
    Lower values indicate stronger topic concentration — a TYPE B
    distortion signal.

    Literature: Boydstun et al. (2014) use normalized entropy to
    measure media attention diversity.
    """
    counts = topic_counts[topic_counts > 0]
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    H = float(-(probs * np.log(probs)).sum())
    H_max = math.log(len(counts))
    return H / H_max if H_max > 0 else 0.0


def jsd_vs_reference(
    outlet_dist: np.ndarray,
    reference_dist: np.ndarray,
) -> float:
    """Jensen-Shannon Divergence between outlet and reference topic distributions.

    Range: 0 (identical agendas) to 1 (completely disjoint).
    JSD is symmetric and bounded, unlike KL divergence.

    Literature: DiMaggio et al. (2013) use JSD on topic distributions
    to measure ideological/agenda distance between corpora.
    """
    # Add small epsilon to avoid division issues
    eps = 1e-10
    p = outlet_dist + eps
    q = reference_dist + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2) ** 2)


def spearman_rank_correlation(
    outlet_dist: np.ndarray,
    reference_dist: np.ndarray,
) -> tuple[float, float]:
    """Spearman rank correlation of topic prevalence between outlet and reference.

    Range: -1 (perfectly inverted priorities) to +1 (identical priorities).
    Rank-based, so completely size-independent.

    Literature: McCombs & Shaw (1972) used rank-order correlation as
    the original measure of agenda-setting correspondence.

    Returns:
        (rho, p_value) tuple.
    """
    rho, pval = spearmanr(outlet_dist, reference_dist)
    return float(rho), float(pval)


def topk_overlap(
    outlet_top_k: list[int],
    reference_top_k: list[int],
) -> float:
    """Fraction of outlet's top-K topics that appear in reference's top-K.

    Range: 0 (no overlap) to 1 (identical top-K).
    Rank-based and size-independent.

    Literature: Heidenreich et al. (2019) use topic overlap to measure
    media fragmentation.
    """
    if not outlet_top_k:
        return 0.0
    shared = set(outlet_top_k) & set(reference_top_k)
    return len(shared) / len(outlet_top_k)


def coverage_breadth_relative(
    outlet_topic_counts: pd.Series,
    corpus_topic_shares: pd.Series,
    n_outlet_articles: int,
    min_articles: int = 10,
) -> float:
    """Ratio of actual topic coverage to expected coverage given corpus size.

    < 1.0 = narrower topic range than size predicts (TYPE A signal).
    > 1.0 = broader than expected.
    = 1.0 = exactly as broad as random sampling would predict.

    Size-controlled via binomial expectation model.
    """
    from scipy.stats import binom

    actual_covered = int((outlet_topic_counts >= min_articles).sum())
    total_topics = len(corpus_topic_shares)

    if total_topics == 0 or n_outlet_articles == 0:
        return float("nan")

    expected = 0.0
    for p_topic in corpus_topic_shares:
        expected += 1.0 - binom.cdf(min_articles - 1, n_outlet_articles, float(p_topic))

    actual_ratio = actual_covered / total_topics
    expected_ratio = expected / total_topics

    if expected_ratio == 0:
        return float("nan")

    return actual_ratio / expected_ratio


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


def compute_all_h1_metrics(
    merged_articles: pd.DataFrame,
    reference_label: str = "Tagesschau",
    top_k: int = 10,
    min_articles: int = 10,
) -> pd.DataFrame:
    """Compute all H1 metrics for every outlet vs the reference.

    Args:
        merged_articles: article-level frame with merged_topic, outlet_label.
        reference_label: the mainstream reference outlet.
        top_k: number of top topics for overlap metric.
        min_articles: minimum articles for coverage breadth.

    Returns:
        DataFrame with one row per outlet and columns for each metric.
    """
    # Filter to non-outlier articles
    topic_df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic"],
    ].copy()

    all_outlets = sorted(merged_articles["outlet_label"].dropna().unique())
    all_topics = sorted(topic_df["merged_topic"].unique())
    n_topics = len(all_topics)

    # Build outlet × topic count matrix
    outlet_topic_counts = (
        topic_df.groupby(["outlet_label", "merged_topic"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_outlets, columns=all_topics, fill_value=0)
        .astype(float)
    )

    # Corpus-level topic shares (for coverage breadth expectation)
    corpus_topic_totals = outlet_topic_counts.sum(axis=0)
    corpus_total = corpus_topic_totals.sum()
    corpus_topic_shares = corpus_topic_totals / corpus_total if corpus_total > 0 else corpus_topic_totals * 0

    # Reference distribution
    ref_counts = outlet_topic_counts.loc[reference_label]
    ref_dist = (ref_counts / ref_counts.sum()).values if ref_counts.sum() > 0 else ref_counts.values
    ref_top_k = ref_counts.nlargest(top_k).index.tolist()

    rows = []
    for outlet in all_outlets:
        counts = outlet_topic_counts.loc[outlet]
        n_total = int(merged_articles[merged_articles["outlet_label"] == outlet].shape[0])
        n_topic_articles = int(counts.sum())

        # Distribution
        dist = (counts / counts.sum()).values if counts.sum() > 0 else counts.values

        # 1. Normalized entropy
        entropy = normalized_entropy(counts)

        # 2. JSD vs reference
        jsd = jsd_vs_reference(dist, ref_dist)
        if outlet == reference_label:
            jsd = 0.0  # by definition

        # 3. Spearman rank correlation
        rho, rho_pval = spearman_rank_correlation(dist, ref_dist)
        if outlet == reference_label:
            rho, rho_pval = 1.0, 0.0

        # 4. Top-K overlap
        outlet_top_k = counts.nlargest(top_k).index.tolist()
        overlap = topk_overlap(outlet_top_k, ref_top_k)
        if outlet == reference_label:
            overlap = 1.0

        # 5. Coverage breadth (relative)
        breadth = coverage_breadth_relative(
            counts, corpus_topic_shares, n_topic_articles, min_articles
        )

        rows.append({
            "outlet_label": outlet,
            "n_articles": n_total,
            "n_topic_articles": n_topic_articles,
            "corpus_share": round(n_total / len(merged_articles), 4),
            "outlier_rate": round(1 - n_topic_articles / n_total, 4) if n_total > 0 else 0.0,
            "entropy_normalized": round(entropy, 4),
            "jsd_vs_tagesschau": round(jsd, 4),
            "spearman_rho": round(rho, 4),
            "spearman_pval": round(rho_pval, 6),
            "topk_overlap": round(overlap, 4),
            "coverage_breadth_relative": round(breadth, 4),
            "size_warning": "SMALL" if n_topic_articles < 1000 else "OK",
        })

    return pd.DataFrame(rows).sort_values("outlet_label").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_h1_metrics(
    merged_articles: pd.DataFrame,
    reference_label: str = "Tagesschau",
    top_k: int = 10,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    min_articles: int = 10,
) -> pd.DataFrame:
    """Bootstrap 95% CIs for JSD, Spearman rho, and top-K overlap.

    Stratified resampling within each outlet to preserve relative
    corpus sizes while generating sampling distributions.

    Returns:
        DataFrame with columns: outlet_label, metric, point_estimate,
        ci_lower, ci_upper.
    """
    rng = np.random.RandomState(random_state)

    topic_df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic"],
    ].copy()

    all_outlets = sorted(topic_df["outlet_label"].unique())
    all_topics = sorted(topic_df["merged_topic"].unique())

    # Point estimates
    point_metrics = compute_all_h1_metrics(
        merged_articles, reference_label, top_k, min_articles
    )

    # Bootstrap
    boot_results = {outlet: {"jsd": [], "rho": [], "overlap": []} for outlet in all_outlets}

    for b in range(n_bootstrap):
        # Resample within each outlet
        boot_dfs = []
        for outlet in all_outlets:
            outlet_df = topic_df[topic_df["outlet_label"] == outlet]
            boot_sample = outlet_df.sample(n=len(outlet_df), replace=True, random_state=rng)
            boot_dfs.append(boot_sample)
        boot_df = pd.concat(boot_dfs, ignore_index=True)

        # Build count matrix
        counts_matrix = (
            boot_df.groupby(["outlet_label", "merged_topic"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=all_outlets, columns=all_topics, fill_value=0)
            .astype(float)
        )

        ref_counts = counts_matrix.loc[reference_label]
        ref_dist = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else ref_counts * 0
        ref_top_k_ids = ref_counts.nlargest(top_k).index.tolist()

        for outlet in all_outlets:
            o_counts = counts_matrix.loc[outlet]
            o_dist = o_counts / o_counts.sum() if o_counts.sum() > 0 else o_counts * 0

            jsd = jsd_vs_reference(o_dist.values, ref_dist.values)
            if outlet == reference_label:
                jsd = 0.0
            boot_results[outlet]["jsd"].append(jsd)

            rho, _ = spearman_rank_correlation(o_dist.values, ref_dist.values)
            if outlet == reference_label:
                rho = 1.0
            boot_results[outlet]["rho"].append(rho)

            o_top_k = o_counts.nlargest(top_k).index.tolist()
            overlap = topk_overlap(o_top_k, ref_top_k_ids)
            if outlet == reference_label:
                overlap = 1.0
            boot_results[outlet]["overlap"].append(overlap)

    # Assemble results
    rows = []
    metric_map = {
        "jsd": "jsd_vs_tagesschau",
        "rho": "spearman_rho",
        "overlap": "topk_overlap",
    }
    for outlet in all_outlets:
        for boot_key, point_col in metric_map.items():
            samples = np.array(boot_results[outlet][boot_key])
            point_val = float(
                point_metrics.loc[
                    point_metrics["outlet_label"] == outlet, point_col
                ].iloc[0]
            )
            rows.append({
                "outlet_label": outlet,
                "metric": point_col,
                "point_estimate": round(point_val, 4),
                "ci_lower": round(float(np.percentile(samples, 2.5)), 4),
                "ci_upper": round(float(np.percentile(samples, 97.5)), 4),
                "boot_mean": round(float(samples.mean()), 4),
                "boot_std": round(float(samples.std()), 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------


def classify_distortion(
    metrics_df: pd.DataFrame,
    reference_label: str = "Tagesschau",
    breadth_threshold: float = 0.90,
    jsd_threshold: float = 0.10,
    entropy_threshold: float = 0.85,
) -> pd.DataFrame:
    """Classify each outlet's distortion type based on metric thresholds.

    Types:
        REFERENCE — the mainstream baseline
        TYPE A — breadth restriction (covers fewer topics than expected)
        TYPE B — concentration (focuses heavily on subset of topics)
        TYPE AB — both breadth restriction and concentration
        UNCLEAR — no strong signal on either dimension
    """
    df = metrics_df.copy()
    types = []

    for _, row in df.iterrows():
        if row["outlet_label"] == reference_label:
            types.append("REFERENCE")
            continue

        type_a = row["coverage_breadth_relative"] < breadth_threshold
        type_b = (
            row["jsd_vs_tagesschau"] >= jsd_threshold
            or row["entropy_normalized"] < entropy_threshold
        )

        if type_a and type_b:
            types.append("TYPE AB")
        elif type_a:
            types.append("TYPE A")
        elif type_b:
            types.append("TYPE B")
        else:
            types.append("UNCLEAR")

    df["distortion_type"] = types
    return df


# ---------------------------------------------------------------------------
# Chi-squared per-topic significance tests
# ---------------------------------------------------------------------------


def chi_squared_topic_tests(
    merged_articles: pd.DataFrame,
    reference_label: str = "Tagesschau",
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Test per-topic over/under-representation for each outlet vs reference.

    For each (outlet, topic) pair, runs a 2x2 chi-squared test:
        - Rows: outlet vs reference
        - Columns: topic vs all-other-topics

    Returns a DataFrame with one row per (outlet, topic) pair including
    observed/expected counts, fold change, raw p-value, and corrected p-value.

    Literature: Standard contingency table approach for comparing
    topic proportions across corpora.

    Args:
        merged_articles: article-level frame with merged_topic, outlet_label.
        reference_label: the mainstream reference outlet.
        correction: multiple testing correction method (statsmodels).
            'fdr_bh' = Benjamini-Hochberg FDR (default).

    Returns:
        DataFrame sorted by corrected p-value.
    """
    from scipy.stats import chi2_contingency
    from statsmodels.stats.multitest import multipletests

    topic_df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic"],
    ].copy()

    all_topics = sorted(topic_df["merged_topic"].unique())
    alt_outlets = sorted(
        o for o in topic_df["outlet_label"].unique() if o != reference_label
    )

    # Pre-compute counts
    ref_df = topic_df[topic_df["outlet_label"] == reference_label]
    ref_total = len(ref_df)
    ref_topic_counts = ref_df["merged_topic"].value_counts()

    # Get display labels if available
    label_map = {}
    if "merged_display_label" in merged_articles.columns:
        label_map = dict(
            merged_articles.dropna(subset=["merged_display_label"])
            .drop_duplicates("merged_topic")[["merged_topic", "merged_display_label"]]
            .values
        )

    rows = []
    for outlet in alt_outlets:
        out_df = topic_df[topic_df["outlet_label"] == outlet]
        out_total = len(out_df)
        out_topic_counts = out_df["merged_topic"].value_counts()

        for topic in all_topics:
            a = int(out_topic_counts.get(topic, 0))       # outlet, this topic
            b = out_total - a                               # outlet, other topics
            c = int(ref_topic_counts.get(topic, 0))        # ref, this topic
            d = ref_total - c                               # ref, other topics

            table = np.array([[a, b], [c, d]])

            # Skip if any marginal is 0
            if table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
                continue

            chi2, pval, dof, expected = chi2_contingency(table, correction=True)

            outlet_share = a / out_total if out_total > 0 else 0
            ref_share = c / ref_total if ref_total > 0 else 0
            fold_change = outlet_share / ref_share if ref_share > 0 else float("inf")

            rows.append({
                "outlet_label": outlet,
                "merged_topic": topic,
                "topic_label": label_map.get(topic, f"Topic {topic}"),
                "outlet_count": a,
                "outlet_total": out_total,
                "outlet_share": round(outlet_share, 4),
                "ref_count": c,
                "ref_total": ref_total,
                "ref_share": round(ref_share, 4),
                "diff_pp": round((outlet_share - ref_share) * 100, 2),  # percentage points
                "fold_change": round(fold_change, 2),
                "chi2": round(chi2, 2),
                "p_raw": pval,
            })

    result = pd.DataFrame(rows)

    # Multiple testing correction
    if len(result) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            result["p_raw"], method=correction
        )
        result["p_corrected"] = pvals_corrected
        result["significant"] = reject

    return result.sort_values("p_corrected").reset_index(drop=True)


def summarize_significant_topics(
    chi_df: pd.DataFrame,
    alpha: float = 0.01,
    min_diff_pp: float = 1.0,
) -> pd.DataFrame:
    """Filter chi-squared results to significant over/under-representations.

    Args:
        chi_df: output of chi_squared_topic_tests().
        alpha: significance threshold on corrected p-value.
        min_diff_pp: minimum difference in percentage points.

    Returns:
        Filtered DataFrame with only significant, substantial differences.
    """
    mask = (
        (chi_df["p_corrected"] < alpha)
        & (chi_df["diff_pp"].abs() >= min_diff_pp)
    )
    return chi_df[mask].copy()


# ---------------------------------------------------------------------------
# Pairwise outlet distance matrix
# ---------------------------------------------------------------------------


def pairwise_outlet_jsd(
    merged_articles: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise JSD between all outlet topic distributions.

    Returns a symmetric N×N DataFrame (outlets as both index and columns).
    Can be used for hierarchical clustering and dendrogram visualization.

    Literature: Muller & Freudenthaler (2022) clustered outlets by
    topic proportions to identify media landscape structure.
    """
    topic_df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic"],
    ].copy()

    all_outlets = sorted(topic_df["outlet_label"].unique())
    all_topics = sorted(topic_df["merged_topic"].unique())

    # Build outlet × topic proportion matrix
    counts = (
        topic_df.groupby(["outlet_label", "merged_topic"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_outlets, columns=all_topics, fill_value=0)
        .astype(float)
    )
    props = counts.div(counts.sum(axis=1), axis=0)

    n = len(all_outlets)
    jsd_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = jsd_vs_reference(props.iloc[i].values, props.iloc[j].values)
            jsd_matrix[i, j] = d
            jsd_matrix[j, i] = d

    return pd.DataFrame(jsd_matrix, index=all_outlets, columns=all_outlets)


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------


def weekly_topic_proportions(
    merged_articles: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Compute weekly topic proportions per outlet.

    Returns a long-form DataFrame with columns:
        outlet_label, week, merged_topic, count, total, proportion

    Literature: Vargo & Guo (2017) used daily topic proportions for
    intermedia agenda-setting analysis.
    """
    df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic", date_col],
    ].copy()

    # Handle mixed timezone dates: strip timezone info before parsing
    raw_dates = df[date_col].astype(str).str.replace(r"\+\d{2}:\d{2}$", "", regex=True)
    df[date_col] = pd.to_datetime(raw_dates, errors="coerce")
    df = df.dropna(subset=[date_col])
    df["week"] = df[date_col].dt.to_period("W").dt.start_time

    # Count per (outlet, week, topic)
    counts = (
        df.groupby(["outlet_label", "week", "merged_topic"])
        .size()
        .reset_index(name="count")
    )

    # Total per (outlet, week)
    totals = (
        df.groupby(["outlet_label", "week"])
        .size()
        .reset_index(name="total")
    )

    result = counts.merge(totals, on=["outlet_label", "week"])
    result["proportion"] = result["count"] / result["total"]

    return result.sort_values(["outlet_label", "week", "merged_topic"]).reset_index(drop=True)


def time_lagged_correlations(
    weekly_props: pd.DataFrame,
    topic: int,
    reference_label: str = "Tagesschau",
    max_lag: int = 4,
) -> pd.DataFrame:
    """Compute time-lagged Pearson correlations for a specific topic.

    For each alt outlet, correlates its weekly topic proportion with
    Tagesschau's proportion at lags -max_lag to +max_lag weeks.

    Positive lag = alt outlet LEADS Tagesschau (alt at t, TS at t+lag).
    Negative lag = alt outlet FOLLOWS Tagesschau.

    Literature: Vargo & Guo (2017), Field et al. (2018) use lagged
    correlations to identify agenda-setting direction.

    Returns:
        DataFrame with columns: outlet_label, lag, correlation, p_value
    """
    from scipy.stats import pearsonr

    topic_data = weekly_props[weekly_props["merged_topic"] == topic].copy()

    # Pivot to wide: weeks as index, outlets as columns
    pivot = topic_data.pivot_table(
        index="week", columns="outlet_label", values="proportion", fill_value=0
    )

    if reference_label not in pivot.columns:
        return pd.DataFrame()

    ref_series = pivot[reference_label]
    alt_outlets = [c for c in pivot.columns if c != reference_label]

    rows = []
    for outlet in alt_outlets:
        alt_series = pivot[outlet]
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                # Alt leads: compare alt[:-lag] with ref[lag:]
                a = alt_series.iloc[:-lag].values if lag < len(alt_series) else np.array([])
                r = ref_series.iloc[lag:].values if lag < len(ref_series) else np.array([])
            elif lag < 0:
                # Alt follows: compare alt[-lag:] with ref[:lag]
                a = alt_series.iloc[-lag:].values
                r = ref_series.iloc[:lag].values
            else:
                a = alt_series.values
                r = ref_series.values

            if len(a) < 4 or len(r) < 4 or len(a) != len(r):
                continue

            corr, pval = pearsonr(a, r)
            rows.append({
                "outlet_label": outlet,
                "merged_topic": topic,
                "lag": lag,
                "correlation": round(float(corr), 4),
                "p_value": round(float(pval), 4),
                "n_weeks": len(a),
            })

    return pd.DataFrame(rows)


def rolling_jsd(
    weekly_props: pd.DataFrame,
    reference_label: str = "Tagesschau",
    window: int = 4,
) -> pd.DataFrame:
    """Compute rolling JSD between each outlet and reference over time.

    Uses a sliding window of `window` weeks. Shows how agenda distance
    evolves over the study period.

    Returns:
        DataFrame with columns: outlet_label, week_end, rolling_jsd
    """
    all_topics = sorted(weekly_props["merged_topic"].unique())
    topic_idx = {t: i for i, t in enumerate(all_topics)}
    n_topics = len(all_topics)

    weeks = sorted(weekly_props["week"].unique())
    all_outlets = sorted(weekly_props["outlet_label"].unique())
    alt_outlets = [o for o in all_outlets if o != reference_label]

    # Pre-build per-(outlet, week) count vectors for speed
    count_vectors = {}
    for (outlet, week), grp in weekly_props.groupby(["outlet_label", "week"]):
        vec = np.zeros(n_topics)
        for _, row in grp.iterrows():
            vec[topic_idx[row["merged_topic"]]] = row["count"]
        count_vectors[(outlet, week)] = vec

    rows = []
    for outlet in alt_outlets:
        for i in range(window - 1, len(weeks)):
            ref_dist = np.zeros(n_topics)
            out_dist = np.zeros(n_topics)

            for j in range(i - window + 1, i + 1):
                w = weeks[j]
                if (reference_label, w) in count_vectors:
                    ref_dist += count_vectors[(reference_label, w)]
                if (outlet, w) in count_vectors:
                    out_dist += count_vectors[(outlet, w)]

            if ref_dist.sum() > 0 and out_dist.sum() > 0:
                ref_dist = ref_dist / ref_dist.sum()
                out_dist = out_dist / out_dist.sum()
                jsd = jsd_vs_reference(out_dist, ref_dist)
                rows.append({
                    "outlet_label": outlet,
                    "week_end": weeks[i],
                    "rolling_jsd": round(float(jsd), 4),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Format metrics for display / thesis inclusion."""
    display_cols = [
        "outlet_label",
        "n_articles",
        "entropy_normalized",
        "jsd_vs_tagesschau",
        "spearman_rho",
        "topk_overlap",
        "coverage_breadth_relative",
        "size_warning",
    ]
    df = metrics_df[display_cols].copy()
    df.columns = [
        "Outlet",
        "Articles",
        "Entropy",
        "JSD",
        "Spearman ρ",
        "Top-K Overlap",
        "Breadth (rel.)",
        "Size",
    ]
    return df
