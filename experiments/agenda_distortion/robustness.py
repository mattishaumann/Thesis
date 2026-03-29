"""
H1 Experiment — Robustness Module
===================================
Checks that H1 findings are not artifacts of:
    1. Specific BERTopic parameters (sensitivity analysis)
    2. Random label assignment (permutation test)
    3. Corpus size imbalance (downsampling test)

Each check returns a structured result that can be logged
in the iteration log and visualized in notebook 03.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics import (
    compute_all_h1_metrics,
    jsd_vs_reference,
    normalized_entropy,
)


# ---------------------------------------------------------------------------
# 1. Permutation test
# ---------------------------------------------------------------------------


@dataclass
class PermutationResult:
    """Result of permuting outlet labels to test H1 significance."""

    metric: str
    observed: dict[str, float]  # outlet -> observed value
    null_distribution: dict[str, list[float]]  # outlet -> list of permuted values
    p_values: dict[str, float]  # outlet -> p-value

    def summary(self) -> pd.DataFrame:
        rows = []
        for outlet in self.observed:
            null = np.array(self.null_distribution[outlet])
            rows.append({
                "outlet_label": outlet,
                "observed": round(self.observed[outlet], 4),
                "null_mean": round(float(null.mean()), 4),
                "null_std": round(float(null.std()), 4),
                "p_value": round(self.p_values[outlet], 4),
                "significant_05": self.p_values[outlet] < 0.05,
            })
        return pd.DataFrame(rows)


def permutation_test(
    merged_articles: pd.DataFrame,
    metric: str = "jsd_vs_tagesschau",
    n_perms: int = 1000,
    reference_label: str = "Tagesschau",
    random_state: int = 42,
) -> PermutationResult:
    """Test whether observed metric values differ from random label assignment.

    Shuffles outlet labels while keeping topic assignments fixed.
    If the observed metric is more extreme than 95% of permuted values,
    the signal is unlikely to be an artifact of the modeling pipeline.
    """
    rng = np.random.RandomState(random_state)

    # Observed metrics
    observed_metrics = compute_all_h1_metrics(merged_articles, reference_label)
    observed = dict(
        zip(observed_metrics["outlet_label"], observed_metrics[metric])
    )

    outlets = sorted(observed.keys())
    null_dist = {o: [] for o in outlets}

    topic_df = merged_articles[["outlet_label", "merged_topic"]].copy()

    for _ in range(n_perms):
        # Shuffle outlet labels
        shuffled = topic_df.copy()
        shuffled["outlet_label"] = rng.permutation(shuffled["outlet_label"].values)

        # Rebuild full frame with shuffled labels
        perm_articles = merged_articles.copy()
        perm_articles["outlet_label"] = shuffled["outlet_label"]

        perm_metrics = compute_all_h1_metrics(perm_articles, reference_label)
        for _, row in perm_metrics.iterrows():
            null_dist[row["outlet_label"]].append(row[metric])

    # Compute p-values (one-tailed: observed >= permuted for JSD/entropy,
    # observed <= permuted for correlation/overlap)
    lower_is_extreme = metric in ("spearman_rho", "topk_overlap", "entropy_normalized")
    p_values = {}
    for outlet in outlets:
        null = np.array(null_dist[outlet])
        obs = observed[outlet]
        if lower_is_extreme:
            p_values[outlet] = float((null <= obs).mean())
        else:
            p_values[outlet] = float((null >= obs).mean())

    return PermutationResult(
        metric=metric,
        observed=observed,
        null_distribution=null_dist,
        p_values=p_values,
    )


# ---------------------------------------------------------------------------
# 2. Downsampling test
# ---------------------------------------------------------------------------


def downsample_test(
    merged_articles: pd.DataFrame,
    target_size: int | None = None,
    n_resamples: int = 50,
    reference_label: str = "Tagesschau",
    random_state: int = 42,
) -> pd.DataFrame:
    """Test H1 metric stability when all outlets are downsampled to equal size.

    If target_size is None, uses the size of the smallest outlet.
    This directly controls for corpus size imbalance.

    Returns:
        DataFrame with mean ± std of each metric across resamples.
    """
    rng = np.random.RandomState(random_state)

    outlet_sizes = merged_articles.groupby("outlet_label").size()
    if target_size is None:
        target_size = int(outlet_sizes.min())

    outlets = sorted(outlet_sizes.index.tolist())
    all_results = []

    for i in range(n_resamples):
        sampled_dfs = []
        for outlet in outlets:
            outlet_df = merged_articles[merged_articles["outlet_label"] == outlet]
            n = min(target_size, len(outlet_df))
            sampled = outlet_df.sample(n=n, replace=False, random_state=rng)
            sampled_dfs.append(sampled)

        sampled_articles = pd.concat(sampled_dfs, ignore_index=True)
        metrics = compute_all_h1_metrics(sampled_articles, reference_label)
        metrics["resample_id"] = i
        all_results.append(metrics)

    combined = pd.concat(all_results, ignore_index=True)

    metric_cols = [
        "entropy_normalized",
        "jsd_vs_tagesschau",
        "spearman_rho",
        "topk_overlap",
        "coverage_breadth_relative",
    ]

    summary_rows = []
    for outlet in outlets:
        outlet_data = combined[combined["outlet_label"] == outlet]
        row = {"outlet_label": outlet, "n_per_resample": target_size}
        for col in metric_cols:
            vals = outlet_data[col].dropna()
            row[f"{col}_mean"] = round(float(vals.mean()), 4)
            row[f"{col}_std"] = round(float(vals.std()), 4)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# 3. Sensitivity analysis (min_similarity threshold)
# ---------------------------------------------------------------------------


def sensitivity_analysis(
    project_root,
    similarity_values: list[float] | None = None,
    *,
    output_dir=None,
) -> pd.DataFrame:
    """Re-merge models at different min_similarity thresholds and compare metrics.

    This tests whether the choice of merge threshold materially changes
    the H1 conclusions.

    Returns:
        DataFrame with metrics for each (outlet, min_similarity) combination.
    """
    from modeling import IterationParams, load_saved_models, merge_models, build_article_frame

    if similarity_values is None:
        similarity_values = [0.5, 0.6, 0.7, 0.8]

    models = load_saved_models(project_root)
    all_results = []

    # Load documents once
    import sys
    from pathlib import Path

    module_dir = Path(project_root) / "1a_BERTopic"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from merged_outlets_analysis import load_all_prepared_documents, combine_prepared_documents

    prepared = load_all_prepared_documents(project_root)
    combined = combine_prepared_documents(prepared)

    for sim in similarity_values:
        print(f"  Testing min_similarity={sim}...")
        params = IterationParams(min_similarity=sim)
        merged_model = merge_models(models, params)
        n_topics = int(
            merged_model.get_topic_info()
            .loc[merged_model.get_topic_info()["Topic"] != -1]
            .shape[0]
        )

        merged_articles, _ = build_article_frame(merged_model, combined, params)
        metrics = compute_all_h1_metrics(merged_articles)
        metrics["min_similarity"] = sim
        metrics["n_merged_topics"] = n_topics
        all_results.append(metrics)

    return pd.concat(all_results, ignore_index=True)
