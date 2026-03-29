"""
H1 Experiment — Modeling Module
================================
Trains per-outlet BERTopic models, merges them into a unified topic space,
and builds the article-level dataframe needed for all downstream H1 metrics.

Architecture:
    Per-outlet models (tuned to each corpus size) → BERTopic.merge_models()
    → unified topic space → per-outlet topic distributions → H1 KPIs.

Why merged models (not a single global model)?
    A global model lets large outlets (Tagesschau ~6k, RT ~4.5k) dominate
    topic discovery, drowning out niche topics from small outlets
    (Antispiegel ~565).  Per-outlet models discover each outlet's topics
    independently; the merge creates a *union* of all discovered topics.
    For H1 (agenda distortion), we need to see what each outlet talks about
    — including topics mainstream ignores entirely.

Literature:
    - Grootendorst (2022): BERTopic — neural topic modeling with c-TF-IDF
    - Jacobi et al. (2016): topic models for media content analysis
    - McCombs & Shaw (1972): agenda-setting theory

Usage:
    from modeling import run_iteration
    result = run_iteration(project_root, iteration_id="v1")
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _setup_paths(project_root: Path) -> None:
    """Ensure 1a_BERTopic is importable."""
    module_dir = project_root / "1a_BERTopic"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))


@dataclass
class IterationParams:
    """All parameters that define a single modeling iteration.

    Changing any of these should produce a new iteration entry
    in the iteration log.
    """

    # Merge
    min_similarity: float = 0.7

    # Outlier reduction (applied uniformly to merged model)
    outlier_strategy: str = "c-tf-idf"
    outlier_threshold: float = 0.10

    # UMAP for article-level 2D projection
    umap_n_neighbors: int = 10
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    # General
    random_state: int = 42
    min_articles_for_coverage: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IterationResult:
    """Everything produced by a single modeling iteration."""

    iteration_id: str
    params: IterationParams
    merged_articles: pd.DataFrame
    merged_topic_info: pd.DataFrame
    outlet_summary: dict[str, dict]  # per-outlet stats
    n_topics: int
    outlier_rates: dict[str, float]
    duration_seconds: float


def load_saved_models(
    project_root: Path,
) -> dict[str, Any]:
    """Load pre-trained per-outlet BERTopic models from local_outputs/.

    Returns:
        dict mapping outlet_key to loaded BERTopic model.
    """
    _setup_paths(project_root)
    from merged_outlets_analysis import OUTLET_SPECS, resolve_model_paths

    # Lazy import BERTopic
    from bertopic import BERTopic

    candidates = [
        project_root / "1a_BERTopic" / "local_outputs",
        project_root / "1a_BERTopic" / "outputs",
    ]
    model_paths = resolve_model_paths(candidates)

    models = {}
    for key, path in model_paths.items():
        print(f"  Loading {OUTLET_SPECS[key].label} from {path.name}...")
        models[key] = BERTopic.load(path)

    return models


def merge_models(
    models: dict[str, Any],
    params: IterationParams,
) -> Any:
    """Merge per-outlet BERTopic models into a unified topic space.

    Uses BERTopic.merge_models() with c-TF-IDF similarity matching.
    The min_similarity threshold controls how aggressively topics are
    merged — lower values create fewer unified topics.

    Args:
        models: dict mapping outlet_key to BERTopic model.
        params: iteration parameters (uses min_similarity).

    Returns:
        Merged BERTopic model.
    """
    from bertopic import BERTopic

    model_list = list(models.values())
    base_model = model_list[0]
    other_models = model_list[1:]

    merged = BERTopic.merge_models(
        [base_model] + other_models,
        min_similarity=params.min_similarity,
    )
    return merged


def reduce_outliers(
    merged_model: Any,
    docs: list[str],
    topics: list[int],
    params: IterationParams,
    embeddings: np.ndarray | None = None,
) -> list[int]:
    """Apply uniform outlier reduction to merged model assignments.

    Using the same strategy + threshold for all outlets ensures
    topic distributions are comparable post-reduction.

    Falls back from c-tf-idf to embeddings strategy if the merged
    model's vectorizer is not fitted (common after merge_models).

    Args:
        merged_model: merged BERTopic model.
        docs: list of document strings.
        topics: original topic assignments.
        params: iteration parameters.
        embeddings: pre-computed document embeddings (needed for embeddings strategy).

    Returns:
        Updated topic assignments with fewer outliers.
    """
    strategy = params.outlier_strategy
    threshold = params.outlier_threshold

    try:
        new_topics = merged_model.reduce_outliers(
            docs,
            topics,
            strategy=strategy,
            threshold=threshold,
        )
        return new_topics
    except Exception as e:
        if "Vocabulary not fitted" in str(e) and strategy == "c-tf-idf":
            print(f"  ⚠ c-tf-idf outlier reduction failed (merged model vectorizer not fitted).")
            print(f"    Falling back to 'embeddings' strategy with threshold={threshold}.")
            # Embeddings strategy uses cosine similarity to nearest topic centroid
            new_topics = merged_model.reduce_outliers(
                docs,
                topics,
                strategy="embeddings",
                threshold=threshold,
                embeddings=embeddings,
            )
            return new_topics
        raise


def build_article_frame(
    merged_model: Any,
    combined_prepared: pd.DataFrame,
    params: IterationParams,
    *,
    apply_outlier_reduction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign all documents to merged topics and build analysis-ready frame.

    Args:
        merged_model: merged BERTopic model.
        combined_prepared: concatenated prepared documents from all outlets.
        params: iteration parameters.
        apply_outlier_reduction: whether to apply outlier reduction.

    Returns:
        (merged_articles, merged_topic_info) tuple.
    """
    from umap import UMAP
    from merged_outlets_analysis import enrich_topic_info_with_display

    docs = combined_prepared["document"].tolist()

    # Transform all docs through the merged model
    topics, probabilities = merged_model.transform(docs)

    # Extract embeddings (needed for UMAP and optionally for outlier reduction)
    embeddings = merged_model._extract_embeddings(docs, method="document")

    # Optionally reduce outliers
    if apply_outlier_reduction:
        topics = reduce_outliers(
            merged_model, docs, topics, params, embeddings=embeddings
        )
    reducer = UMAP(
        n_neighbors=params.umap_n_neighbors,
        n_components=2,
        min_dist=params.umap_min_dist,
        metric=params.umap_metric,
        random_state=params.random_state,
    )
    coords = reducer.fit_transform(embeddings)

    # Build enriched topic info
    merged_topic_info = enrich_topic_info_with_display(merged_model.get_topic_info())
    display_label_map = dict(
        zip(merged_topic_info["Topic"], merged_topic_info["DisplayLabel"])
    )
    display_topic_map = dict(
        zip(merged_topic_info["Topic"], merged_topic_info["DisplayTopic"])
    )

    # Assemble the article frame
    frame = combined_prepared.copy()
    frame["merged_topic"] = list(topics)
    frame["merged_probability"] = (
        list(probabilities) if probabilities is not None else None
    )
    frame["merged_display_topic"] = (
        frame["merged_topic"].map(display_topic_map).astype("Int64")
    )
    frame["merged_display_label"] = (
        frame["merged_topic"].map(display_label_map).fillna("Outliers")
    )
    frame["umap_x"] = coords[:, 0]
    frame["umap_y"] = coords[:, 1]

    return frame, merged_topic_info


def _compute_outlet_summary(
    merged_articles: pd.DataFrame,
) -> dict[str, dict]:
    """Per-outlet stats for the iteration log."""
    summary = {}
    for label in sorted(merged_articles["outlet_label"].unique()):
        outlet_df = merged_articles[merged_articles["outlet_label"] == label]
        n = len(outlet_df)
        n_outlier = int((outlet_df["merged_topic"] == -1).sum())
        n_topics = int(
            outlet_df.loc[outlet_df["merged_topic"] != -1, "merged_topic"].nunique()
        )
        summary[label] = {
            "n_articles": n,
            "n_outliers": n_outlier,
            "outlier_rate": round(n_outlier / n, 3) if n > 0 else 0.0,
            "n_topics_covered": n_topics,
        }
    return summary


def run_iteration(
    project_root: Path,
    iteration_id: str = "v1",
    params: IterationParams | None = None,
    *,
    models: dict[str, Any] | None = None,
) -> IterationResult:
    """Execute a full modeling iteration.

    Steps:
        1. Load pre-trained per-outlet models (or use provided ones)
        2. Merge into unified topic space
        3. Load & prepare all outlet documents
        4. Assign docs to merged topics (with outlier reduction)
        5. Compute per-outlet summary stats

    Args:
        project_root: path to thesis repo root.
        iteration_id: label for this iteration (e.g. "v1", "v2_lower_sim").
        params: modeling parameters. Defaults to IterationParams().
        models: pre-loaded models dict (skip loading if provided).

    Returns:
        IterationResult with all data needed for metrics + visualization.
    """
    _setup_paths(project_root)
    params = params or IterationParams()
    t0 = time.time()

    # Step 1: Load models
    if models is None:
        print(f"[{iteration_id}] Loading per-outlet models...")
        models = load_saved_models(project_root)
    else:
        print(f"[{iteration_id}] Using provided models.")

    # Step 2: Merge
    print(f"[{iteration_id}] Merging models (min_similarity={params.min_similarity})...")
    merged_model = merge_models(models, params)
    n_topics = int(
        merged_model.get_topic_info()
        .loc[merged_model.get_topic_info()["Topic"] != -1]
        .shape[0]
    )
    print(f"  → {n_topics} merged topics discovered")

    # Step 3: Load prepared documents
    print(f"[{iteration_id}] Preparing documents...")
    from merged_outlets_analysis import (
        load_all_prepared_documents,
        combine_prepared_documents,
    )

    prepared_by_outlet = load_all_prepared_documents(project_root)
    combined = combine_prepared_documents(prepared_by_outlet)
    print(f"  → {len(combined):,} total documents")

    # Step 4: Build article frame
    print(f"[{iteration_id}] Assigning documents to merged topics...")
    merged_articles, merged_topic_info = build_article_frame(
        merged_model, combined, params
    )

    # Step 5: Summary
    outlet_summary = _compute_outlet_summary(merged_articles)
    outlier_rates = {
        label: stats["outlier_rate"] for label, stats in outlet_summary.items()
    }

    duration = time.time() - t0
    print(f"[{iteration_id}] Done in {duration:.0f}s")

    return IterationResult(
        iteration_id=iteration_id,
        params=params,
        merged_articles=merged_articles,
        merged_topic_info=merged_topic_info,
        outlet_summary=outlet_summary,
        n_topics=n_topics,
        outlier_rates=outlier_rates,
        duration_seconds=round(duration, 1),
    )


def run_global_model(
    project_root: Path,
    iteration_id: str = "global_v1",
    params: IterationParams | None = None,
) -> IterationResult:
    """Train a single global BERTopic on all documents (no per-outlet merging).

    This serves as a parallel validation of the merged-model approach.
    If JSD rankings agree between global and merged models, the merged
    approach is validated. If they diverge, we need to investigate why.

    Literature: Muller & Freudenthaler (2022), Jacobi et al. (2016) use
    single global models as the standard approach.
    """
    _setup_paths(project_root)
    params = params or IterationParams()
    t0 = time.time()

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP as UMAPModel
    from hdbscan import HDBSCAN

    from merged_outlets_analysis import (
        load_all_prepared_documents,
        combine_prepared_documents,
        enrich_topic_info_with_display,
        EMBEDDING_MODEL_NAME,
    )

    # Load all documents
    print(f"[{iteration_id}] Loading documents...")
    prepared = load_all_prepared_documents(project_root)
    combined = combine_prepared_documents(prepared)
    docs = combined["document"].tolist()
    print(f"  -> {len(docs):,} documents")

    # Embed all documents
    print(f"[{iteration_id}] Embedding documents...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64)

    # Configure BERTopic components for global model
    umap_model = UMAPModel(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=params.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=25,
        min_samples=5,
        metric="euclidean",
        prediction_data=True,
    )

    # Fit global model
    print(f"[{iteration_id}] Fitting global BERTopic...")
    global_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )
    topics, probs = global_model.fit_transform(docs, embeddings=embeddings)

    n_topics = int(
        global_model.get_topic_info()
        .loc[global_model.get_topic_info()["Topic"] != -1]
        .shape[0]
    )
    print(f"  -> {n_topics} topics discovered")

    # 2D UMAP for visualization
    print(f"[{iteration_id}] Computing 2D UMAP...")
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=params.umap_n_neighbors,
        n_components=2,
        min_dist=params.umap_min_dist,
        metric=params.umap_metric,
        random_state=params.random_state,
    )
    coords = reducer.fit_transform(embeddings)

    # Build topic info
    topic_info = enrich_topic_info_with_display(global_model.get_topic_info())
    display_label_map = dict(zip(topic_info["Topic"], topic_info["DisplayLabel"]))
    display_topic_map = dict(zip(topic_info["Topic"], topic_info["DisplayTopic"]))

    # Assemble article frame
    frame = combined.copy()
    frame["merged_topic"] = list(topics)
    frame["merged_probability"] = list(probs) if probs is not None else None
    frame["merged_display_topic"] = frame["merged_topic"].map(display_topic_map).astype("Int64")
    frame["merged_display_label"] = frame["merged_topic"].map(display_label_map).fillna("Outliers")
    frame["umap_x"] = coords[:, 0]
    frame["umap_y"] = coords[:, 1]

    outlet_summary = _compute_outlet_summary(frame)
    outlier_rates = {
        label: stats["outlier_rate"] for label, stats in outlet_summary.items()
    }

    duration = time.time() - t0
    print(f"[{iteration_id}] Done in {duration:.0f}s")

    return IterationResult(
        iteration_id=iteration_id,
        params=params,
        merged_articles=frame,
        merged_topic_info=topic_info,
        outlet_summary=outlet_summary,
        n_topics=n_topics,
        outlier_rates=outlier_rates,
        duration_seconds=round(duration, 1),
    )


def save_iteration(result: IterationResult, output_dir: Path) -> None:
    """Persist iteration outputs to disk."""
    iter_dir = output_dir / result.iteration_id
    iter_dir.mkdir(parents=True, exist_ok=True)

    result.merged_articles.to_csv(iter_dir / "merged_articles.csv", index=False)
    result.merged_topic_info.to_csv(iter_dir / "merged_topic_info.csv", index=False)

    meta = {
        "iteration_id": result.iteration_id,
        "params": result.params.to_dict(),
        "n_topics": result.n_topics,
        "outlier_rates": result.outlier_rates,
        "outlet_summary": result.outlet_summary,
        "duration_seconds": result.duration_seconds,
    }
    with open(iter_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved iteration '{result.iteration_id}' to {iter_dir}")


def load_iteration(output_dir: Path, iteration_id: str) -> IterationResult:
    """Load a previously saved iteration from disk."""
    iter_dir = output_dir / iteration_id

    with open(iter_dir / "meta.json") as f:
        meta = json.load(f)

    merged_articles = pd.read_csv(iter_dir / "merged_articles.csv")
    merged_topic_info = pd.read_csv(iter_dir / "merged_topic_info.csv")

    return IterationResult(
        iteration_id=meta["iteration_id"],
        params=IterationParams(**meta["params"]),
        merged_articles=merged_articles,
        merged_topic_info=merged_topic_info,
        outlet_summary=meta["outlet_summary"],
        n_topics=meta["n_topics"],
        outlier_rates=meta["outlier_rates"],
        duration_seconds=meta["duration_seconds"],
    )


def _first_non_empty(series: pd.Series) -> object:
    """Return the first non-null, non-empty value from a series."""

    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return np.nan


def build_topic_info_from_article_frame(
    merged_articles: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct topic-level metadata from the saved article frame.

    This is a lightweight fallback for experiment notebooks when the
    dedicated `merged_topic_info.csv` file is missing but the article-level
    export from `1a_BERTopic/local_outputs/` is available.
    """

    frame = merged_articles.copy()
    frame["merged_topic"] = pd.to_numeric(frame["merged_topic"], errors="coerce").astype("Int64")
    if "merged_display_topic" in frame.columns:
        frame["merged_display_topic"] = pd.to_numeric(
            frame["merged_display_topic"], errors="coerce"
        ).astype("Int64")

    topic_info = (
        frame.groupby("merged_topic", dropna=False)
        .agg(
            Count=("merged_topic", "size"),
            DisplayTopic=("merged_display_topic", _first_non_empty),
            KeywordLabel=("topic_keyword_label", _first_non_empty),
            TopicLabel=("topic_label", _first_non_empty),
            DisplayLabel=("topic_display_label", _first_non_empty),
            ManualTopicName=("manual_topic_name", _first_non_empty),
            MergedDisplayLabel=("merged_display_label", _first_non_empty),
        )
        .reset_index()
        .rename(columns={"merged_topic": "Topic"})
    )

    topic_info["DisplayTopic"] = pd.to_numeric(
        topic_info["DisplayTopic"], errors="coerce"
    ).astype("Int64")
    topic_info["KeywordLabel"] = topic_info["KeywordLabel"].fillna(topic_info["TopicLabel"])
    topic_info["DisplayLabel"] = topic_info["DisplayLabel"].fillna(
        topic_info["MergedDisplayLabel"]
    )
    topic_info["TopicLabel"] = topic_info["TopicLabel"].fillna(topic_info["DisplayLabel"])
    topic_info["Name"] = topic_info["KeywordLabel"].fillna(topic_info["DisplayLabel"])
    topic_info["Representation"] = topic_info["KeywordLabel"].fillna("").astype(str).str.split("_")

    ordered_columns = [
        "Topic",
        "DisplayTopic",
        "Count",
        "Name",
        "Representation",
        "KeywordLabel",
        "TopicLabel",
        "DisplayLabel",
        "ManualTopicName",
    ]
    topic_info = topic_info.loc[:, ordered_columns]
    topic_info = topic_info.sort_values(
        ["Topic", "DisplayTopic"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    return topic_info


def ensure_iteration_outputs(
    project_root: Path,
    output_dir: Path,
    *,
    preferred_iteration_id: str = "v2",
    fallback_iteration_id: str = "v1",
) -> tuple[str, pd.DataFrame, pd.DataFrame, Path]:
    """Return experiment CSVs, materializing `preferred_iteration_id` when needed.

    Resolution order:
    1. Existing `outputs/<preferred>/merged_articles.csv`
    2. Rebuild `outputs/<preferred>/` from
       `1a_BERTopic/local_outputs/merged_articles_with_topics.csv`
    3. Existing `outputs/<fallback>/merged_articles.csv`
    """

    preferred_dir = output_dir / preferred_iteration_id
    fallback_dir = output_dir / fallback_iteration_id

    preferred_articles = preferred_dir / "merged_articles.csv"
    preferred_topics = preferred_dir / "merged_topic_info.csv"
    fallback_articles = fallback_dir / "merged_articles.csv"
    fallback_topics = fallback_dir / "merged_topic_info.csv"

    if preferred_articles.exists() and preferred_topics.exists():
        return (
            preferred_iteration_id,
            pd.read_csv(preferred_articles, low_memory=False),
            pd.read_csv(preferred_topics, low_memory=False),
            preferred_dir,
        )

    _setup_paths(project_root)
    from merged_outlets_analysis import load_exported_article_topic_dataset

    local_articles = load_exported_article_topic_dataset(project_root)
    if local_articles is None:
        if fallback_articles.exists() and fallback_topics.exists():
            return (
                fallback_iteration_id,
                pd.read_csv(fallback_articles, low_memory=False),
                pd.read_csv(fallback_topics, low_memory=False),
                fallback_dir,
            )
        raise FileNotFoundError(
            "Could not find experiment outputs (v2/v1) or the local merged article export."
        )

    topic_info = build_topic_info_from_article_frame(local_articles)
    preferred_dir.mkdir(parents=True, exist_ok=True)
    local_articles.to_csv(preferred_articles, index=False)
    topic_info.to_csv(preferred_topics, index=False)

    outlet_summary = _compute_outlet_summary(local_articles)
    meta = {
        "iteration_id": preferred_iteration_id,
        "params": IterationParams(
            min_similarity=0.7,
            outlier_strategy="none",
            outlier_threshold=0.0,
        ).to_dict(),
        "n_topics": int(topic_info.loc[topic_info["Topic"] != -1, "Topic"].nunique()),
        "outlier_rates": {
            label: stats["outlier_rate"] for label, stats in outlet_summary.items()
        },
        "outlet_summary": outlet_summary,
        "duration_seconds": 0.0,
        "source_note": (
            "Reconstructed from 1a_BERTopic/local_outputs/merged_articles_with_topics.csv"
        ),
    }
    with open(preferred_dir / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f"Materialized '{preferred_iteration_id}' in {preferred_dir}")
    return preferred_iteration_id, local_articles, topic_info, preferred_dir
