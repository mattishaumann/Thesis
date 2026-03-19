from __future__ import annotations

import importlib
import os
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import numpy as np
import pandas as pd

try:
    from .stopwords_de import get_german_stopwords
except ImportError:
    from stopwords_de import get_german_stopwords


WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


@dataclass(slots=True)
class BERTopicV2Config:
    """Minimal settings for BERTopic with spaCy-based topic representation."""

    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    spacy_model_name: str = "de_core_news_md"
    language: str = "multilingual"
    top_n_words: int = 10
    nr_topics: int | str | None = None
    calculate_probabilities: bool = False
    low_memory: bool = True
    min_tokens: int = 8
    deduplicate: bool = True
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 8
    max_df: int | float = 0.85
    extra_stopwords: tuple[str, ...] = ()
    umap_n_neighbors: int = 30
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 40
    hdbscan_min_samples: int | None = 10
    random_state: int = 42


@dataclass(slots=True)
class BERTopicV2ConfigTitle(BERTopicV2Config):
    """Title-focused defaults for a second BERTopic run on headlines."""

    min_tokens: int = 2
    deduplicate: bool = True
    min_df: int | float = 1
    max_df: int | float = 0.98
    umap_n_neighbors: int = 10
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = 3


def _require_dependency(package_name: str, install_hint: str):
    """Import a dependency and raise a clear install message if missing."""

    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name == package_name:
            raise ImportError(
                f"Missing optional dependency '{package_name}'. Install it with '{install_hint}'."
            ) from exc
        raise ImportError(
            f"Failed to import optional dependency '{package_name}' because a nested dependency "
            f"is missing: '{exc.name}'. Original error: {exc}"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            f"Failed to import optional dependency '{package_name}'. "
            f"The package is installed, but one of its imports failed: {exc}"
        ) from exc


@contextmanager
def _suppress_numba_cache_during_import():
    """Disable numba caching hooks to avoid broken packaged UMAP imports."""

    try:
        dispatcher_module = importlib.import_module("numba.core.dispatcher")
        ufuncbuilder_module = importlib.import_module("numba.np.ufunc.ufuncbuilder")
    except ImportError:
        yield
        return

    dispatcher_cls = dispatcher_module.Dispatcher
    ufunc_dispatcher_cls = ufuncbuilder_module.UFuncDispatcher

    original_dispatcher_enable = dispatcher_cls.enable_caching
    original_ufunc_enable = ufunc_dispatcher_cls.enable_caching
    original_mplconfigdir = os.environ.get("MPLCONFIGDIR")

    dispatcher_cls.enable_caching = lambda self: None
    ufunc_dispatcher_cls.enable_caching = lambda self: None
    os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "mplconfig"))

    try:
        yield
    finally:
        dispatcher_cls.enable_caching = original_dispatcher_enable
        ufunc_dispatcher_cls.enable_caching = original_ufunc_enable
        if original_mplconfigdir is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = original_mplconfigdir


def clean_text(text: Any) -> str:
    """Normalize whitespace and coerce missing values to empty text."""

    if pd.isna(text):
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def _token_count(text: str) -> int:
    """Count word-like tokens in a text."""

    return len(TOKEN_PATTERN.findall(text))


def prepare_documents(
    df: pd.DataFrame,
    text_col: str,
    *,
    config: BERTopicV2Config | None = None,
    id_col: str | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    """Build a minimal BERTopic-ready table with cleaned documents."""

    config = config or BERTopicV2Config()
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in dataframe.")
    if id_col and id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in dataframe.")

    prepared = df.copy().reset_index().rename(columns={"index": "original_index"})
    prepared["document_id"] = (
        prepared[id_col].astype(str) if id_col else prepared["original_index"].astype(str)
    )
    prepared["source_name"] = source_name if source_name is not None else None
    prepared["document"] = prepared[text_col].map(clean_text)
    prepared["token_count"] = prepared["document"].map(_token_count)

    prepared = prepared.loc[(prepared["document"] != "") & (prepared["token_count"] >= config.min_tokens)]
    if config.deduplicate:
        prepared = prepared.drop_duplicates(subset=["document"])

    return prepared.reset_index(drop=True)


def build_vectorizer(config: BERTopicV2Config | None = None, *, stop_words: list[str] | None = None):
    """Create a simple CountVectorizer for c-TF-IDF topic words."""

    config = config or BERTopicV2Config()
    sklearn_text = _require_dependency("sklearn.feature_extraction.text", "pip install scikit-learn")
    CountVectorizer = sklearn_text.CountVectorizer

    effective_stop_words = stop_words if stop_words is not None else get_german_stopwords(config.extra_stopwords)
    return CountVectorizer(
        stop_words=effective_stop_words,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        token_pattern=r"(?u)\b\w\w+\b",
    )


def build_embedding_model(config: BERTopicV2Config | None = None):
    """Load sentence-transformers embeddings."""

    config = config or BERTopicV2Config()
    sentence_transformers = _require_dependency(
        "sentence_transformers",
        "pip install sentence-transformers",
    )
    SentenceTransformer = sentence_transformers.SentenceTransformer
    return SentenceTransformer(config.embedding_model_name)


def build_umap_model(config: BERTopicV2Config | None = None):
    """Create a small UMAP model for BERTopic dimensionality reduction."""

    config = config or BERTopicV2Config()
    with _suppress_numba_cache_during_import():
        umap_module = _require_dependency("umap", "pip install umap-learn")
    UMAP = umap_module.UMAP

    return UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric="cosine",
        random_state=config.random_state,
    )


def build_hdbscan_model(config: BERTopicV2Config | None = None):
    """Create a small HDBSCAN model for BERTopic clustering."""

    config = config or BERTopicV2Config()
    hdbscan_module = _require_dependency("hdbscan", "pip install hdbscan")
    HDBSCAN = hdbscan_module.HDBSCAN

    return HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )


def build_representation_model(config: BERTopicV2Config | None = None):
    """Create spaCy POS-based representation model for cleaner topic labels."""

    config = config or BERTopicV2Config()
    with _suppress_numba_cache_during_import():
        representation_module = _require_dependency("bertopic.representation", "pip install bertopic")
    PartOfSpeech = representation_module.PartOfSpeech
    return PartOfSpeech(config.spacy_model_name)


def build_topic_model(
    config: BERTopicV2Config | None = None,
    *,
    stop_words: list[str] | None = None,
    embedding_model: Any | None = None,
):
    """Build BERTopic with multilingual embeddings and spaCy topic representation."""

    config = config or BERTopicV2Config()
    with _suppress_numba_cache_during_import():
        bertopic_module = _require_dependency("bertopic", "pip install bertopic")
    BERTopic = bertopic_module.BERTopic

    return BERTopic(
        embedding_model=embedding_model or build_embedding_model(config),
        vectorizer_model=build_vectorizer(config, stop_words=stop_words),
        umap_model=build_umap_model(config),
        hdbscan_model=build_hdbscan_model(config),
        representation_model=build_representation_model(config),
        language=config.language,
        top_n_words=config.top_n_words,
        nr_topics=config.nr_topics,
        calculate_probabilities=config.calculate_probabilities,
        low_memory=config.low_memory,
        verbose=True,
    )


def run_bertopic_pipeline(
    df: pd.DataFrame,
    text_col: str,
    *,
    config: BERTopicV2Config | None = None,
    id_col: str | None = None,
    source_name: str | None = None,
    stop_words: list[str] | None = None,
    embedding_model: Any | None = None,
) -> dict[str, Any]:
    """Prepare docs, fit BERTopic, and return reusable result tables."""

    config = config or BERTopicV2Config()
    prepared = prepare_documents(
        df,
        text_col,
        config=config,
        id_col=id_col,
        source_name=source_name,
    )
    if prepared.empty:
        raise ValueError("No documents left after preparation. Lower min_tokens or check input text.")

    docs = prepared["document"].tolist()

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    try:
        import torch

        torch.manual_seed(config.random_state)
    except ImportError:
        pass

    topic_model = build_topic_model(
        config,
        stop_words=stop_words,
        embedding_model=embedding_model,
    )
    topics, probs = topic_model.fit_transform(docs)

    topic_info = topic_model.get_topic_info()
    doc_info = topic_model.get_document_info(docs)
    doc_info = pd.concat(
        [
            prepared[["original_index", "document_id", "source_name"]].reset_index(drop=True),
            doc_info.reset_index(drop=True),
        ],
        axis=1,
    )

    return {
        "config": config,
        "prepared_documents": prepared,
        "docs": docs,
        "topic_model": topic_model,
        "topics": topics,
        "probabilities": probs,
        "topic_info": topic_info,
        "doc_info": doc_info,
    }
