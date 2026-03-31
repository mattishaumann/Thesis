from __future__ import annotations
import html
import importlib
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir
from typing import Any
import random
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

try:
    from .bertopic_config import BERTopicConfig
    from .stopwords_de import get_german_stopwords
except ImportError:
    from bertopic_config import BERTopicConfig
    from stopwords_de import get_german_stopwords


# URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
# EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
# HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
# Cleaning patterns for German news articles, tuned to remove common boilerplate while preserving semantic content.
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
MAX_SINGLE_LINE_BOILERPLATE_CHARS = 80
    

def _require_dependency(package_name: str, install_hint: str):
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{package_name}'. Install it with '{install_hint}'."
        ) from exc


@contextmanager
def _suppress_spacy_during_bertopic_import():
    """Prevent BERTopic from importing the optional spaCy-based POS module."""

    sentinel = object()
    previous = sys.modules.get("spacy", sentinel)
    sys.modules["spacy"] = None
    try:
        yield
    finally:
        if previous is sentinel:
            sys.modules.pop("spacy", None)
        else:
            sys.modules["spacy"] = previous


@contextmanager
def _suppress_numba_cache_during_import():
    """Disable numba's import-time caching hooks for broken packaged installs."""

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


def clean_text(text: Any, config: BERTopicConfig | None = None) -> str:
    """Apply light preprocessing that keeps semantic context for embeddings."""

    config = config or BERTopicConfig()
    if pd.isna(text):
        return ""

    cleaned = html.unescape(str(text))
    cleaned = cleaned.replace("\xa0", " ")

    for pattern in config.boilerplate_patterns:
        compiled_pattern = re.compile(pattern)

        # Guard against boilerplate regexes wiping whole single-line articles.
        # Some corpus texts are already flattened, so broad line-anchored patterns
        # like "^Mehr ...:.*$" can otherwise remove substantive articles that
        # happen to start with words such as "Mehr" or "Auch bei".
        def _replace_boilerplate(match: re.Match[str]) -> str:
            matched_text = match.group(0)
            normalized_match = WHITESPACE_PATTERN.sub(" ", matched_text).strip()
            if "\n" not in matched_text and len(normalized_match) > MAX_SINGLE_LINE_BOILERPLATE_CHARS:
                return matched_text
            return " "

        cleaned = compiled_pattern.sub(_replace_boilerplate, cleaned)

    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    if config.lowercase:
        cleaned = cleaned.lower()
    return cleaned


def _token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _build_preparation_tables(
    df: pd.DataFrame,
    text_col: str,
    config: BERTopicConfig | None = None,
    *,
    id_col: str | None = None,
    source_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the prepared BERTopic table and a row-level preprocessing audit."""

    config = config or BERTopicConfig()
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in dataframe.")
    if id_col and id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in dataframe.")

    prepared = df.copy()
    prepared = prepared.reset_index().rename(columns={"index": "original_index"})
    prepared["document_id"] = (
        prepared[id_col].astype(str) if id_col else prepared["original_index"].astype(str)
    )
    prepared["source_name"] = source_name if source_name is not None else None
    prepared["document"] = prepared[text_col].map(lambda value: clean_text(value, config))
    prepared["document_length"] = prepared["document"].str.len()
    prepared["token_count"] = prepared["document"].map(_token_count)

    audit = prepared.copy()
    audit["passes_min_text_chars"] = audit["document_length"] >= config.min_text_chars
    audit["passes_min_tokens"] = audit["token_count"] >= config.min_tokens
    audit["passes_min_filters"] = audit["passes_min_text_chars"] & audit["passes_min_tokens"]
    audit["is_duplicate_document"] = False

    if config.deduplicate:
        duplicate_mask = audit.loc[audit["passes_min_filters"], "document"].duplicated(keep="first")
        audit.loc[audit["passes_min_filters"], "is_duplicate_document"] = duplicate_mask.to_numpy()

    audit["included_in_model"] = audit["passes_min_filters"] & ~audit["is_duplicate_document"]

    def _exclusion_reason(row: pd.Series) -> str | None:
        reasons: list[str] = []
        if not row["passes_min_text_chars"]:
            reasons.append("short_text")
        if not row["passes_min_tokens"]:
            reasons.append("too_few_tokens")
        if row["is_duplicate_document"]:
            reasons.append("duplicate_clean_document")
        return "|".join(reasons) if reasons else None

    audit["exclusion_reason"] = audit.apply(_exclusion_reason, axis=1)
    prepared = audit.loc[audit["included_in_model"]].copy()
    prepared = prepared.reset_index(drop=True)
    audit = audit.reset_index(drop=True)

    prepared = prepared.drop(
        columns=[
            "passes_min_text_chars",
            "passes_min_tokens",
            "passes_min_filters",
            "is_duplicate_document",
            "included_in_model",
            "exclusion_reason",
        ],
        errors="ignore",
    )
    return prepared, audit


def prepare_documents(
    df: pd.DataFrame,
    text_col: str,
    config: BERTopicConfig | None = None,
    *,
    id_col: str | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    """Return a cleaned document table ready for BERTopic."""

    prepared, _ = _build_preparation_tables(
        df,
        text_col,
        config,
        id_col=id_col,
        source_name=source_name,
    )
    return prepared


def prepare_documents_with_audit(
    df: pd.DataFrame,
    text_col: str,
    config: BERTopicConfig | None = None,
    *,
    id_col: str | None = None,
    source_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned document table plus a row-level preprocessing audit."""

    return _build_preparation_tables(
        df,
        text_col,
        config,
        id_col=id_col,
        source_name=source_name,
    )


def build_vectorizer(
    config: BERTopicConfig | None = None,
    *,
    stop_words: list[str] | None = None,
):
    """Build the c-TF-IDF vectorizer used for topic representations."""

    config = config or BERTopicConfig()
    sklearn_text = _require_dependency(
        "sklearn.feature_extraction.text",
        "pip install scikit-learn",
    )
    CountVectorizer = sklearn_text.CountVectorizer

    effective_stop_words = stop_words
    if effective_stop_words is None:
        effective_stop_words = get_german_stopwords(extra=config.extra_stopwords)

    return CountVectorizer(
        stop_words=effective_stop_words,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        token_pattern=r"(?u)\b\w\w+\b",
    )


def build_umap_model(config: BERTopicConfig | None = None):
    """Build the default dimensionality reduction stage."""

    config = config or BERTopicConfig()
    with _suppress_numba_cache_during_import():
        umap_module = _require_dependency("umap", "pip install umap-learn")
    UMAP = umap_module.UMAP

    return UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.umap_random_state,
    )


def build_hdbscan_model(config: BERTopicConfig | None = None):
    """Build the default clustering stage."""

    config = config or BERTopicConfig()
    hdbscan_module = _require_dependency("hdbscan", "pip install hdbscan")
    HDBSCAN = hdbscan_module.HDBSCAN

    return HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric=config.hdbscan_metric,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        prediction_data=config.hdbscan_prediction_data,
    )


def build_embedding_model(config: BERTopicConfig | None = None):
    """Create the sentence-transformers embedding model used by BERTopic."""

    config = config or BERTopicConfig()
    sentence_transformers = _require_dependency(
        "sentence_transformers",
        "pip install sentence-transformers",
    )
    SentenceTransformer = sentence_transformers.SentenceTransformer
    return SentenceTransformer(config.embedding_model_name)


def build_topic_model(
    config: BERTopicConfig | None = None,
    *,
    stop_words: list[str] | None = None,
    embedding_model: Any | None = None,
):
    """Create a standard BERTopic model with multilingual embeddings."""

    config = config or BERTopicConfig()
    with _suppress_numba_cache_during_import(), _suppress_spacy_during_bertopic_import():
        bertopic_module = _require_dependency("bertopic", "pip install bertopic")
    BERTopic = bertopic_module.BERTopic

    effective_embedding_model = embedding_model or build_embedding_model(config)
    vectorizer_model = build_vectorizer(config, stop_words=stop_words)
    umap_model = build_umap_model(config)
    hdbscan_model = build_hdbscan_model(config)

    return BERTopic(
        embedding_model=effective_embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
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
    config: BERTopicConfig | None = None,
    *,
    id_col: str | None = None,
    source_name: str | None = None,
    stop_words: list[str] | None = None,
    embedding_model: Any | None = None,
) -> dict[str, Any]:
    """Clean documents, fit BERTopic, and return reusable result tables."""

    config = config or BERTopicConfig()
    prepared, preparation_audit = prepare_documents_with_audit(
        df,
        text_col,
        config,
        id_col=id_col,
        source_name=source_name,
    )
    if prepared.empty:
        raise ValueError("No documents left after preprocessing. Relax the filters in BERTopicConfig.")

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
    context_columns = [
        column
        for column in ("original_index", "document_id", "source_name", "row_id", "source", "Date", "Title")
        if column in prepared.columns
    ]
    doc_info = pd.concat(
        [
            prepared[context_columns].reset_index(drop=True),
            doc_info.reset_index(drop=True),
        ],
        axis=1,
    )

    return {
        "config": config,
        "prepared_documents": prepared,
        "preparation_audit": preparation_audit,
        "docs": docs,
        "topic_model": topic_model,
        "topics": topics,
        "probabilities": probs,
        "topic_info": topic_info,
        "doc_info": doc_info,
    }
