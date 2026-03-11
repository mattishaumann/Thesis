from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(slots=True)
class BERTopicConfig:
    """Shared defaults for a reusable BERTopic pipeline."""

    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    language: str = "multilingual"
    low_memory: bool = True
    calculate_probabilities: bool = True
    top_n_words: int = 10
    nr_topics: int | str | None = None

    min_text_chars: int = 50 # Minimum number of characters in a document for it to be included in the topic modeling process.
    min_tokens: int = 8 # Minimum number of tokens in a document for it to be included in the topic modeling process.
    remove_urls: bool = True
    remove_emails: bool = True
    lowercase: bool = False
    deduplicate: bool = True

    # Patterns to remove from documents before topic modeling, such as boilerplate text or common phrases 
    boilerplate_patterns: tuple[str, ...] = (
        r"(?im)^lesen sie auch.*$",
        r"(?im)^mehr .*:.*$",
        r"(?im)^auch bei .*:.*$",
        r"(?im)^quelle:.*$",
    )

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.9

    umap_n_neighbors: int = 30
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    umap_random_state: int = 42

    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int | None = 10
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_prediction_data: bool = True

    random_state: int = 42
    output_dir: Path = Path("BERTopic/outputs")
    extra_stopwords: tuple[str, ...] = field(default_factory=tuple)


def full_text_config(**overrides) -> BERTopicConfig:
    """Return explicit full-text defaults for BERTopic runs."""

    base = BERTopicConfig() # just returns the version from above
    return replace(base, **overrides) if overrides else base


def title_config(**overrides) -> BERTopicConfig:
    """Return title-focused defaults for BERTopic runs."""

    base = BERTopicConfig( # just returns the version from above
        min_text_chars=8,
        min_tokens=2,
        min_df=1,
        max_df=0.95,
        hdbscan_min_cluster_size=15,
        hdbscan_min_samples=5,
    )
    return replace(base, **overrides) if overrides else base


def make_title_config(**overrides) -> BERTopicConfig:
    """Backward-compatible alias for title_config."""

    return title_config(**overrides)
