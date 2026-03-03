from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BERTopicConfig:
    """Shared defaults for a reusable BERTopic pipeline."""

    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    language: str = "multilingual"
    low_memory: bool = True
    calculate_probabilities: bool = False
    top_n_words: int = 10
    nr_topics: int | str | None = None

    min_text_chars: int = 50
    min_tokens: int = 8
    remove_urls: bool = True
    remove_emails: bool = True
    lowercase: bool = False
    deduplicate: bool = True

    boilerplate_patterns: tuple[str, ...] = (
        r"(?im)^lesen sie auch.*$",
        r"(?im)^mehr .*:.*$",
        r"(?im)^auch bei .*:.*$",
        r"(?im)^quelle:.*$",
    )

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 8
    max_df: int | float = 0.85

    umap_n_neighbors: int = 30
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    hdbscan_min_cluster_size: int = 40
    hdbscan_min_samples: int | None = 10
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_prediction_data: bool = True

    random_state: int = 42
    output_dir: Path = Path("BERTopic/outputs")
    extra_stopwords: tuple[str, ...] = field(default_factory=tuple)
