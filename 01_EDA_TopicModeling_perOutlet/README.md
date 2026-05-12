# 01 — EDA and Per-Outlet Topic Modeling

This folder is where the **per-outlet exploratory data analysis**, the **outlet-specific cleaning**, and the **individual BERTopic runs** (one model per outlet) happen. The merge across outlets and all downstream analysis live in [`../02_TopicModeling/`](../02_TopicModeling/).

## Why per-outlet, not one pooled model

Tagesschau and RT DE alone make up about 53% of the corpus, so a single pooled BERTopic fit would be dominated by those two outlets. Each outlet is therefore fit separately, with its own UMAP neighborhood, HDBSCAN `min_cluster_size`, and `min_df`, before merging.

## Shared pipeline

The three-stage BERTopic pipeline used for every outlet lives in `BERTopic_configuration/`:

1. **Embeddings** — SBERT `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual).
2. **Dimensionality reduction** — UMAP, 5 components, cosine metric, `random_state=42`.
3. **Clustering** — HDBSCAN, Euclidean metric, EOM cluster selection, prediction data on.

Topic representation uses class-based TF-IDF (c-TF-IDF) with unigrams + bigrams, a curated German stopword list, and capitalization retained. Outlier reduction is done via c-TF-IDF reassignment at threshold 0.10 (0.05 for Tichys Einblick).

## Subfolders

- `BERTopic_configuration/` — the shared pipeline + stopword list + per-outlet configuration.
- `EDA_BERTopic_per_model/` — per-outlet EDA notebooks (one per outlet) with cleaning, descriptive statistics, and the BERTopic run itself.
- `Outputs_individual_models/` — saved per-outlet BERTopic models and topic tables that feed the merge step in [`../02_TopicModeling/`](../02_TopicModeling/).
