# Merged_BERTopic_All_Outlets.ipynb Explained

This note explains the current merged-topic workflow in [Merged_BERTopic_All_Outlets.ipynb](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/Merged_BERTopic_All_Outlets.ipynb).

## What This Notebook Is

This notebook is the canonical merged-topic workflow for the thesis.

It combines two things:
- the seven saved **outlet-specific** BERTopic models from [local_outputs](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/local_outputs)
- the canonical combined article corpus in [df_combined.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined.csv)

Important lineage:
- The seven loaded models are **not** the legacy overall model from [07_OverallTM.ipynb](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/07_OverallTM.ipynb).
- They were trained separately in the outlet notebooks in `00_Initial EDA/01_*.ipynb` to `08_*.ipynb`.
- Those cleaned outlet corpora were later concatenated into [df_combined.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined.csv).
- The merged notebook therefore merges outlet-specific topic spaces first, then applies the merged topic space back onto the canonical combined article corpus.

## Why `df_combined` And The Merged Corpus Can Differ

The notebook now starts from [df_combined.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined.csv), so the article universe is aligned.

The remaining row difference comes only from BERTopic document preparation in [bertopic_pipeline.py](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/bertopic_pipeline.py):
- minimum text length
- minimum token count
- cleaned-document deduplication
- boilerplate cleaning

So:
- `df_combined.csv` is the canonical input corpus
- `combined_prepared` is the BERTopic-ready subset after those preparation rules

That difference is expected. It is no longer caused by rebuilding the corpus from a different raw-data path.

## Notebook Flow

The notebook has 14 cells and runs top to bottom.

### Cell 0

High-level orientation.

It states:
- the seven models are outlet-specific
- the merged assignment basis is `df_combined.csv`
- the notebook outputs topic tables, a 3D UMAP, and keyword-comparison tables

### Cell 1

Environment and path setup.

It:
- finds the repo root
- adds [1a_BERTopic](/private/tmp/thesis_merge_main_20260329/1a_BERTopic) to `sys.path`
- sets the embedding model
- sets `MIN_SIMILARITY = 0.7`
- points model loading only at [local_outputs](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/local_outputs)
- defines the save path for the optional merged model

### Cell 2

Conceptual explanation of the seven loaded models.

This cell makes explicit that:
- the models are not from the legacy overall TM
- they come from the outlet notebooks
- article-level assignment is rerun on `df_combined.csv`

### Cell 3

Imports the helper functions from [merged_outlets_analysis.py](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/merged_outlets_analysis.py), resolves the seven model paths, and loads the saved BERTopic models into:
- `loaded_models["tagesschau"]`
- `loaded_models["rt"]`
- `loaded_models["antispiegel"]`
- `loaded_models["tichys"]`
- `loaded_models["nius"]`
- `loaded_models["compact"]`
- `loaded_models["deutschlandkurier"]`

This is where `tm_ts`, `tm_rt`, `tm_as`, `tm_te`, `tm_ns`, `tm_cm`, and `tm_dk` come from.

### Cell 4

Short markdown header for the merge step.

### Cell 5

Creates the merged BERTopic model.

It:
- calls `BERTopic.merge_models(models_to_merge, min_similarity=MIN_SIMILARITY)`
- stores the result in `merged_model`
- builds `merged_topic_info_display`
- optionally saves the merged model to [merged_all_outlets_model](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/local_outputs/merged_all_outlets_model)

### Cell 6

Short markdown explanation for the `df_combined`-based assignment step.

This is the key corpus-alignment cell. It states that:
- raw outlet-specific cleaning is already baked into `df_combined`
- the notebook does not rebuild rows from raw loaders
- `row_id` comes from `df_combined`

### Cell 7

Builds the merged article dataset from [df_combined.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined.csv).

It:
- loads `df_combined`
- splits it back into outlet slices
- applies BERTopic preparation outlet by outlet
- shows a `prepared_summary` table
- concatenates the prepared slices into `combined_prepared`
- applies `merged_model` to get one topic assignment per prepared article
- exports:
  - [merged_articles_with_topics.csv](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/local_outputs/merged_articles_with_topics.csv)
  - [df_combined_with_topics.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined_with_topics.csv)

Important output meaning:
- [merged_articles_with_topics.csv](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/local_outputs/merged_articles_with_topics.csv) contains the BERTopic-ready article subset
- [df_combined_with_topics.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined_with_topics.csv) keeps the full canonical corpus and marks rows that did not survive BERTopic preparation as unmatched

### Cell 8

Short markdown header for the merged topic list.

### Cell 9

Builds the full merged-topic reference table.

This table is the easiest place to inspect:
- `Topic`
- `DisplayTopic`
- `Count`
- `Name`
- `DisplayLabel`

### Cell 10

Short markdown header for the 3D UMAP.

### Cell 11

Builds the 3D UMAP view of the merged topic space.

It uses:
- `build_merged_article_umap_3d(...)`
- `plot_merged_topic_umap_3d(...)`

The 3D plot is article-level, not topic-centroid only.

### Cell 12

Short markdown header for the keyword summary.

### Cell 13

Rebuilds the topic representations from the final merged article-topic corpus and then exports keyword tables.

Main outputs:
- `top_topic_size_df`
- `topic_keyword_weights_df`
- `topic_keyword_comparison_df`
- `top_topic_rank_table_df`
- `all_topic_keyword_weights_df`
- `all_topic_rank_table_df`

Why this matters:
- the saved merged model was originally persisted with only `10` topic terms per topic
- the final cell now refreshes that representation to `20` terms from the actual assigned article corpus
- the all-topic rank table gives one row per topic with `keyword_01 ... keyword_20`, which is the easiest format for comparing nearby topics such as multiple Russia-related clusters

## The Most Important Helper Functions

These live in [merged_outlets_analysis.py](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/merged_outlets_analysis.py).

### `resolve_model_paths(...)`

Resolves the seven saved outlet model folders from [local_outputs](/private/tmp/thesis_merge_main_20260329/1a_BERTopic/local_outputs).

### `load_df_combined(...)`

Loads the canonical combined article corpus from [df_combined.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined.csv).

### `split_df_combined_by_outlet(...)`

Splits `df_combined` back into the seven outlet slices.

### `load_all_prepared_documents_from_df_combined(...)`

Applies BERTopic preparation outlet by outlet directly to the `df_combined` slices.

This is the main corpus-prep function for the notebook.

### `build_merged_article_frame(...)`

Applies the merged BERTopic model to the prepared article corpus and returns:
- article-level topic assignments
- display-ready topic info
- the fitted 2D UMAP model

### `export_df_combined_with_topics(...)`

Writes [df_combined_with_topics.csv](/private/tmp/thesis_merge_main_20260329/00_Initial%20EDA/df_combined_with_topics.csv), which is the full canonical corpus enriched with the merged-topic columns.

### `build_topic_keyword_summary(...)`

Builds the side-by-side keyword comparison tables used in the final notebook cell.

### `refresh_topic_representations(...)`

Recomputes topic keywords from the final assigned merged-article corpus.

Use this when the saved BERTopic artifact only exposes `10` keywords per topic but you want `20` or more.

### `build_topic_keyword_rank_table(...)`

Builds one row per topic with `keyword_01`, `keyword_02`, ... columns formatted as `keyword (weight)`.

This is the easiest export for manual topic naming.

## What `keyword_weight` Means

`keyword_weight` is **not** a probability.

It is the BERTopic c-TF-IDF weight for a term inside a topic.

Interpretation:
- high `keyword_weight` = the word is more characteristic of this topic relative to the other topics
- low `keyword_weight` = the word appears, but it is less distinctive

Conceptually, BERTopic computes this by:
1. treating each topic as one aggregated document
2. computing term frequency inside that topic
3. downweighting words that are common across many topics

That is why `keyword_weight` is useful for comparing nearby topics such as different Russia-related clusters: it reflects distinctiveness, not simple frequency.
