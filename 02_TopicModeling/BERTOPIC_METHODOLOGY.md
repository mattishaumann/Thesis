# BERTopic Methodology: Per-Outlet Models and Merged Topic Space

This document records the modeling decisions made for the per-outlet and merged BERTopic pipeline, with academic justification for each choice. It is intended as a reference for writing the thesis methods section.

## Finalized Thesis Run

For the thesis, the authoritative merged-model result is the frozen snapshot
`02_TopicModeling/outputs/frozen_merged_runs/2026-04-03_merged_v1`.

This distinction matters:

- `Merged_BERTopic_Step_By_Step.ipynb` documents the build procedure for the merged model.
- The frozen snapshot is the finalized empirical run used for downstream thesis analysis.
- Later reruns may overwrite the canonical merged output path and can drift from the finalized run even when the workflow is similar.

Methodologically, the thesis should therefore distinguish between:

1. the documented build procedure
2. the finalized archived run actually used for results

This is a reproducibility and audit-trail measure, not a contradiction. The
frozen snapshot preserves the exact model artefacts and article-level topic
assignments that underpin the reported results.

---

## 1. Why BERTopic, not LDA

Classical Latent Dirichlet Allocation (LDA; Blei et al., 2003) assumes a bag-of-words representation and treats all words as exchangeable, which causes it to conflate semantically distinct uses of the same word. BERTopic (Grootendorst, 2022) instead embeds documents using a pre-trained language model, clusters them in embedding space via HDBSCAN, and then extracts topic keywords using a class-based variant of TF-IDF (c-TF-IDF). This approach has three advantages for this project:

1. **Contextual embeddings**: The multilingual MiniLM model captures meaning rather than surface form, which is critical for German news where the same political term (e.g., "Migration") appears across very different frames.
2. **No fixed topic count**: HDBSCAN finds the number of clusters from the data; forcing a fixed *k* (as in LDA) would require arbitrary prior specification across seven corpora of very different sizes.
3. **Outlier-aware clustering**: HDBSCAN explicitly assigns low-density documents to an outlier class (-1) rather than forcing every document into a topic - this is methodologically honest for news text, which contains many one-off articles that do not belong to any stable theme.

**Citation**: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

---

## 2. Embedding model

**Choice**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

This model was trained on multilingual parallel corpora using knowledge distillation from a larger English model (Reimers & Gurevych, 2020). It produces 384-dimensional sentence embeddings and has been widely validated on semantic similarity tasks across 50+ languages including German. It was chosen over larger multilingual models (e.g., `multilingual-e5-large`) because:

- It runs on CPU without excessive memory requirements for a ~22k-document corpus
- It is the standard embedding model in the BERTopic literature for multilingual corpora
- It produces sufficient separation between news topics for HDBSCAN to cluster effectively

The embedding model is fixed across all seven outlets and the merged model to ensure that distances in embedding space are comparable.

**Citations**:
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.
- Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. *EMNLP 2020*.

---

## 3. Dimensionality reduction: UMAP

**Why UMAP before clustering?** HDBSCAN's distance metric degrades in high-dimensional space (the "curse of dimensionality"; Bellman, 1961). Reducing 384-dimensional embeddings to 5 dimensions via UMAP (McInnes et al., 2018) preserves local neighborhood structure while making distances meaningful for clustering.

**Fixed parameters across all outlets**:
- `n_components=5` (BERTopic standard; 5 dims capture sufficient local structure for HDBSCAN)
- `metric="cosine"` (appropriate for embedding vectors, which live on a hypersphere)
- `random_state=42` (reproducibility)

**Per-outlet tuned parameters**:
- `n_neighbors` (15-30): controls how local vs. global the manifold is. Smaller values → more local structure → better for smaller corpora. See per-outlet table in Section 6.
- `min_dist` (0.0-0.05): controls cluster tightness. 0.0 forces tighter clusters; 0.05 allows slight spread.

**Citation**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.

---

## 4. Clustering: HDBSCAN

HDBSCAN (Campello et al., 2013; McInnes et al., 2017) is a density-based hierarchical clustering algorithm. Unlike k-means or LDA, it does not require specifying the number of clusters in advance and assigns low-density points to an outlier class (-1).

**Fixed parameters across all outlets**:
- `cluster_selection_method="eom"` (Excess of Mass; selects stable clusters from the hierarchy)
- `metric="euclidean"` (applied in the 5-dimensional UMAP space)
- `prediction_data=True` (required for `reduce_outliers()` and `transform()`)

**Key tuned parameter: `min_cluster_size`**
This controls the minimum number of documents required to form a topic. It is the primary lever for adjusting topic granularity and **must be scaled to corpus size**. If `min_cluster_size` is too small relative to the corpus, HDBSCAN over-splits into microtopics. If too large, it under-splits into coarse super-topics.

See per-outlet parameter table in Section 6.

**Citations**:
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD 2013*.
- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density-based clustering. *Journal of Open Source Software, 2*(11).

---

## 5. Why `nr_topics` (forced topic reduction) was removed

An earlier iteration of the Compact and Tichys Einblick notebooks used `BERTopic.reduce_topics(nr_topics=25)` after outlier reduction. This was removed for the following reasons:

**What `reduce_topics()` does**: It iteratively merges the two most similar topics (by c-TF-IDF cosine similarity) until the target count is reached. The merged topics' representations are averaged together.

**Problem 1 - blurred representations**: Forced merging averages c-TF-IDF vectors across semantically distinct topics. When these blurred representations are then fed into `BERTopic.merge_models()`, their similarity to topics from other outlets is artificially distorted. A forced-merged topic "German politics + AfD + Bundestag elections" will match (or fail to match) Tagesschau topics that would correctly match only one of its constituent sub-topics.

**Problem 2 - methodological inconsistency**: If Compact and Tichys were capped at 25 topics while other outlets were left at their natural counts (18-74), the per-outlet models would not be comparable. The granularity of the merged topic space is determined by the *union* of all per-outlet topics; artificially compressing two outlets makes their contribution to the merged space structurally different.

**Problem 3 - the right tool is `min_cluster_size`**: Topic count should emerge from the data. The correct way to control granularity is `min_cluster_size`: larger values yield fewer, coarser topics; smaller values yield more, finer-grained topics. This is a principled, data-driven approach documented in the BERTopic and HDBSCAN literature (Grootendorst, 2022; McInnes et al., 2017).

**How to justify removal in the thesis**: Report that all per-outlet models use HDBSCAN's natural cluster discovery without post-hoc reduction. State that `min_cluster_size` was tuned per outlet proportionally to corpus size (see Section 6), and that the resulting topic counts were validated by inspecting topic coherence and docs-per-topic ratios.

---

## 6. Per-outlet model parameters (final state, April 2, 2026)

All models saved in `02_TopicModeling/outputs/`. All use `min_topic_size=10`, `top_n_words=30`, `n_gram_range=(1,1)`.

| Outlet | Docs | `min_cluster_size` | `min_samples` | `n_neighbors` | `min_dist` | Outlier reduction | Topics | Outlier% | Docs/topic |
|--------|------|--------------------|--------------|--------------|-----------|-------------------|--------|----------|-----------|
| Tagesschau | 6,319 | 35 | 2 | 25 | 0.0 | c-tf-idf, 0.10 | 55 | 11.8% | 101.4 |
| RT DE | 4,560 | 14 | 4 | 18 | 0.05 | c-tf-idf, 0.10 | 74 | 12.1% | 54.2 |
| Nius | 3,266 | 15 | 10 | 30 | 0.0 | c-tf-idf, 0.10 | 38 | 15.1% | 73.0 |
| Tichys Einblick | 2,756 | 10 | 10 | 15 | 0.0 | c-tf-idf, 0.05 | 51* | 1.1% | 53.4 |
| Compact | 1,486 | 10 | 3 | 18 | 0.05 | c-tf-idf, 0.10 | TBD† | TBD | TBD |
| DK | 1,465 | 20 | 10 | 30 | 0.0 | c-tf-idf, 0.10 | 18 | 11.9% | 71.7 |
| Anti-Spiegel | 565 | 8 | 2 | 10 | 0.05 | c-tf-idf, 0.10 | 30 | 6.5% | 17.6 |

*Tichys went from 24 (hard-capped) to 51 (natural) after removing `reduce_topics(25)`.
†Compact rerun pending as of Apr 2, 2026.

**Rationale for `min_cluster_size` scaling**:
- Tagesschau (35): largest corpus, conservative clustering to avoid over-splitting 6k articles
- RT (14): medium-large, slightly tighter to capture RT's geographically diverse topic mix
- Nius/DK (15/20): medium corpora, default-range settings
- Tichys (10): tuned down to capture the breadth of a politically diverse outlet
- Compact (10): small corpus, small mcs needed to produce enough topics
- Anti-Spiegel (8): smallest corpus (565 docs); mcs must be very small or almost no clusters form. `min_samples=2` also loosened to reduce sensitivity to noise in a thin corpus

**On the Tichys outlier threshold (0.05 vs 0.10)**:
The stricter threshold (0.05) was deliberately chosen for Tichys because the baseline model produced an unusually high outlier rate. Lowering the threshold more aggressively reassigns borderline documents, resulting in a 1.1% outlier rate. This is a valid per-outlet tuning decision - the threshold controls how aggressively outlier documents are forced into the nearest topic by c-TF-IDF similarity. The 0.05 threshold for Tichys is higher-quality than forcing all outlets to use 0.10, because it reflects the actual density structure of that corpus.

**To report in thesis**: State that `reduce_outliers(strategy="c-tf-idf", threshold=0.10)` was applied to all outlets after initial clustering, with threshold=0.05 for Tichys Einblick where the baseline outlier rate warranted more aggressive reassignment. Cite Grootendorst (2022) for the c-TF-IDF outlier reduction procedure.

---

## 7. Outlier rates: methodological assessment

HDBSCAN explicitly separates "noise" from clusters, which means some documents will always be unassigned (topic -1). The observed outlier rates (1.1%-15.1%) are within the expected range for news corpora and do not indicate a modeling failure.

**How to interpret each rate**:

| Outlet | Outlier% | Interpretation |
|--------|----------|----------------|
| Tagesschau | 11.8% | Normal for a large, diverse public broadcaster corpus |
| RT DE | 12.1% | Normal; RT covers many geographically dispersed topics |
| Nius | 15.1% | Elevated but expected - Nius is a general-interest outlet with high topical diversity (crime, lifestyle, politics, celebrity) that resists neat clustering |
| Tichys | 1.1% | Very low; c-tf-idf threshold=0.05 reassigns most borderline docs |
| Compact | TBD | - |
| DK | 11.9% | Normal |
| Anti-Spiegel | 6.5% | Low; small corpus with thematically coherent content (Russia/Ukraine focus) |

**Academic justification for accepting these rates**: Outlier documents are not "lost" - they are reassigned during the `transform()` step in the merged model via nearest-neighbor lookup. The outlier rate in per-outlet models reflects the corpus's true topical coherence and is not a quality failure. This is consistent with the BERTopic framework's design philosophy (Grootendorst, 2022) and with the interpretation of HDBSCAN noise points in the density-based clustering literature (Campello et al., 2013).

**To report in thesis**: Report per-outlet outlier rates as a descriptive statistic. State that documents initially assigned to the outlier class were subsequently reassigned during the merged-model transformation step. Note that Nius's elevated outlier rate (15.1%) reflects genuine topical diversity rather than modeling failure, citing the outlet's broad editorial scope.

---

## 8. The merge strategy: why `BERTopic.merge_models()`

**Core problem**: A single global BERTopic model trained on all 22k articles would let large outlets (Tagesschau: 6,319; RT: 4,560) dominate topic discovery. Smaller outlets (Anti-Spiegel: 565) would not generate enough within-outlet density to form distinct clusters, and their unique topics would be absorbed into mainstream clusters or assigned to outliers.

**Solution**: Train 7 independent per-outlet models, then merge into a unified topic space using `BERTopic.merge_models()`.

**What `merge_models()` does** (Grootendorst, 2022):
1. Takes all topic representations (c-TF-IDF vectors) from all input models
2. Computes pairwise cosine similarity across all topics from all outlets
3. Topics with similarity ≥ `min_similarity` (0.7) are merged into one canonical topic
4. Topics below the threshold remain as distinct topics in the unified space
5. The result is a new BERTopic model with a unified vocabulary and topic set

**Why `min_similarity=0.7`**: This threshold was chosen as the standard BERTopic merge threshold. It is high enough to prevent spurious merging of distinct topics (e.g., "Ukraine war" and "German coalition politics" are both political but should not merge) and low enough to merge genuinely equivalent topics across outlets (e.g., "Israel-Gaza conflict" in Tagesschau and in RT should merge). Values in the range 0.65-0.75 are standard in the BERTopic literature for multi-corpus merging.

**What this enables analytically**:
- Every outlet's articles can be assigned to the same canonical topic set via `transform()`
- Direct comparison of per-outlet topic distributions is valid (same topic IDs, same semantics)
- Small-outlet topics are preserved - if Anti-Spiegel has a unique topic with no equivalent in Tagesschau, it appears in the merged model as its own topic with very low Tagesschau frequency

**Citation**: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*. See also the BERTopic documentation: https://maartengr.github.io/BERTopic/getting_started/merge/merge.html

---

## 9. Full reference list for methods section

- Bellman, R. (1961). *Adaptive Control Processes: A Guided Tour*. Princeton University Press.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993-1022.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining, PAKDD 2013* (pp. 160-172). Springer.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.
- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density-based clustering. *Journal of Open Source Software, 2*(11), 205.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*.
- Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. In *Proceedings of EMNLP 2020*.
