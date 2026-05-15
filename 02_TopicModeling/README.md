# 02 Merged Topic Model and Agenda-Divergence Analysis

This folder is **Mechanism 1 Agenda Divergence**. It takes the per-outlet BERTopic models built in [`../01_EDA_TopicModeling_perOutlet/`](../01_EDA_TopicModeling_perOutlet/) and merges them into a single shared topic space, then runs the agenda-divergence analysis on top.

## Merge step

`BERTopic.merge_models()` is run with `min_similarity = 0.70` on the c-TF-IDF topic embeddings, using Tagesschau as the anchor. **268 source topics collapse to 72 merged topics.** The merge was sensitivity-checked at 0.65 and 0.75. The full run lives in [`BERTopic_merged.ipynb`](BERTopic_merged.ipynb).

## Manual layer

The 72 merged topics were hand-labeled and grouped into **18 thematic clusters** using top-30 c-TF-IDF terms plus representative documents. 

## Quality check

Topic coherence is measured with gensim's `CoherenceModel` (C_V) on the top-10 terms. Per-outlet models score between **0.60 and 0.76**; the merged model scores **0.727**. The threshold used for "coherent" is > 0.5.

## Analysis (Mechanism 1)

On the outlet × topic prevalence matrix:

- **Pairwise Jensen–Shannon divergence** (log2 base, symmetric, bounded [0, 1]) between every outlet pair.
- **Hierarchical clustering** on √JSD (proper metric) with average linkage.
- **Normalized Shannon entropy** per outlet (H / log2 K) as a breadth-of-agenda complement to JSD.

Everything is reported at **both** the topic level (72) and the cluster level (18) as a robustness check.

## Pointers


- [`BERTopic_merged.ipynb`](BERTopic_merged.ipynb): the merge run itself.
- `Topic_Analysis/`: JSD, entropy, hierarchical-clustering analysis notebooks.
