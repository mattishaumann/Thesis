# Thesis Project: Agenda Distortion in German Alternative Media

## Overview

This is an MSc thesis investigating whether German alternative media systematically construct a distorted political reality that poses a structural threat to democratic discourse. The study compares topic coverage across 7 German news outlets (1 mainstream reference, 6 alternative/right-wing/pro-Russian) using BERTopic-based topic modeling. Study period: **August 2025 -- January 2026**.

The core claim being tested: alternative media don't just have a different editorial line -- they systematically narrow the agenda, prioritize different topics, and create a parallel information ecosystem that diverges structurally from mainstream coverage.

---

## Outlets Under Study

| Outlet | Category | Key | Articles (post-cleaning) | Corpus Size |
|--------|----------|-----|--------------------------|-------------|
| **Tagesschau** | Mainstream (public broadcaster) | `tagesschau` | 6,320 | REFERENCE |
| **RT DE** | Pro-Russian (state-funded) | `rt` | 4,560 | Large |
| **NIUS** | Right-Populist | `nius` | 3,269 | Medium |
| **Tichys Einblick** | Right-Populist / Conservative-Libertarian | `tichys` | 2,756 | Medium |
| **Compact** | Right-Wing (banned magazine, online-only) | `compact` | 1,486 | Small |
| **Deutschlandkurier** | Right-Wing | `deutschlandkurier` | 1,484 | Small |
| **Anti-Spiegel** | Pro-Russian (one-man blog) | `antispiegel` | 565 | Very Small |

**Canonical cleaned corpus (`df_combined.csv`)**: 20,440 articles.

**BERTopic-prepared subset used for merged assignment**: 20,358 articles.

**Important context on corpus imbalance**: Tagesschau has 11x more articles than Antispiegel. All metrics are designed to be size-controlled or rank-based to handle this. Corpus size still matters for statistical power (Antispiegel results carry wider confidence intervals).

---

## Three Research Hypotheses

| # | Hypothesis | Pillar | Methods | Status |
|---|-----------|--------|---------|--------|
| **H1** | Alt media focus on a narrower topic set and/or diverge from mainstream topic priorities (agenda distortion) | Agenda Distortion | Merged BERTopic, JSD, Spearman rho, Top-K overlap, Shannon entropy, coverage breadth | **In progress** (v1 complete, robustness done, interpretation pending) |
| **H2** | Alt media portray democratic institutions as corrupt/failing (delegitimization) | Delegitimization | NER, Framing Analysis | Pending |
| **H3** | Alt media use anger-dominant rhetoric vs mainstream (affective mobilization) | Affective Mobilization | Sentiment Analysis, LLM emotion classification | Pending |

**Workflow rule**: Each hypothesis must be fully grounded in literature, run, evaluated, robustness-checked, and documented before proceeding to the next.

---

## Project Structure

```
Thesis/
├── 00_Initial EDA/                          # Per-outlet data cleaning & EDA
│   ├── 01_RT.ipynb                          # RT Deutsch cleaning
│   ├── 02_Compact.ipynb                     # Compact Magazine cleaning
│   ├── 03_Nius.ipynb                        # Nius cleaning
│   ├── 04_Tichys_Einblick.ipynb             # Tichys Einblick cleaning
│   ├── 05_Antispiegel.ipynb                 # Antispiegel cleaning
│   ├── 06_Tagesschau.ipynb                  # Tagesschau cleaning (most complex: nested JSON)
│   ├── 07_OverallTM.ipynb                   # Combined topic model (legacy)
│   ├── 08_Deutschlandkurier.ipynb           # Deutschlandkurier cleaning
│   ├── *_clean.csv                          # Cleaned per-outlet CSVs
│   └── df_combined.csv                      # All outlets combined
│
├── data preprocessing/                      # Corpus-level EDA for thesis
│   ├── 09_Corpus_EDA.ipynb                  # Descriptive statistics & figures
│   ├── figures/                             # EDA figures (PDF for LaTeX)
│   └── tables/                              # EDA tables (CSV)
│
├── 1a_BERTopic/                             # Core BERTopic implementation
│   ├── bertopic_pipeline.py                 # Document prep, model building, training
│   ├── bertopic_config.py                   # BERTopicConfig dataclass (all hyperparameters)
│   ├── stopwords_de.py                      # Curated German stopwords (3 tiers)
│   ├── merged_outlets_analysis.py           # 2.3k-line analysis hub: loaders, specs, viz
│   ├── Merged_BERTopic_All_Outlets.ipynb    # Main merged-topic workflow notebook
│   └── local_outputs/                       # Canonical saved outlet models + merged exports
│
├── experiments/
│   └── agenda_distortion/                   # H1 experiment (current focus)
│       ├── modeling.py                      # Model loading, merging, iteration orchestration
│       ├── metrics.py                       # 5 H1 KPIs (literature-grounded)
│       ├── robustness.py                    # Permutation, downsampling, sensitivity tests
│       ├── visualization.py                 # Publication-quality thesis figures
│       ├── findings.md                      # H1 results tracker (partially filled)
│       ├── iteration_log.md                 # Detailed v1/v2 results + bootstrap CIs
│       ├── 01_model_inspection.ipynb        # Model introspection & topic quality
│       ├── 02_h1_results.ipynb              # KPI computation & primary findings
│       ├── 03_robustness.ipynb              # Validation suite
│       └── outputs/v1/                      # v1 artifacts (CSVs, figures, meta.json)
│
├── docs/
│   └── HYPOTHESES.md                        # Research roadmap
│
├── tex/                                     # Thesis LaTeX (git submodule → Overleaf)
│
├── data/raw/Alternative Medien/             # Raw scraped articles (DO NOT modify)
│
├── .venv/                                   # General Python env (pandas, jupyter)
└── .venv312/                                # ML Python env (BERTopic, transformers, UMAP)
```

---

## Technical Pipeline

### Data Cleaning (per outlet)

Each outlet has a dedicated Jupyter notebook (`00_Initial EDA/01-08_*.ipynb`) that:
1. Loads raw CSVs/XLSX from `data/raw/`
2. Selects and renames columns to standard schema: `Date, Title, Text, URL, source`
3. Cleans outlet-specific artifacts (affiliate links, podcast refs, HTML, timestamps, ads)
4. Filters to Aug 1, 2025 -- Jan 31, 2026
5. Removes duplicates on Title + Text
6. Exports `*_clean.csv`

Those seven `*_clean.csv` files are then concatenated into `00_Initial EDA/df_combined.csv`, which is now the canonical cross-outlet corpus for merged-topic assignment.

**Outlet-specific quirks**:
- **Tagesschau**: Nested JSON content blocks requiring HTML extraction
- **Compact**: 214 source CSVs, multi-encoding issues, affiliate links in 4.5% of articles
- **Nius**: JSON author field, "Show" category = ads (removed)
- **Tichys Einblick**: SoundCloud/podcast artifacts in text
- **Antispiegel**: Podcast titles removed ("Tacheles #"), smallest corpus

### BERTopic Pipeline

**Embedding model**: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)

**Important lineage**:
- The seven saved outlet models in `1a_BERTopic/local_outputs/` are **not** the legacy overall model from `00_Initial EDA/07_OverallTM.ipynb`.
- They are outlet-specific BERTopic models trained separately in the per-outlet notebooks.
- The merged-topic notebook combines those saved outlet models, then applies the merged topic space back to the canonical `df_combined.csv` corpus.

**Document preparation** (`bertopic_pipeline.py`):
- Light text cleaning (HTML unescape, boilerplate regex removal, whitespace normalization)
- No stemming/lemmatization (preserves embedding quality)
- Filters: `min_text_chars >= 50`, `min_tokens >= 8`
- Deduplication

**Vectorizer**: sklearn `CountVectorizer` with `ngram_range=(1,2)`, `min_df=2`, `max_df=0.9`, German stopwords

**Dimensionality reduction**: UMAP (`n_neighbors=30`, `n_components=5`, `min_dist=0.0`, `metric=cosine`)

**Clustering**: HDBSCAN (`cluster_selection_method="eom"`, `prediction_data=True`)

**Per-outlet tuning** (HDBSCAN `min_cluster_size` scaled to corpus size):

| Outlet | min_cluster_size | Rationale |
|--------|-----------------|-----------|
| Tagesschau | 35 | Largest (6.3k) |
| RT | 15 | Large (4.6k) |
| Nius | 15 | Medium (3.3k) |
| Tichys Einblick | 10 | Medium (2.8k) |
| Compact | 20 | Default |
| Deutschlandkurier | 20 | Default |
| Antispiegel | 8 | Smallest (565), also min_df=1 |

**Random state**: `42` everywhere (numpy, UMAP, HDBSCAN, BERTopic)

### Model Merging Strategy

**Why merged models instead of one global model?**
A global model lets large outlets (Tagesschau ~6k, RT ~4.5k) dominate topic discovery, drowning out niche topics from small outlets (Antispiegel ~565). The alternative approach:

1. Train 7 **independent per-outlet BERTopic models** (each tuned to its corpus size)
2. **Merge** via `BERTopic.merge_models(min_similarity=0.7)` into a unified topic space
3. The merge creates a **union** of all discovered topics -- if Antispiegel discovers a niche topic that Tagesschau doesn't cover, it still appears in the merged model
4. All articles are then assigned to merged topics, enabling direct cross-outlet comparison

**v1 result**: 74 merged topics from 7 per-outlet models.

---

## H1: Agenda Distortion -- Detailed Methodology & Results

### Literature Grounding

| Reference | Contribution | Applied as |
|-----------|-------------|------------|
| McCombs & Shaw (1972) | Agenda-setting theory | Spearman rank correlation of topic salience |
| DiMaggio et al. (2013) | JSD for ideological distance | JSD on per-outlet topic distributions vs Tagesschau |
| Boydstun et al. (2014) | Media attention diversity | Normalized Shannon entropy |
| Heidenreich et al. (2019) | Media fragmentation | Top-K topic overlap |
| Jacobi et al. (2016) | Topic models for media analysis | Single merged model, per-outlet distributions |
| Grootendorst (2022) | BERTopic | Contextual topic modeling with c-TF-IDF |

### Five H1 Metrics

Each metric captures a different dimension of agenda distortion. All are size-controlled or rank-based to handle corpus imbalance:

1. **Normalized Shannon Entropy** (Boydstun et al. 2014)
   - Formula: H / log(K), where H = -sum(p_i * log(p_i))
   - Range: 0 (single topic) to 1 (uniform distribution)
   - Lower values = more concentrated coverage = TYPE A signal
   - Measures: how evenly an outlet distributes attention across topics

2. **Jensen-Shannon Divergence vs Tagesschau** (DiMaggio et al. 2013)
   - Symmetric divergence between outlet's topic distribution and Tagesschau's
   - Range: 0 (identical) to 1 (completely different)
   - Higher values = more distorted agenda
   - Measures: overall distributional distance from mainstream

3. **Spearman Rank Correlation (rho)** (McCombs & Shaw 1972)
   - Correlation between outlet's topic ranking and Tagesschau's ranking
   - Range: -1 (inverted priorities) to +1 (identical priorities)
   - Size-independent (rank-based)
   - Measures: whether outlet prioritizes the same topics as mainstream

4. **Top-K Overlap** (Heidenreich et al. 2019)
   - Fraction of outlet's top-10 topics that appear in Tagesschau's top-10
   - Range: 0 (no overlap) to 1 (identical top-10)
   - Rank-based, size-independent
   - Measures: agreement on the most salient topics

5. **Coverage Breadth (Relative)** (custom, binomial model)
   - Ratio of actual topics covered (>= 10 articles) to expected coverage given corpus size
   - < 1.0 = narrower than random sampling = TYPE A signal
   - > 1.0 = broader than expected
   - Measures: whether outlet restricts its topic range

### Distortion Classification System

Based on metric thresholds, each outlet is classified as:

| Type | Definition | Signal |
|------|-----------|--------|
| **REFERENCE** | Tagesschau (mainstream baseline) | -- |
| **TYPE A** | Narrower topic coverage (low entropy, low breadth) | Focus restriction |
| **TYPE B** | Different topic priorities (low Spearman rho, low Top-K) | Priority divergence |
| **TYPE AB** | Both narrower AND different | Combined distortion |
| **UNCLEAR** | Doesn't clearly fit any pattern | Needs investigation |

### Iteration v1 -- Results

**Parameters**: `min_similarity=0.7`, outlier reduction via embeddings fallback (threshold=0.10)

**Corpus**: 20,455 articles across 7 outlets, 74 merged topics

This is the historical pre-alignment v1 run. The current canonical merged workflow now starts from `df_combined.csv` instead, so corpus accounting in the live notebook is `20,440` cleaned rows and `20,358` BERTopic-prepared rows.

| Outlet | Articles | Entropy | JSD | Spearman rho | Top-K Overlap | Breadth (rel.) | Classification |
|--------|----------|---------|-----|--------------|---------------|----------------|----------------|
| **Tagesschau** | 6,272 | 0.930 | 0.000 | +1.000 | 1.00 | 0.898 | REFERENCE |
| **RT** | 4,556 | 0.823 | 0.154 | +0.567 | 0.60 | 0.880 | TYPE AB |
| **Tichys Einblick** | 2,756 | 0.903 | 0.148 | +0.460 | 0.30 | 0.925 | TYPE B |
| **Nius** | 3,266 | 0.880 | 0.192 | +0.308 | 0.40 | 0.839 | TYPE AB |
| **Compact** | 1,580 | 0.878 | 0.213 | +0.253 | 0.50 | 0.780 | TYPE AB |
| **Deutschlandkurier** | 1,460 | 0.845 | 0.219 | +0.437 | 0.40 | 0.744 | TYPE AB |
| **Antispiegel** | 565 | 0.734 | 0.398 | +0.347 | 0.40 | 0.657 | TYPE AB (SMALL) |

### Key Findings from v1

1. **All alternative outlets show measurable agenda distortion** relative to Tagesschau, but the degree and type vary considerably by outlet.

2. **Antispiegel shows the strongest distortion** (JSD=0.398, entropy=0.734, breadth=0.657) -- highest divergence, most concentrated, narrowest coverage. However, it is also the smallest corpus (565 articles), so these results need careful bootstrap validation. Bootstrap CIs are tight ([0.390, 0.444] for JSD) and do not overlap with other outlets, suggesting Antispiegel is genuinely in a different distortion category.

3. **Tichys Einblick is a unique outlier**: it has the *highest* breadth of all outlets (0.925, even broader than Tagesschau at 0.898!) but the *lowest* Top-K overlap (0.30). This means Tichys covers *many* topics but prioritizes *very different* ones from mainstream -- a pure TYPE B distortion (different agenda, not narrower agenda). This is the most interesting finding for the thesis.

4. **RT is the least divergent alternative outlet** (JSD=0.154, Spearman rho=0.567 -- highest among alt media). Despite being Russian state-funded media, its topic priorities most closely mirror Tagesschau's. This could mean RT strategically covers the same topics as mainstream but with different framing (testable in H2/H3).

5. **All Spearman correlations are positive but weak** (0.25--0.57). No outlet *inverts* mainstream priorities -- they *shift* them. This suggests a modification of the mainstream agenda rather than a wholesale replacement.

6. **Top-K overlap is uniformly low** (0.30--0.60). All alternative outlets devote their top-10 topics to substantially different subjects than Tagesschau's top-10.

7. **Right-wing outlets (Compact, Deutschlandkurier) cluster together** with similar JSD (~0.21--0.22), moderate Spearman rho (~0.25--0.44), and reduced breadth (0.74--0.78). They show a consistent TYPE AB pattern.

### Iteration v2 -- Without Outlier Reduction

v1 had a suspicious 0% outlier rate (the embeddings-based fallback was too aggressive). v2 was run **without outlier reduction** to compare.

**Outlier rates without reduction**:
| Outlet | Outlier Rate |
|--------|-------------|
| Tagesschau | 2.1% |
| RT | 3.1% |
| Antispiegel | 3.5% |
| Nius | 4.2% |
| Tichys Einblick | 6.6% |
| Deutschlandkurier | 6.7% |
| Compact | 10.6% |

**Critical finding**: Metrics are **remarkably stable** between v1 and v2. JSD differences are < 0.01, Spearman rho differences are < 0.02. The outlier reduction has negligible impact on core findings. Main difference: Top-K overlap varies by +/-0.1 (expected, since reassigning outliers shifts which topics make the top-10).

**Decision**: Use v2 (no outlier reduction) as primary results to avoid the methodological concern about the embeddings fallback. The 2--10% outlier rates are acceptable.

### Robustness Checks

#### Bootstrap 95% Confidence Intervals (N=500)

**JSD vs Tagesschau** (all CIs well above 0 -- statistically meaningful):

| Outlet | Point Estimate | 95% CI |
|--------|---------------|--------|
| Antispiegel | 0.398 | [0.390, 0.444] |
| Compact | 0.213 | [0.207, 0.242] |
| Deutschlandkurier | 0.219 | [0.212, 0.251] |
| Nius | 0.192 | [0.185, 0.212] |
| RT | 0.153 | [0.147, 0.171] |
| Tichys Einblick | 0.148 | [0.143, 0.168] |

**Spearman rho** (all positive, all CIs exclude 0):

| Outlet | Point Estimate | 95% CI |
|--------|---------------|--------|
| Compact | +0.253 | [0.183, 0.297] |
| Nius | +0.308 | [0.247, 0.358] |
| Antispiegel | +0.347 | [0.261, 0.415] |
| Deutschlandkurier | +0.437 | [0.354, 0.480] |
| Tichys Einblick | +0.460 | [0.380, 0.492] |
| RT | +0.567 | [0.503, 0.604] |

**Top-K Overlap**:

| Outlet | Point Estimate | 95% CI |
|--------|---------------|--------|
| Tichys Einblick | 0.30 | [0.10, 0.40] |
| Antispiegel | 0.40 | [0.30, 0.50] |
| Deutschlandkurier | 0.40 | [0.30, 0.40] |
| Nius | 0.40 | [0.30, 0.40] |
| Compact | 0.50 | [0.40, 0.60] |
| RT | 0.60 | [0.50, 0.60] |

**Key takeaways from bootstrap**:
- All JSD CIs are tight and far above 0 -- agenda divergence is statistically real
- Spearman CIs are wider but all exclude 0 -- ordering correlation is significant
- CIs do **not** overlap between Antispiegel and the rest on JSD -- Antispiegel is genuinely in a different distortion category
- RT and Tichys CIs overlap on JSD -- cannot distinguish them statistically (both ~0.15)

#### Permutation Test (N=500)

Outlet labels were shuffled while keeping topic assignments fixed. All outlets' observed JSD values are far above the null distribution (null mean ~0.006--0.03). **All p < 0.05** -- the agenda divergence is not a random artifact.

#### Downsampling Test (equalized to 565 articles)

All outlets downsampled to Antispiegel's size. JSD ordering remains stable: Antispiegel > Compact ~ Deutschlandkurier > Nius > Tichys > RT. Mean JSD increases slightly but direction intact. **Findings are not driven by corpus size imbalance.**

### Open Questions & Decision Points for H1

1. **Are 74 merged topics the right granularity?** Need sensitivity testing at `min_similarity` = 0.5, 0.6, 0.7, 0.8 to see if outlet rankings change.

2. **Should the outlier threshold be adjusted?** v2 (no outlier reduction) is now the primary approach, which avoids this question, but a principled outlier strategy would be better for the thesis write-up.

3. **Compact's 10.6% outlier rate** (highest) -- is this a data quality issue or a genuine signal that Compact covers more niche/incoherent topics?

4. **Tichys Einblick's TYPE B profile** -- this is the most interesting finding. Should we dig deeper into *which* topics Tichys covers that mainstream doesn't?

5. **RT's similarity to Tagesschau** -- does this mean RT mirrors mainstream topics but adds framing spin? This has direct implications for H2 (delegitimization framing).

6. **Should there be a v3 iteration?** Or are the current results stable enough to finalize H1 and move to H2?

---

## Evolution of Approaches (What Was Tried)

### Approach 1: Binary Global Models (Abandoned)

**What**: Two global BERTopic models -- one for "alternative media" (all 6 alt outlets combined) and one for "mainstream" (Tagesschau alone).

**Notebooks**: `experiments/agenda_distortion/01_topic_modeling.ipynb` (legacy), `02_comparison_and_kpis.ipynb` (legacy)

**Method**: Train separate BERTopic models, align topics via embedding similarity (threshold=0.65), compute JSD between the two distributions.

**Why abandoned**:
- Binary grouping ("alt media" vs "mainstream") hides per-outlet variation -- Antispiegel and RT are very different despite both being "alternative"
- Combining all alt outlets into one corpus lets large outlets (RT ~4.5k) dominate over small ones (Antispiegel ~565)
- A global model on the combined alt corpus discovers "average" topics, not outlet-specific ones
- Topic alignment via embedding similarity is fragile and requires manual threshold tuning

### Approach 2: Per-Outlet Merged Models (Current)

**What**: 7 independent per-outlet BERTopic models, merged via `BERTopic.merge_models()` into a unified topic space.

**Notebooks**: `experiments/agenda_distortion/01_model_inspection.ipynb` (active), `02_h1_results.ipynb` (active), `03_robustness.ipynb` (active)

**Why this is better**:
- Each outlet discovers its own topics independently (no large-outlet dominance)
- The merge creates a union of all topics -- niche topics from small outlets are preserved
- Per-outlet topic distributions can be compared directly in the unified space
- Enables per-outlet distortion profiling (not just binary comparison)
- Supports the 5-metric framework with literature grounding

**Iterations**:
- **v1**: With outlier reduction (embeddings fallback, threshold=0.10). 74 topics, 0% outlier rate (too aggressive). Metrics computed but concern about artificial topic inflation.
- **v2**: Without outlier reduction. 74 topics, 2--10% outlier rates (reasonable). Metrics nearly identical to v1 -- confirmed that findings are not driven by outlier handling.
- **Bootstrap validation**: 500 resamples, all CIs exclude 0, Antispiegel confirmed as distinct category.
- **Permutation test**: All p < 0.05, findings are not random artifacts.
- **Downsampling test**: Rankings stable when equalizing corpus sizes.

### Approach 3: LLM-Validated Topics (Experimental)

**Notebook**: `experiments/agenda_distortion/04_validation_loop.ipynb`

**What**: Use Claude API to assess topic coherence and agenda-distortion plausibility for top alternative-media topics.

**Status**: Experimental, not yet integrated into main findings. Could serve as a qualitative validation layer.

---

## Key Technical Details for Consulting

### Embedding Model Choice
`paraphrase-multilingual-MiniLM-L12-v2` -- chosen for multilingual support (German), good balance of quality and speed, widely used in BERTopic literature. Produces 384-dimensional embeddings.

### Stopword System
Three tiers: 129 base German stopwords + 9 project-specific terms (artikel, bild, video, etc.) + 6 outlet names. Tichys Einblick gets additional SoundCloud/podcast-related stopwords. Custom function `get_german_stopwords()` supports per-outlet customization.

### Output Artifacts (v1)
Located in `experiments/agenda_distortion/outputs/v1/`:
- `merged_articles.csv` -- 20,455 rows, each article with merged topic assignment + UMAP coords from the historical v1 run
- `merged_topic_info.csv` -- 74 topics with c-TF-IDF labels, counts, representative docs
- `h1_metrics.csv` -- 7 rows (one per outlet), 12 columns (all metrics)
- `h1_classification.csv` -- distortion type per outlet
- `h1_bootstrap.csv` -- 95% CIs for JSD, Spearman rho, Top-K
- `robustness_permutation_jsd.csv`, `robustness_permutation_rho.csv` -- null distributions
- `robustness_downsample.csv` -- downsampled metric means and stds
- `figures/` -- 8 publication-quality PDFs (metrics panel, heatmaps, radar, forest plots)
- `meta.json` -- iteration parameters, topic count, outlier rates, timing

### Visualization Style
All figures use consistent thesis styling via `THESIS_COLORS` (per-outlet hex colors) and `THESIS_RC` (matplotlib rcParams). Exported as 300 dpi PDF for LaTeX inclusion.

---

## What Comes Next

### Immediate (H1 finalization)
- Decide whether v2 results are final or if a v3 iteration is needed
- Run `min_similarity` sensitivity analysis (0.5, 0.6, 0.7, 0.8)
- Finalize `findings.md` with interpretation
- Write H1 section in thesis LaTeX

### After H1
- **H2 (Delegitimization)**: NER to extract institution mentions, framing analysis to detect delegitimizing frames. RT's similarity to Tagesschau makes it a key test case -- does it cover the same topics but frame institutions negatively?
- **H3 (Affective Mobilization)**: Sentiment/emotion analysis, potentially LLM-based. Compare emotional register across outlets.

### Infrastructure
- Thesis LaTeX in `tex/` submodule (synced with Overleaf)
- Two Python environments: `.venv/` (general) and `.venv312/` (ML stack with BERTopic, UMAP, HDBSCAN, sentence-transformers)
