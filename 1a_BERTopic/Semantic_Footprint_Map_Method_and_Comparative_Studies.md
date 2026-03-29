# Semantic Footprint Map: Method, Interpretation, and Comparative Studies

## What This Visualization Does

The semantic footprint map is an outlet-level visualization of the shared semantic space produced by the merged BERTopic model. Its purpose is to show not just **which topics** an outlet covers, but **how broadly or narrowly** that outlet is distributed across the merged topic space.

In the thesis context, this matters because H1 is not only about topic presence or absence. It is also about **agenda concentration**:

- some outlets may cover relatively few topics overall
- others may technically cover many topics but still cluster strongly around a small number of semantic regions

The map is therefore designed to complement the quantitative H1 indicators by making the outlet's distribution in the merged semantic space visible.

## What the Plot Shows

- **Grey points**: all articles in the full merged corpus
- **Red points**: the articles of one selected outlet
- **Red KDE cloud**: a smoothed density overlay showing where that outlet is most concentrated
- **Topic labels**: only the merged topics in which the selected outlet has a meaningful number of articles
- **Annotation box**: article count, number of merged topics covered, and coverage share

The title format is:

`{Outlet} — Semantic Footprint in Merged Topic Space`

## Why This Plot Exists

The standard BERTopic document visualizations are useful, but they are not designed for the specific comparative task in this thesis:

1. compare multiple outlets in one shared merged topic space
2. isolate one outlet at a time against the full corpus
3. show both **coverage breadth** and **concentration density**
4. support interpretation of agenda distortion rather than only topic coherence

So this plot is best understood as a **custom comparative adaptation** built on top of standard BERTopic/UMAP document visualization logic.

## How the Semantic Footprint Map Is Constructed

The method has four main steps.

### 1. Load separately trained outlet models

Each outlet has its own saved BERTopic model:

- `ts_model`
- `rt_model`
- `as_model`
- `te_model`
- `ns_model`
- `compact_model`
- `dk_model`

These are loaded from `1a_BERTopic/local_outputs` first, with tracked outputs as fallback.

### 2. Merge the outlet models into one shared topic space

The shared topic space is created with:

```python
merged_model = BERTopic.merge_models(
    models_to_merge,
    min_similarity=0.7,
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
```

This is important methodologically because the thesis is not fitting one BERTopic model from scratch on the pooled corpus. Instead, it first models each outlet separately and then aligns those topic spaces through model merging.

That means the merged space should be interpreted as a **post hoc aligned semantic space**, not as a single jointly estimated topic model from the start.

### 3. Reconstruct article-level assignments in the merged space

The merged model is then applied to the prepared documents:

```python
topics, probabilities = merged_model.transform(docs)
embeddings = merged_model._extract_embeddings(docs, method="document")
```

This produces:

- one merged topic assignment per article
- one document embedding per article

### 4. Reduce document embeddings to 2D with UMAP

The document embeddings are projected to 2D:

```python
reducer = UMAP(
    n_neighbors=10,
    n_components=2,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)
coords = reducer.fit_transform(embeddings)
```

The resulting coordinates are stored as:

- `umap_x`
- `umap_y`

These coordinates are only for visualization. The axes themselves are not substantively interpretable.

## How to Read the Map

### Breadth

If the red points are distributed across many regions of the grey corpus cloud, the outlet has relatively broad semantic reach.

### Concentration

If the red points cluster tightly in only a few semantic regions, the outlet is more concentrated. This is especially visible when the KDE cloud forms one or two dense islands rather than a broad, diffuse footprint.

### Selective Emphasis

If an outlet appears across multiple areas but has very dense red regions in only a few places, that suggests the outlet does not merely cover those themes, but disproportionately emphasizes them.

### Topic Presence vs Topic Weight

This distinction is central to H1:

- an outlet may cover many merged topics
- but still devote much more attention to a smaller semantic core

The semantic footprint map makes that distinction visible.

## What the KDE Cloud Adds

The KDE cloud is a **kernel density estimate** over the highlighted outlet's 2D points. In plain terms, it smooths the red point cloud so that dense zones become visually apparent.

Without KDE:

- the map shows where points are
- but concentration patterns can be hard to read, especially with many overlapping documents

With KDE:

- dense zones become visible as semantic “mass”
- narrow versus diffuse agenda structure becomes easier to compare across outlets

So the KDE cloud is not there to create new information. It is there to make the distribution already present in the point cloud easier to interpret.

## What the Plot Is Not

The plot should not be over-claimed. It is not:

- a causal model of agenda setting
- a statistical test by itself
- a literal map with interpretable x/y axes
- proof that two outlets are “close” in a strict metric sense just because their clouds overlap visually

It is best treated as an **exploratory comparative visualization** that must be interpreted together with the H1 measures:

- coverage breadth relative
- entropy
- KL divergence from Tagesschau
- topic dominance
- top-topic overlap with Tagesschau

## Why This Fits the Thesis

The core thesis idea is that alternative media may distort the agenda in two ways:

- **TYPE A**: breadth restriction, meaning fewer topics than corpus size would predict
- **TYPE B**: concentration distortion, meaning a strong clustering around selected themes even if topic breadth is not low

The semantic footprint map is especially useful for **TYPE B** because concentration is much easier to understand visually than through a table alone.

## Is This an Established Plot in the Literature?

Based on the literature search, the safest conclusion is:

- the individual ingredients are established
- the exact full plot design used here does not appear to be a standard named plot
- the figure is best described as a **custom synthesis of existing visualization techniques**

That means:

- **UMAP document-space visualization** is standard
- **KDE over a 2D embedding space** is standard as a visualization move
- **highlighting one subset against a grey background corpus** is a common visualization logic
- but the exact combination used here for comparative BERTopic outlet analysis appears to be your own adapted research visualization

## Closest Technical Precedents

### 1. BERTopic document visualization

Official BERTopic documentation explicitly supports document-level 2D visualizations based on embeddings reduced to two dimensions.

Source:

- BERTopic documentation, `visualize_documents`:  
  https://maartengr.github.io/BERTopic/getting_started/visualization/visualize_documents.html

Relevance:

- this is the clearest direct precedent for plotting BERTopic documents in a 2D semantic space
- your plot extends this logic rather than replacing it

### 2. UMAP as the dimensionality reduction basis

UMAP is the dimensionality reduction method used to produce the 2D semantic coordinates.

Source:

- McInnes, Healy, Saul, and Großberger (2018), *UMAP: Uniform Manifold Approximation and Projection*, JOSS  
  https://joss.theoj.org/papers/10.21105/joss.00861

Relevance:

- this is the methodological basis for the 2D layout
- it supports using a 2D manifold projection for exploratory semantic structure visualization

### 3. KDE as a standard smoothing technique

Kernel density estimation is not novel here. It is a classical statistical smoothing method.

Source:

- Rosenblatt (1956), *Remarks on Some Nonparametric Estimates of a Density Function*  
  bibliographic listing with DOI and article details:  
  https://celebratio.org/Rosenblatt_M/article/789/

Relevance:

- this is the theoretical origin of KDE-style density estimation
- in your plot, KDE is used as a visualization overlay, not as the analytical core

### 4. UMAP + KDE contour visualization precedent

A particularly relevant precedent is work that combines UMAP projection with KDE contour visualization in embedding space.

Source:

- Wang et al. (2023), *DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models*, ACL 2023  
  https://aclanthology.org/2023.acl-long.51/

Relevance:

- not a BERTopic or news-bias paper
- but methodologically relevant because it shows that UMAP + density contour visualization is an accepted way to summarize structure in a 2D embedding space

## Closest Comparative Study Precedents

These are not exact plot precedents, but they are conceptually close to what the thesis is doing: comparing media sources, content selection, and ideological divergence across outlets.

### 1. Cross-article comparison for partisan event selection

Source:

- Liu et al. (2023), *All Things Considered: Detecting Partisan Events from News Media with Cross-Article Comparison*, EMNLP 2023  
  https://aclanthology.org/2023.emnlp-main.957/

Why it matters:

- this paper is directly relevant to the idea that media bias can appear through **content selection**, not only wording
- that is conceptually close to your agenda-distortion framing

### 2. Informational bias through factual content selection

Source:

- Fan et al. (2019), *In Plain Sight: Media Bias Through the Lens of Factual Reporting*, EMNLP 2019  
  https://aclanthology.org/D19-1664/

Why it matters:

- this study is important because it argues that bias is not only lexical
- the thesis similarly focuses on **what gets emphasized or selected**, not just how it is phrased

### 3. Diverse viewpoint identification in news aggregation

Source:

- Carlebach et al. (2020), *News Aggregation with Diverse Viewpoint Identification Using Neural Embeddings and Semantic Understanding Models*  
  https://aclanthology.org/2020.argmining-1.7/

Why it matters:

- this is a comparative news setting using semantic representations to distinguish viewpoints across articles
- it is not the same method, but it supports the general research logic of comparing outlet-level semantic patterns

### 4. Comparative partisan vs mainstream outlet analysis

Source:

- Potthast et al. (2018), *A Stylometric Inquiry into Hyperpartisan and Fake News*  
  https://aclanthology.org/P18-1022/

Why it matters:

- this paper is less about topic space and more about style
- but it is a useful comparative precedent for systematically contrasting mainstream and partisan publishers

## What Is Original in Your Use Case

The likely original part is not KDE or UMAP individually. It is the **combined research design**:

1. separately trained BERTopic models for several outlets
2. model merging into one shared semantic space
3. article-level projection into the merged space
4. outlet-specific highlight maps against the full corpus
5. KDE used to show outlet concentration
6. interpretation tied explicitly to agenda distortion

So the strongest novelty claim is not:

> “We invented a new statistical visualization.”

The safer and more accurate claim is:

> “We develop a custom comparative semantic-footprint visualization for outlet-level agenda concentration within a merged BERTopic space.”

## Suggested Thesis Wording

You can describe it like this:

> To visualize outlet-specific agenda structure within the merged topic space, I construct a semantic footprint map. Articles are embedded in a shared BERTopic document space and projected to two dimensions with UMAP. For each outlet, all corpus articles are shown as a grey background, while the outlet's own articles are highlighted in red. A KDE overlay is added to make outlet-specific concentration within the semantic space easier to interpret. The resulting map is used as an exploratory comparative visualization of topical breadth and concentration, complementing the quantitative H1 measures.

## Short Position on Novelty

The most defensible summary is:

- **KDE cloud**: established technique, not invented here
- **BERTopic document map**: established technique, supported by BERTopic docs
- **exact semantic footprint map**: best described as a custom synthesis tailored to comparative outlet analysis

## References

- BERTopic document visualization docs:  
  https://maartengr.github.io/BERTopic/getting_started/visualization/visualize_documents.html
- BERTopic visualization overview:  
  https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html
- McInnes et al. (2018), UMAP:  
  https://joss.theoj.org/papers/10.21105/joss.00861
- Rosenblatt (1956), KDE origins listing:  
  https://celebratio.org/Rosenblatt_M/article/789/
- Wang et al. (2023), DiffusionDB, ACL 2023:  
  https://aclanthology.org/2023.acl-long.51/
- Liu et al. (2023), cross-article partisan event comparison, EMNLP 2023:  
  https://aclanthology.org/2023.emnlp-main.957/
- Fan et al. (2019), informational bias in news, EMNLP 2019:  
  https://aclanthology.org/D19-1664/
- Carlebach et al. (2020), diverse viewpoint identification in news:  
  https://aclanthology.org/2020.argmining-1.7/
- Potthast et al. (2018), hyperpartisan vs mainstream comparison:  
  https://aclanthology.org/P18-1022/
