# 04 Emotion Detection

This folder is **Mechanism 3 Emotional Amplification**: whether alternative outlets lean on **anger** and **fear** more than Tagesschau, and where that concentrates.

## Model

A German-language **ELECTRA** model, pre-trained by the German NLP Group and fine-tuned by **Widmann & Wich (2023)** on ~10,000 crowd-coded sentences from German parliamentary speeches and political-party Facebook posts. It is a multi-label classifier over **8 discrete emotions**:

> anger, fear, disgust, sadness, joy, enthusiasm, pride, hope

**Focus emotions for this thesis: anger (primary) and fear.**

## Where the model is run

1. **Full corpus** at document level.
2. **The delegitimization context windows** produced by [`../03_Framing/`](../03_Framing/), this is what links Mechanism 2 and Mechanism 3.

## Topic-aware comparison (key methodological move)

Raw outlet-level emotion scores can pick up topic vocabulary — war and crisis articles score high on anger regardless of tone (Quandt et al., 2020). To mitigate this confound, every comparison is **within topic cluster, outlet vs. Tagesschau, holding topic constant**, using the 18 thematic clusters from [`../02_TopicModeling/`](../02_TopicModeling/).

## Significance and validation

- **Mann–Whitney U** (two-sided) per outlet × geography and outlet × topic-cluster cell, chosen because emotion scores are bounded in [0, 1], right-skewed, and non-normal.
- **Multiple-testing control**: Benjamini–Hochberg FDR at α = 0.05.
- **Effect-size floor**: rank-biserial |r| ≥ 0.10, so trivially small differences at large n do not get flagged as "significant".
- **Sample-size floor**: cells with n < 20 on either side are excluded and grayed in the heatmaps.
- **Qualitative validation**: for the 5 largest positive anger deviations vs. Tagesschau (excluding the Misc cluster, n ≥ 20 both sides), 15 articles per cell were read in German — top-anger of the alt outlet, low-anger of the same alt outlet, and top-anger Tagesschau on the same cluster (75 articles total). The read looked for exclamation density and polemical lexicon (*Lüge*, *Skandal*, *Wahnsinn*, etc.).

## Framing ↔ emotion link

Both **Pearson's r** and **Spearman's ρ** are reported between the per-cluster delegitimization rate (from Mechanism 2) and the per-cluster emotion scores. 

## Aggregation

The unit of analysis is the **outlet × topic-cluster** cell. When summarizing "alternative outlets" as a group, the **macro-mean** (mean of per-outlet means) is used so that larger outlets do not dominate.
