# Thesis Codebase Overview

This repository contains the code behind a Master's thesis that compares **six German alternative news outlets** against **Tagesschau** (the public-broadcaster mainstream baseline) along three mechanisms associated with delegitimizing or polarizing media discourse.

## The three mechanisms

The analysis is organized around three mechanisms. Each one has its own top-level folder for the analysis, on top of a shared corpus and a shared topic model.

1. **Agenda Divergence**: which topics each outlet covers compared to Tagesschau. Uses per-outlet BERTopic models that are merged into a single shared topic space, then compared with Jensen–Shannon divergence and Shannon entropy. → [`02_TopicModeling/`](02_TopicModeling/)
2. **Delegitimization**: *how* each outlet frames mainstream media when it mentions it. Mainstream-media mentions are extracted with NER plus a hand-curated regex of pejorative terms, then frame-classified with a 2-shot LLM prompt over four bias frames (plus neutral / irrelevant). → [`03_Framing/`](03_Framing/)
3. **Emotional Amplification**: whether alternative outlets lean on **anger** and **fear** more than Tagesschau, and where that concentrates. Uses a fine-tuned German ELECTRA model (Widmann & Wich, 2023) over 8 discrete emotions, with comparisons made *within* topic cluster to avoid topic-vocabulary confounds. → [`04_EmotionDetection/`](04_EmotionDetection/)

## Folder structure

| Folder | What it contains |
| --- | --- |
| [`00_DataSource/`](00_DataSource/) | Raw outlet exports and cleaned per-outlet CSVs. 20,440 articles across 7 outlets, 2025-08-01 to 2026-01-31. **Data is confidential and not uploaded — see folder README.** |
| [`01_EDA_TopicModeling_perOutlet/`](01_EDA_TopicModeling_perOutlet/) | Per-outlet exploratory analysis, outlet-specific cleaning, and the individual BERTopic runs (one model per outlet). Foundation for Mechanism 1. |
| [`02_TopicModeling/`](02_TopicModeling/) | **Mechanism 1: Agenda Divergence.** Merge of the per-outlet models into 72 shared topics (manually grouped into 18 thematic clusters), plus the JSD / entropy analysis. |
| [`03_Framing/`](03_Framing/) | **Mechanism 2: Delegitimization.** NER + regex filter for mainstream-media mentions → 3-sentence context windows → LLM frame coding → delegitimization-rate analysis. |
| [`04_EmotionDetection/`](04_EmotionDetection/) | **Mechanism 3: Emotional Amplification.** GELECTRA emotion classification, run both on the full corpus and on the Mechanism 2 context windows; significance testing and the framing ↔ emotion link. |

Each mechanism folder has its own `README.md` with the methodological detail (data flow, model choices, validation, thresholds). 


## A note on reproducibility

The corpus underlying every analysis here is **confidential** and is therefore not included in this repository. The code is uploaded for **transparency**, so that the methodology can be inspected, not to be re-run end-to-end. Dependencies are listed in [`requirements.txt`](requirements.txt) for reference.
