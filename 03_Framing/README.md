# 03 Delegitimization

This folder is **Mechanism 2 Delegitimization**: measuring *how* each outlet frames mainstream media when it mentions it.


## Two-step pipeline

### 1. Filtering: find mainstream-media mentions

Lives in [`01_NER_Filtering/`](01_NER_Filtering/).

Mainstream-media mentions are identified via **NER** (manually reviewed) plus a **hand-curated regex** for terms NER misses (`Mainstreammedien`, `Staatsmedien`, `Lügenpresse`, etc.). The final filter contains **67 terms** (42 named entities + 25 collective/pejorative terms, a full list in Appendix C of the thesis). Tagesschau self-references are stripped.

For each hit, a **3-sentence context window** is extracted (previous + hit + next sentence); overlapping windows are merged.

**Output: 11,975 mentions across 7,204 articles.**

### 2. Frame classification: LLM coding

Lives in [`02_Framing_Classification_Run/`](02_Framing_Classification_Run/).

The codebook is based on Gravesteijn et al. (2024) and subsequently adapted and contains **4 bias frames + neutral + irrelevant**:

- agenda / allegiance bias
- distortion / manipulation bias
- disinformation / falsehood bias
- failure / incompetence bias

The prompt is 2-shot per class, with a per-class explanation, indicator terms, an example sentence, and a full corpus-drawn example all three annotators agreed on. It returns `{label, evidence_span}`. Final prompts are in Appendix D.1 (EN) / D.2 (DE) of the thesis.

Models tested: GPT-5.4, GPT-5.4 mini, GPT-5 mini. **Final choice: GPT-5.4 mini** — tied on accuracy with GPT-5 mini at **67.3%**, picked because it is the newer model.

## Validation

Two human-labeled samples were used: (i) a random sample n=110 proportional to outlet, and (ii) a stratified sample n=100 across bias classes, because neutral otherwise dominates. Three annotators independently labeled each sample and ground truth was taken as the majority vote. Reported metrics: accuracy, macro-F1, and Cohen's κ, both for the classifier and for inter-annotator agreement. Error analysis fed back into prompt refinement — the key fix was distinguishing outlet-voice framing from outlets quoting another actor's framing.

## Analysis

Lives in [`03_Framing_Analysis/`](03_Framing_Analysis/).

The headline metric is the **delegitimization rate**: the share of mentions classified into any of the 4 bias frames (i.e. not neutral and not irrelevant) per outlet. A secondary view is the frame distribution within each outlet.

## Other contents

- [`outputs/`](outputs/): classification outputs, intermediate tables, and figures.
- [`utils/`](utils/): shared helper modules used by the notebooks.
