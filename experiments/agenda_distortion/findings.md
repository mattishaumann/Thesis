# Findings: Hypothesis 1 — Agenda Distortion

**Hypothesis**: Alternative media engage in agenda-setting, focusing on a narrower set of topics to steer the conversation.

**Status**: [ ] Not yet run | [ ] In progress | [ ] Supported | [ ] Not supported | [ ] Inconclusive

---

## Literature Grounding

- **Agenda-setting theory**: McCombs & Shaw (1972) — media don't tell people what to think, but what to think *about*. Alt media with a narrower topic focus exert stronger agenda-setting pressure.
- **Topic divergence across media ecosystems**: DiMaggio et al. (2013) — JSD on topic distributions as a measure of ideological/agenda distance between corpora.
- **BERTopic methodology**: Grootendorst (2022) — contextual topic modeling superior to LDA for short, heterogeneous news texts.
- **Class imbalance in corpus comparison**: Bootstrap CIs and normalized prevalence are standard corrections (cite your methods accordingly when writing up).

---

## Results

### KPIs
| Metric | Alt Media | Mainstream | Delta / Score | 95% CI | Interpretation |
|---|---|---|---|---|---|
| Num topics | | | | | |
| Topic diversity | | | | | |
| JSD (prevalence) | | | — | | |
| Avg topic cosine sim | | | — | | |

### Top Over-Represented Topics in Alt Media
| Topic | Top Words | Prevalence (Alt) | Prevalence (Main) | LLM Label | Coherent? |
|---|---|---|---|---|---|
| | | | | | |

### Visual Evidence
- UMAP separation observed: [Y/N — describe]
- Prevalence heatmap pattern: [describe]

---

## Validation
- Parameter sensitivity: [stable/fragile]
- Null model test: [passed/failed]
- Imbalance robustness: [holds/drops when downsampled]

---

## Interpretation
[Fill in after running notebooks 01–04]

Does this support the agenda distortion hypothesis?
What would strengthen or weaken this finding?
What should change before moving to Hypothesis 2 (Delegitimization)?

---

## Open Questions & Next Steps
- [ ]
- [ ]
