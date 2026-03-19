# Thesis Project: Comparative Topic Modeling of German News Outlets

## What this is
MSc thesis comparing topic coverage across 7 German news outlets (1 mainstream, 6 alternative/right-wing/pro-Russian) using BERTopic. Study period: Aug 2025 – Jan 2026.

## Outlets
| Outlet | Category | Clean CSV | ~Articles |
|--------|----------|-----------|-----------|
| Tagesschau | Mainstream | `tagesschau_clean.csv` | 6,320 |
| NIUS | Right-Populist | `nius_clean.csv` | 3,269 |
| Tichys Einblick | Right-Populist | `tichys_clean.csv` | 2,756 |
| Compact | Right-Wing | `compact_clean.csv` | needs verification |
| Deutschland-Kurier | Right-Wing | `dkurier_clean.csv` | needs verification |
| RT DE | Pro-Russian | `rt_de_clean.csv` | 4,560 |
| Anti-Spiegel | Pro-Russian | `antispiegel_clean.csv` | 565 |

## Project structure
```
data/raw/Alternative Medien/   ← scraped articles (DO NOT modify)
data preprocessing/
  01–08_*.ipynb                ← per-outlet cleaning (numbered by outlet)
  09_Corpus_EDA.ipynb          ← corpus-level EDA for thesis
  *_clean.csv                  ← cleaned outlet CSVs (Date, Title, Text, source)
  overall_df_combined.csv      ← all outlets combined
  figures/                     ← EDA figures (PDF for LaTeX)
  tables/                      ← EDA tables (CSV)
BERTopic/
  bertopic_pipeline.py         ← main modelling pipeline
  bertopic_config.py           ← BERTopicConfig dataclass
  stopwords_de.py              ← curated German stopwords
  Outlet_Comparison.ipynb      ← cross-outlet topic similarity
  outputs/                     ← saved models + topic tables
tex/                           ← thesis LaTeX (git submodule → Overleaf)
```

## Key conventions
- All clean CSVs have columns: `Date, Title, Text, source`
- Dates are normalised to `YYYY-MM-DD` (no timezone)
- Outlet-specific cleaning removes ads/boilerplate unique to each site
- BERTopic pipeline applies additional filtering: `min_tokens=8`, `min_text_chars=50`
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
- Random state: `42` everywhere

## Workflow
1. Raw data → per-outlet cleaning notebooks (01–08)
2. Clean CSVs → EDA notebook (09)
3. Clean CSVs → BERTopic pipeline → per-outlet models
4. Combined corpus → overall topic model (07_OverallTM)
5. Cross-outlet comparison (Outlet_Comparison.ipynb)

## LaTeX submodule
The `tex/` directory is a git submodule pointing to `Thesis-Latex.git` (Overleaf-synced).
Use `/pull-overleaf` to fetch colleagues' changes.

## Python environments
- `.venv/` — general (pandas, numpy, matplotlib, jupyter)
- `.venv312/` — ML stack (BERTopic, sentence-transformers, UMAP, HDBSCAN)
