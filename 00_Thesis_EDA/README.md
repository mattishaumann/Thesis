# Thesis EDA Workspace

This folder is reserved for the exploratory data analysis outputs that are intended to support the written thesis directly.

Keep this folder separate from:

- [`00_Initial EDA/`](../00_Initial%20EDA/), which contains earlier exploratory notebooks and outlet-level draft work
- [`1a_BERTopic/`](../1a_BERTopic/), which contains the actual topic-modeling pipeline and shared BERTopic code

## Recommended Use

Use the notebooks here for the material that is most likely to be cited, exported, or shown in the thesis text:

- `01_Final_Corpus_EDA.ipynb`
  Use this for the thesis-facing overview of the final cleaned analytic corpus.
  This is the main place for:
  - corpus size by outlet
  - corpus size by month
  - publication timeline
  - article-length distributions
  - normalized outlet comparisons that may be shown in Section 3.3

- `02_Preprocessing_Audit.ipynb`
  Use this for the methodological audit trail behind cleaning decisions.
  This is the place for:
  - before/after counts where available
  - duplicate and boilerplate checks
  - outlet-specific removals or adjustments
  - compact tables that justify preprocessing decisions in Section 3.4 or the appendix

## Practical Rule

- If the figure or table should appear in the thesis body, build it here.
- If the notebook is mainly for testing, trial plots, or exploratory BERTopic diagnostics, keep it out of this folder.

## Output Folders

- `outputs/figures/`
  Store thesis-ready figures here.
- `outputs/tables/`
  Store exported thesis tables here.
- `outputs/appendix/`
  Store optional appendix-only diagnostics here.

## Recommended Sequence

1. Inspect the raw corpus at a high level.
2. Perform outlet-specific diagnostics and cleaning in the outlet notebooks.
3. Build the thesis-facing EDA for the final cleaned corpus in `01_Final_Corpus_EDA.ipynb`.
4. Summarize the cleaning logic and before/after effects in `02_Preprocessing_Audit.ipynb`.

This means the main descriptive EDA shown in the thesis should usually be post-cleaning, while the preprocessing audit remains available as methodological justification.
