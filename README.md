# Thesis

Master thesis repository for preprocessing, exploratory topic modeling, and supporting analysis notebooks.

## Repository Layout

- `data preprocessing/`
  Contains the main outlet-level preprocessing and BERTopic notebooks.
- `BERTopic/`
  Shared BERTopic pipeline code, configuration, stopword handling, and saved model outputs.
- `Initial EDA/`
  Early exploratory notebooks and draft analyses.
- `data/`
  Raw, processed, and experimental data folders.
- `code/`
  Additional scripts and analysis code used alongside the notebooks.

## Main Notebook Workflow

The current preprocessing/topic-modeling workflow is centered in `data preprocessing/`:

- `01_RT.ipynb`
- `02_Compact.ipynb`
- `03_Nius.ipynb`
- `04_Tichys_Einblick.ipynb`
- `05_Antispiegel.ipynb`
- `06_Tagesschau.ipynb`
- `08_Deutschlandkurier.ipynb`

These notebooks:

1. clean one outlet-specific corpus,
2. export a cleaned CSV used for downstream combination,
3. run exploratory BERTopic modeling for that outlet,
4. apply final outlier reduction on the selected BERTopic model,
5. save the reduced final BERTopic model to `BERTopic/outputs/`.

The combined notebook is:

- `07_OverallTM.ipynb`

This notebook loads the cleaned outlet CSVs and builds the overall cross-outlet corpus.

## Cleaned Outlet Data

The cleaned corpora currently expected by the overall notebook are stored in `data preprocessing/`:

- `rt_de_clean.csv`
- `compact_clean.csv`
- `nius_clean.csv`
- `tichys_clean.csv`
- `antispiegel_clean.csv`
- `tagesschau_clean.csv`
- `dkurier_clean.csv`

## BERTopic Code

The shared BERTopic components live in:

- `BERTopic/bertopic_pipeline.py`
- `BERTopic/bertopic_config.py`
- `BERTopic/stopwords_de.py`

Saved topic-model outputs are written under `BERTopic/outputs/`.

## Minimal Run Order

If the cleaned corpora or topic models need to be regenerated, the current practical order is:

1. run the outlet notebooks in `data preprocessing/` to refresh cleaned CSVs and outlet-specific BERTopic outputs,
2. rerun `07_OverallTM.ipynb` to build the combined corpus and overall topic model.

## Environment

Install the Python dependencies from:

- `requirements.txt`

Example:

```bash
pip install -r requirements.txt
```

The project is currently notebook-driven, so Jupyter support is required.
