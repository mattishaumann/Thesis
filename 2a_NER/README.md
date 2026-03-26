# 2a_NER Notebook Guide

This folder keeps the current thesis-facing media-framing workflow at the top level.

## Current notebooks

- `Media_Framing_Batch_API_Preparation.ipynb`
  Prepares the final thesis dataset for the OpenAI Batch API, including validation, manifest writing, splitting, and batch-result parsing.
- `Media_Framing_Normal_API_Run.ipynb`
  Runs the same final thesis dataset through the normal OpenAI API with the same extraction logic and result schema.
- `Media_Framing_Results_Analysis.ipynb`
  Reads finished result files and produces the outlet/category summary tables and pivot tables for analysis.

## Core inputs and helpers

- `df_combined.csv`
  Main article dataset used by the notebooks.
- `framing_codebook_prompt.txt`
  Prompt used for framing classification.
- `media_framing_batch_utils.py`
  Shared helper functions used by the media-framing notebooks.

## Archived notebooks

Older exploratory notebooks were moved to `archive_notebooks/` to keep this folder easier to navigate.

- `archive_notebooks/exploration/`
  Older media-framing exploration notebook kept as reference.
- `archive_notebooks/legacy_ner/`
  Earlier NER notebooks kept for reproducibility/reference.
- `archive_notebooks/legacy_framing/`
  Older framing analysis notebook kept for reference.
