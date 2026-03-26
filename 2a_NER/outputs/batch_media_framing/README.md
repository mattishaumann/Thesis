# Media Framing Outputs

This folder is now split into one clean final thesis area and one archive area.

## Final Thesis

Current working folder:
- `thesis_final/`

Subfolders:
- `thesis_final/batch_workflow/`
  Keeps the batch-preparation artifacts and the partial batch-attempt metadata.
  Use this only if you want to reconstruct or inspect the abandoned batch path.
- `thesis_final/normal_api_run/`
  Keeps the actual final full-run files from the normal API workflow.
  This is the main source for the coded thesis results.
- `thesis_final/analysis/`
  Keeps the summary tables and pivot tables derived from the final full run.

Recommended starting points:
- `thesis_final/normal_api_run/media_framing_thesis_sync_results.csv`
- `thesis_final/analysis/media_framing_thesis_outlet_label_summary.csv`
- `thesis_final/analysis/media_framing_thesis_outlet_label_counts_pivot.csv`
- `thesis_final/analysis/media_framing_thesis_outlet_label_share_pivot.csv`

## Archive

Archive folder:
- `archive_old_tests/`

Subfolders:
- `archive_old_tests/legacy_batches/`
  Old pilot and early full-batch test files.
- `archive_old_tests/tagesschau_review/`
  Earlier Tagesschau 50-article inspection artifacts.
- `archive_old_tests/old_sync_analysis_names/`
  Older analysis outputs with the previous `sync_...` naming.

## Notebook Mapping

Notebooks now write to:
- `Media_Framing_Batch_API_Preparation.ipynb` -> `thesis_final/batch_workflow/`
- `Media_Framing_Normal_API_Run.ipynb` -> `thesis_final/normal_api_run/` and `thesis_final/analysis/`
- `Media_Framing_Results_Analysis.ipynb` -> `thesis_final/analysis/`
