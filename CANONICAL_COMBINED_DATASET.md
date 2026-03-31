# Canonical Combined Dataset

This repository should treat [df_combined.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/df_combined.csv) as the single source of truth for the thesis article corpus.

The canonical reference file is:

- [df_combined.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/df_combined.csv)

The reproducible builder is:

- [00_Build_df_combined.ipynb](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/00_Build_df_combined.ipynb)

## What It Contains

- `20,440` articles
- Columns: `Date`, `Title`, `Text`, `source`, `row_id`
- Stable article identifier: `row_id`

Current outlet counts in the canonical file:

- `Tagesschau`: `6320`
- `RT_de`: `4560`
- `Nius`: `3269`
- `Tichys_Einblick`: `2756`
- `Compact`: `1486`
- `Deutschlandkurier`: `1484`
- `Antispiegel`: `565`

## How It Was Built

The canonical combined file is rebuilt from the cleaned outlet CSV exports in [00_Initial EDA](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA):

- [antispiegel_clean.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/antispiegel_clean.csv)
- [compact_clean.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/compact_clean.csv)
- [nius_clean.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/nius_clean.csv)
- [rt_de_clean.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/rt_de_clean.csv)
- [tichys_clean.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/tichys_clean.csv)
- [dkurier_clean.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/dkurier_clean.csv)
- [tagesschau_clean.csv](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/tagesschau_clean.csv)

Exact build order:

1. `Antispiegel`
2. `Compact`
3. `Nius`
4. `RT_de`
5. `Tichys_Einblick`
6. `Deutschlandkurier`
7. `Tagesschau`

Exact transformations:

1. Read each clean CSV and keep only `Date`, `Title`, `Text`, `source`.
2. Parse `Date` with `pd.to_datetime(..., errors="coerce", utc=True)`.
3. For `Tagesschau`, convert timestamps to `Europe/Berlin` before dropping timezone info.
4. Normalize dates to calendar days with `.dt.tz_localize(None).dt.normalize()`.
5. Concatenate the outlet frames in the order above.
6. Reset the combined index.
7. Create `row_id = index + 1`.

This mirrors the historical combination logic preserved in [07_OverallTM.ipynb](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/07_OverallTM.ipynb#L332) and is now exposed as the active rerunnable build notebook in [00_Build_df_combined.ipynb](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/00_Build_df_combined.ipynb).

## Validation

Validate by running [00_Build_df_combined.ipynb](/Users/MattisHaumann/Dev/Thesis/00_Initial%20EDA/00_Build_df_combined.ipynb). It rebuilds the corpus, compares it to the existing canonical file, and raises if any column differs.

## Rules Going Forward

- BERTopic outlet models should be trained from source-specific subsets of the canonical combined dataframe.
- Merged topic assignment tables should map back to the canonical corpus by `row_id`.
- BERTopic should use the canonical combined corpus.
- NER and framing keep their existing hardcoded combined-dataset paths for now.
- Do not commit another combined-corpus CSV to GitHub.
- `07_OverallTM.ipynb` should be treated as legacy provenance, not the active corpus build step.
