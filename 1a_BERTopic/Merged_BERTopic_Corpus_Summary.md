# Merged BERTopic Corpus Summary

This note is the short, thesis-facing explanation of what the merged BERTopic workflow now uses as its basis, why some rows were removed before modeling, and why the final BERTopic-ready subset is slightly smaller than `df_combined.csv`.

## 1. Canonical corpus

- `00_Initial EDA/df_combined.csv` is the canonical cleaned cross-outlet corpus.
- It is created by concatenating the seven outlet-level `*_clean.csv` files from `00_Initial EDA/01_*.ipynb` to `08_*.ipynb`.
- `1a_BERTopic/Merged_BERTopic_All_Outlets.ipynb` now starts from `df_combined.csv`, not from rebuilt raw loaders.
- `row_id` is preserved through the merged assignment workflow and written back out again.

## 2. What was removed before `df_combined`

The point of the outlet notebooks is to do outlet-specific cleaning first, then combine the already cleaned corpora.

### Tagesschau

- Nested JSON article bodies had to be extracted and normalized first.
- HTML/content-block cleanup is part of the Tagesschau cleaning notebook before export to `tagesschau_clean.csv`.

### RT DE

- Standard schema cleanup and deduplication happened in `01_RT.ipynb`.
- No extra special exclusion rule is currently part of the merged explanation beyond that standard cleaning stage.

### NIUS

- `03_Nius.ipynb` explicitly removes rows whose `Categories` field contains `Show`.
- The notebook note says these `Show` entries appear to be advertisements, so they are excluded before `nius_clean.csv` is saved.

### Tichys Einblick

- `04_Tichys_Einblick.ipynb` removes recurring non-editorial promo/format material that surfaced as noise in exploratory BERTopic runs.
- Two explicit groups are excluded before `tichys_clean.csv` is saved:
  - issue-sale / PDF-promo posts
  - `TE-Wecker` roundup / podcast-format posts

### Anti-Spiegel

- `05_Antispiegel.ipynb` removes titles containing `Tacheles #`, described there as podcast advertisement material.
- It also removes recurring podcast announcement pages identified via `Anti-Spiegel-Podcast`, `Den Podcast können Sie hier`, and Spotify/VK announcement patterns.

### Compact

- The earlier Compact-specific ad / affiliate / boilerplate cleanup from `02_Compact.ipynb` stays in place.
- That old Compact cleaning is already baked into `compact_clean.csv`, therefore also into `df_combined.csv`.
- This is the key reason the merged notebook must start from `df_combined.csv`: it preserves the earlier Compact decisions instead of accidentally rebuilding a broader raw corpus.

### Deutschlandkurier

- `08_Deutschlandkurier.ipynb` removes rows with missing text and drops duplicate `Title` + `Text` pairs before saving `dkurier_clean.csv`.

## 3. Why the BERTopic subset is smaller than `df_combined`

Once the cleaned outlet corpora are combined, BERTopic still applies one shared document-preparation step before topic assignment.

That shared preparation step lives in `1a_BERTopic/bertopic_pipeline.py` and currently does:

- light boilerplate cleanup
- whitespace normalization
- `min_text_chars >= 50`
- `min_tokens >= 8`
- duplicate removal on the cleaned `document` field

So the difference is now methodologically clean:

- `df_combined.csv` = cleaned source corpus
- `merged_articles_with_topics.csv` = BERTopic-ready subset that actually received a topic
- `df_combined_with_topics.csv` = full cleaned source corpus plus merged-topic columns, with unmatched rows kept visible

## 4. Current row counts

### Whole corpus

- `df_combined.csv`: `20,440` rows
- BERTopic-ready subset with assigned topics: `20,358` rows
- unmatched after BERTopic preparation: `82` rows

### By outlet

| Outlet | Rows in `df_combined` | Rows with merged topic | Unmatched after BERTopic prep |
|---|---:|---:|---:|
| Tagesschau | 6,320 | 6,272 | 48 |
| RT_de | 4,560 | 4,556 | 4 |
| Antispiegel | 565 | 565 | 0 |
| Tichys_Einblick | 2,756 | 2,753 | 3 |
| Nius | 3,269 | 3,266 | 3 |
| Compact | 1,486 | 1,486 | 0 |
| Deutschlandkurier | 1,484 | 1,460 | 24 |

Interpretation:

- Compact now matches exactly, which is what we wanted.
- The remaining differences are not raw-corpus mismatches anymore.
- They are the result of the shared BERTopic document-preparation filters.

## 5. What the individual outlet models are

The seven loaded models in `Merged_BERTopic_All_Outlets.ipynb` are:

- `tm_ts`
- `tm_rt`
- `tm_as`
- `tm_te`
- `tm_ns`
- `tm_cm`
- `tm_dk`

These are not the old global model from `00_Initial EDA/07_OverallTM.ipynb`.

They are seven outlet-specific BERTopic models trained separately on the cleaned outlet corpora. That is methodologically fine and is in fact the point of the merged-model design:

- each outlet first gets its own topic discovery step
- then those topic spaces are merged with `BERTopic.merge_models(...)`
- then the merged topic space is assigned back to the canonical combined corpus

## 6. The current exports to use

- `00_Initial EDA/df_combined.csv`
  - cleaned source corpus
- `1a_BERTopic/local_outputs/merged_articles_with_topics.csv`
  - BERTopic-ready subset with one topic per modeled article
- `00_Initial EDA/df_combined_with_topics.csv`
  - full cleaned corpus plus topic columns and `topic_match_status`

## 7. One-sentence methods summary

Use this wording if needed:

> Outlet-specific cleaning decisions, including earlier Compact ad/affiliate/boilerplate removal, were preserved by treating `df_combined.csv` as the canonical cleaned corpus; BERTopic was then applied to a filtered, deduplicated modeling subset of that corpus, and topic assignments were merged back to the full corpus via `row_id`.

## 8. Why the topic exports can look different

There are now two different topic-description layers and they should not be confused:

- `1a_BERTopic/local_outputs/merged_topics_overview.csv`
  - canonical topic overview from the saved merged BERTopic model
  - keeps the original topic names such as `1_ukraine_putin_selenskyj_trump`
  - also includes `AssignedCount` from the final article-level corpus
- `1a_BERTopic/local_outputs/merged_topics_overview_refreshed.csv`
  - refreshed descriptive keywords rebuilt from the final assigned article corpus
  - useful when you want more than the originally saved 10 words
  - can shift the top descriptive words quite a lot, even if the topic IDs themselves stay the same

So if the refreshed files look like "different topics", the safer interpretation is:

- topic IDs and article assignments stayed in the same merged-topic space
- but the keyword representation was recomputed from the final assigned corpus, which changes the visible descriptors
