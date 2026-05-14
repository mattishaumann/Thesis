# 00 Data Source

This folder holds the corpus that every later mechanism builds on: the raw outlet exports plus the cleaned version used for modeling.

## Corpus at a glance

- **20,440 German articles** across **7 outlets**, collected from **2025-08-01 to 2026-01-31**.
- Article counts by outlet, grouped by typology:
  - **Mainstream baseline**: Tagesschau (6,320)
  - **Pro-Russian**: RT DE (4,560), Antispiegel (565)
  - **Right-extremist**: Compact (1,486), Deutschlandkurier (1,484)
  - **Right-conservative**: NIUS (3,269), Tichys Einblick (2,756)

## Sources

- **Alternative outlets**: collected via Polisphere exports.
- **Tagesschau**: collected via the Tagesschau API plus an open-source crawler.

## Confidentiality

**The underlying article data is confidential and is therefore not uploaded or provided with this repository.** The folder structure is kept for reference so that the rest of the codebase remains readable.
