"""One-off builder for split NER exploration notebooks. Not imported by analysis code."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent


def cell_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(keepends=True)}


def cell_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def write_nb(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# --- Patterns: entity composition, heatmap, similarity ---
patterns_cells = [
    cell_md(
        """
        # NER exploration: outlet-level patterns

        **What this notebook does:** stacked bar of entity *labels* by outlet, heatmap of top entities, cosine similarity of outlets by entity distribution.

        **Prerequisite:** a dataframe `mentions` with columns at least `source`, `label`, `entity`, `row_id` (as produced in `NER_all.ipynb` after exploding entities).

        Run the setup cell below. It will use `mentions` from memory, or load `2a_NER/outputs/mentions.parquet` if you export it from `NER_all`.
        """
    ),
    cell_code(
        """
        from __future__ import annotations

        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display
        from sklearn.metrics.pairwise import cosine_similarity

        NOTEBOOK_DIR = Path.cwd().resolve()
        if NOTEBOOK_DIR.name == "2a_NER":
            PROJECT_ROOT = NOTEBOOK_DIR.parent
        else:
            PROJECT_ROOT = NOTEBOOK_DIR

        def ensure_mentions() -> pd.DataFrame:
            global mentions
            if "mentions" in globals() and isinstance(mentions, pd.DataFrame) and not mentions.empty:
                return mentions
            parquet_path = NOTEBOOK_DIR / "outputs" / "mentions.parquet"
            if parquet_path.exists():
                mentions = pd.read_parquet(parquet_path)
                print(f"Loaded mentions from {parquet_path}")
                return mentions
            raise FileNotFoundError(
                "Define `mentions` (run NER_all entity cells) or save "
                f"{parquet_path} via mentions.to_parquet(...)"
            )

        mentions = ensure_mentions()
        articles_per_outlet = mentions.groupby("source")["row_id"].nunique().rename("n_articles")
        outlet_order = sorted(mentions["source"].dropna().unique().tolist())
        """
    ),
    cell_md("### Entity label mix by outlet"),
    cell_code(
        """
        label_summary = (
            mentions.groupby(["source", "label"])
            .size()
            .rename("mentions")
            .reset_index()
            .merge(articles_per_outlet.reset_index(), on="source", how="left")
        )

        label_summary["mentions_per_100_articles"] = (
            label_summary["mentions"] / label_summary["n_articles"] * 100
        )

        plot_data = (
            label_summary.pivot(index="source", columns="label", values="mentions_per_100_articles")
            .fillna(0)
            .loc[outlet_order]
        )

        ax = plot_data.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
        ax.set_title("Entity type composition by outlet")
        ax.set_xlabel("Outlet")
        ax.set_ylabel("Mentions per 100 articles")
        ax.legend(title="Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        """
    ),
    cell_md("### Top entities: share of articles per outlet"),
    cell_code(
        """
        TOP_N = 30

        top_global_entities = (
            mentions.groupby("entity")["row_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(TOP_N)
            .index
        )

        heatmap_df = (
            mentions[mentions["entity"].isin(top_global_entities)]
            .groupby(["entity", "source"])["row_id"]
            .nunique()
            .rename("documents")
            .reset_index()
            .merge(articles_per_outlet.reset_index(), on="source", how="left")
        )

        heatmap_df["doc_share_pct"] = heatmap_df["documents"] / heatmap_df["n_articles"] * 100

        heatmap_matrix = (
            heatmap_df.pivot(index="entity", columns="source", values="doc_share_pct")
            .fillna(0)
            .reindex(columns=outlet_order)
        )

        heatmap_matrix = heatmap_matrix.loc[heatmap_matrix.mean(axis=1).sort_values(ascending=False).index]

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_matrix, cmap="YlOrRd", linewidths=0.3)
        plt.title("Top entities across outlets (share of articles mentioning the entity, %)")
        plt.xlabel("Outlet")
        plt.ylabel("Entity")
        plt.tight_layout()
        plt.show()
        """
    ),
    cell_md("### Outlet similarity (cosine) from entity distributions"),
    cell_code(
        """
        similarity_df = (
            mentions.groupby(["source", "entity"])["row_id"]
            .nunique()
            .rename("documents")
            .reset_index()
            .merge(articles_per_outlet.reset_index(), on="source", how="left")
        )

        similarity_df["doc_share"] = similarity_df["documents"] / similarity_df["n_articles"]

        entity_matrix = (
            similarity_df.pivot(index="source", columns="entity", values="doc_share")
            .fillna(0)
            .reindex(outlet_order)
        )

        cosine_sim = pd.DataFrame(
            cosine_similarity(entity_matrix),
            index=entity_matrix.index,
            columns=entity_matrix.index,
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(cosine_sim, annot=True, cmap="Blues", vmin=0, vmax=1)
        plt.title("Outlet similarity based on entity distributions")
        plt.tight_layout()
        plt.show()
        """
    ),
]

# --- GPT pipeline: regex + sample + API ---
gpt_cells = [
    cell_md(
        """
        # GPT framing pipeline: regex contexts → sample → API

        **Flow:** load corpus → build `MASTER_PATTERN` → `media_context_df` / `media_article_df` → sample hits → (optional) export → OpenAI Responses API → `results_df`.

        Requires `OPENAI_API_KEY` in `.env` or environment for API cells.
        """
    ),
    cell_code(
        """
        import re
        from pathlib import Path

        import pandas as pd
        from IPython.display import display

        NOTEBOOK_DIR = Path.cwd().resolve()
        if NOTEBOOK_DIR.name == "2a_NER":
            PROJECT_ROOT = NOTEBOOK_DIR.parent
        else:
            PROJECT_ROOT = NOTEBOOK_DIR

        def load_combined_corpus() -> pd.DataFrame:
            if "df" in globals():
                return df
            candidates = [
                PROJECT_ROOT / "data preprocessing" / "overall_df_combined.csv",
                PROJECT_ROOT / "data" / "raw" / "df_combined.csv",
            ]
            for path in candidates:
                if path.exists():
                    print(f"Loaded: {path}")
                    return pd.read_csv(path)
            raise FileNotFoundError(
                "Could not find corpus CSV. Tried:\\n  - " + "\\n  - ".join(str(p) for p in candidates)
            )

        df = load_combined_corpus()

        search_df = df[["row_id", "source", "Title", "Text"]].copy()
        search_df = search_df[
            search_df["source"].fillna("").astype(str).str.casefold() != "tagesschau"
        ].copy()

        MEDIA_PATTERNS = [
            r"\\bard\\b",
            r"\\bzdf\\b",
            r"\\b(?:der\\s+)?spiegel\\b",
            r"\\btagesschau(?:\\.de)?\\b",
            r"\\btagesschau24\\b",
            r"\\breuters\\b",
            r"\\bjan\\s+böhmermann\\b",
            r"\\bndr(?:\\s+info)?\\b",
            r"\\bnorddeutscher\\s+rundfunk\\b",
            r"\\bbild-zeitung\\b|\\bdie\\s+bild\\b",
            r"\\bpolitico\\b",
            r"\\bswr\\b",
            r"\\bsüdwestrundfunk\\b",
            r"\\bdpa\\b",
            r"\\bwdr\\b",
            r"\\bwestdeutscher\\s+rundfunk\\b",
            r"\\bfaz\\b",
            r"\\bfrankfurter\\s+allgemeine(?:n)?\\s+zeitung\\b",
            r"\\börr\\b",
            r"\\bhandelsblatt\\b",
            r"\\bdeutschlandfunk\\b",
            r"\\btagesspiegel\\b",
            r"\\btaz\\b",
            r"\\bbr\\b",
            r"\\bbayerischer\\s+rundfunk\\b",
            r"\\bberliner\\s+zeitung\\b",
            r"\\brbb\\b",
            r"\\brundfunk\\s+berlin-brandenburg\\b",
            r"\\bcorrectiv\\b",
            r"\\bsz\\b",
            r"\\bsüddeutsch(?:e|en)\\s+zeitung\\b",
            r"\\bstern\\b",
            r"\\bdas\\s+(?-i:Erste)\\b"
            r"\\bthe\\s+european\\b",
            r"\\brtl\\b",
            r"\\bmdr\\b",
            r"\\bmitteldeutscher\\s+rundfunk\\b",
            r"\\brnd\\b",
            r"\\bredaktionsnetzwerk\\s+deutschland\\b",
            r"\\bcaren\\s+miosga\\b",
            r"\\brheinisch(?:e|en)\\s+post\\b",
            r"\\b(?:markus\\s+)?lanz\\b",
            r"\\beuronews\\b",
        ]

        MASTER_PATTERN = re.compile(
            "|".join(f"(?:{p})" for p in MEDIA_PATTERNS),
            flags=re.IGNORECASE,
        )


        def normalize_text(text):
            text = "" if pd.isna(text) else str(text)
            text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
            text = re.sub(r"\\s+", " ", text)
            return text.strip()


        def combine_title_text(title, text):
            title = normalize_text(title)
            text = normalize_text(text)
            if title and text and not re.search(r"[.!?…:;]$", title):
                title = title + "."
            return f"{title} {text}".strip()


        def split_sentences(text):
            text = normalize_text(text)
            if not text:
                return []
            sentences = re.split(r"(?<=[.!?…])\\s+(?=[A-ZÄÖÜ0-9\\\"'“„(])", text)
            return [s.strip() for s in sentences if s and s.strip()]


        def extract_context_windows(text, pattern, window=1):
            sentences = split_sentences(text)
            if not sentences:
                return []
            matched_indices = [i for i, sentence in enumerate(sentences) if pattern.search(sentence)]
            if not matched_indices:
                return []
            merged_windows = []
            for idx in matched_indices:
                start = max(0, idx - window)
                end = min(len(sentences), idx + window + 1)
                if merged_windows and start <= merged_windows[-1][1]:
                    merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], end))
                else:
                    merged_windows.append((start, end))
            return [" ".join(sentences[s:e]) for s, e in merged_windows]


        search_df["combined_text"] = [
            combine_title_text(title, text) for title, text in zip(search_df["Title"], search_df["Text"])
        ]

        EXTRA_KEYWORDS = [
            "Mainstreammedien", "Staatsmedien", "Staatsfunk", "Qualitätsmedien", "Staatssender",
            "Lügenpresse", "Haltungsjournalisten", "Gleichschaltung", "Mainstreampresse", "Altmedien",
            "Systemmedien", "Qualitätsjournalismus", "Alternativmedien", "Gesternmedien", "Haltungsmedien",
            "Westmedien", "Regierungsmedien", "Linkspresse", "Qualitätspresse", "Haltungsjournalismus",
            "Propagandamedien", "Propagandasender", "Propagandamaschine", "Medienpropaganda", "Staatsrundfunk",
        ]
        EXTRA_KEYWORD_PATTERNS = [rf"\\b{re.escape(k)}\\b" for k in EXTRA_KEYWORDS]

        MASTER_PATTERN = re.compile(
            "|".join(f"(?:{p})" for p in MEDIA_PATTERNS + EXTRA_KEYWORD_PATTERNS),
            flags=re.IGNORECASE,
        )

        candidate_df = search_df[search_df["combined_text"].str.contains(MASTER_PATTERN, na=False)].copy()
        print(f"Candidate articles (excluding Tagesschau): {len(candidate_df):,}")

        rows = []
        for row in candidate_df.itertuples(index=False):
            for context in extract_context_windows(row.combined_text, MASTER_PATTERN, window=1):
                rows.append(
                    {
                        "row_id": row.row_id,
                        "source": row.source,
                        "Title": row.Title,
                        "Text": row.Text,
                        "context_window": context,
                    }
                )

        media_context_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
        print(f"Context windows: {len(media_context_df):,}")
        display(media_context_df.head())

        media_article_df = (
            media_context_df.groupby(["row_id", "source", "Title", "Text"], as_index=False)
            .agg(context_window=("context_window", "\\n\\n---\\n\\n".join))
        )
        print(f"Unique articles in filtered set: {len(media_article_df):,}")
        display(media_article_df.head())
        """
    ),
    cell_md("### Hit term × outlet (regex matches in context windows)"),
    cell_code(
        """
        if "MASTER_PATTERN" not in globals():
            raise NameError("Run the previous cell first.")

        hit_rows = []
        for row in media_context_df[["row_id", "source", "context_window"]].itertuples(index=False):
            text = "" if pd.isna(row.context_window) else str(row.context_window)
            for m in MASTER_PATTERN.finditer(text):
                hit_rows.append(
                    {
                        "row_id": row.row_id,
                        "source": row.source,
                        "hit": m.group(0).strip().casefold(),
                    }
                )

        hits_df = pd.DataFrame(hit_rows)

        hit_pivot = hits_df.groupby(["hit", "source"]).size().unstack(fill_value=0)
        ordered_cols = [s for s in outlet_order if s in hit_pivot.columns] + [
            c for c in hit_pivot.columns if c not in outlet_order
        ]
        hit_pivot = hit_pivot.reindex(columns=ordered_cols)
        hit_pivot["Total_hits"] = hit_pivot.sum(axis=1)
        hit_pivot = hit_pivot.sort_values("Total_hits", ascending=False)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", None)

        article_pivot = (
            hits_df.drop_duplicates(["row_id", "source", "hit"])
            .groupby(["hit", "source"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=ordered_cols, fill_value=0)
        )

        print(f"All matched terms (raw hits): {len(hit_pivot):,}")
        display(hit_pivot)
        print(f"All matched terms (unique articles): {len(article_pivot):,}")
        display(article_pivot)
        article_pivot["Total_articles"] = article_pivot.sum(axis=1)
        article_pivot = article_pivot.sort_values("Total_articles", ascending=False)
        display(hit_pivot.head(100))
        display(article_pivot.head(100))
        """
    ),
]

# Append GPT sample + export + API from original — read from file to avoid huge string here
import json as _json

_src = _json.loads((ROOT / "NER_exploration_text_REFERENCE.ipynb").read_text(encoding="utf-8"))
_by_id = {"".join(c.get("source", []))[:80]: c for c in _src["cells"]}

def src_cells_containing(substr: str) -> str:
    for c in _src["cells"]:
        s = "".join(c.get("source", []))
        if substr in s:
            return s
    raise KeyError(substr)

# Cells 7,8,9,10,11 from reference
for key in ["SAMPLE_N_PER_OUTLET = 25", 'output_path = Path("2a_NER/outputs', 'MODEL_NAME = "gpt-5-mini"', "api_key, api_key_source", "results_df.info()"]:
    gpt_cells.append(cell_code(src_cells_containing(key)))

# --- Analysis ---
analysis_cells = [
    cell_md(
        """
        # Framing analysis: taxonomy + outlets

        Loads the GPT results CSV (with `main_role`, `subcategory`) and summarizes distributions, outlet comparisons, and charts.

        **Standalone:** run after `NER_framing_GPT_pipeline.ipynb` or ensure `2a_NER/outputs/media_framing_gpt5mini_entity_context_results_v2.csv` exists.
        """
    ),
    cell_code(
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from matplotlib.colors import TwoSlopeNorm

        NOTEBOOK_DIR = Path.cwd().resolve()
        if NOTEBOOK_DIR.name == "2a_NER":
            PROJECT_ROOT = NOTEBOOK_DIR.parent
        else:
            PROJECT_ROOT = NOTEBOOK_DIR

        RESULTS_ANALYSIS_PATH = NOTEBOOK_DIR / "outputs" / "media_framing_gpt5mini_entity_context_results_v2.csv"

        if "results_df" not in globals() or results_df.empty:
            if not RESULTS_ANALYSIS_PATH.exists():
                raise FileNotFoundError(f"Results file not found: {RESULTS_ANALYSIS_PATH}")
            results_df = pd.read_csv(RESULTS_ANALYSIS_PATH)

        analysis_df = results_df.copy()
        required_cols = ["row_id", "source", "hit_text", "main_role", "subcategory", "evidence"]
        missing_cols = [col for col in required_cols if col not in analysis_df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in results_df: {missing_cols}")

        analysis_df["source"] = analysis_df["source"].fillna("unknown").astype(str).str.strip()
        analysis_df["main_role"] = analysis_df["main_role"].fillna("missing").astype(str).str.strip()
        analysis_df["subcategory"] = analysis_df["subcategory"].fillna("missing").astype(str).str.strip()
        analysis_df["evidence"] = analysis_df["evidence"].fillna("").astype(str).str.strip()

        if "sampled_articles_df" in globals():
            sampled_articles_per_source = (
                sampled_articles_df.groupby("source")["row_id"].nunique().rename("sampled_articles")
            )
        else:
            sampled_articles_per_source = (
                analysis_df.groupby("source")["row_id"].nunique().rename("sampled_articles")
            )

        print(f"GPT-coded hit rows: {len(analysis_df):,}")
        print(f"Unique articles in results: {analysis_df['row_id'].nunique():,}")
        display(analysis_df[["row_id", "source", "hit_text", "main_role", "subcategory", "evidence"]].head(10))
        """
    ),
    cell_md("### Overall taxonomy distribution"),
    cell_code(src_cells_containing('overall_role_summary = (')),
    cell_md("### Outlet-level comparison"),
    cell_code(src_cells_containing("outlet_summary_df = (")),
    cell_md("### Visualizations"),
    cell_code(src_cells_containing("role_plot_df = outlet_role_shares")),
    cell_md("### Optional: manual inspection sample"),
    cell_code(src_cells_containing('RANDOM_STATE = 42\nN_PER_OUTLET = 6')),
]

# Fix: GPT pivot cell uses outlet_order — must define in gpt notebook after media dfs
_gpt_insert = cell_code(
    """
    # Outlet column order for pivots (falls back to sorted sources in this corpus)
    outlet_order = sorted(media_context_df["source"].dropna().unique().tolist())
    """
)
# Insert after media_article_df cell (index 1 is first code block) — insert at position 3 before pivot md
gpt_cells.insert(3, _gpt_insert)

write_nb("NER_exploration_patterns.ipynb", patterns_cells)
write_nb("NER_framing_GPT_pipeline.ipynb", gpt_cells)
write_nb("NER_framing_analysis.ipynb", analysis_cells)

# Index notebook
index_cells = [
    cell_md(
        """
        # NER / framing exploration (split notebooks)

        The **full monolithic copy** of the original workflow is saved as **`NER_exploration_text_REFERENCE.ipynb`** (unchanged snapshot).

        Use these focused notebooks:

        | Notebook | Content |
        |----------|---------|
        | **`NER_exploration_patterns.ipynb`** | Entity label mix, top-entity heatmap, outlet similarity (needs `mentions` from `NER_all` or `outputs/mentions.parquet`). |
        | **`NER_framing_GPT_pipeline.ipynb`** | Regex → context windows → GPT sampling → OpenAI API → results CSV. |
        | **`NER_framing_analysis.ipynb`** | Load results CSV; taxonomy tables, outlet comparison, plots. |

        Re-run or clear outputs as needed; paths assume the repo root or `2a_NER/` as the working directory.
        """
    )
]
write_nb("NER_exploration_text.ipynb", index_cells)

print("Wrote: NER_exploration_patterns.ipynb, NER_framing_GPT_pipeline.ipynb, NER_framing_analysis.ipynb, NER_exploration_text.ipynb")
