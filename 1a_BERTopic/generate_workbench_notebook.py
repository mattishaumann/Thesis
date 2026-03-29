from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path(__file__).resolve().parent / "Merged_Topic_Modeling_Workbench.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # Merged Topic Modeling Workbench

        This notebook is the clean working notebook for the merged topic-modeling branch.

        It answers the current work items in order:
        1. Load the saved merged BERTopic model.
        2. Load or rebuild the article-level dataframe with one matched topic per article.
        3. Inspect and save the merged article-topic dataframe.
        4. Decide how topic labels should be handled.
        5. Frame H1 methodologically.
        6. Build topic-prevalence comparison charts against `Tagesschau`.

        Default behavior is optimized for iteration speed: if the saved merged article-topic export already exists, the notebook loads it instead of recomputing embeddings and UMAP.
        """
    ),
    code(
        """
        import importlib
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import display

        # Fast iteration by default: only rebuild the article-topic export if you really changed the model.
        REBUILD_ARTICLE_TOPIC_EXPORT = False
        OUTLET_TO_COMPARE = "rt"
        TOP_N_DIFFERENCE_TOPICS = 15
        SAVE_COMPARISON_FIGURES = False

        PROJECT_ROOT = Path.cwd().resolve()
        while not (PROJECT_ROOT / ".git").exists():
            if PROJECT_ROOT.parent == PROJECT_ROOT:
                raise FileNotFoundError("Could not find the repository root.")
            PROJECT_ROOT = PROJECT_ROOT.parent

        MODULE_ROOT = PROJECT_ROOT / "1a_BERTopic"
        if str(MODULE_ROOT) not in sys.path:
            sys.path.insert(0, str(MODULE_ROOT))

        import merged_outlets_analysis as moa
        moa = importlib.reload(moa)

        pd.set_option("display.max_colwidth", 120)
        pd.set_option("display.max_columns", 30)
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.dpi"] = 140

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Local outputs: {moa.get_local_output_dir(PROJECT_ROOT)}")
        print(f"Rebuild article-topic export: {REBUILD_ARTICLE_TOPIC_EXPORT}")
        """
    ),
    md(
        """
        ## 1. Load The Saved Merged Model

        This gives us the authoritative merged-topic definitions and the `74` substantive merged topics (`75` rows including the outlier topic `-1`).
        """
    ),
    code(
        """
        merged_model, merged_model_path, embedding_reference = moa.load_saved_merged_model(PROJECT_ROOT)
        merged_topic_info = moa.enrich_topic_info_with_display(merged_model.get_topic_info())
        topic_name_template_path = moa.ensure_topic_name_overrides_template(PROJECT_ROOT, merged_topic_info)
        topic_name_overrides = moa.load_topic_name_overrides(PROJECT_ROOT)
        merged_topic_info_named = moa.apply_topic_name_overrides(merged_topic_info, topic_name_overrides)

        non_outlier_topic_count = int(
            merged_topic_info_named.loc[merged_topic_info_named["Topic"] != -1, "Topic"].nunique()
        )

        print(f"Merged model path: {merged_model_path}")
        print(f"Embedding reference: {embedding_reference}")
        print(f"Merged topic rows including outlier: {len(merged_topic_info_named)}")
        print(f"Merged substantive topics excluding outlier: {non_outlier_topic_count}")
        print(f"Topic naming template: {topic_name_template_path}")

        display(
            merged_topic_info_named.loc[
                merged_topic_info_named["Topic"] != -1,
                ["DisplayTopic", "Topic", "topic_label", "TopicNameClean", "Count"],
            ].head(15)
        )
        """
    ),
    md(
        """
        ## 2. Load Or Rebuild The Article-Topic Dataframe

        This is the dataframe you asked for: one row per prepared article, with a matched merged topic and 2D UMAP coordinates.

        The notebook first tries to load the saved export from `local_outputs/merged_articles_with_topics.csv`. If that file is missing, or if you set `REBUILD_ARTICLE_TOPIC_EXPORT = True`, it rebuilds the dataframe from the saved merged model.
        """
    ),
    code(
        """
        article_topic_df = None if REBUILD_ARTICLE_TOPIC_EXPORT else moa.load_exported_article_topic_dataset(PROJECT_ROOT)

        if article_topic_df is None:
            # Rebuild only when needed because this step recomputes embeddings and UMAP for all 20,455 articles.
            prepared_by_outlet = moa.load_all_prepared_documents(PROJECT_ROOT)
            combined_prepared = moa.combine_prepared_documents(prepared_by_outlet)
            merged_articles, merged_topic_info, _ = moa.build_merged_article_frame(
                merged_model,
                combined_prepared,
            )
            topic_name_overrides = moa.load_topic_name_overrides(PROJECT_ROOT)
            article_topic_df = moa.build_article_topic_dataset(
                merged_articles,
                merged_topic_info,
                topic_name_overrides=topic_name_overrides,
            )
            article_topic_exports = moa.export_article_topic_dataset(PROJECT_ROOT, article_topic_df)
        else:
            article_topic_exports = {
                "csv": moa.get_local_output_dir(PROJECT_ROOT) / f"{moa.MERGED_ARTICLES_BASENAME}.csv"
            }

        print(f"Article-topic rows: {len(article_topic_df):,}")
        print(f"Distinct outlets: {article_topic_df['outlet_label'].nunique()}")
        print(
            "Distinct merged topics excluding outlier: "
            f"{article_topic_df.loc[article_topic_df['merged_topic'] != -1, 'merged_topic'].nunique()}"
        )
        print("Saved article-topic exports:")
        for kind, path in article_topic_exports.items():
            print(f"  {kind}: {path}")
        """
    ),
    code(
        """
        # This is the main working dataframe for downstream analysis.
        article_topic_preview = article_topic_df[
            [
                "outlet_label",
                "Title",
                "Date",
                "merged_topic",
                "merged_display_topic",
                "topic_label",
                "URL",
            ]
        ].head(10)
        display(article_topic_preview)

        print("Columns available in article_topic_df:")
        print(article_topic_df.columns.tolist())
        """
    ),
    md(
        """
        ## 3. Topic Naming Strategy

        **Recommendation:** keep the keyword-based labels while you are still iterating on the merged model, coverage logic, and prevalence comparisons. Add short manual topic names only when the topic structure is stable enough for thesis figures and prose.

        Why:
        - Keyword labels are transparent during model debugging.
        - Hand-written names are better for charts and LaTeX, but they become maintenance work if the merged-topic map changes again.
        - The best compromise is to keep both: use keywords as the machine-readable base layer and optional manual names as a presentation layer.

        The CSV below is the presentation-layer file. Fill the `manual_topic_name` column when you are ready. Then rerun the notebook with `REBUILD_ARTICLE_TOPIC_EXPORT = True` once if you want those names propagated into the saved article-topic dataframe.
        """
    ),
    code(
        """
        topic_name_reference = pd.read_csv(topic_name_template_path)
        display(topic_name_reference.head(15))
        """
    ),
    md(
        """
        ## 4. H1 Framing Recommendation

        **Recommendation:** use the merged-topic map as a structured exploratory measurement layer, and use the outlet-level distribution metrics as the actual H1 evidence.

        That means:
        - Treat the `74`-topic merged space as the common measurement frame across outlets.
        - Test "narrowing" with explicit metrics such as topic coverage, normalized entropy, effective topic count, and top-5 topic share.
        - Use amplification/substitution charts as the substantive interpretation of which topics alternative outlets over-weight or under-weight relative to `Tagesschau`.

        This is methodologically stronger than presenting the topic model alone as a direct confirmatory test, because the topic-model construction still contains researcher choices such as stopwords, date window, and merge threshold.
        """
    ),
    code(
        """
        coverage_summary_df = moa.build_outlet_topic_coverage_summary(article_topic_df, merged_topic_info_named)
        prevalence_df = moa.build_topic_prevalence_by_outlet(article_topic_df)
        focus_metrics_df = moa.build_outlet_focus_metrics(
            prevalence_df,
            total_topic_count=non_outlier_topic_count,
        )

        prevalence_exports = moa.save_dataframe_exports(
            prevalence_df,
            moa.get_local_output_dir(PROJECT_ROOT) / "topic_prevalence_by_outlet",
        )
        focus_exports = moa.save_dataframe_exports(
            focus_metrics_df,
            moa.get_local_output_dir(PROJECT_ROOT) / "outlet_focus_metrics",
        )

        print("Saved prevalence exports:")
        for kind, path in prevalence_exports.items():
            print(f"  {kind}: {path}")

        print("Saved focus-metric exports:")
        for kind, path in focus_exports.items():
            print(f"  {kind}: {path}")

        display(coverage_summary_df)
        display(
            focus_metrics_df[
                [
                    "outlet_label",
                    "covered_topic_count",
                    "coverage_share",
                    "normalized_entropy",
                    "effective_topic_count",
                    "top_5_topic_share",
                ]
            ].sort_values(["coverage_share", "normalized_entropy"], ascending=[False, False])
        )
        """
    ),
    code(
        """
        # Compare each outlet's focus metrics directly to Tagesschau as the mainstream baseline.
        tagesschau_row = focus_metrics_df.loc[focus_metrics_df["outlet_key"] == "tagesschau"].iloc[0]
        focus_vs_tagesschau = focus_metrics_df.copy()
        for metric in [
            "coverage_share",
            "normalized_entropy",
            "effective_topic_count",
            "top_5_topic_share",
        ]:
            focus_vs_tagesschau[f"{metric}_minus_tagesschau"] = (
                focus_vs_tagesschau[metric] - float(tagesschau_row[metric])
            )

        display(
            focus_vs_tagesschau[
                [
                    "outlet_label",
                    "covered_topic_count",
                    "coverage_share",
                    "coverage_share_minus_tagesschau",
                    "normalized_entropy",
                    "normalized_entropy_minus_tagesschau",
                    "effective_topic_count",
                    "effective_topic_count_minus_tagesschau",
                    "top_5_topic_share",
                    "top_5_topic_share_minus_tagesschau",
                ]
            ].sort_values("coverage_share_minus_tagesschau", ascending=False)
        )
        """
    ),
    md(
        """
        ## 5. Topic Prevalence: Amplification And Substitution vs `Tagesschau`

        Here the logic is simple:
        - positive share differences mean the alternative outlet over-weights a topic relative to `Tagesschau`
        - negative share differences mean the alternative outlet gives less attention to that topic than `Tagesschau`

        That makes the chart useful for identifying both amplification and substitution away from the mainstream agenda.
        """
    ),
    code(
        """
        comparison_df = moa.build_outlet_vs_baseline_prevalence(
            prevalence_df,
            outlet_key=OUTLET_TO_COMPARE,
            baseline_key="tagesschau",
        )

        display(
            comparison_df[
                [
                    "merged_display_topic",
                    "topic_label",
                    "article_share",
                    "baseline_article_share",
                    "share_diff",
                    "comparison_bucket",
                ]
            ].head(12)
        )

        display(
            comparison_df[
                [
                    "merged_display_topic",
                    "topic_label",
                    "article_share",
                    "baseline_article_share",
                    "share_diff",
                    "comparison_bucket",
                ]
            ].sort_values("share_diff", ascending=True).head(12)
        )
        """
    ),
    code(
        """
        fig, ax = moa.plot_outlet_vs_baseline_prevalence(
            comparison_df,
            top_n=TOP_N_DIFFERENCE_TOPICS,
        )
        plt.show()
        """
    ),
    code(
        """
        # Optional full comparison pass for every alternative outlet against Tagesschau.
        all_comparisons = []
        figure_dir = moa.get_local_output_dir(PROJECT_ROOT) / "topic_prevalence_figures"
        if SAVE_COMPARISON_FIGURES:
            figure_dir.mkdir(parents=True, exist_ok=True)

        for outlet_key in moa.ALT_MEDIA_OUTLET_KEYS:
            comparison = moa.build_outlet_vs_baseline_prevalence(
                prevalence_df,
                outlet_key=outlet_key,
                baseline_key="tagesschau",
            )
            all_comparisons.append(comparison)

            fig, ax = moa.plot_outlet_vs_baseline_prevalence(
                comparison,
                top_n=TOP_N_DIFFERENCE_TOPICS,
            )
            plt.show()

            if SAVE_COMPARISON_FIGURES:
                output_path = figure_dir / f"{outlet_key}_vs_tagesschau_topic_prevalence.png"
                fig.savefig(output_path, bbox_inches="tight")

        all_comparisons_df = pd.concat(all_comparisons, ignore_index=True)
        comparison_exports = moa.save_dataframe_exports(
            all_comparisons_df,
            moa.get_local_output_dir(PROJECT_ROOT) / "topic_prevalence_vs_tagesschau",
        )

        print("Saved outlet-vs-Tagesschau comparison exports:")
        for kind, path in comparison_exports.items():
            print(f"  {kind}: {path}")
        """
    ),
    md(
        """
        ## Suggested Next Moves

        1. Keep working with `article_topic_df` as the canonical article-level merged-topics dataset.
        2. If you are still changing the merged model, leave topic names as keywords for now.
        3. Once the merged-topic map is stable, fill `merged_topic_name_overrides.csv` with short manual names.
        4. Use the focus-metric table for the narrowing claim in H1 and the prevalence charts for substantive interpretation.
        """
    ),
]


nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(NOTEBOOK_PATH)
