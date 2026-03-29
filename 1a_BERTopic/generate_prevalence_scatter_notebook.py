from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path(__file__).resolve().parent / "Topic_Prevalence_Map_vs_Tagesschau.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # Topic Prevalence Map vs Tagesschau

        This notebook maps each merged topic in a two-axis prevalence space:

        - **x-axis:** `Tagesschau` topic prevalence
        - **y-axis:** selected outlet topic prevalence

        Interpretation:
        - points **above** the diagonal indicate **amplification** relative to `Tagesschau`
        - points **below** the diagonal indicate **substitution away from** the mainstream agenda

        The notebook is designed for fast iteration. It loads the saved article-level merged-topic dataframe if available and only rebuilds it when explicitly requested.
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

        REBUILD_ARTICLE_TOPIC_EXPORT = False
        OUTLET_TO_COMPARE = "rt"
        LABEL_COUNT_PER_SIDE = 6
        MIN_LABEL_SHARE_DIFF = 0.01
        SAVE_FIGURES = False

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
        print(f"Compare outlet: {OUTLET_TO_COMPARE}")
        """
    ),
    md(
        """
        ## 1. Load The Article-Level Merged Topic Dataset

        This is the working dataframe with one matched merged topic per article.
        """
    ),
    code(
        """
        article_topic_df = None if REBUILD_ARTICLE_TOPIC_EXPORT else moa.load_exported_article_topic_dataset(PROJECT_ROOT)

        if article_topic_df is None:
            merged_model, _, _ = moa.load_saved_merged_model(PROJECT_ROOT)
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
        print("Saved article-topic exports:")
        for kind, path in article_topic_exports.items():
            print(f"  {kind}: {path}")

        display(
            article_topic_df[
                [
                    "outlet_label",
                    "Title",
                    "merged_topic",
                    "merged_display_topic",
                    "topic_label",
                ]
            ].head(10)
        )
        """
    ),
    md(
        """
        ## 2. Build Topic Prevalence Tables

        Each topic share is the within-outlet share of articles assigned to that merged topic.
        """
    ),
    code(
        """
        prevalence_df = moa.build_topic_prevalence_by_outlet(article_topic_df)
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
        """
    ),
    md(
        """
        ## 3. Main Prevalence Map

        This is the core substitution/amplification chart.
        """
    ),
    code(
        """
        fig, ax = moa.plot_outlet_vs_baseline_prevalence_scatter(
            comparison_df,
            label_count_per_side=LABEL_COUNT_PER_SIDE,
            min_label_share_diff=MIN_LABEL_SHARE_DIFF,
        )
        plt.show()
        """
    ),
    md(
        """
        ## 4. Strongest Amplified And Substituted Topics

        These tables surface the largest deviations from the `Tagesschau` topic distribution.
        """
    ),
    code(
        """
        strongest_amplified = comparison_df.sort_values("share_diff", ascending=False).head(12)
        strongest_substituted = comparison_df.sort_values("share_diff", ascending=True).head(12)

        print("Strongest amplified topics:")
        display(
            strongest_amplified[
                [
                    "merged_display_topic",
                    "topic_label",
                    "article_share",
                    "baseline_article_share",
                    "share_diff",
                ]
            ]
        )

        print("Strongest substituted topics:")
        display(
            strongest_substituted[
                [
                    "merged_display_topic",
                    "topic_label",
                    "article_share",
                    "baseline_article_share",
                    "share_diff",
                ]
            ]
        )
        """
    ),
    md(
        """
        ## 5. All Alternative Outlets

        This cell lets you scan the prevalence map for every alternative outlet against `Tagesschau`.
        """
    ),
    code(
        """
        figure_dir = moa.get_local_output_dir(PROJECT_ROOT) / "topic_prevalence_scatter_figures"
        if SAVE_FIGURES:
            figure_dir.mkdir(parents=True, exist_ok=True)

        for outlet_key in moa.ALT_MEDIA_OUTLET_KEYS:
            comparison = moa.build_outlet_vs_baseline_prevalence(
                prevalence_df,
                outlet_key=outlet_key,
                baseline_key="tagesschau",
            )
            fig, ax = moa.plot_outlet_vs_baseline_prevalence_scatter(
                comparison,
                label_count_per_side=LABEL_COUNT_PER_SIDE,
                min_label_share_diff=MIN_LABEL_SHARE_DIFF,
            )
            plt.show()

            if SAVE_FIGURES:
                output_path = figure_dir / f"{outlet_key}_vs_tagesschau_prevalence_scatter.png"
                fig.savefig(output_path, bbox_inches="tight")
        """
    ),
    md(
        """
        ## Suggested Reading Of The Plot

        Use the chart in this order:
        1. Check whether most points cluster close to the diagonal or diverge strongly.
        2. Look at the labeled points above the line to identify topics the outlet amplifies.
        3. Look at the labeled points below the line to identify topics it downplays or substitutes away from.
        4. Combine this with the broader focus metrics from the main workbench when writing H1.
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
