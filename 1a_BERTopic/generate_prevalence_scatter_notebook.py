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

        This notebook builds the screenshot-style prevalence map:

        - **x-axis:** `Tagesschau` topic prevalence
        - **y-axis:** outlet topic prevalence
        - **diagonal:** parity
        - **top-left:** substitution zone
        - **upper-above-diagonal:** amplification zone
        - **bottom-right:** mainstream-only

        The primary chart is a **salient multi-outlet map**. It places selected topic-outlet points from across the alternative outlets into that shared x/y space, which is closer to your sketch than a separate scatter for one outlet only.
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
        SAVE_FIGURES = False

        # These thresholds define the screenshot-style zones.
        SUBSTITUTION_BASELINE_MAX = 0.035
        SUBSTITUTION_OUTLET_MIN = 0.07
        AMPLIFICATION_BASELINE_MIN = 0.03
        AMPLIFICATION_MARGIN = 0.01
        AMPLIFICATION_Y_MIN = 0.08
        MAINSTREAM_ONLY_BASELINE_MIN = 0.045
        MAINSTREAM_ONLY_MARGIN = 0.01
        MAINSTREAM_ONLY_OUTLET_MAX = 0.045
        PARITY_MARGIN = 0.006
        PARITY_MIN_BASELINE_SHARE = 0.045
        POINTS_PER_ZONE = 4

        # Edit this mapping if you want different outlet-family labels or colors.
        OUTLET_GROUP_MAP = {
            "rt": "Pro-Russian / right-extremist",
            "antispiegel": "Pro-Russian / right-extremist",
            "compact": "Pro-Russian / right-extremist",
            "deutschlandkurier": "Pro-Russian / right-extremist",
            "nius": "Populist-alternative",
            "tichys": "Populist-alternative",
        }
        OUTLET_GROUP_COLORS = {
            "Pro-Russian / right-extremist": "#d67a4d",
            "Populist-alternative": "#5c8fd6",
        }

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
        """
    ),
    md(
        """
        ## 1. Load The Article-Level Merged Topic Dataset

        This is the saved dataframe with one matched merged topic per article.
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
        for kind, path in article_topic_exports.items():
            print(f"{kind}: {path}")

        display(
            article_topic_df[
                [
                    "outlet_label",
                    "Title",
                    "merged_display_topic",
                    "topic_label",
                ]
            ].head(10)
        )
        """
    ),
    md(
        """
        ## 2. Build The Shared Prevalence Space

        `all_comparisons_df` contains one row per outlet-topic comparison against `Tagesschau`.
        """
    ),
    code(
        """
        prevalence_df = moa.build_topic_prevalence_by_outlet(article_topic_df)
        all_comparisons_df = moa.build_all_outlet_vs_baseline_prevalence(
            prevalence_df,
            baseline_key="tagesschau",
        )

        all_comparisons_df["outlet_group"] = all_comparisons_df["outlet_key"].map(OUTLET_GROUP_MAP).fillna("Other")
        display(
            all_comparisons_df[
                [
                    "outlet_label",
                    "topic_label",
                    "article_share",
                    "baseline_article_share",
                    "share_diff",
                ]
            ].head(12)
        )
        """
    ),
    md(
        """
        ## 3. Select Salient Points For The Screenshot-Style Map

        This step intentionally does **not** plot every single topic-outlet pair. Instead it picks the most salient points in each conceptual zone so the map stays interpretable.
        """
    ),
    code(
        """
        salient_points_df = moa.build_salient_prevalence_map_points(
            all_comparisons_df,
            substitution_baseline_max=SUBSTITUTION_BASELINE_MAX,
            substitution_outlet_min=SUBSTITUTION_OUTLET_MIN,
            amplification_baseline_min=AMPLIFICATION_BASELINE_MIN,
            amplification_margin=AMPLIFICATION_MARGIN,
            mainstream_only_baseline_min=MAINSTREAM_ONLY_BASELINE_MIN,
            mainstream_only_margin=MAINSTREAM_ONLY_MARGIN,
            parity_margin=PARITY_MARGIN,
            parity_min_baseline_share=PARITY_MIN_BASELINE_SHARE,
            points_per_zone=POINTS_PER_ZONE,
        )
        salient_points_df["outlet_group"] = salient_points_df["outlet_key"].map(OUTLET_GROUP_MAP).fillna("Other")

        display(
            salient_points_df[
                [
                    "map_zone",
                    "outlet_label",
                    "topic_short_label",
                    "baseline_article_share",
                    "article_share",
                    "share_diff",
                ]
            ].sort_values(["map_zone", "share_diff"], ascending=[True, False])
        )
        """
    ),
    md(
        """
        ## 4. Screenshot-Style Prevalence Map

        This is the chart closest to your sketch.
        """
    ),
    code(
        """
        fig, ax = moa.plot_salient_prevalence_map(
            salient_points_df,
            reference_df=all_comparisons_df,
            group_col="outlet_group",
            color_map=OUTLET_GROUP_COLORS,
            substitution_baseline_max=SUBSTITUTION_BASELINE_MAX,
            substitution_outlet_min=SUBSTITUTION_OUTLET_MIN,
            amplification_baseline_min=AMPLIFICATION_BASELINE_MIN,
            amplification_y_min=AMPLIFICATION_Y_MIN,
            mainstream_only_baseline_min=MAINSTREAM_ONLY_BASELINE_MIN,
            mainstream_only_outlet_max=MAINSTREAM_ONLY_OUTLET_MAX,
        )
        plt.show()
        """
    ),
    md(
        """
        ## 5. Detailed Tables Behind The Map

        These are useful when you want to justify why a given point appears in a zone.
        """
    ),
    code(
        """
        display(
            salient_points_df.loc[salient_points_df["map_zone"] == "Substitution zone", [
                "outlet_label",
                "topic_label",
                "baseline_article_share",
                "article_share",
                "share_diff",
            ]]
        )

        display(
            salient_points_df.loc[salient_points_df["map_zone"] == "Amplification zone", [
                "outlet_label",
                "topic_label",
                "baseline_article_share",
                "article_share",
                "share_diff",
            ]]
        )

        display(
            salient_points_df.loc[salient_points_df["map_zone"] == "Mainstream only", [
                "outlet_label",
                "topic_label",
                "baseline_article_share",
                "article_share",
                "share_diff",
            ]]
        )
        """
    ),
    md(
        """
        ## 6. Optional Detailed Diagnostic For One Outlet

        This is still useful as a secondary diagnostic, but it is **not** the main screenshot-style chart.
        """
    ),
    code(
        """
        comparison_df = moa.build_outlet_vs_baseline_prevalence(
            prevalence_df,
            outlet_key=OUTLET_TO_COMPARE,
            baseline_key="tagesschau",
        )

        fig, ax = moa.plot_outlet_vs_baseline_prevalence_scatter(
            comparison_df,
            label_count_per_side=6,
            min_label_share_diff=0.01,
        )
        plt.show()
        """
    ),
    md(
        """
        ## Reading Advice

        Use the screenshot-style map like this:
        1. top-left: topics the outlet world elevates that are barely present in `Tagesschau`
        2. above parity with non-trivial baseline presence: topics that are shared but much more heavily weighted by the outlet
        3. bottom-right: topics that remain comparatively mainstream-only
        4. near parity: topics where the outlet and `Tagesschau` weight the topic similarly
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
