from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path(__file__).resolve().parent / "H1_Agenda_Distortion_Analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # H1 Agenda Distortion Analysis

        **H1 / RQ1**

        To what extent do German alternative media outlets engage in agenda distortion by focusing on a narrower set of topics and actors compared to mainstream media?

        This notebook answers H1 with the merged `74`-topic space as a common measurement frame across outlets.

        The logic is:
        1. measure **topic breadth**: how much of the merged topic space each outlet covers
        2. measure **topic concentration**: how strongly each outlet concentrates attention within that space
        3. inspect **amplification and substitution**: which topics are over-weighted, under-weighted, or absent relative to `Tagesschau`

        The key distinction is important:
        - an outlet can cover many topics but still distort the agenda by weighting a small subset far more strongly
        - so H1 should be answered with both **coverage** and **distributional concentration**
        """
    ),
    code(
        """
        import importlib
        import sys
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Markdown, display
        from matplotlib.ticker import PercentFormatter

        REBUILD_ARTICLE_TOPIC_EXPORT = False
        SAVE_FIGURES = False

        # Reuse the same prevalence-map thresholds as the dedicated prevalence notebook.
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

        pd.set_option("display.max_columns", 40)
        pd.set_option("display.max_colwidth", 140)
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.dpi"] = 140

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Local outputs: {moa.get_local_output_dir(PROJECT_ROOT)}")
        """
    ),
    md(
        """
        ## 1. Load The Article-Level Merged Topic Dataset

        This is the main working dataframe: one row per prepared article, with its matched merged topic.
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
        ## 2. Build H1 Summary Tables

        `focus_metrics_df` is the core outlet-level evidence table for H1.
        """
    ),
    code(
        """
        prevalence_df = moa.build_topic_prevalence_by_outlet(article_topic_df)
        focus_metrics_df = moa.build_outlet_focus_metrics(prevalence_df)
        tagesschau_row = focus_metrics_df.loc[focus_metrics_df["outlet_key"] == "tagesschau"].iloc[0]
        focus_vs_tagesschau_df = focus_metrics_df.copy()
        for metric in [
            "coverage_share",
            "normalized_entropy",
            "effective_topic_count",
            "top_5_topic_share",
        ]:
            focus_vs_tagesschau_df[f"{metric}_minus_tagesschau"] = (
                focus_vs_tagesschau_df[metric] - float(tagesschau_row[metric])
            )

        display(
            focus_vs_tagesschau_df[
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
            ].sort_values(["coverage_share", "normalized_entropy"], ascending=[False, False])
        )
        """
    ),
    md(
        """
        ## 3. Visual 1: Topic Breadth

        These charts answer the simplest breadth question first:
        how much of the shared `74`-topic space does each outlet actually cover?
        """
    ),
    code(
        """
        plot_df = focus_metrics_df.sort_values("coverage_share", ascending=True).copy()
        plot_df["highlight"] = np.where(plot_df["outlet_key"] == "tagesschau", "#222222", "#b35c2e")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        axes[0].barh(plot_df["outlet_label"], plot_df["covered_topic_count"], color=plot_df["highlight"])
        axes[0].set_title("Merged topics covered")
        axes[0].set_xlabel("Number of merged topics")

        axes[1].barh(plot_df["outlet_label"], plot_df["coverage_share"], color=plot_df["highlight"])
        axes[1].set_title("Coverage share of merged topic space")
        axes[1].set_xlabel("Share of 74 merged topics")
        axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))

        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 4. Visual 2: Topic Concentration

        Breadth is only one side of H1. These charts test whether outlets weight the topic space more narrowly than `Tagesschau`.
        """
    ),
    code(
        """
        plot_df = focus_metrics_df.sort_values("normalized_entropy", ascending=True).copy()
        plot_df["highlight"] = np.where(plot_df["outlet_key"] == "tagesschau", "#222222", "#4f83cc")

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        axes[0].barh(plot_df["outlet_label"], plot_df["normalized_entropy"], color=plot_df["highlight"])
        axes[0].set_title("Normalized topic entropy")
        axes[0].set_xlabel("Higher = broader weighting")

        axes[1].barh(plot_df["outlet_label"], plot_df["effective_topic_count"], color=plot_df["highlight"])
        axes[1].set_title("Effective topic count")
        axes[1].set_xlabel("Higher = less concentrated")

        axes[2].barh(plot_df["outlet_label"], plot_df["top_5_topic_share"], color=plot_df["highlight"])
        axes[2].set_title("Top-5 topic share")
        axes[2].set_xlabel("Higher = more concentrated")
        axes[2].xaxis.set_major_formatter(PercentFormatter(1.0))

        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 5. Visual 3: Breadth vs Concentration Summary

        This plot is the compact H1 summary:
        - **x-axis:** topic-space coverage
        - **y-axis:** normalized entropy
        - **point size:** effective topic count

        Outlets closer to the upper-right are broader; outlets lower or further left are narrower.
        """
    ),
    code(
        """
        summary_df = focus_metrics_df.copy()
        summary_df["point_size"] = 40 + (summary_df["effective_topic_count"] * 10)
        summary_df["point_color"] = np.where(
            summary_df["outlet_key"] == "tagesschau",
            "#222222",
            "#c46f3a",
        )

        fig, ax = plt.subplots(figsize=(8.5, 7))
        ax.scatter(
            summary_df["coverage_share"],
            summary_df["normalized_entropy"],
            s=summary_df["point_size"],
            c=summary_df["point_color"],
            alpha=0.85,
            edgecolors="white",
            linewidths=1.0,
        )

        for _, row in summary_df.iterrows():
            ax.text(
                row["coverage_share"] + 0.005,
                row["normalized_entropy"] + 0.003,
                row["outlet_label"],
                fontsize=10,
                ha="left",
                va="bottom",
            )

        ax.set_xlim(0.5, 1.03)
        ax.set_ylim(0.6, 0.96)
        ax.set_xlabel("Coverage share of 74-topic space")
        ax.set_ylabel("Normalized topic entropy")
        ax.set_title("H1 summary: breadth and concentration of outlet agendas")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 6. Visual 4: Amplification And Substitution Map

        This uses the screenshot-style prevalence map to show *which* topics are amplified, substituted, or relatively mainstream-only.
        """
    ),
    code(
        """
        all_comparisons_df = moa.build_all_outlet_vs_baseline_prevalence(
            prevalence_df,
            baseline_key="tagesschau",
        )
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

        display(
            salient_points_df[
                [
                    "map_zone",
                    "outlet_label",
                    "topic_label",
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
        ## 7. Direct H1 Answer

        This cell turns the metrics into a compact written answer.
        """
    ),
    code(
        """
        alt_df = focus_metrics_df.loc[focus_metrics_df["outlet_key"] != "tagesschau"].copy()

        lower_entropy_count = int((alt_df["normalized_entropy"] < float(tagesschau_row["normalized_entropy"])).sum())
        lower_effective_count = int((alt_df["effective_topic_count"] < float(tagesschau_row["effective_topic_count"])).sum())
        higher_top5_count = int((alt_df["top_5_topic_share"] > float(tagesschau_row["top_5_topic_share"])).sum())
        lower_coverage_count = int((alt_df["coverage_share"] < float(tagesschau_row["coverage_share"])).sum())

        strongest_narrowers = alt_df.sort_values(
            ["normalized_entropy", "effective_topic_count", "coverage_share"],
            ascending=[True, True, True],
        )["outlet_label"].tolist()

        answer_lines = [
            "### H1 Interpretation",
            "",
            f"- `Tagesschau` is the broadest benchmark in the merged topic space: coverage share `{tagesschau_row['coverage_share']:.0%}`, normalized entropy `{tagesschau_row['normalized_entropy']:.3f}`, effective topic count `{tagesschau_row['effective_topic_count']:.1f}`, top-5 share `{tagesschau_row['top_5_topic_share']:.1%}`.",
            f"- `{lower_entropy_count}` of `{len(alt_df)}` alternative outlets have lower entropy than `Tagesschau`.",
            f"- `{lower_effective_count}` of `{len(alt_df)}` alternative outlets have a lower effective topic count than `Tagesschau`.",
            f"- `{higher_top5_count}` of `{len(alt_df)}` alternative outlets have a higher top-5 topic share than `Tagesschau`.",
            f"- `{lower_coverage_count}` of `{len(alt_df)}` alternative outlets cover a smaller share of the merged topic space than `Tagesschau`.",
            "",
            "**Conclusion:** H1 is supported, but mainly through **distributional concentration and selective salience**, not through universal topic absence.",
            "",
            "More precisely:",
            "- Some outlets do narrow the topic landscape in raw breadth terms, especially `Antispiegel` and `Deutschlandkurier`.",
            "- Others, such as `RT`, still cover much of the shared topic space, but weight it far more unevenly than `Tagesschau`.",
            "- That means agenda distortion in this dataset operates primarily as **amplification and substitution within a shared topic space**, with outright narrowing strongest for a subset of outlets rather than the whole alternative-media field equally.",
            "",
            f"Outlets with the clearest narrowing profile in this run: `{', '.join(strongest_narrowers[:3])}`.",
        ]

        display(Markdown(\"\\n\".join(answer_lines)))
        """
    ),
    md(
        """
        ## Suggested Thesis Wording

        If you want a concise thesis-style answer, the evidence in this notebook supports a formulation like this:

        > German alternative media do not uniformly reduce topic breadth in the sense of covering entirely different agendas from `Tagesschau`. However, they do consistently redistribute salience more narrowly: compared with `Tagesschau`, they show lower topic entropy, fewer effective topics, and stronger concentration in a small number of top issues. Agenda distortion therefore appears primarily as selective amplification and substitution within a shared topic space, with the strongest outright narrowing found in outlets such as `Antispiegel` and `Deutschlandkurier`.
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
