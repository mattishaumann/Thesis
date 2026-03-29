from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = (
    Path(__file__).resolve().parent / "07_substitution_amplification_scatter.ipynb"
)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # Substitution-Amplification Scatter for H1

        This notebook builds the publication-style scatter plot requested for the H1 agenda-distortion chapter.

        It does four things in a reproducible order:

        1. loads `v2` experiment data when available and otherwise materializes it from the saved merged article-topic export
        2. computes within-outlet topic prevalence shares relative to `Tagesschau`
        3. selects one signature topic per alternative outlet plus 2-3 convergence topics
        4. exports the thesis-ready scatter figure and prints the diagnostic summary
        """
    ),
    code(
        """
        import sys
        from pathlib import Path

        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display

        NOTEBOOK_DIR = Path.cwd()
        if not (NOTEBOOK_DIR / "modeling.py").exists():
            if (NOTEBOOK_DIR / "experiments" / "agenda_distortion" / "modeling.py").exists():
                NOTEBOOK_DIR = NOTEBOOK_DIR / "experiments" / "agenda_distortion"
            else:
                raise FileNotFoundError("Run this notebook from experiments/agenda_distortion or the repo root.")

        PROJECT_ROOT = NOTEBOOK_DIR.parent.parent
        OUTPUT_DIR = NOTEBOOK_DIR / "outputs"

        if str(NOTEBOOK_DIR) not in sys.path:
            sys.path.insert(0, str(NOTEBOOK_DIR))
        if str(PROJECT_ROOT / "1a_BERTopic") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "1a_BERTopic"))

        import modeling
        import visualization as viz
        import merged_outlets_analysis as moa

        pd.set_option("display.max_columns", 30)
        pd.set_option("display.max_colwidth", 120)
        plt.rcParams["figure.dpi"] = 140

        print(f"Notebook dir: {NOTEBOOK_DIR}")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Output dir: {OUTPUT_DIR}")
        """
    ),
    md(
        """
        ## 1. Load The Preferred Experiment Iteration

        The notebook prefers `v2` because that is the no-outlier-reduction result set. If those CSVs are missing, it reconstructs them from the saved article-level merged-topic export in `1a_BERTopic/local_outputs/`.
        """
    ),
    code(
        """
        iteration_id, merged_articles, merged_topic_info, iteration_dir = modeling.ensure_iteration_outputs(
            PROJECT_ROOT,
            OUTPUT_DIR,
            preferred_iteration_id="v2",
            fallback_iteration_id="v1",
        )

        merged_articles["merged_topic"] = pd.to_numeric(
            merged_articles["merged_topic"], errors="coerce"
        ).astype("Int64")
        if "merged_display_topic" in merged_articles.columns:
            merged_articles["merged_display_topic"] = pd.to_numeric(
                merged_articles["merged_display_topic"], errors="coerce"
            ).astype("Int64")

        print(f"Using iteration: {iteration_id}")
        print(f"Iteration directory: {iteration_dir}")
        print(f"Articles: {len(merged_articles):,}")
        print(f"Non-outlier topics: {merged_articles.loc[merged_articles['merged_topic'] != -1, 'merged_topic'].nunique()}")
        print(f"Outliers: {(merged_articles['merged_topic'] == -1).sum():,}")

        display(merged_articles.head(3))
        display(merged_topic_info.head(10))
        """
    ),
    md(
        """
        ## 2. Build The Topic Label Review Table

        BERTopic labels are not publication-ready. This step creates a suggested short English label for every merged topic and writes an editable override template to `outputs/v2/topic_label_overrides.csv` if it does not already exist.
        """
    ),
    code(
        """
        label_override_path = iteration_dir / "topic_label_overrides.csv"
        label_review_path = iteration_dir / "topic_label_review.csv"

        topic_label_review_df = viz.build_topic_label_review(
            merged_topic_info,
            override_path=label_override_path,
        )
        topic_label_review_df.to_csv(label_review_path, index=False)

        print(f"Label override template: {label_override_path}")
        print(f"Label review export: {label_review_path}")
        display(topic_label_review_df.head(15))
        """
    ),
    md(
        """
        ## 3. Compute Prevalence Shares And Select Plot Points

        Shares are always calculated within each outlet's own article total, excluding outlier topic `-1`.
        """
    ),
    code(
        """
        prevalence_df = moa.build_topic_prevalence_by_outlet(merged_articles, include_outliers=False)
        all_comparisons_df = moa.build_all_outlet_vs_baseline_prevalence(
            prevalence_df,
            baseline_key="tagesschau",
        )

        signature_df = viz.select_signature_topics(
            all_comparisons_df,
            topic_label_review_df,
            min_signature_share=0.03,
            substitution_cutoff=0.02,
            amplification_multiplier=1.5,
        )
        convergence_df = viz.select_convergence_topics(
            prevalence_df,
            topic_label_review_df,
            n_topics=3,
            min_baseline_share=0.015,
            min_mean_share=0.012,
            min_all_share=0.003,
        )
        plot_points_df = viz.build_substitution_amplification_points(signature_df, convergence_df)

        signature_df.to_csv(iteration_dir / "signature_topics.csv", index=False)
        convergence_df.to_csv(iteration_dir / "convergence_topics.csv", index=False)
        plot_points_df.to_csv(iteration_dir / "substitution_amplification_points.csv", index=False)

        print("Signature topics:")
        display(signature_df)
        print("Convergence topics:")
        display(convergence_df)
        """
    ),
    md(
        """
        ## 4. Review The Final Label Mapping Used In The Figure

        This table is the exact topic-name mapping used in the chart. If you want a different label, edit the override CSV and rerun this notebook.
        """
    ),
    code(
        """
        plotted_topic_ids = sorted(
            set(signature_df["merged_topic"].tolist()) | set(convergence_df["merged_topic"].tolist())
        )
        plotted_label_map_df = topic_label_review_df.loc[
            topic_label_review_df["merged_topic"].isin(plotted_topic_ids)
        ].copy()

        display(plotted_label_map_df)
        """
    ),
    md(
        """
        ## 5. Export The Scatter Figure

        The figure is saved in both PDF and PNG format at 300 dpi.
        """
    ),
    code(
        """
        figure_dir = iteration_dir / "figures"
        pdf_path = figure_dir / "substitution_amplification_scatter.pdf"
        png_path = figure_dir / "substitution_amplification_scatter.png"

        fig = viz.plot_substitution_amplification_scatter(
            plot_points_df,
            save_path=pdf_path,
            png_preview_path=png_path,
        )
        plt.show()

        print(f"PDF: {pdf_path}")
        print(f"PNG: {png_path}")
        """
    ),
    md(
        """
        ## 6. Diagnostic Summary

        This cell prints the exact summary requested for the H1 write-up and figure captioning.
        """
    ),
    code(
        """
        prevalence_wide = (
            prevalence_df.pivot_table(
                index="merged_topic",
                columns="outlet_key",
                values="article_share",
                fill_value=0.0,
            )
            .reindex(columns=["tagesschau", "rt", "antispiegel", "compact", "deutschlandkurier", "nius", "tichys"], fill_value=0.0)
        )

        print("1. Signature topics")
        print("=" * 90)
        for _, row in signature_df.iterrows():
            overlap_note = row["overlap_with_other_top5"] if str(row["overlap_with_other_top5"]).strip() else "none"
            fallback_note = "yes" if bool(row["used_visibility_fallback"]) else "no"
            print(
                f"{row['outlet_label']:<22} "
                f"topic={int(row['merged_topic']):>2} "
                f"raw='{row['raw_label']}' "
                f"plot='{row['plot_label']}' "
                f"outlet_share={row['outlet_share']:.2%} "
                f"tagesschau_share={row['tagesschau_share']:.2%} "
                f"diff={row['share_difference']:.2%} "
                f"zone={row['zone']:<13} "
                f"visibility_fallback={fallback_note} "
                f"shared_top5={overlap_note}"
            )

        print()
        print("2. Convergence topics")
        print("=" * 90)
        for _, row in convergence_df.iterrows():
            topic_id = int(row["merged_topic"])
            shares = prevalence_wide.loc[topic_id]
            share_text = ", ".join(
                f"{outlet}={shares[outlet]:.2%}"
                for outlet in prevalence_wide.columns
            )
            print(
                f"topic={topic_id:>2} "
                f"plot='{row['plot_label']}' "
                f"tagesschau={row['tagesschau_share']:.2%} "
                f"alt_mean={row['alt_mean_share']:.2%} "
                f"diag_gap={row['diagonal_gap']:.2%} "
                f"alt_std={row['alt_std_share']:.2%}"
            )
            print(f"  shares: {share_text}")

        print()
        print("3. Shared alternative-agenda overlap")
        print("=" * 90)
        overlap_rows = signature_df.loc[
            signature_df["overlap_with_other_top5"].fillna("").astype(str).str.strip() != ""
        ]
        if overlap_rows.empty:
            print("No signature topic also appears in another outlet's top-5 topics.")
        else:
            for _, row in overlap_rows.iterrows():
                print(
                    f"{row['outlet_label']}: '{row['plot_label']}' also appears in top-5 of {row['overlap_with_other_top5']}"
                )
        """
    ),
    md(
        """
        ## 7. H1 Reading Aid

        The plot is most useful as a compact visual summary for the H1 argument:

        - **Substitution** means a topic is highly salient for one outlet while nearly absent in `Tagesschau`.
        - **Amplification** means both cover the topic, but the outlet gives it much more weight.
        - **Convergence** marks topics that remain relatively similar across the media field.

        In this dataset, the expected H1 reading is:

        - pro-Russian outlets cluster around Russia-Ukraine signature topics, often in substitution or strong amplification positions
        - domestic right-wing and populist-alternative outlets cluster around party competition, migration, and conflict-oriented domestic politics
        - only a small set of topics sits close to parity across the full outlet landscape
        """
    ),
]


notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"] = {
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

NOTEBOOK_PATH.write_text(nbf.writes(notebook), encoding="utf-8")
print(f"Wrote notebook to {NOTEBOOK_PATH}")
