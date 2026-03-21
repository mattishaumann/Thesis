from __future__ import annotations

import ast
import html
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bertopic_config import BERTopicConfig
from bertopic_pipeline import prepare_documents


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TICHYS_EXTRA_STOPWORDS = (
    "amazon",
    "spotify",
    "soundcloud",
    "itunes",
    "abonnieren",
    "abonniere",
    "abonnement",
    "abo",
    "angezeigt",
    "anzeigen",
    "inhalte",
    "einverstanden",
)

TEXT_TYPES = {"text", "headline"}
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class OutletSpec:
    key: str
    label: str
    model_name: str
    text_col: str
    id_col: str
    config_factory: Callable[[], BERTopicConfig]
    loader: Callable[[Path], pd.DataFrame]


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing .git")


def read_csv_resilient(csv_path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                csv_path,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="skip",
            )
        except Exception:
            pass

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                csv_path,
                encoding=encoding,
                sep=None,
                engine="python",
                low_memory=False,
                on_bad_lines="skip",
            )
        except Exception:
            pass

    raise ValueError(f"Could not read: {csv_path}")


def _load_csv_tree(base_dir: Path, source_name: str) -> pd.DataFrame:
    csv_files = sorted(base_dir.rglob("*.csv"))
    parts = []
    for csv_file in csv_files:
        part = read_csv_resilient(csv_file)
        part["source"] = source_name
        part["source_file"] = csv_file.name
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _remove_mehr_zum_thema(text: object) -> object:
    if pd.isna(text):
        return text
    return re.sub(r"\n*\s*Mehr zum Thema.*$", "", str(text), flags=re.DOTALL).strip()


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _extract_tagesschau_content(content_raw: object) -> str:
    if pd.isna(content_raw):
        return ""
    try:
        blocks = ast.literal_eval(content_raw) if isinstance(content_raw, str) else content_raw
    except Exception:
        return _clean_html(str(content_raw))

    if not isinstance(blocks, list):
        return _clean_html(str(blocks))

    out: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in TEXT_TYPES and "value" in block:
            value = _clean_html(str(block["value"]))
            if value:
                if block_type == "headline":
                    out.append(f"\n{value}\n")
                else:
                    out.append(value)
    return "\n".join(piece for piece in out if piece).strip()


def _extract_author_name(authors_str: object) -> str | None:
    if pd.isna(authors_str) or authors_str is None:
        return None

    authors_text = str(authors_str)
    matches = re.findall(r"'name':\s*'([^']*)'", authors_text)
    if matches:
        return ", ".join(matches)

    matches = re.findall(r'"name":\s*"([^"]*)"', authors_text)
    if matches:
        return ", ".join(matches)

    return None


def _remove_trailing_nius_reference(text: object) -> object:
    if pd.isna(text) or text is None:
        return text
    pattern = r"\n*(?:Mehr|Auch|Lesen\s+[Ss]ie\s+auch)\s+(?:bei\s+)?NIUS\s*:.*$"
    return re.sub(pattern, "", str(text), flags=re.IGNORECASE | re.DOTALL).strip()


def load_rt_dataframe(project_root: Path) -> pd.DataFrame:
    rt_file = project_root / "data" / "raw" / "Alternative Medien" / "RT_de.xlsx"
    df = pd.read_excel(rt_file)
    df["source"] = "RT_de"
    df["source_file"] = rt_file.name

    df_clean = df.drop(columns=["Full_Text"], errors="ignore").copy()
    df_clean["Text"] = (
        df_clean["Text"]
        .astype("string")
        .fillna("")
        .map(_remove_mehr_zum_thema)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], format="%Y-%m-%d", errors="coerce")
    df_clean = df_clean[
        (df_clean["Date"] >= pd.Timestamp("2025-08-01"))
        & (df_clean["Date"] < pd.Timestamp("2026-02-01"))
    ].copy()

    dup_title = df_clean.duplicated(subset=["Title"], keep="first")
    dup_text = df_clean.duplicated(subset=["Text"], keep="first")
    df_clean = df_clean.loc[~(dup_title | dup_text)].copy()
    return df_clean


def load_compact_dataframe(project_root: Path) -> pd.DataFrame:
    base_dir = project_root / "data" / "raw" / "Alternative Medien" / "Compact"
    df = _load_csv_tree(base_dir, "Compact")
    df_clean = df.copy()
    dates = pd.to_datetime(df_clean["Date"], errors="coerce")
    mask = (dates >= "2025-08-01") & (dates <= "2026-01-31")
    df_clean = df_clean.loc[mask].copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
    keep_cols = [col for col in ["Date", "Title", "Inhalt", "URL", "source", "source_file"] if col in df_clean.columns]
    df_clean = df_clean.loc[:, keep_cols].copy()
    df_clean["Inhalt"] = df_clean["Inhalt"].map(_remove_mehr_zum_thema)
    return df_clean


def load_nius_dataframe(project_root: Path) -> pd.DataFrame:
    base_dir = project_root / "data" / "raw" / "Alternative Medien" / "Nius_Rohdaten_neu"
    df = _load_csv_tree(base_dir, "Nius")

    df_clean = df[["title", "categories", "day", "authors", "article_text", "url", "source", "source_file"]].copy()
    df_clean = df_clean.rename(
        columns={
            "title": "Title",
            "article_text": "Text",
            "day": "Date",
            "authors": "Authors",
            "categories": "Categories",
            "url": "URL",
        }
    )
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
    df_clean = df_clean.dropna().copy()

    dup_title_mask = df_clean.duplicated(subset=["Title"], keep=False)
    dup_text_mask = df_clean.duplicated(subset=["Text"], keep=False)
    df_clean = df_clean.loc[~(dup_title_mask | dup_text_mask)].copy()

    df_clean["Authors"] = df_clean["Authors"].map(_extract_author_name)
    df_clean["Text"] = df_clean["Text"].map(_remove_trailing_nius_reference).astype("string")
    df_clean["Text"] = df_clean["Text"].str.replace("\n", " ", regex=False)
    df_clean = df_clean[
        (df_clean["Date"] >= "2025-08-01")
        & (df_clean["Date"] <= "2026-01-31")
    ].copy()
    df_clean = df_clean[~df_clean["Categories"].astype("string").str.contains("Show", na=False)].copy()
    return df_clean


def load_tichys_dataframe(project_root: Path) -> pd.DataFrame:
    base_dir = project_root / "data" / "raw" / "Alternative Medien" / "Tichy's Einblick"
    df = _load_csv_tree(base_dir, "Tichys_Einblick")

    df_clean = df.rename(
        columns={
            "title": "Title",
            "date": "Date",
            "url": "URL",
            "author": "Author",
            "article_text": "Text",
        }
    )[["Title", "Date", "URL", "Author", "Text", "source", "source_file"]].copy()

    dates = pd.to_datetime(df_clean["Date"], errors="coerce")
    mask = (dates >= pd.Timestamp("2025-08-01")) & (dates <= pd.Timestamp("2026-01-31"))
    df_clean = df_clean.loc[mask].copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")

    issue_sale_mask = (
        df_clean["Title"].fillna("").str.contains(r"^Tichys Einblick \d{2}-\d{4}:", regex=True)
        | df_clean["URL"].fillna("").str.contains(r"/daili-es-sentials/tichys-einblick-\d{2}-\d{4}", regex=True)
        | df_clean["Text"].fillna("").str.contains(
            r"im Handel oder direkt als PDF erhältlich|direkt als PDF erhältlich",
            case=False,
            regex=True,
        )
    )
    te_wecker_mask = (
        df_clean["Title"].fillna("").str.contains(r"TE-Wecker am", case=False, regex=True)
        | df_clean["URL"].fillna("").str.contains(r"/podcast/te-wecker-am-", case=False, regex=True)
    )
    df_clean = df_clean.loc[~(issue_sale_mask | te_wecker_mask)].copy()
    return df_clean


def load_antispiegel_dataframe(project_root: Path) -> pd.DataFrame:
    base_dir = project_root / "data" / "raw" / "Alternative Medien" / "Antispiegel"
    df = _load_csv_tree(base_dir, "Antispiegel")

    df_clean = df.copy()
    dates = pd.to_datetime(df_clean["Date"], errors="coerce")
    mask = (dates >= pd.Timestamp("2025-08-01")) & (dates <= pd.Timestamp("2026-01-31"))
    df_clean = df_clean.loc[mask].copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")

    df_clean = df_clean[~df_clean["Title"].astype("string").str.contains("Tacheles #", na=False)].copy()

    podcast_mask = (
        df_clean["Text"].fillna("").str.contains(r"Anti-Spiegel-Podcast", case=False, regex=True)
        | df_clean["Text"].fillna("").str.contains(r"Den Podcast k[oö]nnen Sie hier", case=False, regex=True)
        | (
            df_clean["Text"].fillna("").str.contains(r"spotify", case=False, regex=True)
            & df_clean["Text"].fillna("").str.contains(r"\bVK\b|Netzwerk VK|russischen VK", case=False, regex=True)
        )
    )
    df_clean = df_clean.loc[~podcast_mask].copy()
    return df_clean


def load_tagesschau_dataframe(project_root: Path) -> pd.DataFrame:
    base_dir = project_root / "data" / "raw" / "Alternative Medien" / "Tagesschau"
    df = _load_csv_tree(base_dir, "Tagesschau")

    df = df.rename(columns={"updateCheckUrl": "URL"})
    cols_to_keep = [
        "title",
        "date",
        "tags",
        "URL",
        "content",
        "topline",
        "ressort",
        "regionId",
        "type",
        "source",
        "source_file",
    ]
    df_clean = df.loc[:, [col for col in cols_to_keep if col in df.columns]].copy()
    df_clean["content_clean"] = df_clean["content"].map(_extract_tagesschau_content)
    df_clean = df_clean.rename(columns={"date": "Date", "content_clean": "Text", "title": "Title"})

    dates = pd.to_datetime(df_clean["Date"], errors="coerce", utc=True).dt.tz_convert("Europe/Berlin")
    mask = (dates >= pd.Timestamp("2025-08-01", tz="Europe/Berlin")) & (
        dates < pd.Timestamp("2026-02-01", tz="Europe/Berlin")
    )
    df_clean = df_clean.loc[mask].copy()
    df_clean["Date"] = pd.to_datetime(dates.loc[mask], errors="coerce").dt.normalize()
    df_clean["Text"] = (
        df_clean["Text"]
        .fillna("")
        .astype("string")
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df_clean


def load_deutschlandkurier_dataframe(project_root: Path) -> pd.DataFrame:
    base_dir = project_root / "data" / "raw" / "Alternative Medien" / "Deutschlandkurier"
    df = _load_csv_tree(base_dir, "Deutschlandkurier")

    df_clean = df[["title", "categories", "created", "article_text", "url", "source", "source_file"]].copy()
    df_clean.columns = ["Title", "Categories", "Date", "Text", "URL", "source", "source_file"]
    df_clean = df_clean[df_clean["Text"].notna()].copy()
    df_clean = df_clean.drop_duplicates(subset=["Title", "Text"], keep="first").copy()
    df_clean["Date"] = (
        pd.to_datetime(df_clean["Date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    df_clean = df_clean[
        (df_clean["Date"] >= "2025-08-01")
        & (df_clean["Date"] <= "2026-01-31")
    ].copy()
    return df_clean


OUTLET_SPECS: dict[str, OutletSpec] = {
    "tagesschau": OutletSpec(
        key="tagesschau",
        label="Tagesschau",
        model_name="ts_model",
        text_col="Text",
        id_col="URL",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=35,
            hdbscan_min_samples=2,
            umap_n_neighbors=25,
            umap_min_dist=0.0,
        ),
        loader=load_tagesschau_dataframe,
    ),
    "rt": OutletSpec(
        key="rt",
        label="RT",
        model_name="rt_model",
        text_col="Text",
        id_col="URL",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=15,
            hdbscan_min_samples=3,
            max_df=0.85,
            umap_n_neighbors=40,
            umap_min_dist=0.05,
        ),
        loader=load_rt_dataframe,
    ),
    "antispiegel": OutletSpec(
        key="antispiegel",
        label="Antispiegel",
        model_name="as_model",
        text_col="Text",
        id_col="URL",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=8,
            hdbscan_min_samples=2,
            min_df=1,
            max_df=0.95,
            umap_n_neighbors=10,
            umap_min_dist=0.05,
            nr_topics=None,
        ),
        loader=load_antispiegel_dataframe,
    ),
    "tichys": OutletSpec(
        key="tichys",
        label="Tichys Einblick",
        model_name="te_model",
        text_col="Text",
        id_col="URL",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=10,
            umap_n_neighbors=15,
            extra_stopwords=TICHYS_EXTRA_STOPWORDS,
        ),
        loader=load_tichys_dataframe,
    ),
    "nius": OutletSpec(
        key="nius",
        label="Nius",
        model_name="ns_model",
        text_col="Text",
        id_col="URL",
        config_factory=lambda: BERTopicConfig(hdbscan_min_cluster_size=15),
        loader=load_nius_dataframe,
    ),
    "compact": OutletSpec(
        key="compact",
        label="Compact",
        model_name="compact_model",
        text_col="Inhalt",
        id_col="URL",
        config_factory=BERTopicConfig,
        loader=load_compact_dataframe,
    ),
    "deutschlandkurier": OutletSpec(
        key="deutschlandkurier",
        label="Deutschlandkurier",
        model_name="dk_model",
        text_col="Text",
        id_col="URL",
        config_factory=BERTopicConfig,
        loader=load_deutschlandkurier_dataframe,
    ),
}

ALT_MEDIA_OUTLET_KEYS: tuple[str, ...] = (
    "rt",
    "antispiegel",
    "tichys",
    "nius",
    "compact",
    "deutschlandkurier",
)


def resolve_model_path(model_name: str, candidates: list[Path]) -> Path:
    for candidate in candidates:
        model_path = candidate / model_name
        if model_path.exists():
            return model_path
    raise FileNotFoundError(
        f"Could not find {model_name} in: " + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_model_paths(candidates: list[Path]) -> dict[str, Path]:
    return {
        key: resolve_model_path(spec.model_name, candidates)
        for key, spec in OUTLET_SPECS.items()
    }


def prepare_outlet_documents(project_root: Path, key: str) -> pd.DataFrame:
    spec = OUTLET_SPECS[key]
    df = spec.loader(project_root)
    prepared = prepare_documents(
        df,
        text_col=spec.text_col,
        config=spec.config_factory(),
        id_col=spec.id_col,
        source_name=spec.label,
    ).copy()
    prepared["outlet_key"] = spec.key
    prepared["outlet_label"] = spec.label
    return prepared


def load_all_prepared_documents(project_root: Path) -> dict[str, pd.DataFrame]:
    return {
        key: prepare_outlet_documents(project_root, key)
        for key in OUTLET_SPECS
    }


def combine_prepared_documents(prepared_by_outlet: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ordered = [prepared_by_outlet[key] for key in OUTLET_SPECS]
    return pd.concat(ordered, ignore_index=True)


def enrich_topic_info_with_display(topic_info: pd.DataFrame) -> pd.DataFrame:
    topic_info = topic_info.copy()
    topic_info["TopicNameClean"] = topic_info["Name"].astype("string").str.replace(r"^\d+_", "", regex=True)

    ranked_topics = (
        topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Count"]]
        .sort_values(["Count", "Topic"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranked_topics["DisplayTopic"] = range(1, len(ranked_topics) + 1)
    display_topic_map = dict(zip(ranked_topics["Topic"], ranked_topics["DisplayTopic"]))

    topic_info["DisplayTopic"] = topic_info["Topic"].map(display_topic_map).astype("Int64")
    topic_info["DisplayLabel"] = topic_info.apply(
        lambda row: "Outliers"
        if row["Topic"] == -1
        else f"Topic {int(row['DisplayTopic'])} -- {row['TopicNameClean']}",
        axis=1,
    )
    return topic_info


def build_merged_article_frame(
    merged_model,
    combined_prepared: pd.DataFrame,
    *,
    umap_n_neighbors: int = 10,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, UMAP]:
    docs = combined_prepared["document"].tolist()
    topics, probabilities = merged_model.transform(docs)
    embeddings = merged_model._extract_embeddings(docs, method="document")

    reducer = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=2,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)

    merged_topic_info = enrich_topic_info_with_display(merged_model.get_topic_info())
    display_label_map = dict(zip(merged_topic_info["Topic"], merged_topic_info["DisplayLabel"]))
    display_topic_map = dict(zip(merged_topic_info["Topic"], merged_topic_info["DisplayTopic"]))

    frame = combined_prepared.copy()
    frame["merged_topic"] = list(topics)
    frame["merged_probability"] = list(probabilities) if probabilities is not None else None
    frame["merged_display_topic"] = frame["merged_topic"].map(display_topic_map).astype("Int64")
    frame["merged_display_label"] = frame["merged_topic"].map(display_label_map).fillna("Outliers")
    frame["umap_x"] = coords[:, 0]
    frame["umap_y"] = coords[:, 1]
    return frame, merged_topic_info, reducer


def plot_merged_topic_umap(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    top_n: int = 20,
    figsize: tuple[int, int] = (15, 15),
    point_size: int = 3,
    alpha: float = 0.35,
):
    topic_info = merged_topic_info.copy()
    top_topic_ids = (
        topic_info.loc[topic_info["Topic"] != -1]
        .nsmallest(top_n, "DisplayTopic")["Topic"]
        .tolist()
    )

    plot_df = merged_articles.copy()
    plot_df.loc[~plot_df["merged_topic"].isin(top_topic_ids), "merged_topic"] = -1

    fig, ax = plt.subplots(figsize=figsize)
    other = plot_df.loc[plot_df["merged_topic"] == -1]
    focus = plot_df.loc[plot_df["merged_topic"] != -1]

    ax.scatter(
        other["umap_x"],
        other["umap_y"],
        c="#d9d9d9",
        s=point_size,
        alpha=0.18,
        linewidths=0,
    )
    scatter = ax.scatter(
        focus["umap_x"],
        focus["umap_y"],
        c=focus["merged_display_topic"],
        s=point_size,
        alpha=alpha,
        cmap="tab20",
        linewidths=0,
    )
    scatter.set_clim(1, max(top_n, 1))

    centroids = focus.groupby("merged_topic", as_index=False)[["umap_x", "umap_y"]].mean()
    label_map = dict(zip(topic_info["Topic"], topic_info["DisplayLabel"]))
    for _, row in centroids.iterrows():
        topic = int(row["merged_topic"])
        ax.text(
            row["umap_x"],
            row["umap_y"],
            label_map.get(topic, f"Topic {topic}"),
            fontsize=10,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
        )

    ax.set_title(f"Merged BERTopic UMAP -- Top {top_n} Topics")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig, ax


def plot_outlet_highlight_umap(
    merged_articles: pd.DataFrame,
    outlet_key: str,
    alt_media_only: bool = False,
    show_kde: bool = True,
    min_label_articles: int = 15,
    *,
    merged_topic_info: pd.DataFrame | None = None,
    figsize: tuple[int, int] = (14, 10),
    dpi: int = 150,
    background_point_size: int = 7,
    highlight_size_range: tuple[int, int] = (8, 60),
    other_color: str = "#8F8F8F",
    highlight_color: str = "#d65a5a",
    background_alpha: float = 0.16,
    highlight_alpha: float = 0.74,
    label_fontsize: int = 9,
    max_labels: int = 12,
):
    spec = OUTLET_SPECS[outlet_key]
    plot_df = merged_articles.copy()
    if alt_media_only:
        plot_df = plot_df.loc[plot_df["outlet_key"].isin(ALT_MEDIA_OUTLET_KEYS)].copy()

    highlight = plot_df["outlet_key"] == outlet_key
    highlight_df = plot_df.loc[highlight].copy()
    topic_counts = highlight_df.loc[highlight_df["merged_topic"] != -1, "merged_topic"].value_counts()
    covered_topics = topic_counts.index.tolist()
    total_topic_count = int(plot_df.loc[plot_df["merged_topic"] != -1, "merged_topic"].nunique())
    covered_topic_count = len(covered_topics)
    coverage_share = covered_topic_count / total_topic_count if total_topic_count else 0.0
    article_count = int(len(highlight_df))
    total_article_count = int(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F7F7")
    ax.scatter(
        plot_df["umap_x"],
        plot_df["umap_y"],
        c=other_color,
        s=background_point_size,
        alpha=background_alpha,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    if not highlight_df.empty:
        topic_count_map = topic_counts.to_dict()
        if (highlight_df["merged_topic"] == -1).any():
            topic_count_map[-1] = int((highlight_df["merged_topic"] == -1).sum())

        count_series = highlight_df["merged_topic"].map(topic_count_map).fillna(1).astype(float)
        size_min, size_max = highlight_size_range
        if count_series.max() > count_series.min():
            scaled_sizes = size_min + (
                (count_series - count_series.min()) / (count_series.max() - count_series.min())
            ) * (size_max - size_min)
        else:
            scaled_sizes = pd.Series(
                np.full(len(count_series), (size_min + size_max) / 2),
                index=count_series.index,
            )

        if show_kde and len(highlight_df) >= 3:
            try:
                from scipy.stats import gaussian_kde

                x = highlight_df["umap_x"].to_numpy()
                y = highlight_df["umap_y"].to_numpy()
                values = np.vstack([x, y])
                kde = gaussian_kde(values)

                x_pad = max((plot_df["umap_x"].max() - plot_df["umap_x"].min()) * 0.05, 1e-6)
                y_pad = max((plot_df["umap_y"].max() - plot_df["umap_y"].min()) * 0.05, 1e-6)
                x_grid = np.linspace(plot_df["umap_x"].min() - x_pad, plot_df["umap_x"].max() + x_pad, 200)
                y_grid = np.linspace(plot_df["umap_y"].min() - y_pad, plot_df["umap_y"].max() + y_pad, 200)
                xx, yy = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = np.reshape(kde(positions), xx.shape)

                positive = zz[zz > 0]
                if positive.size >= 4:
                    levels = np.quantile(positive, [0.55, 0.72, 0.86, 0.96])
                    levels = np.unique(levels)
                    if len(levels) >= 2:
                        ax.contourf(
                            xx,
                            yy,
                            zz,
                            levels=levels,
                            colors=[highlight_color],
                            alpha=0.24,
                            zorder=2,
                        )
                        ax.contour(
                            xx,
                            yy,
                            zz,
                            levels=levels,
                            colors=highlight_color,
                            alpha=0.60,
                            linewidths=1.0,
                            zorder=3,
                        )
            except Exception as exc:
                warnings.warn(f"Skipping KDE overlay for {spec.label}: {exc}", RuntimeWarning)

        ax.scatter(
            highlight_df["umap_x"],
            highlight_df["umap_y"],
            c=highlight_color,
            s=scaled_sizes.to_numpy(),
            alpha=highlight_alpha,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )

    if covered_topics:
        eligible_topic_counts = topic_counts.loc[topic_counts >= min_label_articles].head(max_labels)
        label_topic_ids = eligible_topic_counts.index.tolist()
        if label_topic_ids:
            if merged_topic_info is not None:
                label_rows = merged_topic_info.loc[merged_topic_info["Topic"].isin(label_topic_ids)].copy()
                label_text_map = dict(zip(label_rows["Topic"], label_rows["DisplayLabel"]))
            else:
                label_text_map = (
                    plot_df.loc[plot_df["merged_topic"].isin(label_topic_ids), ["merged_topic", "merged_display_label"]]
                    .drop_duplicates(subset=["merged_topic"])
                    .set_index("merged_topic")["merged_display_label"]
                    .to_dict()
                )

            for topic_id in label_topic_ids:
                topic_points = plot_df.loc[plot_df["merged_topic"] == topic_id, ["umap_x", "umap_y"]].to_numpy()
                if len(topic_points) == 0:
                    continue

                highlight_points = highlight_df.loc[
                    highlight_df["merged_topic"] == topic_id,
                    ["umap_x", "umap_y"],
                ].to_numpy()
                if len(highlight_points) == 0:
                    continue

                centroid = highlight_points.mean(axis=0)
                distances = ((topic_points - centroid) ** 2).sum(axis=1)
                label_xy = topic_points[distances.argmin()]
                label_text = str(label_text_map.get(topic_id, topic_id))

                ax.text(
                    label_xy[0],
                    label_xy[1],
                    label_text,
                    fontsize=label_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "#DDDDDD",
                        "alpha": 0.92,
                        "boxstyle": "round,pad=0.18",
                    },
                    zorder=5,
                )

    ax.text(
        0.02,
        0.98,
        (
            f"Outlet: {spec.label}\n"
            f"Articles: {article_count:,}\n"
            f"Topics covered: {covered_topic_count}/{total_topic_count}\n"
            f"Coverage: {coverage_share:.0%}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "boxstyle": "round,pad=0.5",
        },
        zorder=6,
    )

    ax.set_title(
        f"{spec.label} -- Semantic Footprint in Merged Topic Space",
        fontsize=14,
        pad=28,
    )
    ax.text(
        0.5,
        1.01,
        f"Red = outlet articles | Grey = full corpus ({article_count:,} / {total_article_count:,} articles)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig, ax


def build_outlet_topic_coverage_summary(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    outlet_keys: tuple[str, ...] = ALT_MEDIA_OUTLET_KEYS,
) -> pd.DataFrame:
    total_topic_count = int(merged_topic_info.loc[merged_topic_info["Topic"] != -1, "Topic"].nunique())
    rows: list[dict[str, object]] = []

    for outlet_key in outlet_keys:
        spec = OUTLET_SPECS[outlet_key]
        outlet_df = merged_articles.loc[merged_articles["outlet_key"] == outlet_key].copy()
        covered_topic_count = int(outlet_df.loc[outlet_df["merged_topic"] != -1, "merged_topic"].nunique())
        outlier_articles = int((outlet_df["merged_topic"] == -1).sum())
        article_count = int(len(outlet_df))
        coverage_share = covered_topic_count / total_topic_count if total_topic_count else 0.0

        rows.append(
            {
                "Outlet": spec.label,
                "Articles": article_count,
                "Merged_Topics_Covered": covered_topic_count,
                "Coverage_Share": coverage_share,
                "Outlier_Articles": outlier_articles,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["Merged_Topics_Covered", "Articles", "Outlet"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _legacy_run_h1_analysis_v1(*args, **kwargs):
    """Removed: old exec-based H1 runner. Use run_h1_analysis() instead."""
    return run_h1_analysis(*args, **kwargs)


def _DELETED_run_h1_analysis(*args, **kwargs):
    """Runs all measures, prints diagnostics, and produces all thesis figures
    in the correct presentation order. Returns a structured results dict
    so the notebook requires only a single cell to execute the full H1 block.

    Thesis relevance:
        Assembles the complete evidence stack for H1 (Agenda Distortion):
        TYPE A (breadth restriction) + TYPE B (concentration distortion).

    Args:
        merged_articles: Article-level merged BERTopic dataframe.
        source_colors: Optional dict mapping outlet_label to hex color.
        top_n_topics: Number of top corpus topics for heatmap columns.
        top_n_overlap: Number of top topics per outlet for overlap comparison.
        min_articles: Minimum articles for a topic to count as covered.
        save_dir: Optional path. If provided, saves all figures as PDF into
            save_dir / "h1_figures/".

    Returns:
        dict: Structured results containing measures, overlap summary, and
        all generated figures.
    """
    import json

    try:
        from IPython.display import display
    except Exception:
        display = None

    def _resolve_h1_helpers():
        helper_names = [
            "compute_outlet_topic_measures",
            "compute_topic_overlap_with_tagesschau",
            "plot_outlet_topic_heatmap",
            "print_imbalance_report",
        ]
        helpers = {name: globals().get(name) for name in helper_names}
        plot_measures_func = globals().get("plot_outlet_topic_measures")

        if all(callable(helper) for helper in helpers.values()) and callable(plot_measures_func):
            return helpers, plot_measures_func

        notebook_path = MODULE_DIR / "Merged_BERTopic_All_Outlets.ipynb"
        if notebook_path.exists():
            with notebook_path.open("r", encoding="utf-8") as fh:
                notebook = json.load(fh)

            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue

                cell_source = "".join(cell.get("source", []))
                if "def compute_outlet_topic_measures" not in cell_source:
                    continue

                namespace = dict(globals())
                exec(cell_source, namespace, namespace)
                for name in helper_names:
                    helpers[name] = helpers[name] or namespace.get(name)
                plot_measures_func = plot_measures_func or namespace.get("plot_outlet_topic_measures")
                break

        missing = [name for name, helper in helpers.items() if not callable(helper)]
        if missing:
            raise NameError(
                "run_h1_analysis requires the following helper functions to be defined: "
                + ", ".join(missing)
            )

        return helpers, plot_measures_func

    def _plot_outlet_topic_measures_local(measures_df, source_colors=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150, sharey=True)
        fig.patch.set_facecolor("white")

        plot_df = (
            measures_df.copy()
            .sort_values(["entropy_normalized", "outlet_label"], ascending=[True, True])
            .reset_index(drop=True)
        )
        y_pos = np.arange(len(plot_df))
        colors = [
            source_colors.get(label, "#C44E52") if source_colors is not None else "#C44E52"
            for label in plot_df["outlet_label"]
        ]

        panels = [
            (
                "entropy_normalized",
                "Entropy (normalized)",
                "Lower = more concentrated topic distribution",
                None,
            ),
            (
                "coverage_breadth_relative",
                "Coverage Breadth Relative",
                "Lower than 1.0 = narrower than corpus size predicts",
                1.0,
            ),
            (
                "kl_from_tagesschau",
                "KL Divergence from Tagesschau",
                "Higher = greater divergence from mainstream topic priorities",
                0.0,
            ),
            (
                "topic_dominance_score",
                "Topic Dominance Score",
                "Higher = stronger within-topic agenda dominance",
                None,
            ),
        ]

        for ax, (column, title, xlabel, reference_line) in zip(axes.flat, panels):
            ax.set_facecolor("#FFFFFF")
            ax.barh(y_pos, plot_df[column], color=colors, alpha=0.9)
            if reference_line is not None:
                ax.axvline(reference_line, color="#777777", linestyle="--", linewidth=1.1)
            ax.set_title(title, fontsize=12, weight="semibold", pad=10)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.grid(axis="x", color="#E0E0E0", linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(plot_df["outlet_label"])
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(plot_df["outlet_label"])
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels([])
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([])

        for ax in axes.flat:
            ax.invert_yaxis()

        fig.tight_layout()
        return fig

    helpers, plot_outlet_topic_measures_func = _resolve_h1_helpers()
    compute_outlet_topic_measures_func = helpers["compute_outlet_topic_measures"]
    compute_topic_overlap_with_tagesschau_func = helpers["compute_topic_overlap_with_tagesschau"]
    plot_outlet_topic_heatmap_func = helpers["plot_outlet_topic_heatmap"]
    print_imbalance_report_func = helpers["print_imbalance_report"]

    measures_df = compute_outlet_topic_measures_func(merged_articles, min_articles=min_articles)

    print("=" * 70)
    print("H1 ANALYSIS -- CORPUS IMBALANCE REPORT")
    print("=" * 70)
    print_imbalance_report_func(measures_df)

    overlap_df = compute_topic_overlap_with_tagesschau_func(merged_articles, top_n=top_n_overlap)

    print()
    print("=" * 70)
    print(f"H1 ANALYSIS -- TOPIC OVERLAP WITH TAGESSCHAU (top {top_n_overlap} topics)")
    print("=" * 70)
    outlet_width = max(
        len("Outlet"),
        int(overlap_df["outlet_label"].astype(str).map(len).max()) if not overlap_df.empty else 0,
    )
    print(f"{'Outlet':<{outlet_width}} {'Overlap':>7} {'Share':>7}  Divergent topics")
    for row in (
        overlap_df.sort_values(["overlap_share", "outlet_label"], ascending=[True, True]).itertuples(index=False)
    ):
        divergent_topics = list(row.divergent_topics) if isinstance(row.divergent_topics, (list, tuple)) else []
        divergent_preview = " | ".join(str(topic) for topic in divergent_topics[:3]) or "--"
        if len(divergent_topics) > 3:
            divergent_preview = f"{divergent_preview} ..."
        print(
            f"{row.outlet_label:<{outlet_width}} "
            f"{int(row.overlap_count):>7} "
            f"{float(row.overlap_share):>6.0%}  "
            f"{divergent_preview}"
        )

    if callable(plot_outlet_topic_measures_func):
        fig_measures = plot_outlet_topic_measures_func(measures_df, source_colors=source_colors)
    else:
        fig_measures = _plot_outlet_topic_measures_local(measures_df, source_colors=source_colors)
    fig_measures.patch.set_facecolor("white")
    fig_measures.suptitle(
        "H1 Evidence: Agenda Distortion Measures per Outlet",
        fontsize=16,
        weight="bold",
        y=1.01,
    )
    fig_measures.tight_layout()
    if display is not None:
        display(fig_measures)

    fig_heatmap_outlet = plot_outlet_topic_heatmap_func(
        merged_articles,
        top_n_topics=top_n_topics,
        normalize="outlet",
        source_colors=source_colors,
        measures_df=measures_df,
        min_label=10,
    )
    fig_heatmap_outlet.patch.set_facecolor("white")
    fig_heatmap_outlet.text(
        0.5,
        -0.02,
        "Rows sorted by entropy_normalized ascending (most distorted at top). "
        "Darker = higher share of outlet's own articles in that topic (TYPE B). "
        "White = absent or below min threshold (TYPE A).",
        ha="center",
        fontsize=9,
        color="#555555",
        transform=fig_heatmap_outlet.transFigure,
    )
    fig_heatmap_outlet.tight_layout()
    if display is not None:
        display(fig_heatmap_outlet)

    fig_heatmap_topic = plot_outlet_topic_heatmap_func(
        merged_articles,
        top_n_topics=top_n_topics,
        normalize="topic",
        source_colors=source_colors,
        measures_df=measures_df,
        min_label=10,
    )
    fig_heatmap_topic.patch.set_facecolor("white")
    fig_heatmap_topic.text(
        0.5,
        -0.02,
        "Darker = outlet accounts for a larger share of that topic's total corpus articles. "
        "Reveals agenda monopolization -- one outlet dominating a topic cluster.",
        ha="center",
        fontsize=9,
        color="#555555",
        transform=fig_heatmap_topic.transFigure,
    )
    fig_heatmap_topic.tight_layout()
    if display is not None:
        display(fig_heatmap_topic)

    total_topics = int(merged_articles.loc[merged_articles["merged_topic"] != -1, "merged_topic"].nunique())
    baseline = (top_n_overlap / total_topics) if total_topics else 0.0
    overlap_plot_df = (
        overlap_df.sort_values(["overlap_share", "outlet_label"], ascending=[True, True]).reset_index(drop=True)
    )

    fig_overlap, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig_overlap.patch.set_facecolor("white")
    ax.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(overlap_plot_df))
    overlap_colors = [
        source_colors.get(label, "#C44E52") if source_colors is not None else "#C44E52"
        for label in overlap_plot_df["outlet_label"]
    ]
    ax.barh(y_pos, overlap_plot_df["overlap_share"], color=overlap_colors, alpha=0.9)
    ax.axvline(baseline, color="#777777", linestyle="--", linewidth=1.1)
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(overlap_plot_df["outlet_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of outlet's top topics also in Tagesschau's top topics")
    ax.set_title(
        f"Topic Overlap with Tagesschau Top {top_n_overlap} Topics",
        fontsize=14,
        weight="bold",
        pad=18,
    )
    ax.text(
        0.0,
        1.02,
        "Higher = alternative outlet prioritizes similar topics to mainstream",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for idx, row in overlap_plot_df.iterrows():
        x_pos = min(float(row["overlap_share"]) + 0.02, 0.98)
        ax.text(
            x_pos,
            idx,
            f"{int(row['overlap_count'])}/{top_n_overlap} topics shared",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
        )

    fig_overlap.tight_layout()
    if display is not None:
        display(fig_overlap)

    if save_dir is not None:
        figure_dir = Path(save_dir) / "h1_figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        save_plan = [
            ("h1_01_measures_barchart.pdf", fig_measures),
            ("h1_02_heatmap_outlet_normalized.pdf", fig_heatmap_outlet),
            ("h1_03_heatmap_topic_normalized.pdf", fig_heatmap_topic),
            ("h1_04_topic_overlap.pdf", fig_overlap),
        ]
        for filename, fig in save_plan:
            filepath = figure_dir / filename
            fig.savefig(filepath, bbox_inches="tight")
            print(f"Saved: {filepath}")

    return {
        "measures_df": measures_df,
        "overlap_df": overlap_df,
        "fig_measures": fig_measures,
        "fig_heatmap_outlet": fig_heatmap_outlet,
        "fig_heatmap_topic": fig_heatmap_topic,
        "fig_overlap": fig_overlap,
    }
    # H1 evidence type: BOTH


# ─────────────────────────────────────────────────────────────────────────────
# THESIS DESIGN CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

THESIS_COLORS = {
    "Tagesschau": "#4878CF",
    "RT": "#B22222",
    "Antispiegel": "#D4472A",
    "Tichys Einblick": "#6A994E",
    "Nius": "#E07B39",
    "Compact": "#7B5EA7",
    "Deutschlandkurier": "#3A7D7B",
}
# Semantic grouping: reds = pro-Russian, greens/ambers = right-populist,
# purples/teals = right-wing, blue = mainstream reference.
# All colors are print-safe and distinguishable in greyscale.

THESIS_RC = {
    "font.family": "serif",
    "font.serif": ["Georgia", "DejaVu Serif", "serif"],
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "#FFFFFF",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "grid.color": "#E8E8E8",
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.labelcolor": "#333333",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
}
# Always use via: with plt.rc_context(THESIS_RC): ...
# Never set globally -- always scope per figure.


def _thesis_axis_style(ax) -> None:
    """Applies consistent MSc thesis spine/grid style to a single Axes.

    Removes top/right spines, softens remaining spines, adds light x-grid.
    Call immediately after subplot creation, before adding data.

    Args:
        ax: matplotlib Axes object to style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.7, linestyle="-")
    ax.grid(axis="y", visible=False)


def _thesis_bar_labels(
    ax,
    values,
    positions,
    fmt: str = "{:.2f}",
    offset: float = 0.008,
) -> None:
    """Adds clean value labels at the right end of horizontal bars.

    Args:
        ax: matplotlib Axes containing horizontal bars.
        values: iterable of float -- bar widths (the plotted values).
        positions: iterable of float -- y-coordinates of bar centers.
        fmt: Python format string for the label text.
        offset: horizontal gap between bar tip and label text.
    """
    for val, pos in zip(values, positions):
        if pd.isna(val):
            continue
        ax.text(
            float(val) + offset,
            float(pos),
            fmt.format(float(val)),
            va="center",
            ha="left",
            fontsize=8,
            color="#555555",
        )


def _resolve_colors(source_colors) -> dict:
    """Returns source_colors if provided, else falls back to THESIS_COLORS.

    Args:
        source_colors: dict mapping outlet_label to hex color string, or None.

    Returns:
        dict: guaranteed non-None color mapping.
    """
    if source_colors is not None:
        return source_colors
    return THESIS_COLORS


def _save_thesis_figure(fig, path) -> None:
    """Saves a figure as a thesis-ready PDF.

    Always uses dpi=300, white background, tight bounding box.
    Creates parent directories if they do not exist.

    Args:
        fig: matplotlib Figure to save.
        path: pathlib.Path or str -- should end in .pdf
    """
    from pathlib import Path as _Path

    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    fig.savefig(
        path,
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
        edgecolor="none",
    )
    print(f"Saved: {path}")


def compute_outlet_topic_measures(merged_articles, min_articles=10) -> pd.DataFrame:
    """Computes four complementary measures of agenda distortion per outlet.

    Designed to separate TYPE A (breadth restriction) from TYPE B
    (concentration distortion).

    Thesis relevance:
        Directly operationalizes H1 (Agenda Distortion).

    Args:
        merged_articles: Article-level merged BERTopic assignments.
        min_articles: Minimum number of articles required for a topic to count
            as covered by an outlet.

    Returns:
        pd.DataFrame: Outlet-level H1 measures.
    """
    import math

    def _binomial_tail_prob_ge_k(n, p, k, scipy_state):
        if k <= 0:
            return 1.0
        if n < k or p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0

        try:
            if scipy_state["binom"] is None:
                from scipy.stats import binom

                scipy_state["binom"] = binom
            return float(1.0 - scipy_state["binom"].cdf(k - 1, n, p))
        except Exception:
            if not scipy_state["warned"]:
                warnings.warn(
                    "scipy is unavailable; using a recursive binomial tail fallback for coverage_breadth_expected.",
                    RuntimeWarning,
                )
                scipy_state["warned"] = True

            q = 1.0 - p
            log_p0 = n * math.log(q) if q > 0 else float("-inf")
            if log_p0 < -745:
                return 1.0

            pmf = math.exp(log_p0)
            cdf = pmf
            upper = min(k - 1, n)
            for j in range(0, upper):
                ratio = ((n - j) / (j + 1)) * (p / q) if q > 0 else float("inf")
                pmf *= ratio
                cdf += pmf
            return float(min(max(1.0 - cdf, 0.0), 1.0))

    topic_df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic"],
    ].copy()
    columns = [
        "outlet_label",
        "n_articles",
        "corpus_share",
        "entropy",
        "entropy_normalized",
        "coverage_breadth_raw",
        "coverage_breadth_expected",
        "coverage_breadth_relative",
        "kl_from_tagesschau",
        "topic_dominance_score",
    ]
    if topic_df.empty:
        return pd.DataFrame(columns=columns)
        # H1 evidence type: BOTH

    all_outlets = sorted(merged_articles["outlet_label"].dropna().unique().tolist())
    all_topics = sorted(topic_df["merged_topic"].unique().tolist())
    total_topics = len(all_topics)

    outlet_topic_counts = (
        topic_df.groupby(["outlet_label", "merged_topic"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_outlets, columns=all_topics, fill_value=0)
        .astype(float)
    )
    outlet_topic_totals = outlet_topic_counts.sum(axis=1).astype(int)
    outlet_article_totals = (
        merged_articles.groupby("outlet_label")
        .size()
        .reindex(all_outlets, fill_value=0)
        .astype(int)
    )
    topic_totals = outlet_topic_counts.sum(axis=0).astype(float)
    corpus_topic_total = float(outlet_topic_totals.sum())
    corpus_topic_share = topic_totals / corpus_topic_total if corpus_topic_total else topic_totals * 0.0

    tagesschau_label = "Tagesschau"
    if tagesschau_label not in outlet_topic_counts.index:
        raise KeyError("Tagesschau not found in merged_articles['outlet_label'].")

    tag_counts = outlet_topic_counts.loc[tagesschau_label].astype(float)
    tag_probs = (tag_counts + 1.0) / (float(tag_counts.sum()) + total_topics)

    scipy_state = {"binom": None, "warned": False}
    rows = []
    corpus_article_total = int(len(merged_articles))

    for outlet_label in all_outlets:
        counts = outlet_topic_counts.loc[outlet_label].astype(float)
        n_articles = int(outlet_article_totals.loc[outlet_label])
        n_topic_articles = int(outlet_topic_totals.loc[outlet_label])
        corpus_share = n_articles / corpus_article_total if corpus_article_total else float("nan")

        if n_topic_articles > 0:
            probs = counts / n_topic_articles
            nonzero_probs = probs[probs > 0]
            # Entropy shape is size-independent, but estimates are noisier for n < 1000.
            entropy = float(-(nonzero_probs * nonzero_probs.map(math.log)).sum())
            entropy_normalized = entropy / math.log(total_topics) if total_topics > 1 else 0.0
        else:
            entropy = float("nan")
            entropy_normalized = float("nan")

        covered_topics = int((counts >= min_articles).sum())
        coverage_breadth_raw = covered_topics / total_topics if total_topics else float("nan")

        expected_covered = 0.0
        if n_topic_articles > 0 and total_topics > 0:
            for p_topic in corpus_topic_share.tolist():
                expected_covered += _binomial_tail_prob_ge_k(
                    n_topic_articles,
                    float(p_topic),
                    min_articles,
                    scipy_state,
                )
        coverage_breadth_expected = expected_covered / total_topics if total_topics else float("nan")
        # Primary H1 breadth measure -- size-controlled.
        coverage_breadth_relative = (
            coverage_breadth_raw / coverage_breadth_expected
            if coverage_breadth_expected and not pd.isna(coverage_breadth_expected)
            else float("nan")
        )

        if n_topic_articles > 0:
            smooth_counts = counts + 1.0
            smooth_probs = smooth_counts / smooth_counts.sum()
            ratio = smooth_probs / tag_probs
            # KL compares distribution shapes, not sizes -- relatively robust, but noisier for small n.
            kl_from_tagesschau = float((smooth_probs * ratio.map(math.log)).sum())
            if outlet_label == tagesschau_label:
                kl_from_tagesschau = 0.0
        else:
            kl_from_tagesschau = float("nan")

        qualifying_counts = counts[counts >= min_articles]
        if qualifying_counts.empty:
            topic_dominance_score = float("nan")
        else:
            # Size-controlled dominance -- best for TYPE B detection.
            topic_dominance_score = float(
                (qualifying_counts / topic_totals.reindex(qualifying_counts.index)).mean()
            )

        rows.append(
            {
                "outlet_label": outlet_label,
                "n_articles": n_articles,
                "corpus_share": corpus_share,
                "entropy": entropy,
                "entropy_normalized": entropy_normalized,
                "coverage_breadth_raw": coverage_breadth_raw,
                "coverage_breadth_expected": coverage_breadth_expected,
                "coverage_breadth_relative": coverage_breadth_relative,
                "kl_from_tagesschau": kl_from_tagesschau,
                "topic_dominance_score": topic_dominance_score,
            }
        )

    return pd.DataFrame(rows).sort_values("outlet_label").reset_index(drop=True)
    # H1 evidence type: BOTH


def compute_topic_overlap_with_tagesschau(merged_articles, top_n=10) -> pd.DataFrame:
    """Computes top-topic overlap with Tagesschau for each alternative outlet.

    Low overlap indicates agenda divergence from the mainstream reference.
    The comparison is based on ranked topic lists rather than raw counts,
    so it is not confounded by outlet corpus size in the same way as raw
    coverage counts.

    Thesis relevance:
        Simplest direct test of H1 -- do alternative outlets prioritize the
        same topics as the mainstream reference?

    Args:
        merged_articles: Article-level merged BERTopic assignments.
        top_n: Number of top topics per outlet to compare.

    Returns:
        pd.DataFrame: One row per non-Tagesschau outlet with overlap metrics.
    """
    topic_df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic", "merged_display_label"],
    ].copy()
    columns = [
        "outlet_label",
        "top_n_topics",
        "tagesschau_top_n",
        "overlap_count",
        "overlap_share",
        "divergent_topics",
    ]
    if topic_df.empty:
        return pd.DataFrame(columns=columns)
        # H1 evidence type: BOTH

    label_map = {}
    for topic_id, label in (
        topic_df[["merged_topic", "merged_display_label"]]
        .drop_duplicates(subset=["merged_topic"])
        .itertuples(index=False)
    ):
        label_map[topic_id] = label if pd.notna(label) else str(topic_id)

    counts_df = (
        topic_df.groupby(["outlet_label", "merged_topic"])
        .size()
        .rename("n")
        .reset_index()
    )
    tagesschau_label = "Tagesschau"
    if tagesschau_label not in counts_df["outlet_label"].unique():
        raise KeyError("Tagesschau not found in merged_articles['outlet_label'].")

    tagesschau_top_ids = (
        counts_df.loc[counts_df["outlet_label"] == tagesschau_label]
        .sort_values(["n", "merged_topic"], ascending=[False, True])
        .head(top_n)["merged_topic"]
        .tolist()
    )
    tagesschau_top_labels = [label_map.get(topic_id, str(topic_id)) for topic_id in tagesschau_top_ids]

    rows = []
    for outlet_label in sorted(counts_df["outlet_label"].unique().tolist()):
        if outlet_label == tagesschau_label:
            continue

        outlet_top_ids = (
            counts_df.loc[counts_df["outlet_label"] == outlet_label]
            .sort_values(["n", "merged_topic"], ascending=[False, True])
            .head(top_n)["merged_topic"]
            .tolist()
        )
        overlap_ids = [topic_id for topic_id in outlet_top_ids if topic_id in tagesschau_top_ids]
        divergent_ids = [topic_id for topic_id in outlet_top_ids if topic_id not in tagesschau_top_ids]

        rows.append(
            {
                "outlet_label": outlet_label,
                "top_n_topics": [label_map.get(topic_id, str(topic_id)) for topic_id in outlet_top_ids],
                "tagesschau_top_n": tagesschau_top_labels,
                "overlap_count": len(overlap_ids),
                "overlap_share": len(overlap_ids) / len(outlet_top_ids) if outlet_top_ids else float("nan"),
                "divergent_topics": [label_map.get(topic_id, str(topic_id)) for topic_id in divergent_ids],
            }
        )

    return pd.DataFrame(rows).sort_values(["overlap_share", "outlet_label"]).reset_index(drop=True)
    # H1 evidence type: BOTH


def plot_outlet_topic_measures(measures_df, source_colors=None):
    """Plots the core H1 outlet-level distortion measures."""
    colors = _resolve_colors(source_colors)
    plot_df = (
        measures_df.copy()
        .sort_values(["entropy_normalized", "outlet_label"], ascending=[True, True])
        .reset_index(drop=True)
    )
    y_pos = np.arange(len(plot_df))
    color_values = [colors.get(label, "#999999") for label in plot_df["outlet_label"]]
    label_values = plot_df["outlet_label"].tolist()
    kl_tick_labels = [
        f"{label} ⚠" if int(n_articles) < 1000 else label
        for label, n_articles in zip(plot_df["outlet_label"], plot_df["n_articles"])
    ]
    panel_specs = [
        (
            "coverage_breadth_relative",
            "Coverage Breadth Relative",
            "Relative breadth",
            "{:.2f}",
        ),
        (
            "entropy_normalized",
            "Topic Entropy (normalized)",
            "Entropy (0–1)",
            "{:.2f}",
        ),
        (
            "kl_from_tagesschau",
            "KL Divergence from Tagesschau",
            "KL divergence",
            "{:.2f}",
        ),
        (
            "topic_dominance_score",
            "Topic Dominance Score",
            "Mean within-topic share",
            "{:.2f}",
        ),
    ]

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=150, sharey=False)
        fig.patch.set_facecolor("white")

        for ax, (column, title, xlabel, fmt) in zip(axes, panel_specs):
            _thesis_axis_style(ax)
            bar_values = plot_df[column].astype(float).to_numpy()
            bar_positions = y_pos
            ax.barh(
                bar_positions,
                bar_values,
                height=0.55,
                color=color_values,
                linewidth=0,
                alpha=0.92,
            )
            ax.set_yticks(bar_positions)
            if column == "kl_from_tagesschau":
                ax.set_yticklabels(kl_tick_labels)
                ax.annotate(
                    "⚠ n < 1,000: KL estimate less stable",
                    xy=(0, -0.18),
                    xycoords="axes fraction",
                    fontsize=8,
                    color="#888888",
                    ha="left",
                )
            else:
                ax.set_yticklabels(label_values)

            if column == "coverage_breadth_relative":
                ax.axvline(
                    1.0,
                    color="#AAAAAA",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=0,
                )
            ax.set_title(title, fontsize=12, fontweight="semibold", color="#1a1a1a", pad=10)
            ax.set_xlabel(xlabel, fontsize=10, color="#333333")
            _thesis_bar_labels(ax, bar_values, bar_positions, fmt=fmt)
            ax.invert_yaxis()

        fig.tight_layout()
    return fig


def plot_outlet_topic_heatmap(
    merged_articles,
    top_n_topics=25,
    normalize="outlet",
    source_colors=None,
    measures_df=None,
    classification_df=None,
    min_label=10,
):
    """Plots an outlet-by-topic heatmap for H1 analysis.

    Rows are outlets and columns are the most frequent merged topics in the
    corpus plus an optional "Other topics" column. The plot can emphasize
    within-outlet concentration or within-topic outlet dominance depending
    on the selected normalization.

    Thesis relevance:
        Single figure that shows both breadth (how many topics an outlet shows
        up in) and concentration (how strongly it clusters in a subset of
        topics) at the same time.

    Args:
        merged_articles: Article-level merged BERTopic assignments.
        top_n_topics: Number of globally frequent merged topics to display.
        normalize: Either "outlet" or "topic".
        source_colors: Optional mapping from outlet label to text color.
        measures_df: Optional output of compute_outlet_topic_measures used to
            sort outlets by entropy_normalized ascending.
        classification_df: Optional classification dataframe for thesis row order.
        min_label: Minimum raw count required before a cell is annotated.

    Returns:
        matplotlib.figure.Figure: The rendered heatmap figure.
    """
    if normalize not in {"outlet", "topic"}:
        raise ValueError("normalize must be either 'outlet' or 'topic'.")

    colors = _resolve_colors(source_colors)
    topic_df = merged_articles.loc[
        merged_articles["merged_topic"] != -1,
        ["outlet_label", "merged_topic", "merged_display_label"],
    ].copy()

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(20, 6), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#FAFAFA")

        if topic_df.empty:
            ax.text(0.5, 0.5, "No non-outlier merged topics available.", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            fig.tight_layout()
            return fig
            # H1 evidence type: BOTH

        topic_label_map = {}
        for topic_id, label in (
            topic_df[["merged_topic", "merged_display_label"]]
            .drop_duplicates(subset=["merged_topic"])
            .itertuples(index=False)
        ):
            topic_label_map[topic_id] = label if pd.notna(label) else str(topic_id)

        all_outlets = sorted(topic_df["outlet_label"].dropna().unique().tolist())
        if classification_df is not None and {
            "outlet_label",
            "distortion_type",
            "kl_from_tagesschau",
        }.issubset(classification_df.columns):
            type_order = ["REFERENCE", "TYPE A", "TYPE AB", "TYPE B", "UNCLEAR"]
            type_map = {label: idx for idx, label in enumerate(type_order)}
            ranked = classification_df.loc[:, ["outlet_label", "distortion_type", "kl_from_tagesschau"]].copy()
            ranked["distortion_rank"] = ranked["distortion_type"].map(type_map).fillna(len(type_order))
            ranked = ranked.sort_values(
                ["distortion_rank", "kl_from_tagesschau", "outlet_label"],
                ascending=[True, False, True],
            )
            outlet_order = [label for label in ranked["outlet_label"].tolist() if label in all_outlets]
            outlet_order.extend([label for label in all_outlets if label not in outlet_order])
        elif measures_df is not None and {"outlet_label", "entropy_normalized"}.issubset(measures_df.columns):
            ranked = (
                measures_df[["outlet_label", "entropy_normalized"]]
                .drop_duplicates(subset=["outlet_label"])
                .sort_values(["entropy_normalized", "outlet_label"], ascending=[True, True])
            )
            outlet_order = [label for label in ranked["outlet_label"].tolist() if label in all_outlets]
            outlet_order.extend([label for label in all_outlets if label not in outlet_order])
        else:
            outlet_order = all_outlets

        topic_totals = topic_df.groupby("merged_topic").size().sort_values(ascending=False)
        selected_topic_ids = topic_totals.head(top_n_topics).index.tolist()

        outlet_topic_counts = (
            topic_df.groupby(["outlet_label", "merged_topic"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=outlet_order, fill_value=0)
            .astype(float)
        )
        outlet_non_outlier_totals = outlet_topic_counts.sum(axis=1)

        top_topic_counts = outlet_topic_counts.reindex(columns=selected_topic_ids, fill_value=0).copy()
        other_topic_counts = outlet_non_outlier_totals - top_topic_counts.sum(axis=1)
        other_share_series = other_topic_counts.div(outlet_non_outlier_totals.replace(0, pd.NA)).fillna(0.0)
        other_max_share = float(other_share_series.max()) if not other_share_series.empty else 0.0

        heat_counts = top_topic_counts.copy()
        include_other = other_max_share > 0.15
        if include_other:
            heat_counts["Other topics"] = other_topic_counts

        if normalize == "outlet":
            denom = outlet_non_outlier_totals.replace(0, pd.NA)
            heat_values = heat_counts.div(denom, axis=0).fillna(0.0)
            colorbar_label = "Share of outlet articles"
            cmap = "RdPu"
        else:
            topic_denoms = topic_totals.reindex(selected_topic_ids).replace(0, pd.NA)
            heat_values = top_topic_counts.div(topic_denoms, axis=1).fillna(0.0)
            if include_other:
                heat_values["Other topics"] = other_share_series
            colorbar_label = "Share of topic articles"
            cmap = "YlOrBr"

        column_labels = [topic_label_map.get(topic_id, str(topic_id)) for topic_id in selected_topic_ids]
        if include_other:
            column_labels = column_labels + ["Other topics"]
        heat_values.columns = column_labels
        heat_counts.columns = column_labels

        matrix = heat_values.to_numpy(dtype=float)
        vmax = float(matrix.max()) if matrix.size else 1.0
        if vmax <= 0:
            vmax = 1.0

        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(column_labels)))
        ax.set_xticklabels(column_labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(outlet_order)))
        ax.set_yticklabels(outlet_order)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(10)
            lbl.set_fontweight("semibold")
            if lbl.get_text() in colors:
                lbl.set_color(colors[lbl.get_text()])

        ncols = len(column_labels)
        nrows = len(outlet_order)
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        threshold = 0.55 * vmax
        for row_idx in range(len(outlet_order)):
            for col_idx in range(len(column_labels)):
                raw_count = int(heat_counts.iat[row_idx, col_idx])
                if raw_count < min_label:
                    continue
                cell_value = float(heat_values.iat[row_idx, col_idx])
                ax.text(
                    col_idx,
                    row_idx,
                    f"{raw_count}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if cell_value >= threshold else "#333333",
                )

        ax.set_title(
            f"Outlet × Topic Heatmap in Merged Topic Space ({normalize}-normalized)",
            fontsize=12,
            fontweight="semibold",
            color="#1a1a1a",
            pad=10,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        colorbar.set_label(colorbar_label, fontsize=10, color="#333333")
        fig.tight_layout()
    return fig
    # H1 evidence type: BOTH


def print_imbalance_report(measures_df) -> None:
    """Prints a formatted corpus-imbalance diagnostic table.

    Run this before reporting H1 results so that outlets with noisier
    estimates are flagged explicitly in the methods section.

    Args:
        measures_df: Output of compute_outlet_topic_measures.

    Returns:
        None
    """
    required_columns = [
        "outlet_label",
        "n_articles",
        "corpus_share",
        "entropy_normalized",
        "coverage_breadth_relative",
        "kl_from_tagesschau",
        "topic_dominance_score",
    ]
    missing = [column for column in required_columns if column not in measures_df.columns]
    if missing:
        raise KeyError(f"measures_df is missing required columns: {missing}")

    def _fmt_float(value):
        return "nan" if pd.isna(value) else f"{value:.3f}"

    working = measures_df.loc[:, required_columns].copy()
    working = working.sort_values(["n_articles", "outlet_label"], ascending=[False, True]).reset_index(drop=True)

    header = (
        f"{'outlet':<20} {'n_articles':>10} {'corpus_share':>12} {'flag':>9} "
        f"{'entropy_norm':>14} {'breadth_rel':>14} {'kl_from_ts':>12} {'dominance':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in working.itertuples(index=False):
        flag = "⚠ SMALL" if row.n_articles < 1000 else "✓ OK"
        print(
            f"{str(row.outlet_label):<20} "
            f"{int(row.n_articles):>10,} "
            f"{row.corpus_share:>12.1%} "
            f"{flag:>9} "
            f"{_fmt_float(row.entropy_normalized):>14} "
            f"{_fmt_float(row.coverage_breadth_relative):>14} "
            f"{_fmt_float(row.kl_from_tagesschau):>12} "
            f"{_fmt_float(row.topic_dominance_score):>12}"
        )

    print()
    print(
        "Measures most robust to corpus imbalance: coverage_breadth_relative, "
        "topic_dominance_score (both size-controlled). Use entropy and KL "
        "divergence for large outlets only, or report with caveat for n < 1000."
    )
    # H1 evidence type: BOTH


def h1_classify_outlets(measures_df, overlap_df):
    """Classifies outlets into H1 distortion types."""
    classification_df = measures_df.copy()
    overlap_cols = ["outlet_label", "overlap_count", "overlap_share", "divergent_topics"]
    available_overlap_cols = [col for col in overlap_cols if col in overlap_df.columns]
    classification_df = classification_df.merge(
        overlap_df.loc[:, available_overlap_cols],
        on="outlet_label",
        how="left",
    )
    classification_df["overlap_share"] = classification_df["overlap_share"].fillna(1.0)
    classification_df["dominance_ratio"] = classification_df["topic_dominance_score"].div(
        classification_df["corpus_share"].replace(0, pd.NA)
    )
    alt_df = classification_df.loc[classification_df["outlet_label"] != "Tagesschau"].copy()
    kl_cutoff = float(alt_df["kl_from_tagesschau"].median()) if not alt_df.empty else 0.0
    dominance_cutoff = float(alt_df["dominance_ratio"].median()) if not alt_df.empty else 1.0
    entropy_cutoff = float(alt_df["entropy_normalized"].median()) if not alt_df.empty else 1.0

    distortion_types = []
    rationales = []
    for row in classification_df.itertuples(index=False):
        if row.outlet_label == "Tagesschau":
            distortion_types.append("REFERENCE")
            rationales.append("Mainstream reference outlet.")
            continue

        type_a_signal = pd.notna(row.coverage_breadth_relative) and row.coverage_breadth_relative < 0.90
        type_b_signal = (
            (pd.notna(row.dominance_ratio) and row.dominance_ratio > max(1.15, dominance_cutoff))
            or (pd.notna(row.kl_from_tagesschau) and row.kl_from_tagesschau >= kl_cutoff and row.kl_from_tagesschau > 0)
            or (pd.notna(row.entropy_normalized) and row.entropy_normalized < entropy_cutoff and row.entropy_normalized < 0.85)
        )

        if type_a_signal and type_b_signal:
            distortion_type = "TYPE AB"
            rationale = "Breadth restriction plus concentrated divergence from Tagesschau."
        elif type_a_signal:
            distortion_type = "TYPE A"
            rationale = "Narrower topic breadth than corpus size predicts."
        elif type_b_signal:
            distortion_type = "TYPE B"
            rationale = "Broad topic reach, but concentrated or dominant emphasis within selected topics."
        else:
            distortion_type = "UNCLEAR"
            rationale = "No strong distortion signal on breadth or concentration thresholds."

        distortion_types.append(distortion_type)
        rationales.append(rationale)

    classification_df["distortion_type"] = distortion_types
    classification_df["classification_rationale"] = rationales
    type_order = ["REFERENCE", "TYPE A", "TYPE AB", "TYPE B", "UNCLEAR"]
    classification_df["distortion_rank"] = classification_df["distortion_type"].map(
        {label: idx for idx, label in enumerate(type_order)}
    )
    classification_df = classification_df.sort_values(
        ["distortion_rank", "kl_from_tagesschau", "outlet_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return classification_df


def h1_evidence_table(classification_df):
    """Builds a clean evidence table for thesis export."""
    table_cols = [
        "outlet_label",
        "distortion_type",
        "n_articles",
        "corpus_share",
        "coverage_breadth_relative",
        "entropy_normalized",
        "kl_from_tagesschau",
        "topic_dominance_score",
        "overlap_share",
        "classification_rationale",
    ]
    available_cols = [col for col in table_cols if col in classification_df.columns]
    return classification_df.loc[:, available_cols].copy()


def h1_narrative_summary(classification_df, measures_df):
    """Creates a short narrative summary for H1 interpretation."""
    rows = []
    for row in classification_df.itertuples(index=False):
        if row.outlet_label == "Tagesschau":
            continue
        overlap_share = getattr(row, "overlap_share", float("nan"))
        breadth = getattr(row, "coverage_breadth_relative", float("nan"))
        entropy_norm = getattr(row, "entropy_normalized", float("nan"))
        kl_value = getattr(row, "kl_from_tagesschau", float("nan"))
        dominance = getattr(row, "topic_dominance_score", float("nan"))
        rows.append(
            (
                f"{row.outlet_label}: {row.distortion_type}. "
                f"Breadth relative = {breadth:.2f}; entropy = {entropy_norm:.2f}; "
                f"KL from Tagesschau = {kl_value:.2f}; dominance = {dominance:.2f}; "
                f"top-topic overlap with Tagesschau = {overlap_share:.0%}. "
                f"{row.classification_rationale}"
            )
        )
    return "\n".join(rows)


def h1_plot_classification_radar(classification_df, source_colors=None):
    """Plots thesis-style radar charts for outlet distortion classification."""
    colors = _resolve_colors(source_colors)
    plot_df = classification_df.loc[classification_df["outlet_label"] != "Tagesschau"].copy()
    if plot_df.empty:
        with plt.rc_context(THESIS_RC):
            fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
            ax.text(0.5, 0.5, "No alternative outlets available.", ha="center", va="center")
            ax.set_axis_off()
        return fig

    ref_row = classification_df.loc[classification_df["outlet_label"] == "Tagesschau"]
    if ref_row.empty:
        raise KeyError("Tagesschau reference row not found in classification_df.")
    ref_row = ref_row.iloc[0]

    work_df = classification_df.copy()
    work_df["breadth_restriction_signal"] = (
        1.0 - work_df["coverage_breadth_relative"].clip(lower=0.0, upper=1.5).div(1.5)
    )
    work_df["concentration_signal"] = 1.0 - work_df["entropy_normalized"].clip(lower=0.0, upper=1.0)
    kl_max = float(work_df["kl_from_tagesschau"].max()) if pd.notna(work_df["kl_from_tagesschau"]).any() else 1.0
    if kl_max <= 0:
        kl_max = 1.0
    work_df["kl_signal"] = work_df["kl_from_tagesschau"].fillna(0.0).div(kl_max).clip(lower=0.0, upper=1.0)
    work_df["dominance_ratio"] = work_df["topic_dominance_score"].div(work_df["corpus_share"].replace(0, pd.NA))
    ratio_max = float(work_df["dominance_ratio"].replace([np.inf, -np.inf], np.nan).max())
    if pd.isna(ratio_max) or ratio_max <= 1.0:
        ratio_max = 1.0
    work_df["dominance_signal"] = (
        (work_df["dominance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0) - 1.0) / max(ratio_max - 1.0, 1e-9)
    ).clip(lower=0.0, upper=1.0)
    work_df["overlap_signal"] = 1.0 - work_df["overlap_share"].fillna(1.0).clip(lower=0.0, upper=1.0)

    metrics = [
        ("breadth_restriction_signal", "Breadth restriction"),
        ("concentration_signal", "Concentration"),
        ("kl_signal", "KL divergence"),
        ("dominance_signal", "Topic dominance"),
        ("overlap_signal", "Low overlap"),
    ]
    metric_cols = [metric for metric, _ in metrics]
    metric_labels = [label for _, label in metrics]
    angle_values = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angle_values, [angle_values[0]]])

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150, subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("white")
        axes_flat = np.array(axes).reshape(-1)

        ref_values = work_df.loc[work_df["outlet_label"] == "Tagesschau", metric_cols].iloc[0].to_numpy(dtype=float)
        ref_values = np.concatenate([ref_values, [ref_values[0]]])

        for ax, row in zip(axes_flat, plot_df.itertuples(index=False)):
            outlet_color = colors.get(row.outlet_label, "#777777")
            outlet_values = work_df.loc[work_df["outlet_label"] == row.outlet_label, metric_cols].iloc[0].to_numpy(dtype=float)
            outlet_values = np.concatenate([outlet_values, [outlet_values[0]]])

            ax.set_facecolor("#FAFAFA")
            ax.set_ylim(0.0, 1.0)
            ax.spines["polar"].set_visible(False)
            ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
            ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.6)
            ax.set_rticks([1 / 3, 2 / 3, 1.0])
            ax.set_yticklabels(["", "", ""])
            ax.set_xticks(angle_values)
            ax.set_xticklabels(metric_labels, fontsize=7, color="#888888")

            ax.fill(angles, ref_values, color="#4878CF", alpha=0.08, linewidth=0)
            ax.plot(angles, outlet_values, color=outlet_color, linewidth=2.0)

            ax.set_title(
                f"{row.outlet_label}",
                fontsize=10,
                fontweight="semibold",
                color="#1a1a1a",
                pad=10,
            )
            ax.text(
                0.5,
                1.06,
                f"({row.distortion_type})",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                style="italic",
                color="#555555",
            )
            ax.annotate(
                "●",
                xy=(0.90, 0.92),
                xycoords="axes fraction",
                fontsize=14,
                color=outlet_color,
                ha="center",
            )

        for ax in axes_flat[len(plot_df):]:
            ax.set_axis_off()

        fig.tight_layout()
    return fig


def run_h1_argument(
    merged_articles,
    source_colors=None,
    top_n_topics=25,
    top_n_overlap=10,
    min_articles=10,
    save_dir=None,
) -> dict:
    """Runs the thesis H1 argument workflow without duplicate figures."""
    colors = _resolve_colors(source_colors)

    # ── Measures ──────────────────────────────────────────────────────────────
    measures_df = compute_outlet_topic_measures(
        merged_articles, min_articles=min_articles
    )
    overlap_df = compute_topic_overlap_with_tagesschau(
        merged_articles, top_n=top_n_overlap
    )

    # ── Diagnostics (always first) ────────────────────────────────────────────
    print("=" * 68)
    print("H1 ANALYSIS -- CORPUS IMBALANCE REPORT")
    print("=" * 68)
    print_imbalance_report(measures_df)

    # ── Interpretation ────────────────────────────────────────────────────────
    classification_df = h1_classify_outlets(measures_df, overlap_df)
    narrative = h1_narrative_summary(classification_df, measures_df)
    evidence_table = h1_evidence_table(classification_df)

    # ── Figures -- each called EXACTLY ONCE ───────────────────────────────────
    fig_radar = h1_plot_classification_radar(
        classification_df, source_colors=colors
    )
    fig_measures = plot_outlet_topic_measures(
        measures_df, source_colors=colors
    )
    fig_heatmap_outlet = plot_outlet_topic_heatmap(
        merged_articles,
        top_n_topics=top_n_topics,
        normalize="outlet",
        source_colors=colors,
        measures_df=measures_df,
        classification_df=classification_df,
    )
    fig_heatmap_topic = plot_outlet_topic_heatmap(
        merged_articles,
        top_n_topics=top_n_topics,
        normalize="topic",
        source_colors=colors,
        measures_df=measures_df,
        classification_df=classification_df,
    )

    overlap_df_sorted = overlap_df.sort_values(
        ["overlap_share", "outlet_label"],
        ascending=[True, True],
    ).reset_index(drop=True)
    with plt.rc_context(THESIS_RC):
        fig_overlap, ax = plt.subplots(figsize=(9, 5), dpi=150)
        fig_overlap.patch.set_facecolor("white")
        ax.set_facecolor("#FAFAFA")
        _thesis_axis_style(ax)

        bar_y_positions = np.arange(len(overlap_df_sorted))
        overlap_colors = [colors.get(label, "#999999") for label in overlap_df_sorted["outlet_label"]]
        ax.barh(
            bar_y_positions,
            overlap_df_sorted["overlap_share"],
            height=0.55,
            color=overlap_colors,
            linewidth=0,
            alpha=0.92,
        )
        total_topics = int(merged_articles.loc[merged_articles["merged_topic"] != -1, "merged_topic"].nunique())
        baseline = (top_n_overlap / total_topics) if total_topics else 0.0
        ax.axvline(baseline, color="#AAAAAA", linewidth=1.0, linestyle="--", zorder=0)
        max_val = float(overlap_df["overlap_share"].max()) if not overlap_df.empty else 1.0
        ax.set_xlim(0.0, min(max_val + 0.18, 1.0))
        ax.set_yticks(bar_y_positions)
        ax.set_yticklabels(overlap_df_sorted["outlet_label"])
        ax.invert_yaxis()
        ax.set_xlabel("Share of outlet's top topics also in Tagesschau's top topics", fontsize=10, color="#333333")
        ax.set_title(
            f"Topic Overlap with Tagesschau Top {top_n_overlap} Topics",
            fontsize=12,
            fontweight="semibold",
            color="#1a1a1a",
            pad=10,
        )
        _thesis_bar_labels(
            ax,
            overlap_df_sorted["overlap_share"],
            bar_y_positions,
            fmt="{:.0%}",
        )
        fig_overlap.tight_layout()

    if save_dir is not None:
        out = Path(save_dir) / "h1_argument"
        out.mkdir(parents=True, exist_ok=True)
        _save_thesis_figure(fig_radar, out / "h1_01_radar.pdf")
        _save_thesis_figure(fig_measures, out / "h1_02_measures.pdf")
        _save_thesis_figure(fig_heatmap_outlet, out / "h1_03_heatmap_outlet.pdf")
        _save_thesis_figure(fig_heatmap_topic, out / "h1_04_heatmap_topic.pdf")
        _save_thesis_figure(fig_overlap, out / "h1_05_overlap.pdf")
        evidence_table.to_csv(out / "h1_evidence_table.csv", index=False)
        with open(out / "h1_narrative.txt", "w", encoding="utf-8") as f:
            f.write(narrative)
        print(f"\nAll H1 outputs saved to: {out}")

    return {
        "measures_df": measures_df,
        "overlap_df": overlap_df,
        "classification_df": classification_df,
        "evidence_table": evidence_table,
        "narrative": narrative,
        "fig_radar": fig_radar,
        "fig_measures": fig_measures,
        "fig_heatmap_outlet": fig_heatmap_outlet,
        "fig_heatmap_topic": fig_heatmap_topic,
        "fig_overlap": fig_overlap,
    }


def run_h1_analysis(
    merged_articles,
    source_colors=None,
    top_n_topics=25,
    top_n_overlap=10,
    min_articles=10,
    save_dir=None,
) -> dict:
    """Compatibility wrapper that routes H1 analysis through the thesis path."""
    return run_h1_argument(
        merged_articles,
        source_colors=source_colors,
        top_n_topics=top_n_topics,
        top_n_overlap=top_n_overlap,
        min_articles=min_articles,
        save_dir=save_dir,
    )
