from __future__ import annotations

import ast
import html
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = MODULE_DIR.parent


def configure_runtime_environment(project_root: Path | None = None) -> dict[str, Path]:
    """Point caches at writable repo-local directories before heavy imports happen."""

    root = (project_root or DEFAULT_PROJECT_ROOT).resolve()
    numba_cache_dir = root / ".numba_cache"
    mpl_config_dir = root / ".mplconfig"
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    return {
        "project_root": root,
        "numba_cache_dir": numba_cache_dir,
        "mpl_config_dir": mpl_config_dir,
    }


RUNTIME_ENV = configure_runtime_environment()

from umap import UMAP

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bertopic_config import BERTopicConfig
from bertopic_pipeline import prepare_documents


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MERGED_MODEL_SUBDIR = "merged_all_outlets_model"
TOPIC_NAME_OVERRIDES_FILENAME = "merged_topic_name_overrides.csv"
MERGED_ARTICLES_BASENAME = "merged_articles_with_topics"

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


def get_local_output_dir(project_root: Path) -> Path:
    output_dir = project_root / "1a_BERTopic" / "local_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_embedding_model_reference() -> str:
    """Prefer the newest cached local snapshot, otherwise fall back to the model name."""

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
        / "snapshots"
    )
    if snapshots_dir.exists():
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])
    return EMBEDDING_MODEL_NAME


def resolve_raw_data_root(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "raw",
        project_root / "data source" / "raw",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find raw data root. Tried: " + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any expected path. Tried: " + ", ".join(str(candidate) for candidate in candidates)
    )


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
    raw_root = resolve_raw_data_root(project_root)
    rt_file = resolve_existing_path(
        raw_root / "Alternative Medien" / "RT_de.xlsx",
    )
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
    raw_root = resolve_raw_data_root(project_root)
    base_dir = resolve_existing_path(
        raw_root / "Alternative Medien" / "Compact",
    )
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
    raw_root = resolve_raw_data_root(project_root)
    base_dir = resolve_existing_path(
        raw_root / "Alternative Medien" / "Nius_Rohdaten_neu",
        raw_root / "Alternative Medien" / "Nius",
    )
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
    raw_root = resolve_raw_data_root(project_root)
    base_dir = resolve_existing_path(
        raw_root / "Alternative Medien" / "Tichy's Einblick",
    )
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
    raw_root = resolve_raw_data_root(project_root)
    base_dir = resolve_existing_path(
        raw_root / "Alternative Medien" / "Antispiegel",
    )
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
    raw_root = resolve_raw_data_root(project_root)
    base_dir = resolve_existing_path(
        raw_root / "Alternative Medien" / "Tagesschau",
    )
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
    raw_root = resolve_raw_data_root(project_root)
    base_dir = resolve_existing_path(
        raw_root / "Alternative Medien" / "Deutschlandkurier",
    )
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


def resolve_merged_model_path(project_root: Path) -> Path:
    candidates = [
        get_local_output_dir(project_root),
        project_root / "1a_BERTopic" / "outputs",
        project_root / "BERTopic" / "outputs",
    ]
    return resolve_model_path(MERGED_MODEL_SUBDIR, candidates)


def load_saved_merged_model(project_root: Path):
    """Load the locally saved merged BERTopic model with a stable embedding reference."""

    from bertopic import BERTopic

    model_path = resolve_merged_model_path(project_root)
    embedding_model = resolve_embedding_model_reference()
    return BERTopic.load(model_path, embedding_model=embedding_model), model_path, embedding_model


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
        else f"Topic {int(row['DisplayTopic'])} — {row['TopicNameClean']}",
        axis=1,
    )
    return topic_info


def build_topic_name_reference(merged_topic_info: pd.DataFrame) -> pd.DataFrame:
    """Prepare a topic label table that can later receive hand-written names."""

    topic_ref = merged_topic_info.loc[merged_topic_info["Topic"] != -1].copy()
    topic_ref = topic_ref[
        ["Topic", "DisplayTopic", "TopicNameClean", "DisplayLabel", "Count"]
    ].rename(
        columns={
            "Topic": "merged_topic",
            "DisplayTopic": "merged_display_topic",
            "TopicNameClean": "keyword_label",
            "DisplayLabel": "display_label",
            "Count": "article_count",
        }
    )
    topic_ref["manual_topic_name"] = pd.NA
    topic_ref["topic_label"] = topic_ref["display_label"]
    return topic_ref


def ensure_topic_name_overrides_template(
    project_root: Path,
    merged_topic_info: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a CSV template for optional hand-curated topic names."""

    output_path = get_local_output_dir(project_root) / TOPIC_NAME_OVERRIDES_FILENAME
    if output_path.exists() and not overwrite:
        return output_path

    topic_reference = build_topic_name_reference(merged_topic_info)
    topic_reference.to_csv(output_path, index=False)
    return output_path


def load_topic_name_overrides(project_root: Path) -> pd.DataFrame | None:
    """Read the optional topic-name override file if the user has filled it in."""

    csv_path = get_local_output_dir(project_root) / TOPIC_NAME_OVERRIDES_FILENAME
    if not csv_path.exists():
        return None

    overrides = pd.read_csv(csv_path)
    required = {"merged_topic", "manual_topic_name"}
    if not required.issubset(overrides.columns):
        missing = ", ".join(sorted(required - set(overrides.columns)))
        raise ValueError(f"Topic name override file is missing required columns: {missing}")

    overrides = overrides.loc[:, [col for col in overrides.columns if col in {"merged_topic", "manual_topic_name"}]].copy()
    overrides["merged_topic"] = overrides["merged_topic"].astype(int)
    overrides["manual_topic_name"] = overrides["manual_topic_name"].astype("string").str.strip()
    overrides = overrides.loc[overrides["manual_topic_name"].notna() & (overrides["manual_topic_name"] != "")]
    return overrides if not overrides.empty else None


def apply_topic_name_overrides(
    merged_topic_info: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replace keyword-heavy display labels with optional manual topic names."""

    enriched = merged_topic_info.copy()
    enriched["topic_label"] = enriched["DisplayLabel"]
    enriched["manual_topic_name"] = pd.NA
    if overrides is None or overrides.empty:
        return enriched

    override_map = overrides.set_index("merged_topic")["manual_topic_name"].to_dict()
    enriched["manual_topic_name"] = enriched["Topic"].map(override_map).astype("string")
    named_mask = enriched["manual_topic_name"].notna() & (enriched["Topic"] != -1)
    enriched.loc[named_mask, "topic_label"] = enriched.loc[named_mask].apply(
        lambda row: f"Topic {int(row['DisplayTopic'])} — {row['manual_topic_name']}",
        axis=1,
    )
    return enriched


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


def build_article_topic_dataset(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    topic_name_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a clean article-level table with one matched merged topic per article."""

    topic_info = apply_topic_name_overrides(merged_topic_info, topic_name_overrides)
    topic_columns = topic_info[
        ["Topic", "DisplayTopic", "TopicNameClean", "DisplayLabel", "topic_label", "manual_topic_name"]
    ].rename(
        columns={
            "Topic": "merged_topic",
            "DisplayTopic": "merged_display_topic",
            "TopicNameClean": "topic_keyword_label",
            "DisplayLabel": "topic_display_label",
        }
    )

    article_topic_df = merged_articles.merge(
        topic_columns,
        on=["merged_topic", "merged_display_topic"],
        how="left",
    )

    article_topic_df["topic_label"] = article_topic_df["topic_label"].fillna("Outliers")
    article_topic_df["topic_keyword_label"] = article_topic_df["topic_keyword_label"].fillna("Outliers")
    article_topic_df["topic_display_label"] = article_topic_df["topic_display_label"].fillna("Outliers")

    preferred_columns = [
        "document_id",
        "source_name",
        "outlet_key",
        "outlet_label",
        "Title",
        "Date",
        "URL",
        "document",
        "document_length",
        "token_count",
        "merged_topic",
        "merged_display_topic",
        "topic_keyword_label",
        "manual_topic_name",
        "topic_label",
        "topic_display_label",
        "merged_probability",
        "umap_x",
        "umap_y",
        "source_file",
    ]
    ordered_columns = [col for col in preferred_columns if col in article_topic_df.columns]
    trailing_columns = [col for col in article_topic_df.columns if col not in ordered_columns]
    return article_topic_df.loc[:, ordered_columns + trailing_columns].copy()


def save_dataframe_exports(
    df: pd.DataFrame,
    output_base: Path,
) -> dict[str, Path]:
    """Persist a dataframe as CSV and, when available, parquet."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    csv_path = output_base.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    saved_paths["csv"] = csv_path

    try:
        parquet_path = output_base.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        saved_paths["parquet"] = parquet_path
    except Exception:
        pass

    return saved_paths


def export_article_topic_dataset(
    project_root: Path,
    article_topic_df: pd.DataFrame,
    *,
    basename: str = MERGED_ARTICLES_BASENAME,
) -> dict[str, Path]:
    """Write the merged article-topic dataframe to local_outputs for reuse."""

    output_base = get_local_output_dir(project_root) / basename
    return save_dataframe_exports(article_topic_df, output_base)


def load_exported_article_topic_dataset(
    project_root: Path,
    *,
    basename: str = MERGED_ARTICLES_BASENAME,
) -> pd.DataFrame | None:
    """Load a previously saved merged article-topic dataframe when available."""

    output_dir = get_local_output_dir(project_root)
    parquet_path = output_dir / f"{basename}.parquet"
    csv_path = output_dir / f"{basename}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        loaded = pd.read_csv(csv_path, low_memory=False)
        if "Date" in loaded.columns:
            loaded["Date"] = pd.to_datetime(loaded["Date"], errors="coerce", utc=True).dt.tz_convert(None)
        for column in ("merged_topic", "merged_display_topic"):
            if column in loaded.columns:
                loaded[column] = pd.to_numeric(loaded[column], errors="coerce").astype("Int64")
        for column in ("umap_x", "umap_y", "merged_probability"):
            if column in loaded.columns:
                loaded[column] = pd.to_numeric(loaded[column], errors="coerce")
        return loaded
    return None


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

    ax.set_title(f"Merged BERTopic UMAP — Top {top_n} Topics")
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
    background_point_size: int = 6,
    highlight_size_range: tuple[int, int] = (8, 60),
    other_color: str = "#AAAAAA",
    highlight_color: str = "#c62828",
    background_alpha: float = 0.10,
    highlight_alpha: float = 0.82,
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
                            alpha=0.30,
                            zorder=2,
                        )
                        ax.contour(
                            xx,
                            yy,
                            zz,
                            levels=levels,
                            colors=highlight_color,
                            alpha=0.70,
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
        f"{spec.label} — Semantic Footprint in Merged Topic Space",
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


def build_topic_prevalence_by_outlet(
    article_topic_df: pd.DataFrame,
    *,
    include_outliers: bool = False,
) -> pd.DataFrame:
    """Count topic prevalence within each outlet and convert counts to within-outlet shares."""

    prevalence_df = article_topic_df.copy()
    if not include_outliers:
        prevalence_df = prevalence_df.loc[prevalence_df["merged_topic"] != -1].copy()

    counts = (
        prevalence_df.groupby(
            ["outlet_key", "outlet_label", "merged_topic", "merged_display_topic", "topic_keyword_label", "topic_label"],
            dropna=False,
        )
        .size()
        .rename("article_count")
        .reset_index()
    )

    outlet_totals = (
        prevalence_df.groupby(["outlet_key", "outlet_label"], dropna=False)
        .size()
        .rename("outlet_total_articles")
        .reset_index()
    )

    counts = counts.merge(outlet_totals, on=["outlet_key", "outlet_label"], how="left")
    counts["article_share"] = counts["article_count"] / counts["outlet_total_articles"]
    return counts.sort_values(
        ["outlet_label", "article_share", "article_count", "merged_display_topic"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_outlet_focus_metrics(
    prevalence_df: pd.DataFrame,
    *,
    total_topic_count: int | None = None,
) -> pd.DataFrame:
    """Summarize how broad or concentrated each outlet's topic distribution is."""

    rows: list[dict[str, object]] = []
    if total_topic_count is None:
        total_topic_count = int(prevalence_df["merged_topic"].nunique())

    for (outlet_key, outlet_label), group in prevalence_df.groupby(["outlet_key", "outlet_label"], dropna=False):
        shares = group["article_share"].astype(float).to_numpy()
        shares = shares[shares > 0]
        if shares.size == 0:
            continue

        shannon_entropy = float(-(shares * np.log(shares)).sum())
        normalized_entropy = (
            shannon_entropy / np.log(total_topic_count)
            if total_topic_count and total_topic_count > 1
            else np.nan
        )
        concentration_hhi = float(np.square(shares).sum())
        effective_topic_count = float(1.0 / concentration_hhi) if concentration_hhi > 0 else np.nan

        rows.append(
            {
                "outlet_key": outlet_key,
                "outlet_label": outlet_label,
                "covered_topic_count": int(group["merged_topic"].nunique()),
                "coverage_share": int(group["merged_topic"].nunique()) / total_topic_count if total_topic_count else np.nan,
                "top_5_topic_share": float(group["article_share"].nlargest(min(5, len(group))).sum()),
                "shannon_entropy": shannon_entropy,
                "normalized_entropy": normalized_entropy,
                "topic_concentration_hhi": concentration_hhi,
                "effective_topic_count": effective_topic_count,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["coverage_share", "normalized_entropy", "effective_topic_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_outlet_vs_baseline_prevalence(
    prevalence_df: pd.DataFrame,
    *,
    outlet_key: str,
    baseline_key: str = "tagesschau",
) -> pd.DataFrame:
    """Compare one outlet's topic shares against a baseline outlet such as Tagesschau."""

    outlet_df = prevalence_df.loc[prevalence_df["outlet_key"] == outlet_key].copy()
    baseline_df = prevalence_df.loc[prevalence_df["outlet_key"] == baseline_key].copy()
    if outlet_df.empty:
        raise ValueError(f"No prevalence data found for outlet '{outlet_key}'.")
    if baseline_df.empty:
        raise ValueError(f"No prevalence data found for baseline '{baseline_key}'.")

    comparison = outlet_df.merge(
        baseline_df[
            [
                "merged_topic",
                "merged_display_topic",
                "topic_keyword_label",
                "topic_label",
                "article_count",
                "outlet_total_articles",
                "article_share",
            ]
        ].rename(
            columns={
                "article_count": "baseline_article_count",
                "outlet_total_articles": "baseline_total_articles",
                "article_share": "baseline_article_share",
                "topic_label": "baseline_topic_label",
            }
        ),
        on=["merged_topic", "merged_display_topic", "topic_keyword_label"],
        how="outer",
    )

    outlet_label = OUTLET_SPECS[outlet_key].label
    baseline_label = OUTLET_SPECS[baseline_key].label

    comparison["outlet_key"] = outlet_key
    comparison["outlet_label"] = outlet_label
    comparison["baseline_key"] = baseline_key
    comparison["baseline_label"] = baseline_label
    comparison["topic_label"] = comparison["topic_label"].fillna(comparison["baseline_topic_label"])
    comparison["article_count"] = comparison["article_count"].fillna(0).astype(int)
    comparison["baseline_article_count"] = comparison["baseline_article_count"].fillna(0).astype(int)
    comparison["outlet_total_articles"] = comparison["outlet_total_articles"].fillna(
        int(outlet_df["outlet_total_articles"].iloc[0])
    )
    comparison["baseline_total_articles"] = comparison["baseline_total_articles"].fillna(
        int(baseline_df["outlet_total_articles"].iloc[0])
    )
    comparison["article_share"] = comparison["article_count"] / comparison["outlet_total_articles"]
    comparison["baseline_article_share"] = (
        comparison["baseline_article_count"] / comparison["baseline_total_articles"]
    )
    comparison["share_diff"] = comparison["article_share"] - comparison["baseline_article_share"]

    comparison["share_ratio"] = np.where(
        comparison["baseline_article_share"] > 0,
        comparison["article_share"] / comparison["baseline_article_share"],
        np.nan,
    )
    comparison["comparison_bucket"] = np.where(
        comparison["share_diff"] > 0,
        "Amplified vs baseline",
        np.where(
            comparison["share_diff"] < 0,
            "Substituted away from baseline",
            "Same prevalence",
        ),
    )

    return comparison.sort_values(
        ["share_diff", "article_share", "baseline_article_share"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def plot_outlet_vs_baseline_prevalence(
    comparison_df: pd.DataFrame,
    *,
    top_n: int = 15,
    figsize: tuple[int, int] = (12, 8),
    positive_color: str = "#c62828",
    negative_color: str = "#1565c0",
):
    """Plot the largest share differences against a baseline as a diverging bar chart."""

    plot_df = comparison_df.copy()
    plot_df = plot_df.loc[plot_df["merged_topic"] != -1].copy()
    plot_df["absolute_share_diff"] = plot_df["share_diff"].abs()
    plot_df = plot_df.sort_values(
        ["absolute_share_diff", "share_diff", "article_share"],
        ascending=[False, False, False],
    ).head(top_n)
    plot_df = plot_df.sort_values("share_diff", ascending=True).reset_index(drop=True)
    plot_df["share_diff_pct"] = plot_df["share_diff"] * 100
    colors = np.where(plot_df["share_diff"] >= 0, positive_color, negative_color)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(plot_df["topic_label"], plot_df["share_diff_pct"], color=colors, alpha=0.85)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Percentage-point difference in topic share")
    ax.set_ylabel("")
    ax.set_title(
        f"{plot_df['outlet_label'].iloc[0]} vs {plot_df['baseline_label'].iloc[0]}: amplification (+) and substitution (-)"
    )

    for _, row in plot_df.iterrows():
        annotation = f"{row['article_share']:.1%} vs {row['baseline_article_share']:.1%}"
        x_pos = row["share_diff_pct"]
        x_text = x_pos + 0.2 if x_pos >= 0 else x_pos - 0.2
        ax.text(
            x_text,
            row["topic_label"],
            annotation,
            va="center",
            ha="left" if x_pos >= 0 else "right",
            fontsize=9,
            color="#333333",
        )

    fig.tight_layout()
    return fig, ax


def plot_outlet_vs_baseline_prevalence_scatter(
    comparison_df: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (10, 10),
    point_size_scale: float = 2800.0,
    min_point_size: float = 28.0,
    label_count_per_side: int = 6,
    min_label_share_diff: float = 0.01,
    amplify_color: str = "#c62828",
    substitute_color: str = "#1565c0",
    neutral_color: str = "#7f8c8d",
):
    """Scatter one outlet's topic shares against the baseline outlet's shares."""

    plot_df = comparison_df.loc[comparison_df["merged_topic"] != -1].copy()
    if plot_df.empty:
        raise ValueError("Comparison dataframe does not contain any non-outlier topics.")

    plot_df["combined_share"] = plot_df["article_share"] + plot_df["baseline_article_share"]
    plot_df["absolute_share_diff"] = plot_df["share_diff"].abs()
    plot_df["point_size"] = min_point_size + (plot_df["combined_share"] * point_size_scale)

    color_map = {
        "Amplified vs baseline": amplify_color,
        "Substituted away from baseline": substitute_color,
        "Same prevalence": neutral_color,
    }
    plot_df["point_color"] = plot_df["comparison_bucket"].map(color_map).fillna(neutral_color)

    outlet_label = str(plot_df["outlet_label"].iloc[0])
    baseline_label = str(plot_df["baseline_label"].iloc[0])
    max_share = float(
        max(
            plot_df["article_share"].max(),
            plot_df["baseline_article_share"].max(),
        )
    )
    axis_limit = max_share * 1.08 if max_share > 0 else 0.01

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # Diagonal reference: points above the line are amplified relative to baseline.
    ax.fill_between([0, axis_limit], [0, 0], [0, axis_limit], color=substitute_color, alpha=0.05, zorder=0)
    ax.fill_between([0, axis_limit], [0, axis_limit], [axis_limit, axis_limit], color=amplify_color, alpha=0.05, zorder=0)
    ax.plot([0, axis_limit], [0, axis_limit], linestyle="--", color="#333333", linewidth=1.2, zorder=1)

    for bucket in [
        "Substituted away from baseline",
        "Same prevalence",
        "Amplified vs baseline",
    ]:
        bucket_df = plot_df.loc[plot_df["comparison_bucket"] == bucket]
        if bucket_df.empty:
            continue
        ax.scatter(
            bucket_df["baseline_article_share"],
            bucket_df["article_share"],
            s=bucket_df["point_size"],
            c=bucket_df["point_color"],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.6,
            label=bucket,
            zorder=2,
        )

    label_candidates = pd.concat(
        [
            plot_df.loc[plot_df["share_diff"] >= min_label_share_diff].nlargest(label_count_per_side, "share_diff"),
            plot_df.loc[plot_df["share_diff"] <= -min_label_share_diff].nsmallest(label_count_per_side, "share_diff"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["merged_topic"])

    for _, row in label_candidates.iterrows():
        x_value = float(row["baseline_article_share"])
        y_value = float(row["article_share"])
        x_offset = 8
        y_offset = 8 if y_value >= x_value else -10
        va = "bottom" if y_value >= x_value else "top"
        ax.annotate(
            str(row["topic_label"]),
            xy=(x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
            va=va,
            bbox={
                "facecolor": "white",
                "edgecolor": "#DDDDDD",
                "alpha": 0.92,
                "boxstyle": "round,pad=0.18",
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#999999",
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=3,
        )

    ax.set_xlim(0, axis_limit)
    ax.set_ylim(0, axis_limit)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(f"{baseline_label} topic prevalence")
    ax.set_ylabel(f"{outlet_label} topic prevalence")
    ax.set_title(f"{outlet_label} vs {baseline_label}: topic prevalence map")
    ax.text(
        0.02,
        0.98,
        "Above diagonal: amplification\nBelow diagonal: substitution",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        bbox={
            "facecolor": "white",
            "edgecolor": "#DDDDDD",
            "alpha": 0.9,
            "boxstyle": "round,pad=0.35",
        },
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", loc="lower right")
    fig.tight_layout()
    return fig, ax


def build_all_outlet_vs_baseline_prevalence(
    prevalence_df: pd.DataFrame,
    *,
    baseline_key: str = "tagesschau",
    outlet_keys: tuple[str, ...] = ALT_MEDIA_OUTLET_KEYS,
) -> pd.DataFrame:
    """Stack outlet-vs-baseline prevalence comparisons for multiple outlets."""

    frames = [
        build_outlet_vs_baseline_prevalence(
            prevalence_df,
            outlet_key=outlet_key,
            baseline_key=baseline_key,
        )
        for outlet_key in outlet_keys
    ]
    return pd.concat(frames, ignore_index=True)


def build_salient_prevalence_map_points(
    all_comparisons_df: pd.DataFrame,
    *,
    substitution_baseline_max: float = 0.035,
    substitution_outlet_min: float = 0.07,
    amplification_baseline_min: float = 0.03,
    amplification_margin: float = 0.01,
    mainstream_only_baseline_min: float = 0.045,
    mainstream_only_margin: float = 0.01,
    parity_margin: float = 0.006,
    parity_min_baseline_share: float = 0.045,
    points_per_zone: int = 4,
) -> pd.DataFrame:
    """Select the most salient topic-outlet points for a screenshot-style prevalence map."""

    plot_df = all_comparisons_df.loc[all_comparisons_df["merged_topic"] != -1].copy()
    if plot_df.empty:
        raise ValueError("No non-outlier topics available for the prevalence map.")

    plot_df["combined_share"] = plot_df["article_share"] + plot_df["baseline_article_share"]
    plot_df["absolute_share_diff"] = plot_df["share_diff"].abs()
    plot_df["topic_short_label"] = (
        plot_df["topic_label"]
        .astype("string")
        .str.replace(r"^Topic \d+\s+[—-]\s+", "", regex=True)
        .str.replace("_", " ", regex=False)
        .str.strip()
        .str.title()
    )
    plot_df["label_text"] = plot_df["topic_short_label"] + "\n" + plot_df["outlet_label"]

    substitutions = (
        plot_df.loc[
            (plot_df["baseline_article_share"] <= substitution_baseline_max)
            & (plot_df["article_share"] >= substitution_outlet_min)
        ]
        .sort_values(["article_share", "share_diff"], ascending=[False, False])
        .drop_duplicates(subset=["merged_topic"])
        .head(points_per_zone)
        .assign(map_zone="Substitution zone")
    )

    amplifications = (
        plot_df.loc[
            (plot_df["baseline_article_share"] >= amplification_baseline_min)
            & (plot_df["share_diff"] >= amplification_margin)
        ]
        .sort_values(["share_diff", "article_share"], ascending=[False, False])
        .drop_duplicates(subset=["merged_topic"])
        .head(points_per_zone)
        .assign(map_zone="Amplification zone")
    )

    mainstream_only = (
        plot_df.loc[
            (plot_df["baseline_article_share"] >= mainstream_only_baseline_min)
            & (plot_df["share_diff"] <= -mainstream_only_margin)
        ]
        .sort_values(["baseline_article_share", "share_diff"], ascending=[False, True])
        .drop_duplicates(subset=["merged_topic"])
        .head(points_per_zone)
        .assign(map_zone="Mainstream only")
    )

    near_parity = plot_df.loc[
        (plot_df["absolute_share_diff"] <= parity_margin)
        & (plot_df["baseline_article_share"] >= parity_min_baseline_share)
    ].copy()
    if near_parity.empty:
        near_parity = plot_df.loc[
            plot_df["absolute_share_diff"] <= max(parity_margin, 0.01)
        ].copy()
    near_parity = (
        near_parity.sort_values(["combined_share", "absolute_share_diff"], ascending=[False, True])
        .drop_duplicates(subset=["merged_topic"])
        .head(points_per_zone)
        .assign(map_zone="Near parity")
    )

    selected = pd.concat(
        [substitutions, amplifications, mainstream_only, near_parity],
        ignore_index=True,
    ).drop_duplicates(subset=["outlet_key", "merged_topic"])

    return selected.reset_index(drop=True)


def plot_salient_prevalence_map(
    salient_points_df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame | None = None,
    group_col: str = "outlet_group",
    label_col: str = "label_text",
    color_map: dict[str, str] | None = None,
    neutral_color: str = "#b8b8b8",
    figsize: tuple[int, int] = (12, 8),
    point_size: int = 220,
    substitution_baseline_max: float = 0.035,
    substitution_outlet_min: float = 0.07,
    amplification_baseline_min: float = 0.045,
    amplification_y_min: float = 0.08,
    mainstream_only_baseline_min: float = 0.08,
    mainstream_only_outlet_max: float = 0.045,
):
    """Plot a screenshot-style prevalence map with named conceptual zones."""

    plot_df = salient_points_df.copy()
    if plot_df.empty:
        raise ValueError("No salient prevalence-map points to plot.")

    reference = reference_df if reference_df is not None else plot_df
    axis_limit = float(
        max(
            reference["article_share"].max(),
            reference["baseline_article_share"].max(),
        )
    )
    axis_limit = axis_limit * 1.08 if axis_limit > 0 else 0.01

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    zone_specs = [
        (
            Rectangle(
                (0, substitution_outlet_min),
                substitution_baseline_max,
                max(axis_limit - substitution_outlet_min, 0),
                facecolor="#f3ebe4",
                edgecolor="none",
                alpha=0.65,
                zorder=0,
            ),
            "Substitution zone\nNot in mainstream",
            (substitution_baseline_max * 0.12, axis_limit * 0.92),
        ),
        (
            Rectangle(
                (amplification_baseline_min, amplification_y_min),
                max(axis_limit - amplification_baseline_min, 0),
                max(axis_limit - amplification_y_min, 0),
                facecolor="#f5efe0",
                edgecolor="none",
                alpha=0.55,
                zorder=0,
            ),
            "Amplification zone\nSame topic, much higher weight",
            (amplification_baseline_min + 0.01, axis_limit * 0.92),
        ),
        (
            Rectangle(
                (mainstream_only_baseline_min, 0),
                max(axis_limit - mainstream_only_baseline_min, 0),
                mainstream_only_outlet_max,
                facecolor="#eef2e8",
                edgecolor="none",
                alpha=0.6,
                zorder=0,
            ),
            "Mainstream only",
            (mainstream_only_baseline_min + 0.01, mainstream_only_outlet_max * 0.92),
        ),
    ]

    for rect, zone_label, zone_xy in zone_specs:
        ax.add_patch(rect)
        ax.text(
            zone_xy[0],
            zone_xy[1],
            zone_label,
            ha="left",
            va="top",
            fontsize=11,
            color="#666666",
            zorder=1,
        )

    ax.plot([0, axis_limit], [0, axis_limit], linestyle="--", color="#cfcfcf", linewidth=1.4, zorder=1)
    ax.text(axis_limit, axis_limit, "parity", ha="left", va="bottom", fontsize=10, color="#8f8f8f")

    point_colors = []
    for _, row in plot_df.iterrows():
        if row.get("map_zone") in {"Near parity", "Mainstream only"}:
            point_colors.append(neutral_color)
        elif color_map and group_col in plot_df.columns:
            point_colors.append(color_map.get(str(row[group_col]), neutral_color))
        else:
            point_colors.append(neutral_color)
    plot_df["point_color"] = point_colors

    ax.scatter(
        plot_df["baseline_article_share"],
        plot_df["article_share"],
        s=point_size,
        c=plot_df["point_color"],
        alpha=0.92,
        edgecolors="white",
        linewidths=1.0,
        zorder=2,
    )

    for _, row in plot_df.iterrows():
        label = str(row[label_col]) if label_col in plot_df.columns else str(row["topic_label"])
        ax.text(
            float(row["baseline_article_share"]) + 0.003,
            float(row["article_share"]) + 0.003,
            label,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#333333",
            zorder=3,
        )

    ax.set_xlim(0, axis_limit)
    ax.set_ylim(0, axis_limit)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Tagesschau topic prevalence")
    ax.set_ylabel("Outlet topic prevalence")
    ax.set_title("Topic prevalence map: substitution and amplification vs Tagesschau")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig, ax
