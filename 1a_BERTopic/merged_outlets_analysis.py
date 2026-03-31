from __future__ import annotations

import ast
import html
import json
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from umap import UMAP

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
PROJECT_ROOT_DIR = MODULE_DIR.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from bertopic_config import BERTopicConfig
from bertopic_pipeline import prepare_documents_with_audit, run_bertopic_pipeline


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CANONICAL_COMBINED_RELATIVE_PATH = Path("00_Initial EDA") / "df_combined.csv"
DEFAULT_TOPICS_EXPORT_RELATIVE_PATH = Path("data") / "processed" / "df_combined_with_merged_topics.csv"
DEFAULT_THESIS_TOPIC_EXPORT_RELATIVE_PATH = Path("data") / "processed" / "df_combined_with_topic.csv"
DEFAULT_MERGED_ARTICLES_CACHE_RELATIVE_PATH = Path("data") / "processed" / "merged_articles_with_umap.csv"
DEFAULT_MERGED_TOPIC_INFO_CACHE_RELATIVE_PATH = Path("data") / "processed" / "merged_topic_info_display.csv"
DEFAULT_MERGED_ANALYSIS_METADATA_RELATIVE_PATH = Path("data") / "processed" / "merged_analysis_metadata.json"
DEFAULT_MERGED_MODEL_RELATIVE_PATH = Path("1a_BERTopic") / "local_outputs" / "merged_all_outlets_model"
DEFAULT_FROZEN_MERGED_RUNS_RELATIVE_DIR = Path("1a_BERTopic") / "local_outputs" / "frozen_merged_runs"
REQUIRED_COMBINED_COLUMNS = ("Date", "Title", "Text", "source", "row_id")

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


def get_canonical_combined_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT_DIR
    return root / CANONICAL_COMBINED_RELATIVE_PATH


def load_canonical_combined_df(project_root: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(get_canonical_combined_path(project_root))
    missing = [column for column in REQUIRED_COMBINED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Combined dataframe is missing required columns: {missing}")
    if df["row_id"].duplicated().any():
        raise ValueError("Combined dataframe contains duplicate row_id values.")
    return df


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


def load_canonical_outlet_dataframe(project_root: Path, source_name: str) -> pd.DataFrame:
    canonical_df = load_canonical_combined_df(project_root)
    subset = canonical_df.loc[canonical_df["source"] == source_name].copy()
    if subset.empty:
        raise ValueError(f"No rows found for source '{source_name}'.")
    return subset.reset_index(drop=True)


OUTLET_SPECS: dict[str, OutletSpec] = {
    "tagesschau": OutletSpec(
        key="tagesschau",
        label="Tagesschau",
        model_name="ts_model",
        text_col="Text",
        id_col="row_id",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=35,
            hdbscan_min_samples=2,
            umap_n_neighbors=25,
            umap_min_dist=0.0,
        ),
        loader=lambda project_root: load_canonical_outlet_dataframe(project_root, "Tagesschau"),
    ),
    "rt": OutletSpec(
        key="rt",
        label="RT",
        model_name="rt_model",
        text_col="Text",
        id_col="row_id",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=15,
            hdbscan_min_samples=3,
            max_df=0.85,
            umap_n_neighbors=40,
            umap_min_dist=0.05,
        ),
        loader=lambda project_root: load_canonical_outlet_dataframe(project_root, "RT_de"),
    ),
    "antispiegel": OutletSpec(
        key="antispiegel",
        label="Antispiegel",
        model_name="as_model",
        text_col="Text",
        id_col="row_id",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=8,
            hdbscan_min_samples=2,
            min_df=1,
            max_df=0.95,
            umap_n_neighbors=10,
            umap_min_dist=0.05,
            nr_topics=None,
        ),
        loader=lambda project_root: load_canonical_outlet_dataframe(project_root, "Antispiegel"),
    ),
    "tichys": OutletSpec(
        key="tichys",
        label="Tichys Einblick",
        model_name="te_model",
        text_col="Text",
        id_col="row_id",
        config_factory=lambda: BERTopicConfig(
            hdbscan_min_cluster_size=10,
            umap_n_neighbors=15,
            extra_stopwords=TICHYS_EXTRA_STOPWORDS,
        ),
        loader=lambda project_root: load_canonical_outlet_dataframe(project_root, "Tichys_Einblick"),
    ),
    "nius": OutletSpec(
        key="nius",
        label="Nius",
        model_name="ns_model",
        text_col="Text",
        id_col="row_id",
        config_factory=lambda: BERTopicConfig(hdbscan_min_cluster_size=15),
        loader=lambda project_root: load_canonical_outlet_dataframe(project_root, "Nius"),
    ),
    "compact": OutletSpec(
        key="compact",
        label="Compact",
        model_name="compact_model",
        text_col="Text",
        id_col="row_id",
        config_factory=BERTopicConfig,
        loader=lambda project_root: load_canonical_outlet_dataframe(project_root, "Compact"),
    ),
    "deutschlandkurier": OutletSpec(
        key="deutschlandkurier",
        label="Deutschlandkurier",
        model_name="dk_model",
        text_col="Text",
        id_col="row_id",
        config_factory=BERTopicConfig,
        loader=lambda project_root: load_canonical_outlet_dataframe(project_root, "Deutschlandkurier"),
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

OUTLET_COLOR_MAP: dict[str, str] = {
    "tagesschau": "#1b9e77",
    "rt": "#d95f02",
    "antispiegel": "#7570b3",
    "tichys": "#e7298a",
    "nius": "#66a61e",
    "compact": "#e6ab02",
    "deutschlandkurier": "#1f78b4",
}


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


def prepare_outlet_documents_with_audit(project_root: Path, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = OUTLET_SPECS[key]
    df = spec.loader(project_root)
    prepared, audit = prepare_documents_with_audit(
        df,
        text_col=spec.text_col,
        config=spec.config_factory(),
        id_col=spec.id_col,
        source_name=spec.label,
    )
    prepared = prepared.copy()
    audit = audit.copy()
    prepared["outlet_key"] = spec.key
    prepared["outlet_label"] = spec.label
    audit["outlet_key"] = spec.key
    audit["outlet_label"] = spec.label
    return prepared, audit


def prepare_outlet_documents(project_root: Path, key: str) -> pd.DataFrame:
    prepared, _ = prepare_outlet_documents_with_audit(project_root, key)
    return prepared


def load_all_prepared_documents(project_root: Path) -> dict[str, pd.DataFrame]:
    return {
        key: prepare_outlet_documents(project_root, key)
        for key in OUTLET_SPECS
    }


def load_all_prepared_documents_with_audits(
    project_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    prepared_by_outlet: dict[str, pd.DataFrame] = {}
    audit_by_outlet: dict[str, pd.DataFrame] = {}
    for key in OUTLET_SPECS:
        prepared, audit = prepare_outlet_documents_with_audit(project_root, key)
        prepared_by_outlet[key] = prepared
        audit_by_outlet[key] = audit
    return prepared_by_outlet, audit_by_outlet


def combine_prepared_documents(prepared_by_outlet: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ordered = [prepared_by_outlet[key] for key in OUTLET_SPECS]
    return pd.concat(ordered, ignore_index=True)


def combine_preparation_audits(audit_by_outlet: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ordered = [audit_by_outlet[key] for key in OUTLET_SPECS]
    return pd.concat(ordered, ignore_index=True)


def train_outlet_topic_model(
    project_root: Path,
    outlet_key: str,
    *,
    embedding_model=None,
    stop_words: list[str] | None = None,
) -> dict[str, object]:
    spec = OUTLET_SPECS[outlet_key]
    df = spec.loader(project_root)
    result = run_bertopic_pipeline(
        df,
        text_col=spec.text_col,
        config=spec.config_factory(),
        id_col=spec.id_col,
        source_name=spec.label,
        stop_words=stop_words,
        embedding_model=embedding_model,
    )
    return result


def save_outlet_topic_model(
    topic_model,
    output_dir: Path,
    outlet_key: str,
    *,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    clear_existing: bool = True,
) -> Path:
    spec = OUTLET_SPECS[outlet_key]
    save_dir = output_dir / spec.model_name
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    if clear_existing and save_dir.exists():
        shutil.rmtree(save_dir)
    topic_model.save(
        save_dir,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_model_name,
    )
    return save_dir


def attach_merged_topics_to_canonical_corpus(
    project_root: Path,
    merged_articles: pd.DataFrame,
    *,
    preparation_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    canonical_df = load_canonical_combined_df(project_root)
    assignment_columns = [
        "row_id",
        "document_id",
        "merged_topic",
        "merged_probability",
        "merged_display_topic",
        "merged_display_label",
        "outlet_key",
        "outlet_label",
    ]
    assignments = merged_articles.loc[:, [col for col in assignment_columns if col in merged_articles.columns]].copy()
    if assignments["row_id"].duplicated().any():
        duplicate_ids = assignments.loc[assignments["row_id"].duplicated(), "row_id"].astype(str).tolist()[:10]
        raise ValueError(f"Merged assignments contain duplicate row_id values: {duplicate_ids}")
    assignments = assignments.rename(
        columns={
            "document_id": "merged_document_id",
            "outlet_key": "merged_outlet_key",
            "outlet_label": "merged_outlet_label",
        }
    )

    if preparation_audit is None:
        _, audit_by_outlet = load_all_prepared_documents_with_audits(project_root)
        preparation_audit = combine_preparation_audits(audit_by_outlet)

    audit_columns = [
        "row_id",
        "document_id",
        "included_in_model",
        "exclusion_reason",
        "document_length",
        "token_count",
        "outlet_key",
        "outlet_label",
    ]
    audit = preparation_audit.loc[:, [col for col in audit_columns if col in preparation_audit.columns]].copy()
    if audit["row_id"].duplicated().any():
        duplicate_ids = audit.loc[audit["row_id"].duplicated(), "row_id"].astype(str).tolist()[:10]
        raise ValueError(f"Preparation audit contains duplicate row_id values: {duplicate_ids}")
    audit = audit.rename(
        columns={
            "document_id": "prepared_document_id",
            "included_in_model": "included_in_merged_tm_input",
            "exclusion_reason": "merged_tm_exclusion_reason",
            "document_length": "merged_tm_document_length",
            "token_count": "merged_tm_token_count",
            "outlet_key": "prepared_outlet_key",
            "outlet_label": "prepared_outlet_label",
        }
    )

    enriched = canonical_df.merge(audit, on="row_id", how="left", validate="1:1")
    enriched = enriched.merge(assignments, on="row_id", how="left", validate="1:1")
    return enriched


def export_canonical_corpus_with_merged_topics(
    project_root: Path,
    merged_articles: pd.DataFrame,
    output_path: Path | None = None,
    *,
    preparation_audit: pd.DataFrame | None = None,
) -> Path:
    export_path = output_path or (project_root / DEFAULT_TOPICS_EXPORT_RELATIVE_PATH)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = attach_merged_topics_to_canonical_corpus(
        project_root,
        merged_articles,
        preparation_audit=preparation_audit,
    )
    enriched.to_csv(export_path, index=False)
    return export_path


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


def rebuild_topic_info_from_assignments(
    base_topic_info: pd.DataFrame,
    assigned_topics: pd.Series | list[int],
) -> pd.DataFrame:
    assigned_series = pd.Series(assigned_topics, dtype="Int64", name="Topic").dropna()
    article_counts = (
        assigned_series.astype(int)
        .value_counts(sort=False)
        .rename_axis("Topic")
        .reset_index(name="Count")
    )

    topic_info = base_topic_info.copy()
    topic_info = topic_info.drop(columns=["Count", "DisplayTopic", "DisplayLabel", "TopicNameClean"], errors="ignore")
    topic_info = topic_info.merge(article_counts, on="Topic", how="inner")
    topic_info["Count"] = topic_info["Count"].astype(int)
    topic_info = topic_info.sort_values(["Topic"], ascending=[True]).reset_index(drop=True)
    return enrich_topic_info_with_display(topic_info)


def save_merged_model(
    merged_model,
    project_root: Path,
    output_dir: Path | None = None,
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    overwrite: bool = True,
) -> Path:
    model_dir = output_dir or (project_root / DEFAULT_MERGED_MODEL_RELATIVE_PATH)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if model_dir.exists() and overwrite:
        shutil.rmtree(model_dir)
    merged_model.save(
        model_dir,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_model,
    )
    return model_dir


def load_saved_merged_model(
    project_root: Path,
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    model_dir: Path | None = None,
):
    from bertopic import BERTopic

    resolved_model_dir = model_dir or (project_root / DEFAULT_MERGED_MODEL_RELATIVE_PATH)
    if not resolved_model_dir.exists():
        raise FileNotFoundError(
            f"Missing saved merged model at {resolved_model_dir}. Run the merged build notebook first."
        )
    merged_model = BERTopic.load(resolved_model_dir)
    return merged_model, resolved_model_dir


def build_merged_analysis_metadata(
    project_root: Path,
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    merged_model_path: Path | None = None,
    articles_path: Path | None = None,
    topic_info_path: Path | None = None,
    metadata_overrides: dict | None = None,
) -> dict:
    resolved_model_path = merged_model_path or (project_root / DEFAULT_MERGED_MODEL_RELATIVE_PATH)
    resolved_articles_path = articles_path or (project_root / DEFAULT_MERGED_ARTICLES_CACHE_RELATIVE_PATH)
    resolved_topic_info_path = topic_info_path or (project_root / DEFAULT_MERGED_TOPIC_INFO_CACHE_RELATIVE_PATH)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "merged_model_path": str(resolved_model_path.relative_to(project_root)),
        "merged_articles_cache_path": str(resolved_articles_path.relative_to(project_root)),
        "merged_topic_info_cache_path": str(resolved_topic_info_path.relative_to(project_root)),
        "article_rows": int(len(merged_articles)),
        "article_outliers_raw": int((merged_articles["merged_topic"] == -1).sum()),
        "article_topics_non_null": int(merged_articles["merged_topic"].notna().sum()),
        "topic_rows": int(len(merged_topic_info)),
        "topic_count_sum": int(merged_topic_info["Count"].sum()),
        "topic_counts_source": "article_assignments",
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    return metadata


def freeze_merged_run_snapshot(
    project_root: Path,
    run_label: str,
    *,
    merged_model_path: Path | None = None,
    articles_path: Path | None = None,
    topic_info_path: Path | None = None,
    thesis_topic_export_path: Path | None = None,
    rich_topic_export_path: Path | None = None,
    metadata_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    resolved_model_path = merged_model_path or (project_root / DEFAULT_MERGED_MODEL_RELATIVE_PATH)
    resolved_articles_path = articles_path or (project_root / DEFAULT_MERGED_ARTICLES_CACHE_RELATIVE_PATH)
    resolved_topic_info_path = topic_info_path or (project_root / DEFAULT_MERGED_TOPIC_INFO_CACHE_RELATIVE_PATH)
    resolved_thesis_topic_export_path = thesis_topic_export_path or (
        project_root / DEFAULT_THESIS_TOPIC_EXPORT_RELATIVE_PATH
    )
    resolved_rich_topic_export_path = rich_topic_export_path or (project_root / DEFAULT_TOPICS_EXPORT_RELATIVE_PATH)
    resolved_metadata_path = metadata_path or (project_root / DEFAULT_MERGED_ANALYSIS_METADATA_RELATIVE_PATH)

    snapshot_dir = project_root / DEFAULT_FROZEN_MERGED_RUNS_RELATIVE_DIR / run_label
    if snapshot_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Frozen merged-run snapshot already exists at {snapshot_dir}. "
                "Choose a new run label or explicitly allow overwrite."
            )
        shutil.rmtree(snapshot_dir)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_model_dir = snapshot_dir / "merged_model"
    shutil.copytree(resolved_model_path, snapshot_model_dir)
    shutil.copy2(resolved_articles_path, snapshot_dir / resolved_articles_path.name)
    shutil.copy2(resolved_topic_info_path, snapshot_dir / resolved_topic_info_path.name)
    shutil.copy2(resolved_thesis_topic_export_path, snapshot_dir / resolved_thesis_topic_export_path.name)
    shutil.copy2(resolved_rich_topic_export_path, snapshot_dir / resolved_rich_topic_export_path.name)
    shutil.copy2(resolved_metadata_path, snapshot_dir / resolved_metadata_path.name)
    return snapshot_dir


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

    frame = combined_prepared.copy()
    frame["merged_topic"] = list(topics)
    frame["merged_probability"] = list(probabilities) if probabilities is not None else None
    merged_topic_info = rebuild_topic_info_from_assignments(merged_model.get_topic_info(), frame["merged_topic"])
    display_label_map = dict(zip(merged_topic_info["Topic"], merged_topic_info["DisplayLabel"]))
    display_topic_map = dict(zip(merged_topic_info["Topic"], merged_topic_info["DisplayTopic"]))
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


def build_df_combined_with_topic(
    project_root: Path,
    merged_articles: pd.DataFrame,
    *,
    preparation_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = attach_merged_topics_to_canonical_corpus(
        project_root,
        merged_articles,
        preparation_audit=preparation_audit,
    )
    ordered_columns = list(REQUIRED_COMBINED_COLUMNS)
    thesis_df = enriched.loc[:, ordered_columns].copy()
    thesis_df["Topic"] = enriched["merged_display_topic"].astype("Int64")
    return thesis_df


def export_df_combined_with_topic(
    project_root: Path,
    merged_articles: pd.DataFrame,
    output_path: Path | None = None,
    *,
    preparation_audit: pd.DataFrame | None = None,
) -> Path:
    export_path = output_path or (project_root / DEFAULT_THESIS_TOPIC_EXPORT_RELATIVE_PATH)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    thesis_df = build_df_combined_with_topic(
        project_root,
        merged_articles,
        preparation_audit=preparation_audit,
    )
    thesis_df.to_csv(export_path, index=False)
    return export_path


def export_merged_analysis_cache(
    project_root: Path,
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    articles_output_path: Path | None = None,
    topic_info_output_path: Path | None = None,
    metadata_output_path: Path | None = None,
    merged_model_path: Path | None = None,
    metadata_overrides: dict | None = None,
) -> tuple[Path, Path, Path]:
    articles_path = articles_output_path or (project_root / DEFAULT_MERGED_ARTICLES_CACHE_RELATIVE_PATH)
    topic_info_path = topic_info_output_path or (project_root / DEFAULT_MERGED_TOPIC_INFO_CACHE_RELATIVE_PATH)
    metadata_path = metadata_output_path or (project_root / DEFAULT_MERGED_ANALYSIS_METADATA_RELATIVE_PATH)
    articles_path.parent.mkdir(parents=True, exist_ok=True)
    topic_info_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    merged_articles.to_csv(articles_path, index=False)
    merged_topic_info.to_csv(topic_info_path, index=False)
    metadata = build_merged_analysis_metadata(
        project_root,
        merged_articles,
        merged_topic_info,
        merged_model_path=merged_model_path,
        articles_path=articles_path,
        topic_info_path=topic_info_path,
        metadata_overrides=metadata_overrides,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return articles_path, topic_info_path, metadata_path


def load_merged_analysis_cache(
    project_root: Path,
    *,
    articles_path: Path | None = None,
    topic_info_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    resolved_articles_path = articles_path or (project_root / DEFAULT_MERGED_ARTICLES_CACHE_RELATIVE_PATH)
    resolved_topic_info_path = topic_info_path or (project_root / DEFAULT_MERGED_TOPIC_INFO_CACHE_RELATIVE_PATH)
    resolved_metadata_path = metadata_path or (project_root / DEFAULT_MERGED_ANALYSIS_METADATA_RELATIVE_PATH)

    if not resolved_articles_path.exists():
        raise FileNotFoundError(
            f"Missing merged articles cache at {resolved_articles_path}. Run the merged build notebook first."
        )
    if not resolved_topic_info_path.exists():
        raise FileNotFoundError(
            f"Missing merged topic-info cache at {resolved_topic_info_path}. Run the merged build notebook first."
        )
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing merged analysis metadata at {resolved_metadata_path}. Run the merged build notebook first."
        )

    merged_articles = pd.read_csv(resolved_articles_path)
    merged_topic_info = pd.read_csv(resolved_topic_info_path)
    metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))

    for column in ("row_id", "merged_topic", "merged_display_topic"):
        if column in merged_articles.columns:
            merged_articles[column] = pd.array(merged_articles[column], dtype="Int64")
    for column in ("Topic", "DisplayTopic"):
        if column in merged_topic_info.columns:
            merged_topic_info[column] = pd.array(merged_topic_info[column], dtype="Int64")

    return merged_articles, merged_topic_info, metadata


def validate_merged_analysis_cache(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    metadata: dict,
) -> list[str]:
    issues: list[str] = []
    article_rows = int(len(merged_articles))
    article_outliers = int((merged_articles["merged_topic"] == -1).sum())
    article_topic_sum = int(merged_articles["merged_topic"].notna().sum())
    topic_count_sum = int(merged_topic_info["Count"].sum())

    if article_rows != int(metadata.get("article_rows", -1)):
        issues.append("Metadata article_rows does not match cached merged articles.")
    if article_outliers != int(metadata.get("article_outliers_raw", -1)):
        issues.append("Metadata outlier count does not match cached merged articles.")
    if article_topic_sum != int(metadata.get("article_topics_non_null", -1)):
        issues.append("Metadata assigned-topic count does not match cached merged articles.")
    if topic_count_sum != int(metadata.get("topic_count_sum", -1)):
        issues.append("Metadata topic_count_sum does not match cached topic info.")
    if article_topic_sum != topic_count_sum:
        issues.append("Cached topic-info counts do not sum to the cached article-level assignments.")
    if metadata.get("topic_counts_source") != "article_assignments":
        issues.append("Merged topic-info metadata is not marked as article-assignment-based.")

    return issues


def build_topic_outlet_frequency_table(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    include_outliers: bool = False,
) -> pd.DataFrame:
    topic_info = merged_topic_info.copy()
    if not include_outliers:
        topic_info = topic_info.loc[topic_info["Topic"] != -1].copy()

    outlet_labels = [spec.label for spec in OUTLET_SPECS.values()]
    counts = (
        merged_articles.groupby(["merged_topic", "outlet_label"]).size().unstack(fill_value=0)
        if not merged_articles.empty
        else pd.DataFrame(columns=outlet_labels)
    )

    if not include_outliers and -1 in counts.index:
        counts = counts.drop(index=-1)

    counts = counts.reindex(index=topic_info["Topic"].tolist(), fill_value=0)
    counts = counts.reindex(columns=outlet_labels, fill_value=0)

    table = topic_info.loc[:, ["Topic", "DisplayTopic", "DisplayLabel"]].copy()
    table = table.rename(columns={"DisplayLabel": "Topic_Label"})
    table["Total_Articles"] = counts.sum(axis=1).to_numpy()
    for outlet_label in outlet_labels:
        table[outlet_label] = counts[outlet_label].to_numpy()

    sort_display = table["DisplayTopic"].fillna(10**9)
    return (
        table.assign(_sort_display=sort_display)
        .sort_values(["_sort_display", "Topic"], ascending=[True, True])
        .drop(columns="_sort_display")
        .reset_index(drop=True)
    )


def build_final_topic_list(
    merged_topic_info: pd.DataFrame,
    *,
    include_outliers: bool = False,
    prefix_start: int = 1,
) -> pd.DataFrame:
    topic_info = merged_topic_info.copy()

    if "TopicNameClean" not in topic_info.columns or "DisplayTopic" not in topic_info.columns:
        topic_info = enrich_topic_info_with_display(topic_info)

    if not include_outliers:
        topic_info = topic_info.loc[topic_info["Topic"] != -1].copy()

    ordered = (
        topic_info.sort_values(["DisplayTopic", "Topic"], ascending=[True, True])
        .reset_index(drop=True)
        .copy()
    )
    ordered["FinalTopic"] = pd.array(range(prefix_start, prefix_start + len(ordered)), dtype="Int64")
    ordered["FinalTopicName"] = ordered.apply(
        lambda row: f"{int(row['FinalTopic'])}_{row['TopicNameClean']}",
        axis=1,
    )
    ordered["FinalTopicLabel"] = ordered.apply(
        lambda row: f"Topic {int(row['FinalTopic'])} — {row['TopicNameClean']}",
        axis=1,
    )

    return (
        ordered.loc[
            :,
            [
                "FinalTopic",
                "Topic",
                "FinalTopicName",
                "FinalTopicLabel",
                "Count",
                "Name",
                "DisplayTopic",
                "DisplayLabel",
                "TopicNameClean",
                "Representation",
            ],
        ]
        .rename(
            columns={
                "Topic": "Topic_Raw",
                "Name": "OriginalName",
            }
        )
        .reset_index(drop=True)
    )


def _truncate_topic_label(label: object, max_chars: int = 42) -> str:
    text = str(label)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_topic_label_for_plot(
    topic_id: int,
    merged_topic_info: pd.DataFrame,
    *,
    include_raw_topic_id: bool = True,
    max_chars: int = 46,
) -> str:
    row = merged_topic_info.loc[merged_topic_info["Topic"] == topic_id]
    if row.empty:
        return f"Topic {topic_id}"
    topic_row = row.iloc[0]
    display_topic = topic_row.get("DisplayTopic")
    topic_name = str(topic_row.get("TopicNameClean", topic_row.get("DisplayLabel", topic_id)))
    base_label = (
        f"Topic {int(display_topic)}"
        if pd.notna(display_topic)
        else f"Topic {topic_id}"
    )
    if include_raw_topic_id and topic_id != -1:
        base_label = f"{base_label} [raw {topic_id}]"
    return _truncate_topic_label(f"{base_label} — {topic_name}", max_chars=max_chars)


def plot_outlet_colored_topic_umap(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    top_n: int = 10,
    figsize: tuple[int, int] = (16, 11),
    point_size: int = 6,
    alpha: float = 0.40,
    include_raw_topic_id_in_labels: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for outlet_key, spec in OUTLET_SPECS.items():
        outlet_df = merged_articles.loc[merged_articles["outlet_key"] == outlet_key]
        if outlet_df.empty:
            continue
        ax.scatter(
            outlet_df["umap_x"],
            outlet_df["umap_y"],
            c=OUTLET_COLOR_MAP.get(outlet_key, "#666666"),
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=spec.label,
        )

    top_topic_ids = (
        merged_topic_info.loc[merged_topic_info["Topic"] != -1]
        .nsmallest(top_n, "DisplayTopic")["Topic"]
        .tolist()
    )

    centroids = (
        merged_articles.loc[merged_articles["merged_topic"].isin(top_topic_ids)]
        .groupby("merged_topic", as_index=False)[["umap_x", "umap_y"]]
        .mean()
    )
    for _, row in centroids.iterrows():
        topic_id = int(row["merged_topic"])
        label = _format_topic_label_for_plot(
            topic_id,
            merged_topic_info,
            include_raw_topic_id=include_raw_topic_id_in_labels,
        )
        ax.text(
            row["umap_x"],
            row["umap_y"],
            label,
            fontsize=9,
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

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=OUTLET_COLOR_MAP.get(outlet_key, "#666666"),
            markeredgecolor="none",
            markersize=7,
            label=spec.label,
        )
        for outlet_key, spec in OUTLET_SPECS.items()
        if outlet_key in merged_articles["outlet_key"].unique()
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True, title="Outlet")

    ax.set_title(
        f"Merged BERTopic UMAP — Articles Colored by Outlet, Top {top_n} Display Topics Labeled"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig, ax


def _normalized_shannon_entropy(proportions: pd.Series) -> float:
    values = proportions.to_numpy(dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return float("nan")
    if len(proportions) <= 1:
        return 0.0
    entropy = -np.sum(values * np.log(values))
    return float(entropy / np.log(len(proportions)))


def _jensen_shannon_divergence(p: pd.Series, q: pd.Series) -> float:
    p_values = p.to_numpy(dtype=float)
    q_values = q.to_numpy(dtype=float)
    m_values = 0.5 * (p_values + q_values)

    def _kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl_divergence(p_values, m_values) + 0.5 * _kl_divergence(q_values, m_values)


def _ranked_topic_ids(counts: pd.Series, top_k: int) -> list[int]:
    ranking = (
        counts.rename("count")
        .rename_axis("topic")
        .reset_index()
        .sort_values(["count", "topic"], ascending=[False, True])
    )
    return ranking.head(top_k)["topic"].astype(int).tolist()


def build_agenda_distortion_scorecard(
    merged_articles: pd.DataFrame,
    merged_topic_info: pd.DataFrame,
    *,
    reference_outlet_key: str = "tagesschau",
    top_k: int = 10,
    min_topic_articles: int = 10,
) -> pd.DataFrame:
    from scipy.stats import binom

    topic_ids = (
        merged_topic_info.loc[merged_topic_info["Topic"] != -1]
        .sort_values(["DisplayTopic", "Topic"], ascending=[True, True])["Topic"]
        .astype(int)
        .tolist()
    )

    assigned = merged_articles.loc[merged_articles["merged_topic"] != -1].copy()
    counts = pd.crosstab(assigned["outlet_key"], assigned["merged_topic"])
    counts = counts.reindex(index=list(OUTLET_SPECS.keys()), columns=topic_ids, fill_value=0)
    proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    reference_distribution = proportions.loc[reference_outlet_key]
    reference_top_topics = set(_ranked_topic_ids(counts.loc[reference_outlet_key], top_k))
    global_topic_distribution = counts.sum(axis=0) / max(int(counts.to_numpy().sum()), 1)
    total_topic_count = len(topic_ids)

    rows: list[dict[str, object]] = []
    for outlet_key, spec in OUTLET_SPECS.items():
        outlet_df = merged_articles.loc[merged_articles["outlet_key"] == outlet_key].copy()
        article_count = int(len(outlet_df))
        outlier_articles = int((outlet_df["merged_topic"] == -1).sum())
        assigned_articles = int(article_count - outlier_articles)
        outlet_distribution = proportions.loc[outlet_key]
        outlet_counts = counts.loc[outlet_key]

        actual_topics_covered = int((outlet_counts >= min_topic_articles).sum())
        expected_topics_covered = float(
            np.sum(
                [
                    binom.sf(min_topic_articles - 1, assigned_articles, float(topic_probability))
                    for topic_probability in global_topic_distribution.to_numpy(dtype=float)
                ]
            )
        )
        actual_coverage_share = actual_topics_covered / total_topic_count if total_topic_count else float("nan")
        expected_coverage_share = expected_topics_covered / total_topic_count if total_topic_count else float("nan")
        coverage_breadth = (
            actual_coverage_share / expected_coverage_share
            if np.isfinite(expected_coverage_share) and expected_coverage_share > 0
            else float("nan")
        )

        rows.append(
            {
                "Outlet": spec.label,
                "Articles": article_count,
                "Assigned_Articles": assigned_articles,
                "Outlier_Articles": outlier_articles,
                "Outlier_Rate": (outlier_articles / article_count) if article_count else float("nan"),
                "Entropy": _normalized_shannon_entropy(outlet_distribution),
                "JSD_vs_Tagesschau": _jensen_shannon_divergence(outlet_distribution, reference_distribution),
                "Spearman_vs_Tagesschau": float(outlet_distribution.corr(reference_distribution, method="spearman")),
                f"Top{top_k}_Overlap_vs_Tagesschau": (
                    len(set(_ranked_topic_ids(outlet_counts, top_k)) & reference_top_topics) / top_k
                ),
                "Topics_Covered_ge_10": actual_topics_covered,
                "Expected_Topics_ge_10": expected_topics_covered,
                "Coverage_Breadth": coverage_breadth,
            }
        )

    return pd.DataFrame(rows)
