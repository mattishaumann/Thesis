from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_SOURCE_ORDER = [
    "Antispiegel",
    "Compact",
    "Deutschlandkurier",
    "Nius",
    "RT_de",
    "Tagesschau",
    "Tichys_Einblick",
]

# Keep the media-target list close to the original notebook so the matching logic
# stays comparable to the current workflow.
DEFAULT_MEDIA_PATTERNS = [
    r"\bard\b",
    r"\bzdf\b",
    r"\b(?:der\s+)?spiegel\b",
    r"\btagesschau(?:\.de)?\b",
    r"\btagesschau24\b",
    r"\breuters\b",
    r"\bjan\s+böhmermann\b",
    r"\bndr(?:\s+info)?\b",
    r"\bnorddeutscher\s+rundfunk\b",
    r"\bbild-zeitung\b|\bdie\s+bild\b",
    r"\bpolitico\b",
    r"\bswr\b",
    r"\bsüdwestrundfunk\b",
    r"\bdpa\b",
    r"\bwdr\b",
    r"\bwestdeutscher\s+rundfunk\b",
    r"\bfaz\b",
    r"\bfrankfurter\s+allgemeine(?:n)?\s+zeitung\b",
    r"\börr\b",
    r"\bhandelsblatt\b",
    r"\bdeutschlandfunk\b",
    r"\btagesspiegel\b",
    r"\btaz\b",
    r"\bbr\b",
    r"\bbayerischer\s+rundfunk\b",
    r"\bberliner\s+zeitung\b",
    r"\brbb\b",
    r"\brundfunk\s+berlin-brandenburg\b",
    r"\bcorrectiv\b",
    r"\bsz\b",
    r"\bsüddeutsch(?:e|en)\s+zeitung\b",
    r"\bstern\b",
    r"\bdas\s+(?-i:Erste)\b",
    r"\bthe\s+european\b",
    r"\brtl\b",
    r"\bmdr\b",
    r"\bmitteldeutscher\s+rundfunk\b",
    r"\brnd\b",
    r"\bredaktionsnetzwerk\s+deutschland\b",
    r"\bcaren\s+miosga\b",
    r"\brheinisch(?:e|en)\s+post\b",
    r"\b(?:markus\s+)?lanz\b",
    r"\beuronews\b",
]

DEFAULT_EXTRA_KEYWORDS = [
    "Mainstreammedien",
    "Staatsmedien",
    "Staatsfunk",
    "Qualitätsmedien",
    "Staatssender",
    "Lügenpresse",
    "Haltungsjournalisten",
    "Gleichschaltung",
    "Mainstreampresse",
    "Altmedien",
    "Systemmedien",
    "Qualitätsjournalismus",
    "Alternativmedien",
    "Gesternmedien",
    "Haltungsmedien",
    "Westmedien",
    "Regierungsmedien",
    "Linkspresse",
    "Qualitätspresse",
    "Haltungsjournalismus",
    "Propagandamedien",
    "Propagandasender",
    "Propagandamaschine",
    "Medienpropaganda",
    "Staatsrundfunk",
]

# Only exclude direct Tagesschau-brand self-references for Tagesschau articles
# themselves. This stays intentionally narrow by default: the user wanted
# Tagesschau included, but only obvious self-mentions removed. If pilot review
# still shows too much ARD-network boilerplate, this set can be widened later
# with evidence from the sampled contexts.
TAGESSCHAU_SELF_MENTION_NORMALIZED = {
    "Tagesschau",
    "Tagesschau24",
    "ARD",
    "Das Erste",
}

NORMALIZATION_MAP = {
    "spiegel": "Spiegel",
    "bild": "Bild",
    "bild-zeitung": "Bild-Zeitung",
    "tagesschau": "Tagesschau",
    "tagesschau.de": "Tagesschau",
    "tagesschau24": "Tagesschau24",
    "ard": "ARD",
    "erste": "Das Erste",
    "das erste": "Das Erste",
    "ndr": "NDR",
    "ndr info": "NDR Info",
    "norddeutscher rundfunk": "NDR",
    "wdr": "WDR",
    "swr": "SWR",
    "br": "BR",
    "bayerischer rundfunk": "BR",
    "mdr": "MDR",
    "mitteldeutscher rundfunk": "MDR",
    "rbb": "rbb",
    "rundfunk berlin-brandenburg": "rbb",
    "zdf": "ZDF",
    "faz": "FAZ",
    "frankfurter allgemeine zeitung": "Frankfurter Allgemeine Zeitung",
    "frankfurter allgemeinen zeitung": "Frankfurter Allgemeine Zeitung",
    "sz": "SZ",
    "süddeutsche zeitung": "Süddeutsche Zeitung",
    "süddeutschen zeitung": "Süddeutsche Zeitung",
    "taz": "taz",
    "rtl": "RTL",
    "dpa": "dpa",
    "örr": "ÖRR",
    "reuters": "Reuters",
    "politico": "Politico",
    "deutschlandfunk": "Deutschlandfunk",
    "tagesspiegel": "Tagesspiegel",
    "stern": "Stern",
    "correctiv": "Correctiv",
    "the european": "The European",
    "euronews": "Euronews",
    "rheinische post": "Rheinische Post",
    "rheinischen post": "Rheinische Post",
    "redaktionsnetzwerk deutschland": "RND",
    "lanz": "Markus Lanz",
    "caren miosga": "Caren Miosga",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÄÖÜ0-9\"'“„(])")
WHITESPACE_RE = re.compile(r"\s+")
SOURCE_ORDER_INDEX = {source: idx for idx, source in enumerate(DEFAULT_SOURCE_ORDER)}


@dataclass(frozen=True)
class BatchCostEstimate:
    n_requests: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_cost_usd: float


@dataclass(frozen=True)
class RuntimeEstimate:
    n_requests: int
    estimated_input_tokens: int
    request_bound_minutes: float
    token_bound_minutes: float
    lower_bound_minutes: float
    estimated_minutes_with_overhead: float


LEGACY_RESULT_COLUMNS = [
    "hit_id",
    "row_id",
    "source",
    "Title",
    "hit_text",
    "context_idx",
    "count_hits",
    "count_unique_entities",
    "context_window",
    "model",
    "response_id",
    "category",
    "evidence",
    "raw_response_json",
]


def normalize_text(text: Any) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def combine_title_text(title: Any, text: Any) -> str:
    title = normalize_text(title)
    text = normalize_text(text)

    if title and text and not re.search(r"[.!?…:;]$", title):
        title = title + "."

    return f"{title} {text}".strip()


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(cleaned) if part and part.strip()]


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def sort_by_source_order(frame: pd.DataFrame, *, source_col: str = "source") -> pd.DataFrame:
    if frame.empty or source_col not in frame.columns:
        return frame

    work = frame.copy()
    work["_source_order"] = work[source_col].map(SOURCE_ORDER_INDEX).fillna(len(SOURCE_ORDER_INDEX)).astype(int)
    work = work.sort_values(["_source_order", source_col]).drop(columns="_source_order")
    return work.reset_index(drop=True)


def compile_master_pattern(
    media_patterns: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> re.Pattern:
    media_patterns = media_patterns or DEFAULT_MEDIA_PATTERNS
    extra_keywords = extra_keywords or DEFAULT_EXTRA_KEYWORDS
    extra_keyword_patterns = [rf"\b{re.escape(keyword)}\b" for keyword in extra_keywords]
    return re.compile(
        "|".join(f"(?:{pattern})" for pattern in media_patterns + extra_keyword_patterns),
        flags=re.IGNORECASE,
    )


def normalize_hit_text(hit: str) -> str:
    hit = normalize_text(hit)
    hit_lower = re.sub(r"^(der|die|das)\s+", "", hit.casefold())
    return NORMALIZATION_MAP.get(hit_lower, hit)


def should_exclude_hit(source: str, normalized_hit: str) -> tuple[bool, str | None]:
    if normalize_text(source).casefold() == "tagesschau" and normalized_hit in TAGESSCHAU_SELF_MENTION_NORMALIZED:
        return True, "tagesschau_self_reference"
    return False, None


def collect_sentence_hits(
    sentence: str,
    source: str,
    row_id: Any,
    title: str,
    pattern: re.Pattern,
    sentence_idx: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for match in pattern.finditer(sentence):
        raw_hit = normalize_text(match.group(0))
        normalized_hit = normalize_hit_text(raw_hit)
        exclude, reason = should_exclude_hit(source, normalized_hit)
        hit_record = {
            "row_id": row_id,
            "source": source,
            "Title": title,
            "sentence_idx": sentence_idx,
            "sentence_text": sentence,
            "raw_hit": raw_hit,
            "normalized_hit": normalized_hit,
        }
        if exclude:
            excluded.append({**hit_record, "exclusion_reason": reason})
        else:
            kept.append(hit_record)

    return kept, excluded


def merge_sentence_windows(hit_sentence_indices: list[int], n_sentences: int, window: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for sentence_idx in hit_sentence_indices:
        start = max(0, sentence_idx - window)
        end = min(n_sentences, sentence_idx + window + 1)
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def extract_media_contexts(
    df: pd.DataFrame,
    *,
    pattern: re.Pattern,
    window: int = 1,
) -> dict[str, pd.DataFrame]:
    has_date = "Date" in df.columns
    available_context_columns = ["row_id", "source", "Title", "Text"]
    if has_date:
        available_context_columns.append("Date")

    context_rows: list[dict[str, Any]] = []
    kept_hit_rows: list[dict[str, Any]] = []
    excluded_hit_rows: list[dict[str, Any]] = []

    for row in df[available_context_columns].itertuples(index=False):
        combined_text = combine_title_text(row.Title, row.Text)
        sentences = split_sentences(combined_text)
        if not sentences:
            continue

        sentence_kept_hits: list[dict[str, Any]] = []
        hit_sentence_indices: list[int] = []

        for sentence_idx, sentence in enumerate(sentences):
            kept, excluded = collect_sentence_hits(
                sentence=sentence,
                source=row.source,
                row_id=row.row_id,
                title=row.Title,
                pattern=pattern,
                sentence_idx=sentence_idx,
            )
            if kept:
                hit_sentence_indices.append(sentence_idx)
                sentence_kept_hits.extend(kept)
            if excluded:
                excluded_hit_rows.extend(excluded)

        if not hit_sentence_indices:
            continue

        kept_hit_rows.extend(sentence_kept_hits)
        merged_windows = merge_sentence_windows(hit_sentence_indices, len(sentences), window=window)

        for context_idx, (start, end) in enumerate(merged_windows, start=1):
            window_hits = [
                hit for hit in sentence_kept_hits
                if start <= int(hit["sentence_idx"]) < end
            ]
            if not window_hits:
                continue

            raw_hits = ordered_unique([hit["raw_hit"] for hit in window_hits])
            normalized_hits = ordered_unique([hit["normalized_hit"] for hit in window_hits])
            context_text = " ".join(sentences[start:end]).strip()

            context_rows.append(
                {
                    "row_id": row.row_id,
                    "source": row.source,
                    "Title": row.Title,
                    "Text": row.Text,
                    "context_idx": context_idx,
                    "context_window": context_text,
                    # We deliberately keep the pipe-joined representation because the
                    # classification unit is the context window, not each individual
                    # regex hit. This preserves "one context = one request" while still
                    # exposing all retained targets inside that context.
                    "hit_text": " | ".join(normalized_hits),
                    "raw_hit_text": " | ".join(raw_hits),
                    "count_hits": len(window_hits),
                    "count_unique_entities": len(normalized_hits),
                }
            )
            if has_date:
                context_rows[-1]["Date"] = getattr(row, "Date", None)

    media_context_df = pd.DataFrame(context_rows)
    kept_hits_df = pd.DataFrame(kept_hit_rows)
    excluded_hits_df = pd.DataFrame(excluded_hit_rows)

    if media_context_df.empty:
        empty_article_columns = ["row_id", "source", "Title", "Text", "context_window", "hit_text", "context_count"]
        if has_date:
            empty_article_columns.insert(2, "Date")
        media_article_df = pd.DataFrame(columns=empty_article_columns)
    else:
        article_group_cols = ["row_id", "source", "Title", "Text"]
        if has_date and "Date" in media_context_df.columns:
            article_group_cols.insert(2, "Date")
        media_article_df = (
            media_context_df.groupby(article_group_cols, as_index=False, dropna=False)
            .agg(
                context_window=("context_window", "\n\n---\n\n".join),
                hit_text=("hit_text", lambda values: " | ".join(ordered_unique(" | ".join(values).split(" | ")))),
                context_count=("context_idx", "nunique"),
            )
        )

    outlet_summary_df = (
        media_article_df.groupby("source")
        .agg(
            candidate_articles=("row_id", "nunique"),
            candidate_context_windows=("context_count", "sum"),
        )
        .reset_index()
        .pipe(sort_by_source_order)
        if not media_article_df.empty
        else pd.DataFrame(columns=["source", "candidate_articles", "candidate_context_windows"])
    )

    exclusion_summary_df = (
        excluded_hits_df.groupby(["source", "normalized_hit", "exclusion_reason"])
        .size()
        .rename("excluded_matches")
        .reset_index()
        .pipe(sort_by_source_order)
        .sort_values(["source", "excluded_matches", "normalized_hit"], ascending=[True, False, True])
        if not excluded_hits_df.empty
        else pd.DataFrame(columns=["source", "normalized_hit", "exclusion_reason", "excluded_matches"])
    )

    return {
        "media_context_df": media_context_df,
        "media_article_df": media_article_df,
        "kept_hits_df": kept_hits_df,
        "excluded_hits_df": excluded_hits_df,
        "outlet_summary_df": outlet_summary_df,
        "exclusion_summary_df": exclusion_summary_df,
    }


def add_batch_ids(media_context_df: pd.DataFrame) -> pd.DataFrame:
    work = media_context_df.copy()
    work["hit_id"] = work.apply(
        lambda row: hashlib.md5(
            f"{row['row_id']}|{row['context_idx']}|{row['context_window']}".encode("utf-8")
        ).hexdigest()[:16],
        axis=1,
    )
    work["custom_id"] = work["hit_id"].map(lambda hit_id: f"media-frame-{hit_id}")
    return work


def build_hit_input(prompt_template: str, row: dict[str, Any]) -> str:
    return prompt_template.format(
        context=row["context_window"],
        entity_mention=row["hit_text"],
    )


def build_batch_request_body(
    row: dict[str, Any],
    *,
    prompt_template: str,
    model_name: str,
    analysis_instructions: str,
    frame_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "instructions": analysis_instructions,
        "input": build_hit_input(prompt_template, row),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "media_bias_frame_v2",
                "strict": True,
                "schema": frame_schema,
            }
        },
    }


def build_batch_requests_df(
    media_context_df: pd.DataFrame,
    *,
    prompt_template: str,
    model_name: str,
    analysis_instructions: str,
    frame_schema: dict[str, Any],
) -> pd.DataFrame:
    work = add_batch_ids(media_context_df)
    request_records: list[dict[str, Any]] = []

    for row in work.to_dict("records"):
        body = build_batch_request_body(
            row,
            prompt_template=prompt_template,
            model_name=model_name,
            analysis_instructions=analysis_instructions,
            frame_schema=frame_schema,
        )
        request_records.append(
            {
                "custom_id": row["custom_id"],
                "hit_id": row["hit_id"],
                "row_id": row["row_id"],
                "source": row["source"],
                "Title": row["Title"],
                "hit_text": row["hit_text"],
                "context_idx": row["context_idx"],
                "count_hits": row["count_hits"],
                "count_unique_entities": row["count_unique_entities"],
                "context_window": row["context_window"],
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )

    return pd.DataFrame(request_records)


def write_jsonl_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_batch_jsonl(batch_requests_df: pd.DataFrame, output_path: Path) -> None:
    jsonl_records = batch_requests_df[["custom_id", "method", "url", "body"]].to_dict("records")
    write_jsonl_records(jsonl_records, output_path)


def validate_batch_requests(batch_requests_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    if batch_requests_df.empty:
        errors.append("batch_requests_df is empty")
        return errors

    if batch_requests_df["custom_id"].duplicated().any():
        errors.append("custom_id values are not unique")

    for row in batch_requests_df.to_dict("records"):
        if row["method"] != "POST":
            errors.append(f"{row['custom_id']}: method must be POST")
        if row["url"] != "/v1/responses":
            errors.append(f"{row['custom_id']}: url must be /v1/responses")
        if not isinstance(row["body"], dict):
            errors.append(f"{row['custom_id']}: body must be a dict")
            continue
        if not row["body"].get("model"):
            errors.append(f"{row['custom_id']}: missing model")
        if not row["body"].get("input"):
            errors.append(f"{row['custom_id']}: missing input")
        schema = (
            row["body"]
            .get("text", {})
            .get("format", {})
            .get("schema", {})
        )
        if not schema:
            errors.append(f"{row['custom_id']}: missing JSON schema")

    return errors


def sample_batch_requests(
    batch_requests_df: pd.DataFrame,
    *,
    n_per_source: int,
    random_state: int = 42,
) -> pd.DataFrame:
    sampled_frames: list[pd.DataFrame] = []
    for source in DEFAULT_SOURCE_ORDER:
        frame = batch_requests_df[batch_requests_df["source"] == source]
        if frame.empty:
            continue
        sampled_frames.append(frame.sample(n=min(n_per_source, len(frame)), random_state=random_state))

    remaining_sources = [
        source for source in batch_requests_df["source"].dropna().astype(str).unique()
        if source not in DEFAULT_SOURCE_ORDER
    ]
    for source in sorted(remaining_sources):
        frame = batch_requests_df[batch_requests_df["source"] == source]
        sampled_frames.append(frame.sample(n=min(n_per_source, len(frame)), random_state=random_state))

    if not sampled_frames:
        return batch_requests_df.iloc[0:0].copy()

    return pd.concat(sampled_frames, ignore_index=True)


def split_batch_requests_by_estimated_tokens(
    batch_requests_df: pd.DataFrame,
    *,
    max_estimated_input_tokens: int,
    max_requests_per_batch: int = 50_000,
) -> list[pd.DataFrame]:
    if batch_requests_df.empty:
        return []
    if max_estimated_input_tokens <= 0:
        raise ValueError("max_estimated_input_tokens must be positive")
    if max_requests_per_batch <= 0:
        raise ValueError("max_requests_per_batch must be positive")

    chunk_indices: list[list[int]] = []
    current_chunk_indices: list[int] = []
    current_chunk_tokens = 0

    for idx, row in batch_requests_df.iterrows():
        row_tokens = estimate_tokens_heuristic(json.dumps(row["body"], ensure_ascii=False))
        should_split = (
            current_chunk_indices
            and (
                current_chunk_tokens + row_tokens > max_estimated_input_tokens
                or len(current_chunk_indices) >= max_requests_per_batch
            )
        )
        if should_split:
            chunk_indices.append(current_chunk_indices)
            current_chunk_indices = []
            current_chunk_tokens = 0

        current_chunk_indices.append(idx)
        current_chunk_tokens += row_tokens

    if current_chunk_indices:
        chunk_indices.append(current_chunk_indices)

    return [
        batch_requests_df.loc[indices].reset_index(drop=True)
        for indices in chunk_indices
    ]


def split_requests_by_rate_budget(
    batch_requests_df: pd.DataFrame,
    *,
    max_requests_per_window: int,
    max_estimated_input_tokens_per_window: int,
) -> list[pd.DataFrame]:
    if batch_requests_df.empty:
        return []
    if max_requests_per_window <= 0:
        raise ValueError("max_requests_per_window must be positive")
    if max_estimated_input_tokens_per_window <= 0:
        raise ValueError("max_estimated_input_tokens_per_window must be positive")

    chunk_indices: list[list[int]] = []
    current_chunk_indices: list[int] = []
    current_chunk_tokens = 0

    for idx, row in batch_requests_df.iterrows():
        row_tokens = estimate_tokens_heuristic(json.dumps(row["body"], ensure_ascii=False))
        should_split = (
            current_chunk_indices
            and (
                len(current_chunk_indices) >= max_requests_per_window
                or current_chunk_tokens + row_tokens > max_estimated_input_tokens_per_window
            )
        )
        if should_split:
            chunk_indices.append(current_chunk_indices)
            current_chunk_indices = []
            current_chunk_tokens = 0

        current_chunk_indices.append(idx)
        current_chunk_tokens += row_tokens

    if current_chunk_indices:
        chunk_indices.append(current_chunk_indices)

    return [
        batch_requests_df.loc[indices].reset_index(drop=True)
        for indices in chunk_indices
    ]


def estimate_runtime_from_rate_limits(
    *,
    n_requests: int,
    estimated_input_tokens: int,
    requests_per_minute_limit: int,
    input_tokens_per_minute_limit: int,
    overhead_factor: float = 1.5,
) -> RuntimeEstimate:
    if n_requests < 0:
        raise ValueError("n_requests must be non-negative")
    if estimated_input_tokens < 0:
        raise ValueError("estimated_input_tokens must be non-negative")
    if requests_per_minute_limit <= 0:
        raise ValueError("requests_per_minute_limit must be positive")
    if input_tokens_per_minute_limit <= 0:
        raise ValueError("input_tokens_per_minute_limit must be positive")
    if overhead_factor <= 0:
        raise ValueError("overhead_factor must be positive")

    request_bound_minutes = n_requests / requests_per_minute_limit
    token_bound_minutes = estimated_input_tokens / input_tokens_per_minute_limit
    lower_bound_minutes = max(request_bound_minutes, token_bound_minutes)

    return RuntimeEstimate(
        n_requests=n_requests,
        estimated_input_tokens=estimated_input_tokens,
        request_bound_minutes=request_bound_minutes,
        token_bound_minutes=token_bound_minutes,
        lower_bound_minutes=lower_bound_minutes,
        estimated_minutes_with_overhead=lower_bound_minutes * overhead_factor,
    )


def read_env_value(name: str, *, project_root: Path) -> tuple[str | None, str | None]:
    env_value = os.getenv(name)
    if env_value:
        return env_value, "environment variable"

    env_candidates = [
        project_root / ".env",
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'"), str(env_path)

    return None, None


def upload_batch_file(api_key: str, jsonl_path: Path) -> dict[str, Any]:
    with jsonl_path.open("rb") as handle:
        response = requests.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"purpose": "batch"},
            files={"file": (jsonl_path.name, handle, "application/jsonl")},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()


def create_batch_job(
    api_key: str,
    *,
    input_file_id: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }
    if metadata:
        payload["metadata"] = metadata

    response = requests.post(
        "https://api.openai.com/v1/batches",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def retrieve_batch_job(api_key: str, batch_id: str) -> dict[str, Any]:
    response = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def download_openai_file(api_key: str, file_id: str, output_path: Path) -> Path:
    response = requests.get(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180,
    )
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")
    return output_path


def extract_output_text_from_responses_body(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_chunks: list[str] = []
    for item in response_body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                text_chunks.append(content["text"])
    return "\n".join(text_chunks).strip()


def build_legacy_result_row(
    manifest_row: dict[str, Any],
    response_body: dict[str, Any],
    *,
    default_model_name: str | None = None,
) -> dict[str, Any]:
    parsed_output = json.loads(extract_output_text_from_responses_body(response_body))
    result_row = {
        **{
            key: manifest_row.get(key, "")
            for key in [
                "hit_id",
                "row_id",
                "source",
                "Title",
                "hit_text",
                "context_idx",
                "count_hits",
                "count_unique_entities",
                "context_window",
            ]
        },
        "model": default_model_name or response_body.get("model", ""),
        "response_id": response_body.get("id", ""),
        "category": parsed_output["category"],
        "evidence": parsed_output["evidence"],
        "raw_response_json": json.dumps(response_body, ensure_ascii=False),
    }
    return {column: result_row.get(column, "") for column in LEGACY_RESULT_COLUMNS}


def parse_batch_output_file(
    output_path: Path,
    manifest_df: pd.DataFrame,
    *,
    default_model_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = manifest_df.set_index("custom_id").to_dict("index")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for raw_line in output_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        custom_id = record.get("custom_id")
        manifest_row = manifest.get(custom_id, {})

        if record.get("error"):
            errors.append(
                {
                    "custom_id": custom_id,
                    **manifest_row,
                    "error_type": "batch_error",
                    "error_payload": json.dumps(record.get("error"), ensure_ascii=False),
                }
            )
            continue

        response = record.get("response") or {}
        response_body = response.get("body") or {}
        if response.get("status_code") != 200:
            errors.append(
                {
                    "custom_id": custom_id,
                    **manifest_row,
                    "error_type": "http_error",
                    "status_code": response.get("status_code"),
                    "error_payload": json.dumps(response_body, ensure_ascii=False),
                }
            )
            continue

        try:
            results.append(
                build_legacy_result_row(
                    manifest_row,
                    response_body,
                    default_model_name=default_model_name,
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(
                {
                    "custom_id": custom_id,
                    **manifest_row,
                    "error_type": "parse_error",
                    "status_code": response.get("status_code"),
                    "error_payload": str(exc),
                    "raw_response_json": json.dumps(response_body, ensure_ascii=False),
                }
            )

    return pd.DataFrame(results, columns=LEGACY_RESULT_COLUMNS), pd.DataFrame(errors)


def estimate_tokens_heuristic(text: str) -> int:
    # Simple conservative estimate for planning. For German prose plus JSON
    # scaffolding, 4 characters per token is a reasonable rough heuristic.
    return int(math.ceil(len(text) / 4))


def estimate_batch_cost(
    batch_requests_df: pd.DataFrame,
    *,
    output_tokens_per_request: int,
    batch_input_price_per_million: float,
    batch_output_price_per_million: float,
) -> BatchCostEstimate:
    estimated_input_tokens = 0
    for row in batch_requests_df.to_dict("records"):
        body = row["body"]
        estimated_input_tokens += estimate_tokens_heuristic(
            json.dumps(body, ensure_ascii=False)
        )

    estimated_output_tokens = len(batch_requests_df) * int(output_tokens_per_request)
    estimated_total_cost_usd = (
        estimated_input_tokens / 1_000_000 * batch_input_price_per_million
        + estimated_output_tokens / 1_000_000 * batch_output_price_per_million
    )

    return BatchCostEstimate(
        n_requests=len(batch_requests_df),
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_total_cost_usd=estimated_total_cost_usd,
    )


def write_manifest_csv(batch_requests_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_cols = [
        "custom_id",
        "hit_id",
        "row_id",
        "source",
        "Title",
        "hit_text",
        "context_idx",
        "count_hits",
        "count_unique_entities",
        "context_window",
    ]
    manifest_cols = [col for col in manifest_cols if col in batch_requests_df.columns]
    batch_requests_df[manifest_cols].to_csv(output_path, index=False, encoding="utf-8")
