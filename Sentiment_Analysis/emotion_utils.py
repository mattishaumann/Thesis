"""
Shared utilities for GELECTRA-based discrete emotion classification (Widmann & Wich 2023).
Each title is one sentence and one document — no sentence splitting or aggregation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Authoritative order for 3x8emotions ELECTRA (matches helper/inferencing.py in tweedmann/3x8emotions).
# config.json uses LABEL_0..LABEL_7; index i corresponds to EMOTION_LABELS[i].
EMOTION_LABELS: list[str] = [
    "anger",
    "fear",
    "disgust",
    "sadness",
    "joy",
    "enthusiasm",
    "pride",
    "hope",
]

EMOTION_COLS: list[str] = [f"emotion_{e}" for e in EMOTION_LABELS]


def _read_csv_utf8(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    # Strip UTF-8 BOM / whitespace from headers (e.g. "\ufeffsource" -> "source")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def resolve_outlet_column(df: pd.DataFrame) -> str:
    """Pick outlet/source column from common names (case-sensitive after header normalize)."""
    # NOTE: must be a tuple of separate strings — a single string like ("a, b, c") iterates
    # character-by-character and will never match a column name.
    for name in ("source", "Source", "outlet", "Outlet", "datasource", "Datasource"):
        if name in df.columns:
            return name
    # Case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for key in ("source", "outlet", "datasource"):
        if key in lower_map:
            return lower_map[key]
    raise KeyError(
        "Could not find outlet column. Expected one of: source, outlet, datasource "
        f"(columns present: {list(df.columns)})"
    )


def load_and_clean_titles(filepath: str | Path) -> pd.DataFrame:
    """
    Load CSV and return a copy with cleaned `Title` column.
    Drops empty/invalid titles and rows with fewer than 3 words.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = _read_csv_utf8(path).copy()
    if "Title" not in df.columns:
        raise KeyError(f"Expected column 'Title' in {path}; got {list(df.columns)}")

    # Stringify and strip (NaN -> "nan" then drop)
    df["Title"] = df["Title"].fillna("").astype(str).str.strip()
    mask_bad = (
        (df["Title"] == "")
        | (df["Title"].str.lower() == "nan")
        | (df["Title"].str.lower() == "none")
    )
    df = df.loc[~mask_bad].copy()

    n_words = df["Title"].str.split().str.len()
    df = df.loc[n_words >= 3].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def load_emotion_label_order(model_dir: Path) -> list[str]:
    """
    Prefer label names from config.json if meaningful; else use EMOTION_LABELS.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return list(EMOTION_LABELS)
    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)
    id2label = cfg.get("id2label") or {}
    if not id2label:
        return list(EMOTION_LABELS)
    # Map index -> label string
    ordered: list[str | None] = [None] * len(EMOTION_LABELS)
    for k, v in id2label.items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(ordered):
            ordered[i] = str(v)
    if all(x and not str(x).startswith("LABEL_") for x in ordered):
        return [str(x).lower() for x in ordered]  # type: ignore[list-item]
    return list(EMOTION_LABELS)


def assert_model_files(model_dir: Path) -> None:
    """Raise with download instructions if fine-tuned weights are missing."""
    model_dir = model_dir.resolve()
    bin_file = model_dir / "pytorch_model.bin"
    cfg = model_dir / "config.json"
    if not bin_file.is_file() or not cfg.is_file():
        raise FileNotFoundError(
            "Fine-tuned GELECTRA weights not found.\n"
            f"Expected: {bin_file}\n"
            f"         {cfg}\n\n"
            "Download `pytorch_model.bin` (and ensure `config.json` is present) from the "
            "3x8emotions GitHub release:\n"
            "  https://github.com/tweedmann/3x8emotions/releases\n"
            "Place files under:\n"
            "  Sentiment_Analysis/models/final/german-nlp-group/electra-base-german-uncased/\n"
            "See README.md in this folder for the full layout."
        )


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_stratified_sample(
    df: pd.DataFrame,
    outlet_col: str,
    n_total: int = 100,
    min_per_outlet: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    1) Sample floor(n_total / n_outlets) per outlet (capped by availability).
    2) Ensure each outlet with >= min_per_outlet rows has at least min_per_outlet in the sample.
    3) Top up at random from remaining rows until len >= n_total (or pool exhausted).
    Outlets with fewer than min_per_outlet rows after cleaning contribute all their rows in step 1
    when we enforce minimums (only what exists).
    """
    rng = np.random.default_rng(seed)
    work = df.reset_index(drop=True).copy()
    if outlet_col not in work.columns:
        raise KeyError(f"Column {outlet_col!r} not in dataframe")

    outlets = work[outlet_col].dropna().unique().tolist()
    n_out = len(outlets)
    if n_out == 0:
        return work.iloc[0:0].copy()

    base = n_total // n_out
    picked: set[int] = set()

    # Phase 1: floor(n_total / n_outlets) per outlet
    for o in outlets:
        pos = work.index[work[outlet_col] == o].tolist()
        avail = len(pos)
        if avail == 0:
            continue
        take = min(base, avail)
        if take == 0 and avail > 0:
            take = min(1, avail)  # at least one row per outlet if any data exists
        chosen = rng.choice(pos, size=take, replace=False)
        picked.update(int(x) for x in chosen)

    # Phase 2: raise to min_per_outlet where possible
    for o in outlets:
        pos = work.index[work[outlet_col] == o].tolist()
        avail = len(pos)
        if avail < min_per_outlet:
            continue
        have = sum(1 for p in pos if p in picked)
        need = min_per_outlet - have
        if need <= 0:
            continue
        not_picked = [p for p in pos if p not in picked]
        add = min(need, len(not_picked))
        if add > 0:
            chosen = rng.choice(not_picked, size=add, replace=False)
            picked.update(int(x) for x in chosen)

    # Phase 3: top-up
    all_pos = set(work.index.tolist())
    while len(picked) < n_total:
        remaining = list(all_pos - picked)
        if not remaining:
            break
        need = min(n_total - len(picked), len(remaining))
        extra = rng.choice(remaining, size=need, replace=False)
        picked.update(int(x) for x in extra)

    out = work.loc[sorted(picked)].copy()
    return out.reset_index(drop=True)


def run_emotion_inference(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = 512,
) -> pd.DataFrame:
    """
    Run batched inference; apply softmax over logits so probabilities sum to ~1 per row.
    (Training was multi-label with sigmoid in the official repo; softmax is used here for
    a single dominant-emotion distribution — see README.)
    """
    if device is None:
        device = get_device()
    model.eval()
    model.to(device)

    all_probs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)

    stacked = np.vstack(all_probs)
    return pd.DataFrame(stacked, columns=EMOTION_COLS)


def get_dominant_emotion(prob_row: pd.Series) -> str:
    """Argmax over emotion_* columns."""
    vals = prob_row[[c for c in EMOTION_COLS if c in prob_row.index]]
    if vals.empty:
        return ""
    idx = int(vals.values.argmax())
    return EMOTION_LABELS[idx]


def load_model_and_tokenizer(
    model_dir: Path,
    base_model_name: str = "german-nlp-group/electra-base-german-uncased",
) -> tuple[Any, Any, list[str]]:
    assert_model_files(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    labels = load_emotion_label_order(model_dir)
    if len(labels) != len(EMOTION_LABELS):
        labels = list(EMOTION_LABELS)
    return model, tokenizer, labels
