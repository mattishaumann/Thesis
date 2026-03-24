#!/usr/bin/env python3
"""
Stratified sample to validate the pipeline before the full corpus run.
Use `--text-column` / `EMOTION_TEXT_COLUMN` to match `03_emotion_pipeline.py`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from emotion_utils import (
    EMOTION_COLS,
    EMOTION_LABELS,
    assert_model_files,
    effective_min_words,
    effective_text_column,
    get_device,
    get_dominant_emotion,
    get_stratified_sample,
    load_and_clean_texts,
    load_model_and_tokenizer,
    resolve_outlet_column,
    run_emotion_inference,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = REPO_ROOT / "00_Initial EDA" / "df_combined.csv"
DEFAULT_MODEL_DIR = (
    SCRIPT_DIR
    / "models"
    / "final"
    / "german-nlp-group"
    / "electra-base-german-uncased"
)
SAMPLE_OUT = SCRIPT_DIR / "outputs" / "emotion_sample_results.csv"


def _print_mean_probs(by_outlet: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    # Reorder columns to emotion order
    cols = [c for c in EMOTION_COLS if c in by_outlet.columns]
    sub = by_outlet[cols]
    print(sub.to_string(float_format=lambda x: f"{x:.3f}"))


def _top_snippets_per_emotion(
    df: pd.DataFrame, text_col: str, k: int = 5, preview_chars: int = 200
) -> None:
    for i, emo in enumerate(EMOTION_LABELS):
        col = EMOTION_COLS[i]
        if col not in df.columns:
            continue
        top = df.nlargest(k, col)
        print(f"\n=== Top {k} snippets for: {emo} ===")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            snippet = str(row[text_col])[:preview_chars]
            score = float(row[col])
            print(f'{rank}. "{snippet}" ({score:.3f})')


def main() -> None:
    p = argparse.ArgumentParser(description="Stratified-sample emotion inference (sanity check)")
    p.add_argument("--data", type=Path, default=None, help="Input CSV (default: EMOTION_DATA_PATH)")
    p.add_argument("--model-dir", type=Path, default=None, help="Model dir (default: EMOTION_MODEL_DIR)")
    p.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="CSV column to classify (default: Title, or EMOTION_TEXT_COLUMN)",
    )
    p.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Min word count after cleaning (default: 3, or EMOTION_MIN_WORDS)",
    )
    args = p.parse_args()

    data_path = Path(
        args.data if args.data is not None else os.environ.get("EMOTION_DATA_PATH", DEFAULT_DATA)
    )
    model_dir = Path(
        args.model_dir
        if args.model_dir is not None
        else os.environ.get("EMOTION_MODEL_DIR", DEFAULT_MODEL_DIR)
    )
    text_col = effective_text_column(args.text_column)
    min_w = effective_min_words(args.min_words)

    print(f"Loading data from {data_path}...")
    df_full = load_and_clean_texts(data_path, text_column=text_col, min_words=min_w)
    outlet_col = resolve_outlet_column(df_full)
    n_original = len(df_full)
    print(
        f"Cleaned rows: {n_original} (column={text_col!r}, min_words={min_w}); "
        f"{df_full[outlet_col].nunique()} outlets"
    )

    sample = get_stratified_sample(
        df_full, outlet_col, n_total=10000, min_per_outlet=100, seed=42
    )
    print(f"\nStratified sample: {len(sample)} rows (min 5 per outlet where available)\n")
    print("Sample distribution:")
    for name, cnt in sample[outlet_col].value_counts().sort_index().items():
        print(f"  {name}: {cnt} rows")

    assert_model_files(model_dir)
    device = get_device()
    print(f"\nRunning GELECTRA emotion inference on {len(sample)} rows (device: {device})...")
    model, tokenizer, _ = load_model_and_tokenizer(model_dir)
    texts = sample[text_col].tolist()
    probs = run_emotion_inference(
        texts, model, tokenizer, batch_size=32, device=device
    )
    result = pd.concat([sample.reset_index(drop=True), probs], axis=1)
    result["emotion_dominant"] = probs.apply(get_dominant_emotion, axis=1)

    # Summary tables
    mean_by_outlet = result.groupby(outlet_col, observed=True)[EMOTION_COLS].mean()
    mean_overall = pd.DataFrame(
        [result[EMOTION_COLS].mean().tolist()],
        columns=EMOTION_COLS,
        index=["Overall"],
    )
    _print_mean_probs(mean_by_outlet, "Mean emotion probabilities by outlet")
    _print_mean_probs(mean_overall, "Mean emotion probabilities (corpus)")

    print("\n=== Dominant emotion distribution (full sample) ===")
    print(result["emotion_dominant"].value_counts().to_string())

    print("\n=== Dominant emotion by outlet (counts) ===")
    ct = pd.crosstab(result[outlet_col], result["emotion_dominant"])
    print(ct.to_string())

    _top_snippets_per_emotion(result, text_col=text_col, k=5)

    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SAMPLE_OUT, index=False, encoding="utf-8")
    print(f"\nResults saved to {SAMPLE_OUT}")

    # Preview: first 2 rows of probabilities as JSON-like line (optional)
    print("\nPreview (first 2 rows, emotion columns):")
    print(result[EMOTION_COLS + ["emotion_dominant"]].head(2).to_string())


if __name__ == "__main__":
    main()
