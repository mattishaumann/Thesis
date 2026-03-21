#!/usr/bin/env python3
"""
Stratified sample (>=500 titles, >=5 per outlet where possible) to validate the pipeline.
Run this before `emotion_pipeline.py` on the full corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from emotion_utils import (
    EMOTION_COLS,
    EMOTION_LABELS,
    assert_model_files,
    get_device,
    get_dominant_emotion,
    get_stratified_sample,
    load_and_clean_titles,
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


def _top_titles_per_emotion(
    df: pd.DataFrame, text_col: str = "Title", k: int = 5
) -> None:
    for i, emo in enumerate(EMOTION_LABELS):
        col = EMOTION_COLS[i]
        if col not in df.columns:
            continue
        top = df.nlargest(k, col)
        print(f"\n=== Top {k} titles for: {emo} ===")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            title = str(row[text_col])[:200]
            score = float(row[col])
            print(f'{rank}. "{title}" ({score:.3f})')


def main() -> None:
    data_path = Path(os.environ.get("EMOTION_DATA_PATH", DEFAULT_DATA))
    model_dir = Path(os.environ.get("EMOTION_MODEL_DIR", DEFAULT_MODEL_DIR))

    print(f"Loading data from {data_path}...")
    df_full = load_and_clean_titles(data_path)
    outlet_col = resolve_outlet_column(df_full)
    n_original = len(df_full)
    print(f"Cleaned titles: {n_original} rows across {df_full[outlet_col].nunique()} outlets")

    sample = get_stratified_sample(
        df_full, outlet_col, n_total=10000, min_per_outlet=100, seed=42
    )
    print(f"\nStratified sample: {len(sample)} titles (min 5 per outlet where available)\n")
    print("Sample distribution:")
    for name, cnt in sample[outlet_col].value_counts().sort_index().items():
        print(f"  {name}: {cnt} titles")

    assert_model_files(model_dir)
    device = get_device()
    print(f"\nRunning GELECTRA emotion inference on {len(sample)} titles (device: {device})...")
    model, tokenizer, _ = load_model_and_tokenizer(model_dir)
    texts = sample["Title"].tolist()
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

    _top_titles_per_emotion(result, text_col="Title", k=5)

    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SAMPLE_OUT, index=False, encoding="utf-8")
    print(f"\nResults saved to {SAMPLE_OUT}")

    # Preview: first 2 rows of probabilities as JSON-like line (optional)
    print("\nPreview (first 2 rows, emotion columns):")
    print(result[EMOTION_COLS + ["emotion_dominant"]].head(2).to_string())


if __name__ == "__main__":
    main()
