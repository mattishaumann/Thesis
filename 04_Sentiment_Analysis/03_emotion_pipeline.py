#!/usr/bin/env python3
"""
Full-corpus GELECTRA discrete emotion inference (Widmann & Wich 2023).

By default each row's `Title` is the document. Use `--text-column` (or `EMOTION_TEXT_COLUMN`)
to run on full article text or any other string column. Long texts are truncated to 512 tokens
at inference; see `emotion_utils` docstring for caveats.

Run after validating with `03a_emotion_sample_test.py`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from emotion_utils import (
    assert_model_files,
    effective_min_words,
    effective_text_column,
    get_dominant_emotion,
    get_device,
    load_and_clean_texts,
    load_model_and_tokenizer,
    resolve_outlet_column,
    run_emotion_inference,
)

# Default paths relative to repository root (parent of Sentiment_Analysis/)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = REPO_ROOT / "01_EDAperOutlet" / "outputs" / "df_combined.csv"
DEFAULT_MODEL_DIR = (
    SCRIPT_DIR
    / "models"
    / "final"
    / "german-nlp-group"
    / "electra-base-german-uncased"
)
DEFAULT_OUT = SCRIPT_DIR / "outputs" / "emotion_full_results.csv"


def run_pipeline(
    data_path: Path | None = None,
    model_dir: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 32,
    text_column: str | None = None,
    min_words: int | None = None,
) -> pd.DataFrame:
    data_path = Path(data_path or os.environ.get("EMOTION_DATA_PATH", DEFAULT_DATA))
    model_dir = Path(model_dir or os.environ.get("EMOTION_MODEL_DIR", DEFAULT_MODEL_DIR))
    output_path = Path(output_path or os.environ.get("EMOTION_OUTPUT_PATH", DEFAULT_OUT))
    col = effective_text_column(text_column)
    mw = effective_min_words(min_words)

    print(f"Loading data from {data_path}...", flush=True)
    df = load_and_clean_texts(data_path, text_column=col, min_words=mw)
    outlet_col = resolve_outlet_column(df)
    print(
        f"Cleaned rows: {len(df)}; text column: {col!r} (min_words={mw}); outlet: {outlet_col!r}",
        flush=True,
    )

    assert_model_files(model_dir)
    device = get_device()
    print(f"Loading model from {model_dir} (device: {device})...", flush=True)
    model, tokenizer, _labels = load_model_and_tokenizer(model_dir)

    texts = df[col].tolist()
    print(f"Running inference on {len(texts)} documents...", flush=True)
    probs = run_emotion_inference(
        texts, model, tokenizer, batch_size=batch_size, device=device
    )

    out = pd.concat([df.reset_index(drop=True), probs], axis=1)
    out["emotion_dominant"] = probs.apply(get_dominant_emotion, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows to {output_path}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Full-corpus emotion classification on titles")
    p.add_argument("--data", type=Path, default=None, help="Path to df_combined.csv")
    p.add_argument("--model-dir", type=Path, default=None, help="Dir with pytorch_model.bin")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=32)
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
        help="Drop rows with fewer than this many words after cleaning (default: 3, or EMOTION_MIN_WORDS)",
    )
    args = p.parse_args()
    run_pipeline(
        data_path=args.data,
        model_dir=args.model_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        text_column=args.text_column,
        min_words=args.min_words,
    )


if __name__ == "__main__":
    main()
