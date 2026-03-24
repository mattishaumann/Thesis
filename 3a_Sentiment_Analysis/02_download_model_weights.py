#!/usr/bin/env python3
"""
Download the fine-tuned ELECTRA weights (pytorch_model.bin) from the official
3x8emotions GitHub release (~445 MB). config.json is already in this folder.

Usage (from anywhere):
  python Sentiment_Analysis/download_model_weights.py

Or:
  cd Sentiment_Analysis
  python download_model_weights.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Official release asset (tweedmann/3x8emotions, tag electra-model)
WEIGHTS_URL = (
    "https://github.com/tweedmann/3x8emotions/releases/download/electra-model/"
    "pytorch_model.bin"
)
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_DIR = (
    SCRIPT_DIR
    / "models"
    / "final"
    / "german-nlp-group"
    / "electra-base-german-uncased"
)
TARGET_FILE = TARGET_DIR / "pytorch_model.bin"


def _report(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    read = min(block_num * block_size, total_size)
    pct = 100 * read // total_size
    mb_done = read / (1024 * 1024)
    mb_tot = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  Downloading: {pct:3d}%  ({mb_done:.1f} / {mb_tot:.1f} MB)")
    sys.stdout.flush()


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TARGET_DIR / "config.json"
    if not cfg.is_file():
        print(
            "ERROR: config.json is missing from:\n", TARGET_DIR,
            "\nRestore it from the Thesis repo (it ships with Sentiment_Analysis).",
            file=sys.stderr,
        )
        sys.exit(1)

    if TARGET_FILE.is_file():
        size = TARGET_FILE.stat().st_size
        if size > 400_000_000:  # ~400 MB — looks complete
            print(f"Already present: {TARGET_FILE}\n  Size: {size / (1024**2):.1f} MB — skipping download.")
            return
        print(f"Removing incomplete file ({size / (1024**2):.1f} MB) and re-downloading...")
        TARGET_FILE.unlink()

    print("Downloading fine-tuned weights from 3x8emotions (this can take several minutes)...")
    print(f"  URL: {WEIGHTS_URL}")
    print(f"  To:  {TARGET_FILE}\n")

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, TARGET_FILE, reporthook=_report)
    except Exception as e:
        print(f"\n\nDownload failed: {e}", file=sys.stderr)
        if TARGET_FILE.is_file():
            TARGET_FILE.unlink(missing_ok=True)
        sys.exit(1)

    print("\n\nDone.")
    final = TARGET_FILE.stat().st_size
    print(f"  Saved: {TARGET_FILE}\n  Size: {final / (1024**2):.1f} MB")
    if final < 100_000_000:
        print(
            "  WARNING: File seems too small. Try again or download manually from:\n"
            "  https://github.com/tweedmann/3x8emotions/releases/tag/electra-model",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
