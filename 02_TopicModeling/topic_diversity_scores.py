"""Compute per-outlet topic-space diversity metrics for the merged BERTopic model.

Two metrics:
  TD score   — Bianchi et al. 2021: |unique words| / (10 × n_active_topics)
  Emb. div.  — Mean pairwise cosine distance between topic embeddings

Both computed over active topics (≥10 articles from that outlet).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "02_TopicModeling" / "outputs" / "merged_all_outlets_model"

OUTLET_COLS = [
    "Tagesschau", "RT", "Antispiegel",
    "Tichys Einblick", "Nius", "Compact", "Deutschlandkurier",
]
ACTIVE_THRESHOLD = 10


def load_topic_words(path: Path) -> dict[int, list[str]]:
    df = pd.read_csv(path)
    df = df[df["Topic"] != -1]
    result = {}
    for _, row in df.iterrows():
        words = ast.literal_eval(row["Representation"])
        result[int(row["Topic"])] = words
    return result


def load_topic_embeddings(path: Path) -> np.ndarray:
    with safe_open(str(path), framework="numpy") as f:
        return f.get_tensor("topic_embeddings")  # shape (73, 384), row 0 = outlier


def topic_diversity(words_per_topic: list[list[str]]) -> float:
    all_tokens = [w for words in words_per_topic for w in words]
    if not all_tokens:
        return float("nan")
    return len(set(all_tokens)) / len(all_tokens)


def embedding_diversity(embs: np.ndarray) -> float:
    if len(embs) < 2:
        return float("nan")
    dists = pdist(embs, metric="cosine")
    return float(dists.mean())


def main() -> None:
    counts = pd.read_csv(DATA / "merged_topic_outlet_counts.csv")
    topic_words = load_topic_words(DATA / "merged_topic_info_top30_words.csv")
    embeddings = load_topic_embeddings(MODEL_DIR / "topic_embeddings.safetensors")

    rows = []
    for outlet in OUTLET_COLS:
        active = counts[counts[outlet] >= ACTIVE_THRESHOLD]["Topic"].tolist()
        active = [t for t in active if t != -1]

        words_list = [topic_words[t] for t in active if t in topic_words]
        emb_matrix = np.array([embeddings[t + 1] for t in active if t in topic_words])

        rows.append({
            "Outlet": outlet,
            "N_Active_Topics_ge10": len(active),
            "TD_score": topic_diversity(words_list),
            "Embedding_Diversity": embedding_diversity(emb_matrix),
        })

    result = pd.DataFrame(rows)
    out_path = DATA / "topic_diversity_scores.csv"
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved to {out_path}")

    # Merge with existing agenda scores for a combined view
    agenda = pd.read_csv(DATA / "merged_agenda_distortion_scores.csv")
    combined = agenda.merge(result, on="Outlet", how="left")
    combined_path = DATA / "merged_agenda_distortion_scores_full.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Combined scores saved to {combined_path}")


if __name__ == "__main__":
    main()
