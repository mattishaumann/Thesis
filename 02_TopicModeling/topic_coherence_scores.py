"""Compute C_V topic coherence for the seven per-outlet BERTopic models and the
merged 72-topic cross-outlet model.

Coherence: C_V (Roeder, Both & Hinneburg, 2015), top-10 c-TF-IDF terms per
topic. Scored with gensim's ``CoherenceModel`` against each model's own
modelling-input corpus. Tokenisation matches the BERTopic CountVectorizer
pattern ``\\b\\w\\w+\\b`` and uses the project's curated German stop-word list
from ``02_TopicModeling/01_TopicCreation/stopwords_de.py``.

Reads:
    data/processed/df_combined_with_merged_topics.csv
    02_TopicModeling/outputs/{outlet}_model/topics.json (x7)
    02_TopicModeling/outputs/merged_all_outlets_model/topics.json

Writes:
    data/processed/topic_coherence_scores.csv
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "02_TopicModeling" / "outputs"

sys.path.insert(0, str(ROOT / "02_TopicModeling" / "01_TopicCreation"))
from stopwords_de import get_german_stopwords  # noqa: E402

SOURCE_TO_MODEL: dict[str, str] = {
    "Tagesschau": "ts_model",
    "RT_de": "rt_model",
    "Antispiegel": "as_model",
    "Compact": "compact_model",
    "Deutschlandkurier": "dk_model",
    "Nius": "ns_model",
    "Tichys_Einblick": "te_model",
}

TOP_N = 10
TOKEN_RE = re.compile(r"\b\w\w+\b", flags=re.UNICODE)


def tokenise(text: object, stopwords: set[str]) -> list[str]:
    if not isinstance(text, str):
        return []
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in stopwords]


def topic_words(topics_json_path: Path, top_n: int = TOP_N) -> list[list[str]]:
    """Load BERTopic top-N representations, excluding the -1 outlier topic.

    Bigrams in c-TF-IDF representations (e.g., "social media") are split into
    unigrams so they appear in the unigram-tokenised reference corpus.
    """
    with open(topics_json_path) as f:
        j = json.load(f)
    reps = j["topic_representations"]

    topics: list[list[str]] = []
    for k in sorted(reps.keys(), key=int):
        if int(k) == -1:
            continue
        flat: list[str] = []
        for word, _ in reps[k][:top_n]:
            flat.extend(word.split())
        topics.append(flat[:top_n])
    return topics


def coherence(topics: list[list[str]], texts: list[list[str]]) -> float:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    dictionary = Dictionary(texts)
    cm = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        topn=TOP_N,
        processes=1,
    )
    return float(cm.get_coherence())


def main() -> None:
    warnings.filterwarnings("ignore")

    stopwords = set(get_german_stopwords())

    df = pd.read_csv(
        DATA / "df_combined_with_merged_topics.csv",
        usecols=["source", "included_in_merged_tm_input", "Title", "Text"],
    )
    df = df[df["included_in_merged_tm_input"]].copy()
    df["Text"] = df["Text"].fillna("")
    df["tokens"] = (df["Title"].fillna("") + " " + df["Text"]).map(
        lambda t: tokenise(t, stopwords)
    )

    rows = []
    for source, mdir in SOURCE_TO_MODEL.items():
        sub = df.loc[df["source"] == source]
        texts = sub["tokens"].tolist()
        topics = topic_words(MODEL_DIR / mdir / "topics.json")
        rows.append({
            "outlet": source,
            "model": mdir,
            "n_docs": len(texts),
            "n_topics": len(topics),
            "C_V": coherence(topics, texts),
        })

    texts = df["tokens"].tolist()
    topics = topic_words(MODEL_DIR / "merged_all_outlets_model" / "topics.json")
    rows.append({
        "outlet": "Merged (cross-outlet)",
        "model": "merged_all_outlets_model",
        "n_docs": len(texts),
        "n_topics": len(topics),
        "C_V": coherence(topics, texts),
    })

    result = pd.DataFrame(rows)
    out_path = DATA / "topic_coherence_scores.csv"
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
