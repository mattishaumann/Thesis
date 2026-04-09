"""Shared path resolution for framing notebooks (run from repo root or 03_Framing/)."""

from __future__ import annotations

from pathlib import Path


def resolve_framing_paths() -> tuple[Path, Path, Path]:
    """
    Return (thesis_root, framing_03_dir, outputs_dir).

    Finds ``03_Framing/outputs/framing_gpt_results.csv`` from common cwd choices.
    """
    cwd = Path.cwd().resolve()

    if cwd.name == "03_Framing":
        out = cwd / "outputs"
        if (out / "framing_gpt_results.csv").is_file():
            return cwd.parent, cwd, out

    if cwd.name == "Media_Framing_Analysis":
        framing = cwd.parent
        out = framing / "outputs"
        if framing.name == "03_Framing" and (out / "framing_gpt_results.csv").is_file():
            return framing.parent, framing, out

    for base in (cwd, cwd.parent):
        framing = base / "03_Framing"
        out = framing / "outputs"
        if (out / "framing_gpt_results.csv").is_file():
            return base, framing, out

    raise FileNotFoundError(
        "Could not find 03_Framing/outputs/framing_gpt_results.csv. "
        "Start Jupyter from the Thesis repo root or from 03_Framing/."
    )


def topic_table_candidates(thesis_root: Path) -> list[Path]:
    """Possible locations for the topic-merged corpus CSV (repo layout varies)."""
    return [
        thesis_root / "02_TopicModeling" / "outputs" / "df_combined_topic_only_new.csv",
        thesis_root / "1a_BERTopic" / "outputs" / "df_combined_topic_only_new.csv",
    ]
