from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_COMBINED_RELATIVE_PATH = Path("00_Initial EDA") / "df_combined.csv"
DEFAULT_TOPICS_EXPORT_RELATIVE_PATH = Path("data") / "processed" / "df_combined_with_merged_topics.csv"
REQUIRED_COMBINED_COLUMNS = ("Date", "Title", "Text", "source", "row_id")


@dataclass(frozen=True)
class CleanCorpusSpec:
    key: str
    source: str
    filename: str
    needs_berlin_timezone_normalization: bool = False


CLEAN_CORPUS_SPECS: tuple[CleanCorpusSpec, ...] = (
    CleanCorpusSpec("antispiegel", "Antispiegel", "antispiegel_clean.csv"),
    CleanCorpusSpec("compact", "Compact", "compact_clean.csv"),
    CleanCorpusSpec("nius", "Nius", "nius_clean.csv"),
    CleanCorpusSpec("rt_de", "RT_de", "rt_de_clean.csv"),
    CleanCorpusSpec("tichys", "Tichys_Einblick", "tichys_clean.csv"),
    CleanCorpusSpec("dkurier", "Deutschlandkurier", "dkurier_clean.csv"),
    CleanCorpusSpec(
        "tagesschau",
        "Tagesschau",
        "tagesschau_clean.csv",
        needs_berlin_timezone_normalization=True,
    ),
)

CLEAN_CORPUS_BY_KEY = {spec.key: spec for spec in CLEAN_CORPUS_SPECS}
CLEAN_CORPUS_BY_SOURCE = {spec.source: spec for spec in CLEAN_CORPUS_SPECS}


def find_project_root(start: Path | None = None) -> Path:
    candidate_start = (start or Path.cwd()).resolve()
    for candidate in (candidate_start, *candidate_start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing .git")


def get_canonical_combined_path(project_root: Path | None = None) -> Path:
    root = find_project_root(project_root)
    path = root / CANONICAL_COMBINED_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Canonical combined corpus not found: {path}")
    return path


def find_clean_csv(project_root: Path, filename: str) -> Path:
    candidates = (
        project_root / "00_Initial EDA" / filename,
        project_root / "data preprocessing" / filename,
        project_root / filename,
        project_root / "data" / "raw" / filename,
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing cleaned file: {filename}")


def _normalize_dates(series: pd.Series, *, berlin_timezone: bool) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if berlin_timezone:
        parsed = parsed.dt.tz_convert("Europe/Berlin")
    return parsed.dt.tz_localize(None).dt.normalize()


def _load_clean_corpus_frame(project_root: Path, spec: CleanCorpusSpec) -> pd.DataFrame:
    csv_path = find_clean_csv(project_root, spec.filename)
    frame = pd.read_csv(csv_path)
    required_columns = ["Date", "Title", "Text", "source"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"{spec.filename} is missing required columns: {missing_columns}")

    frame = frame[required_columns].copy()
    frame["Date"] = _normalize_dates(
        frame["Date"],
        berlin_timezone=spec.needs_berlin_timezone_normalization,
    )
    return frame


def build_canonical_combined_df(project_root: Path | None = None) -> pd.DataFrame:
    root = find_project_root(project_root)
    non_tagesschau_specs = [spec for spec in CLEAN_CORPUS_SPECS if spec.key != "tagesschau"]
    parts = [_load_clean_corpus_frame(root, spec) for spec in non_tagesschau_specs]

    tagesschau_spec = CLEAN_CORPUS_BY_KEY["tagesschau"]
    parts.append(_load_clean_corpus_frame(root, tagesschau_spec))

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.reset_index(drop=True)
    combined["row_id"] = combined.index + 1
    return combined.loc[:, list(REQUIRED_COMBINED_COLUMNS)]


def validate_combined_df_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COMBINED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Combined dataframe is missing required columns: {missing_columns}")
    if df["row_id"].duplicated().any():
        raise ValueError("Combined dataframe contains duplicate row_id values.")


def load_canonical_combined_df(project_root: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(get_canonical_combined_path(project_root))
    validate_combined_df_columns(df)
    return df


def load_canonical_outlet_df(
    source_name: str,
    project_root: Path | None = None,
    *,
    canonical_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = canonical_df if canonical_df is not None else load_canonical_combined_df(project_root)
    validate_combined_df_columns(df)
    subset = df.loc[df["source"] == source_name].copy()
    if subset.empty:
        raise ValueError(f"No rows found for source '{source_name}'.")
    return subset.reset_index(drop=True)


def compare_dataframes_as_strings(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    mismatch_columns: list[str] = []
    if left.columns.tolist() != right.columns.tolist():
        return ["__columns__"]
    if left.shape != right.shape:
        return ["__shape__"]

    for column in left.columns:
        left_values = left[column].astype("string").fillna("<NA>")
        right_values = right[column].astype("string").fillna("<NA>")
        if not left_values.equals(right_values):
            mismatch_columns.append(column)
    return mismatch_columns


def validate_reference_matches_build(project_root: Path | None = None) -> None:
    reference = load_canonical_combined_df(project_root)
    rebuilt = build_canonical_combined_df(project_root)
    mismatch_columns = compare_dataframes_as_strings(reference, rebuilt)
    if mismatch_columns:
        raise AssertionError(
            "Rebuilt combined dataframe does not match the canonical reference. "
            f"Mismatched columns: {mismatch_columns}"
        )


def write_built_combined_df(output_path: Path, project_root: Path | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt = build_canonical_combined_df(project_root)
    rebuilt.to_csv(output_path, index=False)
    return output_path


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or rebuild the canonical thesis combined dataframe.",
    )
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help="Assert that the clean-csv rebuild matches the canonical df_combined.csv reference.",
    )
    parser.add_argument(
        "--write-output",
        type=Path,
        help="Optional output path for a rebuilt combined dataframe.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.validate_reference:
        validate_reference_matches_build()
        print(f"Reference validated: {CANONICAL_COMBINED_RELATIVE_PATH}")

    if args.write_output:
        output_path = write_built_combined_df(args.write_output)
        print(f"Wrote rebuilt combined dataframe to: {output_path}")

    if not args.validate_reference and not args.write_output:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
