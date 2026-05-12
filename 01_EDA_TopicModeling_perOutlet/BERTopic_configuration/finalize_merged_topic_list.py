from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing .git")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a final ordered merged-topic list from the cached merged BERTopic topic-info file. "
            "This relabels the topic-name prefix according to the final merged topic order without "
            "touching the underlying BERTopic model."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to merged_topic_info_display.csv. Defaults to the cached merged topic-info file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to data/processed/merged_topic_final_list.csv",
    )
    parser.add_argument(
        "--prefix-start",
        type=int,
        default=1,
        help="Start number for the relabeled prefix. Use 1 for thesis-style topics, 0 for BERTopic-style zero indexing.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include the outlier topic in the final list.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = find_project_root(Path.cwd())
    module_root = project_root / "02_TopicModeling"
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

    import merged_outlets_analysis as moa

    input_path = args.input or (project_root / moa.DEFAULT_MERGED_TOPIC_INFO_CACHE_RELATIVE_PATH)
    output_path = args.output or (project_root / "data" / "processed" / "merged_topic_final_list.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing merged topic-info cache at {input_path}. Run the merged build notebook first."
        )

    topic_info = pd.read_csv(input_path)
    for column in ("Topic", "DisplayTopic", "Count"):
        if column in topic_info.columns:
            topic_info[column] = pd.array(topic_info[column], dtype="Int64")

    final_topic_list = moa.build_final_topic_list(
        topic_info,
        include_outliers=args.include_outliers,
        prefix_start=args.prefix_start,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_topic_list.to_csv(output_path, index=False)

    print(f"Saved final topic list to: {output_path}")
    print(final_topic_list.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main())
