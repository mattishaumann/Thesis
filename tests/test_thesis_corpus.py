import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis_corpus import (  # noqa: E402
    build_canonical_combined_df,
    compare_dataframes_as_strings,
    get_canonical_combined_path,
    load_canonical_combined_df,
    load_canonical_outlet_df,
)


class ThesisCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]

    def test_built_combined_matches_canonical_reference(self) -> None:
        rebuilt = build_canonical_combined_df(self.project_root)
        reference = load_canonical_combined_df(self.project_root)

        mismatch_columns = compare_dataframes_as_strings(rebuilt, reference)

        self.assertEqual(mismatch_columns, [])

    def test_load_canonical_outlet_df_uses_expected_source_counts(self) -> None:
        canonical_df = load_canonical_combined_df(self.project_root)
        expected_counts = canonical_df["source"].value_counts().to_dict()

        for source_name, expected_count in expected_counts.items():
            outlet_df = load_canonical_outlet_df(
                source_name,
                self.project_root,
                canonical_df=canonical_df,
            )
            self.assertEqual(len(outlet_df), expected_count)
            self.assertTrue((outlet_df["source"] == source_name).all())

    def test_canonical_reference_path_exists(self) -> None:
        path = get_canonical_combined_path(self.project_root)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "df_combined.csv")


if __name__ == "__main__":
    unittest.main()
