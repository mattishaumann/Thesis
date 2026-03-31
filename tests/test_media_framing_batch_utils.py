import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2a_NER"))

from media_framing_batch_utils import (  # noqa: E402
    add_batch_ids,
    build_legacy_result_row,
    build_batch_requests_df,
    compile_master_pattern,
    estimate_runtime_from_rate_limits,
    extract_media_contexts,
    parse_batch_output_file,
    split_batch_requests_by_estimated_tokens,
    split_requests_by_rate_budget,
)


class MediaFramingBatchUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pattern = compile_master_pattern()

    def test_extract_media_contexts_without_date_keeps_article_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "row_id": 1,
                    "source": "Nius",
                    "Title": "Testtitel",
                    "Text": "Der Spiegel berichtet darüber.",
                }
            ]
        )

        extraction = extract_media_contexts(df, pattern=self.pattern, window=1)

        self.assertEqual(len(extraction["media_context_df"]), 1)
        self.assertEqual(len(extraction["media_article_df"]), 1)
        self.assertNotIn("Date", extraction["media_article_df"].columns)

    def test_tagesschau_self_filter_excludes_lowercase_das_erste(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "row_id": 11,
                    "source": "Tagesschau",
                    "Title": "Medienblick",
                    "Text": "Im Hintergrund meldete das Erste die Nachricht zuerst. Deutschlandfunk widersprach.",
                }
            ]
        )

        extraction = extract_media_contexts(df, pattern=self.pattern, window=1)
        media_context_df = extraction["media_context_df"]
        excluded_hits_df = extraction["excluded_hits_df"]

        self.assertEqual(len(media_context_df), 1)
        self.assertEqual(media_context_df.iloc[0]["hit_text"], "Deutschlandfunk")
        self.assertIn("Das Erste", excluded_hits_df["normalized_hit"].tolist())

    def test_multiple_alias_hits_collapse_to_single_entity_per_context(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "row_id": 21,
                    "source": "Nius",
                    "Title": "TV-Debatte",
                    "Text": (
                        "Markus Lanz kritisierte im ZDF die aktuelle Debatte. "
                        "Lanz sagte spaeter, das ZDF habe falsch berichtet."
                    ),
                }
            ]
        )

        extraction = extract_media_contexts(df, pattern=self.pattern, window=1)
        row = extraction["media_context_df"].iloc[0]

        self.assertEqual(row["hit_text"], "Markus Lanz | ZDF")
        self.assertEqual(row["count_hits"], 4)
        self.assertEqual(row["count_unique_entities"], 2)

    def test_add_batch_ids_matches_legacy_hash_shape(self) -> None:
        media_context_df = pd.DataFrame(
            [
                {
                    "row_id": 7,
                    "source": "RT_de",
                    "Title": "Titel",
                    "hit_text": "Spiegel",
                    "context_idx": 2,
                    "count_hits": 1,
                    "count_unique_entities": 1,
                    "context_window": "Kontext",
                }
            ]
        )

        result = add_batch_ids(media_context_df)
        expected_hit_id = hashlib.md5("7|2|Kontext".encode("utf-8")).hexdigest()[:16]

        self.assertEqual(result.iloc[0]["hit_id"], expected_hit_id)
        self.assertEqual(result.iloc[0]["custom_id"], f"media-frame-{expected_hit_id}")

    def test_build_batch_requests_df_omits_date_column(self) -> None:
        media_context_df = pd.DataFrame(
            [
                {
                    "row_id": 31,
                    "source": "Compact",
                    "Date": "2025-08-01",
                    "Title": "Titel",
                    "hit_text": "Spiegel",
                    "context_idx": 1,
                    "count_hits": 1,
                    "count_unique_entities": 1,
                    "context_window": "Kontext",
                }
            ]
        )

        batch_requests_df = build_batch_requests_df(
            media_context_df,
            prompt_template="Context: {context}\nMention: {entity_mention}",
            model_name="gpt-5-mini",
            analysis_instructions="Return JSON.",
            frame_schema={"type": "object", "properties": {}, "required": []},
        )

        self.assertNotIn("Date", batch_requests_df.columns)
        self.assertEqual(
            batch_requests_df.columns.tolist(),
            [
                "custom_id",
                "hit_id",
                "row_id",
                "source",
                "Title",
                "hit_text",
                "context_idx",
                "count_hits",
                "count_unique_entities",
                "context_window",
                "method",
                "url",
                "body",
            ],
        )

    def test_parse_batch_output_file_returns_legacy_result_shape(self) -> None:
        manifest_df = pd.DataFrame(
            [
                {
                    "custom_id": "media-frame-abc123",
                    "hit_id": "abc123",
                    "row_id": 99,
                    "source": "Nius",
                    "Title": "Titel",
                    "hit_text": "Spiegel",
                    "context_idx": 1,
                    "count_hits": 1,
                    "count_unique_entities": 1,
                    "context_window": "Kontext",
                }
            ]
        )

        output_record = {
            "custom_id": "media-frame-abc123",
            "response": {
                "status_code": 200,
                "body": {
                    "id": "resp_123",
                    "output_text": json.dumps({"category": "NEUTRAL", "evidence": ""}),
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            output_path.write_text(json.dumps(output_record) + "\n", encoding="utf-8")
            results_df, errors_df = parse_batch_output_file(
                output_path,
                manifest_df,
                default_model_name="gpt-5-mini",
            )

        self.assertTrue(errors_df.empty)
        self.assertEqual(
            results_df.columns.tolist(),
            [
                "hit_id",
                "row_id",
                "source",
                "Title",
                "hit_text",
                "context_idx",
                "count_hits",
                "count_unique_entities",
                "context_window",
                "model",
                "response_id",
                "category",
                "evidence",
                "raw_response_json",
            ],
        )
        self.assertEqual(results_df.iloc[0]["model"], "gpt-5-mini")
        self.assertEqual(results_df.iloc[0]["response_id"], "resp_123")
        self.assertEqual(results_df.iloc[0]["category"], "NEUTRAL")

    def test_build_legacy_result_row_matches_expected_columns(self) -> None:
        manifest_row = {
            "hit_id": "abc123",
            "row_id": 99,
            "source": "Nius",
            "Title": "Titel",
            "hit_text": "Spiegel",
            "context_idx": 1,
            "count_hits": 1,
            "count_unique_entities": 1,
            "context_window": "Kontext",
        }
        response_body = {
            "id": "resp_123",
            "model": "gpt-5-mini",
            "output_text": json.dumps({"category": "NEUTRAL", "evidence": ""}),
        }

        result_row = build_legacy_result_row(manifest_row, response_body)

        self.assertEqual(
            list(result_row.keys()),
            [
                "hit_id",
                "row_id",
                "source",
                "Title",
                "hit_text",
                "context_idx",
                "count_hits",
                "count_unique_entities",
                "context_window",
                "model",
                "response_id",
                "category",
                "evidence",
                "raw_response_json",
            ],
        )
        self.assertEqual(result_row["model"], "gpt-5-mini")
        self.assertEqual(result_row["response_id"], "resp_123")

    def test_split_batch_requests_by_estimated_tokens_preserves_order(self) -> None:
        batch_requests_df = pd.DataFrame(
            [
                {
                    "custom_id": "media-frame-a",
                    "body": {"input": "x" * 80},
                },
                {
                    "custom_id": "media-frame-b",
                    "body": {"input": "y" * 80},
                },
                {
                    "custom_id": "media-frame-c",
                    "body": {"input": "z" * 10},
                },
            ]
        )

        chunks = split_batch_requests_by_estimated_tokens(
            batch_requests_df,
            max_estimated_input_tokens=30,
            max_requests_per_batch=1,
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["custom_id"].tolist(), ["media-frame-a"])
        self.assertEqual(chunks[1]["custom_id"].tolist(), ["media-frame-b"])
        self.assertEqual(chunks[2]["custom_id"].tolist(), ["media-frame-c"])

    def test_split_requests_by_rate_budget_respects_request_and_token_limits(self) -> None:
        batch_requests_df = pd.DataFrame(
            [
                {"custom_id": "media-frame-a", "body": {"input": "x" * 80}},
                {"custom_id": "media-frame-b", "body": {"input": "y" * 80}},
                {"custom_id": "media-frame-c", "body": {"input": "z" * 10}},
            ]
        )

        chunks = split_requests_by_rate_budget(
            batch_requests_df,
            max_requests_per_window=2,
            max_estimated_input_tokens_per_window=30,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["custom_id"].tolist(), ["media-frame-a"])
        self.assertEqual(chunks[1]["custom_id"].tolist(), ["media-frame-b", "media-frame-c"])

    def test_estimate_runtime_from_rate_limits_uses_slower_bound(self) -> None:
        estimate = estimate_runtime_from_rate_limits(
            n_requests=1_000,
            estimated_input_tokens=5_000_000,
            requests_per_minute_limit=500,
            input_tokens_per_minute_limit=500_000,
            overhead_factor=1.5,
        )

        self.assertAlmostEqual(estimate.request_bound_minutes, 2.0)
        self.assertAlmostEqual(estimate.token_bound_minutes, 10.0)
        self.assertAlmostEqual(estimate.lower_bound_minutes, 10.0)
        self.assertAlmostEqual(estimate.estimated_minutes_with_overhead, 15.0)


if __name__ == "__main__":
    unittest.main()
