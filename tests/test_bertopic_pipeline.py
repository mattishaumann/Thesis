import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1a_BERTopic"))

from bertopic_config import BERTopicConfig  # noqa: E402
from bertopic_pipeline import clean_text, prepare_documents_with_audit  # noqa: E402


class BERTopicPipelinePreparationTest(unittest.TestCase):
    def test_prepare_documents_with_audit_preserves_row_id_and_reasons(self) -> None:
        df = pd.DataFrame(
            [
                {"row_id": 1, "source": "Nius", "Text": "Alpha beta gamma"},
                {"row_id": 2, "source": "Nius", "Text": "Alpha beta gamma"},
                {"row_id": 3, "source": "Nius", "Text": "Hi"},
                {"row_id": 4, "source": "Nius", "Text": "Onlyoneword"},
            ]
        )
        config = BERTopicConfig(min_text_chars=5, min_tokens=2, deduplicate=True)

        prepared, audit = prepare_documents_with_audit(
            df,
            text_col="Text",
            config=config,
            id_col="row_id",
            source_name="Nius",
        )

        self.assertEqual(prepared["row_id"].tolist(), [1])
        self.assertEqual(prepared["document_id"].tolist(), ["1"])

        audit = audit.set_index("row_id")
        self.assertTrue(bool(audit.loc[1, "included_in_model"]))
        self.assertTrue(pd.isna(audit.loc[1, "exclusion_reason"]))
        self.assertEqual(audit.loc[2, "exclusion_reason"], "duplicate_clean_document")
        self.assertEqual(audit.loc[3, "exclusion_reason"], "short_text|too_few_tokens")
        self.assertEqual(audit.loc[4, "exclusion_reason"], "too_few_tokens")

    def test_clean_text_keeps_long_single_line_articles_that_match_boilerplate_prefixes(self) -> None:
        config = BERTopicConfig()
        article = (
            "Mehr als zehn Jahre lang war der Aktivist Alaa Abdel Fattah inhaftiert. "
            "Ikone der Demokratiebewegung: Er wurde international unterstützt und blieb "
            "dennoch im Gefängnis."
        )

        cleaned = clean_text(article, config)

        self.assertEqual(cleaned, article)

    def test_clean_text_still_removes_short_boilerplate_lines(self) -> None:
        config = BERTopicConfig()
        boilerplate = "Mehr zum Thema: Weitere Nachrichten aus Berlin"

        cleaned = clean_text(boilerplate, config)

        self.assertEqual(cleaned, "")


if __name__ == "__main__":
    unittest.main()
