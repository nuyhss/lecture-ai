from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcript_postprocess import (  # noqa: E402
    clean_words,
    normalize_text,
    normalize_transcript_schema,
    validate_transcript_json,
)


class TranscriptPostprocessTests(unittest.TestCase):
    def test_normalize_text_removes_repeated_punctuation_noise(self) -> None:
        self.assertEqual(normalize_text("hello,,,,,,,,,"), "hello")
        self.assertEqual(
            normalize_text("and,, no issue,,,,,,,,"),
            "and, no issue",
        )
        self.assertEqual(normalize_text(",,,,,,,,,,"), "")

    def test_clean_words_skips_noise_and_invalid_timestamps(self) -> None:
        cleaned = clean_words(
            [
                {"word": ",,,,,", "start": 0.0, "end": 0.5, "score": 0.1},
                {"word": "token,,,,", "start": 0.5, "end": 0.9, "score": 0.9},
                {"word": "reverse", "start": 2.0, "end": 1.0, "score": 0.3},
                {"word": "missing"},
            ]
        )

        self.assertEqual(
            cleaned,
            [{"word": "token", "start": 0.5, "end": 0.9, "score": 0.9}],
        )

    def test_normalize_transcript_schema_filters_punctuation_only_segments(self) -> None:
        normalized = normalize_transcript_schema(
            {
                "language": "ko",
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": ",,,,,,,,,,,"},
                    {
                        "start": 3.0,
                        "end": 6.0,
                        "text": "and,, no issue,,,,,,,,",
                        "words": [
                            {"word": "and,,", "start": 3.0, "end": 3.5},
                            {"word": "no", "start": 3.6, "end": 4.0},
                            {"word": "issue,,,,,", "start": 4.3, "end": 5.1},
                        ],
                    },
                ],
            }
        )

        self.assertEqual(normalized["language"], "ko")
        self.assertEqual(len(normalized["segments"]), 1)
        self.assertEqual(
            normalized["segments"][0]["text"],
            "and, no issue",
        )

    def test_validate_transcript_json_rejects_invalid_segments(self) -> None:
        with self.assertRaises(ValueError):
            validate_transcript_json(
                {
                    "segments": [
                        {"start": 2.0, "end": 1.0, "text": "reverse"}
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
