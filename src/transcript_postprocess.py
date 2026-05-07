from __future__ import annotations

import re


PUNCTUATION_CHARS = ",.;:!?~"
REPEATED_PUNCTUATION_PATTERN = re.compile(rf"([{re.escape(PUNCTUATION_CHARS)}])\1+")
LEADING_NOISE_PATTERN = re.compile(rf"^\s*[{re.escape(PUNCTUATION_CHARS)}]{{3,}}\s*")
TRAILING_NOISE_PATTERN = re.compile(rf"\s*[{re.escape(PUNCTUATION_CHARS)}]{{3,}}\s*$")
PUNCTUATION_ONLY_PATTERN = re.compile(rf"^\s*[{re.escape(PUNCTUATION_CHARS)}]+\s*$")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    if PUNCTUATION_ONLY_PATTERN.fullmatch(text):
        return ""

    text = LEADING_NOISE_PATTERN.sub("", text)
    text = TRAILING_NOISE_PATTERN.sub("", text)
    text = REPEATED_PUNCTUATION_PATTERN.sub(r"\1", text)
    text = re.sub(rf"\s+([{re.escape(PUNCTUATION_CHARS)}])", r"\1", text)

    return text.strip()


def has_meaningful_text(text: str) -> bool:
    compact = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return bool(compact)


def clean_words(words: list[dict] | None) -> list[dict]:
    cleaned_words = []
    for word in words or []:
        cleaned_text = normalize_text(word.get("word", ""))
        start = word.get("start")
        end = word.get("end")

        if start is None or end is None:
            continue
        if float(end) < float(start):
            continue
        if not cleaned_text or not has_meaningful_text(cleaned_text):
            continue

        cleaned_words.append(
            {
                "word": cleaned_text,
                "start": float(start),
                "end": float(end),
                **({"score": word["score"]} if "score" in word else {}),
            }
        )

    return cleaned_words


def clean_segments(segments: list[dict]) -> list[dict]:
    cleaned_segments = []
    for segment in segments:
        cleaned_text = normalize_text(segment.get("text", ""))
        cleaned_words = clean_words(segment.get("words"))

        if cleaned_words:
            words_text = " ".join(word["word"] for word in cleaned_words)
            if not cleaned_text or not has_meaningful_text(cleaned_text):
                cleaned_text = words_text

        cleaned_segments.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": cleaned_text,
                **({"words": cleaned_words} if cleaned_words else {}),
            }
        )

    return cleaned_segments


def filter_noise_segments(segments: list[dict]) -> list[dict]:
    filtered_segments = []
    for segment in segments:
        duration = float(segment["end"]) - float(segment["start"])
        text = segment.get("text", "")
        words = segment.get("words", [])

        if duration <= 0:
            continue
        if not has_meaningful_text(text):
            continue
        if not words and len(re.sub(r"\s+", "", text)) <= 1:
            continue

        filtered_segments.append(segment)

    return filtered_segments


def normalize_transcript_schema(result: dict) -> dict:
    normalized_segments = filter_noise_segments(clean_segments(result.get("segments", [])))
    return {
        "language": result.get("language"),
        "segments": normalized_segments,
    }


def validate_transcript_json(result: dict) -> None:
    if not isinstance(result.get("segments"), list):
        raise ValueError("Transcript result must contain a segments list.")

    for index, segment in enumerate(result["segments"]):
        for key in ("start", "end", "text"):
            if key not in segment:
                raise ValueError(f"Segment {index} is missing required key: {key}")

        if float(segment["end"]) < float(segment["start"]):
            raise ValueError(f"Segment {index} has an end time earlier than start time.")

        if "words" in segment:
            for word_index, word in enumerate(segment["words"]):
                for key in ("word", "start", "end"):
                    if key not in word:
                        raise ValueError(
                            f"Segment {index} word {word_index} is missing required key: {key}"
                        )

