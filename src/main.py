from __future__ import annotations

import argparse
from pathlib import Path

from extract_audio import extract_audio
from transcribe import save_result, transcribe


def run_pipeline(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
    batch_size: int = 16,
) -> Path:
    extract_audio(video_path, audio_path)
    result = transcribe(
        audio_path=audio_path,
        model_name=model_name,
        language=language,
        batch_size=batch_size,
    )
    return save_result(result, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the first lecture-ai MVP pipeline: video -> audio -> transcript JSON."
    )
    parser.add_argument(
        "--input",
        default="data/input/lecture.mp4",
        help="Path to the input lecture video.",
    )
    parser.add_argument(
        "--audio-output",
        default="data/audio/lecture.wav",
        help="Path to the extracted WAV audio.",
    )
    parser.add_argument(
        "--json-output",
        default="data/output/transcript.json",
        help="Path to the transcript JSON file.",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="WhisperX model name.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code such as ko or en.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Transcription batch size.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output = run_pipeline(
        video_path=args.input,
        audio_path=args.audio_output,
        output_path=args.json_output,
        model_name=args.model,
        language=args.language,
        batch_size=args.batch_size,
    )
    print(f"Pipeline completed: {output}")
