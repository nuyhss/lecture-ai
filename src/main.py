from __future__ import annotations

import argparse
from pathlib import Path

from download_video import download_video
from extract_audio import extract_audio
from transcribe import save_result, transcribe


def run_pipeline(
    video_path: str | Path | None,
    audio_path: str | Path,
    output_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
    batch_size: int | None = None,
    input_url: str | None = None,
    download_path: str | Path = "data/input/lecture.mp4",
    device: str = "auto",
    compute_type: str | None = None,
) -> Path:
    if input_url:
        video_path = download_video(input_url, download_path)

    if video_path is None:
        raise ValueError("Either video_path or input_url must be provided.")

    extract_audio(video_path, audio_path)
    result = transcribe(
        audio_path=audio_path,
        model_name=model_name,
        language=language,
        batch_size=batch_size,
        device=device,
        compute_type=compute_type,
    )
    return save_result(result, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the first lecture-ai MVP pipeline: video -> audio -> transcript JSON."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the local lecture video file.",
    )
    parser.add_argument(
        "--input-url",
        default=None,
        help="Lecture video URL to download before transcription.",
    )
    parser.add_argument(
        "--download-output",
        default="data/input/lecture.mp4",
        help="Path to save the downloaded video when --input-url is used.",
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
        default=None,
        help="Transcription batch size. Defaults to 16 on CUDA and 4 on CPU.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. Use cpu to force CPU-only testing.",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Optional WhisperX compute type override.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    video_input = args.input
    if video_input is None and args.input_url is None:
        video_input = "data/input/lecture.mp4"

    output = run_pipeline(
        video_path=video_input,
        audio_path=args.audio_output,
        output_path=args.json_output,
        model_name=args.model,
        language=args.language,
        batch_size=args.batch_size,
        input_url=args.input_url,
        download_path=args.download_output,
        device=args.device,
        compute_type=args.compute_type,
    )
    print(f"Pipeline completed: {output}")
