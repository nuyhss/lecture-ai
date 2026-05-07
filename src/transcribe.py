from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import whisperx


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_compute_type(device: str, compute_type: str | None) -> str:
    if compute_type:
        return compute_type
    return "float16" if device == "cuda" else "int8"


def resolve_batch_size(device: str, batch_size: int | None) -> int:
    if batch_size is not None:
        return batch_size
    return 16 if device == "cuda" else 4


def transcribe(
    audio_path: str | Path,
    model_name: str = "base",
    language: str | None = None,
    batch_size: int | None = None,
    device: str = "auto",
    compute_type: str | None = None,
) -> dict:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    device = resolve_device(device)
    compute_type = resolve_compute_type(device, compute_type)
    batch_size = resolve_batch_size(device, batch_size)

    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    audio = whisperx.load_audio(str(audio_path))

    transcribe_kwargs = {"batch_size": batch_size}
    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(audio, **transcribe_kwargs)
    detected_language = result.get("language") or language

    if detected_language:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        result["language"] = detected_language

    return result


def save_result(result: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe lecture audio with WhisperX and save JSON output."
    )
    parser.add_argument(
        "--input",
        default="data/audio/lecture.wav",
        help="Path to the WAV audio file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/transcript.json",
        help="Path to the JSON transcript output.",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="WhisperX model name, for example tiny, base, small, medium, large-v3.",
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
        help="Optional WhisperX compute type. Defaults to float16 on CUDA and int8 on CPU.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = transcribe(
        audio_path=args.input,
        model_name=args.model,
        language=args.language,
        batch_size=args.batch_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    output = save_result(result, args.output)

    for segment in result.get("segments", []):
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text", "").strip()
        print(f"{start:.2f} {end:.2f} {text}")

    print(f"Transcript saved: {output}")
