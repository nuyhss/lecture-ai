from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio

from extract_audio import resolve_ffmpeg_binary
from transcript_postprocess import normalize_transcript_schema, validate_transcript_json


if not hasattr(torchaudio, "AudioMetaData"):
    class AudioMetaData:
        pass

    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends() -> list[str]:
        return ["ffmpeg"]

    torchaudio.list_audio_backends = list_audio_backends

if not hasattr(torchaudio, "get_audio_backend"):
    def get_audio_backend() -> str:
        return "ffmpeg"

    torchaudio.get_audio_backend = get_audio_backend

if not hasattr(torchaudio, "set_audio_backend"):
    def set_audio_backend(_backend: str) -> None:
        return None

    torchaudio.set_audio_backend = set_audio_backend

import whisperx


def load_audio_with_resolved_ffmpeg(file: str, sr: int = 16000) -> np.ndarray:
    cmd = [
        resolve_ffmpeg_binary(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to load audio: {exc.stderr.decode()}") from exc

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


whisperx_audio = importlib.import_module("whisperx.audio")
whisperx_audio.load_audio = load_audio_with_resolved_ffmpeg
whisperx.load_audio = load_audio_with_resolved_ffmpeg


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


def load_model(model_name: str, device: str, compute_type: str):
    return whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        vad_method="silero",
    )


def run_whisper(
    model,
    audio_path: str | Path,
    batch_size: int,
    language: str | None,
) -> dict:
    audio = whisperx.load_audio(str(audio_path))
    transcribe_kwargs = {"batch_size": batch_size}
    if language:
        transcribe_kwargs["language"] = language
    return model.transcribe(audio, **transcribe_kwargs)


def transcribe_to_raw_result(
    audio_path: str | Path,
    model_name: str,
    language: str | None,
    batch_size: int,
    device: str,
    compute_type: str,
) -> dict:
    model = load_model(model_name, device, compute_type)
    return run_whisper(model, audio_path, batch_size, language)


def align_timestamps(
    raw_result: dict,
    audio_path: str | Path,
    device: str,
    language: str | None,
) -> dict:
    detected_language = raw_result.get("language") or language
    if not detected_language:
        return raw_result

    audio = whisperx.load_audio(str(audio_path))
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    aligned_result = whisperx.align(
        raw_result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    aligned_result["language"] = detected_language
    return aligned_result


def process_raw_result(
    raw_result: dict,
    audio_path: str | Path,
    device: str,
    language: str | None,
) -> dict:
    aligned_result = align_timestamps(raw_result, audio_path, device, language)
    result = normalize_transcript_schema(aligned_result)
    validate_transcript_json(result)
    return result


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

    raw_result = transcribe_to_raw_result(
        audio_path=audio_path,
        model_name=model_name,
        language=language,
        batch_size=batch_size,
        device=device,
        compute_type=compute_type,
    )
    return process_raw_result(raw_result, audio_path, device, language)


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
