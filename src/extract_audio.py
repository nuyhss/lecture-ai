from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def resolve_ffmpeg_binary() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise FileNotFoundError(
            "ffmpeg executable was not found in PATH, and imageio-ffmpeg is not installed. "
            "Install FFmpeg system-wide or add imageio-ffmpeg to the Python environment."
        ) from exc

    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: str | Path, output_path: str | Path) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    ffmpeg_bin = resolve_ffmpeg_binary()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract mono 16kHz WAV audio from a lecture video."
    )
    parser.add_argument(
        "--input",
        default="data/input/lecture.mp4",
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--output",
        default="data/audio/lecture.wav",
        help="Path to the generated WAV file.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output = extract_audio(args.input, args.output)
    print(f"Audio extracted: {output}")
