from __future__ import annotations

import argparse
from pathlib import Path

from yt_dlp import YoutubeDL


def resolve_ffmpeg_location() -> str | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    return imageio_ffmpeg.get_ffmpeg_exe()


def download_video(url: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template = str(output_path.with_suffix(".%(ext)s"))
    options = {
        "outtmpl": template,
        "format": "bv*+ba/b",
        "merge_output_format": output_path.suffix.lstrip(".") or "mp4",
        "noplaylist": True,
        "quiet": False,
    }
    ffmpeg_location = resolve_ffmpeg_location()
    if ffmpeg_location:
        options["ffmpeg_location"] = ffmpeg_location

    with YoutubeDL(options) as downloader:
        downloader.extract_info(url, download=True)

    if output_path.exists():
        return output_path

    candidates = sorted(output_path.parent.glob(f"{output_path.stem}.*"))
    if not candidates:
        raise FileNotFoundError(
            f"Video download completed but no output file was found for: {output_path.stem}"
        )

    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a lecture video from a URL to a local file."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Lecture video URL such as a YouTube link.",
    )
    parser.add_argument(
        "--output",
        default="data/input/lecture.mp4",
        help="Path to the downloaded video file.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output = download_video(args.url, args.output)
    print(f"Video downloaded: {output}")
