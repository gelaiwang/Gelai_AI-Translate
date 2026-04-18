from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gelai-translate")
    parser.add_argument("--config", help="config.yaml path", metavar="PATH")
    sub = parser.add_subparsers(dest="step", required=True)

    step1 = sub.add_parser("step1", help="Download videos")
    step1.add_argument("--source", required=True, help="Video or playlist URL")
    step1.add_argument("--workdir", type=Path, default=None, help="Output directory")
    step1.add_argument("--fmt", default=None, help="yt-dlp format string override")
    step1.add_argument("--download-video", dest="download_video", action="store_true", default=None)
    step1.add_argument("--playlist-items", default=None, help="Playlist item selector for yt-dlp")
    step1.add_argument("--force-redownload-id", default=None, help="Comma-separated IDs to remove from download record before running")
    step1.add_argument("--fallback-clients", default=None, help="Comma-separated fallback clients")
    step1.add_argument("--fallback-formats", default=None, help="Comma-separated fallback formats")
    step1.add_argument("--print-formats-on-fail", action="store_true", default=None, help="Print available formats when all fallbacks fail")

    for name, help_text in [
        ("step2", "Ingest local files and run ASR"),
        ("step3", "Generate translated subtitles"),
        ("step4", "Render bilingual burned-in videos"),
    ]:
        step_parser = sub.add_parser(name, help=help_text)
        step_parser.add_argument("--workdir", type=Path, default=None, help="Working directory")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        os.environ["GELAI_CONFIG"] = str(Path(args.config).expanduser().resolve())

    if args.step == "step1":
        from pipeline.step1_download import run

        return run(
            config_path=args.config,
            workdir=args.workdir,
            source=args.source,
            fmt=args.fmt,
            download_video=args.download_video,
            playlist_items=args.playlist_items,
            force_redownload_id=args.force_redownload_id,
            fallback_clients=args.fallback_clients,
            fallback_formats=args.fallback_formats,
            print_formats_on_fail=args.print_formats_on_fail,
        )

    if args.step == "step2":
        from pipeline.step2_ingest import run

        return run(workdir=args.workdir, config_path=args.config)

    if args.step == "step3":
        from pipeline.step3_translate import run

        return run(workdir=args.workdir, config_path=args.config)

    if args.step == "step4":
        from pipeline.step4_render import run

        return run(workdir=args.workdir, config_path=args.config)

    parser.error(f"Unsupported step: {args.step}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
