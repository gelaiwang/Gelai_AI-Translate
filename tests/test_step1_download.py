from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.support import (
    build_fake_config_module,
    build_fake_rich_console_module,
    build_fake_rich_progress_module,
    build_fake_step1_config_module,
    build_fake_youtube_metadata_module,
    build_fake_yt_dlp_module,
    load_module,
)


class Step1DownloadTests(unittest.TestCase):
    def load_step1_module(self, workdir: Path):
        return load_module(
            f"test_step1_download_target_{workdir.name}",
            "pipeline/step1_download.py",
            {
                "config": build_fake_step1_config_module(workdir),
                "core.youtube_metadata": build_fake_youtube_metadata_module(),
                "rich.console": build_fake_rich_console_module(),
                "rich.progress": build_fake_rich_progress_module(),
                "yt_dlp": build_fake_yt_dlp_module(),
            },
        )

    def test_save_downloaded_ids_writes_valid_json_without_temp_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            module = self.load_step1_module(workdir)
            module.save_downloaded_ids({"vid_b", "vid_a"})

            record_path = workdir / "downloaded_ids.json"
            payload = json.loads(record_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["downloaded_ids"], ["vid_a", "vid_b"])
            self.assertIn("last_updated", payload)
            self.assertEqual(list(workdir.glob("downloaded_ids.*.tmp")), [])

    def test_load_downloaded_ids_returns_empty_set_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            module = self.load_step1_module(workdir)
            (workdir / "downloaded_ids.json").write_text("{invalid", encoding="utf-8")

            self.assertEqual(module.load_downloaded_ids(), set())

    def test_add_downloaded_id_persists_updated_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            module = self.load_step1_module(workdir)

            updated = module.add_downloaded_id("new_id", {"old_id"})

            self.assertEqual(updated, {"old_id", "new_id"})
            self.assertEqual(module.load_downloaded_ids(), {"old_id", "new_id"})


if __name__ == "__main__":
    unittest.main()
