from __future__ import annotations

import unittest
from pathlib import Path

from tests.support import build_fake_srt_module, load_module


translation_context = load_module(
    "test_translation_context_target",
    "core/translation_context.py",
    {"srt": build_fake_srt_module()},
)
srt = build_fake_srt_module()


class TranslationContextTests(unittest.TestCase):
    def make_subtitle(self, index: int, content: str):
        return srt.Subtitle(index=index, start=0, end=1, content=content)

    def test_normalize_publish_date_supports_multiple_input_shapes(self) -> None:
        self.assertEqual(translation_context.normalize_publish_date("20260416", None), "2026-04-16")
        self.assertEqual(translation_context.normalize_publish_date("2026-04-16", None), "2026-04-16")
        self.assertEqual(translation_context.normalize_publish_date(None, 1_713_225_600), "2024-04-16")

    def test_detect_series_marker_extracts_episode_information(self) -> None:
        self.assertEqual(
            translation_context.detect_series_marker("Lecture 12 - Market Cycles"),
            (True, "lecture", 12),
        )
        self.assertEqual(
            translation_context.detect_series_marker("Standalone Interview"),
            (False, "", None),
        )

    def test_normalize_series_key_collapses_punctuation(self) -> None:
        self.assertEqual(
            translation_context.normalize_series_key("Berkshire Hathaway's Letters 2026"),
            "berkshire_hathaways_letters_2026",
        )

    def test_prepare_context_payload_truncates_long_srt(self) -> None:
        subtitles = [self.make_subtitle(1, "A" * 60), self.make_subtitle(2, "B" * 60)]
        payload = translation_context.prepare_context_payload(subtitles, max_chars=80)
        self.assertIn("\n...\n", payload)
        self.assertGreater(len(payload), 80)

    def test_validate_context_text_requires_expected_markers(self) -> None:
        self.assertFalse(translation_context.validate_context_text("plain text"))
        self.assertTrue(translation_context.validate_context_text("【翻译风格基准】\nfoo\n【微型术语表】\nbar"))

    def test_read_translation_context_file_rejects_invalid_structure(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / "translation_context.txt"
            context_path.write_text("invalid", encoding="utf-8")
            self.assertEqual(translation_context.read_translation_context_file(context_path), "")

    def test_resolve_translation_context_text_reuses_existing_valid_file(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            context_path = project_dir / "translation_context.txt"
            existing = "【翻译风格基准】\nalpha\n【微型术语表】\nbeta\n"
            context_path.write_text(existing, encoding="utf-8")
            infos: list[str] = []
            warnings: list[str] = []

            result = translation_context.resolve_translation_context_text(
                subtitles=[self.make_subtitle(1, "hello")],
                project_dir=project_dir,
                file_name="translation_context.txt",
                force_regenerate=False,
                enabled=True,
                default_context="fallback",
                source_max_chars=1000,
                template_content="{metadata_block}\n{english_srt_excerpt}",
                escape_braces=lambda text: text,
                generate_translation_context_text=lambda prompt: "should not be used",
                generate_series_name_text=lambda prompt: "系列",
                print_info=infos.append,
                print_warning=warnings.append,
            )

            self.assertEqual(result, existing.strip())
            self.assertTrue(any("可复用" in message for message in infos))
            self.assertEqual(warnings, [])

    def test_resolve_translation_context_text_generates_and_persists_when_enabled(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            infos: list[str] = []
            warnings: list[str] = []
            generated = "【翻译风格基准】\n风格\n【微型术语表】\n术语\n"

            result = translation_context.resolve_translation_context_text(
                subtitles=[self.make_subtitle(1, "hello"), self.make_subtitle(2, "world")],
                project_dir=project_dir,
                file_name="translation_context.txt",
                force_regenerate=False,
                enabled=True,
                default_context="fallback",
                source_max_chars=1000,
                template_content="{metadata_block}\n{english_srt_excerpt}",
                escape_braces=lambda text: text,
                generate_translation_context_text=lambda prompt: generated,
                generate_series_name_text=lambda prompt: "系列",
                print_info=infos.append,
                print_warning=warnings.append,
            )

            self.assertEqual(result, generated)
            self.assertEqual((project_dir / "translation_context.txt").read_text(encoding="utf-8"), generated)
            self.assertTrue(any("已生成" in message for message in infos))
            self.assertEqual(warnings, [])

    def test_resolve_translation_context_text_falls_back_to_default_on_invalid_output(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            warnings: list[str] = []
            result = translation_context.resolve_translation_context_text(
                subtitles=[self.make_subtitle(1, "hello")],
                project_dir=Path(tmpdir),
                file_name="translation_context.txt",
                force_regenerate=False,
                enabled=True,
                default_context="fallback context",
                source_max_chars=1000,
                template_content="{metadata_block}\n{english_srt_excerpt}",
                escape_braces=lambda text: text,
                generate_translation_context_text=lambda prompt: "invalid output",
                generate_series_name_text=lambda prompt: "系列",
                print_info=lambda message: None,
                print_warning=warnings.append,
            )

            self.assertEqual(result, "fallback context")
            self.assertTrue(any("不符合预期结构" in message for message in warnings))


if __name__ == "__main__":
    unittest.main()
