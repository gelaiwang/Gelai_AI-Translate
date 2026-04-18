from __future__ import annotations

import unittest

from tests.support import build_fake_srt_module, load_module


translation_text = load_module(
    "test_translation_text_target",
    "core/translation_text.py",
    {"srt": build_fake_srt_module()},
)
srt = build_fake_srt_module()


class TranslationTextTests(unittest.TestCase):
    def make_subtitle(self, index: int, content: str):
        return srt.Subtitle(index=index, start=0, end=1, content=content)

    def test_preprocess_lines_with_numbers_adds_one_based_prefixes(self) -> None:
        self.assertEqual(
            translation_text.preprocess_lines_with_numbers(["alpha", "beta"]),
            ["1: alpha", "2: beta"],
        )

    def test_create_sliding_window_chunks_resets_invalid_overlap(self) -> None:
        chunks = translation_text.create_sliding_window_chunks(
            lines=["a", "b", "c", "d", "e"],
            window_size=3,
            overlap=3,
        )
        self.assertEqual(chunks, [(0, 3, ["a", "b", "c"]), (3, 5, ["d", "e"])])

    def test_extract_lines_from_output_strips_numbering_and_blanks(self) -> None:
        text = "\n1: first line\n2. second line\nthird line\n\n"
        self.assertEqual(
            translation_text.extract_lines_from_output(text),
            ["first line", "second line", "third line"],
        )

    def test_merge_overlapping_translations_prefers_more_central_line(self) -> None:
        merged = translation_text.merge_overlapping_translations(
            chunk_results=[
                (0, 4, ["a1", "a2", "a3", "a4"]),
                (2, 6, ["b3", "b4", "b5", "b6"]),
            ],
            total_lines=6,
        )
        self.assertEqual(
            merged,
            {1: "a1", 2: "a2", 3: "a3", 4: "a4", 5: "b5", 6: "b6"},
        )

    def test_reconstruct_subtitles_from_lines_preserves_layout(self) -> None:
        original_batch = [
            self.make_subtitle(1, "Hello\nworld"),
            self.make_subtitle(2, "Second"),
        ]
        rebuilt = translation_text.reconstruct_subtitles_from_lines(
            original_batch=original_batch,
            translated_text="\ufeff你好\n世界\n第二句\n",
            line_counts=[2, 1],
        )
        self.assertEqual([item.content for item in rebuilt], ["你好\n世界", "第二句"])
        self.assertEqual([item.index for item in rebuilt], [1, 2])

    def test_reconstruct_subtitles_from_lines_rejects_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2, got 1"):
            translation_text.reconstruct_subtitles_from_lines(
                original_batch=[self.make_subtitle(1, "hello"), self.make_subtitle(2, "world")],
                translated_text="only one line",
                line_counts=[1, 1],
            )

    def test_augment_prompt_with_hints_deduplicates_and_keeps_recent_history(self) -> None:
        augmented = translation_text.augment_prompt_with_hints(
            "base prompt",
            [
                "too few lines",
                "too few lines",
                "bad punctuation",
                "format drift",
                "final mismatch",
            ],
        )
        self.assertNotIn("too few lines", augmented)
        self.assertIn("- bad punctuation", augmented)
        self.assertIn("- format drift", augmented)
        self.assertIn("- final mismatch", augmented)

    def test_record_batch_error_hint_deduplicates_latest_and_trims_history(self) -> None:
        storage: dict[int, list[str]] = {}
        translation_text.record_batch_error_hint(storage, 0, "first")
        translation_text.record_batch_error_hint(storage, 0, "first")
        translation_text.record_batch_error_hint(storage, 0, "second")
        translation_text.record_batch_error_hint(storage, 0, "third")
        translation_text.record_batch_error_hint(storage, 0, "fourth")
        translation_text.record_batch_error_hint(storage, 0, "fifth")
        translation_text.record_batch_error_hint(storage, 0, "sixth")
        translation_text.record_batch_error_hint(storage, 0, "seventh")
        self.assertEqual(storage[0], ["fifth", "sixth", "seventh"])


if __name__ == "__main__":
    unittest.main()
