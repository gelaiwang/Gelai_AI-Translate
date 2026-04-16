from __future__ import annotations

import unittest

from tests.support import (
    build_fake_config_module,
    build_fake_google_llm_module,
    build_fake_openai_module,
    load_module,
)


translation_runtime = load_module(
    "test_translation_runtime_target",
    "core/translation_runtime.py",
    {
        "openai": build_fake_openai_module(),
        "config": build_fake_config_module(),
        "core.google_llm": build_fake_google_llm_module(),
    },
)


class TranslationRuntimeTests(unittest.TestCase):
    def test_determine_provider_plan_adds_gemini_fallback_for_vertex(self) -> None:
        plan, retries = translation_runtime.determine_provider_plan(
            translation_service="vertex",
            active_model_name="gemini-3-flash-preview",
            service_display_name="Vertex AI",
            gemini_config_ok=True,
            gemini_retry_model_name="gemini-3-flash-preview",
        )
        self.assertEqual(
            plan,
            [
                ("vertex", "gemini-3-flash-preview", "Vertex AI"),
                ("gemini", "gemini-3-flash-preview", "Gemini API"),
            ],
        )
        self.assertEqual(retries, 2)

    def test_determine_provider_plan_keeps_single_provider_for_non_vertex(self) -> None:
        plan, retries = translation_runtime.determine_provider_plan(
            translation_service="gemini",
            active_model_name="gemini-3-flash-preview",
            service_display_name="Gemini API",
            gemini_config_ok=True,
        )
        self.assertEqual(plan, [("gemini", "gemini-3-flash-preview", "Gemini API")])
        self.assertEqual(retries, 3)

    def test_sanitize_model_for_filename_replaces_unsafe_characters(self) -> None:
        self.assertEqual(
            translation_runtime.sanitize_model_for_filename("gemini/3 flash:preview"),
            "gemini_3_flash_preview",
        )

    def test_short_stem_for_filename_truncates_and_normalizes(self) -> None:
        stem = "This is a very long batch stem with spaces and punctuation!!!"
        shortened = translation_runtime.short_stem_for_filename(stem, max_len=20)
        self.assertEqual(shortened, "This_is_a_very_long")
        self.assertLessEqual(len(shortened), 20)


if __name__ == "__main__":
    unittest.main()
