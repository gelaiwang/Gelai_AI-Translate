from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.support import (
    build_fake_llm_translate_config_module,
    build_fake_openai_module,
    build_fake_rich_module,
    build_fake_rich_panel_module,
    build_fake_srt_module,
    build_fake_google_llm_module,
    build_fake_translation_batches_module,
    build_fake_translation_context_module,
    build_fake_translation_prompts_module,
    build_fake_translation_runtime_module,
    build_fake_translation_text_module,
    build_fake_translation_validation_module,
    load_module,
)


class LlmTranslateRuntimeTests(unittest.TestCase):
    def load_llm_translate_module(self, config_module, module_suffix: str):
        return load_module(
            f"test_llm_translate_target_{module_suffix}",
            "core/llm_translate.py",
            {
                "config": config_module,
                "openai": build_fake_openai_module(),
                "rich": build_fake_rich_module(),
                "rich.panel": build_fake_rich_panel_module(),
                "srt": build_fake_srt_module(),
                "core.google_llm": build_fake_google_llm_module(),
                "core.translation_context": build_fake_translation_context_module(),
                "core.translation_text": build_fake_translation_text_module(),
                "core.translation_runtime": build_fake_translation_runtime_module(),
                "core.translation_batches": build_fake_translation_batches_module(),
                "core.translation_prompts": build_fake_translation_prompts_module(),
                "core.translation_validation": build_fake_translation_validation_module(),
            },
        )

    def test_import_is_lazy_and_missing_gemini_key_fails_only_on_runtime_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_module = build_fake_llm_translate_config_module(workdir=Path(tmpdir), service="gemini")
            config_module.GEMINI_API_KEY = ""
            module = self.load_llm_translate_module(config_module, "missing_gemini")

            self.assertFalse(module._RUNTIME_INITIALIZED)
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY.*未设置"):
                module._ensure_runtime_initialized()

    def test_runtime_initialization_sets_active_model_for_local_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_module = build_fake_llm_translate_config_module(workdir=Path(tmpdir), service="local")
            module = self.load_llm_translate_module(config_module, "local_ok")

            module._ensure_runtime_initialized()

            self.assertTrue(module._RUNTIME_INITIALIZED)
            self.assertEqual(module.SERVICE_DISPLAY_NAME, "Local LLM")
            self.assertEqual(module.ACTIVE_MODEL_NAME, "qwen3:30b")
            self.assertEqual(module.ACTIVE_API_KEY, "ollama")

    def test_invalid_translation_service_raises_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_module = build_fake_llm_translate_config_module(workdir=Path(tmpdir), service="broken")
            module = self.load_llm_translate_module(config_module, "bad_service")

            with self.assertRaisesRegex(RuntimeError, "TRANSLATION_SERVICE"):
                module._ensure_runtime_initialized()


if __name__ == "__main__":
    unittest.main()
