from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich import print as rprint
from rich.panel import Panel


def load_prompt_template(file_path: Path, required_placeholders: Optional[list[str]] = None) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            template_content = f.read()
        for placeholder in required_placeholders or []:
            if placeholder not in template_content:
                rprint(Panel(f"[bold red]错误: 提示模板文件 '{file_path}'\n缺少必要的占位符 '{placeholder}'。[/bold red]", title="提示模板错误", border_style="red"))
                raise SystemExit(1)
        return template_content
    except FileNotFoundError:
        rprint(Panel(f"[bold red]错误: 提示模板文件 '{file_path}' 未找到。[/bold red]", title="文件未找到", border_style="red"))
        raise SystemExit(1)
    except Exception as e:
        rprint(Panel(f"[bold red]错误: 读取提示模板文件 '{file_path}' 时发生错误: {e}[/bold red]", title="文件读取错误", border_style="red"))
        raise SystemExit(1)


def load_local_stage_prompts(base_dir: Path) -> tuple[str, str]:
    stage1_path = base_dir / "translate_local_stage1.txt"
    stage2_path = base_dir / "translate_local_stage2.txt"

    stage1_content = ""
    stage2_content = ""

    if stage1_path.exists():
        stage1_content = stage1_path.read_text(encoding="utf-8")
    if stage2_path.exists():
        stage2_content = stage2_path.read_text(encoding="utf-8")

    return stage1_content, stage2_content
