# services/llm_service/utils.py
"""
Универсальные утилиты для работы с промптами и ответами LLM.
"""

import json
import re
from typing import Any, Dict

def safe_format(template: str, mapping: Dict[str, Any]) -> str:
    """
    Безопасно форматирует строковый шаблон, заменяя плейсхолдеры `{key}` на значения из `mapping`.
    Если значение является словарем или списком, оно сериализуется в JSON.
    Args:
        template (str): Шаблон строки.
        mapping (Dict[str, Any]): Словарь с значениями для подстановки.
    Returns:
        str: Отформатированная строка.
    """
    if not template:
        return template

    def _repl(match: re.Match) -> str:
        key = match.group(1)
        if key not in mapping:
            return "{" + key + "}"
        val = mapping[key]
        if isinstance(val, (dict, list, tuple)):
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
        return str(val)

    placeholder_re = re.compile(r"\{([a-zA-Z0-9_]+)\}")
    return placeholder_re.sub(_repl, template)

def strip_code_fences(text: Any) -> str:
    """
    Удаляет markdown-разметку (тройные и одинарные обратные кавычки) из текста.
    Args:
        text (Any): Входной текст. Если не строка, преобразуется в строку.
    Returns:
        str: Очищенный текст.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    # Удаляем тройные кавычки
    if s.startswith("```") and s.endswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    # Удаляем одинарные кавычки
    if s.startswith("`") and s.endswith("`") and len(s) > 2:
        return s.strip("`").strip()
    return s