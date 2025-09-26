# src/agents/synthesizer/validator.py
# coding: utf-8
"""
Вспомогательные функции: парсинг и валидация JSON-ответа от LLM.
"""

import json
from typing import Tuple, Any, Dict

def parse_json_safe(raw: str) -> Tuple[Dict[str, Any], str]:
    """
    Попытаться распарсить JSON из строки raw. Возвращает (obj, error).
    Если парсинг успешен — error == "".
    Если не успешен — obj == {} и error содержит диагностическое сообщение.
    Стремимся извлечь JSON, даже если вокруг есть code fences или текст — очищаем.
    """
    if raw is None:
        return {}, "empty response"

    text = raw.strip()

    # Уберём code fences ```json ... ``` или ``` ... ```
    if text.startswith("```") and text.endswith("```"):
        # убрать тройные backticks
        parts = text.split("```")
        # найти JSON-like часть
        for p in parts:
            p = p.strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break

    # Попробуем парсинг напрямую
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, ""
        else:
            return {}, "parsed JSON is not an object"
    except Exception as e:
        # Попытки найти JSON внутри текста (по первому символу '{' и последнему '}')
        try:
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                candidate = text[first:last+1]
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj, ""
        except Exception:
            pass
        return {}, f"json parse error: {e}"
