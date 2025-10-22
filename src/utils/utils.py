"""
Утилиты для работы с графом выполнения.
Содержит общие вспомогательные функции, используемые несколькими узлами.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Optional
import logging



LOG = logging.getLogger(__name__)


# src/utils/utils.py

def build_tool_registry_snapshot(agent_registry) -> Dict[str, Any]:
    if agent_registry is None:
        return {}
    snapshot = {}
    EXCLUDED_AGENTS = {"DataAnalysisAgent", "ResultValidatorAgent", "StepResultRelayAgent"}
    for name, entry in agent_registry.tool_registry.items():
        if name in EXCLUDED_AGENTS:
            continue
        if not isinstance(entry, dict):
            continue
        # ВСЕГДА включаем агент, даже если операции пустые
        try:
            operations = agent_registry._resolve_operations(name, entry)
        except Exception as e:
            LOG.warning("Не удалось разрешить операции для агента %s: %s", name, e)
            operations = {}
        safe_meta = {
            "title": entry.get("title", ""),
            "description": entry.get("description", ""),
            "operations": {}
        }
        for op_name, op_meta in operations.items():
            safe_meta["operations"][op_name] = {
                "kind": op_meta.get("kind", "direct"),
                "description": op_meta.get("description", ""),
                "params": op_meta.get("params", {}),
                "outputs": op_meta.get("outputs", {})
            }
        # Даже если operations пустой — добавляем агента!
        snapshot[name] = safe_meta
    return snapshot

def extract_thinking_response(text: str) -> tuple[str, str]:
    """
    Разделяет рассуждения и ответ в ответе Qwen3-4B-Thinking.
    
    Возвращает:
        (reasoning, answer)
    """
    if not text:
        return "", ""
    
    # Стандартные разделители, используемые Qwen
    separators = [
        r"### FINAL ANSWER:",
        r"### Answer:",
        r"####",
        r"```json",
        r"```"
    ]
    
    # Пытаемся найти разделитель
    for sep in separators:
        if sep in text:
            parts = text.split(sep, 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
            return reasoning, answer
    
    # Если разделитель не найден, пытаемся извлечь JSON
    json_text = extract_json_from_text(text)
    if json_text:
        return text.replace(json_text, ""), json_text
    
    # Если все не получается, возвращаем полный текст как рассуждения
    return text, ""

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Извлекает JSON из произвольного текста, возвращаемого LLM.
    Поддерживаются два варианта:
    1. Fenced-блоки (```json ... ```)
    2. Brace-matching (поиск первой { и последней })
    """
    if not text:
        return None
    
    # 1. fenced block
    fenced = re.findall(r"```(?:json)?\s*([\s\S]+?)\s*```", text, flags=re.MULTILINE)
    if fenced:
        return fenced[0].strip()
    
    # 2. brace matching
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    
    return None