# src/agents/PlannerAgent/utils.py
"""
Вспомогательные функции для PlannerAgent:
- Формирование сообщений для LLM на разных этапах планирования
- Извлечение JSON из ответа модели
- Генерация пустого шаблона плана
"""
from typing import Optional
import logging
import re

LOG = logging.getLogger(__name__)

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