# src/services/llm_service/model/request.py
"""
Стандартизированные модели запроса к LLM.

LLMRequest — основной контракт для передачи промпта в LLM.
Поддерживает multi-turn диалог через список LLMMessage.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LLMMessage:
    """
    Одно сообщение в диалоге с LLM.

    Атрибуты:
        role (str): Роль отправителя — "system", "user", "assistant"
        content (str): Текст сообщения
    """
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMRequest:
    """
    Структурированный запрос к LLM.

    Атрибуты:
        messages (List[LLMMessage]): История диалога
        temperature (float): Температура генерации (0.0–1.0)
        max_tokens (int): Максимальное количество генерируемых токенов
        top_p (float): Параметр nucleus sampling
    """
    messages: List[LLMMessage]
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 0.9