# src/services/llm_service/model/response.py
"""
Стандартизированная модель ответа от LLM.

LLMResponse инкапсулирует:
- Сырой текст (`raw_text`)
- Рассуждения (`thinking`) — для моделей типа Qwen-Thinking
- Основной ответ (`answer`)
- JSON-ответ (`json_answer`) — если удалось распарсить
- Метаданные (например, `tokens_used`)

Также содержит статический метод `from_raw`, который парсит любой сырой ответ.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    """
    Структурированный ответ от LLM.

    Атрибуты:
        raw_text (str): Полный сырой ответ модели
        thinking (str): Рассуждения (если модель их генерирует)
        answer (str): Основной ответ (без рассуждений)
        json_answer (Optional[Dict]): Распарсенный JSON (если есть)
        tokens_used (int): Оценка использованных токенов
        metadata (Dict): Дополнительные метаданные (ошибки, backend и т.д.)
    """
    raw_text: str
    thinking: str = ""
    answer: str = ""
    json_answer: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_raw(text: str) -> "LLMResponse":
        """
        Парсит сырой текстовый ответ от LLM и возвращает структурированный LLMResponse.

        Поддерживает:
        - Qwen-специфичные теги: `thinking ... thinking_end`
        - JSON в fenced-блоках (```json ... ```)
        - JSON в фигурных скобках

        Args:
            text (str): Сырой ответ от модели

        Returns:
            LLMResponse: Структурированный объект ответа
        """
        raw_text = text or ""
        thinking = ""
        answer = raw_text
        json_answer = None

        # 1. Обработка Qwen-специфичных тегов рассуждений
        thinking_match = re.search(r"thinking(.*?)thinking_end", raw_text, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            answer = re.sub(r"thinking.*?thinking_end", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()

        # 2. Извлечение JSON из fenced-блоков или фигурных скобок
        json_text = None
        # Fenced block
        fenced = re.findall(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_text, flags=re.MULTILINE)
        if fenced:
            json_text = fenced[0].strip()
        else:
            # Brace matching
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = raw_text[start : end + 1].strip()

        if json_text:
            try:
                json_answer = json.loads(json_text)
                # Если JSON содержит поле "answer", используем его как основной ответ
                if isinstance(json_answer, dict) and "answer" in json_answer:
                    answer = str(json_answer["answer"])
            except json.JSONDecodeError:
                pass  # Игнорируем ошибки парсинга

        return LLMResponse(
            raw_text=raw_text,
            thinking=thinking,
            answer=answer,
            json_answer=json_answer
        )