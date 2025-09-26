# services/llm_service/adapters/openai_adapter.py
"""
Адаптер для работы с API OpenAI.
Преобразует вызовы в формат, ожидаемый библиотекой OpenAI.
"""

import logging
import os
from typing import Any, Dict, Optional

from .base import BaseLLMAdapter

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

class OpenAIAdapter(BaseLLMAdapter):
    """Адаптер для моделей OpenAI."""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует адаптер.
        Args:
            config (Dict[str, Any]): Конфигурация, содержащая `model_name`, `temperature`, `api_key`.
        """
        try:
            import openai
        except ImportError:
            raise RuntimeError("Библиотека 'openai' не установлена. Установите ее с помощью 'pip install openai'.")

        self.api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Не указан OPENAI_API_KEY.")

        openai.api_key = self.api_key
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 512))

    def generate(self, prompt: str) -> str:
        """Генерирует ответ с помощью OpenAI API."""
        import openai
        try:
            resp = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            LOG.exception("Ошибка при вызове OpenAI API: %s", e)
            raise

def create_openai_adapter(config: Dict[str, Any]) -> Optional[OpenAIAdapter]:
    """Фабричная функция для создания адаптера OpenAI."""
    try:
        return OpenAIAdapter(config)
    except Exception as e:
        LOG.info("Не удалось создать адаптер OpenAI: %s", e)
        return None