# services/llm_service/adapters/base.py
"""
Базовый класс для адаптеров LLM.
Определяет общий интерфейс, который должны реализовывать все адаптеры.
"""

from abc import ABC, abstractmethod

class BaseLLMAdapter(ABC):
    """Абстрактный базовый класс для адаптеров LLM."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на основе переданного промпта.
        Args:
            prompt (str): Текстовый промпт для модели.
        Returns:
            str: Сгенерированный текст.
        """
        pass