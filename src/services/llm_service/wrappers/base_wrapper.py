# services/llm_service/wrappers/base_wrapper.py
"""
Базовый класс для оберток LLM.
Определяет общий интерфейс.
"""

from abc import ABC, abstractmethod

class BaseLLMWrapper(ABC):
    """Абстрактный базовый класс для оберток LLM."""

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

    @abstractmethod
    def close(self) -> None:
        """Освобождает ресурсы, связанные с моделью."""
        pass