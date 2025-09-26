# services/llm_service/wrappers/dummy_wrapper.py
"""
Мок-реализация LLM для тестов и отладки.
"""

from .base_wrapper import BaseLLMWrapper

class DummyLLM(BaseLLMWrapper):
    """
    Заглушка для LLM. Всегда возвращает предопределенный ответ.
    Используется для тестирования и в случае, когда реальная LLM недоступна.
    """

    def __init__(self, response: str = "SELECT 1;"):
        """
        Инициализирует мок.
        Args:
            response (str): Ответ, который будет возвращаться на любой промпт.
        """
        self._response = response

    def generate(self, prompt: str) -> str:
        """Возвращает предопределенный ответ."""
        return self._response

    def close(self) -> None:
        """Ничего не делает."""
        pass