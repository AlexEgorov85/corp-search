# services/llm_service/wrappers/safe_wrapper.py
"""
Безопасная обертка для LLM-адаптеров.
Унифицирует интерфейс и обеспечивает graceful degradation при ошибках.
"""

import logging
from typing import Any, Dict, Optional

from .base_wrapper import BaseLLMWrapper

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

class LLMSafeWrapper(BaseLLMWrapper):
    """
    Безопасная обертка для LLM-адаптера.
    Пытается вызвать метод `generate` у внутреннего адаптера и обрабатывает возможные ошибки.
    """

    def __init__(self, model_obj: Any, meta: Optional[Dict[str, Any]] = None):
        """
        Инициализирует обертку.
        Args:
            model_obj (Any): Объект адаптера, реализующий метод `generate`.
            meta (Dict[str, Any], optional): Метаданные (провайдер, профиль, путь к модели).
        """
        self._model = model_obj
        self.meta = meta or {}

    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ, оборачивая вызов адаптера в try-except.
        Если адаптер не реализует `generate`, пытается использовать `invoke`, `predict` или `__call__`.
        """
        if self._model is None:
            raise RuntimeError("LLM не инициализирована")

        # Пытаемся вызвать различные методы в порядке приоритета
        methods = ["generate", "invoke", "predict", "__call__"]
        for method_name in methods:
            if hasattr(self._model, method_name):
                try:
                    method = getattr(self._model, method_name)
                    if method_name == "__call__":
                        result = method(prompt)
                    else:
                        result = method(prompt)
                    return self._extract_text(result)
                except Exception as e:
                    LOG.debug("Метод %s не сработал: %s", method_name, e)
                    continue

        # Если ни один метод не сработал
        raise RuntimeError("Не удалось вызвать LLM — неподдерживаемый интерфейс")

    def _extract_text(self, out: Any) -> str:
        """Извлекает текст из ответа LLM, независимо от его формата."""
        try:
            if isinstance(out, str):
                return out
            if isinstance(out, dict):
                # OpenAI-стиль
                if "choices" in out and len(out["choices"]) > 0:
                    choice = out["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                    if "text" in choice:
                        return choice["text"]
                return str(out)
            # LangChain-стиль
            if hasattr(out, "generations") and len(out.generations) > 0:
                first_gen = out.generations[0][0]
                if hasattr(first_gen, "text"):
                    return first_gen.text
            return str(out)
        except Exception:
            LOG.debug("Не удалось извлечь текст из ответа LLM", exc_info=True)
            return ""

    def close(self) -> None:
        """Закрывает внутренний адаптер, если у него есть метод `close`."""
        try:
            if hasattr(self._model, "close"):
                self._model.close()
        except Exception as e:
            LOG.debug("Ошибка при закрытии LLM: %s", e)
        finally:
            self._model = None