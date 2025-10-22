# src/services/llm_service/adapters/llama_cpp_adapter.py
"""
Адаптер для GGUF-моделей через llama_cpp.
Полностью совместим с LLMRequest / LLMResponse и новой архитектурой llm_service.
Особенности:
- Работает только с GGUF-моделями (не с Transformers)
- Поддерживает Qwen-специфичные теги рассуждений (thinking / thinking_end)
- Использует chat-like формат промпта (System/User/Assistant)
- Возвращает tokens_used через llama_cpp metadata
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Tuple, Optional

from .base import BaseLLMAdapter
from src.services.llm_service.model.request import LLMRequest, LLMMessage
from src.services.llm_service.model.response import LLMResponse

LOG = logging.getLogger(__name__)


class LlamaCppAdapter(BaseLLMAdapter):
    """
    Адаптер для GGUF-моделей через llama_cpp.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует адаптер.

        Args:
            config (Dict[str, Any]): Конфигурация с ключами:
                - model_path: путь к .gguf файлу
                - n_ctx: максимальная длина контекста (по умолчанию 16384)
                - n_gpu_layers: число слоёв на GPU (если используется)
                - temperature, top_p, max_tokens: параметры генерации
                - backend_kwargs: дополнительные параметры для Llama(...)
        """
        super().__init__(config)

        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("Для LlamaCppAdapter требуется 'model_path' в конфигурации")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "Не установлен llama_cpp. Выполните: pip install llama-cpp-python"
            ) from e

        self.model_path = model_path
        n_ctx = config.get("n_ctx", 16384)
        n_gpu_layers = config.get("n_gpu_layers", 0)
        n_batch = config.get("n_batch", 512)

        LOG.info(f"Загрузка GGUF-модели {model_path} (n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")

        # Создаём экземпляр модели
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            **config.get("backend_kwargs", {})
        )

        # Определяем, является ли модель Qwen (для обработки thinking-тегов)
        self.is_qwen_model = "qwen" in model_path.lower()

    def _convert_messages_to_chat_format(self, messages: List[LLMMessage]) -> str:
        """
        Конвертирует список LLMMessage в текстовый промпт в формате:
        System: ...
        User: ...
        Assistant: ...
        """
        lines = []
        for msg in messages:
            role = msg.role.lower()
            content = msg.content.strip()
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "tool":
                # llama_cpp не поддерживает tool, используем user
                lines.append(f"User: {content}")
            else:
                LOG.warning(f"Неизвестная роль '{msg.role}', преобразована в 'User'")
                lines.append(f"User: {content}")
        return "\n".join(lines)

    def generate_with_request(self, request: LLMRequest, **kwargs) -> Tuple[str, LLMResponse]:
        """
        Генерирует ответ на основе LLMRequest.
        Возвращает (основной ответ, структурированный LLMResponse).
        """
        try:
            # 1. Формируем промпт
            prompt = self._convert_messages_to_chat_format(request.messages)

            # 2. Генерация через llama_cpp
            gen_kwargs = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": ["</s>", "###", "thinking_end"] if self.is_qwen_model else ["</s>", "###"],
                "echo": False,
            }

            response = self.model(prompt, **gen_kwargs)

            # 3. Извлекаем сырой текст
            raw_text = response["choices"][0]["text"].strip()

            # 4. Парсим через универсальный парсер
            llm_response = LLMResponse.from_raw(raw_text)

            # 5. Дополнительно: если модель Qwen — извлекаем thinking вручную (на случай, если парсер пропустил)
            if self.is_qwen_model and not llm_response.thinking:
                thinking_match = re.search(
                    r"thinking(.*?)thinking_end", raw_text, re.DOTALL | re.IGNORECASE
                )
                if thinking_match:
                    llm_response.thinking = thinking_match.group(1).strip()
                    # Убираем теги из answer
                    clean_answer = re.sub(
                        r"thinking.*?thinking_end", "", raw_text, flags=re.DOTALL | re.IGNORECASE
                    ).strip()
                    llm_response.answer = clean_answer

            # 6. Оценка токенов (приблизительно)
            prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
            llm_response.tokens_used = prompt_tokens + completion_tokens

            return llm_response.answer, llm_response

        except Exception as e:
            LOG.exception("Ошибка в LlamaCppAdapter.generate_with_request")
            error_response = LLMResponse(
                raw_text="",
                thinking="",
                answer="",
                json_answer=None,
                tokens_used=0,
                metadata={"error": str(e)}
            )
            return "", error_response

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Генерирует ответ на основе простого текстового промпта.
        Для совместимости оборачивает промпт в LLMRequest.
        """
        from src.services.llm_service.model.request import LLMMessage, LLMRequest
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content="Ты — полезный помощник."),
                LLMMessage(role="user", content=prompt)
            ],
            **kwargs
        )
        answer, _ = self.generate_with_request(request)
        return answer

    def close(self) -> None:
        """Освобождает ресурсы модели."""
        try:
            if hasattr(self.model, 'close') and callable(self.model.close):
                self.model.close()
            del self.model
        except Exception as e:
            LOG.debug(f"Ошибка при закрытии LlamaCppAdapter: {e}")