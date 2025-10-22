# src/services/llm_service/adapters/universal_transformers_adapter.py
"""
Универсальный адаптер для Hugging Face Transformers.

Особенности:
- Работает с любыми causal LM и seq2seq моделями через AutoModel
- Использует `chat_template` для формирования промпта
- Полностью совместим с LLMRequest / LLMResponse
- Не дублирует функциональность Transformers — только адаптирует её под наш интерфейс

Архитектурные принципы:
- Нет кастомной генерации — всё через `model.generate()`
- Нет ручного управления токенами — всё через `tokenizer`
- Нет избыточных обёрток — адаптер сам является точкой входа
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from src.services.llm_service.adapters.base import BaseLLMAdapter
from src.services.llm_service.model.request import LLMRequest, LLMMessage
from src.services.llm_service.model.response import LLMResponse

LOG = logging.getLogger(__name__)


class UniversalTransformersAdapter(BaseLLMAdapter):
    """
    Адаптер для моделей Hugging Face Transformers.

    Автоматически определяет тип модели (causal или seq2seq).
    Использует chat_template для формирования промпта.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует адаптер.

        Args:
            config (Dict[str, Any]): Конфигурация с ключами:
                - model_path: путь или имя модели на Hugging Face
                - device: "cpu" или "cuda" (опционально)
                - backend_kwargs: дополнительные параметры для AutoModel
        """
        super().__init__(config)
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("Для UniversalTransformersAdapter требуется 'model_path' в конфигурации")

        self.model_path = model_path
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        backend_kwargs = config.get("backend_kwargs", {})

        LOG.info(f"Загрузка модели {model_path} на устройстве {self.device}")

        # Загрузка токенизатора
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Автоматическое определение типа модели
        try:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **backend_kwargs
            )
            self.model_type = "causal"
        except Exception as e1:
            LOG.debug(f"Не удалось загрузить как causal LM: {e1}")
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path, trust_remote_code=True, **backend_kwargs
                )
                self.model_type = "seq2seq"
            except Exception as e2:
                raise RuntimeError(
                    f"Не удалось загрузить модель ни как causal, ни как seq2seq: {e1}, {e2}"
                )

        self.model.to(self.device)
        self.model.eval()

        # Установка pad_token, если отсутствует
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

    def _convert_messages_to_chat_format(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """
        Конвертирует список LLMMessage в формат [{'role': ..., 'content': ...}],
        совместимый с apply_chat_template.
        """
        chat_messages = []
        for msg in messages:
            role = msg.role.lower()
            if role == "tool":
                role = "user"  # Большинство chat_template не поддерживают 'tool'
            elif role not in ("system", "user", "assistant"):
                LOG.warning(f"Неизвестная роль '{msg.role}', преобразована в 'user'")
                role = "user"
            chat_messages.append({"role": role, "content": msg.content})
        return chat_messages

    def _generate_raw(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, request: LLMRequest) -> str:
        """
        Выполняет генерацию на основе токенизированного ввода.
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Убираем входной промпт из ответа (для causal моделей)
        if self.model_type == "causal":
            output_ids = output_ids[:, input_ids.shape[1]:]

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text.strip()

    def generate_with_request(self, request: LLMRequest, **kwargs) -> Tuple[str, LLMResponse]:
        """
        Генерирует ответ на основе LLMRequest.

        Возвращает (основной ответ, структурированный LLMResponse).
        """
        try:
            # 1. Преобразуем сообщения в чат-формат
            chat_messages = self._convert_messages_to_chat_format(request.messages)

            # 2. Применяем chat_template
            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback: простая конкатенация
                prompt = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_messages)
                if not prompt.endswith("Assistant:"):
                    prompt += "\nAssistant:"

            # 3. Токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("n_ctx", 4096) - request.max_tokens,
            )
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            # 4. Генерация
            raw_text = self._generate_raw(input_ids, attention_mask, request)

            # 5. Парсинг ответа
            llm_response = LLMResponse.from_raw(raw_text)
            # Оценка токенов (упрощённо)
            llm_response.tokens_used = len(self.tokenizer.encode(raw_text))

            return llm_response.answer, llm_response

        except Exception as e:
            LOG.exception("Ошибка в UniversalTransformersAdapter.generate_with_request")
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
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            LOG.debug(f"Ошибка при закрытии UniversalTransformersAdapter: {e}")