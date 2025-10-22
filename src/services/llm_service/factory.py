# src/services/llm_service/factory.py
"""
Фабрика LLM с поддержкой нескольких бэкендов:
- transformers → UniversalTransformersAdapter
- llama_cpp → LlamaCppAdapter
- gigachat → GigaChatAdapter (если нужно)
"""

import logging
import threading
from typing import Any, Dict

from src.common.settings import LLM_PROFILES
from src.services.llm_service.adapters.universal_transformers_adapter import UniversalTransformersAdapter
from src.services.llm_service.adapters.llama_cpp_adapter import LlamaCppAdapter
# from src.services.llm_service.adapters.gigachat_adapter import GigaChatAdapter  # опционально

LOG = logging.getLogger(__name__)

_LLM_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def ensure_llm(profile: str):
    """
    Возвращает закэшированный или новый экземпляр LLM по имени профиля.
    Автоматически выбирает адаптер по полю 'backend' в конфигурации.
    """
    with _CACHE_LOCK:
        if profile in _LLM_CACHE:
            return _LLM_CACHE[profile]

        config = LLM_PROFILES.get(profile)
        if not config:
            raise ValueError(f"Профиль '{profile}' не найден в LLM_PROFILES")

        backend = config.get("backend", "transformers").lower()

        if backend == "transformers":
            adapter = UniversalTransformersAdapter(config)
        elif backend == "llama_cpp":
            adapter = LlamaCppAdapter(config)
        # elif backend == "gigachat":
        #     adapter = GigaChatAdapter(config)
        else:
            raise ValueError(f"Неизвестный backend '{backend}' в профиле '{profile}'")

        _LLM_CACHE[profile] = adapter
        LOG.info(f"Загружена модель для профиля '{profile}' (backend={backend}): {config.get('model_path', 'N/A')}")
        return adapter