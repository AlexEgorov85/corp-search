# services/llm_service/cache.py
"""
Модуль для управления кэшем экземпляров LLM.
Обеспечивает потокобезопасный доступ к кэшу с помощью блокировки.
"""

import threading
from typing import Dict, Optional

from src.services.db_service.executor import LOG

from .wrappers import LLMSafeWrapper

# Глобальный кэш и блокировка
_LLM_LOCK = threading.RLock()
_LLM_CACHE: Dict[str, LLMSafeWrapper] = {}

def get_cached_llm(profile: str) -> Optional[LLMSafeWrapper]:
    """Возвращает закэшированный экземпляр LLM для профиля."""
    with _LLM_LOCK:
        return _LLM_CACHE.get(profile)

def cache_llm(profile: str, llm: LLMSafeWrapper) -> None:
    """Кэширует экземпляр LLM для профиля."""
    with _LLM_LOCK:
        _LLM_CACHE[profile] = llm

def close_cached_llm(profile: str) -> None:
    """Закрывает и удаляет из кэша экземпляр LLM для профиля."""
    with _LLM_LOCK:
        inst = _LLM_CACHE.pop(profile, None)
        if inst:
            try:
                inst.close()
            except Exception as e:
                LOG.debug("Ошибка при закрытии LLM (profile=%s): %s", profile, e)

def close_all_cached_llms() -> None:
    """Закрывает и очищает все закэшированные экземпляры LLM."""
    with _LLM_LOCK:
        keys = list(_LLM_CACHE.keys())
        for k in keys:
            close_cached_llm(k)