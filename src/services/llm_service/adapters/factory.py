# services/llm_service/adapters/factory.py
"""
Фабрика для создания адаптеров на основе конфигурации.
Определяет, какой адаптер создать в зависимости от указанного бэкенда.
"""

from typing import Any, Dict, Optional

from .openai_adapter import create_openai_adapter
from .llama_cpp_adapter import create_llama_cpp_adapter

def create_adapter_from_config(config: Dict[str, Any]) -> Optional[Any]:
    """
    Создает и возвращает адаптер на основе конфигурации.
    Args:
        config (Dict[str, Any]): Конфигурация профиля LLM.
    Returns:
        Optional[Any]: Экземпляр адаптера или None.
    """
    backend = (config.get("backend") or "").lower()

    if backend in ("openai",):
        return create_openai_adapter(config)
    elif backend in ("llama_cpp", "llama", "llama-cpp"):
        return create_llama_cpp_adapter(config)
    else:
        # Автоопределение: пробуем OpenAI, затем llama.cpp
        adapter = create_openai_adapter(config)
        if adapter:
            return adapter
        return create_llama_cpp_adapter(config)