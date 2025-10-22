# src/services/llm_service/__init__.py
"""
Единая точка входа для LLM-сервиса.
Предоставляет функцию ensure_llm(profile), которая возвращает
инстанцированный и закэшированный адаптер LLM по имени профиля.
"""
from .factory import ensure_llm

__all__ = ["ensure_llm"]