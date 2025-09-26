# services/llm_service/wrappers/__init__.py
"""
Пакет оберток для унификации интерфейса LLM.
"""

from .safe_wrapper import LLMSafeWrapper
from .dummy_wrapper import DummyLLM

__all__ = ["LLMSafeWrapper", "DummyLLM"]