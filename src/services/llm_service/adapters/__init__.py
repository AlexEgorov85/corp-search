# services/llm_service/adapters/__init__.py
"""
Пакет адаптеров для различных бэкендов LLM.
"""

from .openai_adapter import OpenAIAdapter
from .llama_cpp_adapter import LlamaCppAdapter

__all__ = ["OpenAIAdapter", "LlamaCppAdapter"]