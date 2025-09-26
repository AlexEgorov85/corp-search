# services/llm_service/__init__.py
"""
Основной интерфейс пакета llm_service.
Экспортирует функции для создания, получения и управления экземплярами LLM.
Также экспортирует вспомогательные утилиты для работы с промптами.
"""

from .factory import create_llm, ensure_llm, get_llm, close_llm
from .config import list_profiles, get_agent_config, get_all_agents_config
from .utils import safe_format, strip_code_fences

__all__ = [
    "create_llm",
    "ensure_llm",
    "get_llm",
    "close_llm",
    "list_profiles",
    "get_agent_config",
    "get_all_agents_config",
    "safe_format",
    "strip_code_fences",
]