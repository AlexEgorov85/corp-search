# src/agents/SynthesizerAgent/core.py
"""
SynthesizerAgent — агент для генерации финального ответа на основе результатов шагов плана.

Особенности:
- Не выполняет бизнес-логику напрямую — вся логика вынесена в операцию `synthesize`.
- Наследуется от `BaseAgent`, автоматически загружает операции из папки `operations/`.
- Использует LLM из конфигурации (`llm_profile`).

Пример использования:
>>> agent = agent_registry.instantiate_agent("SynthesizerAgent", control=True)
>>> result = agent.execute_operation("synthesize", params={...}, context=ctx.to_dict())
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent

LOG = logging.getLogger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Контрольный агент для синтеза финального ответа.
    """

    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует агент.
        LLM и операции будут загружены автоматически при первом вызове `execute_operation`.
        """
        super().__init__(descriptor, config)
        LOG.debug("SynthesizerAgent инициализирован.")