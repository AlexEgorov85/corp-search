# src/agents/DataAnalysisAgent/core.py
"""
DataAnalysisAgent — универсальный агент для анализа любых данных.

Особенности:
- Не привязан к предметной области (книги, акты и т.д.).
- Имеет одну операцию `analyze`, которая сама определяет тип данных и стратегию обработки.
- Поддерживает LLM-синтез выводов.
- Полностью совместим с BaseAgent и GraphContext.

Пример использования в executor_node:
>>> agent = agent_registry.instantiate_agent("DataAnalysisAgent")
>>> result = agent.execute_operation("analyze", params={
...     "subquestion_text": "Какие книги написал Пушкин?",
...     "raw_output": [{"title": "Евгений Онегин"}]
... }, context=ctx.to_dict())
>>> print(result.status)  # "ok"
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent

LOG = logging.getLogger(__name__)


class DataAnalysisAgent(BaseAgent):
    """
    Контрольный агент для универсального анализа данных.
    Наследуется от BaseAgent и автоматически загружает операции из папки operations/.
    """

    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует агент.
        Конфигурация может содержать:
          - llm_profile: имя профиля LLM для синтеза выводов
        """
        super().__init__(descriptor, config)
        LOG.debug("DataAnalysisAgent инициализирован.")