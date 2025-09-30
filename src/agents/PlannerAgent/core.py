# src/agents/PlannerAgent/core.py
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent

LOG = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Планировщик, который разбивает сложный вопрос на атомарные подвопросы.
    Использует новую архитектуру BaseAgent с ленивой инициализацией и загрузкой операций из папки operations/.
    """
    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        # Просто вызываем конструктор базового класса. Вся инициализация (LLM, операции) происходит автоматически.
        super().__init__(descriptor, config)
        LOG.debug("PlannerAgent инициализирован.")