# src/agents/ResultValidatorAgent/core.py
"""
ResultValidatorAgent — контрольный агент для валидации результатов шагов.
Особенности:
- Не содержит бизнес-логики — только вызывает LLM через операцию `validate_result`
- Использует `BaseAgent` → автоматическая загрузка операций из папки operations/
- Поддерживает LLM из конфига (`llm_profile`)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent

LOG = logging.getLogger(__name__)


class ResultValidatorAgent(BaseAgent):
    """
    Агент для валидации результатов шагов с помощью LLM.
    """

    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Инициализация агента.
        LLM и операции будут загружены автоматически при первом вызове `execute_operation`.
        """
        super().__init__(descriptor, config)
        LOG.debug("ResultValidatorAgent инициализирован.")