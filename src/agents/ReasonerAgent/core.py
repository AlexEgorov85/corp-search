# src/agents/ReasonerAgent/core.py
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent

LOG = logging.getLogger(__name__)

class ReasonerAgent(BaseAgent):
    """
    Агент рассуждений (ReasonerAgent).
    Управляет 5-этапным пайплайном для одного подвопроса.
    Вся логика сосредоточена в операции `reason_step`.
    """
    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__(descriptor, config)
        LOG.debug("ReasonerAgent инициализирован.")