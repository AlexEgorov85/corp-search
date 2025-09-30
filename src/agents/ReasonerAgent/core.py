# src/agents/ReasonerAgent/core.py
from __future__ import annotations
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent
import logging

LOG = logging.getLogger(__name__)

class ReasonerAgent(BaseAgent):
    """
    ReasonerAgent — агент принятия решений (ReAct-style).
    Все стадии реализованы как отдельные операции в папке operations/.
    """
    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__(descriptor, config or {})
        LOG.debug("ReasonerAgent инициализирован.")