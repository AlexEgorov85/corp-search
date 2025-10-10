from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent

LOG = logging.getLogger(__name__)


class ResultValidatorAgent(BaseAgent):
    """
    Контрольный агент для валидации результата шага с помощью LLM.
    Не использует эвристики — полностью делегирует решение LLM.
    """

    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__(descriptor, config)
        LOG.debug("ResultValidatorAgent инициализирован.")