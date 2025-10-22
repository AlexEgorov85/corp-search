from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent

LOG = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Агент-планировщик, который разбивает сложный вопрос на атомарные подвопросы.
    
    Особенности реализации:
    - Использует новую архитектуру с LLMRequest для формирования запросов
    - Полностью соответствует обновленному интерфейсу LLM
    - Сохраняет все необходимые метаданные для отладки и анализа
    - Обеспечивает совместимость со всеми типами LLM через единый интерфейс
    
    Пример использования:
    ```
    planner = PlannerAgent(descriptor, config={"llm_profile": "qwen_thinking"})
    result = planner.execute_operation("plan", {
        "question": "Какие книги написал Пушкин?",
        "tool_registry_snapshot": tool_registry
    })
    ```
    """
    
    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует агент-планировщик.
        
        Args:
            descriptor: Метаданные агента из реестра (name, title, implementation и т.д.)
            config: Конфигурация агента (llm_profile, db_uri и другие параметры)
        """
        super().__init__(descriptor, config)
        LOG.debug("PlannerAgent инициализирован.")