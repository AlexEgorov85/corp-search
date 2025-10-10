from __future__ import annotations
import enum
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from src.model.agent_result import AgentResult

LOG = logging.getLogger(__name__)


class OperationKind(str, enum.Enum):
    DIRECT = "direct"
    VALIDATION = "validation"
    SEMANTIC = "semantic"
    CONTROL = "control"

    def __str__(self) -> str:
        return self.value


class BaseOperation(ABC):
    """
    Базовый класс для всех операций агентов.
    
    Каждая операция должна:
      - реализовать метод `run`
      - определить атрибуты класса: `description`, `params_schema`, `outputs_schema`
    
    Пример:
        class MyOp(BaseOperation):
            description = "Описание операции"
            params_schema = {"param1": {"type": "string", "required": True}}
            outputs_schema = {"result": {"type": "string"}}
            
            def run(self, params: dict, context: dict, agent) -> AgentResult:
                ...
    """
    kind: OperationKind = OperationKind.DIRECT
    description: str = "Базовая операция"
    params_schema: Dict[str, Any] = {}
    outputs_schema: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Валидация при создании подкласса
        if not isinstance(cls.kind, OperationKind):
            raise TypeError(f"'kind' должен быть OperationKind, получено: {type(cls.kind)}")
        if not isinstance(cls.description, str) or not cls.description.strip():
            raise ValueError(f"'description' должен быть непустой строкой в {cls.__name__}")
        if not isinstance(cls.params_schema, dict):
            raise TypeError(f"'params_schema' должен быть dict в {cls.__name__}")
        if not isinstance(cls.outputs_schema, dict):
            raise TypeError(f"'outputs_schema' должен быть dict в {cls.__name__}")

    @abstractmethod
    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        pass

    @classmethod
    def get_manifest(cls) -> Dict[str, Any]:
        return {
            "kind": str(cls.kind),
            "description": cls.description,
            "params": cls.params_schema,
            "outputs": cls.outputs_schema,
        }