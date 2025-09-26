# src/agents/base.py
# coding: utf-8
"""
BaseAgent — минималистичный и строгий базовый класс для агентов.
Контракт:
 - descriptor (AgentEntry) обязателен и должен содержать поля:
     - name, title, description, implementation, operations (dict)
 - Каждая операция в operations обязана содержать:
     - kind: "direct" | "composed"
     - description: str (не пустая)
 - Перед вызовом execute_operation агент должен быть инициализирован методом initialize()
 - Методы обработки операций:
     - _run_direct_operation(self, op_name: str, params: dict, context: dict) -> AgentResult
     - _run_composed_operation(self, op_name: str, params: dict, context: dict) -> AgentResult
   — обязательно возвращают AgentResult

Пример использования:
    agent = MyAgent(descriptor=descriptor, config=config)
    agent.initialize()
    res: AgentResult = agent.execute_operation("get_last_book", {"author": "Пушкин"})
    agent.close()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple, Callable

from src.services.results.agent_result import AgentResult  # обязателен в проекте

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class BaseAgent:
    """
    Строгий базовый класс агентов.

    Обязательные контракты:
    - descriptor: dict с ключами name, title, description, implementation, operations
    - operations: dict оперaций; у каждой операции обязателен 'kind' и 'description'
    - дочерние классы реализуют _run_direct_operation и/или _run_composed_operation
      и возвращают AgentResult
    """

    # минимальный набор обязательных полей в descriptor
    _REQUIRED_DESCRIPTOR_KEYS = {"name", "title", "description", "implementation", "operations"}

    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            descriptor: AgentEntry (обязательный)
            config: runtime-конфиг (опционально)
        """
        if not isinstance(descriptor, dict):
            raise ValueError("descriptor is required and must be a dict (AgentEntry).")
        missing = self._REQUIRED_DESCRIPTOR_KEYS - set(descriptor.keys())
        if missing:
            raise ValueError(f"descriptor missing required keys: {sorted(missing)}")

        self.descriptor: Dict[str, Any] = descriptor
        self.config: Dict[str, Any] = dict(config or {})
        # нормализуем операции и проверим их контракт
        self.operations: Dict[str, Dict[str, Any]] = {}
        self._load_and_validate_operations(self.descriptor["operations"])

        self._initialized: bool = False
        self.hooks: Dict[str, Callable[..., Any]] = {}

        LOG.debug("BaseAgent initialized: %s operations=%s", self.name, list(self.operations.keys()))

    # -------------------------
    # Простые свойства дескриптора
    # -------------------------
    @property
    def name(self) -> str:
        return str(self.descriptor["name"])

    @property
    def title(self) -> str:
        return str(self.descriptor.get("title", self.name))

    @property
    def description(self) -> str:
        return str(self.descriptor.get("description", ""))

    # -------------------------
    # Загрузка и валидация операций
    # -------------------------
    def _load_and_validate_operations(self, ops: Dict[str, Any]) -> None:
        """
        Нормализует и валидирует operations из descriptor.
        Требования к каждой операции:
          - это dict
          - содержит 'kind' == 'direct'|'composed'
          - содержит непустую 'description'
        """
        if not isinstance(ops, dict):
            raise ValueError("descriptor['operations'] must be a dict of operations")

        normalized: Dict[str, Dict[str, Any]] = {}
        for op_name, meta in ops.items():
            if not isinstance(meta, dict):
                raise ValueError(f"operation '{op_name}' meta must be a dict")
            kind = meta.get("kind")
            if kind not in ("direct", "composed"):
                raise ValueError(f"operation '{op_name}': 'kind' must be 'direct' or 'composed'")
            desc = meta.get("description")
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"operation '{op_name}': 'description' is required and must be non-empty string")
            normalized[op_name] = dict(meta, kind=kind, description=desc.strip())
        self.operations = normalized

    # -------------------------
    # Lifecycle
    # -------------------------
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента (подключения, LLM, кэши и т.п.).
        Дочерний класс может переопределить и вызвать super().initialize(config).

        Обязательное поведение: установить self._initialized = True в конце.
        """
        if config:
            self.config.update(config)
        self._initialized = True
        LOG.debug("Agent %s initialized; config keys: %s", self.name, list(self.config.keys()))

    def close(self) -> None:
        """
        Освобождение ресурсов. Переопределять при наличии соединений.
        Должно привести к состоянию _initialized == False.
        """
        self._initialized = False
        LOG.debug("Agent %s closed", self.name)

    # -------------------------
    # Выполнение операций
    # -------------------------
    def execute_operation(self, operation: str, params: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Выполнить операцию агента.

        Требования:
         - агент должен быть инициализирован (self._initialized == True)
         - operation должен присутствовать в descriptor['operations']
         - дочерние методы _run_direct_operation/_run_composed_operation обязаны вернуть AgentResult

        Возвращает AgentResult. В случае исключений возвращается AgentResult.error(...)
        """
        if not self._initialized:
            raise RuntimeError(f"Agent '{self.name}' must be initialized before execute_operation()")

        if operation not in self.operations:
            raise KeyError(f"Unknown operation '{operation}' for agent '{self.name}'")

        params = params or {}
        context = context or {}

        op_meta = self.operations[operation]
        kind = op_meta["kind"]

        # Лёгкая проверка обязательных параметров (если schema указана в op_meta['params'])
        ok, reason = self._basic_validate_params(params, op_meta.get("params"))
        if not ok:
            return AgentResult.error(f"params validation failed: {reason}")

        start = time.time()
        try:
            if kind == "direct":
                result = self._run_direct_operation(operation, params, context)
            else:  # composed
                result = self._run_composed_operation(operation, params, context)

            if not isinstance(result, AgentResult):
                raise TypeError(f"Operation handler must return AgentResult, got {type(result)}")

            # enrich metadata
            meta = getattr(result, "metadata", {}) or {}
            meta.setdefault("agent", self.name)
            meta.setdefault("operation", operation)
            meta.setdefault("elapsed_s", time.time() - start)
            result.metadata = meta
            return result

        except Exception as exc:  # intentionally broad: we convert any run-time error to AgentResult.error
            LOG.exception("Agent %s operation %s failed", self.name, operation)
            return AgentResult.error(f"operation '{operation}' failed: {exc}")

    # -------------------------
    # Методы, которые должны реализовать потомки
    # -------------------------
    def _run_direct_operation(self, op_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """
        Реализуйте в дочернем классе логику direct-операции.
        Обязательно вернуть AgentResult.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _run_direct_operation")

    def _run_composed_operation(self, op_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """
        Реализуйте пайплайн-логику для composed-операций, если требуется.
        По умолчанию — бросает исключение.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement composed operations '{op_name}'")

    # -------------------------
    # Простая валидация параметров (light-weight)
    # -------------------------
    @staticmethod
    def _basic_validate_params(params: Dict[str, Any], params_schema: Optional[Any]) -> Tuple[bool, str]:
        """
        Очень простая проверка: если params_schema — dict и содержит ключи с {required: True},
        проверяем наличие этих ключей в params.
        Если params_schema == "freeform" или None — пропускаем.
        """
        if not params_schema:
            return True, "no schema"
        if isinstance(params_schema, str) and params_schema == "freeform":
            return True, "freeform"
        if not isinstance(params_schema, dict):
            return True, "schema not dict"

        missing = []
        for k, v in params_schema.items():
            if isinstance(v, dict):
                if v.get("required") and k not in params:
                    missing.append(k)
            else:
                # если определён как просто "required" (строка) — трактуем как обязательный
                if str(v).lower() in ("required", "req", "r") and k not in params:
                    missing.append(k)
        if missing:
            return False, f"missing required params: {missing}"
        return True, "ok"

    # -------------------------
    # Hooks / утилиты
    # -------------------------
    def register_hook(self, name: str, func: Callable[..., Any]) -> None:
        """Зарегистрировать hook (тесты/расширения)."""
        self.hooks[name] = func

    def call_hook(self, name: str, *args, **kwargs) -> Any:
        """Вызвать зарегистрированный hook."""
        fn = self.hooks.get(name)
        if not fn:
            return None
        return fn(*args, **kwargs)

    def describe(self) -> Dict[str, Any]:
        """Краткое описание агента (для логов/UI)."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "operations": list(self.operations.keys()),
            "config_keys": list(self.config.keys()),
        }
