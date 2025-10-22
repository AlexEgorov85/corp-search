# src/agents/base.py
# coding: utf-8
"""
BaseAgent — унифицированный базовый класс для всех агентов.
ОСНОВНЫЕ ПРИНЦИПЫ:
1. Агенты НЕ требуют ручного вызова .initialize() — инициализация происходит автоматически при первом execute_operation.
2. LLM автоматически создаётся из config["llm_profile"] через ensure_llm(profile).
3. Операции размещаются в папке operations/ рядом с core.py: каждый файл <operation_name>.py должен содержать
   класс Operation, унаследованный от BaseOperation.
4. Агент должен наследоваться от BaseAgent и реализовывать только бизнес-логику в файлах operations/.

СТРУКТУРА АГЕНТА:
src/agents/MyAgent/
├── __init__.py
├── core.py                 # from .core import MyAgent
└── operations/
    ├── op1.py              # class Operation(BaseOperation): ...
    └── op2.py              # class Operation(BaseOperation): ...

КОНФИГУРАЦИЯ:
В реестре (control_registry.py или tool_registry.py) указывается:
{
  "config": {
    "llm_profile": "default",   # ← имя профиля из LLM_PROFILES
    "db_uri": "...",            # ← любые параметры
    ...
  }
}

ПРИМЕР ИСПОЛЬЗОВАНИЯ В НОДЕ:
agent = agent_registry.instantiate_agent("MyAgent", control=True)
result = agent.execute_operation("op1", {"param": "value"})  # LLM и операции инициализируются автоматически
"""

from __future__ import annotations
import importlib.util
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from src.agents.operations_base import BaseOperation
from src.model.agent_result import AgentResult
from src.services.llm_service import ensure_llm  # ← обновлённый импорт

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class BaseAgent:
    """
    Базовый класс для всех агентов.

    Атрибуты:
        descriptor (Dict[str, Any]): Метаданные агента из реестра (name, title, implementation и т.д.).
        config (Dict[str, Any]): Конфигурация агента (из поля "config" в реестре).
        llm (Optional[Any]): Экземпляр LLM, если в config указан "llm_profile".
        _operations (Dict[str, type[BaseOperation]]): Кэш загруженных классов операций из папки operations/.
        _initialized (bool): Флаг, показывающий, была ли выполнена инициализация.
    """

    # Обязательные поля в descriptor
    _REQUIRED_DESCRIPTOR_KEYS = {"name", "title", "description", "implementation"}

    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует базовый агент.

        Args:
            descriptor (Dict[str, Any]): Метаданные агента из реестра.
            config (Optional[Dict[str, Any]]): Конфигурация агента.

        Raises:
            ValueError: Если в descriptor отсутствуют обязательные поля.
        """
        if not isinstance(descriptor, dict):
            raise ValueError("descriptor must be a dict (AgentEntry).")
        missing = self._REQUIRED_DESCRIPTOR_KEYS - set(descriptor.keys())
        if missing:
            raise ValueError(f"descriptor missing required keys: {sorted(missing)}")

        self.descriptor: Dict[str, Any] = descriptor
        self.config: Dict[str, Any] = dict(config or {})
        self.llm: Optional[Any] = None
        # Храним классы операций (не экземпляры!)
        self._operations: Dict[str, type[BaseOperation]] = {}
        self._initialized: bool = False
        LOG.debug("BaseAgent инициализирован: %s", self.name)

    # -------------------------
    # Свойства дескриптора
    # -------------------------

    @property
    def name(self) -> str:
        """Возвращает имя агента из descriptor['name']."""  
        return str(self.descriptor["name"])

    @property
    def title(self) -> str:
        """Возвращает человекочитаемое название из descriptor['title']."""  
        return str(self.descriptor.get("title", self.name))

    @property
    def description(self) -> str:
        """Возвращает описание из descriptor['description']."""  
        return str(self.descriptor.get("description", ""))

    # -------------------------
    # Ленивая инициализация (выполняется автоматически)
    # -------------------------

    def _lazy_initialize(self) -> None:
        """Выполняет инициализацию при первом вызове execute_operation."""
        if self._initialized:
            return

        # === Автоматическая инициализация LLM ===
        llm_profile = self.config.get("llm_profile")
        if llm_profile:
            try:
                self.llm = ensure_llm(llm_profile)
                if self.llm is None:
                    LOG.warning("LLM не создана для профиля '%s' в агенте %s", llm_profile, self.name)
                else:
                    LOG.debug("LLM успешно создана для агента %s (профиль: %s)", self.name, llm_profile)
            except Exception as e:
                LOG.exception("Ошибка инициализации LLM для агента %s: %s", self.name, e)
                self.llm = None

        # === Загрузка операций из папки operations/ ===
        self._load_operations_from_folder()
        self._initialized = True
        LOG.debug("Агент %s инициализирован; config keys: %s", self.name, list(self.config.keys()))

    def _load_operations_from_folder(self) -> None:
        """
        Сканирует папку operations/ рядом с модулем агента и загружает все операции.
        Каждый файл <operation_name>.py должен содержать:
          - класс Operation, унаследованный от BaseOperation.
        """
        try:
            agent_module = self.__class__.__module__
            spec = importlib.util.find_spec(agent_module)
            if spec is None or spec.origin is None:
                raise ValueError(f"Не удалось найти origin для модуля {agent_module}")

            agent_file = Path(spec.origin).resolve()
            operations_dir = agent_file.parent / "operations"

            if not operations_dir.exists():
                LOG.debug("Папка operations не найдена для агента %s", self.name)
                return

            ops = {}
            for op_file in operations_dir.glob("*.py"):
                if op_file.name.startswith("_"):
                    continue
                op_name = op_file.stem
                try:
                    spec_op = importlib.util.spec_from_file_location(
                        f"{agent_module}.operations.{op_name}", op_file
                    )
                    if spec_op is None:
                        LOG.warning("Не удалось создать spec для %s", op_file)
                        continue
                    mod = importlib.util.module_from_spec(spec_op)
                    spec_op.loader.exec_module(mod)

                    # Требуем наличие класса Operation
                    if not hasattr(mod, "Operation"):
                        LOG.error("Файл %s не содержит класса 'Operation'", op_file)
                        continue
                    op_cls = getattr(mod, "Operation")
                    if not (inspect.isclass(op_cls) and issubclass(op_cls, BaseOperation)):
                        LOG.error("Operation в %s не наследуется от BaseOperation", op_file)
                        continue
                    ops[op_name] = op_cls
                    LOG.debug("Загружена операция %s для агента %s", op_name, self.name)
                except Exception as e:
                    LOG.exception("Ошибка загрузки операции %s из %s: %s", op_name, op_file, e)
            self._operations = ops
        except Exception as e:
            LOG.exception("Ошибка при загрузке операций для агента %s: %s", self.name, e)

    # -------------------------
    # Методы для AgentRegistry (анализ без инициализации)
    # -------------------------

    @staticmethod
    def _load_operations_from_module_path(module_path: str) -> Dict[str, Any]:
        """
        Анализирует папку operations/ и возвращает манифесты операций.
        Используется AgentRegistry для валидации без создания экземпляра агента.
        """
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None or spec.origin is None:
                return {}
            agent_file = Path(spec.origin).resolve()
            operations_dir = agent_file.parent / "operations"
            if not operations_dir.exists():
                return {}

            ops = {}
            for op_file in operations_dir.glob("*.py"):
                if op_file.name.startswith("_"):
                    continue
                op_name = op_file.stem
                try:
                    spec_op = importlib.util.spec_from_file_location(
                        f"{module_path}.operations.{op_name}", op_file
                    )
                    if spec_op is None:
                        continue
                    mod = importlib.util.module_from_spec(spec_op)
                    spec_op.loader.exec_module(mod)
                    if not hasattr(mod, "Operation"):
                        continue
                    op_cls = getattr(mod, "Operation")
                    if not (inspect.isclass(op_cls) and issubclass(op_cls, BaseOperation)):
                        continue
                    ops[op_name] = op_cls.get_manifest()
                except Exception as e:
                    LOG.debug("Не удалось получить манифест для %s: %s", op_name, e)
                    ops[op_name] = {
                        "kind": "direct",
                        "description": f"Операция {op_name} (манифест недоступен)"
                    }
            return ops
        except Exception as e:
            LOG.debug("Не удалось загрузить операции для модуля %s: %s", module_path, e)
            return {}

    @classmethod
    def discover_operations(cls, descriptor: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обнаруживает операции агента из папки operations/ без инициализации LLM и без создания экземпляра.
        Используется AgentRegistry для валидации.
        """
        impl = descriptor.get("implementation")
        if not impl or ":" not in impl:
            return {}
        module_path, _ = impl.rsplit(":", 1)
        return cls._load_operations_from_module_path(module_path)

    # -------------------------
    # Выполнение операций
    # -------------------------

    def execute_operation(
        self,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Выполняет операцию агента.

        Логика:
        1. Вызывает _lazy_initialize() для гарантии инициализации.
        2. Проверяет существование операции в self._operations.
        3. Создаёт экземпляр операции и вызывает метод run().
        4. Оборачивает результат в AgentResult с метаданными.

        Args:
            operation (str): Имя операции (без расширения .py).
            params (Optional[Dict[str, Any]]): Параметры операции.
            context (Optional[Dict[str, Any]]): Контекст выполнения.

        Returns:
            AgentResult: Результат выполнения операции.

        Raises:
            KeyError: Если операция не найдена.
        """
        # === Автоматическая инициализация ===
        self._lazy_initialize()

        if operation not in self._operations:
            available = list(self._operations.keys())
            raise KeyError(f"Операция '{operation}' не найдена у агента '{self.name}'. Доступны: {available}")

        params = params or {}
        context = context or {}

        op_cls = self._operations[operation]  # Это класс, унаследованный от BaseOperation
        start = time.time()
        try:
            # Создаём экземпляр операции
            op_instance = op_cls()
            # Вызываем метод run
            result = op_instance.run(params, context, self)
            if not isinstance(result, AgentResult):
                raise TypeError(f"Операция должна вернуть AgentResult, получено: {type(result)}")

            # Явно устанавливаем поля agent и operation
            if result.agent is None:
                result.agent = self.name
            if result.operation is None:
                result.operation = operation

            # Добавляем elapsed_s в metadata
            meta = getattr(result, "metadata", {}) or {}
            meta.setdefault("elapsed_s", time.time() - start)
            result.metadata = meta

            return result
        except Exception as exc:
            LOG.exception("Агент %s: ошибка при выполнении операции %s", self.name, operation)
            return AgentResult.error("operation_execution", f"Операция '{operation}' завершилась с ошибкой: {exc}")

    # -------------------------
    # Утилиты
    # -------------------------

    def describe(self) -> Dict[str, Any]:
        """Возвращает краткое описание агента."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "operations": list(self._operations.keys()),
            "config_keys": list(self.config.keys()),
        }