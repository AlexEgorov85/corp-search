# src/common/agent_registry.py
# coding: utf-8
"""
AgentRegistry — менеджер реестра агентов (tools и control agents) для проекта.

Назначение:
- Поддерживать единый программный интерфейс доступа к TOOL_REGISTRY и CONTROL_REGISTRY.
- Валидировать записи агентом по единой схеме AgentEntry.
- Импортировать (smoke-test) и инстанцировать реализации (implementation: "module:Attr").
- Предоставлять утилиты для поиска агентов/операций.

Как использовать (примеры):
>>> from src.common.agent_registry import AgentRegistry
>>> ar = AgentRegistry()  # автоматически подхватит src.common.tool_registry (и control_registry, если есть)
>>> ar.validate_all()     # проверит структуру реестров
>>> entry = ar.get_agent_entry("BooksLibraryAgent")
>>> op = ar.get_operation("BooksLibraryAgent", "get_last_book")
>>> cls_or_fn = ar.get_implementation("BooksLibraryAgent")  # import и вернуть объект
>>> instance = ar.instantiate_agent("BooksLibraryAgent")   # создать экземпляр (если это класс)

ПЕРЕЧЕНЬ ОСНОВНЫХ МЕТОДОВ:
- list_agents(control=False) -> list[str]
- get_agent_entry(name, control=False) -> dict
- get_operation(agent_name, op_name, control=False) -> dict
- find_agents_by_operation(op_name, control=False) -> list[(agent_name, entry)]
- get_implementation(agent_name, control=False) -> object (class or function)
- instantiate_agent(agent_name, control=False) -> instance or callable wrapper
- validate_all() -> raises ValueError если что-то не так
- validate_implementations() -> проверяет импортируемость implementation для всех записей

Принципы реализации:
- "operations" — единственный source-of-truth, каждому operation обязателен флаг "kind" и "description".
- "implementation" — строка формата "module.path:Attr". Обязательное поле.
- В конструкторе класса-агента пытаемся передать config (если конструктор принимает именованный аргумент 'config'
  или **kwargs). В противном случае пробуем без аргументов.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import types
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# Попытка импортировать реестры по умолчанию (если они есть в проекте).
def _load_registry_module(module_path: str) -> Optional[Dict[str, Any]]:
    """
    Попытаться импортировать модуль-реестр и вернуть TOOL_REGISTRY/CONTROL_REGISTRY,
    если модуль доступен и содержит соответствующее имя.
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        LOG.debug("Модуль реестра %s не найден.", module_path)
        return None

    # Возможные имена переменных в модуле
    for varname in ("TOOL_REGISTRY", "CONTROL_REGISTRY"):
        if hasattr(mod, varname):
            val = getattr(mod, varname)
            if isinstance(val, dict):
                LOG.debug("Загружен реестр %s из %s", varname, module_path)
                return val
    # fallback: вернуть весь модуль как mapping, но мы ожидаем словарь
    LOG.debug("Модуль %s импортирован, но не содержит TOOL_REGISTRY/CONTROL_REGISTRY.", module_path)
    return None


class AgentRegistry:
    """
    Управляет набором зарегистрированных агентов.

    Конструктор:
        AgentRegistry(tool_registry=None, control_registry=None, validate_on_init=False)

    Если tool_registry/control_registry отсутствуют, пытается импортировать:
     - src.common.tool_registry.TOOL_REGISTRY
     - src.common.control_registry.CONTROL_REGISTRY

    Параметры:
      - validate_on_init: если True — выполнить validate_all() при создании (может бросить исключение).
    """

    # Ключевые поля и требования к структуре AgentEntry
    _REQUIRED_TOP_LEVEL = {"name", "title", "description", "implementation", "operations"}

    def __init__(
        self,
        tool_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        control_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        validate_on_init: bool = False,
    ) -> None:
        # Загружаем реестры: либо из аргументов, либо пробуем импортировать модули
        self.tool_registry = tool_registry or _load_registry_module("src.common.tool_registry") or {}
        self.control_registry = control_registry or _load_registry_module("src.common.control_registry") or {}

        # Кеш импортированных реализаций (module:attr -> object)
        self._impl_cache: Dict[str, Any] = {}

        if validate_on_init:
            self.validate_all()

    # -----------------------------
    # Базовые методы доступа
    # -----------------------------
    def list_agents(self, control: bool = False) -> List[str]:
        """Вернуть список имён агентов (ключи). По умолчанию — tool_registry."""
        reg = self.control_registry if control else self.tool_registry
        return list(reg.keys())

    def get_agent_entry(self, name: str, control: bool = False) -> Dict[str, Any]:
        """Получить запись агента по имени. Бросает KeyError если не найден."""
        reg = self.control_registry if control else self.tool_registry
        if name not in reg:
            raise KeyError(f"Agent '{name}' not found in {'control' if control else 'tool'} registry.")
        return reg[name]

    def get_operation(self, agent_name: str, op_name: str, control: bool = False) -> Dict[str, Any]:
        """Получить описание операции указанного агента."""
        entry = self.get_agent_entry(agent_name, control=control)
        ops = entry.get("operations") or {}
        if op_name not in ops:
            raise KeyError(f"Operation '{op_name}' not found in agent '{agent_name}'.")
        return ops[op_name]

    def find_agents_by_operation(self, op_name: str, control: bool = False) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Найти агентов, которые реализуют операцию с именем op_name.
        Возвращает список (agent_name, entry).
        """
        reg = self.control_registry if control else self.tool_registry
        found = []
        for name, entry in reg.items():
            ops = entry.get("operations") or {}
            if op_name in ops:
                found.append((name, entry))
        return found

    # -----------------------------
    # Импорт и инстанцирование реализаций
    # -----------------------------
    @staticmethod
    def _parse_implementation(impl: str) -> Tuple[str, str]:
        """
        Разбить implementation 'module.path:Attr' -> (module_path, attr_name).
        Бросает ValueError при неверном формате.
        """
        if not isinstance(impl, str) or ":" not in impl:
            raise ValueError(f"Invalid implementation format: {impl!r}. Expected 'module.path:Attr'.")
        module_path, attr = impl.split(":", 1)
        module_path = module_path.strip()
        attr = attr.strip()
        if not module_path or not attr:
            raise ValueError(f"Invalid implementation format: {impl!r}. Expected 'module.path:Attr'.")
        return module_path, attr

    def _import_implementation(self, implementation: str) -> Any:
        """
        Импортировать объект по строке implementation и кешировать результат.
        Возвращает объект (класс или функцию).
        """
        if implementation in self._impl_cache:
            return self._impl_cache[implementation]

        module_path, attr = self._parse_implementation(implementation)
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            LOG.exception("Ошибка импорта модуля %s: %s", module_path, e)
            raise

        if not hasattr(module, attr):
            # Возможно attr это вложенный атрибут через точку, попробуем по цепочке
            if "." in attr:
                cur = module
                for part in attr.split("."):
                    if hasattr(cur, part):
                        cur = getattr(cur, part)
                    else:
                        LOG.error("Attribute %s not found in module path %s", attr, module_path)
                        raise AttributeError(f"Attribute {attr} not found in module {module_path}")
                impl_obj = cur
            else:
                LOG.error("Attribute %s not found in module %s", attr, module_path)
                raise AttributeError(f"Attribute {attr} not found in module {module_path}")
        else:
            impl_obj = getattr(module, attr)

        self._impl_cache[implementation] = impl_obj
        return impl_obj

    def get_implementation(self, agent_name: str, control: bool = False) -> Any:
        """
        Получить импортируемый объект (класс или функцию) для агента.
        Бросает KeyError/ValueError/AttributeError при проблемах.
        """
        entry = self.get_agent_entry(agent_name, control=control)
        impl = entry.get("implementation")
        if not impl:
            raise ValueError(f"Agent '{agent_name}' has no 'implementation' field.")
        return self._import_implementation(impl)

    def instantiate_agent(self, agent_name: str, control: bool = False, *, override_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Инстанцировать агента:
        - Если implementation — класс, пытаемся создать экземпляр.
          Пытаемся передать config (entry['config']) в конструктор если он принимает именованный аргумент 'config'
          или **kwargs. Если не принимает — вызываем без аргументов.
        - Если implementation — callable (функция), возвращаем сам объект (функцию), чтобы Executor мог её вызвать.

        Возвращаем либо экземпляр класса, либо callable.
        """
        impl_obj = self.get_implementation(agent_name, control=control)
        entry = self.get_agent_entry(agent_name, control=control)
        config = override_config if override_config is not None else entry.get("config", {}) or {}

        # Если это класс (type) — инстанцируем
        if inspect.isclass(impl_obj):
            ctor_sig = inspect.signature(impl_obj)
            # Проверяем, принимает ли конструктор именованный аргумент 'config' или **kwargs
            try:
                parameters = ctor_sig.parameters
            except Exception:
                parameters = {}

            accepts_config = False
            accepts_kwargs = False
            for p in parameters.values():
                if p.kind in (p.VAR_KEYWORD,):
                    accepts_kwargs = True
                if p.name == "config":
                    accepts_config = True

            try:
                if accepts_config:
                    instance = impl_obj(config=config)
                elif accepts_kwargs:
                    instance = impl_obj(**(config or {}))
                elif len(parameters) == 0 or all(p.default is not inspect.Parameter.empty or p.kind in (p.VAR_POSITIONAL,) for p in parameters.values()):
                    # Пустой конструктор или все параметры имеют дефолты
                    instance = impl_obj()
                else:
                    # Конструктор требует позиционные аргументы — попробуем создать без аргументов и, если не получится, падём
                    instance = impl_obj()
            except Exception as e:
                LOG.exception("Не удалось инстанцировать агент %s: %s", agent_name, e)
                raise

            return instance

        # Если это callable (функция) — возвращаем callable (агент-функция)
        if callable(impl_obj):
            return impl_obj

        # Необрабатываемый тип
        raise TypeError(f"Implementation for agent '{agent_name}' is not a class or callable: {type(impl_obj)}")

    # -----------------------------
    # Валидация структуры
    # -----------------------------
    def _validate_agent_entry(self, name: str, entry: Dict[str, Any]) -> None:
        """
        Проверка базовой структуры AgentEntry для одного агента.
        Бросает ValueError при несоответствии.
        """
        missing = self._REQUIRED_TOP_LEVEL - set(entry.keys())
        if missing:
            raise ValueError(f"Agent '{name}': missing required top-level fields: {sorted(missing)}")

        impl = entry.get("implementation")
        if not isinstance(impl, str) or ":" not in impl:
            raise ValueError(f"Agent '{name}': 'implementation' must be a string 'module:Attr'.")

        operations = entry.get("operations")
        if not isinstance(operations, dict) or not operations:
            raise ValueError(f"Agent '{name}': 'operations' must be a non-empty dict.")

        for op_name, op in operations.items():
            if not isinstance(op, dict):
                raise ValueError(f"Agent '{name}' operation '{op_name}' must be a dict.")
            kind = op.get("kind")
            if kind not in ("direct", "composed"):
                raise ValueError(f"Agent '{name}' operation '{op_name}': invalid kind '{kind}'. Expected 'direct' or 'composed'.")
            desc = op.get("description")
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"Agent '{name}' operation '{op_name}': 'description' is required and must be a non-empty string.")
            # params/outputs can be 'freeform' or structures; no strict schema here

            # Дополнительная проверка для composed
            if kind == "composed":
                # Если указан allowed_tables — должен быть список строк
                allowed = op.get("allowed_tables")
                if allowed is not None and (not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed)):
                    raise ValueError(f"Agent '{name}' operation '{op_name}': 'allowed_tables' must be a list of strings if provided.")

    def validate_all(self) -> None:
        """
        Пройтись по всем записям в tool_registry и control_registry и проверить структуру.
        Бросает ValueError при первой проблеме.
        """
        # tools
        for name, entry in self.tool_registry.items():
            self._validate_agent_entry(name, entry)
        # control
        for name, entry in self.control_registry.items():
            self._validate_agent_entry(name, entry)
        LOG.info("AgentRegistry: validation passed for %d tools and %d control agents.", len(self.tool_registry), len(self.control_registry))

    def validate_implementations(self, *, control: bool = False, fail_on_error: bool = False) -> List[Tuple[str, Exception]]:
        """
        Попытаться импортировать все implementation и вернуть список ошибок (если есть).
        Если fail_on_error=True — при первой ошибке бросаем исключение.
        Возвращаем список (agent_name, Exception) для всех провалившихся импортов.
        """
        reg = self.control_registry if control else self.tool_registry
        errors = []
        for name, entry in reg.items():
            impl = entry.get("implementation")
            try:
                self._import_implementation(impl)
            except Exception as e:
                LOG.exception("Validate implementation failed for %s: %s", name, e)
                errors.append((name, e))
                if fail_on_error:
                    raise
        return errors

    # -----------------------------
    # Утилиты поиска
    # -----------------------------
    def find_agents(self, predicate) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Универсальная функция поиска: predicate(name, entry) -> bool.
        Возвращает список (name, entry) из tool_registry подходящих под предикат.
        """
        results = []
        for name, entry in self.tool_registry.items():
            try:
                if predicate(name, entry):
                    results.append((name, entry))
            except Exception:
                LOG.exception("Predicate failed on agent %s", name)
        return results

    # -----------------------------
    # Полезные "удобные" методы
    # -----------------------------
    def get_agent_operations(self, agent_name: str, control: bool = False) -> List[str]:
        """Вернуть список имён операций агента."""
        entry = self.get_agent_entry(agent_name, control=control)
        return list((entry.get("operations") or {}).keys())

    def dump_registry_summary(self) -> Dict[str, Any]:
        """Вернуть краткую сводку реестра: list agents и operations count."""
        def summarize(reg: Dict[str, Dict[str, Any]]):
            return {name: {"operations": len(entry.get("operations") or {}), "title": entry.get("title")} for name, entry in reg.items()}

        return {"tools": summarize(self.tool_registry), "control": summarize(self.control_registry)}


# -----------------------------
# Простой демонстрационный main (не выполняется при импорте модуля)
# -----------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    ar = AgentRegistry(validate_on_init=False)
    try:
        ar.validate_all()
    except Exception as e:
        LOG.error("Validation failed: %s", e)

    # Попробуем импортировать реализации и показать сводку
    errs = ar.validate_implementations(fail_on_error=False)
    if errs:
        LOG.warning("Found %d implementation import errors.", len(errs))
    else:
        LOG.info("All implementations imported successfully (smoke-test).")

    print("Registry summary:")
    print(json.dumps(ar.dump_registry_summary(), indent=2, ensure_ascii=False))
