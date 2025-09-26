# run_planner.py
# coding: utf-8
"""
Утилита для локального запуска PlannerAgent.
Изменения:
- Добавлена обертка для LLM, соответствующая ожидаемому интерфейсу PlannerAgent
- Обновлен способ передачи LLM в конфигурацию
- Улучшена обработка результатов
- Добавлена поддержка текущей структуры PlannerAgent
"""

from __future__ import annotations
import inspect
import json
import logging
import sys
from typing import Any, Dict, List, Callable
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("run_planner")

# Импортируем необходимые компоненты
try:
    from src.agents.PlannerAgent.core import PlannerAgent
except Exception as e:
    LOG.exception("Не удалось импортировать PlannerAgent. Проверьте пути/модули.")
    raise

# Попытка импортировать фабрику LLM
create_llm = None
try:
    from src.services.llm_service.factory import create_llm as _create_llm
    create_llm = _create_llm
    LOG.info("Найдена LLM factory: src.services.llm_service.factory.create_llm")
except Exception as e:
    LOG.info("LLM factory не найдена. Попробуем использовать config llm_callable (если есть).")
    LOG.debug("Исключение при импорте create_llm: %s", str(e))

# Попытка получить TOOL_REGISTRY из настроек
tool_registry_snapshot = None
try:
    import src.common.tool_registry as settings_module
    tool_registry_snapshot = getattr(settings_module, "TOOL_REGISTRY", None)
    if tool_registry_snapshot:
        LOG.info("TOOL_REGISTRY найден в src.common.settings — будет использован snapshot.")
    else:
        LOG.info("TOOL_REGISTRY не найден в src.common.settings — продолжим без snapshot.")
except Exception as e:
    LOG.info("Не удалось импортировать src.common.settings или TOOL_REGISTRY отсутствует — продолжим без snapshot.")
    LOG.debug("Исключение при импорте settings: %s", str(e))

def _safe_create_llm(factory_callable, profile_name: str = "default"):
    """
    Безопасный вызов create_llm: поддерживаем разные сигнатуры фабрики:
      - create_llm(profile="default")
      - create_llm("default")
      - create_llm()
    Возвращает объект llm или пробрасываем исключение.
    """
    if factory_callable is None:
        raise RuntimeError("_safe_create_llm: фабрика не передана")
    
    # Попробуем проанализировать сигнатуру
    try:
        sig = inspect.signature(factory_callable)
        params = sig.parameters
    except Exception:
        # Если сигнатура не доступна — пробуем вызвать простыми вариантами
        try:
            return factory_callable(profile_name)
        except TypeError:
            try:
                return factory_callable()
            except Exception:
                LOG.exception("Не удалось создать LLM через фабрику")
                return None
    
    # Если есть named param 'profile' или 'profile_name' — передаём по имени
    if "profile" in params:
        try:
            return factory_callable(profile=profile_name)
        except Exception:
            pass
    if "profile_name" in params:
        try:
            return factory_callable(profile_name=profile_name)
        except Exception:
            pass
    
    # Если фабрика принимает один позиционный параметр — пробуем вызвать с profile_name
    positional_params = [p for p in params.values() 
                         if p.kind in (inspect.Parameter.POSITIONAL_ONLY, 
                                       inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional_params) == 1:
        try:
            return factory_callable(profile_name)
        except Exception:
            pass
    
    # Наконец, пробуем без аргументов
    try:
        return factory_callable()
    except Exception:
        LOG.exception("Не удалось создать LLM через фабрику")
        return None

def create_llm_wrapper(llm: Any) -> Callable[[List[Dict[str, str]]], str]:
    """
    Создает обертку для LLM, которая соответствует интерфейсу, ожидаемому PlannerAgent.
    PlannerAgent ожидает функцию, которая принимает messages: List[Dict[str, str]] и возвращает строку.
    """
    if llm is None:
        return None
    
    def llm_wrapper(messages: List[Dict[str, str]]) -> str:
        """
        Обертка, которая приводит вызов к ожидаемому формату.
        """
        # Извлекаем только content из сообщений
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Пытаемся вызвать LLM различными способами
        if hasattr(llm, "generate"):
            try:
                return llm.generate(prompt)
            except Exception:
                pass
        
        # Попробуем вызов через invoke
        if hasattr(llm, "invoke"):
            try:
                return llm.invoke(prompt)
            except Exception:
                pass
        
        # Попробуем вызов через predict
        if hasattr(llm, "predict"):
            try:
                return llm.predict(prompt)
            except Exception:
                pass
        
        # Попробуем вызов через __call__
        if callable(llm):
            try:
                return llm(prompt)
            except Exception:
                pass
        
        # Если ничего не сработало, возвращаем заглушку
        LOG.error("Не удалось вызвать LLM для генерации ответа")
        return "Ошибка генерации ответа"
    
    return llm_wrapper

def main() -> int:
    """Основная функция для запуска PlannerAgent."""
    # Описание агента
    planner_descriptor: Dict[str, Any] = {
        "name": "PlannerAgent",
        "title": "PlannerAgent",
        "description": "Агент, формирующий план выполнения запроса.",
        "implementation": "src.agents.PlannerAgent.core:PlannerAgent",
        "operations": {
            "plan": {"description": "generate plan", "kind": "direct"},
            "validate_plan": {"description": "validate plan", "kind": "direct"},
        },
    }
    
    # Конфигурация
    cfg: Dict[str, Any] = {}
    
    # Создаем и настраиваем LLM
    llm = None
    llm_callable = None
    
    if create_llm is not None:
        LOG.info("Попытка создать LLM через фабрику...")
        try:
            # Создаем LLM через безопасную функцию
            llm = _safe_create_llm(create_llm, profile_name="default")
            
            if llm is not None:
                # Создаем обертку для LLM
                llm_callable = create_llm_wrapper(llm)
                if llm_callable:
                    cfg["llm_callable"] = llm_callable
                    LOG.info("LLM обертка создана и добавлена в конфигурацию")
                else:
                    LOG.error("Не удалось создать обертку для LLM")
            else:
                LOG.warning("LLM создана, но является None")
        except Exception as e:
            LOG.exception("Не удалось создать LLM через factory")
    
    # Добавляем snapshot инструментов, если он доступен
    if tool_registry_snapshot:
        try:
            short_snapshot = {}
            for k, v in tool_registry_snapshot.items():
                if not isinstance(v, dict):
                    short_snapshot[k] = {}
                    continue
                short_snapshot[k] = {
                    "title": v.get("title", ""),
                    "description": v.get("description", ""),  # <-- Добавлено описание
                    "operations": v.get("operations", {})
                }
            cfg["tool_registry_snapshot"] = short_snapshot
            LOG.info("Добавлен tool_registry_snapshot в конфигурацию")
        except Exception as e:
            LOG.exception("Ошибка при обработке tool_registry_snapshot")
    
    # Создаем PlannerAgent
    try:
        planner = PlannerAgent(planner_descriptor, config=cfg)
        LOG.info("PlannerAgent создан")
    except Exception as e:
        LOG.exception("Не удалось создать PlannerAgent")
        return 1
    
    # Инициализируем агента
    try:
        planner.initialize()
        #LOG.info("PlannerAgent успешно инициализирован")
    except Exception as e:
        LOG.exception("Ошибка при инициализации PlannerAgent")
        # Попробуем продолжить без инициализации, но с предупреждением
        LOG.warning("Продолжаем с потенциально неработающим агентом")
    
    # Запускаем генерацию плана
    # question = "Найди книги Пушкина и укажи главного героя в последней из них?"
    question = "В каких актах содержится информация  о необоснованных отказах Банка?"
    LOG.info("Генерация плана для вопроса: %s", question)
    
    params = {"question": question}
    if "tool_registry_snapshot" in cfg:
        params["tool_registry_snapshot"] = cfg["tool_registry_snapshot"]
    
    try:
        res = planner.execute_operation("plan", params, {})
        LOG.info("Операция 'plan' выполнена")
    except Exception as e:
        LOG.exception("Ошибка при вызове planner.execute_operation('plan', ...)")
        print("\nPlanner run result:")
        print("Status : error")
        print("Message:", str(e))
        return 1
    
    # Обработка и вывод результата
    status = getattr(res, "status", "unknown")
    message = getattr(res, "content", getattr(res, "message", ""))
    structured = getattr(res, "structured", None)
    
    # Если результат - словарь
    if isinstance(res, dict):
        status = res.get("status", status)
        message = res.get("message", message) or res.get("content", "")
        structured = res.get("structured", res.get("content", res.get("plan", None)))
    
    print("\n" + "=" * 80)
    print("Planner run result:")
    print(f"Status : {status}")
    print(f"Message: {message}")
    
    # Вывод структурированных данных
    if structured:
        print("\nStructured (pretty JSON):")
        try:
            print(json.dumps(structured, ensure_ascii=False, indent=2))
        except Exception:
            print(str(structured))
    else:
        print("\nNo structured payload returned by agent.")
    
    print("=" * 80)
    
    # Закрытие агента
    try:
        planner.close()
        LOG.info("PlannerAgent успешно закрыт")
    except Exception:
        LOG.exception("Ошибка при закрытии PlannerAgent")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())