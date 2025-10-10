# src/graph/nodes/executor.py
"""
Узел выполнения инструментов (Executor Node).

Основная задача:
  - Получить решение от Reasoner (`selected_tool`).
  - При необходимости — **автоматически подставить параметры** из контекста,
    если операция предназначена для работы с контекстом (а не с ручными параметрами).
  - Вызвать агент с этими параметрами.
  - Сохранить результат в контекст через `record_tool_execution_result`.

Особенности:
  - Для операций вроде `validate_result`, `synthesize` параметры **не передаются вручную** из Reasoner.
    Вместо этого Executor **инжектит** нужные данные из `GraphContext`.
  - Это позволяет Reasoner оставаться чистым: он только выбирает инструмент, не заботясь о деталях параметров.
  - Поддержка расширения: легко добавить новые "контекстные" операции.

Примеры:
  - Reasoner выбирает:
      selected_tool = {"agent": "ResultValidatorAgent", "operation": "validate_result"}
    Executor автоматически подставляет:
      params = {
          "subquestion_text": ctx.get_subquestion_text(step_id),
          "raw_output": ctx.get_step_result(step_id)
      }

  - Reasoner выбирает:
      selected_tool = {"agent": "BooksLibraryAgent", "operation": "list_books", "params": {"author": "Пушкин"}}
    Executor оставляет params как есть.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from src.model.agent_result import AgentResult
from src.model.context.base import (
    append_history_event,
    get_current_step_id,
    get_execution_step,
)
from src.model.context.context import GraphContext


LOG = logging.getLogger(__name__)


def _inject_contextual_params(
    tool_call: Dict[str, Any],
    ctx: GraphContext,
    step_id: str
) -> Dict[str, Any]:
    """
    Автоматически подставляет параметры для операций, которые работают с контекстом выполнения.

    Эта функция проверяет имя агента и операции и, если это "контекстная" операция,
    формирует параметры на основе текущего состояния шага и плана.

    Поддерживаемые операции:
      - ResultValidatorAgent.validate_result
      - DataAnalysisAgent.analyze

    Args:
        tool_call (dict): Решение от Reasoner, содержащее "agent", "operation", и опционально "params".
        ctx (GraphContext): Текущий контекст выполнения графа.
        step_id (str): ID текущего шага.

    Returns:
        Dict[str, Any]: Готовые параметры для вызова операции.
    """
    agent_name = tool_call["agent"]
    operation_name = tool_call["operation"]
    base_params = tool_call.get("params", {})

    # === 1. ResultValidatorAgent.validate_result ===
    if agent_name == "ResultValidatorAgent" and operation_name == "validate_result":
        return {
            "subquestion_text": ctx.get_subquestion_text(step_id),
            "raw_output": ctx.get_step_result(step_id)
        }

    # === 2. DataAnalysisAgent (если потребуется) ===
    if agent_name == "DataAnalysisAgent" and operation_name == "analyze":
        return {
            "subquestion_text": ctx.get_subquestion_text(step_id),
            "raw_output": ctx.get_step_result(step_id)
        }

    # === По умолчанию: возвращаем параметры как есть ===
    return base_params


def _execute_tool_call(
    ctx: GraphContext,
    tool_call: Dict[str, Any],
    agent_registry,
    step_id: str
) -> Any:
    """
    Выполняет вызов инструмента через агентский реестр.

    Args:
        ctx (GraphContext): Контекст выполнения.
        tool_call (dict): Описание вызова инструмента.
        agent_registry: Реестр агентов.
        step_id (str): ID шага (для логирования и инжекта параметров).

    Returns:
        AgentResult: Результат выполнения операции.

    Raises:
        RuntimeError: При ошибках инициализации или выполнения.
    """
    agent_name = tool_call["agent"]
    operation = tool_call["operation"]

    # Автоматическая подстановка параметров для контекстных операций
    params = _inject_contextual_params(tool_call, ctx, step_id)

    LOG.info("⚙️  Executor: запускаем %s.%s", agent_name, operation)
    agent = agent_registry.instantiate_agent(agent_name)

    if not agent:
        raise RuntimeError(f"Агент '{agent_name}' не найден в реестре")

    res = agent.execute_operation(operation, params=params, context=ctx.to_dict())

    if not isinstance(res, AgentResult):
        raise RuntimeError(f"Агент '{agent_name}' вернул не AgentResult: {type(res)}")

    if res.status != "ok":
        raise RuntimeError(f"Операция завершилась с ошибкой: {res.error or res.content}")

    return res


def executor_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    """
    Основной узел выполнения инструментов.

    Логика:
      1. Получить текущий шаг из контекста.
      2. Получить решение от Reasoner (`selected_tool`).
      3. Если решение есть — выполнить инструмент.
      4. Сохранить результат (успех или ошибка) в контекст.
      5. Залогировать событие.

    Args:
        state (Dict[str, Any]): Состояние графа (GraphContext в виде dict).
        agent_registry: Реестр агентов (обязателен).

    Returns:
        Dict[str, Any]: Обновлённое состояние графа.
    """
    if agent_registry is None:
        raise ValueError("executor_node: agent_registry is required")

    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)
    step_id = get_current_step_id(ctx)

    if not step_id:
        LOG.warning("⚠️ Executor: нет текущего шага")
        append_history_event(ctx, {"type": "executor_no_step"})
        return ctx.to_dict()

    step = get_execution_step(ctx, step_id)
    if not step:
        LOG.warning("⚠️ Executor: шаг %s не найден", step_id)
        return ctx.to_dict()

    # === Получаем вызов инструмента через API контекста ===
    tool_call = ctx.get_tool_call_for_executor(step_id)
    if not tool_call:
        LOG.info("ℹ️  Executor: нет selected_tool для шага %s", step_id)
        return ctx.to_dict()

    try:
        result = _execute_tool_call(ctx, tool_call, agent_registry, step_id)
        # === Сохраняем результат через API контекста ===
        ctx.record_tool_execution_result(step_id, result)
        LOG.info("✅  Executor: успешно выполнил операцию для шага %s", step_id)
        append_history_event(ctx, {
            "type": "executor_success",
            "step_id": step_id,
            "agent": tool_call["agent"],
            "operation": tool_call["operation"]
        })
    except Exception as e:
        error_msg = str(e)
        LOG.exception("💥 Ошибка при выполнении инструмента для шага %s: %s", step_id, e)
        # === Сохраняем ошибку через API контекста ===
        ctx.record_tool_execution_result(step_id, result=None, error=error_msg)
        append_history_event(ctx, {
            "type": "executor_error",
            "step_id": step_id,
            "error": error_msg
        })

    return ctx.to_dict()