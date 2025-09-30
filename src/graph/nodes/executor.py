# src/graph/nodes/executor.py
# coding: utf-8
"""
executor_node — узел, который выполняет вызовы инструментов (tool agents).

Изменения:
- Перенёс всю логику работы с execution/current_call/steps на GraphContext API
- Сохранил поведение: инстанцируем инструмент из tool registry (control=False),
  вызываем execute_operation(operation, params, context) и в зависимости от AgentResult
  обновляем step.raw_output/final_result/status/error.
- Всегда очищаем current_call в конце обработки (успех/ошибка).
"""
from __future__ import annotations
from typing import Dict, Any

import logging

from src.graph.context import GraphContext
from src.services.results.agent_result import AgentResult

LOG = logging.getLogger(__name__)


def executor_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)

    current_call = ctx.execution.current_call
    if not current_call or not current_call.decision:
        ctx.append_history({"type": "executor_no_current_call"})
        return ctx.to_legacy_state()

    decision = current_call.decision or {}
    step_id = current_call.step_id or ctx.execution.current_subquestion_id

    action = (decision.get("action") or "").lower()
    if action != "call_tool":
        ctx.append_history({"type": "executor_skip_non_calltool", "action": action, "step_id": step_id})
        # leave decision intact (other nodes may handle it), but return state
        return ctx.to_legacy_state()

    tool_name = decision.get("tool")
    operation = decision.get("operation")
    params = decision.get("params", {}) or {}

    if not tool_name or not operation:
        ctx.set_step_error(step_id or "unknown", "Executor: invalid decision missing tool/operation", stage="validate_decision")
        ctx.clear_current_call()
        return ctx.to_legacy_state()

    # Инстанцируем инструмент (tool) из tool_registry
    try:
        tool = agent_registry.instantiate_agent(tool_name, control=False)
    except Exception as e:
        ctx.set_step_error(step_id, f"Executor: instantiate tool '{tool_name}' failed: {e}", stage="instantiate_tool")
        ctx.clear_current_call()
        return ctx.to_legacy_state()

    # Выполним операцию
    try:
        # Поддерживаем интерфейс execute_operation(operation, params, context)
        res = tool.execute_operation(operation, params, context={"step_id": step_id})
    except Exception as e:
        ctx.set_step_error(step_id, f"Executor: tool.execute threw: {e}", stage="execute_tool")
        ctx.clear_current_call()
        return ctx.to_legacy_state()

    if not isinstance(res, AgentResult):
        ctx.set_step_error(step_id, f"Executor: unexpected tool return type: {type(res)}", stage="execute_tool")
        ctx.clear_current_call()
        return ctx.to_legacy_state()

    if res.is_error():
        ctx.set_step_error(step_id, res.content or res.error or res.message or "tool returned error", stage="tool")
        ctx.clear_current_call()
        return ctx.to_legacy_state()

    # Успешный результат
    payload = res.structured if res.structured is not None else (res.content if hasattr(res, "content") else None)
    # Сохраняем как raw_output / final_result в шаге; семантика зависит от инструмента —
    # чтобы не ломать текущую логику, просто положим payload в final_result (как раньше большинство узлов делало)
    ctx.set_step_result(step_id or "unknown", payload)
    ctx.clear_current_call()
    ctx.append_history({"type": "executor_tool_success", "tool": tool_name, "operation": operation, "step_id": step_id, "rows_preview": str(payload)[:200]})
    return ctx.to_legacy_state()
