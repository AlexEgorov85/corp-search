# src/graph/nodes/executor.py
"""
Узел выполнения операций.
Цель: выполнить вызов инструмента для текущего этапа.
Логирование:
  - вход в шаг и этап
  - вызов агента и операции
  - успешный/ошибочный результат
  - завершение этапа
"""
from __future__ import annotations
import logging
from typing import Any, Dict
from src.model.agent_result import AgentResult
from src.model.context.context import GraphContext

LOG = logging.getLogger(__name__)

def executor_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    if agent_registry is None:
        raise ValueError("executor_node: agent_registry is required")
    ctx = GraphContext.from_state_dict(state)
    step_id = ctx.get_current_step_id()
    if not step_id:
        LOG.warning("⚠️ executor_node: нет текущего шага")
        return ctx.to_dict()

    tool_call = ctx.get_current_tool_call(step_id)
    if not tool_call:
        LOG.info("ℹ️ executor_node: нет вызова для шага %s", step_id)
        return ctx.to_dict()

    current_stage = ctx.get_current_stage(step_id)
    LOG.info("⚙️ Выполнение этапа '%s' для шага %s", current_stage, step_id)
    LOG.info("🚀 Запуск %s.%s", tool_call["agent"], tool_call["operation"])
    LOG.debug("📦 Параметры: %s", tool_call["params"])

    try:
        agent = agent_registry.instantiate_agent(tool_call["agent"])
        result = agent.execute_operation(
            tool_call["operation"],
            tool_call["params"],
            context=ctx.to_dict()
        )
        if isinstance(result, AgentResult):
            ctx.record_agent_call(step_id, result)
            if result.status == "ok":
                if current_stage == "validation":
                    ctx.record_validation_result(step_id, result.output)
                else:
                    ctx.record_step_result(step_id, result.output)
                ctx.mark_stage_completed(step_id, current_stage)
                LOG.info("✅ Этап '%s' успешно завершён для шага %s", current_stage, step_id)
                LOG.debug("📤 Результат: %s", result.output)
            else:
                LOG.error("❌ Операция завершилась с ошибкой: %s", result.error)
        else:
            LOG.error("❌ Агент вернул не AgentResult: %s", type(result))
    except Exception as e:
        LOG.exception("💥 Ошибка выполнения в executor_node: %s", e)

    return ctx.to_dict()