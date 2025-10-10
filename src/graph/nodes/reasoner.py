# src/graph/nodes/reasoner.py
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from src.model.agent_result import AgentResult
from src.model.context.base import (
    append_history_event,
    get_current_step_id,
    get_subquestion_text,
)
from src.model.context.context import GraphContext
from src.utils.utils import build_tool_registry_snapshot

LOG = logging.getLogger(__name__)


def reasoner_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    if agent_registry is None:
        raise ValueError("reasoner_node: agent_registry is required")

    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)

    step_id = get_current_step_id(ctx)
    if not step_id:
        LOG.warning("⚠️ reasoner_node: нет текущего шага")
        append_history_event(ctx, {"type": "reasoner_no_step"})
        return ctx.to_dict()

    # ✅ ИСПОЛЬЗУЕМ НОВЫЙ МЕТОД ensure_execution_step
    step = ctx.ensure_execution_step(step_id)

    try:
        subquestion_text = get_subquestion_text(ctx, step_id)
    except Exception as e:
        LOG.error("❌ reasoner_node: не найден текст подвопроса для шага %s: %s", step_id, e)
        return ctx.to_dict()

    # Собираем результаты других шагов
    step_outputs = ctx.get_all_step_results_for_reasoner()

    # Собираем snapshot инструментов
    tool_registry_snapshot = build_tool_registry_snapshot(agent_registry)

    # Состояние шага для промпта
    step_state = ctx.get_step_state_for_reasoner(step_id)

    params = {
        "subquestion": {"id": step_id, "text": subquestion_text},
        "step_state": step_state,
        "step_outputs": step_outputs,
        "tool_registry_snapshot": tool_registry_snapshot,
    }

    LOG.info("🧠 reasoner_node: вызов ReasonerAgent для шага %s", step_id)
    append_history_event(ctx, {"type": "reasoner_call", "step_id": step_id})

    try:
        reasoner_agent = agent_registry.instantiate_agent("ReasonerAgent", control=True)
        agent_result = reasoner_agent.execute_operation("decide_next_stage", params=params, context={})
    except Exception as e:
        LOG.exception("💥 Ошибка при вызове ReasonerAgent: %s", e)
        return ctx.to_dict()

    if not isinstance(agent_result, AgentResult) or agent_result.status != "ok":
        LOG.error("❌ ReasonerAgent вернул ошибку: %s", agent_result.error or agent_result.content)
        return ctx.to_dict()

    decision = agent_result.output
    next_stage = decision.get("next_stage")
    reason = decision.get("reason", "")
    LOG.info("✅ Reasoner принял решение: next_stage=%s, reason=%s", next_stage, reason)

    # ✅ Записываем решение через новый метод
    ctx.record_reasoner_decision(step_id, decision)

    # === finalize ===
    if next_stage == "finalize":
        ctx.mark_step_completed(step_id)
        LOG.info("🏁 Шаг %s завершён", step_id)
        return ctx.to_dict()

    # Любые другие этапы — передаём в executor
    return ctx.to_dict()