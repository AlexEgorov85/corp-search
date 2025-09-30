# src/graph/nodes/reasoner.py
# coding: utf-8
"""
reasoner_node — узел, который вызывает ReasonerAgent.step и применяет его результат.

Изменения:
- Использует GraphContext.from_state_dict() для входного state
- Гарантирует, что при наличии 'decision' в AgentResult.structured current_call записывается через ctx.set_current_call()
- Обновления шага (updated_step) применяются через StepState поля
- Ошибки reasoner'а сохраняются в step.error и в history
- Возвращает ctx.to_legacy_state()
"""
from __future__ import annotations
from typing import Dict, Any, Optional

import logging

from src.graph.context import GraphContext
from src.services.results.agent_result import AgentResult

LOG = logging.getLogger(__name__)


def reasoner_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)

    # Определяем текущий шаг (в порядке: execution.current_subquestion_id -> first step in steps)
    step_id = ctx.execution.current_subquestion_id or (next(iter(ctx.execution.steps.keys())) if ctx.execution.steps else None)
    if not step_id:
        ctx.append_history({"type": "reasoner_no_step"})
        return ctx.to_legacy_state()

    step = ctx.ensure_step(step_id)

    # Инстанцируем ReasonerAgent (control agent)
    if agent_registry is None:
        ctx.append_history({"type": "reasoner_no_registry"})
        return ctx.to_legacy_state()

    try:
        reasoner = agent_registry.instantiate_agent("ReasonerAgent", control=True)
    except Exception as e:
        ctx.append_history({"type": "reasoner_instantiate_failed", "error": str(e)})
        return ctx.to_legacy_state()

    # Подготовка параметров в стиле legacy (чтобы mock-тесты продолжали работать)
    params = {
        "step": step.model_dump() if hasattr(step, "model_dump") else dict(step),
        "execution_state": ctx.execution.model_dump() if hasattr(ctx.execution, "model_dump") else dict(ctx.execution)
    }

    try:
        res = reasoner.execute_operation("step", params, context=None)
    except Exception as e:
        ctx.set_step_error(step_id, f"reasoner.execute threw: {e}", stage="execute")
        return ctx.to_legacy_state()

    # Обработка результата
    if not isinstance(res, AgentResult):
        ctx.append_history({"type": "reasoner_invalid_result_type", "value": str(type(res))})
        return ctx.to_legacy_state()

    if res.is_error():
        ctx.set_step_error(step_id, res.content or res.error or res.message or "Reasoner error", stage="agent")
        return ctx.to_legacy_state()

    structured = res.structured or {}
    # updated_step — применяем безопасно
    updated_step_raw = structured.get("updated_step") or {}
    if updated_step_raw:
        for k, v in dict(updated_step_raw).items():
            try:
                setattr(step, k, v)
            except Exception:
                ctx.memory.setdefault("reasoner_updated_fields", {}).setdefault(step_id, {})[k] = v
        ctx.ensure_step(step_id)

    decision = structured.get("decision")
    if decision:
        # Записываем current_call через API (гарантия консистентности)
        ctx.set_current_call(decision, step_id=step_id)
    else:
        ctx.clear_current_call()

    # Если reasoner сразу возвращает final_answer
    if isinstance(decision, dict) and decision.get("action") == "final_answer":
        ctx.final_answer = decision.get("answer")

    if "next_stage" in structured:
        ctx.append_history({"type": "reasoner_next_stage", "next_stage": structured.get("next_stage")})

    return ctx.to_legacy_state()
