# src/graph/nodes/next_subquestion.py
# coding: utf-8
"""
next_subquestion_node — выбирает следующий подвопрос и устанавливает execution.current_subquestion_id.

Изменения:
- Использует GraphContext API.
- Читает план из ctx.memory['plan'] (если он там есть) или из legacy place ctx.to_legacy_state().get('plan').
- План ожидается как dict с ключом 'subquestions' — список элементов с 'id' и 'text' (как в PlannerAgent output).
- Устанавливает current_subquestion_id на первый незавершённый подвопрос.
- Если все подвопросы завершены — возвращает finished flag (в legacy state) и очищает текущий подвопрос.
"""
from __future__ import annotations
from typing import Dict, Any, Optional

import logging

from src.graph.context_model import GraphContext

LOG = logging.getLogger(__name__)


def _get_plan_from_ctx(ctx: GraphContext) -> Optional[Dict[str, Any]]:
    # Попробуем сначала взять план из memory (planner_node должен был туда положить)
    p = ctx.memory.get("plan")
    if p:
        return p
    # Иначе попробуем из legacy state
    legacy = ctx.to_legacy_state()
    return legacy.get("plan")


def next_subquestion_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)

    # Если уже финиш — ничего не делаем
    legacy = ctx.to_legacy_state()
    if legacy.get("finished"):
        ctx.append_history({"type": "next_subquestion_already_finished"})
        return ctx.to_legacy_state()

    plan = _get_plan_from_ctx(ctx)
    if not plan or not isinstance(plan, dict):
        ctx.append_history({"type": "next_subquestion_no_plan"})
        # mark finished to avoid infinite loops in graph
        legacy_out = ctx.to_legacy_state()
        legacy_out["finished"] = True
        return legacy_out

    subquestions = plan.get("subquestions") or []
    if not subquestions:
        ctx.append_history({"type": "next_subquestion_empty_subquestions"})
        legacy_out = ctx.to_legacy_state()
        legacy_out["finished"] = True
        return legacy_out

    # Найти первый подвопрос, у которого статус не finalized/failed/done
    for sq in subquestions:
        sid = sq.get("id")
        if not sid:
            continue
        step = ctx.execution.steps.get(sid)
        status = step.status if step is not None else None
        if status not in ("finalized", "failed", "done", "executed"):
            # Выбираем этот подвопрос
            ctx.execution.current_subquestion_id = sid
            ctx.append_history({"type": "next_subquestion_selected", "step_id": sid})
            return ctx.to_legacy_state()

    # Если дошли сюда — все подвопросы завершены
    ctx.execution.current_subquestion_id = None
    ctx.append_history({"type": "next_subquestion_all_done"})
    out = ctx.to_legacy_state()
    out["finished"] = True
    return out
