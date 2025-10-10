# src/graph/nodes/next_subquestion.py
"""
next_subquestion_node — узел выбора следующего подвопроса для обработки.
Цель:
- Определить, какой из подвопросов плана должен быть обработан следующим.
- Установить его как текущий шаг в контексте выполнения.
- Завершить граф, если все шаги завершены.

Совместимость:
- Работает с GraphContext из src/model/context/context.py
- Использует только публичные методы контекста
- Поддерживает только Plan (Pydantic), не dict
"""

from __future__ import annotations
from typing import Any, Dict, List
import logging

from src.model.context.base import (
    append_history_event,
    ensure_execution_step,
    get_plan,
    set_current_step_id,
    is_step_completed,
)
from src.model.context.context import GraphContext

LOG = logging.getLogger(__name__)


def next_subquestion_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    """
    Основная функция узла выбора следующего подвопроса.
    Args:
        state (Dict[str, Any]): Входное состояние графа.
        agent_registry: Не используется (передаётся для совместимости).
    Returns:
        Dict[str, Any]: Обновлённое состояние графа.
    """
    # Преобразуем входной state в GraphContext
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)

    # Получаем план через API
    plan = get_plan(ctx)
    if not plan or not plan.subquestions:
        LOG.warning("⚠️  next_subquestion: план отсутствует или пуст")
        append_history_event(ctx, {"type": "next_subquestion_no_plan"})
        return _finish_graph(ctx)

    # Ищем первый незавершённый шаг, зависимости которого выполнены
    for sq in plan.subquestions:
        if not _are_dependencies_satisfied(ctx, sq.depends_on):
            continue
        if is_step_completed(ctx, sq.id):
            continue

        # 🎯 Выбираем этот шаг
        ensure_execution_step(ctx, sq.id)  # гарантирует создание шага
        set_current_step_id(ctx, sq.id)
        LOG.info(f"➡️  next_subquestion: выбран шаг {sq.id}")
        append_history_event(ctx, {"type": "next_subquestion_selected", "step_id": sq.id})
        return ctx.to_dict()

    # 🏁 Все шаги завершены
    LOG.info("🏁  next_subquestion: все шаги завершены")
    set_current_step_id(ctx, None)
    return _finish_graph(ctx)


def _are_dependencies_satisfied(ctx: GraphContext, depends_on: List[str]) -> bool:
    """Проверяет, что все зависимости завершены."""
    for dep_id in depends_on:
        if not is_step_completed(ctx, dep_id):
            return False
    return True


def _finish_graph(ctx: GraphContext) -> Dict[str, Any]:
    """Завершает граф и возвращает состояние с finished=True."""
    out = ctx.to_dict()
    out["finished"] = True
    append_history_event(ctx, {"type": "next_subquestion_all_done"})
    return out