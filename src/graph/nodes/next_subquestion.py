# src/graph/nodes/next_subquestion.py
"""
Узел выбора следующего подвопроса.
Цель: определить, какой шаг выполнять следующим или завершить граф.
Логирование:
  - все шаги завершены → завершение
  - выбран следующий шаг → его ID и текст
"""
from __future__ import annotations
import logging
from typing import Any, Dict
from src.model.context.context import GraphContext

LOG = logging.getLogger(__name__)

def next_subquestion_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    ctx = GraphContext.from_state_dict(state)
    if ctx.all_steps_completed():
        LOG.info("✅ Все шаги завершены. Граф завершает работу.")
        ctx.set_current_step_id(None)
        return ctx.to_dict()

    next_step_id = ctx.select_next_step()
    if next_step_id:
        ctx.start_step(next_step_id)
        subq_text = ctx.get_subquestion_text(next_step_id)
        LOG.info("➡️ Выбран следующий шаг: %s", next_step_id)
        LOG.debug("📄 Текст подвопроса: %s", subq_text)
    else:
        LOG.warning("⚠️ Не найден следующий шаг. Завершаем граф.")
        ctx.set_current_step_id(None)

    return ctx.to_dict()