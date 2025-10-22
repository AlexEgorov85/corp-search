# src/graph/nodes/planner.py
# coding: utf-8
"""
planner_node — узел планировщика.
Цель: сгенерировать план (декомпозицию вопроса на подвопросы) с помощью PlannerAgent.
Контракт:
- Вход: GraphContext с вопросом, установленным через set_question()
- Выход: GraphContext с заполненным ctx.plan как объект Plan (Pydantic)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.model.agent_result import AgentResult
from src.model.context.base import append_history_event, get_question, set_plan
from src.model.context.context import GraphContext
from src.model.context.models import Plan, SubQuestion
from src.utils.utils import build_tool_registry_snapshot

LOG = logging.getLogger(__name__)


def planner_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    """
    Основная функция узла планировщика.
    Логика:
      1. Преобразовать входной state в GraphContext.
      2. Получить вопрос через get_question(ctx).
      3. Если вопрос пуст — завершить с ошибкой.
      4. Инициализировать PlannerAgent.
      5. Собрать snapshot инструментов.
      6. Вызвать операцию "plan".
      7. Сохранить результат в ctx.plan через set_plan(ctx, plan_obj).
      8. В случае ошибки — использовать fallback (1 шаг).
    """
    # 🔁 Преобразуем входной dict в GraphContext
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)
    LOG.info("🔄 planner_node: начало обработки")

    # 📥 Получаем вопрос через API контекста
    question = get_question(ctx) or ""
    if not question.strip():
        LOG.warning("⚠️ planner_node: вопрос отсутствует")
        append_history_event(ctx, {"type": "planner_no_question"})
        return ctx.to_dict()

    LOG.info(f"📝 planner_node: исходный вопрос: {question}")

    # 🧠 Пытаемся использовать PlannerAgent
    if agent_registry is not None:
        planner_agent = None
        try:
            planner_agent = agent_registry.instantiate_agent("PlannerAgent", control=True)
            LOG.debug("✅ planner_node: PlannerAgent успешно создан")
        except Exception as e:
            LOG.error(f"❌ planner_node: ошибка создания PlannerAgent: {e}")
            append_history_event(ctx, {"type": "planner_instantiate_failed", "error": str(e)})

        if planner_agent is not None:
            try:
                # 📦 Собираем snapshot инструментов через AgentRegistry
                tool_registry_snapshot = build_tool_registry_snapshot(agent_registry)
                LOG.debug(
                    f"🛠️ planner_node: собран snapshot инструментов для {len(tool_registry_snapshot)} агентов"
                )

                # 🚀 Вызываем операцию plan
                params = {
                    "question": question,
                    "tool_registry_snapshot": tool_registry_snapshot,
                }
                LOG.info("🧠 planner_node: вызов PlannerAgent.execute_operation('plan', ...)")
                res = planner_agent.execute_operation("plan", params, context={})

                # ✅ Обработка успешного результата
                if isinstance(res, AgentResult) and res.status == "ok":
                    # Используем поле output → plan
                    plan_struct = res.output.get("plan") if isinstance(res.output, dict) else {}
                    LOG.info(f"✅ planner_node: план успешно сгенерирован. Структура: {plan_struct}")

                    # 💾 Сохраняем план как Pydantic-модель Plan
                    subquestions = []
                    raw_subs = plan_struct.get("subquestions", [])
                    if not isinstance(raw_subs, list):
                        LOG.error("❌ planner_node: 'subquestions' не является списком")
                        raw_subs = []

                    for sq in raw_subs:
                        if not isinstance(sq, dict):
                            continue
                        subquestions.append(
                            SubQuestion(
                                id=str(sq.get("id", "")),
                                text=str(sq.get("text", "")),
                                depends_on=sq.get("depends_on", []),  # ← критически важно: сохраняем зависимости
                            )
                        )

                    plan_obj = Plan(subquestions=subquestions)
                    set_plan(ctx, plan_obj)  # ← Используем API контекста
                    append_history_event(
                        ctx,
                        {
                            "type": "planner_agent_generated_plan",
                            "plan_summary": str(plan_struct)[:300],
                        },
                    )
                    return ctx.to_dict()

                else:
                    # ❌ Ошибка от агента
                    error_msg = res.error or str(res)
                    LOG.error(f"❌ planner_node: PlannerAgent вернул ошибку: {error_msg}")
                    append_history_event(
                        ctx,
                        {
                            "type": "planner_agent_failed",
                            "error": error_msg,
                        },
                    )

            except Exception as e:
                LOG.exception(f"💥 planner_node: исключение при вызове PlannerAgent: {e}")
                append_history_event(
                    ctx,
                    {
                        "type": "planner_agent_execute_failed",
                        "error": str(e),
                    },
                )

    # 🛟 FALLBACK: создаём один шаг с исходным вопросом
    LOG.warning("⚠️ planner_node: переход в fallback-режим")
    fallback_plan = Plan(
        subquestions=[
            SubQuestion(id="q1", text=question, depends_on=[])
        ]
    )
    set_plan(ctx, fallback_plan)  # ← Используем API контекста
    LOG.info("🛡️ planner_node: создан fallback-план с шагом q1")
    append_history_event(
        ctx,
        {
            "type": "planner_fallback_created_step",
            "step_id": "q1",
        },
    )
    return ctx.to_dict()