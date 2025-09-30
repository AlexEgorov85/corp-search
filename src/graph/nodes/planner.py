# src/graph/nodes/planner.py
# coding: utf-8
"""
planner_node — узел планировщика.
Цель: сгенерировать план (декомпозицию вопроса на подвопросы) с помощью PlannerAgent.

Контракт:
- Вход: dict с ключом "question"
- Выход: dict с обновлённым execution.steps, current_subquestion_id и memory["plan"]

Пример входа:
{
    "question": "Найди книги Пушкина и укажи главного героя в последней из них?"
}

Пример выхода (после успешного plan):
{
    "question": "...",
    "execution": {
        "current_subquestion_id": "q1",
        "steps": {
            "q1": {"id": "q1", "subquestion_text": "Какие книги написал Пушкин?", "status": "pending"},
            "q2": {"id": "q2", "subquestion_text": "Какая из книг — последняя?", "status": "pending"},
            "q3": {"id": "q3", "subquestion_text": "Кто главный герой в последней книге?", "status": "pending"}
        }
    },
    "memory": {
        "plan": {
            "subquestions": [
                {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []},
                {"id": "q2", "text": "Какая из книг — последняя?", "depends_on": ["q1"]},
                {"id": "q3", "text": "Кто главный герой в последней книге?", "depends_on": ["q2"]}
            ]
        }
    }
}
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from src.graph.context_model import GraphContext
from src.graph.context_ops import (
    get_question,
    set_plan,
    ensure_step,
    set_current_subquestion_id,
    append_history_event,
)
from src.services.results.agent_result import AgentResult

LOG = logging.getLogger(__name__)


def _build_tool_registry_snapshot(full_tool_registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создаёт упрощённую и безопасную копию tool_registry для передачи в PlannerAgent.
    Удаляет чувствительные поля (implementation, config и т.д.), оставляя только:
      - title
      - description
      - operations (с их kind, description, params, outputs)

    Пример входа:
    {
        "BooksLibraryAgent": {
            "implementation": "src.agents.BooksLibraryAgent.core:BooksLibraryAgent",
            "config": {"db_uri": "..."},
            "title": "База книг",
            "description": "Доступ к книгам и авторам",
            "operations": {
                "list_books": {"description": "Получить список книг", "kind": "direct"}
            }
        }
    }

    Пример выхода:
    {
        "BooksLibraryAgent": {
            "title": "База книг",
            "description": "Доступ к книгам и авторам",
            "operations": {
                "list_books": {"description": "Получить список книг", "kind": "direct"}
            }
        }
    }
    """
    snapshot = {}
    for name, meta in full_tool_registry.items():
        if not isinstance(meta, dict):
            snapshot[name] = {}
            continue
        safe_meta = {
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "operations": {}
        }
        ops = meta.get("operations", {})
        if isinstance(ops, dict):
            for op_name, op_meta in ops.items():
                if not isinstance(op_meta, dict):
                    safe_meta["operations"][op_name] = {"description": ""}
                    continue
                safe_meta["operations"][op_name] = {
                    "kind": op_meta.get("kind", "direct"),
                    "description": op_meta.get("description", ""),
                    "params": op_meta.get("params", {}),
                    "outputs": op_meta.get("outputs", {})
                }
        snapshot[name] = safe_meta
    return snapshot


def planner_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    """
    Основная функция узла планировщика.
    Логика:
      1. Преобразовать входной state в GraphContext.
      2. Получить вопрос.
      3. Если вопрос пуст — завершить с ошибкой.
      4. Инициализировать PlannerAgent.
      5. Собрать snapshot инструментов.
      6. Вызвать операцию "plan".
      7. Сохранить результат в ctx.memory["plan"] и создать шаги.
      8. В случае ошибки — использовать fallback (1 шаг).
    """
    # 🔁 Преобразуем входной dict в GraphContext (единственный канонический контекст)
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)
    LOG.info("🔄 planner_node: начало обработки")

    # 📥 Получаем вопрос из контекста (через функцию)
    question = get_question(ctx) or ""
    if not question.strip():
        LOG.warning("⚠️ planner_node: вопрос отсутствует")
        append_history_event(ctx, {"type": "planner_no_question"})
        return ctx.to_legacy_state()

    LOG.info(f"📝 planner_node: исходный вопрос: {question}")

    # 🧠 Пытаемся использовать PlannerAgent
    if agent_registry is not None:
        planner_agent = None
        try:
            # 🧪 Инстанцируем PlannerAgent
            planner_agent = agent_registry.instantiate_agent("PlannerAgent", control=True)
            LOG.debug("✅ planner_node: PlannerAgent успешно создан")
        except Exception as e:
            LOG.error(f"❌ planner_node: ошибка создания PlannerAgent: {e}")
            append_history_event(ctx, {"type": "planner_instantiate_failed", "error": str(e)})

        if planner_agent is not None:
            try:
                # 📦 Собираем snapshot инструментов
                full_tool_registry = agent_registry.tool_registry
                tool_registry_snapshot = _build_tool_registry_snapshot(full_tool_registry)
                LOG.debug(f"🛠️ planner_node: собран snapshot инструментов для {len(tool_registry_snapshot)} агентов")

                # 🚀 Вызываем операцию plan с обязательными параметрами
                params = {
                    "question": question,
                    "tool_registry_snapshot": tool_registry_snapshot
                }
                LOG.info("🧠 planner_node: вызов PlannerAgent.execute_operation('plan', ...)")
                res = planner_agent.execute_operation("plan", params, context={})

                # ✅ Обработка успешного результата
                if isinstance(res, AgentResult) and res.status == "ok":
                    plan_struct = res.structured or res.content or {}
                    LOG.info(f"✅ planner_node: план успешно сгенерирован. Структура: {plan_struct}")

                    # 💾 Сохраняем план в ctx.memory (обязательно!)
                    set_plan(ctx, plan_struct)

                    # ➕ Создаём шаги из подвопросов
                    subqs = plan_struct.get("subquestions") if isinstance(plan_struct, dict) else None
                    if subqs:
                        first_id: Optional[str] = None
                        for s in subqs:
                            sid = s.get("id") or f"sq_{len(ctx.execution.steps) + 1}"
                            text = s.get("text") or s.get("title") or ""
                            LOG.debug(f"➕ planner_node: создаём шаг {sid}: {text}")
                            ensure_step(ctx, sid, subquestion_text=text)
                            if first_id is None:
                                first_id = sid

                        if first_id:
                            set_current_subquestion_id(ctx, first_id)
                            LOG.info(f"🎯 planner_node: установлен текущий подвопрос: {first_id}")

                    append_history_event(ctx, {
                        "type": "planner_agent_generated_plan",
                        "plan_summary": str(plan_struct)[:300]
                    })
                    return ctx.to_legacy_state()

                else:
                    # ❌ Ошибка от агента
                    error_msg = res.content if hasattr(res, "content") else str(res)
                    LOG.error(f"❌ planner_node: PlannerAgent вернул ошибку: {error_msg}")
                    append_history_event(ctx, {
                        "type": "planner_agent_failed",
                        "result": error_msg
                    })

            except Exception as e:
                LOG.exception(f"💥 planner_node: исключение при вызове PlannerAgent: {e}")
                append_history_event(ctx, {
                    "type": "planner_agent_execute_failed",
                    "error": str(e)
                })

    # 🛟 FALLBACK: создаём один шаг с исходным вопросом
    LOG.warning("⚠️ planner_node: переход в fallback-режим")
    step_id = "q1"
    ensure_step(ctx, step_id, subquestion_text=question, status="pending")
    set_current_subquestion_id(ctx, step_id)
    fallback_plan = {"subquestions": [{"id": step_id, "text": question}]}
    set_plan(ctx, fallback_plan)  # ← важно: сохраняем даже fallback-план!
    LOG.info(f"🛡️ planner_node: создан fallback-план с шагом {step_id}")
    append_history_event(ctx, {
        "type": "planner_fallback_created_step",
        "step_id": step_id
    })
    return ctx.to_legacy_state()