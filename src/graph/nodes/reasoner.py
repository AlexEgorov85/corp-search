# src/graph/nodes/reasoner.py
"""
Узел Reasoner.
Цель: принять решение о следующем этапе выполнения шага.
Логирование:
  - вход в шаг
  - вызов ReasonerAgent
  - выбранные гипотезы и уверенность
  - определённый этап (data_fetch, processing, validation)
  - завершение шага
"""
from __future__ import annotations
import logging
from typing import Any, Dict
from src.model.agent_result import AgentResult
from src.model.context.context import GraphContext
from src.utils.utils import build_tool_registry_snapshot

LOG = logging.getLogger(__name__)


def reasoner_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    if agent_registry is None:
        raise ValueError("reasoner_node: agent_registry is required")
    ctx = GraphContext.from_state_dict(state)
    step_id = ctx.get_current_step_id()
    if not step_id:
        LOG.warning("⚠️ reasoner_node: нет текущего шага")
        return ctx.to_dict()

    # === Проверка: завершена ли валидация и провалена ли она? ===
    if ctx.is_stage_completed(step_id, "validation"):
        step = ctx.get_execution_step(step_id)
        validation_result = step.validation_result if step else None
        is_valid = (
            validation_result.get("is_valid", False)
            if isinstance(validation_result, dict)
            else False
        )
        retry_count = getattr(step, "retry_count", 0)

        if not is_valid and retry_count < 2:
            LOG.info("🔁 Валидация провалена, запускаем повторную попытку для шага %s (попытка %d)", step_id, retry_count + 1)
            # Сбрасываем этапы, но сохраняем expected_stages
            step.retry_count += 1
            for stage in step.completed_stages:
                step.completed_stages[stage] = False
            step.raw_output = None
            step.validation_result = None
            step.error = None
            # Reasoner будет вызван снова на следующей итерации
            return ctx.to_dict()
        elif not is_valid:
            LOG.error("❌ Валидация провалена после %d попыток, завершаем шаг %s", retry_count, step_id)
            ctx.mark_step_completed(step_id)
            return ctx.to_dict()

    # === Обычная логика: вызов ReasonerAgent ===
    if ctx.is_step_fully_completed(step_id):
        ctx.mark_step_completed(step_id)
        LOG.info("🏁 Шаг %s завершён", step_id)
        return ctx.to_dict()

    subquestion_text = ctx.get_subquestion_text(step_id)
    LOG.info("🔍 Обработка шага %s: '%s'", step_id, subquestion_text)

    step_outputs = ctx.get_relevant_step_outputs_for_reasoner(step_id)
    tool_registry_snapshot = build_tool_registry_snapshot(agent_registry)

    params = {
        "subquestion": {"id": step_id, "text": subquestion_text},
        "step_state": {"stage": ctx.get_current_stage(step_id)},
        "step_outputs": step_outputs,
        "tool_registry_snapshot": tool_registry_snapshot,
    }

    # Проверка: есть ли уже решение и нет ошибки → используем его
    step = ctx.get_execution_step(step_id)
    existing_decision = step.decision if step else None
    has_error = step.error is not None if step else False

    if existing_decision and not has_error:
        LOG.info("🔄 reasoner_node: использование существующего решения для шага %s", step_id)
        decision = existing_decision
    else:
        try:
            reasoner_agent = agent_registry.instantiate_agent("ReasonerAgent", control=True)
            result = reasoner_agent.execute_operation("decide_next_stage", params)
            if isinstance(result, AgentResult) and result.status == "ok":
                ctx.record_reasoner_decision(step_id, result.output)
                decision = result.output

                # Логируем гипотезы
                hypotheses = decision.get("hypotheses", [])
                LOG.info("🧠 Получено %d гипотез от ReasonerAgent", len(hypotheses))
                for i, hyp in enumerate(hypotheses):
                    LOG.info(
                        "  🧪 Гипотеза %d: %s.%s (уверенность: %.2f) — %s",
                        i,
                        hyp["agent"],
                        hyp["operation"],
                        hyp["confidence"],
                        hyp["reason"],
                    )
                # Логируем выбранную гипотезу
                if "final_decision" in decision:
                    sel_idx = decision["final_decision"].get("selected_hypothesis", 0)
                    if 0 <= sel_idx < len(hypotheses):
                        sel = hypotheses[sel_idx]
                        LOG.info(
                            "  ✅ Выбрана гипотеза %d: %s.%s (уверенность: %.2f)",
                            sel_idx, sel["agent"], sel["operation"], sel["confidence"]
                        )
                # Логируем этапы
                needs_proc = decision.get("postprocessing", {}).get("needed", False)
                needs_val = decision.get("validation", {}).get("needed", True)
                LOG.info("🔧 Этапы: postprocessing=%s, validation=%s", needs_proc, needs_val)
            else:
                LOG.error("❌ Reasoner вернул ошибку: %s", result.error)
                decision = None
        except Exception as e:
            LOG.exception("💥 Ошибка в reasoner_node: %s", e)
            decision = None

    # Установка следующей операции через get_current_tool_call
    tool_call = ctx.get_current_tool_call(step_id)
    if tool_call:
        LOG.info("⚙️ reasoner_node: установка вызова %s.%s для этапа %s",
                 tool_call["agent"], tool_call["operation"], ctx.get_current_stage(step_id))
    else:
        LOG.warning("⚠️ reasoner_node: не удалось определить вызов для текущего этапа")

    return ctx.to_dict()