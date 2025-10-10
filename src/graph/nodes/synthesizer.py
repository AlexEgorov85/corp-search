# src/graph/nodes/synthesizer.py
from __future__ import annotations
from typing import Dict, Any
import logging
from src.model.agent_result import AgentResult
from src.model.context.base import (
    append_history_event,
    get_final_answer,
    set_final_answer,
)
from src.model.context.context import GraphContext

LOG = logging.getLogger(__name__)


def synthesizer_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    """
    Узел синтеза финального ответа.
    Использует только методы контекста:
      - get_final_answer() / set_final_answer()
      - get_all_step_results_for_reasoner() (через ctx)
    """
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)

    # Проверяем, не синтезирован ли уже ответ
    if get_final_answer(ctx) is not None:
        append_history_event(ctx, {"type": "synthesizer_already_done"})
        return ctx.to_dict()

    # === Получаем все результаты шагов через API контекста ===
    step_outputs = ctx.get_all_step_results_for_reasoner()
    if not step_outputs:
        LOG.warning("⚠️ synthesizer_node: нет завершённых шагов для синтеза")
        append_history_event(ctx, {"type": "synthesizer_no_step_outputs"})
        # Устанавливаем заглушку
        set_final_answer(ctx, "Не удалось получить результаты ни одного шага.")
        return ctx.to_dict()

    # Пытаемся вызвать SynthesizerAgent
    if agent_registry is not None:
        try:
            synth_agent = agent_registry.instantiate_agent("SynthesizerAgent", control=True)
        except Exception as e:
            LOG.error("❌ synthesizer_node: ошибка создания SynthesizerAgent: %s", e)
            synth_agent = None

        if synth_agent is not None:
            try:
                question = ctx.get_question()
                plan = ctx.get_plan()
                params = {
                    "step_outputs": step_outputs,
                    "plan": plan.to_dict() if hasattr(plan, "to_dict") else plan,
                    "question": question
                }
                context_for_agent = {
                    "plan": plan,
                    "step_outputs": step_outputs,
                    "question": question
                }
                raw = synth_agent.execute_operation("synthesize", params, context=context_for_agent)

                if isinstance(raw, AgentResult) and raw.status == "ok":
                    structured = raw.output or {}
                    final_answer = structured.get("final_answer") or raw.summary or str(raw.output)
                    set_final_answer(ctx, final_answer)
                    ctx.memory["synth_output"] = structured
                    append_history_event(ctx, {
                        "type": "synthesizer_agent_ok",
                        "summary": str(final_answer)[:200]
                    })
                    return ctx.to_dict()
                else:
                    error_msg = raw.error or raw.content if isinstance(raw, AgentResult) else str(raw)
                    LOG.error("❌ SynthesizerAgent вернул ошибку: %s", error_msg)

            except Exception as e:
                LOG.exception("💥 synthesizer_node: исключение при вызове SynthesizerAgent: %s", e)

    # === Fallback: используем последний результат ===
    last_step_id = list(step_outputs.keys())[-1]
    fallback_answer = step_outputs[last_step_id]
    set_final_answer(ctx, fallback_answer)
    ctx.memory["synth_output"] = {
        "fallback": True,
        "steps_used": list(step_outputs.keys())
    }
    append_history_event(ctx, {
        "type": "synthesizer_fallback_used",
        "step_id": last_step_id
    })
    return ctx.to_dict()