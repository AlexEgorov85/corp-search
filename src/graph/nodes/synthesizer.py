# src/graph/nodes/synthesizer.py
# coding: utf-8
"""
synthesizer_node — собирает финальный ответ на основе outputs шагов и план/metadata.

Изменения:
- Берёт step_outputs из ctx.execution.steps (по статусу 'finalized' или 'done'/'executed')
- Вызывает SynthesizerAgent (control agent) если он доступен, передаём ему plan и step_outputs
- При успешном выполнении — записываем ctx.final_answer и ctx.synth_output
- Возвращаем ctx.to_legacy_state()
"""
from __future__ import annotations
from typing import Dict, Any, List

import logging

from src.graph.context import GraphContext
from src.services.results.agent_result import AgentResult

LOG = logging.getLogger(__name__)


def synthesizer_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)

    # Если final_answer уже есть — ничего не делаем
    if ctx.final_answer is not None:
        ctx.append_history({"type": "synthesizer_already_answer"})
        return ctx.to_legacy_state()

    # Соберём step_outputs: используем шаги со статусом finalized/executed/done
    step_outputs = {}
    for step_id, step in ctx.execution.steps.items():
        if getattr(step, "status", None) in ("finalized", "executed", "done") and getattr(step, "final_result", None) is not None:
            step_outputs[step_id] = step.final_result

    # Если нет результатов — лог и выход
    if not step_outputs:
        ctx.append_history({"type": "synthesizer_no_step_outputs"})
        return ctx.to_legacy_state()

    # Попробуем вызвать SynthesizerAgent (control)
    if agent_registry is not None:
        try:
            synth_agent = agent_registry.instantiate_agent("SynthesizerAgent", control=True)
        except Exception as e:
            ctx.append_history({"type": "synthesizer_instantiate_failed", "error": str(e)})
            synth_agent = None

        if synth_agent is not None:
            try:
                plan = ctx.memory.get("plan") or ctx.to_legacy_state().get("plan")
                # Собираем контекст вызова совместимо с оригинальным кодом
                call_context = {"plan": plan, "step_outputs": step_outputs, "question": ctx.question}
                raw = synth_agent.execute_operation("synthesize", {"step_outputs": step_outputs, "plan": plan, "question": ctx.question}, context=call_context)
                # поддержка AgentResult и legacy dict
                if isinstance(raw, AgentResult):
                    if raw.status == "ok":
                        structured = raw.structured or {}
                        final = structured.get("final_answer") or raw.content
                        ctx.final_answer = final
                        ctx.synth_output = structured
                        ctx.append_history({"type": "synthesizer_agent_ok", "summary": str(final)[:200]})
                        return ctx.to_legacy_state()
                    else:
                        ctx.append_history({"type": "synthesizer_agent_error", "error": raw.error or raw.content})
                        # fallthrough to local summarization
                elif isinstance(raw, dict):
                    # support agents returning legacy dicts
                    if raw.get("status") == "ok":
                        final = raw.get("content") or (raw.get("structured") or {}).get("final_answer")
                        ctx.final_answer = final
                        ctx.synth_output = raw.get("structured")
                        ctx.append_history({"type": "synthesizer_agent_ok_legacy", "summary": str(final)[:200]})
                        return ctx.to_legacy_state()
                    else:
                        ctx.append_history({"type": "synthesizer_agent_error_legacy", "msg": raw.get("content")})
            except Exception as e:
                ctx.append_history({"type": "synthesizer_agent_exception", "error": str(e)})

    # Fallback: простая агрегация — взять последний финализированный результат
    last_step_id = None
    last_res = None
    for sid, res in step_outputs.items():
        last_step_id = sid
        last_res = res
    ctx.final_answer = last_res
    ctx.synth_output = {"fallback": True, "steps_used": list(step_outputs.keys())}
    ctx.append_history({"type": "synthesizer_fallback_used", "step_id": last_step_id})
    return ctx.to_legacy_state()
