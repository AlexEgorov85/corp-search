# src/graph/nodes/synthesizer.py
"""
Узел синтеза финального ответа.
Цель: собрать результаты и сформировать итоговый ответ.
Логирование:
  - вход в синтез
  - вызов SynthesizerAgent
  - успешный/ошибочный результат
  - fallback на последний результат
"""
from __future__ import annotations
import logging
from typing import Any, Dict
from src.model.agent_result import AgentResult
from src.model.context.context import GraphContext

LOG = logging.getLogger(__name__)

def synthesizer_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    ctx = GraphContext.from_state_dict(state)
    if ctx.get_final_answer() is not None:
        LOG.info("✅ Финальный ответ уже синтезирован")
        return ctx.to_dict()

    step_outputs = ctx.get_all_completed_step_results()
    if not step_outputs:
        LOG.warning("⚠️ synthesizer_node: нет завершённых шагов")
        ctx.set_final_answer("Не удалось получить результаты ни одного шага.")
        return ctx.to_dict()

    LOG.info("🎯 Запуск синтеза финального ответа. Шагов: %d", len(step_outputs))

    if agent_registry is None:
        LOG.warning("⚠️ synthesizer_node: agent_registry не передан, используем fallback")
        last_result = list(step_outputs.values())[-1]
        ctx.set_final_answer(str(last_result))
        return ctx.to_dict()

    try:
        synth_agent = agent_registry.instantiate_agent("SynthesizerAgent", control=True)
        result = synth_agent.execute_operation("synthesize", {
            "question": ctx.get_question(),
            "plan": ctx.get_plan().to_dict(),
            "step_outputs": step_outputs
        })
        if isinstance(result, AgentResult) and result.status == "ok":
            final_answer = result.output.get("final_answer", str(result.output))
            ctx.set_final_answer(final_answer)
            LOG.info("✅ Финальный ответ синтезирован")
            LOG.debug("📄 Ответ: %.200s", final_answer)
        else:
            raise ValueError(result.error)
    except Exception as e:
        LOG.warning("⚠️ synthesizer fallback: %s", e)
        last_result = list(step_outputs.values())[-1]
        ctx.set_final_answer(str(last_result))
        LOG.info("🛡️ Использован fallback на последний результат")

    return ctx.to_dict()