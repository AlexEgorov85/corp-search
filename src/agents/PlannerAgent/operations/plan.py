# src/agents/PlannerAgent/operations/plan.py
"""
Операция 'plan' для PlannerAgent.
Цель: сгенерировать структурированный план с полной прозрачностью принятия решений.
Структура итогового JSON:
{
  "reasoning": ["P1: ...", ..., "P5: ..."],
  "planning": { "needed": true|false, "confidence": 0.0–1.0, "reason": "...", "explanation": "..." },
  "subquestions": [
    { "id": "q1", "text": "...", "depends_on": [], "confidence": 0.0–1.0, "reason": "...", "explanation": "..." }
  ],
  "final_decision": { "explanation": "Итоговое резюме в 1–2 предложения" }
}
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult
from src.agents.PlannerAgent.decomposition import DecompositionPhase
LOG = logging.getLogger(__name__)


class Operation(BaseOperation):
    """Операция 'plan' для PlannerAgent."""
    kind = OperationKind.DIRECT
    description = "Сгенерировать декомпозицию вопроса на подвопросы."
    params_schema = {
        "question": {"type": "string", "required": True},
        "tool_registry_snapshot": {"type": "object", "required": True}
    }
    outputs_schema = {
        "type": "object",
        "properties": {
            "plan": {"type": "object"}
        }
    }

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        """Основной метод выполнения операции."""
        question = params.get("question")
        tool_registry = params.get("tool_registry_snapshot", {})
        if not question or not isinstance(question, str):
            return AgentResult.error(
                message="Не указан или неверный параметр 'question'",
                stage="plan_generation",
                input_params=params
            )
        if not tool_registry:
            LOG.warning("Пустой tool_registry_snapshot")

        decomposition_phase = DecompositionPhase(llm=agent.llm, max_retries=3)
        success, decomposition, feedback, diagnostics = decomposition_phase.run(params)
        LOG.debug("Декомпозиция завершена. Успех: %s", success)

        if success:
            return AgentResult.ok(
                stage="plan_generation",
                output={"plan": decomposition},
                summary=f"Успешно сгенерирован план из {len(decomposition.get('subquestions', []))} подвопросов",
                input_params=params,
                thinking=diagnostics.get("thinking"),
                prompt=diagnostics.get("prompt"),
                raw_response=diagnostics.get("raw_response"),
                tokens_used=diagnostics.get("tokens_used")
            )
        else:
            return AgentResult.error(
                stage="plan_generation",
                message=f"Не удалось сгенерировать декомпозицию: {feedback}",
                input_params=params,
                prompt=diagnostics.get("prompt"),
                raw_response=diagnostics.get("raw_response"),
                thinking=diagnostics.get("thinking"),
                tokens_used=diagnostics.get("tokens_used")
            )