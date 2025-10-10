# src/agents/ReasonerAgent/operations/decide_next_stage.py
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Optional
from src.agents.operations_base import BaseOperation, OperationKind
from src.agents.ReasonerAgent.prompts import build_universal_reasoner_prompt
from src.model.agent_result import AgentResult

LOG = logging.getLogger(__name__)

class Operation(BaseOperation):
    kind = OperationKind.CONTROL
    description = "Принимает решение о следующем этапе на основе ответа LLM. Никакой дополнительной логики — только парсинг и валидация."
    params_schema = {
        "subquestion": {"type": "object", "required": True},
        "step_state": {"type": "object", "required": True},
        "step_outputs": {"type": "object", "required": False},
        "tool_registry_snapshot": {"type": "object", "required": True},
    }
    outputs_schema = {
        "next_stage": "string",
        "selected_tool": "object | null",
        "previous_output_ref": "string | null",
        "run_entity_validation": "boolean",
        "run_fetch_data": "boolean",
        "run_process_data": "boolean",
        "retry_count": "integer",
        "validation_feedback": "array | null",
        "reason": "string"
    }

    def _parse_llm_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Безопасный парсинг JSON из ответа LLM."""
        # Попытка извлечь из fenced block
        fenced = re.findall(r"```(?:json)?\s*([\s\S]+?)\s*```", raw, flags=re.MULTILINE)
        if fenced:
            try:
                return json.loads(fenced[0].strip())
            except json.JSONDecodeError:
                pass

        # Попытка найти первый валидный JSON-объект
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1].strip())
            except json.JSONDecodeError:
                pass

        return None

    def _validate_decision(self, decision: Dict[str, Any]) -> tuple[bool, str]:
        required_fields = {
            "next_stage", "selected_tool", "previous_output_ref",
            "run_entity_validation", "run_fetch_data", "run_process_data", "retry_count",
            "validation_feedback", "reason"
        }
        if not required_fields.issubset(decision.keys()):
            missing = required_fields - set(decision.keys())
            return False, f"Отсутствуют обязательные поля: {missing}"

        allowed_stages = {
            "validate_entities", "fetch_data",
            "process_data", "validate_result", "finalize"
        }
        stage = decision["next_stage"]
        if stage not in allowed_stages:
            return False, f"Недопустимое значение next_stage: '{stage}'. Допустимые: {allowed_stages}"

        if not isinstance(decision["reason"], str) or not decision["reason"].strip():
            return False, "Поле 'reason' должно быть непустой строкой"

        return True, ""

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        try:
            subquestion = params["subquestion"]
            step_state = params["step_state"]
            step_outputs = params.get("step_outputs", {})
            tool_registry_snapshot = params["tool_registry_snapshot"]
            question = subquestion["text"]

            # Формируем промпт
            prompt = build_universal_reasoner_prompt(
                question=question,
                step_outputs=step_outputs,
                tool_registry_snapshot=tool_registry_snapshot,
                step_state=step_state
            )

            if agent.llm is None:
                return AgentResult.error("LLM не инициализирована в ReasonerAgent")

            raw_response = agent.llm.generate(prompt)
            LOG.debug("Raw LLM response: %.500s", raw_response)

            decision = self._parse_llm_response(raw_response)
            if decision is None:
                return AgentResult.error("Не удалось извлечь валидный JSON из ответа LLM")

            is_valid, error_msg = self._validate_decision(decision)
            if not is_valid:
                return AgentResult.error(f"Некорректная структура решения от LLM: {error_msg}")

            # ✅ ВАЖНО: НИКАКОЙ ДОПОЛНИТЕЛЬНОЙ ЛОГИКИ!
            # Просто возвращаем то, что сказал LLM.
            # Executor сам решит, что делать с selected_tool или его отсутствием.

            return AgentResult.ok(
                stage="reasoning",
                output=decision,
                summary=f"Принято решение для шага: next_stage={decision.get('next_stage')}"
            )

        except Exception as e:
            LOG.exception("Ошибка в decide_next_stage")
            return AgentResult.error(f"Ошибка в decide_next_stage: {e}")