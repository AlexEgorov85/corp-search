# src/agents/ResultValidatorAgent/operations/validate_result.py
"""
Операция validate_result: делегирует валидацию LLM.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List
from src.agents.ResultValidatorAgent.prompt import build_validation_prompt
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult


LOG = logging.getLogger(__name__)


class Operation(BaseOperation):
    kind = OperationKind.VALIDATION
    description = "Проверяет с помощью LLM, что результат шага отвечает на подвопрос."
    params_schema = {
        "subquestion_text": {"type": "string", "required": True},
        "raw_output": {"type": "any", "required": True},
    }
    outputs_schema = {
        "type": "object",
        "properties": {
            "is_valid": "boolean",
            "feedback": "array",
        },
    }

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Безопасный парсинг JSON из ответа LLM."""
        text = text.strip()
        # Убираем fenced block, если есть
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Попытка найти первый JSON-объект
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Не удалось распарсить JSON из ответа LLM: {text[:200]}...")

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        subquestion = params["subquestion_text"]
        raw_output = params["raw_output"]

        if agent.llm is None:
            return AgentResult.error(
                message="LLM не инициализирована в ResultValidatorAgent",
                stage="result_validation"
            )

        prompt = build_validation_prompt(subquestion, raw_output)
        LOG.debug("LLM prompt for validation:\n%s", prompt)

        try:
            raw_response = agent.llm.generate(prompt)
            LOG.debug("LLM raw response:\n%s", raw_response)

            parsed = self._parse_llm_response(raw_response)

            is_valid = bool(parsed.get("is_valid", False))
            feedback = parsed.get("feedback", [])
            if not isinstance(feedback, list):
                feedback = [str(feedback)]

            output = {"is_valid": is_valid, "feedback": feedback}

            if is_valid:
                return AgentResult.ok(
                    stage="result_validation",
                    output=output,
                    summary=f"Результат для подвопроса '{subquestion}' признан валидным LLM.",
                )
            else:
                error_msg = "; ".join(feedback) or "Результат не прошёл валидацию."
                return AgentResult.error(
                    message=error_msg,
                    stage="result_validation",
                    meta=output,
                    summary=f"Валидация провалена для подвопроса '{subquestion}'.",
                )

        except Exception as e:
            LOG.exception("Ошибка при валидации результата через LLM")
            return AgentResult.error(
                message=f"Ошибка валидации через LLM: {e}",
                stage="result_validation",
                meta={"is_valid": False, "feedback": [str(e)]},
                summary=f"Исключение при вызове LLM для валидации подвопроса '{subquestion}'."
            )