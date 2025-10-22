# src/agents/ResultValidatorAgent/operations/validate_result.py
"""
Операция validate_result: делегирует валидацию LLM с полным контекстом.
Теперь полностью соответствует новой архитектуре с LLMRequest/LLMResponse.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from src.agents.ResultValidatorAgent.prompt import build_validation_prompt
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult
from src.services.llm_service.model.request import LLMMessage, LLMRequest

LOG = logging.getLogger(__name__)


class Operation(BaseOperation):
    kind = OperationKind.VALIDATION
    description = "Проверяет с помощью LLM, что результат шага отвечает на подвопрос."
    params_schema = {
        "subquestion_text": {"type": "string", "required": True},
        "raw_output": {"type": "any", "required": True},
        "agent_calls": {"type": "array", "required": False},
        "step_state": {"type": "object", "required": False},
    }
    outputs_schema = {
        "type": "object",
        "properties": {
            "is_valid": "boolean",
            "confidence": "number",
            "reasoning": "array",
            "explanation": "string"
        },
    }

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        subquestion = params["subquestion_text"]
        raw_output = params["raw_output"]
        agent_calls = params.get("agent_calls", [])
        step_state = params.get("step_state", {})

        if agent.llm is None:
            return AgentResult.error(
                message="LLM не инициализирована в ResultValidatorAgent",
                stage="result_validation"
            )

        # === Формируем промпт ===
        prompt_text = build_validation_prompt(
            subquestion_text=subquestion,
            raw_output=raw_output,
            agent_calls=agent_calls,
            step_state=step_state,
        )

        # === Формируем запрос в формате LLMRequest ===
        request = LLMRequest(
            messages=[
                LLMMessage(role="user", content=prompt_text)
            ],
            temperature=0.0,  # Для валидации — детерминированность
            max_tokens=1024
        )

        try:
            # === Вызываем LLM через единый интерфейс ===
            _, llm_response = agent.llm.generate_with_request(request)

            # === Извлекаем решение ===
            validation = llm_response.json_answer
            if not validation or "validation" not in validation:
                return self._create_error_result(
                    "Не удалось извлечь валидный JSON из ответа LLM",
                    prompt_text,
                    llm_response.raw_text,
                    params,
                    llm_response
                )

            # === Валидация структуры ===
            is_valid, error_msg = self._validate_structure(validation)
            if not is_valid:
                return self._create_error_result(
                    f"Некорректная структура валидации: {error_msg}",
                    prompt_text,
                    llm_response.raw_text,
                    params,
                    llm_response,
                    validation
                )

            # === Формируем результат ===
            val_data = validation["validation"]
            output = {
                "is_valid": val_data["is_valid"],
                "confidence": val_data["confidence"],
                "reasoning": validation["reasoning"],
                "explanation": val_data["explanation"]
            }

            if val_data["is_valid"]:
                return AgentResult.ok(
                    stage="result_validation",
                    output=output,
                    summary=f"Результат для подвопроса '{subquestion}' признан валидным.",
                    input_params=params,
                    thinking=llm_response.thinking,
                    prompt=prompt_text,
                    raw_response=llm_response.raw_text,
                    tokens_used=llm_response.tokens_used
                )
            else:
                return AgentResult.error(
                    message=val_data["explanation"],
                    stage="result_validation",
                    output=output,
                    summary=f"Валидация провалена для подвопроса '{subquestion}'.",
                    input_params=params,
                    thinking=llm_response.thinking,
                    prompt=prompt_text,
                    raw_response=llm_response.raw_text,
                    tokens_used=llm_response.tokens_used
                )

        except Exception as e:
            LOG.exception("Ошибка при валидации результата через LLM")
            return AgentResult.error(
                message=f"Ошибка валидации через LLM: {e}",
                stage="result_validation",
                input_params=params
            )

    def _validate_structure(self, validation: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Валидирует структуру ответа."""
        if "reasoning" not in validation:
            return False, "Отсутствует поле reasoning"
        if not isinstance(validation["reasoning"], list) or len(validation["reasoning"]) != 4:
            return False, "reasoning должен содержать 4 элемента (V1–V4)"

        for i in range(4):
            if not isinstance(validation["reasoning"][i], str) or not validation["reasoning"][i].startswith(f"V{i+1}:"):
                return False, f"Элемент reasoning[{i}] должен начинаться с 'V{i+1}:'"

        if "validation" not in validation:
            return False, "Отсутствует поле validation"

        val = validation["validation"]
        required = ["is_valid", "confidence", "reason", "explanation"]
        for field in required:
            if field not in val:
                return False, f"Отсутствует поле validation.{field}"

        if not (0 <= val.get("confidence", -1) <= 1):
            return False, "Некорректная уверенность в validation.confidence"

        return True, None

    def _create_error_result(
        self,
        message: str,
        prompt: str,
        raw_response: str,
        params: Dict,
        llm_response,
        validation: Optional[Dict] = None
    ) -> AgentResult:
        """Создаёт объект AgentResult с ошибкой и полной диагностикой."""
        LOG.error("ResultValidatorAgent ошибка: %s", message)
        LOG.debug("Промпт: %s", prompt)
        LOG.debug("Сырой ответ LLM: %s", raw_response)
        LOG.debug("Извлечённая валидация: %s", json.dumps(validation, indent=2) if validation else "Нет")

        return AgentResult.error(
            message=message,
            stage="result_validation",
            input_params=params,
            thinking=getattr(llm_response, "thinking", ""),
            prompt=prompt,
            raw_response=raw_response,
            tokens_used=getattr(llm_response, "tokens_used", None)
        )