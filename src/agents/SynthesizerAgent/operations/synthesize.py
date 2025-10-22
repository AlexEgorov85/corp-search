# src/agents/SynthesizerAgent/operations/synthesize.py
"""
Операция `synthesize` — генерация финального ответа.
Теперь полностью соответствует новой архитектуре с LLMRequest/LLMResponse.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional, Tuple
from src.agents.SynthesizerAgent.prompt import build_synthesis_prompt
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult
from src.services.llm_service.model.request import LLMMessage, LLMRequest

LOG = logging.getLogger(__name__)


class Operation(BaseOperation):
    """
    Операция синтеза финального ответа.
    """
    kind = OperationKind.DIRECT
    description = (
        "На основании outputs шагов и метаданных плана формирует итоговый ответ "
        "и вспомогательные данные (evidence)."
    )
    params_schema = {
        "question": {"type": "string", "required": True},
        "plan": {"type": "object", "required": True},
        "step_outputs": {"type": "object", "required": True},
    }
    outputs_schema = {
        "type": "object",
        "properties": {
            "final_answer": "string",
            "confidence": "number",
            "reasoning": "array",
            "explanation": "string"
        }
    }

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        question = params["question"]
        plan = params["plan"]
        step_outputs = params["step_outputs"]

        if not agent.llm:
            return AgentResult.error(
                message="LLM не инициализирована в SynthesizerAgent",
                stage="synthesis"
            )

        # === Формируем промпт ===
        prompt_text = build_synthesis_prompt(
            original_question=question,
            plan=plan,
            step_outputs=step_outputs,
        )

        # === Формируем запрос в формате LLMRequest ===
        request = LLMRequest(
            messages=[
                LLMMessage(role="user", content=prompt_text)
            ],
            temperature=0.0,  # Для синтеза — детерминированность
            max_tokens=2048
        )

        try:
            # === Вызываем LLM через единый интерфейс ===
            _, llm_response = agent.llm.generate_with_request(request)

            # === Извлекаем решение ===
            synthesis = llm_response.json_answer
            if not synthesis or "synthesis" not in synthesis:
                return self._create_error_result(
                    "Не удалось извлечь валидный JSON из ответа LLM",
                    prompt_text,
                    llm_response.raw_text,
                    params,
                    llm_response
                )

            # === Валидация структуры ===
            is_valid, error_msg = self._validate_structure(synthesis)
            if not is_valid:
                return self._create_error_result(
                    f"Некорректная структура синтеза: {error_msg}",
                    prompt_text,
                    llm_response.raw_text,
                    params,
                    llm_response,
                    synthesis
                )

            # === Формируем результат ===
            synth_data = synthesis["synthesis"]
            output = {
                "final_answer": synth_data["final_answer"],
                "confidence": synth_data["confidence"],
                "reasoning": synthesis["reasoning"],
                "explanation": synth_data["explanation"]
            }

            return AgentResult.ok(
                stage="synthesis",
                output=output,
                summary=f"Успешный синтез ответа для вопроса: {question[:50]}...",
                input_params=params,
                thinking=llm_response.thinking,
                prompt=prompt_text,
                raw_response=llm_response.raw_text,
                tokens_used=llm_response.tokens_used
            )

        except Exception as e:
            LOG.exception("Ошибка при синтезе финального ответа через LLM")
            return AgentResult.error(
                message=f"Ошибка синтеза через LLM: {e}",
                stage="synthesis",
                input_params=params
            )

    def _validate_structure(self, synthesis: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Валидирует структуру ответа."""
        if "reasoning" not in synthesis:
            return False, "Отсутствует поле reasoning"
        if not isinstance(synthesis["reasoning"], list) or len(synthesis["reasoning"]) != 4:
            return False, "reasoning должен содержать 4 элемента (S1–S4)"

        for i in range(4):
            if not isinstance(synthesis["reasoning"][i], str) or not synthesis["reasoning"][i].startswith(f"S{i+1}:"):
                return False, f"Элемент reasoning[{i}] должен начинаться с 'S{i+1}:'"

        if "synthesis" not in synthesis:
            return False, "Отсутствует поле synthesis"

        synth = synthesis["synthesis"]
        required = ["final_answer", "confidence", "reason", "explanation"]
        for field in required:
            if field not in synth:
                return False, f"Отсутствует поле synthesis.{field}"

        if not (0 <= synth.get("confidence", -1) <= 1):
            return False, "Некорректная уверенность в synthesis.confidence"

        return True, None

    def _create_error_result(
        self,
        message: str,
        prompt: str,
        raw_response: str,
        params: Dict,
        llm_response,
        synthesis: Optional[Dict] = None
    ) -> AgentResult:
        """Создаёт объект AgentResult с ошибкой и полной диагностикой."""
        LOG.error("SynthesizerAgent ошибка: %s", message)
        LOG.debug("Промпт: %s", prompt)
        LOG.debug("Сырой ответ LLM: %s", raw_response)
        LOG.debug("Извлечённый синтез: %s", json.dumps(synthesis, indent=2) if synthesis else "Нет")

        return AgentResult.error(
            message=message,
            stage="synthesis",
            input_params=params,
            thinking=getattr(llm_response, "thinking", ""),
            prompt=prompt,
            raw_response=raw_response,
            tokens_used=getattr(llm_response, "tokens_used", None)
        )