# src/agents/ReasonerAgent/operations/decide_next_stage.py
"""
Операция decide_next_stage: принимает решение о следующем этапе выполнения шага.
Цель: на основе подвопроса, контекста и доступных инструментов сформировать
структурированное решение в формате JSON, полностью совместимом с новой архитектурой LLM.

Основные компоненты решения:
1. reasoning — массив из 7 ответов на аналитические вопросы (R1–R7)
2. hypotheses — список гипотез (вызовов инструментов) с уверенностью и пояснениями
3. postprocessing — объект с флагом, уверенностью и пояснением необходимости постобработки
4. validation — объект с флагом, уверенностью и пояснением необходимости валидации
5. final_decision — итоговый выбор и человекочитаемое резюме

Архитектурные требования:
- Используется только `agent.llm.generate_with_request(request)`
- Все диагностические поля (`thinking`, `prompt`, `raw_response`, `tokens_used`)
  берутся из `LLMResponse`
- Валидация строгая: все обязательные поля должны присутствовать
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult
from src.agents.ReasonerAgent.prompts import build_universal_reasoner_prompt
from src.services.llm_service.model.request import LLMMessage, LLMRequest

LOG = logging.getLogger(__name__)


class Operation(BaseOperation):
    """
    Операция 'decide_next_stage' для ReasonerAgent.
    """
    kind = OperationKind.CONTROL
    description = "Принимает решение о следующем этапе на основе гипотез с объективной оценкой уверенности."
    params_schema = {
        "subquestion": {"type": "object", "required": True},
        "step_state": {"type": "object", "required": True},
    }

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        """
        Основной метод выполнения операции.
        Логика:
        1. Проверяет инициализацию LLM
        2. Формирует запрос в виде LLMRequest
        3. Вызывает LLM через generate_with_request()
        4. Извлекает решение из LLMResponse.json_answer
        5. Применяет детерминированный выбор гипотезы
        6. Валидирует структуру и возвращает AgentResult
        """
        if not agent.llm:
            return AgentResult.error(
                message="LLM не инициализирована в ReasonerAgent",
                stage="reasoning"
            )

        # === 1. Формируем запрос в стандартизированном формате LLMRequest ===
        request = self._build_request(params, context)
        try:
            prompt_str = request.model_dump_json()
        except Exception:
            prompt_str = str(request)

        try:
            # === 2. Вызываем LLM через единый интерфейс ===
            _, llm_response = agent.llm.generate_with_request(request)

            # === 3. Извлекаем решение из структурированного ответа ===
            decision = llm_response.json_answer
            if not decision:
                try:
                    decision = json.loads(llm_response.answer)
                except (json.JSONDecodeError, TypeError):
                    decision = None

            if not decision:
                return self._create_error_result(
                    "Не удалось извлечь валидный JSON из ответа LLM",
                    prompt_str,
                    llm_response.raw_text,
                    params,
                    llm_response
                )

            # === 4. Применяем детерминированный выбор гипотезы ===
            decision = self._apply_deterministic_selection(decision)

            # === 5. Валидация структуры ===
            is_valid, error_msg = self._validate_decision(decision)
            if not is_valid:
                return self._create_error_result(
                    f"Некорректная структура решения от LLM: {error_msg}",
                    prompt_str,
                    llm_response.raw_text,
                    params,
                    llm_response,
                    decision
                )

            # === 6. Формируем итоговое резюме ===
            hypotheses_count = len(decision.get("hypotheses", []))
            selected_idx = decision["final_decision"]["selected_hypothesis"]
            summary = f"Принято решение для шага: {hypotheses_count} гипотез, выбрана {selected_idx}"

            return AgentResult.ok(
                stage="reasoning",
                output=decision,
                summary=summary,
                input_params=params,
                thinking=llm_response.thinking,
                prompt=prompt_str,
                raw_response=llm_response.raw_text,
                tokens_used=llm_response.tokens_used
            )

        except Exception as e:
            LOG.exception("Ошибка в decide_next_stage")
            return AgentResult.error(
                message=f"Ошибка в decide_next_stage: {str(e)}",
                stage="reasoning",
                input_params=params,
                prompt=prompt_str
            )

    def _build_request(self, params: Dict[str, Any], context: Dict[str, Any]) -> LLMRequest:
        """
        Формирует запрос в стандартизированном формате LLMRequest.
        Использует промпт из prompts.py и конвертирует его в LLMMessage.
        """
        messages = build_universal_reasoner_prompt(
            question=params["subquestion"]["text"],
            step_outputs=context.get("step_outputs", {}),
            tool_registry_snapshot=params.get("tool_registry_snapshot", {}),
            step_state=params["step_state"]
        )
        llm_messages = [
            LLMMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        # === Читаем параметры из конфигурации агента ===
        config = self.agent.config if hasattr(self, 'agent') else {}
        return LLMRequest(
            messages=llm_messages,
            temperature=config.get("LLM_TEMPERATURE", 0.3),
            max_tokens=config.get("LLM_MAX_TOKENS", 2048),
            top_p=config.get("LLM_TOP_P", 0.9)
        )

    def _apply_deterministic_selection(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Применяет детерминированный алгоритм выбора гипотезы.
        Это гарантирует, что даже если LLM ошиблась в final_decision,
        система сама исправит выбор на основе уверенности.
        """
        hypotheses = decision.get("hypotheses", [])
        if not hypotheses:
            decision["final_decision"]["selected_hypothesis"] = -1
            return decision

        # Фильтруем гипотезы с уверенностью < 0.5
        viable = [i for i, h in enumerate(hypotheses) if h.get("confidence", 0) >= 0.5]
        if not viable:
            decision["final_decision"]["selected_hypothesis"] = -1
            return decision

        # Выбираем гипотезу с максимальной уверенностью
        best_idx = max(viable, key=lambda i: hypotheses[i].get("confidence", 0))
        decision["final_decision"]["selected_hypothesis"] = best_idx
        return decision

    def _validate_decision(self, decision: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Валидирует полную структуру решения.
        Проверяет наличие и корректность всех обязательных полей.
        """
        # === 1. Проверка reasoning ===
        if "reasoning" not in decision:
            return False, "Отсутствует поле reasoning"
        reasoning = decision["reasoning"]
        if not isinstance(reasoning, list) or len(reasoning) != 7:
            return False, "reasoning должен содержать ровно 7 элементов (R1–R7)"
        for i in range(7):
            if not isinstance(reasoning[i], str) or not reasoning[i].startswith(f"R{i+1}:"):
                return False, f"Элемент reasoning[{i}] должен начинаться с 'R{i+1}:'"

        # === 2. Проверка гипотез ===
        hypotheses = decision.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            return False, "hypotheses должен быть списком"
        for i, h in enumerate(hypotheses):
            for field in ["agent", "operation", "params", "confidence", "reason", "explanation"]:
                if field not in h:
                    return False, f"Гипотеза #{i} не содержит поля '{field}'"
            if not (0 <= h.get("confidence", -1) <= 1):
                return False, f"Некорректная уверенность в гипотезе #{i}"

        # === 3. Проверка postprocessing ===
        postproc = decision.get("postprocessing")
        if not postproc:
            return False, "Отсутствует поле postprocessing"
        for field in ["needed", "confidence", "reason", "explanation"]:
            if field not in postproc:
                return False, f"postprocessing не содержит поля '{field}'"

        # === 4. Проверка validation ===
        validation = decision.get("validation")
        if not validation:
            return False, "Отсутствует поле validation"
        for field in ["needed", "confidence", "reason", "explanation"]:
            if field not in validation:
                return False, f"validation не содержит поля '{field}'"

        # === 5. Проверка final_decision ===
        final = decision.get("final_decision")
        if not final:
            return False, "Отсутствует поле final_decision"
        if "selected_hypothesis" not in final:
            return False, "final_decision не содержит selected_hypothesis"
        if "explanation" not in final or not isinstance(final["explanation"], str) or len(final["explanation"].strip()) < 10:
            return False, "final_decision.explanation должно содержать человекочитаемое резюме (1–2 предложения)"

        return True, None

    def _create_error_result(
        self,
        message: str,
        prompt: str,
        raw_response: str,
        params: Dict,
        llm_response,
        decision: Optional[Dict] = None
    ) -> AgentResult:
        """
        Создаёт объект AgentResult с ошибкой и полной диагностикой.
        """
        LOG.error("ReasonerAgent ошибка: %s", message)
        LOG.debug("Промпт: %s", prompt)
        LOG.debug("Сырой ответ LLM: %s", raw_response)
        LOG.debug("Извлеченное решение: %s", json.dumps(decision, indent=2) if decision else "Нет")

        return AgentResult.error(
            message=message,
            stage="reasoning",
            input_params=params,
            thinking=getattr(llm_response, "thinking", ""),
            prompt=prompt,
            raw_response=raw_response,
            tokens_used=getattr(llm_response, "tokens_used", None)
        )