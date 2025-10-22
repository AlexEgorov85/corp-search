# src/agents/PlannerAgent/decomposition.py
"""
Фаза декомпозиции вопроса на подвопросы с поддержкой нового формата LLMRequest.
Особенности реализации:
- Использует только новый формат запроса LLMRequest
- Поддерживает несколько попыток генерации при неудачной валидации
- Сохраняет полную диагностику каждого запроса для отладки
- Обрабатывает ответы с тегами рассуждений в специфичных форматах
Основной метод: run() - запускает полный цикл декомпозиции
"""
from typing import Any, Dict, List, Tuple
from src.agents.PlannerAgent.prompt import (
    get_decomposition_system_prompt,
    get_decomposition_user_prompt
)
import logging
from src.services.llm_service.model.request import LLMMessage, LLMRequest
import json
LOG = logging.getLogger(__name__)


class DecompositionPhase:
    """Фаза декомпозиции вопроса на подвопросы."""

    def __init__(self, llm, max_retries: int = 3):
        """Инициализирует фазу декомпозиции."""
        self.llm = llm
        self.max_retries = max_retries

    def run(self, params: Dict[str, Any]) -> Tuple[bool, Any, str, Dict[str, Any]]:
        """Запускает декомпозицию с обработкой ответов через LLMRequest."""
        question = params.get("question")
        tool_registry = params.get("tool_registry_snapshot", {})
        feedback = ""
        last_diagnostics = {
            "prompt": None,
            "raw_response": None,
            "thinking": None,
            "tokens_used": None,
            "json_answer": None
        }

        for attempt in range(1, self.max_retries + 1):
            # Формируем запрос
            request = self._build_request(question, tool_registry, feedback)
            # Сохраняем строковое представление промпта для логирования
            try:
                prompt_str = request.model_dump_json()
            except Exception:
                prompt_str = str(request)
            last_diagnostics["prompt"] = prompt_str

            try:
                # Генерируем ответ через LLMRequest
                _, response = self.llm.generate_with_request(request)
                # Сохраняем диагностические данные
                last_diagnostics.update({
                    "raw_response": response.raw_text,
                    "thinking": response.thinking,
                    "tokens_used": response.tokens_used,
                    "json_answer": response.json_answer
                })

                # Попытка извлечь JSON: сначала из json_answer, потом из answer
                decomposition = response.json_answer
                if not decomposition:
                    try:
                        decomposition = json.loads(response.answer)
                    except (json.JSONDecodeError, TypeError):
                        decomposition = None

                if decomposition and self._validate_decomposition_structure(decomposition):
                    return True, decomposition, feedback, last_diagnostics
                else:
                    feedback = "Ответ LLM содержит некорректную структуру декомпозиции или не является JSON"
            except Exception as e:
                LOG.exception("Ошибка вызова LLM (попытка %d): %s", attempt, e)
                feedback = f"Ошибка вызова LLM: {str(e)}"

            LOG.warning("Попытка %d: %s. Повторная попытка...", attempt, feedback[:100])

        return False, None, feedback, last_diagnostics


    def _build_request(self, question: str, tool_registry: dict, feedback: str) -> LLMRequest:
        system_content = get_decomposition_system_prompt()
        user_content = get_decomposition_user_prompt(question, tool_registry, feedback)
        llm_messages = [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=user_content)
        ]
        # === Читаем параметры из конфигурации агента ===
        config = self.agent.config if hasattr(self, 'agent') else {}
        return LLMRequest(
            messages=llm_messages,
            temperature=config.get("LLM_TEMPERATURE", 0.3),
            max_tokens=config.get("LLM_MAX_TOKENS", 2048),
            top_p=config.get("LLM_TOP_P", 0.9)
        )

    def _validate_decomposition_structure(self, decomposition: Dict[str, Any]) -> bool:
        """Валидирует структуру декомпозиции."""
        required_top = ["reasoning", "planning", "subquestions", "final_decision"]
        if not isinstance(decomposition, dict):
            return False
        for field in required_top:
            if field not in decomposition:
                return False

        # Проверка reasoning (5 вопросов P1–P5)
        reasoning = decomposition["reasoning"]
        if not isinstance(reasoning, list) or len(reasoning) != 5:
            return False
        for i in range(5):
            if not isinstance(reasoning[i], str) or not reasoning[i].startswith(f"P{i+1}:"):
                return False

        # Проверка planning
        planning = decomposition["planning"]
        if not isinstance(planning, dict):
            return False
        for field in ["needed", "confidence", "reason", "explanation"]:
            if field not in planning:
                return False

        # Проверка subquestions
        subquestions = decomposition["subquestions"]
        if not isinstance(subquestions, list):
            return False
        for sq in subquestions:
            if not isinstance(sq, dict):
                return False
            for field in ["id", "text", "depends_on", "confidence", "reason", "explanation"]:
                if field not in sq:
                    return False
            if not isinstance(sq["depends_on"], list):
                return False

        # Проверка final_decision
        final = decomposition["final_decision"]
        if not isinstance(final, dict) or "explanation" not in final:
            return False

        return True