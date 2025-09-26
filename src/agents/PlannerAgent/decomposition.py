# src/agents/PlannerAgent/phases/decomposition.py

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.agents.PlannerAgent.decomposition_prompt import (
    get_decomposition_system_prompt,
    get_decomposition_user_prompt
)
from src.agents.PlannerAgent.decomposition_rules import validate_decomposition
from src.agents.PlannerAgent.utils import extract_json_from_text

LOG = logging.getLogger(__name__)

class DecompositionPhase:
    def __init__(self, llm_callable, max_retries: int = 3):
        self.llm = llm_callable
        self.max_retries = max_retries

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        # Предполагаем, что llm поддерживает messages
        return self.llm(messages=messages)

    def _prepare_messages(self, question: str, tool_registry: dict, feedback: str = "") -> List[Dict[str, str]]:
        system = get_decomposition_system_prompt()
        user = get_decomposition_user_prompt(question, tool_registry)
        if feedback:
            user += f"\n\nПредыдущая попытка не прошла валидацию. Исправьте:\n{feedback}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    def run(self, question: str, tool_registry: dict) -> Tuple[bool, Any, List[str]]:
        """
        Возвращает:
          - success: bool
          - decomposition: dict или None
          - issues: List[str] (если success=False)
        """
        feedback = ""
        for attempt in range(1, self.max_retries + 1):
            messages = self._prepare_messages(question, tool_registry, feedback)
            try:
                print("=========================================")
                print("Промт для плана:", messages)
                print("=========================================")
                raw = self._call_llm(messages)
            except Exception as e:
                LOG.error("Ошибка вызова LLM на этапе декомпозиции: %s", e)
                return False, None, [f"Ошибка LLM: {e}"]

            json_text = extract_json_from_text(raw)
            if not json_text:
                error = "Не удалось извлечь JSON из ответа LLM"
                LOG.warning("Попытка %d: %s. Ответ: %.200s", attempt, error, raw)
                if attempt == self.max_retries:
                    return False, None, [error]
                feedback = "Верните ТОЛЬКО валидный JSON, без пояснений."
                continue

            try:
                decomposition = json.loads(json_text)
            except json.JSONDecodeError as e:
                error = f"Ошибка парсинга JSON: {e}"
                LOG.warning("Попытка %d: %s", attempt, error)
                if attempt == self.max_retries:
                    return False, None, [error]
                feedback = "Ваш ответ не является валидным JSON. Исправьте синтаксис."
                continue

            # Валидация
            is_valid, issues = validate_decomposition(decomposition, tool_registry)
            if is_valid:
                return True, decomposition, []

            LOG.warning("Попытка %d: валидация не пройдена: %s", attempt, issues)
            if attempt == self.max_retries:
                return False, decomposition, issues

            # Формируем фидбэк для следующей попытки
            feedback_lines = []
            for issue in issues:
                rule_id = issue.get("rule_id", "unknown")
                message = issue.get("message", "Ошибка валидации")
                severity = issue.get("severity", "error").upper()
                feedback_lines.append(f"- [{severity}] [{rule_id}] {message}")
            feedback = "\n".join(feedback_lines)

        return False, None, ["Неизвестная ошибка декомпозиции"]