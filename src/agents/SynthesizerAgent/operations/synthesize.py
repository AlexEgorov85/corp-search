# src/agents/SynthesizerAgent/operations/synthesize.py
"""
Операция `synthesize` — генерация финального ответа.

Цель: на основе плана и результатов шагов сформировать человекочитаемый ответ.

Параметры (заполняются автоматически в executor_node):
  - question (str): исходный вопрос пользователя
  - plan (dict): структура плана (список подвопросов)
  - step_outputs (dict): результаты всех завершённых шагов {step_id: raw_output}

Логика:
  1. Формирует промпт для LLM.
  2. Вызывает LLM.
  3. Парсит ответ и возвращает структурированный `AgentResult`.

Пример вызова от executor_node:
>>> params = {
...     "question": "Найди книги Пушкина...",
...     "plan": {"subquestions": [...]},
...     "step_outputs": {"q1": [...], "q2": {...}}
... }
>>> result = op.run(params, context={}, agent=agent_instance)
>>> assert result.status == "ok"
>>> assert "final_answer" in result.output
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult

LOG = logging.getLogger(__name__)


class Operation(BaseOperation):
    """
    Операция синтеза финального ответа.
    """

    # Тип операции — прямой вызов
    kind = OperationKind.DIRECT

    # Описание операции
    description = (
        "На основании outputs шагов и метаданных плана формирует итоговый ответ "
        "и вспомогательные данные (evidence)."
    )

    # Схема входных параметров
    params_schema = {
        "question": {"type": "string", "required": True},
        "plan": {"type": "object", "required": True},
        "step_outputs": {"type": "object", "required": True},
    }

    # Схема выходных данных
    outputs_schema = {
        "type": "object",
        "properties": {
            "final_answer": "string",
            "evidence": "object"
        }
    }

    def _build_prompt(self, question: str, plan: Any, step_outputs: Dict[str, Any]) -> str:
        """
        Формирует промпт для LLM.

        Args:
            question (str): исходный вопрос
            plan (Any): план выполнения (ожидается Plan или dict)
            step_outputs (dict): результаты шагов

        Returns:
            str: готовый промпт
        """
        # 🔑 Преобразуем Plan в dict
        if hasattr(plan, "model_dump"):
            plan_dict = plan.model_dump()
        elif hasattr(plan, "dict"):
            plan_dict = plan.dict()
        else:
            plan_dict = plan

        plan_text = json.dumps(plan_dict, ensure_ascii=False, indent=2)
        
        try:
            outputs_text = json.dumps(step_outputs, ensure_ascii=False, indent=2)
        except Exception:
            outputs_text = str(step_outputs)

        return f"""Ты — эксперт по синтезу ответов.
    На основе плана и результатов шагов сформируй **финальный ответ** на вопрос пользователя.

    ### Исходный вопрос
    {question}

    ### План выполнения
    {plan_text}

    ### Результаты шагов
    {outputs_text}

    ### Инструкция
    - Верни **ТОЛЬКО** валидный JSON в формате:
    {{
        "final_answer": "строка — итоговый ответ",
        "evidence": {{}}  // опционально: ссылки на шаги, цитаты и т.д.
    }}
    - Ответ должен быть кратким, точным и основанным **только** на предоставленных данных.
    - Если данных недостаточно — напиши об этом в final_answer.
    - Никаких пояснений вне JSON.
    """

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Безопасный парсинг JSON из ответа LLM."""
        text = text.strip()
        # Убираем fenced block
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
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Не удалось распарсить JSON: {text[:200]}...")

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        """
        Выполняет синтез финального ответа.

        Args:
            params (dict):
                - question (str): исходный вопрос
                - plan (dict): план выполнения
                - step_outputs (dict): результаты шагов
            context (dict): контекст выполнения (не используется напрямую)
            agent: экземпляр SynthesizerAgent (для доступа к LLM)

        Returns:
            AgentResult: результат синтеза
        """
        question = params["question"]
        plan = params["plan"]
        step_outputs = params["step_outputs"]

        if not agent.llm:
            return AgentResult.error("LLM не инициализирована в SynthesizerAgent")

        try:
            prompt = self._build_prompt(question, plan, step_outputs)
            LOG.debug("Synthesizer prompt:\n%s", prompt)

            raw_response = agent.llm.generate(prompt)
            LOG.debug("Synthesizer raw response:\n%s", raw_response)

            parsed = self._parse_llm_response(raw_response)

            final_answer = parsed.get("final_answer", "").strip()
            evidence = parsed.get("evidence", {})

            if not final_answer:
                final_answer = "Не удалось сгенерировать финальный ответ."

            output = {
                "final_answer": final_answer,
                "evidence": evidence
            }

            return AgentResult.ok(
                stage="synthesis",
                output=output,
                summary=f"Успешный синтез ответа для вопроса: {question[:50]}..."
            )

        except Exception as e:
            LOG.exception("Ошибка в SynthesizerAgent.synthesize")
            return AgentResult.error(
                message=f"Ошибка синтеза ответа: {e}",
                stage="synthesis"
            )