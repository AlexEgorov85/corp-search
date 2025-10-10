# src/agents/DataAnalysisAgent/operations/analyze.py
"""
Операция `analyze` — универсальный анализ данных.

Цель: проанализировать входные данные (`raw_output`) в контексте подвопроса (`subquestion_text`)
и вернуть структурированный результат + human-readable summary.

Параметры (заполняются автоматически в executor_node):
  - subquestion_text (str): текст подвопроса (например, "Какие книги написал Пушкин?")
  - raw_output (any): результат предыдущего шага (например, список книг)

Логика анализа:
  1. Определяет тип данных: таблица, текстовый список, скаляр, пустой.
  2. Применяет соответствующую стратегию:
      - Таблица → агрегация (count, примеры полей)
      - Текст → суммаризация через LLM (если данных не слишком много)
      - Скаляр → валидация и пояснение
  3. Генерирует human-readable summary через LLM (опционально).
  4. Возвращает структурированный AgentResult.

Пример вызова от executor_node:
>>> params = {
...     "subquestion_text": "Какие книги написал Пушкин?",
...     "raw_output": [{"title": "Евгений Онегин", "year": 1833}]
... }
>>> result = op.run(params, context={}, agent=agent_instance)
>>> assert result.status == "ok"
>>> assert "metrics" in result.output
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
    Универсальная операция анализа данных.
    """

    # Тип операции — прямой вызов
    kind = OperationKind.DIRECT

    # Описание операции
    description = "Автоматически анализирует входные данные и возвращает структурированный результат."

    # Схема входных параметров (обязательна для документации и валидации)
    params_schema = {
        "subquestion_text": {"type": "string", "required": True},
        "raw_output": {"type": "any", "required": True},
    }

    # Схема выходных данных
    outputs_schema = {
        "type": "object",
        "properties": {
            "metrics": "object",      # Структурированные метрики
            "summary": "string"       # Человекочитаемое резюме
        }
    }

    def _detect_data_type(self, data: Any) -> str:
        """
        Определяет тип входных данных.

        Возвращает:
            "empty"       — данные отсутствуют или пусты
            "table"       — список словарей (табличные данные)
            "text_list"   — список строк
            "scalar"      — число, строка, булево значение
            "unknown"     — неподдерживаемый тип

        Примеры:
            [] → "empty"
            [{"title": "..."}] → "table"
            ["текст1", "текст2"] → "text_list"
            "Евгений Онегин" → "scalar"
        """
        if data is None:
            return "empty"
        if isinstance(data, list):
            if len(data) == 0:
                return "empty"
            first = data[0]
            if isinstance(first, str):
                return "text_list"
            elif isinstance(first, dict):
                return "table"
        elif isinstance(data, (int, float, str, bool)):
            return "scalar"
        return "unknown"

    def _analyze_table(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Анализ табличных данных.

        Возвращает:
            {
                "row_count": int,
                "columns": List[str],
                "sample": List[Dict]  # первые 3 строки
            }

        Пример:
            [{"title": "Евгений Онегин", "year": 1833}]
            → {"row_count": 1, "columns": ["title", "year"], "sample": [...]}
        """
        return {
            "row_count": len(data),
            "columns": list(data[0].keys()) if data else [],
            "sample": data[:3]
        }

    def _analyze_text_list(self, data: List[str]) -> Dict[str, Any]:
        """
        Анализ списка текстов.

        Возвращает:
            {
                "text_count": int,
                "sample": List[str]  # первые 2 текста
            }
        """
        return {
            "text_count": len(data),
            "sample": data[:2]
        }

    def _synthesize_summary(self, subquestion: str, metrics: Dict[str, Any]) -> str:
        """
        Генерирует human-readable summary через LLM.

        Аргументы:
            subquestion (str): подвопрос
            metrics (dict): структурированные метрики

        Возвращает:
            str: краткое резюме на русском языке

        Пример промпта:
            "Подвопрос: Какие книги написал Пушкин?
             Метрики: {'row_count': 5, 'columns': ['title', 'year']}
             Кратко опиши результат."
        """
        if not self.agent.llm:
            return f"Проанализированы данные для подвопроса: {subquestion}"

        prompt = (
            f"Подвопрос: {subquestion}\n"
            f"Метрики: {json.dumps(metrics, ensure_ascii=False)}\n"
            "Кратко опиши результат на русском языке (1-2 предложения)."
        )
        try:
            raw = self.agent.llm.generate(prompt)
            return raw.strip()
        except Exception as e:
            LOG.warning("Ошибка генерации summary через LLM: %s", e)
            return f"Результат анализа: {metrics}"

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        """
        Выполняет универсальный анализ данных.

        Args:
            params (dict):
                - subquestion_text (str): текст подвопроса
                - raw_output (any): данные для анализа
            context (dict): контекст выполнения (не используется напрямую)
            agent: экземпляр DataAnalysisAgent (для доступа к LLM)

        Returns:
            AgentResult: структурированный результат анализа
        """
        subquestion = params["subquestion_text"]
        raw_output = params["raw_output"]

        try:
            # Шаг 1: Определение типа данных
            data_type = self._detect_data_type(raw_output)

            # Шаг 2: Анализ в зависимости от типа
            if data_type == "empty":
                metrics = {"error": "Данные отсутствуют или пусты"}
            elif data_type == "table":
                metrics = self._analyze_table(raw_output)
            elif data_type == "text_list":
                metrics = self._analyze_text_list(raw_output)
            elif data_type == "scalar":
                metrics = {"value": raw_output}
            else:
                metrics = {"raw": str(raw_output)[:500]}

            # Шаг 3: Генерация human-readable summary
            summary = self._synthesize_summary(subquestion, metrics)

            # Шаг 4: Формирование результата
            output = {
                "metrics": metrics,
                "summary": summary
            }

            return AgentResult.ok(
                stage="data_analysis",
                output=output,
                summary=f"Успешный анализ данных для подвопроса: {subquestion}"
            )

        except Exception as e:
            LOG.exception("Ошибка в DataAnalysisAgent.analyze")
            return AgentResult.error(
                message=f"Ошибка анализа данных: {e}",
                stage="data_analysis"
            )