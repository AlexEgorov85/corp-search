# src/agents/BooksLibraryAgent/operations/get_book_chapters.py
"""
Операция: главы книги.
Используется для запросов вида:
- "Верни главы книги 'Евгений Онегин'"
- "Содержание книги с id=5"

Возвращает список глав с полями:
- chapter_id (int)
- chapter_number (int)
- chapter_text (str)

Соответствует новой архитектуре:
- Использует `AgentResult.ok()` / `AgentResult.error()`
- Заполняет семантические поля: `stage`, `input_params`, `summary`
- Возвращает данные в `output` (не в `structured`)
- Совместима с `GraphContext.record_tool_execution_result()`
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from sqlalchemy import text
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult


class Operation(BaseOperation):
    """
    Операция получения глав книги по book_id.
    """

    # Тип операции — прямой вызов
    kind = OperationKind.DIRECT

    # Описание операции
    description = "Вернуть список глав/фрагментов для заданного book_id."

    # Схема входных параметров
    params_schema = {
        "book_id": {"type": "integer", "required": True},
        "max_fragments": {"type": "integer", "required": False}
    }

    # Схема выходных данных
    outputs_schema = {
        "type": "array",
        "items": {
            "chapter_id": "integer",
            "chapter_number": "integer",
            "chapter_text": "string"
        }
    }

    def run(self, params: Dict[str, Any], context: Dict[str, Any], agent) -> AgentResult:
        """
        Выполняет запрос к БД и возвращает список глав.

        Args:
            params (dict):
                - book_id (int): ID книги
                - max_fragments (int, optional): максимальное число глав (по умолчанию 100)
            context (dict): контекст выполнения (не используется напрямую)
            agent: экземпляр BooksLibraryAgent

        Returns:
            AgentResult: результат операции
        """
        # === Валидация параметров ===
        book_id = params.get("book_id")
        if not isinstance(book_id, int) or book_id <= 0:
            return AgentResult.error(
                message="Параметр 'book_id' обязателен и должен быть положительным целым числом.",
                stage="data_fetch",
                input_params=params,
                summary="Ошибка валидации параметров: book_id"
            )

        max_fragments = min(int(params.get("max_fragments", 100)), 1000)

        # === Проверка подключения к БД ===
        if not hasattr(agent, 'engine') or agent.engine is None:
            return AgentResult.error(
                message="База данных недоступна",
                stage="data_fetch",
                input_params=params,
                summary="Ошибка: отсутствует подключение к БД"
            )

        # === Выполнение SQL-запроса ===
        sql = """
        SELECT c.chapter_id, c.chapter_number, c.chapter_text 
        FROM "Lib".chapters c
        WHERE book_id = :book_id
        ORDER BY c.chapter_number
        LIMIT :limit
        """
        try:
            with agent.engine.connect() as conn:
                res = conn.execute(text(sql), {"book_id": book_id, "limit": max_fragments})
                rows = [dict(row) for row in res.mappings().all()]

            # === Успешный результат ===
            return AgentResult.ok(
                stage="data_fetch",
                input_params=params,
                output=rows,
                summary=f"Успешно получены главы для книги с ID {book_id} (всего: {len(rows)})."
            )

        except Exception as e:
            # === Ошибка выполнения ===
            return AgentResult.error(
                message=f"Ошибка выполнения SQL: {e}",
                stage="data_fetch",
                input_params=params,
                summary=f"Не удалось получить главы для книги с ID {book_id}."
            )