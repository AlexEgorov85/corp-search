# src/agents/BooksLibraryAgent/operations/get_book_chapters.py
"""
Операция: главы книги.

Используется для запросов вида:
- "Верни главы книги 'Евгений Онегин'"
- "Содержание книги с id=123"

Возвращает список глав с текстом и заголовками.
"""
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult
from sqlalchemy import text

class Operation(BaseOperation):
    kind = OperationKind.DIRECT
    description = "Вернуть список глав/фрагментов для заданного book_id."
    params_schema = {
        "book_id": {"type": "integer", "required": True},
        "max_fragments": {"type": "integer", "required": False}
    }
    outputs_schema = {
        "type": "array",
        "items": {"chapter_id": "integer", "title": "string", "text": "string"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        if not hasattr(agent, 'engine') or agent.engine is None:
            return AgentResult.error("База данных недоступна")

        book_id = params["book_id"]
        limit = min(int(params.get("max_fragments", 100)), 1000)
        sql = """
        SELECT chapter_id, title, text
        FROM chapters
        WHERE book_id = :book_id
        ORDER BY chapter_order
        LIMIT :limit
        """
        try:
            with agent.engine.connect() as conn:
                res = conn.execute(text(sql), {"book_id": book_id, "limit": limit})
                rows = [dict(row) for row in res.mappings().all()]
            return AgentResult.ok(structured=rows)
        except Exception as e:
            return AgentResult.error(f"Ошибка выполнения SQL: {e}")