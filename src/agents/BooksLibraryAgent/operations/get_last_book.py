# src/agents/BooksLibraryAgent/operations/get_last_book.py
"""
Операция: последняя книга автора.

Используется для запросов вида:
- "Какая последняя книга у Пушкина?"
- "Последняя опубликованная книга автора"

Сортировка: по году (DESC), затем по id (DESC).
"""
from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from sqlalchemy import text

class Operation(BaseOperation):
    kind = OperationKind.DIRECT
    description = "Вернуть метаданные последней книги указанного автора."
    params_schema = {
        "author": {"type": "string", "required": True},
        "tie_break": {"type": "string", "required": False}
    }
    outputs_schema = {
        "type": "object",
        "properties": {"id": "integer", "title": "string", "year": "integer", "author_id": "integer"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        if not hasattr(agent, 'engine') or agent.engine is None:
            return AgentResult.error("База данных недоступна")

        author = params["author"]
        sql = """
        SELECT b.id, b.title, b.year, b.author_id
        FROM books b
        JOIN authors a ON b.author_id = a.id
        WHERE a.name ILIKE :author
        ORDER BY b.year DESC, b.id DESC
        LIMIT 1
        """
        try:
            with agent.engine.connect() as conn:
                res = conn.execute(text(sql), {"author": f"%{author}%"})
                row = res.mappings().first()
            if row:
                return AgentResult.ok(structured=dict(row))
            else:
                return AgentResult.ok(structured=None, content="Книги не найдены")
        except Exception as e:
            return AgentResult.error(f"Ошибка выполнения SQL: {e}")