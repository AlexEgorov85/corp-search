# src/agents/BooksLibraryAgent/operations/list_books.py
"""
Операция: список книг по фамилии автора.
Используется для запросов вида:
- "Какие книги написал Пушкин?"
- "Книги Достоевского"
Поддерживает только фильтрацию по фамилии автора (поле authors.last_name).
"""
from src.agents.operations_base import BaseOperation, OperationKind
from sqlalchemy import text
from src.model.agent_result import AgentResult

class Operation(BaseOperation):
    kind = OperationKind.DIRECT
    description = "Вернуть список книг по фамилии автора (author)."
    params_schema = {
        "author": {"type": "string", "required": True},
        "limit": {"type": "integer", "required": False}
    }
    outputs_schema = {
        "type": "array",
        "items": {"book_id": "integer", "title": "string", "publication_date": "date", "author_id": "integer"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        if not hasattr(agent, 'engine') or agent.engine is None:
            return AgentResult.error(
                message="База данных недоступна",
                stage="data_fetch",
                input_params=params,
                summary="Ошибка: отсутствует подключение к БД"
            )

        author_last_name = params["author"]
        limit = min(int(params.get("limit", 100)), 1000)
        sql = """
        SELECT b.id as book_id, b.title, b.publication_date, b.author_id
        FROM "Lib".books b
        JOIN "Lib".authors a ON b.author_id = a.id
        WHERE a.last_name = :last_name
        LIMIT :limit
        """

        try:
            with agent.engine.connect() as conn:
                res = conn.execute(text(sql), {"last_name": author_last_name, "limit": limit})
                rows = [dict(row) for row in res.mappings().all()]

            return AgentResult.ok(
                stage="data_fetch",
                input_params=params,
                output=rows,
                summary=f"Успешно получены книги автора '{author_last_name}' (всего: {len(rows)})."
            )

        except Exception as e:
            return AgentResult.error(
                message=f"Ошибка выполнения SQL: {e}",
                stage="data_fetch",
                input_params=params,
                summary=f"Не удалось получить книги автора '{author_last_name}'."
            )