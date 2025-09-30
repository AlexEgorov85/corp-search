# src/agents/BooksLibraryAgent/operations/validate_author.py
"""
Операция: валидация автора.

Используется ReasonerAgent для нормализации сущностей:
- "Пушкн" → "Пушкин, Александр Сергеевич"
- "Достоевский Ф.М." → "Достоевский, Фёдор Михайлович"

Реализует семантический поиск через ILIKE ANY.
"""
from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from sqlalchemy import text

class Operation(BaseOperation):
    kind = OperationKind.VALIDATION
    description = "Валидация автора через семантический поиск."
    params_schema = {
        "candidates": {"type": "array", "items": {"type": "string"}, "required": True}
    }
    outputs_schema = {
        "type": "object",
        "properties": {"authors": "array"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        if not hasattr(agent, 'engine') or agent.engine is None:
            return AgentResult.error("База данных недоступна")

        candidates = params.get("candidates", [])
        if not candidates:
            return AgentResult.ok(structured={"authors": []})

        placeholders = ", ".join([f":cand{i}" for i in range(len(candidates))])
        values = {f"cand{i}": cand for i, cand in enumerate(candidates)}
        sql = f"""
        SELECT DISTINCT name
        FROM authors
        WHERE name ILIKE ANY(ARRAY[{placeholders}])
        """
        try:
            with agent.engine.connect() as conn:
                res = conn.execute(text(sql), values)
                authors = [{"name": row[0]} for row in res.fetchall()]
            return AgentResult.ok(structured={"authors": authors})
        except Exception as e:
            return AgentResult.error(f"Ошибка валидации автора: {e}")