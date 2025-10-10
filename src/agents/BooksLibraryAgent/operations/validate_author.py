# src/agents/BooksLibraryAgent/operations/validate_author.py
from src.agents.operations_base import BaseOperation, OperationKind
from sqlalchemy import text

from src.model.agent_result import AgentResult

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
        candidates = params.get("candidates", [])
        if not hasattr(agent, 'engine') or agent.engine is None:
            return AgentResult.error(
                message="База данных недоступна",
                stage="entity_validation",
                entity_type="author",
                input_params=params,
                summary="Ошибка: база данных недоступна"
            )
        if not candidates:
            return AgentResult.ok(
                stage="entity_validation",
                entity_type="author",
                input_params=params,
                output={"validated": []},
                summary="Проведена валидация сущности 'автор' для пустого списка кандидатов"
            )
        placeholders = ", ".join([f":cand{i}" for i in range(len(candidates))])
        values = {f"cand{i}": cand for i, cand in enumerate(candidates)}
        sql = f"""
        SELECT DISTINCT last_name 
        FROM "Lib".authors
        WHERE last_name ILIKE ANY(ARRAY[{placeholders}])
        """
        try:
            with agent.engine.connect() as conn:
                res = conn.execute(text(sql), values)
                authors = [{"name": row[0]} for row in res.fetchall()]
            return AgentResult.ok(
                stage="entity_validation",
                entity_type="author",
                input_params=params,
                output={"validated": authors},
                summary=f"Проведена валидация сущности 'автор' для кандидатов: {candidates}"
            )
        except Exception as e:
            return AgentResult.error(
                message=f"Ошибка валидации автора: {e}",
                stage="entity_validation",
                entity_type="author",
                input_params=params,
                summary="Ошибка при выполнении SQL-запроса валидации автора"
            )