# src/agents/ReasonerAgent/operations/process_validation.py
from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult

class Operation(BaseOperation):
    kind = OperationKind.SEMANTIC
    description = "Обработка результата валидации и формирование вызова основного инструмента."
    params_schema = {
        "validation_result": {"type": "object", "required": True},
        "original_params": {"type": "object", "required": True}
    }
    outputs_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "tool": {"type": "string"},
            "operation": {"type": "string"},
            "params": {"type": "object"},
            "next_stage": {"type": "string"}
        }
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        validation_result = params.get("validation_result", {})
        original_params = params.get("original_params", {})

        # Нормализуем автора (пример для сущности "author")
        validated_authors = validation_result.get("authors", [])
        if validated_authors:
            canonical = validated_authors[0].get("name", original_params.get("author", ""))
            final_params = {**original_params, "author": canonical}
        else:
            final_params = original_params

        return AgentResult.ok(
            content="proceed_to_execution",
            structured={
                "action": "call_tool",
                "tool": "BooksLibraryAgent",
                "operation": "list_books",
                "params": final_params,
                "next_stage": "analyze_data"
            }
        )