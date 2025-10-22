# src/agents/StepResultRelayAgent/operations/relay_step_result.py
from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult

class Operation(BaseOperation):
    kind = OperationKind.DIRECT
    description = "Возвращает результат выполнения указанного шага из контекста."
    params_schema = {
        "source_step_id": {"type": "string", "required": True}
    }
    outputs_schema = {"type": "any"}

    def run(self, params, context, agent):
        source_step_id = params.get("source_step_id")
        if not source_step_id:
            return AgentResult.error("Требуется параметр source_step_id")

        # Получаем raw_output из контекста
        step_outputs = context.get("step_outputs", {})
        result = step_outputs.get(source_step_id)

        if result is None:
            return AgentResult.error(f"Результат для шага {source_step_id} не найден")

        return AgentResult.ok(
            stage="data_fetch",
            output=result,
            summary=f"Результат шага {source_step_id} успешно передан"
        )