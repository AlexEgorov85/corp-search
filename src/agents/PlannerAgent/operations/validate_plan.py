# src/agents/PlannerAgent/operations/validate_plan.py

from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from src.agents.PlannerAgent.decomposition_rules import validate_decomposition


class Operation(BaseOperation):
    kind = OperationKind.VALIDATION
    description = "Проверить декомпозицию на корректность."
    params_schema = {
        "plan": {"type": "object", "required": True},
        "tool_registry_snapshot": {"type": "object", "required": False}
    }
    outputs_schema = {
        "type": "object",
        "properties": {"ok": "boolean", "issues": "array"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        plan = params.get("plan")
        if not plan or not isinstance(plan, dict):
            return AgentResult.error("Параметр 'plan' обязателен и должен быть словарём.")
        
        tool_registry = params.get("tool_registry_snapshot") or {}
        is_valid, issues = validate_decomposition(plan, tool_registry)
        result = {"ok": is_valid, "issues": issues}
        
        if is_valid:
            return AgentResult.ok(content="plan_validated", structured=result)
        else:
            return AgentResult.ok(content="plan_validation_failed", structured=result)