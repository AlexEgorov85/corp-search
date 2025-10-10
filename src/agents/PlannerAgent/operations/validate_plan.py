# src/agents/PlannerAgent/operations/validate_plan.py
"""
Операция 'validate_plan' для PlannerAgent.

Эта операция проверяет уже сгенерированный план на соответствие правилам
(например, отсутствие циклических зависимостей). Она не использует LLM,
а выполняет чистую программную валидацию.

Согласно новой архитектуре, операция:
1. Принимает параметры через `params`.
2. Возвращает результат строго в виде `AgentResult`.
3. Использует `AgentResult.ok()` даже для случая проваленной валидации,
   потому что сама операция выполнилась успешно, просто результат валидации — отрицательный.
4. Заполняет поля `stage`, `output`, `summary` для полной прозрачности.
"""

from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult
from src.agents.PlannerAgent.decomposition_rules import validate_decomposition


class Operation(BaseOperation):
    """
    Операция валидации плана.
    """
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
        """
        Основной метод выполнения операции.

        Args:
            params (dict): Параметры операции.
                - plan: План для валидации (dict).
                - tool_registry_snapshot: Опциональный снимок реестра (dict).
            context (dict): Контекст выполнения (не используется).
            agent: Экземпляр родительского агента.

        Returns:
            AgentResult: Результат выполнения операции.
        """
        # --- Валидация входных параметров ---
        plan = params.get("plan")
        if not plan or not isinstance(plan, dict):
            return AgentResult.error(
                message="Параметр 'plan' обязателен и должен быть словарём.",
                stage="plan_validation",
                input_params=params
            )

        tool_registry = params.get("tool_registry_snapshot") or {}

        # --- Выполнение валидации ---
        is_valid, issues = validate_decomposition(plan, tool_registry)
        # Формируем структурированный результат валидации
        validation_result = {"ok": is_valid, "issues": issues}

        # Важно: операция сама по себе завершилась успешно (ok),
        # даже если валидация плана провалилась (is_valid=False).
        # Это позволяет Reasoner корректно обработать результат.
        return AgentResult.ok(
            stage="plan_validation",
            output=validation_result,
            summary=f"Валидация плана завершена. Результат: {'успешно' if is_valid else 'ошибки найдены'}.",
            input_params=params
        )