# src/agents/PlannerAgent/operations/plan.py

from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from src.agents.PlannerAgent.decomposition import DecompositionPhase


class Operation(BaseOperation):
    kind = OperationKind.DIRECT
    description = "Сгенерировать декомпозицию вопроса на подвопросы."
    params_schema = {
        "question": {"type": "string", "required": True},
        "tool_registry_snapshot": {"type": "object", "required": True}
    }
    outputs_schema = {
        "type": "object",
        "properties": {"subquestions": "array"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        question = params.get("question")
        if not question or not isinstance(question, str):
            return AgentResult.error("Параметр 'question' обязателен и должен быть строкой.")
        
        tool_registry = params.get("tool_registry_snapshot")
        if not tool_registry:
            return AgentResult.error("Параметр 'tool_registry_snapshot' обязателен для генерации плана.")

        def llm_wrapper(messages: list[dict[str, str]]) -> str:
            if agent.llm is None:
                raise RuntimeError("LLM не была инициализирована для агента PlannerAgent.")
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            return agent.llm.generate(prompt)

        decomposition_phase = DecompositionPhase(llm_callable=llm_wrapper, max_retries=3)
        success, decomposition, issues = decomposition_phase.run(question, tool_registry)
        
        if success:
            return AgentResult.ok(
                content="plan_generated",
                structured={"plan": decomposition}
            )
        else:
            error_msg = "Не удалось сгенерировать валидную декомпозицию. Проблемы: " + "; ".join(
                [issue.get("message", "Неизвестная ошибка") for issue in issues]
            )
            return AgentResult.error(error_msg, metadata={"issues": issues})