# src/agents/ReasonerAgent/operations/validate_result.py

from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from src.agents.ReasonerAgent.prompts import get_validate_result_prompt
import json

class Operation(BaseOperation):
    kind = OperationKind.VALIDATION
    description = "Валидация результата анализа: решает ли он исходный подвопрос."
    params_schema = {
        "subquestion_text": {"type": "string", "required": True},
        "analysis_result": {"type": "any", "required": True}
    }
    outputs_schema = {
        "type": "object",
        "properties": {
            "action": "string",
            "answer": "any",
            "finalized": "boolean"
        }
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        subquestion_text = params.get("subquestion_text", "")
        analysis_result = params.get("analysis_result")
        if analysis_result is None:
            return AgentResult.error("Нет результата анализа для валидации")

        if agent.llm is None:
            return AgentResult.error("LLM не инициализирована")

        prompt = get_validate_result_prompt(subquestion_text, analysis_result)
        try:
            raw = agent.llm.generate(prompt)
            parsed = json.loads(raw)
            if parsed.get("is_solved"):
                return AgentResult.ok(
                    content="final_answer",
                    structured={
                        "action": "final_answer",
                        "answer": analysis_result,
                        "finalized": True
                    }
                )
            else:
                return AgentResult.ok(
                    content="retry_analysis",
                    structured={"action": "retry_analysis"}
                )
        except Exception as e:
            return AgentResult.error(f"Ошибка в validate_result: {e}")