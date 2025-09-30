# src/agents/ReasonerAgent/operations/analyze_data.py

from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from src.agents.ReasonerAgent.prompts import get_analyze_data_prompt
import json

class Operation(BaseOperation):
    kind = OperationKind.SEMANTIC
    description = "Анализ сырых данных и извлечение структурированного ответа."
    params_schema = {
        "subquestion_text": {"type": "string", "required": True},
        "raw_data": {"type": "any", "required": True}
    }
    outputs_schema = {
        "type": "object",
        "properties": {"analysis": "any"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        subquestion_text = params.get("subquestion_text", "")
        raw_data = params.get("raw_data")
        if raw_data is None:
            return AgentResult.error("Нет сырых данных для анализа")

        if agent.llm is None:
            return AgentResult.error("LLM не инициализирована")

        prompt = get_analyze_data_prompt(subquestion_text, raw_data)
        try:
            raw = agent.llm.generate(prompt)
            parsed = json.loads(raw)
            analysis = parsed.get("analysis", raw_data)
            return AgentResult.ok(
                content="analysis_complete",
                structured={"analysis": analysis}
            )
        except Exception as e:
            return AgentResult.error(f"Ошибка в analyze_data: {e}")