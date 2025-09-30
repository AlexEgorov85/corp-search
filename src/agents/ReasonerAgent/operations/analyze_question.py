# src/agents/ReasonerAgent/operations/analyze_question.py
from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from src.agents.ReasonerAgent.prompts import get_reasoning_prompt
import json

class Operation(BaseOperation):
    kind = OperationKind.DIRECT
    description = "Анализ подвопроса: извлечение сущностей, выбор инструмента и формирование вызова."
    params_schema = {
        "subquestion": {"type": "object", "required": True},
        "tool_registry_snapshot": {"type": "object", "required": True}
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
        subquestion = params.get("subquestion")
        if not isinstance(subquestion, dict) or "text" not in subquestion:
            return AgentResult.error("Параметр 'subquestion' должен быть словарём с полем 'text'")
        tool_registry = params.get("tool_registry_snapshot")
        if not tool_registry:
            return AgentResult.error("Параметр 'tool_registry_snapshot' обязателен")
        if agent.llm is None:
            return AgentResult.error("LLM не инициализирована")

        goal = subquestion["text"]
        prompt = get_reasoning_prompt(goal, tool_registry)
        try:
            raw = agent.llm.generate(prompt)
            parsed = json.loads(raw)
        except Exception as e:
            return AgentResult.error(f"Ошибка в analyze_question: {e}")

        entities = parsed.get("entities", [])
        original_params = parsed.get("params", {})

        # === Проверяем, нужна ли валидация ===
        for entity in entities:
            entity_type = entity.get("type")
            normalized = entity.get("normalized")
            for tool_name, tool_meta in tool_registry.items():
                for op_name, op_meta in tool_meta.get("operations", {}).items():
                    if op_meta.get("validates_entity_type") == entity_type:
                        # Возвращаем ВЫЗОВ ВАЛИДАЦИИ через executor
                        return AgentResult.ok(
                            structured={
                                "action": "call_tool",
                                "tool": tool_name,
                                "operation": op_name,
                                "params": {"candidates": [normalized]},
                                "next_stage": "process_validation"
                            }
                        )

        # Если валидация не нужна — вызываем основной инструмент
        return AgentResult.ok(
            structured={
                "action": "call_tool",
                "tool": parsed["selected_tool"],
                "operation": parsed["selected_operation"],
                "params": original_params,
                "next_stage": "analyze_data"
            }
        )