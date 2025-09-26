# src/agents/ReasonerAgent/core.py
from __future__ import annotations
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent
from src.services.results.agent_result import AgentResult
from .stages import (
    analyze_question,
    validate_entities,
    process_validation_result,
    analyze_data,
    validate_result,
    finalize
)
import logging
LOG = logging.getLogger(__name__)

class ReasonerAgent(BaseAgent):
    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__(descriptor, config or {})
        self._call_llm = self.config.get("llm_callable")
        self._tool_registry_snapshot = self.config.get("tool_registry_snapshot", {})

    def _run_direct_operation(self, op_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        if op_name == "decide":
            return self._op_decide(params, context)
        else:
            return AgentResult.error(f"Неизвестная операция: {op_name}")

    def _build_final_params(self, original_params: Dict[str, Any], validation_result: Any) -> Dict[str, Any]:
        """Формирует финальные параметры для основного вызова на основе результата валидации."""
        validated_authors = validation_result.get("authors", [])
        if validated_authors:
            canonical_name = validated_authors[0].get("name", original_params.get("author", ""))
            final_params = original_params.copy()
            final_params["author"] = canonical_name
            return final_params
        return original_params  # Fallback

    def _op_decide(self, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        subquestion = params.get("subquestion")
        if not isinstance(subquestion, dict) or "text" not in subquestion:
            return AgentResult.error("Параметр 'subquestion' должен быть словарём с полем 'text'")

        subq_id = subquestion["id"]
        step_outputs = params.get("step_outputs", {})
        current_stage = params.get("current_stage", "ANALYZE_QUESTION")
        tool_registry = params.get("tool_registry_snapshot") or self._tool_registry_snapshot

        # --- Определение текущей стадии ---
        if step_outputs.get(subq_id, {}).get("finalized"):
            return AgentResult.ok(content="skip", structured={"action": "skip"})

        raw_key = f"{subq_id}_raw"
        if raw_key in step_outputs and "analysis" not in step_outputs.get(subq_id, {}):
            current_stage = "ANALYZE_DATA"

        validated_key = f"{subq_id}_validated"
        analysis_key = f"{subq_id}_analysis"
        if analysis_key in step_outputs and validated_key not in step_outputs:
            current_stage = "VALIDATE_ENTITIES"
        elif validated_key in step_outputs and "final_params" not in step_outputs.get(subq_id, {}):
            current_stage = "PROCESS_VALIDATION"

        LOG.info(f"ReasonerAgent: выполняет стадию {current_stage} для подвопроса {subq_id}")
        try:
            if current_stage == "ANALYZE_QUESTION":
                return analyze_question(subquestion, tool_registry, self._call_llm)
            elif current_stage == "VALIDATE_ENTITIES":
                entities = step_outputs[analysis_key]["entities"]
                return validate_entities(subquestion, entities, tool_registry)
            elif current_stage == "PROCESS_VALIDATION":
                validation_result = step_outputs[validated_key]
                original_params = step_outputs[analysis_key]["params"]
                final_params = self._build_final_params(original_params, validation_result)
                return AgentResult.ok(
                    content="proceed_to_execution",
                    structured={
                        "action": "call_tool",
                        "tool": "BooksLibraryAgent",
                        "operation": "list_books",
                        "params": final_params,
                        "next_stage": "EXECUTE_TOOL"
                    }
                )
            elif current_stage == "ANALYZE_DATA":
                raw_data = step_outputs.get(raw_key)
                if raw_data is None:
                    return AgentResult.error("Нет сырых данных для анализа")
                return analyze_data(subquestion, raw_data, self._call_llm)
            elif current_stage == "VALIDATE_RESULT":
                analysis_result = step_outputs.get(subq_id, {}).get("analysis")
                if analysis_result is None:
                    return AgentResult.error("Нет результата анализа для валидации")
                return validate_result(subquestion, analysis_result, self._call_llm)
            elif current_stage == "FINALIZE":
                analysis_result = step_outputs.get(subq_id, {}).get("analysis")
                if analysis_result is None:
                    return AgentResult.error("Нет результата для финализации")
                return finalize(subquestion, analysis_result)
            else:
                return AgentResult.error(f"Неизвестная стадия: {current_stage}")
        except Exception as e:
            LOG.exception(f"Ошибка выполнения стадии {current_stage}")
            return AgentResult.error(f"Ошибка выполнения стадии {current_stage}: {e}")