# src/agents/ReasonerAgent/stages.py
from typing import Any, Dict, List
from src.services.results.agent_result import AgentResult
from src.services.llm_service.utils import strip_code_fences
import json
import logging
LOG = logging.getLogger(__name__)

def analyze_question(subquestion: Dict[str, Any], tool_registry: Dict[str, Any], llm_callable) -> AgentResult:
    from .prompts import get_reasoning_prompt
    goal = subquestion["text"]
    prompt = get_reasoning_prompt(goal, tool_registry)
    try:
        raw = llm_callable([{"role": "user", "content": prompt}])
        parsed = json.loads(strip_code_fences(raw))
        return AgentResult.ok(
            content="analyze_complete",
            structured={
                "action": "store_analysis",
                "entities": parsed.get("entities", []),
                "selected_tool": parsed["selected_tool"],
                "selected_operation": parsed["selected_operation"],
                "params": parsed["params"]
            }
        )
    except Exception as e:
        LOG.exception("Ошибка в analyze_question")
        return AgentResult.error(f"Ошибка в analyze_question: {e}")


def validate_entities(subquestion: Dict[str, Any], entities: List[Dict], tool_registry: Dict[str, Any]) -> AgentResult:
    """Универсальная валидация сущностей через подходящий инструмент из реестра."""
    for entity in entities:
        entity_type = entity["type"]
        normalized_value = entity["normalized"]
        
        # Ищем инструмент и операцию, которые могут валидировать этот тип сущности
        for tool_name, tool_meta in tool_registry.items():
            for op_name, op_meta in tool_meta.get("operations", {}).items():
                if op_meta.get("validates_entity_type") == entity_type:
                    # Нашли подходящий инструмент!
                    return AgentResult.ok(
                        content="call_tool",
                        structured={
                            "action": "call_tool",
                            "tool": tool_name,
                            "operation": op_name,
                            "params": {"candidates": [normalized_value]},
                            "next_stage": "PROCESS_VALIDATION"
                        }
                    )
    
    # Если подходящего инструмента нет — пропускаем валидацию
    return AgentResult.ok(
        content="validation_skipped",
        structured={"action": "proceed_to_execution"}
    )

def process_validation_result(subquestion: Dict[str, Any], validation_result: Any, original_params: Dict) -> AgentResult:
    """Обработка результата валидации и формирование финальных параметров."""
    try:
        validated_authors = validation_result.get("authors", [])
        if validated_authors:
            # Берём первое каноническое имя
            canonical_name = validated_authors[0].get("name", original_params.get("author", ""))
            # Обновляем параметры для основного вызова
            final_params = original_params.copy()
            final_params["author"] = canonical_name
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
        else:
            # Если валидация не дала результата — используем оригинальные параметры
            return AgentResult.ok(
                content="fallback_execution",
                structured={
                    "action": "call_tool",
                    "tool": "BooksLibraryAgent",
                    "operation": "list_books",
                    "params": original_params,
                    "next_stage": "EXECUTE_TOOL"
                }
            )
    except Exception as e:
        LOG.exception("Ошибка в process_validation_result")
        return AgentResult.error(f"Ошибка обработки валидации: {e}")

def analyze_data(subquestion: Dict[str, Any], raw_data: Any, llm_callable) -> AgentResult:
    from .prompts import get_analyze_data_prompt
    goal = subquestion["text"]
    prompt = get_analyze_data_prompt(goal, raw_data)
    try:
        raw = llm_callable([{"role": "user", "content": prompt}])
        parsed = json.loads(strip_code_fences(raw))
        analysis = parsed.get("analysis", raw_data)
        return validate_result(subquestion, analysis, llm_callable)
    except Exception as e:
        LOG.exception("Ошибка в analyze_data")
        return AgentResult.error(f"Ошибка в analyze_ {e}")

def validate_result(subquestion: Dict[str, Any], analysis_result: Any, llm_callable) -> AgentResult:
    from .prompts import get_validate_result_prompt
    goal = subquestion["text"]
    prompt = get_validate_result_prompt(goal, analysis_result)
    try:
        raw = llm_callable([{"role": "user", "content": prompt}])
        parsed = json.loads(strip_code_fences(raw))
        is_solved = parsed.get("is_solved", False)
        if is_solved:
            return finalize(subquestion, analysis_result)
        else:
            return AgentResult.ok(
                content="retry_analysis",
                structured={"action": "retry_analysis"}
            )
    except Exception as e:
        LOG.exception("Ошибка в validate_result")
        return AgentResult.error(f"Ошибка в validate_result: {e}")

def finalize(subquestion: Dict[str, Any], analysis_result: Any) -> AgentResult:
    return AgentResult.ok(
        content="final_answer",
        structured={
            "action": "final_answer",
            "answer": analysis_result,
            "finalized": True
        }
    )