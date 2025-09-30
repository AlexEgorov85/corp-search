# src/agents/ReasonerAgent/operations/step.py
import logging
from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from typing import Dict, Any

LOG = logging.getLogger(__name__)

class Operation(BaseOperation):
    kind = OperationKind.CONTROL
    description = "Выполнить один микрошаг reasoning'а."
    params_schema = {
        "subquestion": {"type": "object", "required": True},
        "step": {"type": "object", "required": False},
        "execution_state": {"type": "object", "required": False},
        "tool_registry_snapshot": {"type": "object", "required": True},
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        subq = params.get("subquestion", {})
        current_step = params.get("step") or {}
        exec_state = params.get("execution_state") or {}
        tool_registry_snapshot = params.get("tool_registry_snapshot") or {}

        subq_id = subq.get("id")
        subq_text = subq.get("text", "")

        # Определяем следующую стадию из execution_state
        next_stage = (
            exec_state.get("next_stage") or
            current_step.get("next_stage") or
            "analyze_question"
        ).lower()

        LOG.info(f"[{subq_id}] 🧠 Reasoner: выполняет стадию '{next_stage}' для подвопроса: {subq_text}")

        def safe_get(d, *keys, default=None):
            for k in keys:
                if isinstance(d, dict):
                    d = d.get(k, {})
                else:
                    return default
            return d if d != {} else default

        stage_to_op = {
            "analyze_question": (
                "analyze_question",
                {
                    "subquestion": subq,
                    "tool_registry_snapshot": tool_registry_snapshot
                }
            ),
            "process_validation": (
                "process_validation",
                {
                    "validation_result": safe_get(current_step, "validated_params", default={}),
                    "original_params": safe_get(current_step, "analysis", "params", default={})
                }
            ),
            "analyze_data": (
                "analyze_data",
                {
                    "subquestion_text": subq_text,
                    "raw_data": current_step.get("raw_output")
                }
            ),
            "validate_result": (
                "validate_result",
                {
                    "subquestion_text": subq_text,
                    "analysis_result": current_step.get("final_result")
                }
            )
        }

        if next_stage not in stage_to_op:
            LOG.error(f"[{subq_id}] ❌ Reasoner: неизвестная стадия '{next_stage}'")
            return AgentResult.error(f"unknown_stage: {next_stage}")

        op_name, op_params = stage_to_op[next_stage]

        try:
            res = agent.execute_operation(op_name, op_params, context)
        except Exception as e:
            LOG.exception(f"[{subq_id}] 💥 Reasoner: ошибка при вызове операции '{op_name}': {e}")
            return AgentResult.error(f"operation_call_failed: {op_name} -> {e}")

        if res.status != "ok":
            LOG.warning(f"[{subq_id}] ❌ Reasoner: операция '{op_name}' завершилась с ошибкой: {res.error or res.content}")
            return AgentResult.ok(
                structured={
                    "updated_step": {
                        "status": "failed",
                        "error": res.error or str(res.content),
                        "error_stage": next_stage
                    },
                    "decision": {"action": "error", "reason": res.error or str(res.content)},
                    "next_stage": None,
                    "progress": f"stage {next_stage} failed"
                }
            )

        structured = res.structured or {}
        updated_step: Dict[str, Any] = {}

        # Обновляем поля шага
        for field in ("analysis", "validated_params", "raw_output", "final_result", "status", "error", "error_stage"):
            if field in structured:
                updated_step[field] = structured[field]

        reported_next_stage = structured.get("next_stage")
        if reported_next_stage is not None:
            reported_next_stage = str(reported_next_stage).lower()

        decision = None
        if structured.get("action") == "call_tool":
            decision = {
                "action": "call_tool",
                "tool": structured["tool"],
                "operation": structured["operation"],
                "params": structured.get("params", {})
            }
        elif structured.get("action") == "final_answer":
            answer = structured.get("answer") or structured.get("final_result") or structured.get("analysis")
            decision = {"action": "final_answer", "answer": answer}
            updated_step["status"] = "finalized"
            updated_step["final_result"] = answer
            reported_next_stage = None

        LOG.info(f"[{subq_id}] ✅ Reasoner: стадия '{next_stage}' завершена. Следующая стадия: {reported_next_stage}")

        return AgentResult.ok(
            structured={
                "updated_step": updated_step,
                "decision": decision,
                "next_stage": reported_next_stage,
                "progress": f"completed {next_stage}"
            }
        )