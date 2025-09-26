# src/graph/nodes/reasoner.py
from typing import Dict, Any
import logging
from src.agents.registry import AgentRegistry

LOG = logging.getLogger(__name__)


def reasoner_node(state: Dict[str, Any], agent_registry: AgentRegistry) -> Dict[str, Any]:
    if state.get("finished"):
        return {}
    current_id = state.get("current_subquestion_id")
    if not current_id:
        return {"finished": True}

    plan = state.get("plan", {})
    subquestions = {sq["id"]: sq for sq in plan.get("subquestions", [])}
    subquestion = subquestions.get(current_id)
    if not subquestion:
        return {"finished": True}

    agent = agent_registry.instantiate_agent("ReasonerAgent", control=True)
    if not agent:
        return {"finished": True}

    params = {
        "subquestion": subquestion,
        "step_outputs": state.get("step_outputs", {}),
        "tool_registry_snapshot": agent_registry.tool_registry,
        "current_stage": state.get("next_stage", "ANALYZE_QUESTION")
    }

    result = agent.execute_operation("decide", params, state)
    if result.status != "ok":
        return {"finished": True}

    decision = result.structured
    action = decision.get("action")
    updates = {"current_call": {"subquestion_id": current_id, "decision": decision}}

    if action == "store_analysis":
        step_outputs = state.get("step_outputs", {})
        step_outputs[f"{current_id}_analysis"] = decision
        updates["step_outputs"] = step_outputs
    elif action == "final_answer":
        step_outputs = state.get("step_outputs", {})
        step_outputs[current_id] = decision
        updates["step_outputs"] = step_outputs

    next_stage = decision.get("next_stage")
    if next_stage:
        updates["next_stage"] = next_stage

    return updates