# src/graph/nodes/executor.py
from typing import Dict, Any
import logging
from src.agents.registry import AgentRegistry

LOG = logging.getLogger(__name__)


def executor_node(state: Dict[str, Any], agent_registry: AgentRegistry) -> Dict[str, Any]:
    current_call = state.get("current_call", {})
    decision = current_call.get("decision", {})
    action = decision.get("action")
    if action != "call_tool":
        return {}

    tool_name = decision["tool"]
    operation = decision["operation"]
    params = decision["params"]

    tool_agent = agent_registry.instantiate_agent(tool_name, control=False)
    if not tool_agent:
        LOG.error("Executor: не удалось инстанцировать агент %s", tool_name)
        return {}

    result = tool_agent.execute_operation(operation, params, state)
    if result.status != "ok":
        LOG.warning("Executor: операция %s.%s завершилась с ошибкой", tool_name, operation)
        return {}

    subq_id = current_call.get("subquestion_id", "unknown")
    step_outputs = state.get("step_outputs", {})

    if operation == "validate_author":
        step_outputs[f"{subq_id}_validated"] = result.structured
    else:
        step_outputs[f"{subq_id}_raw"] = result.structured or result.content

    return {"step_outputs": step_outputs}