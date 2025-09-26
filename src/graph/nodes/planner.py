# src/graph/nodes/planner.py
from typing import Dict, Any, List
import uuid
import logging
from src.agents.registry import AgentRegistry

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def planner_node(state: Dict[str, Any], agent_registry: AgentRegistry) -> Dict[str, Any]:
    try:
        if state.get("plan"):
            return {}
        question = (state.get("question") or "").strip()
        agent = agent_registry.instantiate_agent("PlannerAgent", control=True)

        result = None
        try:
            result = agent.execute_operation("plan", {"question": question, "tool_registry_snapshot": agent_registry.tool_registry})
        except Exception as e:
            LOG.exception("planner_node: execute_operation failed: %s", e)
            return {"plan": {"subquestions": []}, "plan_id": str(uuid.uuid4()), "finished": True}

        if isinstance(result, dict) and result.get("plan"):
            plan = result.get("plan")
            plan_id = result.get("plan_id") or str(uuid.uuid4())
            LOG.info("planner_node: plan obtained from PlannerAgent (id=%s)", plan_id)
            return {"plan": plan, "plan_id": plan_id}

        if isinstance(result, list):
            plan = {"subquestions": result}
            plan_id = str(uuid.uuid4())
            LOG.info("planner_node: plan (list) obtained from PlannerAgent (id=%s)", plan_id)
            return {"plan": plan, "plan_id": plan_id}

        LOG.exception("planner_node unexpected error: %s", e)
        return {"plan": {"subquestions": []}, "plan_id": str(uuid.uuid4()), "finished": True}

    except Exception as e:
        LOG.exception("planner_node unexpected error: %s", e)
        return {"plan": {"subquestions": []}, "plan_id": str(uuid.uuid4()), "finished": True}