# src/graph/nodes/planner.py
# coding: utf-8
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from src.graph.context import GraphContext
from src.services.results.agent_result import AgentResult

LOG = logging.getLogger(__name__)


def _build_tool_registry_snapshot(full_tool_registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создаёт упрощённую и безопасную копию tool_registry для передачи в PlannerAgent.
    Удаляет чувствительные поля (implementation, config и т.д.), оставляя только:
      - title
      - description
      - operations (с их kind, description, params, outputs)
    """
    snapshot = {}
    for name, meta in full_tool_registry.items():
        if not isinstance(meta, dict):
            snapshot[name] = {}
            continue
        safe_meta = {
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "operations": {}
        }
        ops = meta.get("operations", {})
        if isinstance(ops, dict):
            for op_name, op_meta in ops.items():
                if not isinstance(op_meta, dict):
                    safe_meta["operations"][op_name] = {"description": ""}
                    continue
                safe_meta["operations"][op_name] = {
                    "kind": op_meta.get("kind", "direct"),
                    "description": op_meta.get("description", ""),
                    "params": op_meta.get("params", {}),
                    "outputs": op_meta.get("outputs", {})
                }
        snapshot[name] = safe_meta
    return snapshot


def planner_node(state: Dict[str, Any], agent_registry=None) -> Dict[str, Any]:
    ctx = state if isinstance(state, GraphContext) else GraphContext.from_state_dict(state)
    question = ctx.question or ""
    if not question.strip():
        ctx.append_history({"type": "planner_no_question"})
        return ctx.to_legacy_state()

    if agent_registry is not None:
        try:
            planner_agent = agent_registry.instantiate_agent("PlannerAgent", control=True)
        except Exception as e:
            planner_agent = None
            ctx.append_history({"type": "planner_instantiate_failed", "error": str(e)})

        if planner_agent is not None:
            try:
                # 🔑 Строим snapshot инструментов
                full_tool_registry = agent_registry.tool_registry
                tool_registry_snapshot = _build_tool_registry_snapshot(full_tool_registry)

                # 🔑 Передаём обязательный параметр
                res = planner_agent.execute_operation(
                    "plan",
                    {
                        "question": question,
                        "tool_registry_snapshot": tool_registry_snapshot
                    },
                    context={}
                )

                if isinstance(res, AgentResult) and res.status == "ok":
                    plan_struct = res.structured or res.content or {}
                    ctx.memory["plan"] = plan_struct

                    subqs = plan_struct.get("subquestions") if isinstance(plan_struct, dict) else None
                    if subqs:
                        first_id: Optional[str] = None
                        for s in subqs:
                            sid = s.get("id") or f"sq_{len(ctx.execution.steps) + 1}"
                            if first_id is None:
                                first_id = sid
                            ctx.ensure_step(sid, subquestion_text=s.get("text") or "")
                        if first_id:
                            ctx.execution.current_subquestion_id = first_id
                    ctx.append_history({"type": "planner_agent_generated_plan", "plan_summary": str(plan_struct)[:300]})
                    return ctx.to_legacy_state()
                else:
                    ctx.append_history({"type": "planner_agent_failed", "result": str(res.content if hasattr(res, "content") else res)})
            except Exception as e:
                ctx.append_history({"type": "planner_agent_execute_failed", "error": str(e)})

    # FALLBACK
    step_id = "q1"
    ctx.ensure_step(step_id, subquestion_text=question, status="pending")
    ctx.execution.current_subquestion_id = step_id
    ctx.memory.setdefault("plan", {"subquestions": [{"id": step_id, "text": question}]})
    ctx.append_history({"type": "planner_fallback_created_step", "step_id": step_id})
    return ctx.to_legacy_state()