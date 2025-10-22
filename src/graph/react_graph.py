# src/graph/react_graph.py
"""
Граф выполнения ReAct-цикла.
Маршрутизация:
  planner → next_subquestion → (reasoner ↔ executor) → synthesizer
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.graph.nodes.planner import planner_node
from src.graph.nodes.reasoner import reasoner_node
from src.graph.nodes.executor import executor_node
from src.graph.nodes.next_subquestion import next_subquestion_node
from src.graph.nodes.synthesizer import synthesizer_node
from src.agents.registry import AgentRegistry
from src.model.context.context import GraphContext

def build_react_graph(agent_registry: AgentRegistry):
    def planner(state: GraphContext) -> GraphContext:
        return planner_node(state.to_dict(), agent_registry=agent_registry)

    def reasoner(state: GraphContext) -> GraphContext:
        return reasoner_node(state.to_dict(), agent_registry=agent_registry)

    def executor(state: GraphContext) -> GraphContext:
        return executor_node(state.to_dict(), agent_registry=agent_registry)

    def next_subq(state: GraphContext) -> GraphContext:
        return next_subquestion_node(state.to_dict(), agent_registry=None)

    def synthesizer(state: GraphContext) -> GraphContext:
        return synthesizer_node(state.to_dict(), agent_registry=agent_registry)

    graph = StateGraph(GraphContext)
    graph.add_node("planner", planner)
    graph.add_node("next_subquestion", next_subq)
    graph.add_node("reasoner", reasoner)
    graph.add_node("executor", executor)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "next_subquestion")

    def next_subq_router(ctx: GraphContext) -> str:
        return "reasoner" if ctx.get_current_step_id() else "synthesizer"

    graph.add_conditional_edges("next_subquestion", next_subq_router)

    def reasoner_router(ctx: GraphContext) -> str:
        step_id = ctx.get_current_step_id()
        if not step_id or ctx.is_step_fully_completed(step_id):
            return "next_subquestion"
        return "executor"

    graph.add_conditional_edges("reasoner", reasoner_router)

    # 🔁 Ключевой цикл: executor → reasoner
    graph.add_edge("executor", "reasoner")

    graph.add_edge("synthesizer", END)
    return graph.compile()