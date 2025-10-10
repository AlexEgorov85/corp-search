# src/graph/react_graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.graph.nodes.planner import planner_node
from src.graph.nodes.reasoner import reasoner_node
from src.graph.nodes.executor import executor_node
from src.graph.nodes.next_subquestion import next_subquestion_node
from src.graph.nodes.synthesizer import synthesizer_node
from src.agents.registry import AgentRegistry
from src.model.context.base import (
    get_current_step_id,
    get_execution_step,
    is_step_completed,
)
from src.model.context.context import GraphContext

def build_react_graph(agent_registry: AgentRegistry):
    # Оборачиваем узлы, чтобы передавать agent_registry
    def planner(state: GraphContext) -> GraphContext:
        return planner_node(state.to_dict(), agent_registry=agent_registry)

    def reasoner(state: GraphContext) -> GraphContext:
        return reasoner_node(state.to_dict(), agent_registry=agent_registry)

    def executor(state: GraphContext) -> GraphContext:
        return executor_node(state.to_dict(), agent_registry=agent_registry)

    def synthesizer(state: GraphContext) -> GraphContext:
        return synthesizer_node(state.to_dict(), agent_registry=agent_registry)

    def next_subq(state: GraphContext) -> GraphContext:
        return next_subquestion_node(state.to_dict(), agent_registry=None)

    # Создаём граф с состоянием GraphContext
    graph = StateGraph(GraphContext)
    graph.add_node("planner", planner)
    graph.add_node("next_subquestion", next_subq)
    graph.add_node("reasoner", reasoner)
    graph.add_node("executor", executor)
    graph.add_node("synthesizer", synthesizer)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "next_subquestion")

    def next_subq_router(state: GraphContext) -> str:
        current_step_id = get_current_step_id(state)
        if current_step_id:
            return "reasoner"
        else:
            return "synthesizer"

    graph.add_conditional_edges("next_subquestion", next_subq_router)

    def reasoner_router(state: GraphContext) -> str:
        step_id = get_current_step_id(state)
        if not step_id:
            return "next_subquestion"

        # Проверяем, завершён ли шаг
        if is_step_completed(state, step_id):
            return "next_subquestion"

        # Получаем решение от Reasoner
        step = get_execution_step(state, step_id)
        if not step or not step.decision:
            return "next_subquestion"

        next_stage = step.decision.get("next_stage")
        if next_stage in {"validate_entities", "fetch_data", "process_data", "validate_result"}:
            return "executor"
        else:
            return "next_subquestion"

    graph.add_conditional_edges("reasoner", reasoner_router)
    graph.add_edge("executor", "next_subquestion")
    graph.add_edge("synthesizer", END)
    return graph.compile()