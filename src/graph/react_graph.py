# src/graph/react_graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.graph.context_model import GraphContext
from src.graph.nodes.planner import planner_node
from src.graph.nodes.reasoner import reasoner_node
from src.graph.nodes.executor import executor_node
from src.graph.nodes.next_subquestion import next_subquestion_node
from src.graph.nodes.synthesizer import synthesizer_node
from src.agents.registry import AgentRegistry


def build_react_graph(agent_registry: AgentRegistry):
    """
    Строит ReAct-граф с единым контекстом GraphContext.
    Все узлы получают agent_registry для доступа к tool_registry и control-агентам.
    """

    def planner(state: Dict[str, Any]) -> Dict[str, Any]:
        # Передаём agent_registry в planner_node для формирования tool_registry_snapshot
        return planner_node(state, agent_registry=agent_registry)

    def reasoner(state: Dict[str, Any]) -> Dict[str, Any]:
        return reasoner_node(state, agent_registry=agent_registry)

    def executor(state: Dict[str, Any]) -> Dict[str, Any]:
        return executor_node(state, agent_registry=agent_registry)

    def synthesizer(state: Dict[str, Any]) -> Dict[str, Any]:
        return synthesizer_node(state, agent_registry=agent_registry)

    def next_subq(state: Dict[str, Any]) -> Dict[str, Any]:
        # next_subquestion_node не требует agent_registry — работает с контекстом
        return next_subquestion_node(state, agent_registry=None)

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
        # Проверяем флаг finished
        if getattr(state, "finished", False):
            return "synthesizer"
        # Есть ли текущий подвопрос?
        if state.execution.current_subquestion_id:
            return "reasoner"
        else:
            return "synthesizer"

    graph.add_conditional_edges("next_subquestion", next_subq_router)

    def reasoner_router(state: GraphContext) -> str:
        current_call = state.execution.current_call
        if current_call and current_call.decision:
            action = current_call.decision.get("action")
            if action == "call_tool":
                return "executor"
        return "next_subquestion"

    graph.add_conditional_edges("reasoner", reasoner_router)

    # 🔁 Ключевое: executor → next_subquestion (для обработки следующего шага)
    graph.add_edge("executor", "next_subquestion")
    graph.add_edge("synthesizer", END)

    return graph.compile()