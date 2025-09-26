# src/graph/react_graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from src.graph.nodes.planner import planner_node
from src.graph.nodes.reasoner import reasoner_node
from src.graph.nodes.executor import executor_node
from src.graph.nodes.next_subquestion import next_subquestion_node
from src.graph.nodes.synthesizer import synthesizer_node
from src.agents.registry import AgentRegistry


class AppState(TypedDict, total=False):
    question: str
    agents_config: Dict[str, Any]
    plan: Any
    current_subquestion_id: str
    step_outputs: Dict[str, Any]
    final_answer: str
    finished: bool
    current_call: Dict[str, Any]


def build_react_graph(agent_registry: AgentRegistry):
    def planner(state): return planner_node(state, agent_registry)
    def reasoner(state): return reasoner_node(state, agent_registry)
    def executor(state): return executor_node(state, agent_registry)
    def synthesizer(state): return synthesizer_node(state, agent_registry)
    def next_subq(state): return next_subquestion_node(state)

    graph = StateGraph(AppState)
    graph.add_node("planner", planner)
    graph.add_node("next_subquestion", next_subq)
    graph.add_node("reasoner", reasoner)
    graph.add_node("executor", executor)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "next_subquestion")

    graph.add_conditional_edges(
        "next_subquestion",
        lambda state: "reasoner" if not state.get("finished") else "synthesizer"
    )

    def reasoner_router(state):
        current_call = state.get("current_call", {})
        decision = current_call.get("decision", {})
        action = decision.get("action")
        if action == "call_tool":
            return "executor"
        elif action in ("final_answer", "skip"):
            return "next_subquestion"
        else:
            return "executor"

    graph.add_conditional_edges("reasoner", reasoner_router)
    graph.add_edge("executor", "reasoner")
    graph.add_edge("synthesizer", END)

    return graph.compile()