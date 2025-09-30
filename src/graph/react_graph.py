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
    execution: Dict[str, Any]

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

    def next_subq_router(state):
        if state.get("finished"):
            return "synthesizer"
        current_id = state.get("execution", {}).get("current_subquestion_id")
        if current_id:
            return "reasoner"
        else:
            return "synthesizer"

    graph.add_conditional_edges("next_subquestion", next_subq_router)

    def reasoner_router(state):
        exec_state = state.get("execution", {}) or {}
        current_call = exec_state.get("current_call", {}) or {}
        decision = current_call.get("decision", {}) or {}
        action = decision.get("action")
        if action == "call_tool":
            return "executor"
        else:
            return "next_subquestion"

    graph.add_conditional_edges("reasoner", reasoner_router)

    # 🔑 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: executor → next_subquestion (а не → reasoner)
    graph.add_edge("executor", "next_subquestion")

    graph.add_edge("synthesizer", END)
    return graph.compile()