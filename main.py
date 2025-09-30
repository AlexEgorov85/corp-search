# main.py
import logging
from src.graph.react_graph import build_react_graph
from src.agents.registry import AgentRegistry

logging.basicConfig(level=logging.INFO)
agent_registry = AgentRegistry(validate_on_init=True)
graph = build_react_graph(agent_registry)

init = {
    "question": "Найди книги Пушкина и укажи главного героя в последней из них?",
}
final = graph.invoke(init)
print("FINAL ANSWER:", final.get("final_answer"))
print("Synth output:", final.get("synth_output"))