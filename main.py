# main.py
import logging
from src.graph.react_graph import build_react_graph
from src.agents.registry import AgentRegistry

from src.model.context.base import set_question
from src.model.context.context import GraphContext

logging.basicConfig(level=logging.INFO)

# Подавляем технические предупреждения от llama_cpp
logging.getLogger("llama_cpp").setLevel(logging.ERROR)

# Создаём реестр агентов
agent_registry = AgentRegistry(validate_on_init=True)

# Собираем граф
graph = build_react_graph(agent_registry)

# === 1. Создаём пустой контекст ===
ctx = GraphContext()

# === 2. Устанавливаем вопрос через API ===
question = "Найди книги Пушкина и укажи последнюю из них?"
# question = "Найди книги Пушкина"
set_question(ctx, question)

# === 3. Запускаем граф ===
final_ctx = graph.invoke(ctx.to_dict())

# === 4. Читаем ответ из memory ===
final_answer = final_ctx.get("memory", {}).get("final_answer")
synth_output = final_ctx.get("memory", {}).get("synth_output")

print("FINAL ANSWER:", final_answer)
print("Synth output:", synth_output)