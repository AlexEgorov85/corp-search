# src/api/main.py
from __future__ import annotations
import logging
from src.graph.react_graph import build_react_graph
from src.agents.registry import AgentRegistry

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# Создаём единый реестр
agent_registry = AgentRegistry(validate_on_init=True)
graph = build_react_graph(agent_registry)

# Опциональный FastAPI
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class QueryRequest(BaseModel):
        question: str

    @app.post("/query")
    async def query_endpoint(req: QueryRequest):
        state = {
            "question": req.question,
            "step_outputs": {},
            "final_answer": None
        }
        return graph.invoke(state)

except Exception:
    pass