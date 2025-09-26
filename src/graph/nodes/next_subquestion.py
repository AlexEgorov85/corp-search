# src/graph/nodes/next_subquestion.py
from typing import Dict, Any
import logging
LOG = logging.getLogger(__name__)

def next_subquestion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выбирает следующий подвопрос для выполнения.
    """
    plan = state.get("plan", {})
    step_outputs = state.get("step_outputs", {})
    subquestions = plan.get("subquestions", [])

    # Найти первый незавершённый подвопрос, у которого выполнены зависимости
    for sq in subquestions:
        sq_id = sq["id"]
        # Проверяем, есть ли финальный результат для этого подвопроса
        if step_outputs.get(sq_id, {}).get("finalized"):
            continue  # уже завершён
        deps = sq.get("depends_on", [])
        if all(dep in step_outputs for dep in deps):
            return {"current_subquestion_id": sq_id}

    # Все подвопросы выполнены
    return {"finished": True}