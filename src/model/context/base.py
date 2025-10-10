# src/model/context/base.py
"""
Базовый класс для контекста графа.
Содержит общие методы для сериализации/десериализации, необходимые для совместимости с LangGraph.
ВАЖНО:
- Все функции ниже работают с GraphContext.
- Используйте термины:
    - "subquestion" — элемент плана (неизменяемый)
    - "execution step" — состояние выполнения (изменяемое)
"""

from __future__ import annotations
from typing import Any, Dict, Optional


from pydantic import BaseModel


class BaseGraphContext(BaseModel):
    """
    Базовый класс контекста графа.
    Предоставляет стандартные методы для работы с LangGraph.
    """
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует объект в словарь для передачи в LangGraph.
        """
        return self.model_dump()

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "BaseGraphContext":
        """
        Создаёт объект из словаря (состояния LangGraph).
        Этот метод должен быть переопределён в дочерних классах для обработки специфической логики.
        """
        return cls(**data)

# === API для узлов графа ===

def get_question(ctx) -> str:
    """Возвращает исходный вопрос."""
    return ctx.get_question()

def set_question(ctx, question: str) -> None:
    """Устанавливает исходный вопрос."""
    ctx.set_question(question)

def get_plan(ctx):
    """Возвращает план (Plan или dict)."""
    return ctx.get_plan()

def set_plan(ctx, plan) -> None:
    """Устанавливает план."""
    ctx.set_plan(plan)

def get_current_step_id(ctx) -> Optional[str]:
    """Возвращает ID текущего шага выполнения."""
    return ctx.get_current_step_id()

def set_current_step_id(ctx, step_id: Optional[str]) -> None:
    """Устанавливает текущий шаг выполнения."""
    ctx.set_current_step_id(step_id)

def get_subquestion_text(ctx, step_id: str) -> str:
    """Возвращает текст подвопроса из плана."""
    return ctx.get_subquestion_text(step_id)

# --- Работа с шагом выполнения ---

def get_execution_step(ctx, step_id: str):
    """Возвращает состояние выполнения шага."""
    return ctx.get_execution_step(step_id)

def ensure_execution_step(ctx, step_id: str):
    """Гарантирует существование шага выполнения."""
    return ctx.ensure_execution_step(step_id)

def update_step_data(ctx, step_id: str, **kwargs) -> None:
    """Обновляет данные шага выполнения."""
    ctx.update_step_data(step_id, **kwargs)

def is_step_completed(ctx, step_id: str) -> bool:
    """Проверяет, завершён ли шаг."""
    return ctx.is_step_completed(step_id)

def get_step_result(ctx, step_id: str):
    """Возвращает результат выполнения шага."""
    return ctx.get_step_result(step_id)

# --- История и память ---

def append_history_event(ctx, event: Dict[str, Any]) -> None:
    """Добавляет событие в историю выполнения."""
    ctx.append_history_event(event)

def get_final_answer(ctx):
    """Возвращает финальный ответ из памяти."""
    return ctx.get_final_answer()

def set_final_answer(ctx, answer) -> None:
    """Устанавливает финальный ответ в память."""
    ctx.set_final_answer(answer)