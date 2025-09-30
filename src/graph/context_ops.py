# src/graph/context_ops.py
"""
ОПЕРАЦИИ ЧТЕНИЯ И ЗАПИСИ НАД ЕДИНЫМ КОНТЕКСТОМ (GraphContext)

Этот файл предоставляет **чистые функции** для безопасной работы с GraphContext.
Цель — изолировать логику доступа к полям состояния и избежать дублирования.

Правила:
- Все функции принимают `ctx: GraphContext` и ничего не возвращают (кроме геттеров).
- Все изменения происходят **in-place** (GraphContext — мутабельный объект).
- Нет прямого доступа к `ctx.execution.steps` из узлов — только через `get_step`, `ensure_step`.

Пример использования в узле:
>>> from src.graph.context_ops import get_question, set_plan, ensure_step
>>> question = get_question(ctx)
>>> set_plan(ctx, {"subquestions": [...]})
>>> ensure_step(ctx, "q1", subquestion_text="Какие книги написал Пушкин?")
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from src.graph.context_model import GraphContext, StepState, CurrentCall


# === Вопрос ===
def get_question(ctx: GraphContext) -> Optional[str]:
    """Возвращает исходный вопрос пользователя."""
    return ctx.question

def set_question(ctx: GraphContext, question: str) -> None:
    """Устанавливает исходный вопрос."""
    ctx.question = question


# === План ===
def get_plan(ctx: GraphContext) -> Optional[Dict[str, Any]]:
    """
    Возвращает план от PlannerAgent.
    План — это dict с ключом 'subquestions', содержащий список подвопросов.
    
    Пример:
    {
        "subquestions": [
            {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []},
            ...
        ]
    }
    """
    return ctx.plan

def set_plan(ctx: GraphContext, plan: Dict[str, Any]) -> None:
    """Устанавливает план (результат работы PlannerAgent)."""
    ctx.plan = plan


# === Шаги ===
def get_step(ctx: GraphContext, step_id: str) -> Optional[StepState]:
    """Возвращает шаг по его ID или None, если шаг не существует."""
    return ctx.execution.steps.get(step_id)

def ensure_step(ctx: GraphContext, step_id: str, **kwargs) -> StepState:
    """
    Создаёт шаг, если его нет, и возвращает объект StepState.
    Любые дополнительные kwargs передаются в конструктор StepState.
    
    Пример:
    >>> step = ensure_step(ctx, "q1", subquestion_text="Какие книги написал Пушкина?")
    """
    if step_id not in ctx.execution.steps:
        ctx.execution.steps[step_id] = StepState(id=step_id, **(kwargs or {}))
    return ctx.execution.steps[step_id]

def update_step(ctx: GraphContext, step_id: str, **fields) -> None:
    """
    Обновляет поля шага. Поля, отсутствующие в StepState, игнорируются.
    
    Пример:
    >>> update_step(ctx, "q1", status="executed", raw_output=[...])
    """
    step = ensure_step(ctx, step_id)
    for k, v in fields.items():
        if hasattr(step, k):
            setattr(step, k, v)


# === Текущий подвопрос ===
def get_current_subquestion_id(ctx: GraphContext) -> Optional[str]:
    """Возвращает ID текущего обрабатываемого подвопроса."""
    return ctx.execution.current_subquestion_id

def set_current_subquestion_id(ctx: GraphContext, step_id: Optional[str]) -> None:
    """Устанавливает текущий подвопрос по его ID."""
    ctx.execution.current_subquestion_id = step_id


# === Текущий вызов (Reasoner → Executor) ===
def get_current_call(ctx: GraphContext) -> Optional[CurrentCall]:
    """Возвращает текущий вызов (буфер между Reasoner и Executor)."""
    return ctx.execution.current_call

def set_current_call(ctx: GraphContext, decision: Dict[str, Any], step_id: Optional[str] = None) -> None:
    """
    Устанавливает текущий вызов. Автоматически добавляет временную метку.
    
    Пример:
    >>> set_current_call(ctx, {"action": "call_tool", "tool": "BooksLibraryAgent", ...}, step_id="q1")
    """
    if step_id is None:
        step_id = ctx.execution.current_subquestion_id
    ctx.execution.current_call = CurrentCall(
        decision=decision,
        step_id=step_id,
        ts=datetime.utcnow()
    )

def clear_current_call(ctx: GraphContext) -> None:
    """Очищает текущий вызов (после успешного или неуспешного выполнения)."""
    ctx.execution.current_call = None


# === Финальный ответ и синтез ===
def get_final_answer(ctx: GraphContext) -> Any:
    """Возвращает финальный ответ (результат работы SynthesizerAgent)."""
    return ctx.final_answer

def set_final_answer(ctx: GraphContext, answer: Any) -> None:
    """Устанавливает финальный ответ."""
    ctx.final_answer = answer

def get_synth_output(ctx: GraphContext) -> Any:
    """Возвращает дополнительные данные от SynthesizerAgent."""
    return ctx.synth_output

def set_synth_output(ctx: GraphContext, output: Any) -> None:
    """Устанавливает дополнительные данные от SynthesizerAgent."""
    ctx.synth_output = output


# === История (журнал событий) ===
def append_history_event(ctx: GraphContext, event: Dict[str, Any]) -> None:
    """
    Добавляет событие в историю выполнения.
    Автоматически добавляет временную метку 'ts'.
    
    Пример:
    >>> append_history_event(ctx, {"type": "planner_agent_generated_plan", "plan_summary": "..."})
    """
    event = dict(event)
    event.setdefault("ts", datetime.utcnow())
    ctx.execution.history.append(event)


# === Проверка завершения ===
def is_finished(ctx: GraphContext) -> bool:
    """
    Проверяет, завершён ли граф.
    В текущей реализации завершение определяется по отсутствию незавершённых шагов,
    но можно добавить явное поле `finished: bool` в GraphContext при необходимости.
    """
    # Пока что возвращаем False — логика завершения реализована в next_subquestion_node
    return False