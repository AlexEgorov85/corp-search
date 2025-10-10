# src/model/context/models.py
"""
Pydantic-модели данных для контекста графа.
Эти модели описывают структуру данных, но не содержат логики.

ВАЖНО:
- Plan и SubQuestion — это НЕИЗМЕНЯЕМАЯ декомпозиция вопроса (спецификация).
- StepExecutionState — это ИЗМЕНЯЕМОЕ состояние выполнения одного подвопроса.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# === 1. ПЛАН: неизменяемая структура подвопросов ===
class SubQuestion(BaseModel):
    id: str
    text: str
    depends_on: List[str] = []

class Plan(BaseModel):
    subquestions: List[SubQuestion] = Field(default_factory=list)


# === 2. ШАГ ВЫПОЛНЕНИЯ: состояние одного подвопроса ===
class StepExecutionState(BaseModel):
    """
    Состояние выполнения одного шага (подвопроса из плана).
    Каждый экземпляр привязан к конкретному SubQuestion по `id`.
    """
    id: str
    text: str = Field(..., description="Исходный текст атомарного подвопроса.")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # --- Состояние выполнения ---
    completed: bool = Field(default=False, description="Флаг завершения шага.")
    error: Optional[str] = Field(default=None, description="Текст последней ошибки.")
    retry_count: int = Field(default=0, description="Число попыток выполнения шага.")
    validation_passed: bool = Field(default=False, description="Флаг успешной валидации результата.")

    # --- Семантический контекст (для Reasoner!) ---
    stage: Optional[str] = Field(default=None, description="Текущая стадия жизненного цикла.")
    decision: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Последнее решение Reasoner'а (next_stage, selected_tool и т.д.)."
    )

    # --- Результаты выполнения ---
    raw_output: Optional[Any] = Field(
        default=None,
        description="Результат выполнения операции (AgentResult.output)."
    )

    # --- История вызовов ---
    agent_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="История всех вызовов агентов в виде сериализованных AgentResult."
    )


# === 3. КОНТЕКСТ ВЫПОЛНЕНИЯ: агрегатор всех шагов ===
class ExecutionContext(BaseModel):
    current_subquestion_id: Optional[str] = None
    subquestions: Dict[str, StepExecutionState] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)