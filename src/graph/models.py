# src/graph/models.py
"""
Модели состояния графа выполнения.

Этот модуль определяет строгие Pydantic-модели для:
- Плана (PlanModel) — неизменяемая структура подвопросов.
- Выполнения (ExecutionState) — динамическое состояние шагов.

Пример использования:
>>> plan = PlanModel(
...     plan_id="p1",
...     subquestions=[SubQuestion(id="q1", text="Какие книги написал Пушкин?")]
... )
>>> step = StepState(id="q1", subquestion_text="Какие книги написал Пушкин?")
>>> exec_state = ExecutionState(steps={"q1": step}, current_subquestion_id="q1")
"""
from typing import Dict, Optional, Literal, Any
from pydantic import BaseModel

# Статусы жизненного цикла шага
StepStatus = Literal[
    "pending",      # Шаг ещё не начат
    "analyzed",     # Сущности извлечены, инструмент выбран
    "validated",    # Параметры проверены
    "executed",     # Инструмент вызван, есть сырые данные
    "finalized",    # Результат валидирован и готов к синтезу
    "failed",       # Ошибка на любом этапе
]

class SubQuestion(BaseModel):
    """
    Модель одного подвопроса в плане.

    Пример:
        SubQuestion(id="q1", text="Какие книги написал Пушкин?", depends_on=[])
    """
    id: str
    text: str
    depends_on: list[str] = []

class PlanModel(BaseModel):
    """
    Неизменяемая модель плана выполнения.

    Поля:
        plan_id: Уникальный идентификатор плана.
        subquestions: Список подвопросов.

    Пример:
        PlanModel(plan_id="p1", subquestions=[...])
    """
    plan_id: str
    subquestions: list[SubQuestion]

    class Config:
        # Делает объект неизменяемым после создания
        frozen = True

class StepState(BaseModel):
    """
    Состояние одного шага (подвопроса) в процессе выполнения.

    Поля:
        id: Идентификатор подвопроса (например, "q1").
        status: Текущий статус шага.
        subquestion_text: Текст подвопроса для отладки.
        analysis: Результат этапа analyze_question.
        validated_params: Результат валидации сущностей.
        raw_output: Сырые данные от инструмента (например, SQL-результат).
        final_result: Финальный результат после validate_result.
        error: Текст ошибки (если status == "failed").
        error_stage: Этап, на котором произошла ошибка.

    Пример:
        StepState(id="q1", status="executed", raw_output=[{"title": "Евгений Онегин"}])
    """
    id: str
    status: StepStatus = "pending"
    subquestion_text: str

    # Промежуточные данные этапов
    analysis: Optional[dict] = None
    validated_params: Optional[dict] = None
    raw_output: Optional[Any] = None
    final_result: Optional[Any] = None

    # Ошибки
    error: Optional[str] = None
    error_stage: Optional[str] = None

class ExecutionState(BaseModel):
    """
    Общее состояние выполнения всех шагов.

    Поля:
        steps: Словарь состояний шагов по их id.
        current_subquestion_id: Текущий обрабатываемый подвопрос.
        current_call: Временный буфер для передачи решения Reasoner → Executor.

    Пример:
        ExecutionState(
            steps={"q1": StepState(...)},
            current_subquestion_id="q1",
            current_call={"subquestion_id": "q1", "decision": {...}}
        )
    """
    steps: Dict[str, StepState] = {}
    current_subquestion_id: Optional[str] = None
    current_call: Optional[Dict[str, Any]] = None