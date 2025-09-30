# src/graph/context_model.py
"""
ЕДИНАЯ МОДЕЛЬ СОСТОЯНИЯ ГРАФА (GraphContext)

Этот файл определяет каноническую Pydantic-модель для всего состояния ReAct-графа.
Она заменяет собой все legacy-словари и устраняет рассогласованность между узлами.

Цель:
- Хранить всё состояние в одном типизированном объекте.
- Гарантировать, что все узлы (planner, reasoner, executor и т.д.) работают с одной и той же структурой.
- Упростить отладку, тестирование и сериализацию.

Структура:
- GraphContext — корневой класс.
  ├── question: str — исходный вопрос пользователя.
  ├── plan: Optional[Dict] — результат работы PlannerAgent (структура подвопросов).
  ├── execution: ExecutionState — динамическое состояние выполнения шагов.
  │     ├── current_subquestion_id: Optional[str] — текущий обрабатываемый шаг.
  │     ├── steps: Dict[str, StepState] — словарь всех шагов по их ID.
  │     ├── current_call: Optional[CurrentCall] — буфер для передачи решения от Reasoner к Executor.
  │     └── history: List[Dict] — журнал событий (для отладки).
  ├── final_answer: Any — итоговый ответ от SynthesizerAgent.
  ├── synth_output: Any — дополнительные данные от синтезатора.
  └── memory: Dict[str, Any] — временный кэш между узлами (например, для промежуточных вычислений).

Пример использования в узле:
>>> from src.graph.context_model import GraphContext
>>> state = GraphContext(question="Какие книги написал Пушкин?")
>>> state.execution.current_subquestion_id = "q1"
>>> state.execution.steps["q1"] = StepState(id="q1", subquestion_text="Какие книги написал Пушкин?")
>>> return state  # LangGraph автоматически сериализует через model_dump()
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# === Типы статусов шага ===
"""
StepStatus — перечисление возможных статусов жизненного цикла шага.

Примеры:
- "pending" → шаг создан, но ещё не обработан.
- "analyzed" → Reasoner извлёк сущности и выбрал инструмент.
- "executed" → Executor получил результат от инструмента.
- "finalized" → результат валидирован и готов к синтезу.
- "failed" → ошибка на любом этапе.
"""
StepStatus = Literal[
    "pending",      # Шаг ещё не начат
    "analyzed",     # Сущности извлечены, инструмент выбран
    "validated",    # Параметры проверены (например, нормализация автора)
    "executed",     # Инструмент вызван, есть сырые данные
    "finalized",    # Результат валидирован и готов к синтезу
    "failed",       # Ошибка на любом этапе
]


# === Подвопрос (неизменяемый элемент плана) ===
"""
SubQuestion — модель одного атомарного подвопроса в плане.

Поля:
- id: str — уникальный идентификатор (например, "q1").
- text: str — формулировка подвопроса (фактологическая, без глаголов действия).
- depends_on: list[str] — список ID подвопросов, от которых зависит текущий.

Пример:
>>> sq = SubQuestion(id="q2", text="Какая из книг Пушкина — последняя?", depends_on=["q1"])
"""
class SubQuestion(BaseModel):
    id: str
    text: str
    depends_on: list[str] = []


# === Состояние одного шага выполнения ===
"""
StepState — динамическое состояние одного шага (подвопроса) в процессе выполнения.

Поля:
- id: str — идентификатор (должен совпадать с SubQuestion.id).
- status: StepStatus — текущий статус шага.
- subquestion_text: str — текст подвопроса (для отладки).
- analysis: Optional[dict] — результат этапа analyze_question (сущности, выбранный инструмент).
- validated_params: Optional[dict] — результат валидации (например, нормализованный автор).
- raw_output: Optional[Any] — сырые данные от инструмента (например, SQL-результат).
- final_result: Optional[Any] — финальный результат после валидации.
- error: Optional[str] — текст ошибки (если status == "failed").
- error_stage: Optional[str] — этап, на котором произошла ошибка.

Пример:
>>> step = StepState(
...     id="q1",
...     subquestion_text="Какие книги написал Пушкин?",
...     status="executed",
...     raw_output=[{"title": "Евгений Онегин"}]
... )
"""
class StepState(BaseModel):
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

    class Config:
        arbitrary_types_allowed = True  # Разрешаем Any-поля


# === Текущий вызов (буфер между Reasoner и Executor) ===
"""
CurrentCall — временный буфер для передачи решения от Reasoner к Executor.

Поля:
- decision: dict — решение Reasoner (например, {"action": "call_tool", "tool": "BooksLibraryAgent", ...}).
- step_id: Optional[str] — ID шага, к которому относится решение.
- ts: datetime — временная метка создания вызова.

Пример:
>>> call = CurrentCall(
...     decision={"action": "call_tool", "tool": "BooksLibraryAgent", "operation": "list_books", "params": {"author": "Пушкин"}},
...     step_id="q1"
... )
"""
class CurrentCall(BaseModel):
    decision: Optional[Dict[str, Any]] = None
    step_id: Optional[str] = None
    ts: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True


# === Общее состояние выполнения всех шагов ===
"""
ExecutionState — агрегирующая модель состояния выполнения.

Поля:
- steps: Dict[str, StepState] — все шаги по их ID.
- current_subquestion_id: Optional[str] — текущий обрабатываемый шаг.
- current_call: Optional[CurrentCall] — буфер для передачи решения.
- history: List[Dict] — журнал событий (для отладки и аудита).

Пример:
>>> exec_state = ExecutionState(
...     steps={"q1": StepState(...)},
...     current_subquestion_id="q1",
...     current_call=CurrentCall(...)
... )
"""
class ExecutionState(BaseModel):
    steps: Dict[str, StepState] = {}
    current_subquestion_id: Optional[str] = None
    current_call: Optional[CurrentCall] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# === Корневая модель состояния графа ===
"""
GraphContext — каноническая модель всего состояния графа.

Поля:
- question: str — исходный вопрос пользователя.
- plan: Optional[Dict] — план от PlannerAgent (структура с subquestions).
- execution: ExecutionState — состояние выполнения шагов.
- final_answer: Any — итоговый ответ (заполняется SynthesizerAgent).
- synth_output: Any — дополнительные данные от синтезатора.
- memory: Dict — временный кэш между узлами (например, для промежуточных вычислений).

Пример полного состояния:
>>> state = GraphContext(
...     question="Найди книги Пушкина и укажи главного героя в последней из них?",
...     plan={
...         "subquestions": [
...             {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []},
...             {"id": "q2", "text": "Какая из книг — последняя?", "depends_on": ["q1"]},
...             {"id": "q3", "text": "Кто главный герой в последней книге?", "depends_on": ["q2"]}
...         ]
...     },
...     execution=ExecutionState(
...         steps={
...             "q1": StepState(id="q1", subquestion_text="Какие книги написал Пушкина?", status="finalized", final_result=[...])
...         },
...         current_subquestion_id="q2"
...     )
... )
"""
class GraphContext(BaseModel):
    question: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    execution: ExecutionState = Field(default_factory=ExecutionState)
    final_answer: Optional[Any] = None
    synth_output: Optional[Any] = None
    memory: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True