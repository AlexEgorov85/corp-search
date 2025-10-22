# src/model/context/models.py
"""
Pydantic-модели данных для контекста графа.

Эти модели описывают структуру данных, но не содержат логики.
ВАЖНО:
- Plan и SubQuestion — это НЕИЗМЕНЯЕМАЯ декомпозиция вопроса (спецификация).
- StepExecutionState — это ИЗМЕНЯЕМОЕ состояние выполнения одного подвопроса.
- Все модели должны быть совместимы с Pydantic для автоматической валидации.

Структура:
1. SubQuestion: определяет структуру подвопроса в плане
2. Plan: содержит список подвопросов
3. StepExecutionState: отслеживает состояние выполнения одного подвопроса
4. ExecutionContext: агрегирует все шаги выполнения
5. GraphState: корневая модель состояния графа

Изменения для поддержки гипотез ReasonerAgent:
- Добавлено поле `hypothesis` в StepExecutionState для хранения информации о выбранной гипотезе
- Добавлены поля для отслеживания необходимости постобработки и валидации
- Добавлено поле `current_call` для хранения информации о текущем вызове
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# === 1. ПЛАН: неизменяемая структура подвопросов ===
class SubQuestion(BaseModel):
    """Описание одного подвопроса из плана.
    
    Подвопросы:
    - Не могут быть изменены после создания плана
    - Определяют порядок и зависимости выполнения
    - Содержат информацию о том, что нужно сделать
    
    Поля:
    - id: Уникальный идентификатор подвопроса (например, "q1")
    - text: Текст подвопроса для обработки
    - depends_on: Список ID подвопросов, от которых зависит текущий
    - operation_hint: Опциональная подсказка для ReasonerAgent о том, какую операцию использовать
    """
    id: str
    text: str
    depends_on: List[str] = Field(default_factory=list)
    operation_hint: Optional[Dict[str, Any]] = None


class Plan(BaseModel):
    """План выполнения, состоящий из подвопросов.
    
    План:
    - Создается PlannerAgent в начале обработки
    - Остается неизменным на протяжении всего выполнения
    - Определяет структуру работы с вопросом
    
    Поля:
    - subquestions: Список подвопросов, составляющих план
    """
    subquestions: List[SubQuestion] = Field(default_factory=list)


# === 2. СОСТОЯНИЕ ВЫПОЛНЕНИЯ: изменяемое состояние для одного подвопроса ===
class StepExecutionState(BaseModel):
    """
    Состояние выполнения одного шага (подвопроса из плана).
    
    Каждый экземпляр привязан к конкретному SubQuestion по `id`.
    
    Основные аспекты:
    - Состояние изменяется в течение выполнения
    - Содержит информацию о текущем этапе обработки
    - Сохраняет результаты и ошибки
    - Хранит информацию о гипотезах и текущих вызовах
    
    Изменения для поддержки ReasonerAgent:
    - Добавлено поле `hypothesis` для хранения информации о выбранной гипотезе
    - Добавлены поля для отслеживания необходимости постобработки и валидации
    - Добавлено поле `current_call` для хранения информации о текущем вызове
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
        description="Последнее решение Reasoner'а (все гипотезы и выбор)."
    )
    
    # --- НОВЫЕ ПОЛЯ для работы с гипотезами ---
    hypothesis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Информация о выбранной гипотезе."
    )
    needs_postprocessing: bool = Field(
        default=False,
        description="Требуется ли постобработка результата"
    )
    needs_validation: bool = Field(
        default=False,
        description="Требуется ли валидация результата"
    )
    postprocessing_confidence: float = Field(
        default=0.0,
        description="Уверенность в необходимости постобработки"
    )
    validation_confidence: float = Field(
        default=0.0,
        description="Уверенность в необходимости валидации"
    )

    # Ожидаемые этапы выполнения
    expected_stages: Dict[str, bool] = Field(
        default_factory=lambda: {"data_fetch": False, "processing": False, "validation": False},
        description="Ожидаемые этапы выполнения"
    )
    
    # Завершенные этапы выполнения
    completed_stages: Dict[str, bool] = Field(
        default_factory=lambda: {"data_fetch": False, "processing": False, "validation": False},
        description="Завершенные этапы выполнения"
    )
    
    # Текущий вызов (очищается после выполнения)
    current_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Текущий вызов для выполнения операции"
    )
    
    # --- Результаты выполнения ---
    raw_output: Optional[Any] = Field(
        default=None,
        description="Результат выполнения операции (AgentResult.output)."
    )

    validation_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Результат последней валидации (is_valid, feedback)."
    )
    
    # --- История вызовов ---
    agent_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="История всех вызовов агентов в виде сериализованных AgentResult."
    )


# === 3. КОНТЕКСТ ВЫПОЛНЕНИЯ: агрегатор всех шагов ===
class ExecutionContext(BaseModel):
    """Контекст выполнения, содержащий все шаги и историю.
    
    Агрегирует информацию по всем подвопросам и отслеживает общий прогресс.
    
    Поля:
    - current_step_id: ID текущего активного шага
    - steps: Словарь состояний выполнения для каждого шага (ключ: id подвопроса)
    - history: История событий выполнения для отладки и аудита
    """
    current_step_id: Optional[str] = None
    steps: Dict[str, StepExecutionState] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)


# === 4. ГЛОБАЛЬНОЕ СОСТОЯНИЕ ГРАФА ===
class GraphState(BaseModel):
    """Корневая модель состояния всего графа выполнения.
    
    Содержит все необходимые данные для работы системы:
    - memory: Временное хранилище данных (например, финальный ответ)
    - execution: Контекст выполнения со всеми шагами
    """
    memory: Dict[str, Any] = Field(default_factory=dict)
    execution: ExecutionContext = Field(default_factory=ExecutionContext)