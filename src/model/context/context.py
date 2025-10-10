# src/model/context/context.py
"""
Основной класс управления контекстом выполнения графа.
Содержит всю логику работы с состоянием в виде методов класса.

ТЕРМИНОЛОГИЯ:
- План (Plan) — декомпозиция вопроса на подвопросы. Не меняется.
- Шаг выполнения (StepExecutionState) — состояние выполнения одного подвопроса.
"""

from __future__ import annotations
from pydantic import Field
from typing import Any, Dict, Optional
from datetime import datetime

from src.model.agent_result import AgentResult
from .base import BaseGraphContext
from .models import Plan, ExecutionContext, StepExecutionState


class GraphContext(BaseGraphContext):
    """
    Главный класс для управления состоянием выполнения графа.
    Инкапсулирует всю логику работы с контекстом.
    """
    # === Поля корневого контекста ===
    question: str = ""
    plan: Plan = Field(default_factory=Plan)
    execution: ExecutionContext = Field(default_factory=ExecutionContext)
    memory: Dict[str, Any] = Field(default_factory=dict)


    # ------------------------------------
    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "GraphContext":
        if isinstance(data, cls):
            return data
        return cls(**data)

    # -------------------------
    # Методы для работы с вопросом и планом
    # -------------------------
    def get_question(self) -> str:
        return self.question

    def set_question(self, question: str) -> None:
        self.question = question

    def get_plan(self) -> Plan:
        return self.plan

    def set_plan(self, plan: Plan) -> None:
        self.plan = plan

    # -------------------------
    # Методы для управления текущим шагом
    # -------------------------
    def get_current_step_id(self) -> Optional[str]:
        return self.execution.current_subquestion_id

    def set_current_step_id(self, step_id: Optional[str]) -> None:
        self.execution.current_subquestion_id = step_id

    def get_subquestion_text(self, step_id: str) -> str:
        """Возвращает текст подвопроса из плана по его ID."""
        for sq in self.plan.subquestions:
            if sq.id == step_id:
                return sq.text
        raise KeyError(f"Подвопрос с ID '{step_id}' не найден в плане.")

    # -------------------------
    # Методы для работы с шагом выполнения
    # -------------------------
    def get_execution_step(self, step_id: str) -> Optional[StepExecutionState]:
        """Возвращает состояние выполнения шага (или None)."""
        return self.execution.subquestions.get(step_id)

    def ensure_execution_step(self, step_id: str) -> StepExecutionState:
        """Создаёт шаг выполнения, если его ещё нет."""
        if step_id not in self.execution.subquestions:
            text = self.get_subquestion_text(step_id)
            step = StepExecutionState(id=step_id, text=text)
            self.execution.subquestions[step_id] = step
        return self.execution.subquestions[step_id]

    def update_step_data(self, step_id: str, **kwargs) -> None:
        """Обновляет поля StepExecutionState."""
        step = self.get_execution_step(step_id)
        if not step:
            return
        for key, value in kwargs.items():
            if not hasattr(step, key):
                raise ValueError(f"Поле '{key}' не существует в StepExecutionState.")
            setattr(step, key, value)

    def mark_step_completed(self, step_id: str) -> None:
        self.update_step_data(step_id, completed=True)

    def mark_step_failed(self, step_id: str, error: str) -> None:
        self.update_step_data(step_id, completed=False, error=error)

    def is_step_completed(self, step_id: str) -> bool:
        """Проверяет, завершён ли шаг."""
        step = self.get_execution_step(step_id)
        return step.completed if step else False

    def get_step_result(self, step_id: str) -> Any:
        """Возвращает результат выполнения шага (raw_output)."""
        step = self.get_execution_step(step_id)
        return step.raw_output if step else None

    # -------------------------
    # Методы для работы с историей
    # -------------------------
    def append_history_event(self, event: Dict[str, Any]) -> None:
        event_with_ts = event.copy()
        event_with_ts["timestamp"] = datetime.utcnow().isoformat()
        self.execution.history.append(event_with_ts)

    # -------------------------
    # Методы для работы с финальным ответом
    # -------------------------
    def get_final_answer(self) -> Any:
        return self.memory.get("final_answer")

    def set_final_answer(self, answer: Any) -> None:
        self.memory["final_answer"] = answer

    # -------------------------
    # Методы для Reasoner
    # -------------------------

    def get_step_state_for_reasoner(self, step_id: str) -> Dict[str, Any]:
        """
        Возвращает словарь с полным состоянием шага, необходимым для Reasoner.
        Используется в reasoner_node для формирования промпта.
        
        Возвращаемые поля:
        - retry_count: int
        - validation_feedback: Optional[List[str]]
        - raw_output: Any (результат предыдущего выполнения)
        - structured: Optional[Dict] (предыдущее решение от Reasoner)
        
        Если шаг не существует — возвращает пустое состояние.
        """
        step = self.get_execution_step(step_id)
        if not step:
            return {
                "retry_count": 0,
                "validation_feedback": None,
                "raw_output": None,
                "structured": None,
                "validation_passed": False,
                "agent_calls": []
            }
        return {
            "retry_count": step.retry_count,
            "validation_feedback": step.error.split("; ") if step.error else None,
            "raw_output": step.raw_output,
            "structured": step.decision,
            "validation_passed": step.validation_passed,
            "agent_calls": step.agent_calls
        }
    
    
    def record_reasoner_decision(self, step_id: str, decision: Dict[str, Any]) -> None:
        """
        Сохраняет решение Reasoner'а в состояние шага.
        Обновляет:
        - decision (structured)
        - retry_count (если указано в решении)
        - validation_feedback (если указано)
        
        Используется в reasoner_node после получения AgentResult от ReasonerAgent.
        """
        if not isinstance(decision, dict):
            raise ValueError("Решение Reasoner должно быть словарём.")
        
        # Обновляем основные поля
        update_fields = {
            "decision": decision,
        }
        
        # Опционально обновляем retry_count и validation_feedback
        if "retry_count" in decision:
            update_fields["retry_count"] = decision["retry_count"]
        if "validation_feedback" in decision:
            update_fields["error"] = "; ".join(decision["validation_feedback"]) if decision["validation_feedback"] else None

        self.update_step_data(step_id, **update_fields)

    def get_all_step_results_for_reasoner(self) -> Dict[str, Any]:
        """
        Возвращает словарь результатов всех завершённых шагов.
        Используется Reasoner для анализа контекста (например, для use_previous).
        
        Формат: {step_id: raw_output}
        """
        results = {}
        for step_id, step in self.execution.subquestions.items():
            if step.completed and step.raw_output is not None:
                results[step_id] = step.raw_output
        return results


    # -------------------------
    # Методы для Executor
    # -------------------------

    def get_tool_call_for_executor(self, step_id: str) -> Optional[Dict[str, Any]]:
        """
        Извлекает selected_tool из решения Reasoner для запуска в Executor.
        
        Ожидаемый формат решения:
        {
            "selected_tool": {
            "agent": "BooksLibraryAgent",
            "operation": "list_books",
            "params": {"author": "Пушкин"}
            },
            ...
        }
        
        Возвращает selected_tool или None, если:
        - шаг не существует
        - решение отсутствует
        - selected_tool отсутствует или null
        """
        step = self.get_execution_step(step_id)
        if not step or not step.decision:
            return None
        selected_tool = step.decision.get("selected_tool")
        if not selected_tool or not isinstance(selected_tool, dict):
            return None
        return selected_tool

    def record_tool_execution_result(self, step_id: str, agent_result: AgentResult) -> None:
        """
        Сохраняет полный результат выполнения инструмента (AgentResult).
        Обновляет:
        - raw_output (если успех)
        - error (если ошибка)
        - agent_calls (всегда!)
        - completed (только при ошибке или finalize)
        """
        if not isinstance(agent_result, AgentResult):
            raise ValueError("Ожидается AgentResult")

        # Добавляем полный результат в историю вызовов
        step = self.ensure_execution_step(step_id)
        step.agent_calls.append(agent_result.to_dict())  # ← сохраняем ВЕСЬ контекст

        if agent_result.status == "error":
            self.update_step_data(
                step_id,
                raw_output=None,
                error=agent_result.error or str(agent_result.content),
                completed=True
            )
        else:
            if agent_result.stage == "result_validation":
                # Это валидация — сохраняем отдельно, НЕ трогаем raw_output
                self.update_step_data(
                    step_id,
                    validation_passed=True,
                    error=None
                )
            else:
                # Это fetch_data, process_data и т.д. — сохраняем как raw_output
                self.update_step_data(
                    step_id,
                    raw_output=agent_result.output,
                    error=None
                )