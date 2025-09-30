# src/graph/context.py
# coding: utf-8
"""
GraphContext — единая модель и API для работы с контекстом/состоянием графа.

Цели:
- Явно типизировать часть state, связанную с исполнением (execution).
- Дать однозначный API для записи/чтения current_call, шагов, результатов и ошибок.
- Поддержать обратную совместимость: model_dump() -> plain dict (для внешнего интерфейса).
- Записывать history (события) для удобства отладки и тестирования.

Пример использования (внутри узла):
    from src.graph.context import GraphContext

    # Если у нас dict state (как раньше) — создаём контекст
    ctx = GraphContext.from_state_dict(state)

    # Убедимся, что шаг существует
    ctx.ensure_step("q1", subquestion_text="Какие книги написал Пушкин?")

    # Сохранить решение reasoner'а
    ctx.set_current_call({"action": "call_tool", "tool": "BooksLibraryAgent", "operation": "validate_author"}, step_id="q1")

    # Сериализовать обратно в dict для возвращения из узла
    return ctx.model_dump()
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ----- Модели состояния шага -----
class StepState(BaseModel):
    id: str
    subquestion_text: Optional[str] = None
    status: str = "pending"           # pending | in_progress | done | error
    analysis: Optional[Any] = None
    final_result: Optional[Any] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class CurrentCall(BaseModel):
    decision: Optional[Dict[str, Any]] = None
    step_id: Optional[str] = None
    ts: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True


class ExecutionState(BaseModel):
    current_subquestion_id: Optional[str] = None
    steps: Dict[str, StepState] = Field(default_factory=dict)
    current_call: Optional[CurrentCall] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class GraphContext(BaseModel):
    """
    Общая модель контекста/состояния графа.

    Основные поля:
      - question: исходная пользовательская формулировка
      - execution: ExecutionState (включает steps/current_call/history)
      - final_answer: optional (заполняется Synthesizer'ом или Reasoner'ом при final_answer)
      - synth_output: optional
      - memory: key-value store для кэша/памяти между узлами
    """
    question: Optional[str] = None
    execution: ExecutionState = Field(default_factory=ExecutionState)
    final_answer: Optional[Any] = None
    synth_output: Optional[Any] = None
    memory: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    # ----------------- Адаптеры / конструкторы -----------------
    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "GraphContext":
        """
        Создать GraphContext из legacy dict. Игнорируем неизвестные поля, переносим
        execution.steps (если они есть) в модели StepState.
        """
        # Prepare minimal kwargs
        kw: Dict[str, Any] = {}
        if not isinstance(state, dict):
            # Если передали уже модель — вернём её
            if isinstance(state, GraphContext):
                return state
            raise TypeError("state must be dict or GraphContext")

        kw["question"] = state.get("question")

        # execution
        exec_raw = state.get("execution", {}) or {}
        execution = ExecutionState()
        # current_subquestion_id
        execution.current_subquestion_id = exec_raw.get("current_subquestion_id") or exec_raw.get("current_subquestion")
        # steps: could be dict of StepState-like dicts
        steps_raw = exec_raw.get("steps", {}) or {}
        for sid, v in steps_raw.items():
            try:
                if isinstance(v, StepState):
                    execution.steps[sid] = v
                elif isinstance(v, dict):
                    # ensure there's an 'id'
                    step_kwargs = dict(v)
                    step_kwargs.setdefault("id", sid)
                    execution.steps[sid] = StepState(**step_kwargs)
                else:
                    # fallback: store minimal
                    execution.steps[sid] = StepState(id=sid, subquestion_text=str(v))
            except Exception:
                execution.steps[sid] = StepState(id=sid, subquestion_text=str(v))

        # current_call: if present in legacy as dict, convert
        cur_call = exec_raw.get("current_call")
        if cur_call:
            try:
                cc = CurrentCall(**cur_call) if isinstance(cur_call, dict) else CurrentCall(decision=getattr(cur_call, "decision", None))
                execution.current_call = cc
            except Exception:
                execution.current_call = CurrentCall(decision=cur_call)
        # history
        history_raw = exec_raw.get("history", []) or []
        execution.history = list(history_raw)

        kw["execution"] = execution
        kw["final_answer"] = state.get("final_answer")
        kw["synth_output"] = state.get("synth_output")
        kw["memory"] = state.get("memory", {}) or {}

        return cls(**kw)

    # ----------------- Утилиты для работы с шагами и текущим вызовом -----------------
    def get_step(self, step_id: str) -> Optional[StepState]:
        return self.execution.steps.get(step_id)

    def ensure_step(self, step_id: str, **kwargs) -> StepState:
        """
        Создать шаг, если его нет. Любые дополнительные kwargs передаются в StepState.
        Возвращает объект StepState (модель).
        """
        if step_id not in self.execution.steps:
            self.execution.steps[step_id] = StepState(id=step_id, **(kwargs or {}))
        return self.execution.steps[step_id]

    def set_current_call(self, decision: Dict[str, Any], step_id: Optional[str] = None) -> None:
        """
        Безопасно установить current_call. Запишем событие в history.
        """
        self.execution.current_call = CurrentCall(
            decision=decision,
            step_id=step_id or self.execution.current_subquestion_id,
            ts=datetime.utcnow()
        )
        self.append_history({"type": "set_current_call", "decision": decision, "step_id": self.execution.current_call.step_id})

    def clear_current_call(self) -> None:
        self.execution.current_call = None
        self.append_history({"type": "clear_current_call"})

    def append_history(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", datetime.utcnow())
        self.execution.history.append(event)

    def set_step_result(self, step_id: str, result: Any) -> None:
        """
        Установить результат шага и пометить статус = done.
        Обрезаем длинные значения в history (резюме).
        """
        step = self.ensure_step(step_id)
        step.final_result = result
        step.status = "done"
        summary = result
        try:
            s = str(result)
            summary = s if len(s) < 400 else s[:400] + "..."
        except Exception:
            summary = "<unserializable>"
        self.append_history({"type": "set_step_result", "step_id": step_id, "result_summary": summary})

    def set_step_error(self, step_id: str, error: str, stage: Optional[str] = None) -> None:
        step = self.ensure_step(step_id)
        step.error = error
        step.status = "error"
        step.error_stage = stage
        self.append_history({"type": "set_step_error", "step_id": step_id, "error": error, "stage": stage})

    # ----------------- Вспомогательные методы -----------------
    def to_legacy_state(self) -> Dict[str, Any]:
        """
        Возвращает обратно совместимый dict, который используют текущие узлы/тесты.
        Это wrapper над model_dump(), но гарантирует, что execution.steps — plain dict of dicts.
        """
        base = self.model_dump()
        # pydantic model_dump уже вернёт вложенные dict-ы; но делаем явный проход для гарантии
        steps = {}
        for sid, s in (self.execution.steps or {}).items():
            if hasattr(s, "model_dump"):
                steps[sid] = s.model_dump()
            else:
                steps[sid] = dict(s)
        base.setdefault("execution", {})
        base["execution"]["steps"] = steps
        # current_call may be model — convert
        if self.execution.current_call:
            base["execution"]["current_call"] = (self.execution.current_call.model_dump()
                                                if hasattr(self.execution.current_call, "model_dump")
                                                else dict(self.execution.current_call))
        return base

    # Alias для удобства
    model_dump_legacy = to_legacy_state
