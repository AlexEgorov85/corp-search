# src/model/context/context.py
"""
Основной класс управления контекстом выполнения графа.
Содержит ВСЮ логику работы с состоянием в виде чистых, самодостаточных методов.
Все методы содержат подробное логирование для отладки и аудита.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.model.agent_result import AgentResult
from src.model.context.models import (
    ExecutionContext,
    Plan,
    StepExecutionState,
    SubQuestion,
)

LOG = logging.getLogger(__name__)


class GraphContext(BaseModel):
    """
    Главный класс для управления состоянием выполнения графа.
    Инкапсулирует всю логику работы с контекстом.
    """

    # === Поля корневого контекста ===
    question: str = ""
    plan: Plan = Field(default_factory=Plan)
    execution: ExecutionContext = Field(default_factory=ExecutionContext)
    memory: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует контекст в словарь для совместимости с LangGraph."""
        return self.model_dump()

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> "GraphContext":
        """Создаёт GraphContext из словаря."""
        ctx = cls()
        if "question" in state_dict:
            ctx.question = state_dict["question"]
        if "plan" in state_dict:
            plan_data = state_dict["plan"]
            if isinstance(plan_data, dict):
                subquestions = [
                    SubQuestion(**sq) for sq in plan_data.get("subquestions", [])
                ]
                ctx.plan = Plan(subquestions=subquestions)
            else:
                ctx.plan = plan_data
        if "execution" in state_dict:
            exec_data = state_dict["execution"]
            steps = {}
            for step_id, step_data in exec_data.get("steps", {}).items():
                steps[step_id] = StepExecutionState(**step_data)
            ctx.execution = ExecutionContext(
                current_step_id=exec_data.get("current_step_id"),
                steps=steps,
                history=exec_data.get("history", []),
            )
        if "memory" in state_dict:
            ctx.memory = state_dict["memory"]
        return ctx

    # ======================================
    # 1. Работа с вопросом и планом ========
    # ======================================

    def set_question(self, question: str) -> None:
        """Устанавливает исходный вопрос."""
        self.question = question
        LOG.info("📝 Установлен исходный вопрос: %s", question[:100])
        self.append_history_event({"type": "question_set", "question": question[:100]})

    def get_question(self) -> str:
        """Возвращает исходный вопрос."""
        return self.question

    def set_plan(self, plan: Plan) -> None:
        """Устанавливает план выполнения."""
        self.plan = plan
        LOG.info("✅ Установлен план с %d подвопросами", len(plan.subquestions))
        self.append_history_event({"type": "plan_set"})

    def get_plan(self) -> Plan:
        """Возвращает план выполнения."""
        return self.plan

    def is_plan_set(self) -> bool:
        """Проверяет, установлен ли план."""
        return len(self.plan.subquestions) > 0

    # ======================================
    # 2. Управление шагами выполнения ======
    # ======================================

    def get_current_step_id(self) -> Optional[str]:
        """Возвращает ID текущего активного шага."""
        return self.execution.current_step_id

    def set_current_step_id(self, step_id: Optional[str]) -> None:
        """Устанавливает текущий шаг выполнения."""
        self.execution.current_step_id = step_id
        if step_id:
            LOG.info("➡️ Установлен текущий шаг: %s", step_id)
            self.append_history_event({"type": "current_step_set", "step_id": step_id})

    def _get_subquestion_by_id(self, step_id: str) -> Optional[SubQuestion]:
        """Вспомогательный метод: найти подвопрос по ID в плане."""
        for sq in self.plan.subquestions:
            if sq.id == step_id:
                return sq
        return None

    def get_subquestion_text(self, step_id: str) -> str:
        """Возвращает текст подвопроса по его ID."""
        sq = self._get_subquestion_by_id(step_id)
        if sq:
            return sq.text
        step = self.get_execution_step(step_id)
        if step and step.text:
            return step.text
        return step_id

    def ensure_execution_step(self, step_id: str) -> StepExecutionState:
        """Гарантирует существование состояния выполнения для шага."""
        if step_id not in self.execution.steps:
            sq = self._get_subquestion_by_id(step_id)
            text = sq.text if sq else step_id
            self.execution.steps[step_id] = StepExecutionState(id=step_id, text=text)
            LOG.debug("🆕 Создано состояние выполнения для шага %s", step_id)
        return self.execution.steps[step_id]

    def get_execution_step(self, step_id: str) -> Optional[StepExecutionState]:
        """Возвращает состояние выполнения шага."""
        return self.execution.steps.get(step_id)

    def is_step_completed(self, step_id: str) -> bool:
        """Проверяет, завершён ли шаг (все этапы выполнены)."""
        return self.is_step_fully_completed(step_id)

    def mark_step_completed(self, step_id: str) -> None:
        """Помечает шаг как завершённый."""
        step = self.get_execution_step(step_id)
        if step:
            step.completed = True
            # LOG.info("🏁 Шаг %s помечен как завершённый", step_id)
            self.append_history_event({"type": "step_completed", "step_id": step_id})

    def all_steps_completed(self) -> bool:
        """Проверяет, завершены ли все шаги в плане."""
        if not self.is_plan_set():
            return True
        for sq in self.plan.subquestions:
            step = self.get_execution_step(sq.id)
            # 🔑 Если шаг не инициализирован или не прошёл через reasoner — не завершён
            if not step or not any(step.expected_stages.values()):
                return False
            if not self.is_step_fully_completed(sq.id):
                return False
        LOG.info("✅ Все шаги завершены")
        return True

    def select_next_step(self) -> Optional[str]:
        """
        Выбирает следующий незавершённый шаг, у которого выполнены зависимости.
        Возвращает ID шага или None, если все завершены.
        """
        if not self.is_plan_set():
            LOG.warning("⚠️ План не установлен, невозможно выбрать следующий шаг")
            return None
        for sq in self.plan.subquestions:
            if self.is_step_fully_completed(sq.id):
                continue
            deps_ok = True
            for dep_id in sq.depends_on:
                if not self.is_step_fully_completed(dep_id):
                    deps_ok = False
                    break
            if deps_ok:
                LOG.debug("➡️ Найден следующий шаг: %s", sq.id)
                return sq.id
        LOG.debug("🔍 Нет незавершённых шагов с выполненными зависимостями")
        return None

    def start_step(self, step_id: str) -> None:
        """Инициализирует шаг как текущий и гарантирует его состояние."""
        self.set_current_step_id(step_id)
        self.ensure_execution_step(step_id)
        LOG.info("🔄 Начато выполнение шага %s: '%s'", step_id, self.get_subquestion_text(step_id))

    # ======================================
    # 3. Управление этапами шага ===========
    # ======================================

    def set_expected_stages(self, step_id: str, stages: Dict[str, bool]) -> None:
        """Устанавливает, какие этапы требуются для шага."""
        step = self.ensure_execution_step(step_id)
        step.expected_stages = stages
        LOG.debug("🔧 Установлены ожидаемые этапы для шага %s: %s", step_id, stages)

    def mark_stage_completed(self, step_id: str, stage: str) -> None:
        """Помечает этап как завершённый."""
        step = self.ensure_execution_step(step_id)
        if stage in step.completed_stages:
            step.completed_stages[stage] = True
            # LOG.info("✅ Этап '%s' завершён для шага %s", stage, step_id)

    def is_stage_completed(self, step_id: str, stage: str) -> bool:
        """Проверяет, завершён ли конкретный этап."""
        step = self.get_execution_step(step_id)
        if not step:
            return False
        return step.completed_stages.get(stage, False)

    def is_step_fully_completed(self, step_id: str) -> bool:
        """Проверяет, завершены ли все ожидаемые этапы шага."""
        step = self.get_execution_step(step_id)
        if not step:
            return False
        # 🔑 Если expected_stages не установлен (все False), шаг НЕ завершён!
        if not any(step.expected_stages.values()):
            return False
        for stage, required in step.expected_stages.items():
            if required and not step.completed_stages.get(stage, False):
                return False
        return True

    def get_current_stage(self, step_id: str) -> str:
        """
        Определяет текущий этап выполнения шага.
        Возвращает: 'data_fetch', 'processing', 'validation' или 'completed'.
        """
        if self.is_step_fully_completed(step_id):
            return "completed"
        step = self.get_execution_step(step_id)
        if not step:
            return "data_fetch"
        if not self.is_stage_completed(step_id, "data_fetch"):
            return "data_fetch"
        if step.expected_stages.get("processing", False) and not self.is_stage_completed(step_id, "processing"):
            return "processing"
        if step.expected_stages.get("validation", False) and not self.is_stage_completed(step_id, "validation"):
            return "validation"
        return "completed"

    # ===================================================
    # 4. Работа с решениями Reasoner и вызовами =========
    # ===================================================

    def record_reasoner_decision(self, step_id: str, decision: Dict[str, Any]) -> None:
        """Сохраняет решение Reasoner и устанавливает expected_stages."""
        step = self.ensure_execution_step(step_id)
        step.decision = decision

        needs_postprocessing = decision.get("needs_postprocessing", False)
        needs_validation = decision.get("needs_validation", True)

        expected_stages = {
            "data_fetch": True,
            "processing": needs_postprocessing,
            "validation": needs_validation,
        }
        self.set_expected_stages(step_id, expected_stages)

        if "hypotheses" in decision and "final_decision" in decision:
            selected_idx = decision["final_decision"].get("selected_hypothesis", 0)
            hypotheses = decision["hypotheses"]
            if 0 <= selected_idx < len(hypotheses):
                step.hypothesis = hypotheses[selected_idx]
                hyp = hypotheses[selected_idx]
                # LOG.info("🧠 Выбрана гипотеза для шага %s: %s.%s (уверенность: %.2f)",
                #          step_id, hyp["agent"], hyp["operation"], hyp["confidence"])

    def get_current_tool_call(self, step_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает вызов инструмента для текущего этапа.
        """
        current_stage = self.get_current_stage(step_id)
        if current_stage == "completed":
            return None

        step = self.get_execution_step(step_id)
        if not step:
            return None

        if current_stage == "data_fetch":
            hyp = step.hypothesis
            if hyp:
                call = {
                    "agent": hyp["agent"],
                    "operation": hyp["operation"],
                    "params": hyp["params"],
                }
                LOG.debug("🛠️ Текущий вызов (data_fetch): %s.%s", call["agent"], call["operation"])
                return call
        elif current_stage == "processing":
            call = {
                "agent": "DataAnalysisAgent",
                "operation": "analyze",
                "params": {
                    "subquestion_text": self.get_subquestion_text(step_id),
                    "raw_output": step.raw_output,
                },
            }
            LOG.debug("🛠️ Текущий вызов (processing): DataAnalysisAgent.analyze")
            return call
        elif current_stage == "validation":
            call = {
                "agent": "ResultValidatorAgent",
                "operation": "validate_result",
                "params": {
                    "subquestion_text": self.get_subquestion_text(step_id),
                    "raw_output": step.raw_output,
                },
            }
            LOG.debug("🛠️ Текущий вызов (validation): ResultValidatorAgent.validate_result")
            return call
        return None
    
    def get_step_hypothesis(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает информацию о выбранной гипотезе для шага."""
        step = self.get_execution_step(step_id)
        return step.hypothesis if step else None

    # ======================================
    # 5. Работа с результатами =============
    # ======================================

    def record_step_result(self, step_id: str, result: Any) -> None:
        """Сохраняет результат операции в состояние шага."""
        step = self.ensure_execution_step(step_id)
        step.raw_output = result
        LOG.debug("📦 Записан результат для шага %s", step_id)

    def record_validation_result(self, step_id: str, result: Any) -> None:
        """Сохраняет результат операции валидации."""
        step = self.ensure_execution_step(step_id)
        step.validation_result = result
        LOG.debug("📦 Записан результат для шага %s", step_id)

    def get_step_result(self, step_id: str) -> Any:
        """Возвращает результат выполнения шага."""
        step = self.get_execution_step(step_id)
        return step.raw_output if step else None

    def get_all_completed_step_results(self) -> Dict[str, Any]:
        """Возвращает результаты всех завершённых шагов."""
        results = {}
        for step_id, step in self.execution.steps.items():
            if self.is_step_fully_completed(step_id) and step.raw_output is not None:
                results[step_id] = step.raw_output
        return results
    
    def get_relevant_step_outputs_for_reasoner(self, step_id: str) -> Dict[str, Any]:
        """
        Возвращает raw_output только для шагов из depends_on текущего подвопроса.
        Используется в reasoner_node для формирования промпта.
        """
        # Находим текущий подвопрос в плане
        current_subq = None
        for sq in self.plan.subquestions:
            if sq.id == step_id:
                current_subq = sq
                break
        if not current_subq:
            return {}

        outputs = {}
        for dep_id in current_subq.depends_on:
            step = self.get_execution_step(dep_id)
            if step and step.raw_output is not None:
                outputs[dep_id] = step.raw_output

        return outputs
    
    def record_agent_call(self, step_id: str, agent_result: AgentResult) -> None:
        """
        Записывает вызов агента в историю шага.
        Вызывается из BaseAgent.execute_operation после выполнения операции.
        
        Args:
            step_id (str): ID шага выполнения
            agent_result (AgentResult): результат выполнения операции
        """
        if not step_id or not agent_result:
            return

        step = self.ensure_execution_step(step_id)
        if not step:
            return

        # Преобразуем AgentResult в сериализуемый словарь
        call_record = {
            "agent": agent_result.agent or "unknown",
            "operation": agent_result.operation or "unknown",
            "status": agent_result.status,
            "stage": agent_result.stage,
            "input_params": agent_result.input_params,
            "output": agent_result.output,
            "summary": agent_result.summary,
            "error": agent_result.error,
            "metadata": agent_result.metadata,
            "timestamp": agent_result.ts,
        }

        step.agent_calls.append(call_record)
        LOG.debug(
            "📝 Записан вызов агента %s.%s для шага %s (статус: %s)",
            call_record["agent"],
            call_record["operation"],
            step_id,
            call_record["status"]
        )

    # ======================================
    # 6. Работа с финальным ответом ========
    # ======================================

    def set_final_answer(self, answer: str) -> None:
        """Устанавливает финальный ответ."""
        self.memory["final_answer"] = answer
        LOG.info("🎯 Установлен финальный ответ: %.200s", answer)
        self.append_history_event({"type": "final_answer_set"})

    def get_final_answer(self) -> Optional[str]:
        """Возвращает финальный ответ."""
        return self.memory.get("final_answer")

    # ======================================
    # 7. Вспомогательные методы ============
    # ======================================

    def append_history_event(self, event: Dict[str, Any]) -> None:
        """Добавляет событие в историю выполнения."""
        event_with_ts = event.copy()
        event_with_ts["timestamp"] = datetime.utcnow().isoformat()
        self.execution.history.append(event_with_ts)

    # ====
    def get_step_state_for_validation(self, step_id: str) -> Dict[str, Any]:
        """
        Возвращает состояние шага для валидатора.
        Используется в get_current_tool_call для передачи в ResultValidatorAgent.
        """
        step = self.get_execution_step(step_id)
        if not step:
            return {}
        return {
            "retry_count": step.retry_count,
            "error": step.error,
            "completed_stages": step.completed_stages,
            "expected_stages": step.expected_stages,
        }