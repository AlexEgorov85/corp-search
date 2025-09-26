# src/agents/PlannerAgent/core.py
from __future__ import annotations
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional
from src.agents.base import BaseAgent
from src.services.results.agent_result import AgentResult
from .decomposition import DecompositionPhase
from .decomposition_rules import validate_decomposition

LOG = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """Упрощённый PlannerAgent: только декомпозиция вопроса на подвопросы."""

    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__(descriptor, config or {})
        self._call_llm = self.config.get("llm_callable")
        self._tool_registry_snapshot = self.config.get("tool_registry_snapshot", {})

    def _op_plan(self, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Генерация плана: только декомпозиция."""
        question = params.get("question")
        if not (isinstance(question, str) and question.strip()):
            return self._make_error_result("Параметр 'question' обязателен и должен быть непустой строкой")

        tool_registry = params.get("tool_registry_snapshot") or self._tool_registry_snapshot
        if not tool_registry:
            return self._make_error_result("tool_registry_snapshot обязателен для планирования")

        # === Единственный этап: Декомпозиция вопроса ===
        decomposition_phase = DecompositionPhase(self._call_llm, max_retries=3)
        success, decomposition, issues = decomposition_phase.run(question, tool_registry)
        if not success:
            return self._make_error_result(
                "Не удалось сгенерировать корректную декомпозицию вопроса",
                structured={"issues": issues, "decomposition": decomposition}
            )

        # Генерируем уникальный ID плана
        plan_id = str(uuid.uuid4())
        decomposition["plan_id"] = plan_id

        # Сохраняем план
        try:
            self._save_plan_to_file(decomposition)
        except Exception as e:
            LOG.warning(f"Не удалось сохранить план в файл: {e}")

        # Успешный результат: возвращаем только subquestions
        return AgentResult.ok(content="plan_generated", structured={"plan": decomposition})

    def _save_plan_to_file(self, plan: Dict[str, Any]) -> None:
        """Сохраняет план в JSON-файл в папке plans."""
        plan_id = plan.get("plan_id", "unknown_plan")
        plans_dir = "plans"
        os.makedirs(plans_dir, exist_ok=True)
        file_path = os.path.join(plans_dir, f"{plan_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        LOG.info(f"План сохранен в файл: {file_path}")

    def _op_validate_plan(self, params: Dict[str, Any]) -> AgentResult:
        """Валидация плана (декомпозиции)."""
        plan = params.get("plan")
        if not isinstance(plan, dict):
            return self._make_error_result("Параметр 'plan' обязателен и должен быть словарём")

        tool_registry = params.get("tool_registry_snapshot") or self._tool_registry_snapshot
        ok, issues = validate_decomposition(plan, tool_registry)
        return AgentResult.ok(
            content="validated",
            structured={"ok": ok, "issues": issues}
        )

    def _make_error_result(self, message: str, structured: Optional[Dict[str, Any]] = None) -> AgentResult:
        return AgentResult.error(message=message, content=message, metadata=structured or {})

    def _run_direct_operation(self, op_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        if op_name == "plan":
            return self._op_plan(params, context)
        elif op_name == "validate_plan":
            return self._op_validate_plan(params)
        else:
            return self._make_error_result(f"Неизвестная операция: {op_name}")
        
