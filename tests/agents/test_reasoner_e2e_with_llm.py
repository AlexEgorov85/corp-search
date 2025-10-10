# tests/agents/test_reasoner_e2e_with_llm.py
"""
End-to-end интеграционный тест для ReasonerAgent с использованием мокированной LLM.
Проверяет полный цикл:
  planner → next_subquestion → reasoner (первый этап)
и корректность решений ReasonerAgent на основе промпта и LLM.

Важно: Тесты используют **текущую реализацию** из ProjectJSON.txt:
- ReasonerAgent использует метод `choose_action_from_context` и операцию `"step"`.
- Формат решения LLM соответствует схеме из `prompts.py`.
- PlannerAgent сохраняет план в `ctx.memory["plan"]`.
"""

import json
import pytest
from unittest.mock import Mock, patch
from src.graph.context_model import GraphContext
from src.graph.nodes.planner import planner_node
from src.graph.nodes.next_subquestion import next_subquestion_node
from src.graph.nodes.reasoner import reasoner_node
from src.agents.registry import AgentRegistry


@pytest.fixture
def mock_llm():
    """Мок LLM, который возвращает заранее заданные ответы по промпту."""
    llm = Mock()
    llm.generate = Mock()
    return llm


@pytest.fixture
def agent_registry(mock_llm):
    """Реестр агентов с мокированной LLM."""
    with patch("src.services.llm_service.factory.ensure_llm", return_value=mock_llm):
        registry = AgentRegistry(validate_on_init=False)
        return registry


def _mock_planner_response(question: str):
    """
    Возвращает фиксированный план для тестов.
    Используется для подмены ответа PlannerAgent.
    """
    if "книги Пушкина" in question.lower():
        return {
            "subquestions": [
                {
                    "id": "q1",
                    "text": "Какие книги написал Пушкин?",
                    "depends_on": [],
                    "operation_hint": {
                        "tool": "BooksLibraryAgent",
                        "operation": "list_books",
                        "params": {"author": "Пушкин"}
                    }
                },
                {
                    "id": "q2",
                    "text": "Какая из книг — последняя?",
                    "depends_on": ["q1"]
                },
            ]
        }
    elif "акты" in question.lower():
        return {
            "subquestions": [
                {
                    "id": "q1",
                    "text": "Какие акты содержат упоминания 'необоснованных отказов'?",
                    "depends_on": [],
                    "operation_hint": {
                        "tool": "AktUVAAgent",
                        "operation": "get_act",
                        "params": {"name": "необоснованных отказов"}
                    }
                }
            ]
        }
    return {
        "subquestions": [
            {
                "id": "q1",
                "text": question,
                "depends_on": []
            }
        ]
    }


# === Сценарий 1: use_previous ===
def test_e2e_reasoner_use_previous(agent_registry, mock_llm):
    """
    ТЗ: Если предыдущий шаг уже дал подходящий результат,
         Reasoner должен вернуть action="use_previous" и ссылку на этот шаг.

    Логика теста:
      1. Запускаем planner → next_subquestion для вопроса о книгах Пушкина.
      2. Вручную добавляем в контекст результат предыдущего шага "q0".
      3. Мокаем LLM так, чтобы она вернула решение use_previous.
      4. Вызываем reasoner_node и проверяем, что current_call содержит правильное решение.
    """
    # 1. Запускаем planner
    init_state = {"question": "Какие книги написал Пушкин?"}
    with patch.object(agent_registry, "instantiate_agent") as mock_instantiate:
        planner_mock = Mock()
        # ВАЖНО: возвращаем AgentResult.ok с content и structured
        planner_mock.execute_operation.return_value = Mock(
            status="ok",
            content="plan_generated",
            structured={"plan": _mock_planner_response(init_state["question"])}
        )
        mock_instantiate.return_value = planner_mock
        state_after_planner = planner_node(init_state, agent_registry=agent_registry)

    # 2. next_subquestion выбирает q1
    state_after_next = next_subquestion_node(state_after_planner, agent_registry=None)

    # 3. Подготавливаем контекст: предыдущий шаг q0 уже дал ответ
    ctx = GraphContext.from_state_dict(state_after_next)
    ctx.execution.steps["q0"] = {
        "id": "q0",
        "subquestion_text": "Какие книги написал Пушкина?",
        "status": "finalized",
        "final_result": [{"title": "Евгений Онегин"}],
    }
    ctx.memory["plan"] = ctx.plan

    # 4. Мокаем ответ LLM для Reasoner (старый формат из prompts.py)
    mock_llm.generate.return_value = json.dumps({
        "action": "use_previous",
        "selected_tool": None,
        "previous_output_ref": "q0",
        "run_entity_validation": False,
        "run_data_fetch": False,
        "run_processing": False,
        "run_result_validation": True,
        "reason": "Используем результат предыдущего шага q0"
    })

    # 5. Запускаем reasoner и проверяем результат
    state_after_reasoner = reasoner_node(ctx.to_legacy_state(), agent_registry=agent_registry)
    ctx_final = GraphContext.from_state_dict(state_after_reasoner)

    # В текущей реализации reasoner_node устанавливает decision в current_call
    assert ctx_final.execution.current_call is not None
    decision = ctx_final.execution.current_call.decision
    assert decision["action"] == "use_previous"
    assert decision["previous_output_ref"] == "q0"


# === Сценарий 2: direct операция ===
def test_e2e_reasoner_direct_operation(agent_registry, mock_llm):
    """
    ТЗ: Если операция имеет kind == "direct",
         Reasoner должен установить run_entity_validation = True.

    Логика теста:
      1. Запускаем planner → next_subquestion.
      2. Мокаем LLM так, чтобы она вернула решение с run_entity_validation=True.
      3. Проверяем, что решение содержит этот флаг.
    """
    init_state = {"question": "Какие книги написал Пушкин?"}
    with patch.object(agent_registry, "instantiate_agent") as mock_instantiate:
        planner_mock = Mock()
        planner_mock.execute_operation.return_value = Mock(
            status="ok",
            content="plan_generated",
            structured={"plan": _mock_planner_response(init_state["question"])}
        )
        mock_instantiate.return_value = planner_mock
        state_after_planner = planner_node(init_state, agent_registry=agent_registry)

    state_after_next = next_subquestion_node(state_after_planner, agent_registry=None)
    ctx = GraphContext.from_state_dict(state_after_next)
    ctx.memory["plan"] = ctx.plan

    # Мокаем ответ LLM: direct операция → нужна валидация
    mock_llm.generate.return_value = json.dumps({
        "action": "full_pipeline",
        "selected_tool": {"agent": "BooksLibraryAgent", "operation": "list_books"},
        "previous_output_ref": None,
        "run_entity_validation": True,    # ← direct → валидация обязательна
        "run_data_fetch": True,
        "run_processing": True,
        "run_result_validation": True,
        "reason": "Операция list_books требует валидации автора"
    })

    state_after_reasoner = reasoner_node(ctx.to_legacy_state(), agent_registry=agent_registry)
    ctx_final = GraphContext.from_state_dict(state_after_reasoner)
    assert ctx_final.execution.current_call is not None
    decision = ctx_final.execution.current_call.decision
    assert decision["run_entity_validation"] is True


# === Сценарий 3: semantic операция ===
def test_e2e_reasoner_semantic_operation(agent_registry, mock_llm):
    """
    ТЗ: Если операция имеет kind == "semantic",
         Reasoner должен установить run_entity_validation = False.

    Логика теста:
      1. Запускаем planner для вопроса об актах.
      2. Мокаем LLM так, чтобы она вернула решение с run_entity_validation=False.
      3. Проверяем, что решение содержит этот флаг.
    """
    init_state = {"question": "Какие акты содержат упоминания 'необоснованных отказов'?"}
    with patch.object(agent_registry, "instantiate_agent") as mock_instantiate:
        planner_mock = Mock()
        planner_mock.execute_operation.return_value = Mock(
            status="ok",
            content="plan_generated",
            structured={"plan": _mock_planner_response(init_state["question"])}
        )
        mock_instantiate.return_value = planner_mock
        state_after_planner = planner_node(init_state, agent_registry=agent_registry)

    state_after_next = next_subquestion_node(state_after_planner, agent_registry=None)
    ctx = GraphContext.from_state_dict(state_after_next)
    ctx.memory["plan"] = ctx.plan

    # Мокаем ответ LLM: semantic операция → валидация не нужна
    mock_llm.generate.return_value = json.dumps({
        "action": "full_pipeline",
        "selected_tool": {"agent": "AktUVAAgent", "operation": "get_act"},
        "previous_output_ref": None,
        "run_entity_validation": False,   # ← semantic → валидация не требуется
        "run_data_fetch": True,
        "run_processing": True,
        "run_result_validation": True,
        "reason": "Операция get_act — semantic, валидация не требуется"
    })

    state_after_reasoner = reasoner_node(ctx.to_legacy_state(), agent_registry=agent_registry)
    ctx_final = GraphContext.from_state_dict(state_after_reasoner)
    assert ctx_final.execution.current_call is not None
    decision = ctx_final.execution.current_call.decision
    assert decision["run_entity_validation"] is False


# === Сценарий 4: fetch_only (пропуск обработки) ===
def test_e2e_reasoner_fetch_only_skip_processing(agent_registry, mock_llm):
    """
    ТЗ: Если операция напрямую отвечает на вопрос,
         Reasoner должен установить run_processing = False.

    Логика теста:
      1. Запускаем planner → next_subquestion.
      2. Мокаем LLM так, чтобы она вернула решение с run_processing=False.
      3. Проверяем, что решение содержит этот флаг.
    """
    init_state = {"question": "Какие книги написал Пушкин?"}
    with patch.object(agent_registry, "instantiate_agent") as mock_instantiate:
        planner_mock = Mock()
        planner_mock.execute_operation.return_value = Mock(
            status="ok",
            content="plan_generated",
            structured={"plan": _mock_planner_response(init_state["question"])}
        )
        mock_instantiate.return_value = planner_mock
        state_after_planner = planner_node(init_state, agent_registry=agent_registry)

    state_after_next = next_subquestion_node(state_after_planner, agent_registry=None)
    ctx = GraphContext.from_state_dict(state_after_next)
    ctx.memory["plan"] = ctx.plan

    # Мокаем ответ LLM: операция напрямую отвечает → обработка не нужна
    mock_llm.generate.return_value = json.dumps({
        "action": "fetch_only",
        "selected_tool": {"agent": "BooksLibraryAgent", "operation": "list_books"},
        "previous_output_ref": None,
        "run_entity_validation": True,
        "run_data_fetch": True,
        "run_processing": False,          # ← обработка не нужна
        "run_result_validation": True,
        "reason": "Операция list_books напрямую отвечает на вопрос"
    })

    state_after_reasoner = reasoner_node(ctx.to_legacy_state(), agent_registry=agent_registry)
    ctx_final = GraphContext.from_state_dict(state_after_reasoner)
    assert ctx_final.execution.current_call is not None
    decision = ctx_final.execution.current_call.decision
    assert decision["run_processing"] is False