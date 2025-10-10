# tests/graph/nodes/test_executor_node.py
"""
Unit-тесты для executor_node.
Покрывают:
1. Успешное выполнение инструмента (validate_entities, fetch_data, process_data).
2. Отсутствие current_call.
3. Отсутствие step_id.
4. Ошибка инстанцирования инструмента.
5. Ошибка выполнения операции.
6. Инструмент вернул ошибку (AgentResult.status != "ok").
"""

import pytest
from unittest.mock import Mock, patch
from src.graph.context_model import GraphContext
from src.graph.context_ops import ensure_step
from src.graph.nodes.executor import executor_node
from src.model.agent_result import AgentResult


@pytest.fixture
def base_context():
    """Базовый контекст с одним шагом."""
    ctx = GraphContext(question="Тестовый вопрос")
    ensure_step(ctx, "q1", subquestion_text="Тестовый подвопрос")
    return ctx


@pytest.fixture
def mock_agent_registry():
    """Мок реестра агентов."""
    registry = Mock()
    return registry


# === Сценарий 1: Успешное выполнение (validate_entities) ===
def test_executor_success_validate_entities(base_context, mock_agent_registry):
    """Тест: успешный вызов validate_author."""
    # Подготавливаем current_call
    from src.graph.context_ops import set_current_call
    set_current_call(base_context, {
        "action": "proceed",
        "tool_call": {
            "tool": "BooksLibraryAgent",
            "operation": "validate_author",
            "params": {"candidates": ["Пушкин"]}
        }
    }, step_id="q1")

    # Мокаем инструмент
    mock_tool = Mock()
    mock_tool.execute_operation.return_value = AgentResult.ok(
        structured={"authors": [{"name": "Пушкин, Александр Сергеевич"}]}
    )
    mock_agent_registry.instantiate_agent.return_value = mock_tool

    # Вызываем executor
    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    # Проверки
    step = ctx.execution.steps["q1"]
    assert step.raw_output == {"authors": [{"name": "Пушкин, Александр Сергеевич"}]}
    assert ctx.execution.current_call is None
    assert any(e["type"] == "executor_success" for e in ctx.execution.history)


# === Сценарий 2: Успешное выполнение (fetch_data) ===
def test_executor_success_fetch_data(base_context, mock_agent_registry):
    """Тест: успешный вызов list_books."""
    from src.graph.context_ops import set_current_call
    set_current_call(base_context, {
        "action": "proceed",
        "tool_call": {
            "tool": "BooksLibraryAgent",
            "operation": "list_books",
            "params": {"author": "Пушкин"}
        }
    }, step_id="q1")

    mock_tool = Mock()
    mock_tool.execute_operation.return_value = AgentResult.ok(
        structured=[{"title": "Евгений Онегин"}, {"title": "Руслан и Людмила"}]
    )
    mock_agent_registry.instantiate_agent.return_value = mock_tool

    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    step = ctx.execution.steps["q1"]
    assert step.raw_output == [{"title": "Евгений Онегин"}, {"title": "Руслан и Людмила"}]
    assert ctx.execution.current_call is None


# === Сценарий 3: Успешное выполнение (process_data) ===
def test_executor_success_process_data(base_context, mock_agent_registry):
    """Тест: успешный вызов map_summarize_documents."""
    from src.graph.context_ops import set_current_call
    set_current_call(base_context, {
        "action": "proceed",
        "tool_call": {
            "tool": "DataAnalysisAgent",
            "operation": "map_summarize_documents",
            "params": {"data": [{"text": "Текст документа"}]}
        }
    }, step_id="q1")

    mock_tool = Mock()
    mock_tool.execute_operation.return_value = AgentResult.ok(
        structured={"summary": "Краткое содержание..."}
    )
    mock_agent_registry.instantiate_agent.return_value = mock_tool

    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    step = ctx.execution.steps["q1"]
    assert step.raw_output == {"summary": "Краткое содержание..."}
    assert ctx.execution.current_call is None


# === Сценарий 4: Отсутствие current_call ===
def test_executor_no_current_call(base_context, mock_agent_registry):
    """Тест: нет current_call → ничего не делаем."""
    # current_call не установлен
    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    assert ctx.execution.current_call is None
    assert any(e["type"] == "executor_no_call" for e in ctx.execution.history)


# === Сценарий 5: Отсутствие step_id ===
def test_executor_no_step_id(base_context, mock_agent_registry):
    """Тест: current_call без step_id и без current_subquestion_id."""
    from src.graph.context_ops import set_current_call
    set_current_call(base_context, {
        "action": "proceed",
        "tool_call": {"tool": "TestAgent", "operation": "test_op"}
    }, step_id=None)
    # Также очищаем current_subquestion_id
    base_context.execution.current_subquestion_id = None

    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    assert ctx.execution.current_call is None
    assert any(e["type"] == "executor_no_step_id" for e in ctx.execution.history)


# === Сценарий 6: Ошибка инстанцирования инструмента ===
def test_executor_instantiate_error(base_context, mock_agent_registry):
    """Тест: ошибка при создании инструмента."""
    from src.graph.context_ops import set_current_call
    set_current_call(base_context, {
        "action": "proceed",
        "tool_call": {"tool": "NonExistentAgent", "operation": "test_op"}
    }, step_id="q1")

    mock_agent_registry.instantiate_agent.side_effect = ValueError("Agent not found")

    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    step = ctx.execution.steps["q1"]
    assert step.status == "failed"
    assert "Instantiate error" in step.error
    assert step.error_stage == "instantiate"
    assert ctx.execution.current_call is None


# === Сценарий 7: Ошибка выполнения операции ===
def test_executor_execute_error(base_context, mock_agent_registry):
    """Тест: исключение при вызове execute_operation."""
    from src.graph.context_ops import set_current_call
    set_current_call(base_context, {
        "action": "proceed",
        "tool_call": {"tool": "BooksLibraryAgent", "operation": "list_books"}
    }, step_id="q1")

    mock_tool = Mock()
    mock_tool.execute_operation.side_effect = RuntimeError("DB connection failed")
    mock_agent_registry.instantiate_agent.return_value = mock_tool

    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    step = ctx.execution.steps["q1"]
    assert step.status == "failed"
    assert "Execute error" in step.error
    assert step.error_stage == "execute"
    assert ctx.execution.current_call is None


# === Сценарий 8: Инструмент вернул ошибку ===
def test_executor_tool_returned_error(base_context, mock_agent_registry):
    """Тест: инструмент вернул AgentResult со статусом error."""
    from src.graph.context_ops import set_current_call
    set_current_call(base_context, {
        "action": "proceed",
        "tool_call": {"tool": "BooksLibraryAgent", "operation": "list_books"}
    }, step_id="q1")

    mock_tool = Mock()
    mock_tool.execute_operation.return_value = AgentResult.error("No books found")
    mock_agent_registry.instantiate_agent.return_value = mock_tool

    result = executor_node(base_context.to_legacy_state(), agent_registry=mock_agent_registry)
    ctx = GraphContext.from_state_dict(result)

    step = ctx.execution.steps["q1"]
    assert step.status == "failed"
    assert "No books found" in step.error
    assert step.error_stage == "tool"
    assert ctx.execution.current_call is None
    