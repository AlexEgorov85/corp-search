# tests/graph/test_react_graph.py
# coding: utf-8
"""
Тесты для React-графа (LangGraph).
Проверяет корректную сборку и выполнение полного цикла.
"""
import pytest
from unittest.mock import Mock, patch
from src.graph.react_graph import build_compiled_graph


@pytest.fixture
def mock_agents_config():
    """Мок конфигурации агентов."""
    return {
        "PlannerAgent": {
            "name": "PlannerAgent",
            "implementation": "mock",
            "operations": {"plan": {"kind": "direct", "description": "test"}},
            "config": {}
        },
        "ReasonerAgent": {
            "name": "ReasonerAgent",
            "implementation": "mock",
            "operations": {"decide": {"kind": "direct", "description": "test"}},
            "config": {}
        },
        "BooksLibraryAgent": {
            "name": "BooksLibraryAgent",
            "implementation": "mock",
            "operations": {"list_books": {"kind": "direct", "description": "test"}},
            "config": {}
        },
        "SynthesizerAgent": {
            "name": "SynthesizerAgent",
            "implementation": "mock",
            "operations": {"execute": {"kind": "direct", "description": "test"}},
            "config": {}
        }
    }


@pytest.fixture
def mock_tool_registry():
    """Мок реестра инструментов."""
    return {
        "BooksLibraryAgent": {
            "title": "Библиотека Книг",
            "description": "Агент для книг.",
            "operations": {
                "list_books": {"description": "Получить список книг"},
            }
        }
    }


def test_react_graph_compilation():
    """Тест: граф компилируется без ошибок."""
    graph = build_compiled_graph()
    assert graph is not None


@patch("src.graph.nodes.planner.AgentRegistry")
@patch("src.graph.nodes.synthesizer.AgentRegistry")
def test_react_graph_full_cycle(
    mock_synth_registry,
    mock_exec_instantiate,
    mock_reason_instantiate,
    mock_plan_registry,
    mock_agents_config,
    mock_tool_registry
):
    # === Мокаем PlannerAgent ===
    mock_planner = Mock()
    mock_planner.execute_operation.return_value.structured = {
        "plan": {
            "subquestions": [
                {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []}
            ]
        }
    }
    mock_plan_registry.return_value.instantiate_agent.return_value = mock_planner

    # === Мокаем ReasonerAgent ===
    mock_reasoner = Mock()
    mock_reasoner.execute_operation.return_value.structured = {
        "action": "call_tool",
        "tool": "BooksLibraryAgent",
        "operation": "list_books",
        "params": {"author": "Пушкин"}
    }
    mock_reason_instantiate.return_value = mock_reasoner

    # === Мокаем Executor (BooksLibraryAgent) ===
    mock_executor = Mock()
    mock_executor.execute_operation.return_value.structured = [
        {"title": "Евгений Онегин", "year": 1833}
    ]
    mock_exec_instantiate.return_value = mock_executor

    # === Мокаем SynthesizerAgent ===
    mock_synthesizer = Mock()
    mock_synthesizer.execute.return_value = {
        "status": "ok",
        "content": "Пушкин написал 'Евгений Онегин'."
    }
    mock_synth_registry.return_value.instantiate_agent.return_value = mock_synthesizer

    # === Запускаем граф ===
    graph = build_compiled_graph()
    init_state = {
        "question": "Какие книги написал Пушкин?",
        "agents_config": mock_agents_config,
        "tool_registry_snapshot": mock_tool_registry
    }
    result = graph.invoke(init_state)

    # === Проверки ===
    assert result["finished"] is True
    assert "final_answer" in result or "synth_output" in result

    # Проверяем, что все агенты были вызваны
    mock_planner.execute_operation.assert_called_once()
    mock_reasoner.execute_operation.assert_called_once()
    mock_executor.execute_operation.assert_called_once()
    mock_synthesizer.execute.assert_called_once()