# tests/agents/test_reasoner_agent.py
# coding: utf-8
"""
Тесты для ReasonerAgent с поддержкой валидации сущностей.
"""
import json
import pytest
from unittest.mock import Mock
from src.agents.ReasonerAgent.core import ReasonerAgent
from src.agents.ReasonerAgent.stages import process_validation_result, validate_entities
from src.services.results.agent_result import AgentResult


@pytest.fixture
def reasoner_descriptor():
    return {
        "name": "ReasonerAgent",
        "title": "Выборщик инструментов (ReasonerAgent)",
        "description": "Декомпозирует шаги плана и подбирает candidate tools/operations из TOOL_REGISTRY.",
        "implementation": "src.agents.ReasonerAgent.core:ReasonerAgent",
        "operations": {
            "decide": {
                "kind": "direct",
                "description": "Принять решение для подвопроса.",
                "params": {"subquestion": {"type": "object", "required": True}},
                "outputs": {"type": "object"},
            }
        },
        "config": {},
    }


@pytest.fixture
def simple_tool_registry():
    return {
        "BooksLibraryAgent": {
            "title": "Библиотека Книг",
            "description": "Агент для книг.",
            "operations": {
                "list_books": {"description": "Получить список книг"},
                "get_last_book": {"description": "Получить последнюю книгу"},
                "validate_author": {
                    "description": "Валидация автора через семантический поиск",
                    "validates_entity_type": "author"
                },
            }
        }
    }


# --- Тест 1: Успешный анализ подвопроса ---
def test_analyze_question_success(reasoner_descriptor, simple_tool_registry):
    """Тест: успешная генерация анализа подвопроса с извлечением сущностей."""

    def mock_llm_call(messages):
        return json.dumps({
            "entities": [
                {"text": "Пушкин", "type": "author", "normalized": "Александр Сергеевич Пушкин"}
            ],
            "selected_tool": "BooksLibraryAgent",
            "selected_operation": "validate_author",
            "params": {"candidates": ["Александр Сергеевич Пушкин"]}
        })

    cfg = {
        "llm_callable": mock_llm_call,
        "tool_registry_snapshot": simple_tool_registry
    }
    agent = ReasonerAgent(reasoner_descriptor, config=cfg)
    agent.initialize()

    subquestion = {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []}
    params = {
        "subquestion": subquestion,
        "step_outputs": {}
    }
    result = agent.execute_operation("decide", params, {})

    assert isinstance(result, AgentResult)
    assert result.status == "ok"
    assert "entities" in result.structured
    assert result.structured["selected_tool"] == "BooksLibraryAgent"
    assert result.structured["selected_operation"] == "validate_author"


# --- Тест 2: Ошибка LLM (невалидный JSON) ---
def test_analyze_question_invalid_json(reasoner_descriptor, simple_tool_registry):
    """Тест: LLM возвращает не-JSON."""

    def bad_llm_call(messages):
        return "Это не JSON!"

    cfg = {
        "llm_callable": bad_llm_call,
        "tool_registry_snapshot": simple_tool_registry
    }
    agent = ReasonerAgent(reasoner_descriptor, config=cfg)
    agent.initialize()

    subquestion = {"id": "q1", "text": "Какие книги написал Пушкин?"}
    params = {
        "subquestion": subquestion,
        "step_outputs": {}
    }
    result = agent.execute_operation("decide", params, {})

    assert isinstance(result, AgentResult)
    assert result.status == "error"
    # Проверяем, что ошибка зафиксирована в metadata или content не None
    assert result.content is not None or result.metadata is not None
    # Более надёжная проверка: просто убедимся, что это ошибка
    # (в реальном проекте можно проверять тип исключения в metadata)


# --- Тест 3: Валидация сущностей (мок-вызов инструмента) ---
def test_validate_entities_flow(reasoner_descriptor, simple_tool_registry):
    """Тест: проверка, что ReasonerAgent возвращает вызов validate_author."""

    def mock_llm_call(messages):
        if "Извлеки все сущности" in messages[0]["content"]:
            return json.dumps({
                "entities": [{"text": "Пушкин", "type": "author", "normalized": "Пушкин"}],
                "selected_tool": "BooksLibraryAgent",
                "selected_operation": "list_books",
                "params": {"author": "Пушкин"}
            })
        return "{}"

    cfg = {
        "llm_callable": mock_llm_call,
        "tool_registry_snapshot": simple_tool_registry
    }
    agent = ReasonerAgent(reasoner_descriptor, config=cfg)
    agent.initialize()

    subquestion = {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []}
    params = {
        "subquestion": subquestion,
        "step_outputs": {}
    }
    result = agent.execute_operation("decide", params, {})

    assert isinstance(result, AgentResult)
    assert result.status == "ok"
    assert result.structured["selected_tool"] == "BooksLibraryAgent"

def test_validate_entities(reasoner_descriptor, simple_tool_registry):
    """Тест: ReasonerAgent возвращает вызов validate_author."""
    entities = [{"text": "Пушкн", "type": "author", "normalized": "Пушкн"}]
    
    result = validate_entities(
        subquestion={"id": "q1", "text": "Книги Пушкна?", "depends_on": []},
        entities=entities,
        tool_registry=simple_tool_registry
    )
    
    assert result.status == "ok"
    assert result.structured["tool"] == "BooksLibraryAgent"
    assert result.structured["operation"] == "validate_author"
    assert result.structured["params"]["candidates"] == ["Пушкн"]

def test_process_validation_result(reasoner_descriptor, simple_tool_registry):
    """Тест: обработка результата валидации и формирование финальных параметров."""
    validation_result = {"authors": [{"name": "Пушкин, Александр Сергеевич"}]}
    original_params = {"author": "Пушкн"}
    
    result = process_validation_result(
        subquestion={"id": "q1", "text": "Книги Пушкна?", "depends_on": []},
        validation_result=validation_result,
        original_params=original_params
    )
    
    assert result.status == "ok"
    assert result.structured["params"]["author"] == "Пушкин, Александр Сергеевич"
    assert result.structured["operation"] == "list_books"