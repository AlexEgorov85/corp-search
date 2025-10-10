# tests/agents/test_planner_agent.py
# coding: utf-8
"""
Тесты для упрощённого PlannerAgent (только декомпозиция на подвопросы).
"""
import json
import pytest
from src.agents.PlannerAgent.core import PlannerAgent
from src.model.agent_result import AgentResult

@pytest.fixture
def planner_descriptor():
    return {
        "name": "PlannerAgent",
        "title": "Планировщик (PlannerAgent)",
        "description": "Декомпозирует вопрос на подвопросы.",
        "implementation": "src.agents.PlannerAgent.core:PlannerAgent",
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
            }
        }
    }

# --- Тест 1: Успешная декомпозиция ---
def test_decomposition_success(planner_descriptor, simple_tool_registry):
    def mock_llm_callable(*, messages, **kwargs):
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        return json.dumps({
            "subquestions": [
                {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []},
                {"id": "q2", "text": "Какая из этих книг — последняя?", "depends_on": ["q1"]},
                {"id": "q3", "text": "Кто главный герой этой книги?", "depends_on": ["q2"]}
            ]
        })

    # Создаём объект, который callable и принимает messages=...
    class CallableMock:
        def __call__(self, *, messages, **kwargs):
            return mock_llm_callable(messages=messages, **kwargs)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("src.services.llm_service.factory.ensure_llm", lambda profile: CallableMock())
        agent = PlannerAgent(planner_descriptor, config={"llm_profile": "mock"})
        result = agent.execute_operation("plan", {
            "question": "Найди книги Пушкина и укажи главного героя в последней из них?",
            "tool_registry_snapshot": simple_tool_registry
        }, {})

    assert result.status == "ok"

# --- Тест 2: Ошибка LLM (невалидный JSON) ---
def test_decomposition_invalid_json(planner_descriptor, simple_tool_registry):
    """Тест: LLM возвращает не-JSON."""
    def bad_llm_call(*args, **kwargs):
        return "Это не JSON!"

    cfg = {
        "llm_profile": "mock",
        "tool_registry_snapshot": simple_tool_registry
    }

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("src.services.llm_service.factory.ensure_llm", lambda profile: bad_llm_call)
        agent = PlannerAgent(planner_descriptor, config=cfg)
        params = {"question": "Найди книги Пушкина."}
        result = agent.execute_operation("plan", params, {})

    assert isinstance(result, AgentResult)
    assert result.status == "error"
    assert result.content is not None
    assert "декомпозиц" in result.content.lower()


# --- Тест 3: Валидация корректной декомпозиции ---
def test_validate_decomposition_success(planner_descriptor, simple_tool_registry):
    """Тест: валидация успешной декомпозиции."""
    agent = PlannerAgent(planner_descriptor, config={"tool_registry_snapshot": simple_tool_registry})
    # .initialize() больше не нужен!

    valid_decomposition = {
        "subquestions": [
            {"id": "q1", "text": "Какие книги написал Пушкин?", "depends_on": []}
        ]
    }
    params = {"plan": valid_decomposition}
    result = agent.execute_operation("validate_plan", params, {})
    assert isinstance(result, AgentResult)
    assert result.status == "ok"
    assert result.structured["ok"] is True
    assert result.structured["issues"] == []

# --- Тест 4: Валидация с циклической зависимостью ---
def test_validate_decomposition_cycle(planner_descriptor, simple_tool_registry):
    """Тест: валидация декомпозиции с циклом."""
    agent = PlannerAgent(planner_descriptor, config={"tool_registry_snapshot": simple_tool_registry})
    # .initialize() больше не нужен!

    cyclic_decomposition = {
        "subquestions": [
            {"id": "q1", "text": "Вопрос 1", "depends_on": ["q2"]},
            {"id": "q2", "text": "Вопрос 2", "depends_on": ["q1"]}
        ]
    }
    params = {"plan": cyclic_decomposition}
    result = agent.execute_operation("validate_plan", params, {})
    assert isinstance(result, AgentResult)
    assert result.status == "ok"
    assert result.structured["ok"] is False
    assert any("циклическ" in issue["message"].lower() for issue in result.structured["issues"])