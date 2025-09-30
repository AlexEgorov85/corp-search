# tests/agents/test_reasoner_agent_full_cycle.py
"""
Интеграционный тест ReasonerAgent с использованием реального контекста (state).
Проверяет полный цикл ReAct для подвопроса:
1. analyze_question → возвращает вызов validate_author
2. executor → сохраняет validated_params
3. process_validation → возвращает вызов list_books
4. executor → сохраняет raw_output
5. analyze_data → возвращает final_result
6. validate_result → завершает шаг
"""
import pytest
from unittest.mock import Mock
from src.graph.models import PlanModel, SubQuestion, StepState
from src.services.results.agent_result import AgentResult


@pytest.fixture
def mock_tool_registry():
    return {
        "BooksLibraryAgent": {
            "name": "BooksLibraryAgent",
            "title": "Библиотека Книг",
            "description": "Агент для книг.",
            "implementation": "src.agents.BooksLibraryAgent.core:BooksLibraryAgent",
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


@pytest.fixture
def mock_agent_registry(mock_tool_registry):
    registry = Mock()
    registry.tool_registry = mock_tool_registry
    return registry


@pytest.fixture
def initial_state():
    plan = PlanModel(
        plan_id="test_plan",
        subquestions=[
            SubQuestion(id="q1", text="Какие книги написал Пушкин?", depends_on=[]),
            SubQuestion(id="q2", text="Какая из книг — последняя?", depends_on=["q1"]),
            SubQuestion(id="q3", text="Кто главный герой в последней книге?", depends_on=["q2"]),
        ]
    )
    return {
        "question": "Найди книги Пушкина и укажи главного героя в последней из них?",
        "plan": plan.model_dump(),
        "execution": {
            "steps": {},
            "current_subquestion_id": "q1",
        }
    }


def test_reasoner_full_cycle(mock_agent_registry, initial_state, mock_tool_registry):
    from src.graph.nodes.reasoner import reasoner_node
    from src.graph.nodes.executor import executor_node

    # === Мокаем ReasonerAgent.step через execute_operation ===
    def mock_reasoner_execute_operation(operation, params, context=None):
        assert operation == "step"
        exec_state = params.get("execution_state", {})
        next_stage = (
            exec_state.get("next_stage") or
            params.get("step", {}).get("next_stage") or
            "analyze_question"
        ).lower()

        if next_stage == "analyze_question":
            return AgentResult.ok(structured={
                "updated_step": {},
                "decision": {
                    "action": "call_tool",
                    "tool": "BooksLibraryAgent",
                    "operation": "validate_author",
                    "params": {"candidates": ["Пушкин"]}
                },
                "next_stage": "process_validation"
            })

        elif next_stage == "process_validation":
            validation_result = params["validation_result"]
            original_params = params["original_params"]
            canonical = validation_result.get("authors", [{}])[0].get("name", "Пушкин")
            return AgentResult.ok(structured={
                "updated_step": {},
                "decision": {
                    "action": "call_tool",
                    "tool": "BooksLibraryAgent",
                    "operation": "list_books",
                    "params": {"author": canonical}
                },
                "next_stage": "analyze_data"
            })

        elif next_stage == "analyze_data":
            return AgentResult.ok(structured={
                "updated_step": {"final_result": [{"title": "Евгений Онегин"}]},
                "decision": {
                    "action": "final_answer",
                    "answer": [{"title": "Евгений Онегин"}]
                },
                "next_stage": None
            })

        else:
            return AgentResult.error(f"Неизвестная стадия: {next_stage}")

    # === Мокаем BooksLibraryAgent.execute_operation ===
    def mock_books_execute_operation(operation, params, context=None):
        if operation == "validate_author":
            return AgentResult.ok(structured={
                "authors": [{"name": "Пушкин, Александр Сергеевич"}]
            })
        elif operation == "list_books":
            return AgentResult.ok(structured=[
                {"title": "Евгений Онегин", "year": 1833},
                {"title": "Руслан и Людмила", "year": 1820}
            ])
        else:
            return AgentResult.error(f"Неизвестная операция: {operation}")

    # === Настройка instantiate_agent ===
    def mock_instantiate_agent(name, control=False):
        if name == "ReasonerAgent":
            agent = Mock()
            agent.execute_operation = mock_reasoner_execute_operation
            return agent
        elif name == "BooksLibraryAgent":
            agent = Mock()
            agent.execute_operation = mock_books_execute_operation
            return agent
        else:
            raise ValueError(f"Неизвестный агент: {name}")

    mock_agent_registry.instantiate_agent.side_effect = mock_instantiate_agent

    # === Шаг 1: Reasoner → analyze_question → возвращает validate_author ===
    state = initial_state.copy()
    state["execution"]["steps"]["q1"] = StepState(
        id="q1",
        subquestion_text="Какие книги написал Пушкин?",
        status="pending"
    ).model_dump()

    result = reasoner_node(state, mock_agent_registry)
    assert "execution" in result
    assert "current_call" in result["execution"]
    call = result["execution"]["current_call"]
    assert call["decision"]["action"] == "call_tool"
    assert call["decision"]["tool"] == "BooksLibraryAgent"
    assert call["decision"]["operation"] == "validate_author"

    # === Шаг 2: Executor → вызывает validate_author ===
    executor_result = executor_node(result, mock_agent_registry)
    step = executor_result["execution"]["steps"]["q1"]
    assert step["status"] == "executed"
    assert step["raw_output"] == {"authors": [{"name": "Пушкин, Александр Сергеевич"}]}

    # === Шаг 3: Reasoner → process_validation → list_books ===
    result2 = reasoner_node(executor_result, mock_agent_registry)
    call2 = result2["execution"]["current_call"]
    assert call2["decision"]["action"] == "call_tool"
    assert call2["decision"]["tool"] == "BooksLibraryAgent"
    assert call2["decision"]["operation"] == "list_books"
    assert call2["decision"]["params"]["author"] == "Пушкин, Александр Сергеевич"

    # === Шаг 4: Executor → list_books ===
    executor_result2 = executor_node(result2, mock_agent_registry)
    step2 = executor_result2["execution"]["steps"]["q1"]
    assert step2["status"] == "executed"
    assert len(step2["raw_output"]) == 2

    # === Шаг 5: Reasoner → analyze_data → final_answer ===
    result3 = reasoner_node(executor_result2, mock_agent_registry)
    step3 = result3["execution"]["steps"]["q1"]
    assert step3["status"] == "finalized"
    assert "final_result" in step3
    print("✅ Тест пройден: ReasonerAgent корректно работает с контекстом и возвращает решения для executor.")