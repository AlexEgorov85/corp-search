# tests/agents/test_reasoner_agent_hybrid.py
"""
Интеграционные тесты для ReasonerAgent с гибридной логикой (правила + LLM).
Покрывают все ключевые сценарии из обновлённого ТЗ:

1. use_previous — если предыдущий шаг уже дал подходящий результат → пропускаем этапы.
2. direct → validate_entities обязателен.
3. semantic → validate_entities пропускается.
4. fetch_only — если операция напрямую отвечает на вопрос → пропускаем обработку (processing).
5. Ретрай при провале валидации результата (макс. 2 попытки).
6. Провал после исчерпания лимита ретраев.

Все тесты работают без LLM (через моки и детерминированную логику).
"""

import pytest
from unittest.mock import Mock
from src.agents.ReasonerAgent.operations.reason_step import Operation
from src.model.agent_result import AgentResult


@pytest.fixture
def base_params():
    """
    Базовый набор параметров для вызова reason_step.
    Включает:
      - подвопрос ("Какие книги написал Пушкин?")
      - пустое состояние шага (next_stage = "analyze_question")
      - пустые step_outputs
      - tool_registry_snapshot с двумя операциями:
          * list_books (kind: "direct")
          * dynamic_query (kind: "semantic")
      - operation_hint = None (по умолчанию)
    """
    return {
        "subquestion": {"id": "q1", "text": "Какие книги написал Пушкин?"},
        "step_state": {"next_stage": "analyze_question"},
        "step_outputs": {},
        "tool_registry_snapshot": {
            "BooksLibraryAgent": {
                "operations": {
                    "list_books": {
                        "kind": "direct",
                        "description": "Вернуть список книг по фамилии автора Пушкин."
                    },
                    "dynamic_query": {
                        "kind": "semantic",
                        "description": "Выполнить динамический запрос к БД через генерацию SQL."
                    }
                }
            }
        },
        "operation_hint": None
    }


# === Сценарий 1: use_previous ===
def test_reason_step_use_previous(base_params):
    """
    ТЗ: Если в step_outputs уже есть результат подходящего предыдущего шага,
         Reasoner должен сразу вернуть action="use_previous" и пропустить все этапы.

    Проверка:
      - Подготавливаем step_outputs с непустым результатом от шага "q0".
      - Вызываем reason_step.
      - Ожидаем: action == "use_previous", previous_step_id == "q0".
    """
    base_params["step_outputs"] = {"q0": [{"title": "Евгений Онегин"}]}
    op = Operation()
    result = op.run(base_params, {}, Mock())
    assert result.status == "ok"
    assert result.structured["action"] == "use_previous"
    assert result.structured["previous_step_id"] == "q0"


# === Сценарий 2: direct → validate_entities ===
def test_reason_step_direct_operation_requires_validation(base_params):
    """
    ТЗ: Если выбранная операция имеет kind == "direct",
         Reasoner обязан запустить этап validate_entities.

    Проверка:
      - Указываем operation_hint с операцией list_books (kind: "direct").
      - Вызываем reason_step.
      - Ожидаем: next_stage == "validate_entities".
    """
    base_params["operation_hint"] = {"tool": "BooksLibraryAgent", "operation": "list_books"}
    op = Operation()
    result = op.run(base_params, {}, Mock())
    assert result.status == "ok"
    assert result.structured["next_stage"] == "validate_entities"


# === Сценарий 3: semantic → пропуск validate_entities ===
def test_reason_step_semantic_operation_skips_validation(base_params):
    """
    ТЗ: Если выбранная операция имеет kind == "semantic",
         Reasoner пропускает этап validate_entities и переходит сразу к fetch_data.

    Проверка:
      - Указываем operation_hint с операцией dynamic_query (kind: "semantic").
      - Вызываем reason_step.
      - Ожидаем: next_stage == "fetch_data".
    """
    base_params["operation_hint"] = {"tool": "BooksLibraryAgent", "operation": "dynamic_query"}
    op = Operation()
    result = op.run(base_params, {}, Mock())
    assert result.status == "ok"
    assert result.structured["next_stage"] == "fetch_data"


# === Сценарий 4: fetch_only (пропуск обработки) ===
def test_reason_step_fetch_only_skips_processing(base_params):
    """
    ТЗ: Если операция напрямую отвечает на подвопрос, Reasoner устанавливает skip_processing = True.
    Поскольку решение принимает LLM, в тесте мокаем LLM так, чтобы она вернула skip_processing=true.
    """
    base_params["operation_hint"] = {
        "tool": "BooksLibraryAgent",
        "operation": "list_books",
        "params": {"author": "Пушкин"}
    }
    # Описание оставляем как в реальном реестре
    base_params["tool_registry_snapshot"]["BooksLibraryAgent"]["operations"]["list_books"]["description"] = \
        "Вернуть список книг по фамилии автора (author)."

    # Мокаем LLM: она должна вернуть {"skip_processing": true}
    mock_llm = Mock()
    mock_llm.generate.return_value = '{"skip_processing": true, "reason": "Операция напрямую отвечает на вопрос"}'
    agent = Mock()
    agent.llm = mock_llm

    op = Operation()
    result = op.run(base_params, {}, agent)

    assert result.status == "ok"
    decision = result.structured
    assert decision.get("skip_processing") is True


# === Сценарий 5: Ретрай при провале валидации ===
def test_reason_step_retry_on_validation_failure(base_params):
    """
    ТЗ: Если на этапе finalize валидация результата вернула is_valid=False,
         Reasoner возвращается на analyze_question с увеличенным retry_count.

    Проверка:
      - Подготавливаем step_state с:
          next_stage = "finalize",
          retry_count = 1,
          validation_result = {"is_valid": False, ...}
      - Вызываем reason_step.
      - Ожидаем: action == "retry", next_stage == "analyze_question", retry_count == 2.
    """
    base_params["step_state"] = {
        "next_stage": "finalize",
        "retry_count": 1,
        "validation_result": {"is_valid": False, "issues": ["Неверный формат"]}
    }
    op = Operation()
    result = op.run(base_params, {}, Mock())
    assert result.status == "ok"
    decision = result.structured
    assert decision["action"] == "retry"
    assert decision["next_stage"] == "analyze_question"
    assert decision["retry_count"] == 2


# === Сценарий 6: Провал после 2 ретраев ===
def test_reason_step_fail_after_max_retries(base_params):
    """
    ТЗ: После 2 неудачных попыток (retry_count == 2) Reasoner помечает шаг как failed.

    Проверка:
      - Подготавливаем step_state с retry_count = 2 и проваленной валидацией.
      - Вызываем reason_step.
      - Ожидаем: action == "fail_step".
    """
    base_params["step_state"] = {
        "next_stage": "finalize",
        "retry_count": 2,
        "validation_result": {"is_valid": False, "issues": ["Ошибка"]}
    }
    op = Operation()
    result = op.run(base_params, {}, Mock())
    assert result.status == "ok"
    assert result.structured["action"] == "fail_step"