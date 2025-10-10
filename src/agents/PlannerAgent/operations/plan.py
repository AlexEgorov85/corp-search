# src/agents/PlannerAgent/operations/plan.py
"""
Операция 'plan' для PlannerAgent.

Эта операция отвечает за генерацию декомпозиции (плана) исходного вопроса на список
атомарных подвопросов. Она использует LLM и встроенный механизм повторных попыток
с валидацией для обеспечения корректности результата.

Согласно новой архитектуре, операция:
1. Принимает параметры через `params`.
2. Возвращает результат строго в виде `AgentResult`.
3. Использует `AgentResult.ok()` для успешного завершения, заполняя семантические поля:
   - `stage`: "planning" — этап генерации плана.
   - `output`: структурированный план (dict).
   - `summary`: краткое человекочитаемое резюме действия.
   - `input_params`: исходные параметры для аудита.
4. Не заполняет поля `agent` и `operation` — это делает BaseAgent автоматически.
"""

from src.agents.operations_base import BaseOperation, OperationKind
from src.model.agent_result import AgentResult
from src.agents.PlannerAgent.decomposition import DecompositionPhase


class Operation(BaseOperation):
    """
    Операция генерации плана.
    """
    # Указываем тип операции как DIRECT (прямой вызов)
    kind = OperationKind.DIRECT
    # Обязательное описание операции
    description = "Сгенерировать декомпозицию вопроса на подвопросы."
    # Схема входных параметров (для документации и валидации на уровне реестра)
    params_schema = {
        "question": {"type": "string", "required": True},
        "tool_registry_snapshot": {"type": "object", "required": True}
    }
    # Схема выходных данных
    outputs_schema = {
        "type": "object",
        "properties": {"subquestions": "array"}
    }

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        """
        Основной метод выполнения операции.

        Args:
            params (dict): Параметры операции.
                - question: Исходный вопрос пользователя (str).
                - tool_registry_snapshot: Снимок реестра инструментов (dict).
            context (dict): Контекст выполнения (не используется в этой операции).
            agent: Экземпляр родительского агента (PlannerAgent), предоставляет доступ к LLM.

        Returns:
            AgentResult: Результат выполнения операции.
        """
        # --- Валидация входных параметров ---
        question = params.get("question")
        if not question or not isinstance(question, str):
            return AgentResult.error(
                message="Параметр 'question' обязателен и должен быть строкой.",
                # Заполняем семантические поля для ошибки
                stage="planning",
                input_params=params
            )

        tool_registry = params.get("tool_registry_snapshot")
        if not tool_registry:
            return AgentResult.error(
                message="Параметр 'tool_registry_snapshot' обязателен для генерации плана.",
                stage="planning",
                input_params=params
            )

        # --- Подготовка LLM-обертки ---
        # Создаем обертку, совместимую с DecompositionPhase, которая ожидает функцию с messages
        def llm_wrapper(messages: list[dict[str, str]]) -> str:
            if agent.llm is None:
                raise RuntimeError("LLM не была инициализирована для агента PlannerAgent.")
            # Форматируем список сообщений в простой промпт для текущего LLM-адаптера
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            return agent.llm.generate(prompt)

        # --- Запуск процесса декомпозиции ---
        decomposition_phase = DecompositionPhase(llm_callable=llm_wrapper, max_retries=3)
        success, decomposition, issues = decomposition_phase.run(question, tool_registry)

        if success:
            # Успешный результат: формируем AgentResult с полным семантическим контекстом
            return AgentResult.ok(
                # Этап выполнения
                stage="planning",
                # Основной структурированный результат
                output={"plan": decomposition},
                # Краткое резюме для логов и отладки
                summary=f"Успешно сгенерирован план из {len(decomposition.get('subquestions', []))} подвопросов.",
                # Сохраняем входные параметры для аудита
                input_params=params
            )
        else:
            # Ошибка: агрегируем все проблемы в читаемое сообщение
            error_msg = "Не удалось сгенерировать валидную декомпозицию. Проблемы: " + "; ".join(
                [issue.get("message", "Неизвестная ошибка") for issue in issues]
            )
            return AgentResult.error(
                message=error_msg,
                stage="planning",
                input_params=params,
                # Дополнительные метаданные с деталями ошибок
                meta={"validation_issues": issues}
            )