# src/agents/BooksLibraryAgent/operations/dynamic_query.py
"""
Операция: динамический запрос через LLM → SQL.

Используется, когда Reasoner не может сопоставить подвопрос с предопределённой операцией.
Примеры:
- "Сколько всего книг в библиотеке?"
- "Какой автор написал больше всего книг?"

Этапы:
1. Генерация SQL через LLM.
2. Валидация SQL против схемы БД и allowed_tables.
3. При провале валидации — повторная генерация с учётом ошибок (до 3 попыток).
4. Выполнение валидного SQL.

Безопасность:
- Только SELECT-запросы.
- Ограничение LIMIT.
- Валидация схемы.
"""
from src.agents.operations_base import BaseOperation, OperationKind
from src.services.results.agent_result import AgentResult
from src.agents.BooksLibraryAgent.prompt import sql_generation_prompt, sql_retry_prompt
from src.agents.BooksLibraryAgent.validation import validate_sql_against_schema
import time

class Operation(BaseOperation):
    kind = OperationKind.SEMANTIC
    description = "Выполнить динамический запрос к БД через генерацию SQL с валидацией и retry."
    params_schema = {
        "question": {"type": "string", "required": True},
        "max_retries": {"type": "integer", "required": False}
    }
    outputs_schema = {"type": "array"}

    def run(self, params: dict, context: dict, agent) -> AgentResult:
        if agent.llm is None:
            return AgentResult.error("LLM не инициализирована для dynamic_query")
        if not hasattr(agent, 'engine') or agent.engine is None:
            return AgentResult.error("База данных недоступна")

        question = params["question"]
        max_retries = min(int(params.get("max_retries", 3)), 5)

        # Обновляем схему и получаем текст
        agent.refresh_schema()
        schema_text = agent.get_schema_text()
        allowed_tables = ", ".join(sorted(agent.allowed_tables))

        last_messages = []
        last_sql = ""

        for attempt in range(1, max_retries + 1):
            # --- Генерация SQL ---
            if attempt == 1:
                prompt = sql_generation_prompt(schema_text, question, allowed_tables)
            else:
                retry_msgs = "\n".join(last_messages)
                prompt = sql_retry_prompt(
                    problems_text=retry_msgs,
                    previous_sql=last_sql,
                    allowed_tables=allowed_tables,
                    hint="Исправьте ошибки и верните ТОЛЬКО SQL SELECT."
                )

            try:
                raw_sql = agent.llm.generate(prompt)
                sql = raw_sql.strip()
                if not sql:
                    raise ValueError("LLM вернула пустой SQL")
            except Exception as e:
                last_messages.append(f"Ошибка генерации SQL (попытка {attempt}): {e}")
                if attempt == max_retries:
                    return AgentResult.error("; ".join(last_messages))
                continue

            last_sql = sql

            # --- Валидация SQL ---
            ok, messages, _ = validate_sql_against_schema(sql, agent.allowed_tables, agent._schema)
            if ok:
                # Выполняем и возвращаем успех
                try:
                    result = agent.execute_sql(sql)
                    return AgentResult.ok(structured=result["rows"])
                except Exception as e:
                    return AgentResult.error(f"Ошибка выполнения SQL: {e}")

            # Валидация провалена
            last_messages.extend(messages)
            if attempt == max_retries:
                return AgentResult.error(
                    "Не удалось сгенерировать валидный SQL. Ошибки: " + "; ".join(last_messages)
                )
            # иначе — продолжаем цикл

        return AgentResult.error("Неизвестная ошибка в dynamic_query")