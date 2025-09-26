# coding: utf-8
"""
Промпты (шаблоны) для BooksLibraryAgent.
Все промпты на русском, строгие — модель должна вернуть ТОЛЬКО SQL SELECT без пояснений.
"""

from typing import Optional
import textwrap


def sql_generation_prompt(schema_text: str, question: str, allowed_tables: str) -> str:
    """
    Шаблон запроса NL -> SQL.
    - schema_text: краткая текстовая сводка доступной схемы (таблицы и столбцы)
    - question: вопрос на NL
    - allowed_tables: запрошенные/разрешённые таблицы (строка, перечисление)

    Важно: промпт строго требует ОДИН SQL-запрос - SELECT. Любые дополнительные пояснения
    модель должна НЕ генерировать. Если модель вернёт code fence, вызывающий код должен
    его очистить (strip_code_fences).
    """
    tpl = textwrap.dedent(
        """
        У тебя есть доступ к базе данных с описанием схемы (ниже).
        Разрешённые таблицы: {allowed_tables}

        Сводка схемы (только для ориентира):
        {schema_text}

        Задача: по следующему вопросу сформировать ТОЛЬКО ОДИН корректный SQL-запрос (SELECT),
        который решает задачу и использует только разрешённые таблицы.
        Возвращай ТОЛЬКО SQL (ничего лишнего), не добавляй текстовые пояснения.
        Если нужен ORDER или LIMIT — включи их явно.

        Вопрос:
        {question}

        ВАЖНО: возвращай только чистый SQL SELECT. Если нужно — используй JOIN.
        """
    )
    return tpl.format(schema_text=schema_text, question=question, allowed_tables=allowed_tables)


def sql_retry_prompt(problems_text: str, previous_sql: str, allowed_tables: str, hint: Optional[str] = None) -> str:
    """
    Промпт для повтора генерации SQL, когда валидация обнаружила проблемы.
    - problems_text: список проблем/ошибок, обнаруженных в предыдущей валидации (строка)
    - previous_sql: предыдущая версия SQL, нужно её исправить
    - allowed_tables: перечисление разрешённых таблиц
    - hint: опциональная подсказка (например: используйте колонку published_at)

    Требует вернуть ТОЛЬКО НОВЫЙ SQL (SELECT).
    """
    tpl = textwrap.dedent(
        """
        Предыдущий SQL:
        {previous_sql}

        Проблемы, найденные при валидации:
        {problems_text}

        Используй только разрешённые таблицы: {allowed_tables}
        Постарайся исправить SQL, учитывая обнаруженные ошибки. Верни ТОЛЬКО НОВЫЙ SQL (SELECT), без пояснений.
        {hint_block}
        """
    )
    hint_block = f"Подсказка: {hint}" if hint else ""
    return tpl.format(previous_sql=previous_sql, problems_text=problems_text, allowed_tables=allowed_tables, hint_block=hint_block)
