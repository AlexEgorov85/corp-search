# src/agents/ReasonerAgent/prompts.py
import textwrap
from typing import Dict, Any

def get_reasoning_prompt(goal: str, tool_registry: Dict[str, Any]) -> str:
    """Генерирует промпт для ANALYZE_QUESTION с извлечением и нормализацией сущностей."""
    tools_desc = "\n".join(
        f"- {name}: {meta.get('title', '')}\n"
        f"  Описание: {meta.get('description', '')}\n"
        f"  Операции: {', '.join(meta.get('operations', {}).keys())}"
        for name, meta in tool_registry.items()
    ) or "(нет доступных инструментов)"

    return textwrap.dedent(f"""
Ты — эксперт по анализу запросов и выбору инструментов.
Задача: для цели «{goal}» выполни следующие действия:
1. Извлеки все сущности (имена, даты, места и т.д.).
2. Определи тип каждой сущности (author, year, genre и т.д.).
3. Нормализуй сущности (исправь опечатки, приведи к полному имени).
4. Выбери лучший инструмент и операцию из списка.
5. Сформулируй точные параметры для вызова.

Доступные инструменты:
{tools_desc}

Верни ТОЛЬКО JSON в формате:
{{
  "entities": [{{"text": "...", "type": "...", "normalized": "..."}}],
  "selected_tool": "...",
  "selected_operation": "...",
  "params": {{...}}
}}
""")

def get_analyze_data_prompt(subquestion_text: str, raw_data: Any) -> str:
    """Генерирует промпт для ANALYZE_DATA."""
    return textwrap.dedent(f"""
Ты — эксперт по анализу данных.
Задача: на основе данных извлеки ответ на вопрос: «{subquestion_text}».
Сырые данные: {raw_data}

Верни ТОЛЬКО JSON в формате:
{{
  "analysis": "структурированный ответ (например, {{\"main_character\": \"...\"}})"
}}
""")

def get_validate_result_prompt(subquestion_text: str, analysis_result: Any) -> str:
    """Генерирует промпт для VALIDATE_RESULT."""
    return textwrap.dedent(f"""
Ты — эксперт-валидатор.
Исходный вопрос: «{subquestion_text}»
Результат анализа: {analysis_result}
Задача: определи, решает ли результат вопрос.
Верни ТОЛЬКО JSON в формате: {{"is_solved": true/false, "reason": "краткое пояснение"}}
""")