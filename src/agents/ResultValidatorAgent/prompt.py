# src/agents/ResultValidatorAgent/prompt.py
"""
Промпт для валидации результата с помощью LLM.

LLM получает:
- подвопрос (text),
- сырые данные (raw_output в JSON-формате).

Задача LLM:
- Определить, отвечает ли `raw_output` на `subquestion_text`.
- Вернуть **ТОЛЬКО** валидный JSON в формате:
  {
    "is_valid": true | false,
    "feedback": ["причина1", "причина2", ...]  // пустой массив, если is_valid=true
  }

ВАЖНО:
- Никаких пояснений вне JSON.
- Если результат частично отвечает — считать невалидным.
- Если raw_output пустой или отсутствует — невалиден.
"""

from typing import Any


from typing import Any


def build_validation_prompt(subquestion_text: str, raw_output: Any) -> str:
    import json
    try:
        output_str = json.dumps(raw_output, ensure_ascii=False, indent=2)
    except Exception:
        output_str = str(raw_output)

    return f"""Ты — валидатор результатов в системе автоматического планирования.
Твоя задача — строго определить, отвечает ли предоставленный результат на заданный подвопрос.

### Подвопрос
{subquestion_text}

### Результат выполнения шага (raw_output)
{output_str}

### Инструкция
- Если результат **полностью и однозначно** отвечает на подвопрос — верни {{ "is_valid": true, "feedback": [] }}.
- Если результат отсутствует, пустой, частичный, не относится к вопросу или непонятен — верни {{ "is_valid": false, "feedback": ["причина"] }}.
- Причина должна быть краткой (1 предложение).
- Верни **ТОЛЬКО** валидный JSON без пояснений, markdown или текста вокруг.

Ответ:"""