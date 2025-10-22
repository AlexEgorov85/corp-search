# src/agents/ResultValidatorAgent/prompt.py
"""
Промпт для валидации результата с полным контекстом и структурированным выводом.
Новая схема:
{
  "reasoning": [
    "V1: Результат полностью отвечает на подвопрос?",
    "V2: Данные соответствуют ожидаемому формату?",
    "V3: Есть ли противоречия в данных?",
    "V4: Достаточно ли данных для ответа?"
  ],
  "validation": {
    "is_valid": true|false,
    "confidence": 0.0–1.0,
    "reason": "Краткий ответ на V1–V4",
    "explanation": "Человекочитаемое обоснование"
  }
}
"""

from typing import Any, Dict, List, Optional
import json

def build_validation_prompt(
    subquestion_text: str,
    raw_output: Any,
    agent_calls: List[Dict] = None,
    step_state: Dict = None,
) -> str:
    """
    Формирует промпт для валидатора с чёткой структурой ответа.
    """
    # --- Форматируем результат ---
    try:
        output_str = json.dumps(raw_output, ensure_ascii=False, indent=2)
    except Exception:
        output_str = str(raw_output)

    # --- Форматируем историю вызовов ---
    calls_str = "Нет вызовов."
    if agent_calls:
        call_lines = []
        for i, call in enumerate(agent_calls, 1):
            agent_name = call.get("agent", "?")
            op = call.get("operation", "?")
            status = call.get("status", "?")
            summary = call.get("summary", "—")
            error = call.get("error", "")
            if error:
                summary += f" [ОШИБКА: {error}]"
            call_lines.append(f"{i}. [{status}] {agent_name}.{op}: {summary}")
        calls_str = "\n".join(call_lines)

    # --- Форматируем состояние шага ---
    state_str = "Нет данных."
    if step_state:
        try:
            state_str = json.dumps(step_state, ensure_ascii=False, indent=2)
        except Exception:
            state_str = str(step_state)

    return f"""Ты — валидатор результатов в системе автоматического планирования.
Твоя задача — строго определить, отвечает ли предоставленный результат на заданный подвопрос,
**с учётом истории выполнения и текущего состояния шага**.

### Подвопрос
{subquestion_text}

### Результат выполнения шага (raw_output)
{output_str}

### История вызовов агентов
{calls_str}

### Состояние шага
{state_str}

### 📌 Обязательный анализ (ответь на V1–V4 и включи в "reasoning")
V1. Результат полностью отвечает на подвопрос?
V2. Данные соответствуют ожидаемому формату?
V3. Есть ли противоречия в данных?
V4. Достаточно ли данных для ответа?

### 📏 СТРУКТУРА ОТВЕТА
{{
  "reasoning": ["V1: ...", "V2: ...", "V3: ...", "V4: ..."],
  "validation": {{
    "is_valid": true|false,
    "confidence": 0.0–1.0,
    "reason": "Краткий ответ на V1–V4",
    "explanation": "Человекочитаемое обоснование (1–2 предложения)"
  }}
}}

### ⚠️ ВАЖНО
- НИКАКОГО ТЕКСТА ВНЕ JSON.
- Начни с '{{', закончи '}}'.
- Не используй markdown, пояснения, комментарии.
"""