# src/agents/ReasonerAgent/prompts.py
from __future__ import annotations
from typing import Dict, Any, Optional, List
import json
import textwrap

def build_universal_reasoner_prompt(
    question: str,
    step_outputs: Optional[Dict[str, Any]] = None,
    tool_registry_snapshot: Optional[Dict[str, Any]] = None,
    step_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Формирует промпт для ReasonerAgent с чёткими инструкциями и без избыточных примеров.
    """
    system_content = textwrap.dedent("""\
        ТЫ — ReasonerAgent в ReAct-системе. ТВОЯ ЗАДАЧА — ВЕРНУТЬ ТОЛЬКО ВАЛИДНЫЙ JSON.

        ### 🔍 Что такое гипотеза?
        Гипотеза — это **один конкретный вызов инструмента**, который может помочь ответить на подвопрос.
        Формат:
        {
          "agent": "<имя агента>",
          "operation": "<имя операции>",
          "params": { /* строго по схеме операции */ },
          "confidence": 0.0–1.0,
          "reason": "Краткий ответ на R3–R6",
          "explanation": "Человекочитаемое обоснование (1–2 предложения)"
        }

        ### 📌 Правила
        1. **Количество гипотез**: обычно 1. Максимум — 2–3.
        2. **Параметры**: заполняй ТОЛЬКО из подвопроса или step_outputs. Нельзя выдумывать!
        3. **Если ответ уже есть в step_outputs** → hypotheses = [], selected_hypothesis = -1.
        4. **Если данных недостаточно** → hypotheses = [], selected_hypothesis = -1.

        ### 🧠 Обязательный анализ (ответь на R1–R7 и включи в "reasoning")
        R1. Какой формат ответа ожидается? (список, скаляр, объект, текст)
        R2. Какие данные из предыдущих шагов уже доступны?
        R3. Достаточно ли этих данных, чтобы ответить без нового вызова?
        R4. Какая операция возвращает данные нужного формата?
        R5. Все ли обязательные параметры операции могут быть заполнены?
        R6. Можно ли использовать более простую операцию (DIRECT вместо SEMANTIC)?
        R7. Требуется ли постобработка результата?

        ### 📏 СТРУКТУРА ОТВЕТА (ОБЯЗАТЕЛЬНО СЛЕДУЙ ЭТОЙ СХЕМЕ)
        {
          "reasoning": ["R1: ...", "R2: ...", ..., "R7: ..."],
          "hypotheses": [ ... ],
          "postprocessing": {
            "needed": true|false,
            "confidence": 0.0–1.0,
            "reason": "...",
            "explanation": "Почему требуется/не требуется постобработка?"
          },
          "validation": {
            "needed": true|false,
            "confidence": 0.0–1.0,
            "reason": "...",
            "explanation": "Почему требуется/не требуется валидация?"
          },
          "final_decision": {
            "selected_hypothesis": -1 или индекс,
            "explanation": "Итоговое резюме выбора (1–2 предложения)"
          }
        }

        ### ⚠️ ВАЖНО
        - НИКАКОГО ТЕКСТА ВНЕ JSON.
        - Начни с '{', закончи '}'.
        - Не используй markdown, пояснения, комментарии.
    """)

    # --- Формируем пользовательскую часть ---
    user_parts = [f"### ❓ Подвопрос\n{question}"]

    if step_outputs:
        user_parts.append("### 📤 Результаты других шагов")
        user_parts.append(json.dumps(step_outputs, ensure_ascii=False, indent=2))
    else:
        user_parts.append("### 📤 Результаты других шагов\nНет")

    if tool_registry_snapshot:
        user_parts.append("### 📚 Доступные инструменты")
        user_parts.append(json.dumps(tool_registry_snapshot, ensure_ascii=False, indent=2))
    else:
        user_parts.append("### 📚 Доступные инструменты\nНет")

    if step_state:
        safe_state = {k: v for k, v in step_state.items() if k in ("retry_count", "validation_feedback")}
        if safe_state:
            user_parts.append("### 🧠 Состояние шага")
            user_parts.append(json.dumps(safe_state, ensure_ascii=False, indent=2))

    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_content.strip()}
    ]