# src/agents/ReasonerAgent/prompts.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json
import textwrap

def build_universal_reasoner_prompt(
    question: str,
    step_outputs: Optional[Dict[str, Any]] = None,
    tool_registry_snapshot: Optional[Dict[str, Any]] = None,
    step_state: Optional[Dict[str, Any]] = None,
) -> str:
    # === 1. Системная роль (инструкция для LLM) ===
    system_role = textwrap.dedent("""\
        Ты — ReasonerAgent в ReAct-системе.
        Твоя задача — управлять полным жизненным циклом выполнения одного подвопроса.

        ### 🔁 Жизненный цикл шага (только эти этапы!)
        1. **validate_entities** — нормализовать сущности (например, автор → "Пушкин") через валидационные операции.
        2. **fetch_data** — выполнить запрос к инструменту (например, `list_books`).
        3. **process_data** — обработать сырые данные (если нужно агрегировать/фильтровать).
        4. **validate_result** — проверить, что результат отвечает на подвопрос.
        5. **finalize** — завершить шаг (устанавливается автоматически после успешной `validate_result`).

        Этапы выполняются **строго по порядку**. Пропуск возможен **только если этап не нужен**.

        ### 📤 Формат ответа (ОБЯЗАТЕЛЬНО)
        Все поля обязательны. Используй `null`, `true`/`false`, и корректные типы.
        Верни **ТОЛЬКО валидный JSON** по схеме:
        ```json
        {{
          "next_stage": "analyze_question" | "validate_entities" | "fetch_data" | "process_data" | "validate_result" | "finalize" | "use_previous",
          "selected_tool": {{
            "agent": <наименование агента>,
            "operation": <наименование операции>,
            "params": {{<парамаетр>: <значение параметра>,}}
          }} | null,
          "previous_output_ref": "id_шага" | null,
          "run_entity_validation": true | false,
          "run_fetch_data": true | false,
          "run_process_data": true | false,
          "retry_count": 0,
          "validation_feedback": ["ошибка1", ...] | null,
          "reason": "Краткое обоснование (одно предложение)"
        }}
	      ```

        ### 📋 Критерии заполнения полей
        - **`next_stage`** (`string`):  
          Один из: `validate_entities`, `fetch_data`, `process_data`, `validate_result`, `finalize`.  
          Выбери **следующий этап**, который нужно выполнить.  
          - `validate_entities` — если подвопрос содержит сущности (автор, жанр и т.д.), требующие нормализации.  
          - `fetch_data` — когда нужно получить данные из инструмента.  
          - `process_data` — если сырые данные требуют агрегации, фильтрации или преобразования.  
          - `validate_result` — **обязателен** перед `finalize`, чтобы убедиться, что результат отвечает на подвопрос.  
          - `finalize` — только если результат прошёл валидацию (`validation_passed == true`) и шаг завершён.

        - **selected_tool** (`object | null`):  
          Обязателен для этапов `validate_entities`, `fetch_data`, `process_data`.  
          Формат:
            {
              "agent": "ИмяАгента",
              "operation": "имя_операции",
              "params": { ... }
            }
          Для `finalize` — `null`.
                                  
        - **previous_output_ref** (`string | null`):  
          Должен содержать `id` шага из `step_outputs`, результат которого можно использовать напрямую. В данном случае этапы validate_entities и fetch_data можно пропустить.
          Во всех остальных случаях — **`null`**.

        - **run_fetch_data** (`boolean`):  
          `true`, если previous_output_ref == null.
          Если previous_output_ref <> null, то нужно определиться достаточно данных previous_output_ref для ответа или нет, если достаточно то ставь false 
                                  
        - **run_entity_validation** (`boolean`):  
          `true`, если требуется валидация сущностей. 
          Если run_fetch_data == true и на нем требуется запуск операции с типом SEMANTIC, то `false`.
                                  
        - **run_process_data** (`boolean`):  
          `true`, если требуется постобработка (обычно при `process_data`).
          Если результат от операции на fetch_data отвечает на подвопрос, то false.

        - **retry_count** (`integer`):  
          Текущее число попыток. Увеличивай при ретрае.
        
        - **`validation_feedback`** (`array[string] | null`):  
          Содержит список ошибок или замечаний от предыдущей валидации результата (этап `validate_result`).  
          - Устанавливается **только** если предыдущая валидация провалилась.  
          - Должен быть `null`, если валидация не проводилась или прошла успешно.  
          - При ретрае используется для уточнения генерации.

        - **reason** (`string`):  
          **Одно предложение** с обоснованием выбора. Пример:  
          *"Подвопрос содержит сущность 'автор', требуется нормализация."*
                                  
        **❗ КРИТИЧЕСКИЕ ПРАВИЛА:**  
          1. **НЕ ПОВТОРЯЙ УСПЕШНЫЕ ЭТАПЫ.** Если в истории есть запись `[ok]` для комбинации `(next_stage, selected_tool)`, **этот этап уже завершён** — переходи к следующему.  
          2. **ПОСЛЕ УСПЕШНОГО `fetch_data` ВСЕГДА СЛЕДУЕТ `validate_result`**, если только не требуется постобработка (`process_data`).  
          3. **`validate_result` ОБЯЗАТЕЛЕН перед `finalize`.**  
          4. **`selected_tool` НЕ МОЖЕТ быть `null`** на этапах `validate_entities`, `fetch_data`, `process_data`, `validate_result`.
          5. **Если `validation_passed == true`, выбирай `next_stage = "finalize"` и не повторяй `validate_result`.**
    """)

    # === 2. Пользовательский контекст ===
    # --- Подвопрос ---
    user_context = f"### ❓ Подвопрос\n{question}\n\n"

    # --- Результаты других шагов ---
    if step_outputs:
        def _safe_json_dumps(obj, **kwargs):
            """Безопасная сериализация с fallback на str()."""
            try:
                return json.dumps(obj, **kwargs)
            except (TypeError, ValueError):
                return str(obj)

        outputs_text = "\n".join(
            f"- {step_id}: {_safe_json_dumps(out, ensure_ascii=False, indent=2)}"
            for step_id, out in step_outputs.items()
        )
        user_context += f"### 📤 Результаты других шагов\n{outputs_text}\n\n"
    else:
        user_context += "### 📤 Результаты других шагов\nНет.\n\n"

        
    # --- Доступные инструменты ---
    if tool_registry_snapshot:
        tools_text = json.dumps(tool_registry_snapshot, ensure_ascii=False, indent=2)
        user_context += f"### 📚 Доступные инструменты\n{tools_text}\n\n"
    else:
        user_context += "### 📚 Доступные инструменты\nНет доступных инструментов.\n\n"

    # --- Состояние текущего шага ---
    if step_state:
        safe_state = {
            k: v for k, v in step_state.items()
            if k in ("retry_count", "validation_feedback", "raw_output", "structured")
        }
        if "raw_output" in safe_state and safe_state["raw_output"] is not None:
            try:
                preview = json.dumps(safe_state["raw_output"], ensure_ascii=False, indent=2)
                if len(preview) > 500:
                    preview = preview[:497] + "..."
                safe_state["raw_output"] = preview
            except Exception:
                safe_state["raw_output"] = str(safe_state["raw_output"])[:500]
        state_text = json.dumps(safe_state, ensure_ascii=False, indent=2)
        user_context += f"### 🧠 Состояние шага\n{state_text}\n\n"
    else:
        user_context += "### 🧠 Состояние шага\nНет данных.\n\n"

    # --- История вызовов агентов (agent_calls) ---
    agent_calls_text = "Нет вызовов."
    if step_state and step_state.get("agent_calls"):
        calls = step_state["agent_calls"]
        if calls:
            entries = []
            for i, call in enumerate(calls[-3:], 1):  # последние 3 вызова
                stage = call.get("stage", "?")
                agent = call.get("agent", "?")
                op = call.get("operation", "?")
                status = call.get("status", "?")
                summary = call.get("summary", "—")
                error = call.get("error", "")
                if error:
                    summary += f" [ОШИБКА: {error}]"
                entries.append(f"{i}. [{status}] {stage} → {agent}.{op}: {summary}")
            agent_calls_text = "\n".join(entries)
    user_context += f"### 📜 История вызовов агентов (НЕ ПОВТОРЯЙ!)\n{agent_calls_text}\n\n"

    # === 3. Примеры ===
    examples = textwrap.dedent("""\
        ### ✅ Примеры

        #### Пример 1: Валидация автора
        ```json
        {
          "next_stage": "validate_entities",
          "selected_tool": {
            "agent": "BooksLibraryAgent",
            "operation": "validate_author",
            "params": {"candidates": ["Толстой"]}
          },
          "previous_output_ref": null,
          "run_entity_validation": true,
          "run_fetch_data": true,
          "run_process_data": true,
          "retry_count": 0,
          "reason": "Подвопрос содержит сущность 'автор', требуется нормализация."
        }
        ```

        #### Пример 2: Ретрай после ошибки
        ```json
        {
          "next_stage": "validate_entities",
          "selected_tool": {
            "agent": "BooksLibraryAgent",
            "operation": "validate_author",
            "params": {"candidates": ["Толстой"]}
          },
          "previous_output_ref": null,
          "run_entity_validation": true,
          "run_fetch_data": true,
          "run_process_data": true,
          "retry_count": 1,
          "reason": "Предыдущая валидация вернула пустой результат, уточняем имя автора."
        }
        ```
       
        #### Пример 3: Данные получены на предыдущем шаге
        ```json
        {
          "next_stage": "process_data",
          ""selected_tool": {
            "agent": "DataAnalysisAgent",
            "operation": "analyze",
            "params": {}
          },
          "previous_output_ref": q1,
          "run_entity_validation": false,
          "run_fetch_data": false,
          "run_process_data": true,
          "retry_count": 0,
          "reason": "Данных полученных на предыдущем подвопросе достаточно чтобы ответить на данный вопрос, проводим их анализ."
        }
        ```

        #### Пример 4: Завершение шага
        ```json
        {
          "next_stage": "finalize",
          "selected_tool": null,
          "previous_output_ref": null,
          "run_validation": false,
          "run_processing": false,
          "retry_count": 0,
          "reason": "Результат прошёл валидацию, шаг завершён."
        }
        ```
    """)

    # === Сборка финального промпта ===
    full_prompt = f"{system_role.strip()}\n\n{user_context.strip()}\n\n{examples.strip()}"
    return full_prompt