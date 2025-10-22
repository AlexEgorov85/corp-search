# src/common/tool_registry.py
# coding: utf-8
"""
TOOL_REGISTRY — централизованный реестр инструментальных агентов (AgentEntry).
Файл поддерживает минималистичную, однозначную структуру записи агентов.

AgentEntry (примерный JSON/Python dict)
{
  # Метаданные (обязательное минимум)
  "name": "<string>",             # уникальный идентификатор (повторяет ключ), используется для логов/реестра
  "title": "<string>",            # человекочитаемое имя (UI, документация)
  "description": "<string>",      # краткое назначение + ограничения (для devs/docs)

  # Где находится реализация (обязательное)
  "implementation": "<module:Class|module:function>",  
                                  # формат module:path — executor импортирует и вызывает
                                  # пример: "src.agents.books_facade:BooksLibraryAgent"

  # runtime-конфигурация, опционально; только для создания/инициализации агента (Executor/DevOps)
  "config": { "db_uri": "...", "llm_profile": "default", ... } (optional)

  # internal/meta — НЕ используем для выбора/валидации; только для devs/доков (опционально)
  "meta": { ... }  # туда можно положить internal_pipeline и notes для разработчиков
}

Правила / рекомендации:
- description в операции — обязательное и краткое поле для Planner/Reasoner/доков.
- implementation — обязателен и должен быть импортируем (формат module:attr).
- meta не должен влиять на логику отбора/выполнения инструментов.
"""

from typing import Dict, Any
import os

from src.common import settings

TOOL_REGISTRY: Dict[str, Any] = {
    "BooksLibraryAgent": {
        "name": "BooksLibraryAgent",
        "title": "База книг и авторов (BooksLibraryAgent)",
        "description": (
            "Доступ к данным библиотеки: таблицы authors, books, chapters, genres и book_genres.\n"
            "Read-only операции: поиск книг, получение метаданных, получение глав.\n"
            "Planner должен ссылаться на операции (operation + params). Агент не генерирует SQL на уровне Planner'а."
        ),
        "implementation": "src.agents.BooksLibraryAgent.core:BooksLibraryAgent",
        "config": {
            "db_uri": settings.POSTGRES_DSN,
            "llm_profile": settings.LLM_DEFAULT_PROFILE,
            "allowed_tables": ["authors", "books", "chapters", "genres", "book_genres"],
            "max_rows": 1000
        },
        "meta": {
            "maintainers": ["platform-team"],
            "notes": "Internal: фасадный агент; SQL pipeline реализован внутри агента и логируется."
        }
    },

    "StepResultRelayAgent": {
        "name": "StepResultRelayAgent",
        "title": "Ретранслятор результата шага",
        "description": "Специальный агент для передачи результата одного шага в другой без повторного вызова.",
        "implementation": "src.agents.StepResultRelayAgent.core:StepResultRelayAgent",
        "config": {},
        "meta": {"role": "internal"}
    },

    "ResultValidatorAgent": {
        "name": "ResultValidatorAgent",
        "title": "Валидатор результата (ResultValidatorAgent)",
        "description": "Универсальный агент для проверки, что результат шага отвечает на подвопрос.",
        "implementation": "src.agents.ResultValidatorAgent.core:ResultValidatorAgent",
        "config": {
            "llm_profile": settings.LLM_DEFAULT_PROFILE
            },
        "meta": {
            "role": "control", 
            "maintainers": ["platform-team"]
            }
    },

    "DataAnalysisAgent": {
        "name": "DataAnalysisAgent",
        "title": "Агент аналитики/агрегации (DataAnalysisAgent)",
        "description": (
            "Инструмент для обработки наборов данных/результатов (map/reduce), суммаризации и агрегаций.\n"
            "Используется для post-processing outputs шагов плана и подготовки данных для финального ответа."
        ),
        "implementation": "src.agents.DataAnalysisAgent.core:DataAnalysisAgent",
        "config": {
            "llm_profile": settings.LLM_DEFAULT_PROFILE,
            "max_map_batch": int(os.environ.get("DATAAGENT_MAX_MAP_BATCH", "500"))
        },
        "meta": {
            "maintainers": ["analytics-team"],
            "notes": "Используется для post-processing и подготовки данных для синтеза ответов."
        }
    }
}