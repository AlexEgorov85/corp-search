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

  # Операции — единственный источник правды для функционала агента (обязательное)
  "operations": {
    "<op_name>": {
      "kind": "direct" | "composed",   # enum: значение обязательно; direct = parametric, composed = pipeline
      "description": "<string>",       # обязательное поле — однозначно описывает операцию
      "params": { ... } | "freeform",  # описание параметров (человеческая подсказка / простой schema)
      "outputs": { ... } | "freeform", # ожидаемые выходы (ссылка на типы или простая структура)
      "priority": <int, optional>,     # Tie-breaker для Reasoner (больше — предпочтительнее)
      # ПОЛЯ НИЖЕ — ТОЛЬКО ДЛЯ composed операций (опционально, но рекомендуется)
      "enforce_validation": <bool, optional>, # если true — composed должна проходить валидацию (security)
      "allowed_tables": ["t1","t2", ...] (optional), # whitelist для composed (если релевантно)
      # "notes" не используется внутри operations — внутренние замечания помещаем в meta
    }
  },

  # runtime-конфигурация, опционально; только для создания/инициализации агента (Executor/DevOps)
  "config": { "db_uri": "...", "llm_profile": "default", ... } (optional)

  # internal/meta — НЕ используем для выбора/валидации; только для devs/доков (опционально)
  "meta": { ... }  # туда можно положить internal_pipeline и notes для разработчиков
}

Правила / рекомендации:
- operations — единственный источник truth по возможностям агента.
- description в операции — обязательное и краткое поле для Planner/Reasoner/доков.
- implementation — обязателен и должен быть импортируем (формат module:attr).
- meta не должен влиять на логику отбора/выполнения инструментов.
"""

from typing import Dict, Any
import os

# runtime values (подхватываем из окружения, если заданы)
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")
LLM_DEFAULT_PROFILE = os.environ.get("LLM_DEFAULT_PROFILE", "default")

TOOL_REGISTRY: Dict[str, Any] = {
    "BooksLibraryAgent": {
        "name": "BooksLibraryAgent",
        "title": "База книг и авторов (BooksLibraryAgent)",
        "description": (
            "Доступ к данным библиотеки: таблицы authors, books, chapters, genres и book_genres.\n"
            "Read-only операции: поиск книг, получение метаданных, получение глав.\n"
            "Planner должен ссылаться на операции (operation + params). Агент не генерирует SQL на уровне Planner'а."
        ),
        # реализация — модуль и класс/фабрика (executor импортирует и инстанцирует)
        "implementation": "src.agents.BooksLibraryAgent.core:BooksLibraryAgent",
        "operations": {
            "list_books": {
                "kind": "direct",
                "description": (
                    "Вернуть список книг по фильтру. Поддерживаемые фильтры: author, year_from, year_to, genre, title_contains."
                ),
                "params": {
                    "author": {"type": "string", "required": False},
                    "year_from": {"type": "integer", "required": False},
                    "year_to": {"type": "integer", "required": False},
                    "genre": {"type": "string", "required": False},
                    "title_contains": {"type": "string", "required": False},
                    "limit": {"type": "integer", "required": False}
                },
                "outputs": {
                    "type": "array",
                    "items": {"id": "integer", "title": "string", "year": "integer", "author_id": "integer"}
                },
                "priority": 30
            },
            "get_last_book": {
                "kind": "direct",
                "description": (
                    "Вернуть метаданные последней (по году) книги указанного автора. В случае равенства годов — "
                    "обрабатывать по параметру tie_break, если он указан."
                ),
                "params": {
                    "author": {"type": "string", "required": True},
                    "tie_break": {"type": "string", "required": False, "description": "например 'latest_by_year_then_id'"}
                },
                "outputs": {
                    "type": "object",
                    "properties": {"id": "integer", "title": "string", "year": "integer", "author_id": "integer"}
                },
                "priority": 40
            },
            "get_book_chapters": {
                "kind": "direct",
                "description": "Вернуть список глав/фрагментов для заданного book_id (включая текст и заголовки).",
                "params": {
                    "book_id": {"type": "integer", "required": True},
                    "max_fragments": {"type": "integer", "required": False, "description": "максимум фрагментов/глав"}
                },
                "outputs": {
                    "type": "array",
                    "items": {"chapter_id": "integer", "title": "string", "text": "string"}
                },
                "priority": 25
            },
            "execute_safe_sql": {
                "kind": "composed",
                "description": (
                    "Выполнить заранее валидированный SQL и вернуть табличный результат. "
                    "SQL должен быть предварительно сгенерирован/проверен компонентами pipeline."
                ),
                "params": {
                    "sql": {"type": "string", "required": True},
                    "params": {"type": "object", "required": False},
                    "max_rows": {"type": "integer", "required": False}
                },
                "outputs": {
                    "type": "array",
                    "items": {"type": "object"}
                },
                "priority": 10,
                "enforce_validation": True,
                "allowed_tables": ["authors", "books", "chapters", "genres", "book_genres"]
            }
        },
        "config": {
            "db_uri": POSTGRES_DSN,
            "llm_profile": LLM_DEFAULT_PROFILE,
            "allowed_tables": ["authors", "books", "chapters", "genres", "book_genres"],
            "max_rows": 1000
        },
        "meta": {
            "maintainers": ["platform-team"],
            "notes": "Internal: фасадный агент; SQL pipeline реализован внутри агента и логируется."
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
        "operations": {
            "map_summarize_documents": {
                "kind": "direct",
                "description": "Применяет summarizer к списку текстовых чанков и возвращает промежуточные результаты (map).",
                "params": {
                    "chunks": {"type": "array", "items": "string", "required": True},
                    "summarizer": {"type": "string", "required": False}
                },
                "outputs": {
                    "type": "array",
                    "items": {"summary": "object"}
                },
                "priority": 20
            },
            "reduce_summaries": {
                "kind": "direct",
                "description": "Агрегирует map-результаты (parts) в итоговую сводку/summary.",
                "params": {
                    "parts": {"type": "array", "items": "object", "required": True}
                },
                "outputs": {
                    "type": "object",
                    "properties": {"summary": "string", "meta": "object"}
                },
                "priority": 20
            }
        },
        "config": {
            "llm_profile": LLM_DEFAULT_PROFILE,
            "max_map_batch": int(os.environ.get("DATAAGENT_MAX_MAP_BATCH", "500"))
        },
        "meta": {
            "maintainers": ["analytics-team"],
            "notes": "Используется для post-processing и подготовки данных для синтеза ответов."
        }
    },
    "AktUVAAgent": {
        "name": "AktUVAAgent",
        "title": "Агент аналитики результатов проверки (AktUVAAgent)",
        "description": (
            "Инструмент для получения инфоримации по проверкам, карманам, актам проверки.\n"
        ),
        "implementation": "src.agents.AktUVAAgent.core:AktUVAAgent",
        "operations": {
            "get_act": {
                "kind": "direct",
                "description": "Получение текста акта по названию проверки (name).",
                "params": {
                    "name": {"type": "string", "required": False}
                },
                "outputs": {
                    "type": "array",
                    "items": {"id": "integer", "title": "string", "date": "date", "content": "string"}
                },
                "priority": 20
            },
            "get_poket": {
                "kind": "direct",
                "description": "Получение текстов актов по карману (poket).",
                "params": {
                    "poket": {"type": "string", "required": False}
                },
                "outputs": {
                    "type": "array",
                    "items": {"id": "integer", "title": "string", "date": "date", "content": "string"}
                },
                "priority": 80
            },
            "find_text": {
                "kind": "direct",
                "description": "Поиск текста или фразы (text) в акте проверки (act_id).",
                "params": {
                    "text": {"type": "string", "required": False},
                    "act_id": {"type": "integer", "required": False}
                },
                "outputs": {
                    "type": "array",
                    "items": {"id": "integer", "title": "string", "date": "date", "content": "string"}
                },
                "priority": 80
            }
        },
        "config": {
            "llm_profile": LLM_DEFAULT_PROFILE,
            "max_map_batch": int(os.environ.get("DATAAGENT_MAX_MAP_BATCH", "500"))
        },
        "meta": {
            "maintainers": ["analytics-team"],
            "notes": "Используется для post-processing и подготовки данных для синтеза ответов."
        }
    }
}

# Конец файла
