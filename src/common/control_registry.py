# src/common/control_registry.py
# coding: utf-8
"""
CONTROL_REGISTRY — реестр контрольных агентов (Planner/Reasoner/Executor/Synthesizer).
Файл хранит только control-агенты и использует тот же унифицированный формат AgentEntry,
что и TOOL_REGISTRY, чтобы обеспечить консистентность и возможность общей валидации.

AgentEntry (примерный JSON/Python dict)
{
  # Метаданные (обязательное минимум)
  "name": "<string>",             # уникальный идентификатор (повторяет ключ), используется для логов/реестра
  "title": "<string>",            # человекочитаемое имя (UI, документация)
  "description": "<string>",      # краткое назначение + ограничения (для devs/docs)

  # Где находится реализация (обязательное)
  "implementation": "<module:Class|module:function>",  
                                  # формат module:path — executor/importer импортирует и вызывает
                                  # пример: "src.agents.planner:PlannerAgent"

  # Операции — единственный источник правды для функционала агента (обязательное)
  "operations": {
    "<op_name>": {
      "kind": "direct" | "composed",   # enum: значение обязательно; direct = parametric, composed = pipeline
      "description": "<string>",       # обязательное поле — однозначно описывает операцию
      "params": { ... } | "freeform",  # описание параметров (человеческая подсказка / простой schema)
      "outputs": { ... } | "freeform", # ожидаемые выходы (ссылка на типы или простая структура)
      "priority": <int, optional>,     # Tie-breaker/важность (опционально)
      # ПОЛЯ НИЖЕ — ТОЛЬКО ДЛЯ composed операций (опционально)
      "enforce_validation": <bool, optional>,
      "allowed_tables": ["t1","t2", ...] (optional)
    }
  },

  # runtime-конфигурация, опционально; только для создания/инициализации агента (Executor/DevOps)
  "config": { "db_uri": "...", "llm_profile": "default", ... } (optional)

  # internal/meta — НЕ используем для выбора/валидации; только для devs/доков (опционально)
  "meta": { "role": "control", "maintainers": [...], ... }
}

Правила:
- Control agents хранятся отдельно от TOOL_REGISTRY и не рассматриваются как candidate tools при подборе инструментов для шагов.
- operations — единственный источник правды по API control-агентов.
- description в operation — обязательно.
- implementation должен быть импортируем (smoke-test на старте рекомендуется).
"""

from typing import Dict, Any
import os

LLM_DEFAULT_PROFILE = os.environ.get("LLM_DEFAULT_PROFILE", "default")

CONTROL_REGISTRY: Dict[str, Any] = {
    "PlannerAgent": {
        "name": "PlannerAgent",
        "title": "Планировщик (PlannerAgent)",
        "description": "Генерирует детализированный план (stages/steps) на основе входной задачи/вопроса.",
        "implementation": "src.agents.PlannerAgent.core:PlannerAgent",
        "operations": {
            "plan": {
                "kind": "direct",
                "description": "Сгенерировать план (stages/steps) для заданного вопроса/context; "
                               "результат — структурированный план с ссылками на operations из TOOL_REGISTRY.",
                "params": {
                    "question": {"type": "string", "required": True},
                    "context": {"type": "object", "required": False},
                    "agent_config_snapshot": {"type": "object", "required": False}
                },
                "outputs": {
                    "type": "object",
                    "properties": {"plan_id": "string", "stages": "array"}
                },
                "priority": 100
            },
            "validate_plan": {
                "kind": "direct",
                "description": "Проверить план на консистентность: все referenced agents/operations существуют, "
                               "зависимости acyclic, обязательные params указаны.",
                "params": {"plan": {"type": "object", "required": True}},
                "outputs": {"type": "object", "properties": {"ok": "boolean", "issues": "array[string]"}}
            }
        },
        "config": {"llm_profile": LLM_DEFAULT_PROFILE},
        "meta": {"role": "control", "maintainers": ["platform-team"], "notes": "Control agent — не включается в candidate tools."}
    },

    "ReasonerAgent": {
        "name": "ReasonerAgent",
        "title": "Выборщик инструментов (ReasonerAgent)",
        "description": "Декомпозирует шаги плана и подбирает candidate tools/operations из TOOL_REGISTRY для каждого шага.",
        "implementation": "src.agents.ReasonerAgent.core:ReasonerAgent",
        "operations": {
            "select_tools": {
                "kind": "direct",
                "description": "Для данного шага/цели определить список подходящих tool_candidates с ранжированием и rationale.",
                "params": {
                    "step_goal": {"type": "string", "required": True},
                    "available_tools": {"type": "array", "items": "string", "required": True},
                    "policy": {"type": "string", "required": False}
                },
                "outputs": {
                    "type": "array",
                    "items": {"tool": "string", "operation": "string", "score": "number", "rationale": "string"}
                },
                "priority": 90
            },
            "score_candidates": {
                "kind": "direct",
                "description": "Оценить/пересчитать скор кандидатов по метрикам (latency, cost, accuracy) и вернуть перечень с ranks.",
                "params": {"candidates": {"type": "array", "required": True}, "policy": {"type": "string", "required": False}},
                "outputs": {"type": "array", "items": {"tool": "string", "operation": "string", "score": "number"}}
            }
        },
        "config": {"selection_policy": "balanced"},
        "meta": {"role": "control", "maintainers": ["platform-team"]}
    },

    "ExecutorAgent": {
        "name": "ExecutorAgent",
        "title": "Исполнитель плана (ExecutorAgent)",
        "description": "Оркеструет выполнение плана: запускает шаги, отслеживает зависимости, собирает outputs и статусы.",
        "implementation": "src.agents.ExecutorAgent.core:ExecutorAgent",
        "operations": {
            "execute_plan": {
                "kind": "direct",
                "description": "Выполнить plan: поддерживает последовательное и параллельное исполнение шагов в соответствии с зависимостями.",
                "params": {
                    "plan": {"type": "object", "required": True},
                    "run_options": {"type": "object", "required": False}
                },
                "outputs": {"type": "object", "properties": {"results": "object", "status": "string"}},
                "priority": 80
            },
            "cancel_plan": {
                "kind": "direct",
                "description": "Попытаться отменить выполнение плана (если поддерживается исполнителем и шаги cancellable).",
                "params": {"plan_id": {"type": "string", "required": True}},
                "outputs": {"type": "object", "properties": {"ok": "boolean", "message": "string"}}
            }
        },
        "config": {"max_concurrency": int(os.environ.get("EXECUTOR_MAX_CONCURRENCY", "4"))},
        "meta": {"role": "control", "maintainers": ["platform-team"]}
    },

    "SynthesizerAgent": {
        "name": "SynthesizerAgent",
        "title": "Синтезатор ответа (SynthesizerAgent)",
        "description": "Агрегирует результаты шагов плана и формирует финальный ответ/артефакты для пользователя.",
        "implementation": "src.agents.SynthesizerAgent.core:SynthesizerAgent",
        "operations": {
            "synthesize": {
                "kind": "direct",
                "description": "На основании outputs шагов и метаданных плана формирует итоговый ответ и вспомогательные данные (evidence).",
                "params": {"step_outputs": {"type": "object", "required": True}, "plan": {"type": "object", "required": True}},
                "outputs": {"type": "object", "properties": {"final_answer": "string", "evidence": "object"}},
                "priority": 70
            }
        },
        "config": {"llm_profile": LLM_DEFAULT_PROFILE},
        "meta": {"role": "control", "maintainers": ["platform-team"]}
    }
}