# src/model/agent_result.py
"""
Универсальный результат выполнения операции агента.

Этот класс служит стандартным контрактом для всех операций во всей системе.
Он содержит не только данные, но и **семантическую информацию** о том,
что было сделано, чтобы Reasoner мог принимать осознанные решения.

Основные поля:
  - status: 'ok' | 'error'
  - stage: этап жизненного цикла (entity_validation, data_fetch, ...)
  - entity_type: тип сущности (author, book_title, genre и т.д.) — только для этапов валидации
  - agent: имя агента, который выполнил операцию (например, "BooksLibraryAgent")
  - operation: имя операции (например, "validate_author")
  - input_params: исходные параметры операции (для аудита и отладки)
  - output: основной результат операции (структурированные данные)
  - summary: краткое текстовое резюме того, что было сделано (человекочитаемое)
  - metadata: дополнительная информация (sql, timing, source и т.д.)
  - error: текст ошибки при status='error'
  - ts: временная метка выполнения
  - thinking: рассуждения модели (если применимо)
  - prompt: сформированный промт, отправленный в LLM
  - raw_response: сырой ответ модели (без обработки)
  - tokens_used: оценка использованных токенов
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional
import time


@dataclass
class AgentResult:
    """
    Универсальный результат выполнения операции агента.
    
    Этот класс служит стандартным контрактом для всех операций во всей системе.
    Он содержит не только данные, но и **семантическую информацию** о том,
    что было сделано, чтобы Reasoner мог принимать осознанные решения.

    Основные поля:
      - status: 'ok' | 'error'
      - stage: этап жизненного цикла (entity_validation, data_fetch, ...)
      - entity_type: тип сущности (author, book_title, genre и т.д.) — только для этапов валидации
      - agent: имя агента, который выполнил операцию (например, "BooksLibraryAgent")
      - operation: имя операции (например, "validate_author")
      - input_params: исходные параметры операции (для аудита и отладки)
      - output: основной результат операции (структурированные данные)
      - summary: краткое текстовое резюме того, что было сделано (человекочитаемое)
      - metadata: дополнительная информация (sql, timing, source и т.д.)
      - error: текст ошибки при status='error'
      - ts: временная метка выполнения
      - thinking: рассуждения модели (если применимо)
      - prompt: сформированный промт, отправленный в LLM
      - raw_response: сырой ответ модели (без обработки)
      - tokens_used: оценка использованных токенов
    """
    # Статус выполнения
    status: str  # "ok" или "error"

    # === Семантический контекст выполнения ===
    stage: Optional[str] = None          # "entity_validation", "data_fetch", "data_processing", "result_validation"
    entity_type: Optional[str] = None    # "author", "book_title", "genre", ...

    # === Источник результата ===
    agent: Optional[str] = None          # Имя агента (например, "BooksLibraryAgent")
    operation: Optional[str] = None      # Имя операции (например, "validate_author")

    # === Вход и выход операции ===
    input_params: Optional[Dict[str, Any]] = None  # Параметры, с которыми была вызвана операция
    output: Optional[Any] = None                   # Основной результат (структурированные данные)

    # === Человекочитаемое резюме ===
    summary: Optional[str] = None        # Например: "Проведена валидация сущности 'автор'"

    # === Метаданные и служебная информация ===
    metadata: Dict[str, Any] = field(default_factory=dict)  # sql, elapsed_s, source и т.д.
    error: Optional[str] = None        # Текст ошибки (если status == "error")
    ts: float = field(default_factory=time.time)  # Временная метка выполнения

    # === ПОЛЯ ДЛЯ АНАЛИЗА РАБОТЫ LLM ===
    thinking: Optional[str] = None
    """Рассуждения модели (если применимо).
    
    Для моделей с встроенными рассуждениями (Qwen3-Thinking) содержит:
    - Мыслительный процесс модели
    - Обоснование выбора гипотез
    - Анализ доступных данных
    - План действий
    
    Позволяет анализировать, почему модель приняла то или иное решение.
    """
    
    prompt: Optional[str] = None
    """Сформированный промт, отправленный в LLM.
    
    Содержит:
    - Системный промт
    - Контекст выполнения
    - Доступные инструменты
    - Историю вызовов
    
    Позволяет проверять корректность формирования промтов
    и отлаживать проблемы на уровне взаимодействия с LLM.
    """
    
    raw_response: Optional[str] = None
    """Сырой ответ модели (без обработки адаптером).
    
    Содержит полный вывод модели, включая:
    - Рассуждения
    - Финальный ответ
    - Форматирующие теги (если применимо)
    
    Позволяет анализировать, как модель интерпретирует промт
    и какие проблемы возникают при парсинге ответа.
    """
    
    tokens_used: Optional[int] = None
    """Оценка использованных токенов при генерации.
    
    Содержит приблизительное количество токенов:
    - Входного промта
    - Сгенерированного ответа
    
    Позволяет оптимизировать использование ресурсов
    и предотвращать превышение лимитов контекста.
    """

    @classmethod
    def ok(
        cls,
        stage: str,
        output: Any = None,
        summary: Optional[str] = None,
        input_params: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
        operation: Optional[str] = None,
        entity_type: Optional[str] = None,
        thinking: Optional[str] = None,
        prompt: Optional[str] = None,
        raw_response: Optional[str] = None,
        tokens_used: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AgentResult":
        """Создаёт успешный результат с полной информацией для анализа.
        
        Args:
            stage: этап выполнения (planning, reasoning, data_fetch и т.д.)
            output: структурированный результат операции
            summary: краткое человекочитаемое описание результата
            input_params: входные параметры операции (для аудита)
            agent: имя агента, выполняющего операцию
            operation: имя операции
            entity_type: тип сущности (для этапов валидации)
            thinking: рассуждения модели (если применимо)
            prompt: сформированный промт
            raw_response: сырой ответ модели
            tokens_used: оценка использованных токенов
            metadata: дополнительные метаданные
            
        Returns:
            Экземпляр AgentResult с status='ok' и всеми указанными полями
            
        Пример:
            >>> result = AgentResult.ok(
            ...     stage="data_fetch",
            ...     output=[{"title": "Евгений Онегин", "year": 1833}],
            ...     summary="Успешно получены книги Пушкина",
            ...     input_params={"author": "Пушкин"}
            ... )
            >>> result.status
            'ok'
        """
        return cls(
            status="ok",
            stage=stage,
            entity_type=entity_type,
            agent=agent,
            operation=operation,
            input_params=input_params,
            output=output,
            summary=summary,
            metadata=metadata or {},
            error=None,
            thinking=thinking,
            prompt=prompt,
            raw_response=raw_response,
            tokens_used=tokens_used
        )

    @classmethod
    def error(
        cls,
        message: str,
        stage: str,
        input_params: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
        operation: Optional[str] = None,
        entity_type: Optional[str] = None,
        thinking: Optional[str] = None,
        prompt: Optional[str] = None,
        raw_response: Optional[str] = None,
        tokens_used: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AgentResult":
        """Создаёт результат с ошибкой, включая информацию для анализа.
        
        Args:
            message: текст ошибки
            stage: этап выполнения, на котором произошла ошибка
            input_params: входные параметры операции (для аудита)
            agent: имя агента, вызвавшего ошибку
            operation: имя операции, вызвавшей ошибку
            entity_type: тип сущности (для этапов валидации)
            thinking: рассуждения модели до ошибки (если применимо)
            prompt: сформированный промт
            raw_response: сырой ответ модели (если есть)
            tokens_used: оценка использованных токенов
            metadata: дополнительные метаданные
            
        Returns:
            Экземпляр AgentResult с status='error' и всеми указанными полями
            
        Пример:
            >>> result = AgentResult.error(
            ...     message="Параметр 'author' обязателен",
            ...     stage="data_fetch",
            ...     input_params={"book_id": 1}
            ... )
            >>> result.status
            'error'
        """
        return cls(
            status="error",
            stage=stage,
            entity_type=entity_type,
            agent=agent,
            operation=operation,
            input_params=input_params,
            output=None,
            summary=message[:200] if message else None,  # Краткое описание ошибки
            metadata=metadata or {},
            error=message,
            thinking=thinking,
            prompt=prompt,
            raw_response=raw_response,
            tokens_used=tokens_used
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь для сериализации и логирования.
        
        Returns:
            Словарь с полями результата, без None-значений для чистоты логов
            
        Пример:
            >>> result = AgentResult.ok(stage="planning", output={"plan": [...]})
            >>> result.to_dict()
            {
                "status": "ok",
                "stage": "planning",
                "output": {"plan": [...]},
                "summary": None,
                ...
            }
        """
        result = {
            "status": self.status,
            "stage": self.stage,
            "entity_type": self.entity_type,
            "agent": self.agent,
            "operation": self.operation,
            "input_params": self.input_params,
            "output": self.output,
            "summary": self.summary,
            "metadata": self.metadata,
            "error": self.error,
            "ts": self.ts,
            "thinking": self.thinking,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "tokens_used": self.tokens_used
        }
        # Удаляем None-значения для чистоты логов и сериализации
        return {k: v for k, v in result.items() if v is not None}