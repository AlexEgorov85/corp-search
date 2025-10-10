# src/services/results/agent_result.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
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
      - meta дополнительная информация (sql, timing, source и т.д.)
      - error: текст ошибки при status='error'
      - ts: временная метка выполнения
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
    ts: float = field(default_factory=time.time)  # Временная метка

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует объект в обычный словарь для сериализации (в контекст графа, логи и т.д.).
        """
        return asdict(self)

    @classmethod
    def ok(
        cls,
        stage: str,
        output: Any = None,
        entity_type: Optional[str] = None,
        agent: Optional[str] = None,
        operation: Optional[str] = None,
        input_params: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        """
        Создаёт успешный результат.
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
        )

    @classmethod
    def error(
        cls,
        message: str,
        stage: Optional[str] = None,
        agent: Optional[str] = None,
        operation: Optional[str] = None,
        input_params: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Создаёт результат с ошибкой.
        """
        return cls(
            status="error",
            stage=stage,
            agent=agent,
            operation=operation,
            input_params=input_params,
            output=None,
            summary=summary,
            metadata=meta or {},
            error=str(message),
        )