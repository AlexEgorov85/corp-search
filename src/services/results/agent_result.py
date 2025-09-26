# coding: utf-8
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional
import time

@dataclass
class AgentResult:
    """
    Простой централизованный результат агента (вариант 2).
    Поля:
      - status: 'ok' или 'error'
      - content: основной результат (строка или структура)
      - structured: опционально — машинно-удобная часть (dict), например {columns, rows}
      - metadata: доп. информация (sql, timing и т.п.)
      - error: текст ошибки при статусе error
      - ts: timestamp (epoch float)
    """
    status: str
    content: Any = None
    structured: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Привести к plain dict для записи в state / логов."""
        return asdict(self)

    @classmethod
    def ok(cls, content: Any = None, structured: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> "AgentResult":
        return cls(status="ok", content=content, structured=structured, metadata=metadata or {}, error=None)

    @classmethod
    def error(cls, message: str, content: Any = None, metadata: Optional[Dict[str, Any]] = None) -> "AgentResult":
        return cls(status="error", content=content, structured=None, metadata=metadata or {}, error=str(message))
