# src/agents/BooksLibraryAgent/core.py
"""
BooksLibraryAgent — агент для предметной области "books/authors".
Все операции вынесены в папку operations/.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional, Set
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from src.agents.base import BaseAgent
from src.common import settings
from src.services.db_service.schema import build_schema_text, refresh_schema_for_tables

LOG = logging.getLogger(__name__)

class BooksLibraryAgent(BaseAgent):
    """
    Агент для доступа к данным по книгам/авторам с NL->SQL функционалом.
    """
    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__(descriptor, config)
        self._engine: Optional[Engine] = None
        self._schema: Dict[str, Dict[str, Any]] = {}
        self._schema_cache_time: float = 0.0
        self.allowed_tables: Set[str] = set()
        self.max_rows: int = 500

        # Инициализация из config
        cfg = self.config.get("config", {}) if isinstance(self.config, dict) else {}
        self.allowed_tables = set([t.lower() for t in cfg.get("allowed_tables", [])])
        db_uri = cfg.get("db_uri", settings.POSTGRES_DSN)

        if db_uri:
            try:
                self._engine = create_engine(db_uri)
                # LOG.info("BooksLibraryAgent: engine создан для db_uri=%s", db_uri)
            except Exception:
                LOG.exception("Ошибка создания engine из db_uri")
        else:
            LOG.warning("BooksLibraryAgent: db_uri отсутствует в config")

    @property
    def engine(self) -> Optional[Engine]:
        return self._engine

    def refresh_schema(self) -> None:
        """Обновить локальный кеш схемы для allowed_tables."""
        if not self.engine:
            self._schema = {}
            return
        try:
            self._schema = refresh_schema_for_tables(self.engine, list(self.allowed_tables))
            self._schema_cache_time = time.time()
        except Exception:
            LOG.exception("Ошибка refresh_schema")
            self._schema = {}

    def get_schema_text(self) -> str:
        """Краткая текстовая сводка схемы для передачи в промпты."""
        return build_schema_text(self._schema)

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        Выполнить SELECT и вернуть словарь {columns, rows}.
        Добавляет LIMIT, если его нет.
        """
        if not self.engine:
            raise RuntimeError("Нет engine для выполнения SQL")
        low = sql.lower()
        safe_sql = sql
        if " limit " not in low:
            safe_sql = f"SELECT * FROM ({sql.rstrip(';')}) AS subq LIMIT {self.max_rows}"
        with self.engine.connect() as conn:
            res = conn.execute(text(safe_sql))
            cols = list(res.keys())
            rows = [dict(zip(cols, row)) for row in res.fetchall()]
        return {"columns": cols, "rows": rows}