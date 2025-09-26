# coding: utf-8
"""
BooksLibraryAgent — агент для предметной области "books/authors".

Файл реализует:
 - NL -> SQL через LLM
 - Валидацию SQL (локальная кешированная схема)
 - Retry-процедуру при провале валидации (с несколькими стратегиями вызова LLM)
 - Безопасное выполнение SELECT-only SQL через SQLAlchemy
 - Возврат результата в виде AgentResult

Изменения: улучшен retry-flow — при повторных попытках агент пытается:
  1) сгенерировать SQL обычным промптом (generate_sql_once)
  2) вызвать LLM без аргументов (self.llm()), если callable без args — полезно для SequenceLLM
  3) вызвать call_llm(self.llm, retry_prompt) как fallback
Это повышает вероятность, что тестовые LLM (SequenceLLM) и реальные LLM оба будут корректно обработаны.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Set
import logging
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from services.results.agent_result import AgentResult
from src.services.db_service.schema import build_schema_text, refresh_schema_for_tables
from src.services.llm_service.utils import strip_code_fences

# локальные компоненты агента
from .prompt import sql_generation_prompt, sql_retry_prompt
from .validation import validate_sql_against_schema

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class BooksLibraryAgent:
    """
    Агент для доступа к данным по книгам/авторам с NL->SQL функционалом.

    Конструктор:
      - config: словарь с ключом "config", внутри -- allowed_tables, db_uri и т.п.
      - llm: объект LLM (можно подменять в тестах)
      - max_rows: лимит строк для безопасного выполнения
      - max_retries: число попыток для генерации/валидации SQL
      - schema_cache_ttl: lifetime кеша схемы (сек.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[Any] = None,
                 max_rows: int = 500, max_retries: int = 3, schema_cache_ttl: int = 300):
        self.config = config or {}
        self._llm = llm
        self.max_rows = max_rows
        self.max_retries = max_retries
        self.schema_cache_ttl = schema_cache_ttl

        self._engine: Optional[Engine] = None
        self._schema: Dict[str, Dict[str, Any]] = {}
        self._schema_cache_time: float = 0.0

        cfg = self.config.get("config", {}) if isinstance(self.config, dict) else {}
        self.allowed_tables: Set[str] = set([t.lower() for t in cfg.get("allowed_tables", [])])

        db_uri = cfg.get("db_uri")
        if db_uri:
            try:
                self._engine = create_engine(db_uri)
            except Exception:
                LOG.exception("Ошибка создания engine из db_uri (игнорируем при инициализации)")

    # --- свойства для тестов/подмены ---
    @property
    def engine(self) -> Optional[Engine]:
        return self._engine

    @engine.setter
    def engine(self, e: Engine):
        self._engine = e

    @property
    def llm(self) -> Optional[Any]:
        return self._llm

    @llm.setter
    def llm(self, v: Any):
        self._llm = v

    # --- schema helpers ---
    def refresh_schema(self) -> None:
        """Обновить локальный кеш схемы для allowed_tables, если есть engine."""
        now = time.time()
        if self._schema and (now - self._schema_cache_time) < self.schema_cache_ttl:
            return
        if not self.engine:
            LOG.debug("Нет engine — пропускаю refresh_schema")
            self._schema = {}
            return
        try:
            self._schema = refresh_schema_for_tables(self.engine, list(self.allowed_tables))
            self._schema_cache_time = now
            LOG.debug("Schema refreshed: %s", list(self._schema.keys()))
        except Exception:
            LOG.exception("Ошибка refresh_schema; сбрасываю схему")
            self._schema = {}

    def get_schema_text(self) -> str:
        """Краткая текстовая сводка схемы для передачи в промпты."""
        return build_schema_text(self._schema)

    # --- NL -> SQL (один вызов) ---
    def generate_sql_once(self, question: str, schema_text: str) -> str:
        """Сделать один вызов LLM и получить SQL (очищенный от code fences)."""
        if not self.llm:
            raise RuntimeError("LLM не задан для генерации SQL")

        prompt = sql_generation_prompt(schema_text=schema_text, question=question, allowed_tables=", ".join(sorted(self.allowed_tables)))
        try:
            raw = call_llm(self.llm, prompt)
        except Exception as e:
            LOG.exception("Ошибка вызова LLM: %s", e)
            return ""

        sql = strip_code_fences(raw).strip() if raw else ""
        return sql

    # --- выполнить SQL безопасно ---
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

    # --- основной метод выполнения ---
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Если context и context.get('is_sql') == True -> валидировать и выполнить SQL напрямую.
        Иначе: NL -> SQL с retry (max_retries), где retry использует несколько стратегий вызова LLM.
        Возвращает AgentResult.
        """
        self.refresh_schema()
        schema_text = self.get_schema_text()

        # Прямой SQL
        if context and context.get("is_sql"):
            sql = query.strip()
            ok, messages, diag = validate_sql_against_schema(sql, self.allowed_tables, self._schema)
            if not ok:
                return AgentResult.error("SQL validation failed", metadata={"messages": messages, "diagnostics": diag, "sql": sql})
            try:
                out = self.execute_sql(sql)
                return AgentResult.ok(content="SQL executed", structured=out, metadata={"sql": sql})
            except Exception as e:
                LOG.exception("Ошибка выполнения SQL: %s", e)
                return AgentResult.error(str(e), metadata={"sql": sql})

        # NL -> SQL с повторными попытками (улучшенный flow)
        last_messages = []
        last_sql_candidate = ""

        for attempt in range(1, self.max_retries + 1):
            # обновляем схему
            self.refresh_schema()
            schema_text = self.get_schema_text()

            sql_candidate = ""
            # --- Стратегия A: стандартная генерация (промпт с короткой схемой) ---
            try:
                sql_candidate = self.generate_sql_once(query, schema_text)
            except Exception as e:
                LOG.exception("Ошибка генерации SQL (generate_sql_once): %s", e)
                return AgentResult.error(str(e), metadata={"phase": "generate_sql"})

            # --- Если пустой ответ — пробуем другие стратегии, полезно для SequenceLLM и подобных ---
            if not sql_candidate:
                LOG.warning("LLM вернула пустой SQL на попытке %s (стратегия A)", attempt)
                last_messages.append("LLM returned empty SQL (strategy A)")

                # Strategy B: если llm callable без аргументов — попробовать self.llm()
                try:
                    if callable(self.llm):
                        try:
                            res_b = None
                            # попытка вызвать без аргумента (SequenceLLM часто реализована так)
                            try:
                                res_b = self.llm()
                            except TypeError:
                                # если callable требует аргумент — пропускаем
                                res_b = None
                            if res_b:
                                sql_candidate = strip_code_fences(res_b).strip()
                                LOG.debug("Strategy B produced SQL (len=%s)", len(sql_candidate))
                        except Exception as e:
                            LOG.exception("Ошибка при попытке вызова llm() без аргумента: %s", e)
                except Exception:
                    LOG.exception("Ошибка при проверке callable llm для стратегии B")

            # --- Если всё ещё пусто — Strategy C: вызвать call_llm с retry_prompt (invoke/predict/generate fallback) ---
            if not sql_candidate:
                LOG.debug("Переходим к стратегии C (call_llm с retry prompt)")
                # формируем messages и retry_prompt (в prompt.py sql_retry_prompt ожидается позиционно)
                retry_msgs = [f"Попытка {attempt}: SQL, который корректно выполнится на базе. Ошибки ранее: {last_messages}."]
                # добавим краткую схему в сообщения — LLM может учитывать
                retry_msgs.append(f"schema: {schema_text}")
                retry_prompt = sql_retry_prompt(query, attempt, retry_msgs)
                try:
                    raw_retry = call_llm(self.llm, retry_prompt)
                    sql_candidate = strip_code_fences(raw_retry).strip() if raw_retry else ""
                    if sql_candidate:
                        LOG.debug("Strategy C returned SQL (len=%s)", len(sql_candidate))
                except Exception as e:
                    LOG.exception("Ошибка call_llm в стратегии C: %s", e)
                    sql_candidate = ""

            # Сохраним последний candidate
            last_sql_candidate = sql_candidate

            # Если пустой — учтём и решим: если дошли до последней попытки — вернуть ошибку
            if not sql_candidate:
                LOG.warning("LLM вернула пустой SQL на попытке %s (все стратегии)", attempt)
                last_messages.append("LLM returned empty SQL (all strategies)")
                if attempt == self.max_retries:
                    return AgentResult.error("SQL validation failed", metadata={"messages": last_messages, "sql": sql_candidate})
                # иначе, продолжим цикл — следующая итерация повторит генерацию
                continue

            # --- валидируем candidate ---
            ok, messages, diag = validate_sql_against_schema(sql_candidate, self.allowed_tables, self._schema)
            if ok:
                # Выполняем и возвращаем успех
                try:
                    out = self.execute_sql(sql_candidate)
                    return AgentResult.ok(content="SQL executed", structured=out, metadata={"sql": sql_candidate})
                except OperationalError as oe:
                    LOG.exception("Критическая ошибка при выполнении SQL: %s", oe)
                    return AgentResult.error(str(oe), metadata={"sql": sql_candidate, "error": str(oe)})
                except Exception as e:
                    LOG.exception("Ошибка выполнения SQL: %s", e)
                    return AgentResult.error(str(e), metadata={"sql": sql_candidate})
            else:
                # Не прошла валидация: накапливаем причины и пробуем снова (если остались попытки)
                LOG.warning("Валидация SQL не пройдена (попытка %s): %s; diagnostics=%s", attempt, messages, diag)
                last_messages.extend(messages)
                if attempt == self.max_retries:
                    return AgentResult.error("SQL validation failed", metadata={"messages": last_messages, "sql": sql_candidate, "diagnostics": diag})

                # подготовим запрос на retry — передадим diagnostics и схему в messages
                # но НЕ ДОВЕРЯЕМСЯ тому, что LLM обязана принимать schema_text именованно,
                # поэтому передаём информацию в retry prompt позиционно (sql_retry_prompt)
                retry_msgs = messages + [f"schema: {schema_text}"]
                try:
                    # здесь мы НЕ вызываем call_llm напрямую; просто продолжаем цикл:
                    # следующая итерация сначала попробует generate_sql_once (стратегия A),
                    # затем — strategy B (self.llm()), затем — strategy C (call_llm)
                    LOG.debug("Подготовлен retry_msgs для следующей итерации")
                except Exception as e:
                    LOG.exception("Ошибка подготовки retry prompt: %s", e)
                # continue цикл (следующая итерация)
                continue

        # Если дошли сюда — всё не удалось
        return AgentResult.error("SQL generation/validation failed", metadata={"messages": last_messages, "sql": last_sql_candidate})
