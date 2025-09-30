# src/agents/BooksLibraryAgent/validation.py
"""
Валидация SQL для BooksLibraryAgent.

Содержит:
 - Функции извлечения используемых таблиц/алиасов и колонок из SQL.
 - Логику проверки использованных таблиц/колонок против кешированной схемы.
 - Проверку соответствия allowed_tables.

Ключевая особенность: корректная поддержка алиасов таблиц (например `books b` -> alias `b` -> base `books`).
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Set, Any
import re
import logging

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# Регексы для извлечения FROM / JOIN и SELECT списка
_from_re = re.compile(r'\bfrom\s+([^\s,()]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?', re.I)
_join_re = re.compile(r'\bjoin\s+([^\s,()]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?', re.I)
_select_re = re.compile(r'\bselect\s+(.*?)\bfrom\b', re.I | re.S)

def extract_tables_and_columns(sql: str) -> Tuple[Set[str], List[Tuple[str, str]], Dict[str, str]]:
    """
    Анализ SQL: возвращает (used_tables, used_columns, alias_map)
      - used_tables: set реальных имён таблиц, найденных в FROM/JOIN (без алиасов)
      - used_columns: список кортежей (table_or_alias, column) — т.к. колонка может быть указана через алиас
      - alias_map: dict alias -> base_table_name
    """
    sql = (sql or "").strip()
    alias_map: Dict[str, str] = {}
    tables: Set[str] = set()
    used_columns: List[Tuple[str, str]] = []

    # Найдём FROM
    for m in _from_re.finditer(sql):
        tbl = m.group(1)
        alias = m.group(2)
        tbl_clean = tbl.split('.')[-1].strip('`"')
        tables.add(tbl_clean.lower())
        if alias:
            alias_map[alias.lower()] = tbl_clean.lower()

    # Найдём JOINs
    for m in _join_re.finditer(sql):
        tbl = m.group(1)
        alias = m.group(2)
        tbl_clean = tbl.split('.')[-1].strip('`"')
        tables.add(tbl_clean.lower())
        if alias:
            alias_map[alias.lower()] = tbl_clean.lower()

    # SELECT: возьмём часть между SELECT и FROM
    sel = ""
    mm = _select_re.search(sql)
    if mm:
        sel = mm.group(1)
    if sel:
        parts = [p.strip() for p in re.split(r',(?![^\\(]*\\))', sel) if p.strip()]
        for p in parts:
            p_no_as = re.split(r'\s+as\s+', p, flags=re.I)[0].strip()
            mcol = re.search(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', p_no_as)
            if mcol:
                used_columns.append((mcol.group(1).lower(), mcol.group(2).lower()))
            else:
                mbare = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)', p_no_as)
                if mbare:
                    used_columns.append(("", mbare.group(1).lower()))

    return tables, used_columns, alias_map

def validate_sql_against_schema(sql: str, allowed_tables: Set[str], schema: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Проверяет SQL:
     - разрешены ли используемые таблицы (с учётом алиасов)
     - существуют ли использованные колонки в кешированной схеме
    Возвращает (ok: bool, messages: List[str], diagnostics: dict)
    """
    ok = True
    messages: List[str] = []
    diagnostics: Dict[str, Any] = {}

    used_tables, used_columns, alias_map = extract_tables_and_columns(sql)

    # Построим множество реальных таблиц
    real_used_tables: Set[str] = set(used_tables)
    for left, col in used_columns:
        if left and left in alias_map:
            real_used_tables.add(alias_map[left])
        elif left:
            real_used_tables.add(left)

    diagnostics["used_tables"] = sorted(list(real_used_tables))
    diagnostics["alias_map"] = alias_map

    # Проверка разрешённых таблиц
    bad_tables = [t for t in real_used_tables if t not in {x.lower() for x in allowed_tables}]
    if bad_tables:
        messages.append("Используются запрещённые/неразрешённые таблицы: " + ", ".join(bad_tables))
        diagnostics["missing_tables"] = bad_tables

    # Проверка колонок
    missing_cols: List[str] = []
    for left, col in used_columns:
        if left:
            tbl = alias_map.get(left, left)
            tbl_meta = schema.get(tbl)
            if not tbl_meta:
                missing_cols.append(f"{col} (table not in cached schema: {left})")
            else:
                col_names = [c["name"].lower() for c in tbl_meta.get("columns", [])]
                if col.lower() not in col_names:
                    missing_cols.append(f"{col} (not in {tbl})")
        else:
            found = any(col.lower() in [c["name"].lower() for c in meta.get("columns", [])] for meta in schema.values())
            if not found:
                missing_cols.append(f"{col} (not found in cached schema)")

    if missing_cols:
        messages.append("Отсутствуют колонки: " + ", ".join(missing_cols))
        diagnostics["missing_columns"] = missing_cols

    ok = len(messages) == 0
    return ok, messages, diagnostics