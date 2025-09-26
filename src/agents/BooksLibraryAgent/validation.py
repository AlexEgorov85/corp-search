# coding: utf-8
"""
Валидация SQL для BooksLibraryAgent.

Содержит:
 - функции извлечения используемых таблиц/алиасов и колонок из SQL
 - логику проверки использованных таблиц/колонок против "кешированной" схемы,
   а также проверки соответствия allowed_tables.

Ключевая правка: корректная поддержка алиасов таблиц (например `books b` -> alias `b` -> base `books`).
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

    Реализован упрощённый парсер на regex; он покрывает стандартные случаи:
      - SELECT a.col, b.x, id FROM books b JOIN authors a ...
      - учитываются алиасы после имени таблицы (с/без AS)
    """
    sql = (sql or "").strip()
    alias_map: Dict[str, str] = {}
    tables: Set[str] = set()
    used_columns: List[Tuple[str, str]] = []

    # Найдём FROM
    for m in _from_re.finditer(sql):
        tbl = m.group(1)
        alias = m.group(2)
        # убрать возможный schema.table или кавычки
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

    # SELECT: возьмём часть между SELECT и FROM (упрощённо)
    sel = ""
    mm = _select_re.search(sql)
    if mm:
        sel = mm.group(1)

    if sel:
        # split by commas, аккуратно
        parts = [p.strip() for p in re.split(r',(?![^\(]*\))', sel) if p.strip()]
        for p in parts:
            # возможны выражения типа "b.title as t" или "max(b.pages) as maxp"
            # берём первую часть (до as) и ищем if alias.col present
            p_no_as = re.split(r'\s+as\s+', p, flags=re.I)[0].strip()
            # если есть пробел (функция), берём до пробела, но аккуратно
            # ищем выражение alias.col или table.col
            mcol = re.search(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', p_no_as)
            if mcol:
                left = mcol.group(1).lower()
                col = mcol.group(2).lower()
                used_columns.append((left, col))
            else:
                # bare column or expression: попытаемся достать идентификатор
                mbare = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)', p_no_as)
                if mbare:
                    used_columns.append(("", mbare.group(1).lower()))

    # Возвращаем set таблиц (реальные имена), used_columns (alias_or_blank, col), alias_map
    return tables, used_columns, alias_map


def validate_sql_against_schema(sql: str, allowed_tables: Set[str], schema: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Проверяет SQL:
     - разрешены ли используемые таблицы (с учётом алиасов, alias -> base table)
     - существуют ли использованные колонки в кешированной схеме

    Возвращает (ok: bool, messages: List[str], diagnostics: dict)
    diagnostics содержит: used_tables, missing_tables, missing_columns, alias_map
    """
    ok = True
    messages: List[str] = []
    diagnostics: Dict[str, Any] = {}

    # Получаем данные из SQL
    used_tables, used_columns, alias_map = extract_tables_and_columns(sql)

    # Построим множество реальных таблиц, учитывая alias_map:
    #  - если использован alias 'b', и alias_map['b'] = 'books' => используем 'books'
    #  - если встречается bare table name 'books' — уже есть
    real_used_tables: Set[str] = set()
    # добавим все таблицы, которые уже были извлечены
    for t in used_tables:
        real_used_tables.add(t.lower())
    # если в used_columns есть ссылки через alias, добавим соответствующие base tables
    for left, col in used_columns:
        if left:
            left_low = left.lower()
            if left_low in alias_map:
                real_used_tables.add(alias_map[left_low])
            else:
                # если left выглядит как реальное имя таблицы — добавим
                real_used_tables.add(left_low)

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
        # если left задан — разрешаем искать в table по alias_map
        if left:
            left_low = left.lower()
            if left_low in alias_map:
                tbl = alias_map[left_low]
                tbl_meta = schema.get(tbl)
                if not tbl_meta:
                    missing_cols.append(f"{col} (table not in cached schema: {left_low})")
                else:
                    col_names = [c["name"].lower() for c in tbl_meta.get("columns", [])]
                    if col.lower() not in col_names:
                        missing_cols.append(f"{col} (not in {tbl})")
            else:
                # left сама по себе может быть именем таблицы
                tbl = left_low
                tbl_meta = schema.get(tbl)
                if not tbl_meta:
                    missing_cols.append(f"{col} (table not in cached schema: {left_low})")
                else:
                    col_names = [c["name"].lower() for c in tbl_meta.get("columns", [])]
                    if col.lower() not in col_names:
                        missing_cols.append(f"{col} (not in {tbl})")
        else:
            # bare column: ищем хотя бы в одной таблице схемы
            found = False
            for tname, meta in schema.items():
                col_names = [c["name"].lower() for c in meta.get("columns", [])]
                if col.lower() in col_names:
                    found = True
                    break
            if not found:
                missing_cols.append(f"{col} (not found in cached schema)")

    if missing_cols:
        messages.append("Отсутствуют колонки: " + ", ".join(missing_cols))
        diagnostics["missing_columns"] = missing_cols

    ok = len(messages) == 0
    return ok, messages, diagnostics
