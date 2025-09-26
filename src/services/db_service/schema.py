# coding: utf-8
"""
Утилиты по инспекции/кэшированию схемы таблиц.
- refresh_schema_for_tables(engine, tables)  — возвращает подробную структуру (columns, pk, fks)
- build_schema_text(schema) — компактная текстовая сводка для вставки в prompt
"""

from typing import Dict, Any, List
import logging

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def refresh_schema_for_tables(engine, tables: List[str]) -> Dict[str, Any]:
    """
    Инспектирует перечисленные таблицы через SQLAlchemy Inspector и возвращает
    структуру вида:
    {
        "table_name": {
            "columns": [{"name": "...", "type": "...", "nullable": True}, ...],
            "pk": ["id"],
            "fks": [ { "constrained_columns": [...], "referred_table": "...", "referred_columns": [...] }, ... ]
        },
        ...
    }

    Если таблица не найдена — записываем пустую структуру и логируем предупреждение.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    schema = {}
    for tbl in tables:
        try:
            cols = inspector.get_columns(tbl)
            pk = inspector.get_pk_constraint(tbl).get("constrained_columns") or []
            fks = inspector.get_foreign_keys(tbl) or []
            # normalize columns: name, type (str), nullable
            normalized_cols = []
            for c in cols:
                normalized_cols.append({"name": c.get("name"), "type": str(c.get("type")), "nullable": bool(c.get("nullable"))})
            schema[tbl] = {"columns": normalized_cols, "pk": pk, "fks": fks}
        except Exception as e:
            LOG.warning("Таблица '%s' не найдена или ошибка инспекции: %s", tbl, e)
            schema[tbl] = {"columns": [], "pk": [], "fks": []}
    return schema


def build_schema_text(schema: Dict[str, Any], max_cols_preview: int = 6) -> str:
    """
    Вернуть компактную текстовую сводку схемы для вставки в prompt.
    Формат:
      - table (col1, col2, col3...)
        FK: colX -> other_table(other_col)
    """
    parts: List[str] = []
    for tbl, meta in schema.items():
        cols = meta.get("columns", [])
        col_names = [c["name"] for c in cols][:max_cols_preview]
        more = "" if len(cols) <= max_cols_preview else ", ..."
        parts.append(f"- {tbl} ({', '.join(col_names)}{more})")
        for fk in meta.get("fks", []):
            constrained = ", ".join(fk.get("constrained_columns", []) or [])
            referred = fk.get("referred_table", "")
            referred_cols = ", ".join(fk.get("referred_columns", []) or [])
            parts.append(f"  FK: {constrained} -> {referred}({referred_cols})")
    return "\n".join(parts)
