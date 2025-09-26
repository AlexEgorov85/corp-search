"""
services/db_service/executor.py

Безопасное выполнение SQL-запросов для агентов.
Адаптировано под требования агентов:
- ограничение выборки
- возврат (rows, metadata)
- логирование ошибок
"""

import logging
from typing import Any, List, Tuple, Dict
from sqlalchemy import text
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import SQLAlchemyError, OperationalError

LOG = logging.getLogger(__name__)


def execute_sql(engine: Engine, sql: str, limit: int = 500) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Выполнить SQL-запрос в безопасном режиме.

    :param engine: SQLAlchemy engine
    :param sql: строка SQL-запроса (без LIMIT)
    :param limit: ограничение выборки
    :return: (rows, metadata)
    """
    # Оборачиваем запрос в подзапрос с лимитом
    safe_sql = f"SELECT * FROM ({sql.strip().rstrip(';')}) AS subq LIMIT {limit}"

    try:
        with engine.connect() as conn:
            res: Result = conn.execute(text(safe_sql))
            rows = [dict(row) for row in res.mappings().all()]
            metadata = {
                "rowcount": len(rows),
                "sql": sql,
                "safe_sql": safe_sql,
            }
            return rows, metadata

    except OperationalError as oe:
        LOG.error("SQL execution error: %s", oe)
        # Возвращаем пустой результат и метаданные с ошибкой
        return [], {"error": str(oe), "sql": sql, "safe_sql": safe_sql}

    except SQLAlchemyError as e:
        LOG.exception("Unexpected SQLAlchemy error: %s", e)
        return [], {"error": str(e), "sql": sql, "safe_sql": safe_sql}
