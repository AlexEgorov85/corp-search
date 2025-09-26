"""
services/db_service/connection.py

Модуль для управления подключением к базе данных.
Создаёт SQLAlchemy engine, хранит кэш подключений.
"""

from typing import Dict
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Кэш подключённых engine по URI
_ENGINE_CACHE: Dict[str, Engine] = {}


def get_engine(db_uri: str) -> Engine:
    """
    Получить SQLAlchemy engine по db_uri.
    Если engine уже создавался — вернуть из кэша.
    """
    if db_uri in _ENGINE_CACHE:
        return _ENGINE_CACHE[db_uri]

    engine = create_engine(db_uri, future=True)
    _ENGINE_CACHE[db_uri] = engine
    return engine
