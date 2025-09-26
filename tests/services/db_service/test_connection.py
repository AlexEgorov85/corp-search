import pytest
from sqlalchemy import text
from services.db_service.connection import get_engine

def test_get_engine_sqlite_memory():
    engine = get_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1
