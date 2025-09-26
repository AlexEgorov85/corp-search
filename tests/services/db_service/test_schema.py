import pytest
from sqlalchemy import text
from services.db_service.connection import get_engine
from services.db_service.schema import inspect_schema

def prepare_test_engine():
    engine = get_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);"))
        conn.execute(text("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER);"))
    return engine

def test_inspect_schema_tables_and_columns():
    engine = prepare_test_engine()
    schema = inspect_schema(engine)
    assert "authors" in schema
    assert "books" in schema
    assert "id" in schema["authors"]
    assert "title" in schema["books"]
