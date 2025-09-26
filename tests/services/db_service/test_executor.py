import pytest
from sqlalchemy import text
from services.db_service.connection import get_engine
from services.db_service.executor import execute_sql

def prepare_books_engine():
    engine = get_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);"))
        conn.execute(text("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER);"))
        conn.execute(text("INSERT INTO authors (id, name) VALUES (1, 'Пушкин');"))
        conn.execute(text("INSERT INTO books (id, title, author_id) VALUES (1, 'Евгений Онегин', 1);"))
    return engine

def test_execute_sql_success():
    engine = prepare_books_engine()
    sql = "SELECT b.id, b.title FROM books b JOIN authors a ON b.author_id = a.id WHERE a.name = 'Пушкин'"
    rows, metadata = execute_sql(engine, sql)
    assert len(rows) == 1
    assert rows[0]["title"] == "Евгений Онегин"
    assert "safe_sql" in metadata

def test_execute_sql_with_error():
    engine = prepare_books_engine()
    bad_sql = "SELECT non_existing_col FROM books"
    rows, metadata = execute_sql(engine, bad_sql)
    assert rows == []
    assert "error" in metadata
