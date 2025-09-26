import psycopg2
from sqlalchemy import create_engine
import logging

# 🔹 DSN для подключения
DSN = "postgresql+psycopg2://new_user:secure_password_123@localhost:5432/postgres"

print("=== 🐘 Прямое подключение через psycopg2 ===")
try:
    schema = "Lib"
    conn = psycopg2.connect(
        dbname="postgres",
        user="new_user",
        password="secure_password_123",
        host="localhost",
        port="5432",
        options=f"-c search_path={schema}",
        client_encoding = "UTF8"
    )
    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {schema};")
        conn.commit()
        cur.execute("SELECT version();")
        print("✅ psycopg2:", cur.fetchone()[0])
    conn.close()
except Exception as e:
    print("❌ psycopg2 ошибка:", e)


print("\n=== ⚙️ Подключение через SQLAlchemy.create_engine ===")
# Включаем логирование SQLAlchemy (покажет DSN и connect_args)
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

try:
    engine = create_engine(
        DSN,
        echo=True,
        connect_args={
            "options": "-c search_path=Lib -c client_encoding=UTF8"
        }
    )
    with engine.connect() as conn:
        result = conn.execute("SELECT version();")
        print("✅ SQLAlchemy:", result.scalar())
except Exception as e:
    print("❌ SQLAlchemy ошибка:", e)
