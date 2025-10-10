# src/utils/settings.py
# coding: utf-8
"""
Централизованные настройки проекта (на русском).

Конвенции:
- LLM_PROFILES: именованные профили LLM. Для каждого профиля должен подниматься один кэшируемый экземпляр LLM.
  Агент, которому нужен LLM, указывает имя профиля через поле 'llm_profile' в своей конфигурации.
"""

from typing import Any, Dict
import os

# Окружение и режим работы
APP_ENV = os.environ.get("APP_ENV", "development")
DEBUG = APP_ENV != "production"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# --------------------------
# Профили LLM (именованные)
# --------------------------
# Каждый профиль описывает параметры и путь к модели/бэкенду.
# Реализация create_llm(profile) в src/utils/config.py должна:
#  - прочитать профиль из LLM_PROFILES[profile]
#  - создать backend-клиент (например LlamaCpp, Ollama, OpenAI и т.п.)
#  - кэшировать и возвращать один экземпляр на профиль (singleton per profile)
LLM_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "backend": os.environ.get("LLM_BACKEND", "llama_cpp"),  # пример: 'llama_cpp', 'ollama', 'openai'
        "model_path": os.environ.get("LLM_MODEL_PATH", r"C:\Qwen3\Qwen3-4B-Instruct-2507-F16.gguf"),
        "n_ctx": int(os.environ.get("LLM_N_CTX", "16384")),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.3")),
        "top_p": float(os.environ.get("LLM_TOP_P", "0.9")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "8192")),
        "backend_kwargs": {
            "n_threads": int(os.environ.get("LLM_THREADS", "8")),
            "use_gpu": os.environ.get("LLM_USE_GPU", "false").lower() in ("1", "true", "yes"),
            "verbose": False, 
        },
    },
    "explore": {
        "backend": "llama_cpp",
        "model_path": os.environ.get("LLM_MODEL_PATH_EXPLORE", r"C:\phi-3\Phi-3-mini-4k-instruct-fp16.gguf"),
        "n_ctx": 4090,
        "temperature": 0.7,
        "max_tokens": 1024,
        "backend_kwargs": {"n_threads": 8},
    },
    "local_quick": {
        "backend": "llama_cpp",
        "model_path": os.environ.get("LLM_MODEL_PATH_QUICK", r"C:\models\alpaca\alpaca-7b.gguf"),
        "n_ctx": 2048,
        "temperature": 0.0,
        "max_tokens": 256,
        "backend_kwargs": {"n_threads": 4},
    },
}

# Профиль по умолчанию (если агент не указал явно)
LLM_DEFAULT_PROFILE = os.environ.get("LLM_DEFAULT_PROFILE", "default")

# --------------------------
# Внешние DSN / параметры
# --------------------------
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql+psycopg2://new_user:secure_password_123@localhost:5432/postgres?client_encoding=utf8"
)

# --------------------------
# LangGraph / рекурсия
# --------------------------
# Значение по умолчанию для ограничения итераций LangGraph (если нужно — можно увеличить)
LANGGRAPH_RECURSION_LIMIT = int(os.environ.get("LANGGRAPH_RECURSION_LIMIT", "25"))

# --------------------------
# Руководство для разработчиков (на русском)
# --------------------------
# Как агент должен получить LLM:
# 1) получить профиль: profile = agent_cfg.get("config", {}).get("llm_profile") or agent_cfg.get("llm_profile") or LLM_DEFAULT_PROFILE
# 2) получить LLM: from src.utils.config import create_llm; llm = create_llm(profile=profile)
# 3) использовать llm для генерации/инференса.
#
# Важно: create_llm(profile) должен кэшировать экземпляры по profile — т.е. повторные вызовы с одним profile
# должны возвращать один и тот же объект LLM (одна поднятая модель).
#
# Пример использования внутри агента (псевдо-код):
#   profile = agent_cfg.get("config", {}).get("llm_profile", "default")
#   from src.utils.config import create_llm
#   llm = create_llm(profile=profile)
#   result = llm.generate(prompt, max_tokens=..., temperature=...)
#
# Если нужно — могу дополнительно подготовить реализацию create_llm(profile) в src/utils/config.py,
# включая кэширование экземпляров и обработку различных backend'ов (llama_cpp, ollama, openai и т.д.).
#
# --------------------------
# Конец файла
# --------------------------
