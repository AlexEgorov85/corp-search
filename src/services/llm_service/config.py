# services/llm_service/config.py
"""
Модуль для работы с конфигурацией профилей LLM и агентов.
Отвечает за чтение настроек из `common.settings` и переменных окружения.
"""

import os
import copy
import logging
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# Попытка импортировать настройки
try:
    from src.common import settings as SETTINGS
except Exception:
    SETTINGS = None
    LOG.debug("Не удалось импортировать common.settings; SETTINGS = None", exc_info=True)

def _get_profiles_from_settings() -> Dict[str, Dict]:
    """Считывает и возвращает словарь LLM_PROFILES из настроек."""
    profiles: Dict[str, Dict] = {}
    if SETTINGS is None:
        return profiles
    try:
        p = getattr(SETTINGS, "LLM_PROFILES", None)
        if isinstance(p, dict):
            profiles.update(p)
    except Exception:
        LOG.debug("Ошибка чтения SETTINGS.LLM_PROFILES", exc_info=True)
    return profiles

def list_profiles() -> List[str]:
    """Возвращает список имен всех доступных профилей LLM."""
    return list(_get_profiles_from_settings().keys())

def _get_profile_config(profile: Optional[str]) -> Dict[str, Any]:
    """
    Получает конфигурацию для указанного профиля.
    Если профиль не указан, используется профиль по умолчанию.
    Если в настройках профиль не найден, используются переменные окружения.
    """
    profiles = _get_profiles_from_settings()
    if profile and profile in profiles:
        return dict(profiles[profile])

    # Используем профиль по умолчанию из настроек
    if SETTINGS is not None:
        try:
            default = getattr(SETTINGS, "LLM_DEFAULT_PROFILE", None)
            if default and default in profiles:
                return dict(profiles[default])
        except Exception:
            LOG.debug("Ошибка чтения LLM_DEFAULT_PROFILE", exc_info=True)

    # Fallback: переменные окружения
    cfg: Dict[str, Any] = {}
    backend = os.getenv("LLM_BACKEND") or os.getenv("LLM_PROVIDER")
    if backend:
        cfg["backend"] = backend
    if os.getenv("LLAMA_MODEL_PATH"):
        cfg.setdefault("backend", "llama_cpp")
        cfg.setdefault("model_path", os.getenv("LLAMA_MODEL_PATH"))
    if os.getenv("OPENAI_API_KEY"):
        cfg.setdefault("backend", "openai")
        cfg.setdefault("openai_api_key", os.getenv("OPENAI_API_KEY"))
    if os.getenv("LLM_N_CTX"):
        try:
            cfg.setdefault("n_ctx", int(os.getenv("LLM_N_CTX")))
        except Exception:
            pass
    if os.getenv("LLM_TEMPERATURE"):
        try:
            cfg.setdefault("temperature", float(os.getenv("LLM_TEMPERATURE")))
        except Exception:
            pass
    return cfg

# --- Утилиты для чтения конфигурации агентов ---
def _read_agents_container_from_settings() -> Dict[str, Dict[str, Any]]:
    """
    Пытается считать конфигурацию агентов из SETTINGS.
    Поддерживает несколько имен: AGENTS_CONFIG, AGENTS, agents_config, agents.
    Возвращает копию словаря или пустой dict.
    """
    if SETTINGS is None:
        return {}
    candidates = ["AGENTS_CONFIG", "AGENTS", "agents_config", "agents"]
    for name in candidates:
        try:
            val = getattr(SETTINGS, name, None)
            if isinstance(val, dict):
                return copy.deepcopy(val)
        except Exception:
            LOG.debug("Ошибка чтения SETTINGS.%s", name, exc_info=True)
    return {}

def get_agent_config(agent_name: str) -> Optional[Dict[str, Any]]:
    """
    Возвращает конфигурацию агента по его имени.
    Поиск нечувствителен к регистру и проверяет поля 'callable' и 'title'.
    Возвращает копию конфигурации или None.
    """
    if not agent_name:
        return None
    agents = _read_agents_container_from_settings()
    if not agents:
        LOG.debug("get_agent_config: контейнер агентов не найден в SETTINGS")
        return None

    # Прямой поиск
    if agent_name in agents:
        return copy.deepcopy(agents[agent_name])

    # Поиск без учета регистра
    lower_name = agent_name.lower()
    for k, v in agents.items():
        if isinstance(k, str) and k.lower() == lower_name:
            return copy.deepcopy(v)

    # Поиск по полям внутри конфигурации агента
    for k, v in agents.items():
        try:
            if isinstance(v, dict):
                name_like = v.get("callable") or v.get("title") or v.get("name")
                if isinstance(name_like, str) and name_like.lower() == lower_name:
                    return copy.deepcopy(v)
        except Exception:
            continue

    LOG.debug("get_agent_config: конфиг агента '%s' не найден", agent_name)
    return None

def get_all_agents_config() -> Dict[str, Dict[str, Any]]:
    """Возвращает конфигурацию всех агентов или пустой словарь."""
    return _read_agents_container_from_settings()