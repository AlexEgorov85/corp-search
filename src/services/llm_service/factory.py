# src/services/llm_service/factory.py
"""Основной API фабрики LLM.
Содержит функции для создания и управления жизненным циклом экземпляров LLM.
Является центральной точкой взаимодействия для остальной системы.
"""

import logging
from typing import Optional, Any

from src.common import settings
from src.services.llm_service.adapters.factory import create_adapter_from_config
from src.services.llm_service.wrappers.dummy_wrapper import DummyLLM
from .config import _get_profile_config, _get_profiles_from_settings
from .cache import get_cached_llm, cache_llm, close_cached_llm, close_all_cached_llms
from .wrappers import LLMSafeWrapper

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def create_llm(profile_name: str) -> Optional[Any]:
    """Создает и оборачивает экземпляр LLM для указанного профиля.

    Args:
        profile_name (str): Имя профиля LLM.

    Returns:
        Optional[Any]: Обернутый экземпляр LLM (LLMSafeWrapper) или None.
    """
    config = _get_profile_config(profile_name)
    if not config:
        LOG.warning("Конфигурация для профиля '%s' не найдена.", profile_name)
        return None

    # Создание адаптера через общую фабрику
    adapter = _create_adapter(config)
    if not adapter:
        LOG.warning("Не удалось создать адаптер для профиля '%s'.", profile_name)
        return None

    # Оборачивание в безопасную обертку
    wrapped_llm = LLMSafeWrapper(adapter)
    LOG.info("Создана LLM для профиля '%s'.", profile_name)
    return wrapped_llm


def _create_adapter(config: dict) -> Optional[Any]:
    """Создает адаптер на основе конфигурации, включая обработку моков.

    Args:
        config (dict): Конфигурация профиля.

    Returns:
        Optional[Any]: Экземпляр адаптера или None.
    """
    # Проверка на мок
    if config.get("mock") is True:
        mock_response = config.get("mock_response", "Mock LLM Response")
        LOG.info("Создается мок-LLM с ответом: %s", mock_response)
        return DummyLLM(response=mock_response)

    # Используем общую фабрику для остальных случаев
    return create_adapter_from_config(config)


def ensure_llm(profile_name: str) -> Optional[Any]:
    """Возвращает закэшированный экземпляр LLM или создает новый и кэширует его.

    Args:
        profile_name (str): Имя профиля LLM.

    Returns:
        Optional[Any]: Обернутый экземпляр LLM (LLMSafeWrapper) или None.
    """
    llm = get_cached_llm(profile_name)
    if llm is None:
        llm = create_llm(profile_name)
        if llm is not None:
            cache_llm(profile_name, llm)
            LOG.debug("LLM для профиля '%s' создана и закэширована.", profile_name)
        else:
            LOG.warning("Не удалось создать LLM для профиля '%s'.", profile_name)
    else:
        LOG.debug("Возвращена кэшированная LLM для профиля '%s'.", profile_name)
    return llm


def get_llm(profile_name: str) -> Optional[Any]:
    """Возвращает закэшированный экземпляр LLM для указанного профиля.

    Args:
        profile_name (str): Имя профиля LLM.

    Returns:
        Optional[Any]: Обернутый экземпляр LLM (LLMSafeWrapper) или None.
    """
    llm = get_cached_llm(profile_name)
    if llm is None:
        LOG.debug("LLM для профиля '%s' не найдена в кэше.", profile_name)
    else:
        LOG.debug("Возвращена кэшированная LLM для профиля '%s'.", profile_name)
    return llm


def close_llm(profile_name: Optional[str] = None) -> None:
    """Закрывает и удаляет из кэша экземпляр LLM.

    Если profile_name не указан, закрывает все кэшированные экземпляры.

    Args:
        profile_name (Optional[str]): Имя профиля LLM.
    """
    if profile_name is None:
        close_all_cached_llms()
        LOG.info("Закрыты все кэшированные LLM.")
    else:
        close_cached_llm(profile_name)
        LOG.info("Закрыта LLM для профиля '%s'.", profile_name)


def list_profiles() -> list[str]:
    """Возвращает список доступных профилей LLM.

    Returns:
        list[str]: Список имен профилей.
    """
    profiles = _get_profiles_from_settings()
    return list(profiles.keys())
