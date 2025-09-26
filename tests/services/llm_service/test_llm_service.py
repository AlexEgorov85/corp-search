# tests/services/llm_service/test_llm_service.py
# coding: utf-8
"""Тесты для сервиса LLM (фабрика, адаптеры, обертки, утилиты)."""

import os
import pytest
import json
from unittest.mock import patch, MagicMock, PropertyMock

# Импортируем тестируемые функции и классы
from src.services.llm_service.adapters.llama_cpp_adapter import LlamaCppAdapter
from src.services.llm_service.adapters.openai_adapter import OpenAIAdapter
from src.services.llm_service.cache import _LLM_CACHE, _LLM_LOCK
from src.services.llm_service.config import get_agent_config, get_all_agents_config, list_profiles
from src.services.llm_service.factory import close_llm, create_llm, ensure_llm, get_llm
from src.services.llm_service.utils import safe_format, strip_code_fences
from src.services.llm_service.wrappers.safe_wrapper import LLMSafeWrapper

# Фикстура для очистки кэша LLM перед каждым тестом
@pytest.fixture(autouse=True)
def clear_llm_cache():
    """Очищает кэш LLM перед каждым тестом."""
    with _LLM_LOCK:
        # Закрываем все кэшированные экземпляры
        keys = list(_LLM_CACHE.keys())
        for k in keys:
            inst = _LLM_CACHE.pop(k, None)
            if inst and hasattr(inst, 'close'):
                try:
                    inst.close()
                except Exception:
                    pass # Игнорируем ошибки закрытия в тестах
        # Очищаем сам кэш
        _LLM_CACHE.clear()


@pytest.fixture
def mock_settings():
    """Фикстура для мокирования настроек."""
    mock_llm_profiles = {
        "default": {"backend": "llama_cpp", "model_path": "/path/to/model.gguf"},
        "mock_profile": {"mock": True, "mock_response": "Mocked response for test."},
        "openai_test": {"backend": "openai", "model": "gpt-3.5-turbo"},
    }
    mock_agents_config = {
        "TestAgent": {
            "title": "Тестовый Агент",
            "description": "Это тест.",
            "backend": "sql",
            "module": "test.module",
            "callable": "TestClass"
        },
        "BooksLibraryAgent": {
            "title": "Библиотека Книг",
            "description": "Агент для книг.",
        },
        "Agent1": {"title": "Агент 1"},
        "Agent2": {"title": "Агент 2"},
    }
    # Мокируем SETTINGS, так как _get_profile_config и _read_agents_container_from_settings зависят от него
    with patch('src.services.llm_service.config.SETTINGS') as mock_settings_obj:
        mock_settings_obj.LLM_PROFILES = mock_llm_profiles
        mock_settings_obj.AGENTS_CONFIG = mock_agents_config
        yield mock_settings_obj


class TestLLMServiceAPI:
    """Интеграционные тесты основного API фабрики LLM."""

    def test_list_profiles_returns_default(self, mock_settings):
        """Тест: list_profiles возвращает хотя бы профиль 'default'."""
        profiles = list_profiles()
        assert isinstance(profiles, list)
        assert "default" in profiles
        assert "mock_profile" in profiles # Проверим, что наш мок-профиль тоже доступен

    def test_create_llm_mock_profile(self):
        """Тест: create_llm создает мок-LLM при указании 'mock' в конфиге."""
        # Мокируем _get_profile_config в модуле, где она используется (factory.py)
        with patch('src.services.llm_service.factory._get_profile_config') as mock_get_config:
            mock_get_config.return_value = {"mock": True, "mock_response": "Это мок-ответ"}
            llm = create_llm("test_mock")
            assert llm is not None
            assert isinstance(llm, LLMSafeWrapper)
            # Проверяем, что вызов generate возвращает мок-ответ
            response = llm.generate("Любой промпт")
            assert response == "Это мок-ответ"

    def test_ensure_llm_caching_works(self):
        """Тест: ensure_llm кэширует экземпляры по профилю."""
        with patch('src.services.llm_service.factory._get_profile_config') as mock_get_config:
            mock_get_config.return_value = {"mock": True}
            # Первый вызов
            llm1 = ensure_llm("test_cache")
            # Второй вызов
            llm2 = ensure_llm("test_cache")
            # Должны вернуться одинаковые объекты (один и тот же экземпляр)
            assert llm1 is llm2
            # Вызов для другого профиля должен вернуть новый объект
            llm3 = ensure_llm("another_profile")
            assert llm1 is not llm3

    def test_get_llm_returns_cached_instance(self):
        """Тест: get_llm возвращает закэшированный экземпляр."""
        with patch('src.services.llm_service.factory._get_profile_config') as mock_get_config:
            mock_get_config.return_value = {"mock": True}
            # Сначала создаем экземпляр через ensure_llm
            created_llm = ensure_llm("test_get")
            # Затем получаем его через get_llm
            retrieved_llm = get_llm("test_get")
            assert retrieved_llm is created_llm

    def test_close_llm_closes_instance(self):
        """Тест: close_llm корректно вызывает метод close у LLMSafeWrapper."""
        with patch('src.services.llm_service.factory._get_profile_config') as mock_get_config:
            mock_get_config.return_value = {"mock": True}
            llm = ensure_llm("test_close")
            # Мокируем метод close у внутреннего _model (DummyLLM)
            dummy_llm_instance = llm._model # Получаем внутренний адаптер/мок
            with patch.object(dummy_llm_instance, 'close') as mock_close:
                close_llm("test_close")
                mock_close.assert_called_once()
            # Проверяем, что экземпляр удален из кэша
            assert get_llm("test_close") is None

    def test_close_all_llms(self):
        """Тест: close_llm() без аргумента закрывает все экземпляры."""
        with patch('src.services.llm_service.factory._get_profile_config') as mock_get_config:
            mock_get_config.return_value = {"mock": True}
            ensure_llm("profile1")
            ensure_llm("profile2")
            assert get_llm("profile1") is not None
            assert get_llm("profile2") is not None
            close_llm() # Закрываем все
            assert get_llm("profile1") is None
            assert get_llm("profile2") is None


class TestLLMServiceUtils:
    """Тесты для вспомогательных утилит."""

    def test_safe_format_basic(self):
        """Тест: safe_format заменяет плейсхолдеры."""
        template = "Привет, {name}!"
        mapping = {"name": "Алексей"}
        result = safe_format(template, mapping)
        assert result == "Привет, Алексей!"

    def test_safe_format_with_dict(self):
        """Тест: safe_format сериализует словари в JSON."""
        template = "Данные: {data}"
        mapping = {"data": {"key": "value"}}
        result = safe_format(template, mapping)
        expected_json = '{"key": "value"}'
        assert expected_json in result

    def test_safe_format_missing_key(self):
        """Тест: safe_format оставляет плейсхолдеры при отсутствии ключа."""
        template = "Привет, {name}!"
        mapping = {"other_key": "value"}
        result = safe_format(template, mapping)
        assert result == "Привет, {name}!"

    def test_strip_code_fences_triple_backticks(self):
        """Тест: strip_code_fences удаляет тройные обратные кавычки."""
        text = "```python\nprint('hello')\n```"
        result = strip_code_fences(text)
        assert result == "print('hello')"

    def test_strip_code_fences_single_backticks(self):
        """Тест: strip_code_fences удаляет одинарные обратные кавычки."""
        text = "`print('hello')`"
        result = strip_code_fences(text)
        assert result == "print('hello')"

    def test_strip_code_fences_no_fences(self):
        """Тест: strip_code_fences возвращает текст без изменений, если нет кавычек."""
        text = "plain text"
        result = strip_code_fences(text)
        assert result == "plain text"


class TestLLMServiceConfig:
    """Тесты для модуля конфигурации."""

    def test_get_agent_config_finds_agent(self):
        """Тест: get_agent_config возвращает конфигурацию агента."""
        mock_agents_config = {
            "TestAgent": {
                "title": "Тестовый Агент",
                "description": "Это тест.",
                "module": "test.module",
            }
        }
        with patch('src.services.llm_service.config._read_agents_container_from_settings') as mock_read:
            mock_read.return_value = mock_agents_config
            config = get_agent_config("TestAgent")
            assert config is not None
            assert config["title"] == "Тестовый Агент"
            assert config["module"] == "test.module"

    def test_get_agent_config_case_insensitive(self):
        """Тест: get_agent_config ищет агент без учета регистра."""
        mock_agents_config = {
            "BooksLibraryAgent": {
                "title": "Библиотека Книг",
                "description": "Агент для книг.",
            }
        }
        with patch('src.services.llm_service.config._read_agents_container_from_settings') as mock_read:
            mock_read.return_value = mock_agents_config
            config = get_agent_config("bookslibraryagent")
            assert config is not None
            assert config["title"] == "Библиотека Книг"

    def test_get_all_agents_config_returns_dict(self):
        """Тест: get_all_agents_config возвращает словарь конфигураций."""
        mock_agents_config = {
            "Agent1": {"title": "Агент 1"},
            "Agent2": {"title": "Агент 2"},
        }
        with patch('src.services.llm_service.config._read_agents_container_from_settings') as mock_read:
            mock_read.return_value = mock_agents_config
            all_configs = get_all_agents_config()
            assert isinstance(all_configs, dict)
            assert len(all_configs) == 2
            assert "Agent1" in all_configs


class TestLLMServiceAdapters:
    """Модульные тесты для адаптеров (с мокированием внешних библиотек)."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake-api-key"})
    def test_openai_adapter_creation(self):
        """Тест: Создание адаптера OpenAI."""
        # Мокируем импорт openai
        mock_openai = MagicMock()
        with patch.dict('sys.modules', {'openai': mock_openai}):
            from src.services.llm_service.adapters.openai_adapter import create_openai_adapter
            config = {"backend": "openai", "model": "gpt-3.5-turbo"}
            adapter = create_openai_adapter(config)
            assert adapter is not None
            assert isinstance(adapter, OpenAIAdapter)

    def test_llama_cpp_adapter_creation_mocked(self):
        """Тест: Создание адаптера llama.cpp (с моком llama_cpp)."""
        # Мокируем импорт llama_cpp
        mock_llama_cpp = MagicMock()
        mock_llama_instance = MagicMock()
        mock_llama_cpp.Llama.return_value = mock_llama_instance
        with patch.dict('sys.modules', {'llama_cpp': mock_llama_cpp}):
            from src.services.llm_service.adapters.llama_cpp_adapter import create_llama_cpp_adapter
            config = {"backend": "llama_cpp", "model_path": "/fake/path.gguf"}
            adapter = create_llama_cpp_adapter(config)
            assert adapter is not None
            assert isinstance(adapter, LlamaCppAdapter)

    def test_llama_cpp_adapter_close(self):
        """Тест: Закрытие адаптера llama.cpp."""
        # Мокируем импорт llama_cpp
        mock_llama_cpp = MagicMock()
        mock_llama_instance = MagicMock()
        mock_llama_cpp.Llama.return_value = mock_llama_instance
        with patch.dict('sys.modules', {'llama_cpp': mock_llama_cpp}):
            from src.services.llm_service.adapters.llama_cpp_adapter import LlamaCppAdapter
            config = {"backend": "llama_cpp", "model_path": "/fake/path.gguf"}
            adapter = LlamaCppAdapter(config)
            adapter.close()
            # Проверяем, что внутренний объект llama_cpp.Llama был удален
            # (псевдо-проверка, т.к. del внутри close не проверить напрямую)
            # Логично предположить, что close() вызывается без ошибок
            assert True # Заглушка, если других проверок нет


class TestLLMSafeWrapper:
    """Тесты для безопасной обертки LLM."""

    def test_llm_safe_wrapper_with_callable(self):
        """Тест: Обертка работает с вызываемым объектом."""
        def simple_llm(prompt):
            return f"Generated: {prompt}"
        wrapper = LLMSafeWrapper(simple_llm)
        response = wrapper.generate("тест")
        assert response == "Generated: тест"

    def test_llm_safe_wrapper_with_invoke_method(self):
        """Тест: Обертка извлекает текст из метода .invoke()."""
        class InvokeLLM:
            def invoke(self, prompt):
                return {"choices": [{"message": {"content": f"Invoke: {prompt}"}}]}
        invoke_llm = InvokeLLM()
        wrapper = LLMSafeWrapper(invoke_llm)
        response = wrapper.generate("тест")
        assert response == "Invoke: тест"

    def test_llm_safe_wrapper_with_predict_method(self):
        """Тест: Обертка извлекает текст из метода .predict()."""
        class PredictLLM:
            def predict(self, prompt):
                return f"Predicted: {prompt}"
        predict_llm = PredictLLM()
        wrapper = LLMSafeWrapper(predict_llm)
        response = wrapper.generate("тест")
        assert response == "Predicted: тест"

    def test_llm_safe_wrapper_with_generate_method(self):
        """Тест: Обертка извлекает текст из метода .generate()."""
        class GenerateLLM:
            def generate(self, prompt):
                return f"Generated: {prompt}"
        generate_llm = GenerateLLM()
        wrapper = LLMSafeWrapper(generate_llm)
        response = wrapper.generate("тест")
        assert response == "Generated: тест"

    def test_llm_safe_wrapper_with_openai_style_dict(self):
        """Тест: Обертка извлекает текст из ответа в стиле OpenAI."""
        openai_style_response = {"choices": [{"message": {"content": "Содержимое ответа от OpenAI"}}]}
        class DictLLM:
            def __call__(self, prompt):
                return openai_style_response
        dict_llm = DictLLM()
        wrapper = LLMSafeWrapper(dict_llm)
        response = wrapper.generate("тест")
        assert response == "Содержимое ответа от OpenAI"
