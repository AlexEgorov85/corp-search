import logging
from typing import Any, Dict, Tuple
from src.services.llm_service.model.request import LLMRequest
from src.services.llm_service.model.response import LLMResponse

LOG = logging.getLogger(__name__)

class BaseLLMAdapter:
    """
    Универсальный базовый адаптер для всех LLM.
    
    Этот класс определяет контракт, которому должны следовать все конкретные адаптеры:
    1. Реализовать метод generate_with_request (основной способ взаимодействия)
    2. Опционально реализовать метод generate (для простых текстовых промптов)
    
    Все адаптеры должны возвращать кортеж (str, LLMResponse) при вызове generate_with_request.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует адаптер с конфигурацией.
        
        Args:
            config: Конфигурация адаптера
        """
        self.config_setting = config or {}
        self.config = {
            "max_tokens": config.get("LLM_MAX_TOKENS", 1024),
            "temperature": config.get("LLM_TEMPERATURE", 0.3),
            "top_p": config.get("LLM_TOP_P", 0.9),
            "stop": config.get("stop", ["###"]),
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Генерирует текстовый ответ на основе простого текстового промпта.
        
        Args:
            prompt: Текстовый промпт для модели
            **kwargs: Дополнительные параметры
            
        Returns:
            str: Сгенерированный текст
            
        Note:
            Этот метод является опциональным. Если адаптер не поддерживает простые текстовые промпты,
            он может вызывать NotImplementedError.
        """
        raise NotImplementedError("Метод generate не реализован в базовом адаптере")
    
    def generate_with_request(self, request: LLMRequest, **kwargs) -> Tuple[str, LLMResponse]:
        """
        Генерирует ответ на основе структурированного запроса LLMRequest.
        
        Args:
            request: Объект запроса в стандартизированном формате
            **kwargs: Дополнительные параметры
            
        Returns:
            tuple[str, LLMResponse]: (основной ответ, структурированный объект ответа)
            
        Raises:
            NotImplementedError: Если адаптер не реализует этот метод
        """
        raise NotImplementedError("Метод generate_with_request не реализован в базовом адаптере")