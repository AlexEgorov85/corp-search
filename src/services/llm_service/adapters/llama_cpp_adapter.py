# coding: utf-8
"""
Адаптер для llama-cpp-python с ленивой инициализацией и адаптацией под конфиг LLM_PROFILES.

Основные возможности:
- Ленивая инициализация модели (initialize() / init_model в конструкторе по флагу).
- Поддержка передачи внешнего экземпляра Llama (dependency injection).
- Учет параметра n_ctx из конфига при инициализации модели.
- Автоматическое ограничение max_tokens перед вызовом модели, чтобы избежать
  ошибки "Requested tokens exceed context window".
- Попытка оценить длину prompt в токенах (если у llama есть tokenize) и учесть её
  при расчёте дозволенных токенов.
- Надёжная нормализация raw-ответа (вытащить content / text / choices и т.п.).
- Извлечение JSON из fenced code blocks или первого сбалансированного `{...}` при необходимости.
- Фабрика create_llama_cpp_adapter(config) — возвращает адаптер (не None), что ожидают тесты.

Примеры:
  # Ленивый адаптер, не грузит модель сразу (удобно для тестов)
  adapter = LlamaCppAdapter(config, init_model=False)

  # Явная инициализация (может бросить ошибку, если путь к модели некорректен)
  adapter.initialize()

  # Генерация (generate автоматически попытается инициализировать модель, если это возможно)
  text = adapter.generate("Сгенерируй JSON-план для задачи ...")

  # Закрытие
  adapter.close()

Примечание: адаптер написан максимально осторожно: он не пытается "клевать" за
приватные поля Llama (например _stack) и корректно работает даже если llama_cpp
замокирован в тестах.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class LlamaCppAdapter:
    """
    Адаптер для Llama (llama_cpp.Llama).

    Параметры конструктора:
        config: словарь с настройками LLM (см. LLM_PROFILES). Ожидаемые ключи:
            - model_path: путь к gguf файлу (строка) или None
            - n_ctx: желаемый context window (int)
            - max_tokens: дефолт max_tokens для генерации
            - temperature, top_p: сэмплинг параметры
            - backend_kwargs: dict с доп. kwargs для конструктора Llama (n_threads, use_gpu и т.п.)
            - safety_margin: (опционально) сколько токенов резервируем от контекста (по умолчанию 64)
        llama_instance: опционально — уже созданный экземпляр llama_cpp.Llama (dependency injection)
        init_model: bool, по умолчанию False — если True, адаптер попытается инициализировать Llama в __init__.
                    Если модель не может быть загружена и init_model=True — исключение будет проброшено.
                    Если init_model=False — адаптер создаётся без загрузки модели (удобно для тестов).
    """

    # Регулярное выражение для fenced code block с JSON внутри (```json ... ``` или ``` {...} ```)
    _FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llama_instance: Optional[Any] = None,
        init_model: bool = False,
    ):
        # Конфиг: читаем и подставляем значения, если есть
        self.config: Dict[str, Any] = dict(config or {})

        # Параметры генерации по умолчанию (берём из конфига LLM_PROFILES)
        # Если в конфиге нет значений — используем безопасные дефолты
        self.default_gen_kwargs: Dict[str, Any] = {
            "max_tokens": int(self.config.get("max_tokens", 1024)),
            "temperature": float(self.config.get("temperature", 0.1)),
            "top_p": float(self.config.get("top_p", 0.9)),
            "stop": self.config.get("stop", ["<|end|>", "</s>"]),
        }

        # backend kwargs (например n_threads, use_gpu)
        self.backend_kwargs: Dict[str, Any] = dict(self.config.get("backend_kwargs") or {})
        # model_path и n_ctx
        self.model_path: Optional[str] = self.config.get("model_path")
        self.config_n_ctx: Optional[int] = None
        try:
            if "n_ctx" in self.config:
                self.config_n_ctx = int(self.config.get("n_ctx"))
        except Exception:
            self.config_n_ctx = None

        # safety margin (сколько токенов оставить резерва)
        self.safety_margin: int = int(self.config.get("safety_margin", 64))

        # внешняя переданная инстанция Llama (dependency injection)
        self._external_llama = llama_instance

        # внутренний экземпляр Llama (создаётся в initialize если необходимо)
        self._llama: Optional[Any] = None

        # флаг — является ли адаптер владельцем экземпляра (если True, он вызывает .close())
        self._owns_llama: bool = False

        # если внешний экземпляр передан — используем его и не "владеем" им
        if self._external_llama is not None:
            self._llama = self._external_llama
            self._owns_llama = False
            LOG.debug("LlamaCppAdapter: использован внешний экземпляр Llama (dependency injection).")

        # если просят инициализировать модель немедленно — делаем это (и пробрасываем исключения)
        if init_model:
            self.initialize(raise_on_failure=True)

    # ------------------------------
    # Инициализация / состояние
    # ------------------------------
    def is_initialized(self) -> bool:
        """Возвращает True, если адаптер имеет доступ к экземпляру Llama."""
        return self._llama is not None

    def initialize(self, raise_on_failure: bool = True) -> None:
        """
        Явная инициализация модели (lazy import + создание Llama).

        Поведение:
          - Если llama уже инициализирована — ничего не делает.
          - Если передан внешний экземпляр — использует его.
          - Иначе пытается importlib.import_module("llama_cpp") и создать класс Llama.
          - Пытается передать параметр n_ctx из конфига (если указан).
          - Если ошибка и raise_on_failure=True -> пробрасывает исключение, иначе логирует и оставляет _llama == None.
        """
        if self._llama is not None:
            LOG.debug("LlamaCppAdapter.initialize: _llama уже инициализирован, пропускаем.")
            return

        if self._external_llama is not None:
            self._llama = self._external_llama
            self._owns_llama = False
            LOG.debug("LlamaCppAdapter.initialize: используем внешний инстанс llama.")
            return

        if not self.model_path:
            msg = "LlamaCppAdapter.initialize: model_path не указан в конфиге, инициализация невозможна."
            LOG.warning(msg)
            if raise_on_failure:
                raise RuntimeError(msg)
            return

        # ленивый импорт — важно для тестов, где sys.modules['llama_cpp'] может быть замокирован
        try:
            llama_cpp = importlib.import_module("llama_cpp")
        except Exception as e:
            LOG.exception("LlamaCppAdapter.initialize: не удалось импортировать 'llama_cpp': %s", e)
            if raise_on_failure:
                raise
            return

        LlamaClass = getattr(llama_cpp, "Llama", None)
        if LlamaClass is None:
            msg = "LlamaCppAdapter.initialize: в модуле 'llama_cpp' не найден класс 'Llama'."
            LOG.error(msg)
            if raise_on_failure:
                raise RuntimeError(msg)
            return

        # формируем init kwargs и учитываем n_ctx из конфига / backend_kwargs
        init_kwargs: Dict[str, Any] = {"model_path": self.model_path}
        init_kwargs.update(self.backend_kwargs or {})
        if self.config_n_ctx is not None:
            init_kwargs["n_ctx"] = int(self.config_n_ctx)
        else:
            # если backend_kwargs содержит n_ctx - он уже в init_kwargs
            pass

        try:
            self._llama = LlamaClass(**init_kwargs)
            self._owns_llama = True
            LOG.info("LlamaCppAdapter.initialize: Llama успешно инициализирован (model_path=%s, n_ctx=%s).",
                     self.model_path, init_kwargs.get("n_ctx"))
        except Exception as e:
            LOG.exception("LlamaCppAdapter.initialize: ошибка создания Llama: %s", e)
            if raise_on_failure:
                raise
            # не падаем — оставим _llama == None для тестов/ленивого поведения
            self._llama = None
            self._owns_llama = False

    # ------------------------------
    # Закрытие адаптера
    # ------------------------------
    def close(self) -> None:
        """
        Корректно закрываем модель, если мы её инициализировали (owns_llama).
        Не пытаемся менять приватные поля Llama, только вызываем public close()/shutdown().
        """
        if not self._llama:
            LOG.debug("LlamaCppAdapter.close: _llama is None — nothing to close.")
            return

        if self._owns_llama:
            try:
                closer = getattr(self._llama, "close", None)
                if callable(closer):
                    closer()
                    LOG.debug("LlamaCppAdapter.close: вызван .close() у Llama.")
            except Exception as e:
                LOG.debug("LlamaCppAdapter.close: исключение при close(): %s", e)

        # отвязываем ссылку
        try:
            del self._llama
        except Exception:
            self._llama = None
        self._owns_llama = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------
    # Помощники для context window и токенизации
    # ------------------------------
    def _get_model_n_ctx(self) -> int:
        """
        Попробовать получить реальный размер контекстного окна (n_ctx) у инициализированной модели.
        Порядок:
          1) self._llama.context_params.n_ctx
          2) self._llama.model_params.n_ctx
          3) self.config['n_ctx'] (если задан)
          4) backend_kwargs['n_ctx'] (если задан)
          5) fallback 512
        Возвращает int >= 1.
        """
        try:
            if self._llama is not None:
                cp = getattr(self._llama, "context_params", None)
                if cp is not None and hasattr(cp, "n_ctx"):
                    return int(getattr(cp, "n_ctx"))
                mp = getattr(self._llama, "model_params", None)
                if mp is not None and hasattr(mp, "n_ctx"):
                    return int(getattr(mp, "n_ctx"))
        except Exception:
            # игнорируем ошибки получения параметров
            pass
        # попытаться взять из конфига/backend_kwargs
        try:
            if self.config_n_ctx is not None:
                return int(self.config_n_ctx)
            if isinstance(self.backend_kwargs, dict) and "n_ctx" in self.backend_kwargs:
                return int(self.backend_kwargs.get("n_ctx"))
        except Exception:
            pass
        # безопасный fallback
        return 512

    def _estimate_prompt_tokens(self, prompt: str) -> Optional[int]:
        """
        Попытаться оценить число токенов в prompt, используя доступный tokenizer у self._llama.
        Если токенизация недоступна — возвращает None.
        Не поднимает исключения.
        """
        if not prompt or self._llama is None:
            return None
        try:
            # Некоторые версии llama_cpp.Llama могут иметь метод tokenize
            tokenize_fn = getattr(self._llama, "tokenize", None)
            if callable(tokenize_fn):
                toks = tokenize_fn(prompt)
                # вероятность, что toks — итерируемая коллекция
                try:
                    return int(len(toks))
                except Exception:
                    return None
        except Exception:
            return None
        return None

    # ------------------------------
    # Нормализация raw-ответа и извлечение JSON
    # ------------------------------
    @staticmethod
    def _text_from_raw(raw: Any) -> str:
        """
        Универсально извлечь текст из raw-ответа, который может быть:
         - str
         - dict с 'choices' -> [{'message': {'content': '...'}}] или [{'text': '...'}]
         - итерируемый поток чанков (stream)
         - любые другие структуры — приводим к str(raw)
        """
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        try:
            if isinstance(raw, dict):
                choices = raw.get("choices")
                if isinstance(choices, (list, tuple)) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        # chat style
                        msg = first.get("message")
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                            return msg.get("content").strip()
                        # older: text
                        if isinstance(first.get("text"), str):
                            return first.get("text").strip()
                        # streaming delta content
                        delta = first.get("delta")
                        if isinstance(delta, dict):
                            for k in ("content", "text"):
                                if isinstance(delta.get(k), str):
                                    return delta.get(k).strip()
                # top-level fields
                for key in ("content", "text", "output"):
                    v = raw.get(key)
                    if isinstance(v, str):
                        return v.strip()
                # fallback to json dump
                try:
                    return json.dumps(raw, ensure_ascii=False)
                except Exception:
                    return str(raw)
            # iterable (stream)
            from collections.abc import Iterable
            if isinstance(raw, Iterable):
                parts = []
                for chunk in raw:
                    if chunk is None:
                        continue
                    if isinstance(chunk, str):
                        parts.append(chunk)
                    elif isinstance(chunk, dict):
                        parts.append(LlamaCppAdapter._text_from_raw(chunk))
                    else:
                        parts.append(str(chunk))
                return "".join(parts).strip()
            # fallback
            return str(raw).strip()
        except Exception:
            try:
                return str(raw)
            except Exception:
                return ""

    @classmethod
    def _extract_fenced_json(cls, text: str) -> Optional[str]:
        """
        Попытаться извлечь содержимое первого fenced code block (```json ... ``` или ``` ... ```).
        Возвращаем None, если не найден.
        """
        if not isinstance(text, str):
            return None
        m = cls._FENCED_JSON_RE.search(text)
        if not m:
            return None
        inner = m.group(1).strip()
        return inner if inner else None

    @staticmethod
    def _extract_first_balanced_json(text: str) -> Optional[str]:
        """
        Поиск первого сбалансированного JSON-объекта: находим первый '{' и подбираем соответствующую '}'.
        Возвращаем подстроку или None.
        """
        if not isinstance(text, str):
            return None
        start = text.find("{")
        if start == -1:
            return None
        stack = []
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                stack.append("}")
            elif ch == "}":
                if not stack:
                    continue
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return None

    # ------------------------------
    # Основные методы генерации
    # ------------------------------
    def generate_with_raw(self, prompt: str, **generate_kwargs) -> Tuple[str, Any]:
        """
        Вызыв модели, возвращая (normalized_text, raw_response).

        Логика:
         - Если модель ещё не инициализирована, пытаемся лениво её создать (initialize(raise_on_failure=False)).
         - Если модель не доступна после попытки — возвращаем ("", None).
         - Подготавливаем call_kwargs путём слияния default_gen_kwargs и generate_kwargs.
         - Безопасно ограничиваем max_tokens в зависимости от model n_ctx и примерной длины prompt.
         - Пробуем вызвать create_chat_completion / create_completion / generate / callable.
         - Возвращаем нормализованный текст и raw объект.
        """
        if self._llama is None:
            # пробуем лениво инициализировать — если не получится, не падаем
            if self.model_path:
                try:
                    self.initialize(raise_on_failure=False)
                except Exception:
                    LOG.debug("LlamaCppAdapter.generate_with_raw: ленивый initialize() провалился.")
            else:
                LOG.debug("LlamaCppAdapter.generate_with_raw: model_path не указан и llama не инициализирована.")
        if self._llama is None:
            return "", None

        # merge kwargs
        call_kwargs: Dict[str, Any] = dict(self.default_gen_kwargs)
        call_kwargs.update({k: v for k, v in (generate_kwargs or {}).items() if v is not None})

        # ----- Safety clamp: рассчитать допустимый max_tokens -----
        model_ctx = self._get_model_n_ctx() or 512
        est_prompt_tokens = self._estimate_prompt_tokens(prompt)
        # допустимое число токенов для генерации: модельный контекст минус оценённый prompt и safety margin
        allowed_max = max(1, model_ctx - (est_prompt_tokens or 0) - int(self.safety_margin))
        # если allowed_max оказался слишком большим (на всякий случай), ограничим им также requested max_tokens
        requested_max = None
        try:
            requested_max = int(call_kwargs.get("max_tokens", call_kwargs.get("max_new_tokens", None) or 0))
        except Exception:
            requested_max = None

        if requested_max is None or requested_max <= 0:
            call_kwargs["max_tokens"] = min(call_kwargs.get("max_tokens", 1024), allowed_max)
        else:
            if requested_max > allowed_max:
                LOG.warning(
                    "LlamaCppAdapter: понижаю requested max_tokens %s -> %s (model_ctx=%s, est_prompt=%s, safety_margin=%s)",
                    requested_max, allowed_max, model_ctx, est_prompt_tokens, self.safety_margin,
                )
                call_kwargs["max_tokens"] = allowed_max
            else:
                call_kwargs["max_tokens"] = requested_max

        raw = None
        try:
            # Prefer chat API if available
            if hasattr(self._llama, "create_chat_completion"):
                messages = [{"role": "user", "content": prompt}]
                raw = self._llama.create_chat_completion(messages=messages, **call_kwargs)
            elif hasattr(self._llama, "create_completion"):
                raw = self._llama.create_completion(prompt=prompt, **call_kwargs)
            elif hasattr(self._llama, "generate"):
                raw = self._llama.generate(prompt=prompt, **call_kwargs)
            else:
                # try callable (positional)
                try:
                    raw = self._llama(prompt, **call_kwargs)
                except TypeError:
                    raw = self._llama(prompt=prompt, **call_kwargs)
        except Exception as e:
            # Логируем подробности и не пробрасываем (потому что вызывающий код умеет сделать fallback)
            LOG.exception("LlamaCppAdapter.generate_with_raw: ошибка при вызове модели: %s", e)
            return "", None

        text = self._text_from_raw(raw)
        return text, raw

    def generate(self, prompt: str, **generate_kwargs) -> str:
        """
        Высокоуровневая генерация: возвращает строку.
        Если ответ содержит JSON в fenced block или первый сбалансированный объект,
        пытаемся вернуть minified JSON (чтобы downstream мог вызвать json.loads).
        """
        text, raw = self.generate_with_raw(prompt, **generate_kwargs)
        if not text:
            return ""

        # 1) try fenced JSON
        fenced = self._extract_fenced_json(text)
        if fenced:
            try:
                parsed = json.loads(fenced)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return fenced

        # 2) try first balanced JSON object
        balanced = self._extract_first_balanced_json(text)
        if balanced:
            try:
                parsed = json.loads(balanced)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return balanced

        # 3) otherwise return normalized text
        return text.strip()

# ------------------------------
# Фабричная функция для adapters.factory
# ------------------------------
def create_llama_cpp_adapter(config: Optional[Dict[str, Any]] = None) -> LlamaCppAdapter:
    """
    Фабрика — возвращает экземпляр LlamaCppAdapter. По умолчанию не инициализирует модель (init_model=False).
    Это ожидаемое поведение для тестов: адаптер создаётся, даже если model_path не существует.
    """
    return LlamaCppAdapter(config=config, init_model=False)
