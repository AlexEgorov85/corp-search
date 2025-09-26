# src/agents/SynthesizerAgent/core.py
# coding: utf-8
import logging
import os
from typing import Any, Dict, Optional

from src.services.llm_service.factory import ensure_llm

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# Поддержка разных версий langchain
try:
    from langchain.chains import LLMChain
except Exception:
    try:
        from langchain import LLMChain  # type: ignore
    except Exception:
        LLMChain = None  # возможно LLMChain недоступен

try:
    from langchain_core.prompts import PromptTemplate
except Exception:
    try:
        from langchain.prompts import PromptTemplate  # type: ignore
    except Exception:
        # очень простой fallback
        class PromptTemplate:
            def __init__(self, input_variables, template):
                self.input_variables = input_variables
                self.template = template

            def format(self, **kwargs):
                return self.template.format(**kwargs)


PROMPT_TEMPLATE = (
    "Вопрос: {question}\n"
    "Контекст: {context}\n"
    "Шаг: {step_id}\n"
    "Сформируй итоговый ответ и краткое reasoning:\n"
)

class SynthesizerAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[Any] = None):
        self.config = config or {}
        # Если LLM не передан — создаём его из профиля
        if llm is None:
            profile = self.config.get("llm_profile") or os.getenv("LLM_DEFAULT_PROFILE", "default")
            self.llm = ensure_llm(profile)
            if self.llm is None:
                LOG.warning(f"SynthesizerAgent: не удалось создать LLM из профиля '{profile}'")
        else:
            self.llm = llm

        self.prompt = PromptTemplate(
            input_variables=["question", "context", "step_id"],
            template=PROMPT_TEMPLATE
        )
        self.chain = None
        if self.llm and LLMChain is not None:
            try:
                self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            except Exception:
                LOG.exception("Failed to create LLMChain for SynthesizerAgent")
                self.chain = None

    def _collect_context_text(self, state: Dict[str, Any]) -> str:
        parts = []
        for sid, out in (state.get("step_outputs") or {}).items():
            if isinstance(out, dict):
                parts.append(f"[{sid}] {out.get('content')}")
            else:
                parts.append(f"[{sid}] {str(out)}")
        return "\n".join(p for p in parts if p)

    def _ensure_step_id(self, state: Dict[str, Any]) -> str:
        # сначала смотрим current_call
        current_call = state.get("current_call")
        if isinstance(current_call, dict):
            sid = current_call.get("step_id") or ""
            if sid:
                return sid
        # если нет — возьмём последний добавленный step в step_outputs
        step_outputs = state.get("step_outputs") or {}
        if step_outputs:
            # взять последний по порядку вставки — Python dict сохраняет порядок
            try:
                last_sid = list(step_outputs.keys())[-1]
                return last_sid or ""
            except Exception:
                pass
        # если ничего нет — вернём пустую строку (ключ должен существовать)
        return ""

    def _call_llm(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Вызывает LLM через chain (если есть), или fallback.
        ВАЖНО: всегда передаём prompt_inputs с ключами question, context, step_id.
        """
        question = state.get("question", "") or ""
        context_text = self._collect_context_text(state) or ""
        step_id = self._ensure_step_id(state) or ""

        prompt_inputs = {
            "question": question,
            "context": context_text,
            "step_id": step_id  # ключ всегда присутствует
        }

        if self.chain:
            try:
                # Попытки вызвать chain в более-менее совместимом виде.
                # 1) prefer __call__ (chain(prompt_inputs) -> dict)
                try:
                    out = self.chain(prompt_inputs)
                except TypeError:
                    # 2) older API: chain.run(**prompt_inputs)
                    out = self.chain.run(**prompt_inputs)
                # normalize out
                if isinstance(out, dict):
                    return out
                if isinstance(out, str):
                    return {"final_answer": out, "reasoning": ""}
                return {"final_answer": str(out), "reasoning": ""}
            except Exception as e:
                LOG.exception("SynthesizerAgent.chain invocation failed: %s", e)
                # идём в fallback
        # fallback
        LOG.warning("No LLM or chain available for SynthesizerAgent; using fallback synthesizer.")
        return {"final_answer": "Нет модели для синтеза ответа", "reasoning": "SynthesizerAgent не смог вызвать LLM"}

    def execute(self, query: str = "", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Интерфейс execute: принимает query/context и возвращает normalized result dict.
        Ожидается формат:
            {"status": "ok", "content": "...", "structured": {...}, "metadata": {...}}
        """
        try:
            # Для совместимости: если нам передали context как state (часто так делается в node)
            state = context if isinstance(context, dict) else ({} if context is None else {})
            # если state пустой — возможно query содержит question (строку) — тогда формируем базовый state
            if not state and isinstance(query, str) and query:
                state = {"question": query}

            res = self._call_llm(state)
            # нормализуем to expected format
            if isinstance(res, dict):
                final_answer = res.get("final_answer") or res.get("content") or ""
                structured = res if res.get("final_answer") else {"final_answer": final_answer}
                return {
                    "status": "ok",
                    "content": final_answer,
                    "structured": structured,
                    "metadata": {"raw": str(res)}
                }
            else:
                return {
                    "status": "ok",
                    "content": str(res),
                    "structured": {"final_answer": str(res)},
                    "metadata": {}
                }
        except Exception as e:
            LOG.exception("SynthesizerAgent.execute failed: %s", e)
            return {
                "status": "error",
                "content": "Нет модели для синтеза ответа",
                "structured": {"final_answer": "Нет модели для синтеза ответа", "reasoning": str(e)},
                "metadata": {}
            }
