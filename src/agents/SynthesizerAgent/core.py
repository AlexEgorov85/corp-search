# src/agents/SynthesizerAgent/core.py
# coding: utf-8
import json
import logging
import os
from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.services.llm_service.factory import ensure_llm

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


PROMPT_TEMPLATE = (
    "Вопрос: {question}\n"
    "Контекст: {context}\n"
    "Шаг: {step_id}\n"
    "Сформируй итоговый ответ и краткое reasoning:\n"
)

class SynthesizerAgent(BaseAgent):
    def __init__(self, descriptor: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        super().__init__(descriptor, config or {})
        # Получаем профиль из config
        profile = self.config.get("llm_profile") or os.getenv("LLM_DEFAULT_PROFILE", "default")
        self.llm = ensure_llm(profile)
        self.prompt_template = PROMPT_TEMPLATE

    def _call_llm(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("question", "")
        context_text = self._collect_context_text(state)
        step_id = self._ensure_step_id(state)

        prompt = self.prompt_template.format(
            question=question,
            context=context_text,
            step_id=step_id
        )

        if self.llm is None:
            return {"final_answer": "Нет модели для синтеза ответа", "reasoning": "LLM не загружена"}

        try:
            response = self.llm.generate(prompt)
            # Пробуем распарсить как JSON
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    return parsed
            except:
                pass
            # Иначе возвращаем как текст
            return {"final_answer": response, "reasoning": ""}
        except Exception as e:
            LOG.exception("Ошибка генерации в SynthesizerAgent: %s", e)
            return {"final_answer": "Ошибка генерации ответа", "reasoning": str(e)}

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
        

