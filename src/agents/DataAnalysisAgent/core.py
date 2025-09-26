# coding: utf-8
"""
DataAnalysisAgent — агрегатор/анализатор данных.

Функциональные обязанности:
 - принимать на вход структурированные данные (rows, summaries, docs) или источник (агент-источник)
 - производить map/reduce обработку для расчёта метрик
 - проводить валидацию и cross-check (validation.py)
 - при необходимости — задействовать LLM для синтеза выводов (prompt.py + llm.py)
 - возвращать AgentResult с полями structured (метрики) и content (короткий human-readable вывод или ссылка на необходимость синтеза)

Параметры и конфигурация берутся через get_agent_config("DataAnalysisAgent") из settings.AGENTS_CONFIG:
  config = {
      "max_map_batch": 200,
      "numeric_keys": ["amount", "value", "score"],
      "synthesizer_enabled": True,
      "synthesizer_model": "local-llm",
      ...
  }
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import logging
import time

from src.agents.base import BaseAgent
from services.results.agent_result import AgentResult

# локальные модули
from .analytics import map_summarize_documents, reduce_aggregate_summaries, streaming_aggregate
from .validation import basic_numeric_checks, cross_check_sum
from .llm import call_llm
from .prompt import synthesize_findings_prompt

from services.llm_service._old_factory import get_agent_config, create_llm

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class DataAnalysisAgent(BaseAgent):
    """
    Публичный класс агента для анализа данных.
    Метод execute принимает:
      - query: может быть:
          * строка — естественноя формулировка: "Посчитай выручку по клиентам за 2023"
          * dict с полями {"action":"analyze","data": [...], "params": {...}} — прямой вызов на данных
      - context: опциональные параметры (например - ссылка на источник, allowed_tables и т.д.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[Any] = None):
        cfg = config or get_agent_config("DataAnalysisAgent") or {}
        self.max_map_batch = int(cfg.get("max_map_batch", 500))
        self.numeric_keys = cfg.get("numeric_keys", ["amount", "value", "score"])
        self.synthesizer_enabled = bool(cfg.get("synthesizer_enabled", True))
        # LLM можно передать в конструктор или взять фабрику
        self.llm = llm or (create_llm() if callable(create_llm) else None)

    # ----------------- Map/Reduce flow -----------------
    def _map_phase(self, chunks: List[Dict[str, Any]], summarizer: Callable[[str], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        chunks: список чанков/документов; summarizer: функция(text)->metrics dict
        """
        return map_summarize_documents(chunks, summarizer, max_chunks=self.max_map_batch)

    def _reduce_phase(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Агрегируем выбранные numeric_keys
        """
        return reduce_aggregate_summaries(summaries, self.numeric_keys)

    # ----------------- Helpers -----------------
    @staticmethod
    def default_summarizer(text: str) -> Dict[str, Any]:
        """
        По умолчанию summarizer — простая эвристика: ищем числа в тексте и считаем их.
        Для production лучше подменить на ChunkSummarizerAgent или LLM summarizer.
        """
        import re
        nums = re.findall(r"[-+]?[0-9]*\\.?[0-9]+", text.replace(",", "."))
        nums = [float(n) for n in nums[:10]]  # берем первые 10 чисел
        return {"amount": sum(nums), "count_numbers": len(nums)}

    # ----------------- Main execute -----------------
    def execute(self, query: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Входные сценарии:
         - Если query is dict and contains 'data': считаем, что данные уже переданы (rows/_chunks)
         - Если query is str: это NL запрос; ожидается, что внешний оркестратор сначала позовёт
           источник (например BooksLibraryAgent / FAISS) и передаст данные в DataAnalysisAgent.
        Возвращаем AgentResult: structured -> агрегированные метрики.
        """
        start = time.time()
        # если непосредственный вызов на данных
        if isinstance(query, dict) and "data" in query:
            data = query["data"]
            params = query.get("params", {}) or {}
            summarizer = params.get("summarizer") or self.default_summarizer
            # map
            summaries = self._map_phase(data, summarizer)
            # reduce
            agg = self._reduce_phase(summaries)
            # basic validation / cross-check
            ok, cross = cross_check_sum(data, group_by=params.get("group_by",""), sum_field=params.get("sum_field",""))
            metadata = {"map_count": len(summaries), "processing_time": time.time()-start, "cross_check": cross}
            # optionally generate human-readable synthesis via LLM
            content = "Аналитика готова"
            if self.synthesizer_enabled and self.llm:
                try:
                    prompt = synthesize_findings_prompt(query.get("query", ""), agg)
                    text = call_llm(self.llm, prompt)
                    content = text.strip()
                except Exception as e:
                    LOG.warning("Synthesizer LLM failed: %s", e)
            return AgentResult(status="ok", content=content, structured={"metrics": agg}, metadata=metadata)

        # если NL-запрос — без данных: попросим внешние агенты дать данные и затем выполнять (контракт)
        if isinstance(query, str):
            # сигнализируем Executor/Orchestrator, что нужен источник данных
            # Возвращаем metadata с требованием: need_data_from: ["reports","clients"] и params
            required_sources = context.get("required_sources") if (context and "required_sources" in context) else ["reports","clients"]
            return AgentResult(
                status="ok",
                content="Требуется получить исходные данные из источников: " + ", ".join(required_sources),
                structured=None,
                metadata={"need_data_from": required_sources, "params": context}
            )

        return AgentResult(status="error", content="Неподдерживаемый формат запроса", structured=None, metadata={})
