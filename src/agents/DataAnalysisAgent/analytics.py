# coding: utf-8
"""
Алгоритмы анализа: map/reduce для суммаризации большого числа документов/чанков,
агрегаторы и инструменты для incremental processing.

Задачи, которые покрывает модуль:
 - map_step: обработать пачку чанков/документов -> вернуть промежуточные суммаризированные показатели
 - reduce_step: агрегировать map-результаты -> итоговые метрики
 - streaming helpers: итеративная агрегация (поддержка больших наборов)
"""

from __future__ import annotations
from typing import List, Dict, Any, Callable, Iterable, Tuple
import math
import statistics

def map_summarize_documents(chunks: Iterable[Dict[str, Any]], summarizer: Callable[[str], Dict[str, Any]], max_chunks: int = 1000) -> List[Dict[str, Any]]:
    """
    Map-операция: каждому чанку вызываем summarizer (функция, которая принимает текст и возвращает dict metric).
    summarizer может быть локальной (н-р, rule-based) или LLM-based (вызов LLM).
    Возвращаем список summary dict'ов.
    """
    out = []
    cnt = 0
    for ch in chunks:
        if cnt >= max_chunks:
            break
        text = ch.get("text") or ch.get("content") or ""
        try:
            s = summarizer(text)
        except Exception as e:
            s = {"error": str(e)}
        # переносим метаданые чанка (source, chunk_id) чтобы потом цитировать
        s["_meta"] = {"source": ch.get("source"), "chunk_id": ch.get("chunk_id")}
        out.append(s)
        cnt += 1
    return out

def reduce_aggregate_summaries(summaries: List[Dict[str, Any]], numeric_keys: List[str]) -> Dict[str, Any]:
    """
    Простая reduce: по списку summary dict'ов агрегируем указанные numeric_keys:
      - count, sum, mean, median, std (если возможно)
    Возвращаем dict с агрегатами по каждому ключу.
    """
    agg = {}
    for k in numeric_keys:
        vals = [s.get(k) for s in summaries if (s.get(k) is not None and isinstance(s.get(k), (int, float)))]
        if not vals:
            agg[k] = {"count": 0, "sum": 0.0, "mean": None, "median": None, "std": None}
            continue
        cnt = len(vals)
        ssum = sum(vals)
        mean = ssum / cnt
        median = statistics.median(vals)
        std = statistics.pstdev(vals) if cnt > 1 else 0.0
        agg[k] = {"count": cnt, "sum": ssum, "mean": mean, "median": median, "std": std}
    return agg

def streaming_aggregate(iterable_summaries: Iterable[Dict[str, Any]], numeric_keys: List[str]) -> Dict[str, Any]:
    """
    Streaming-агрегатор для большого потока: поддерживает инкрементальные подсчёты.
    Возвращает итератор промежуточных состояний (можно не использовать).
    """
    # простая реализация — аккумулируем все, т.к. более сложные алгоритмы требуют дополнительного state
    all_summ = []
    for s in iterable_summaries:
        all_summ.append(s)
    return reduce_aggregate_summaries(all_summ, numeric_keys)
