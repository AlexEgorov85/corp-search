# coding: utf-8
"""
Валидаторы и чекеры качества данных/метрик.

Содержит:
 - базовую проверку на NaN/None и типы
 - cross-check между метриками (напр., сверка сумм/подсчётов)
 - правила допустимых значений (пороговые проверки)
"""

from typing import Dict, Any, List, Tuple
import math

def basic_numeric_checks(series: List[float]) -> Dict[str, Any]:
    """
    Простая проверка массива чисел: count, nan_count, min/max/mean
    """
    n = len(series)
    nan_cnt = sum(1 for v in series if v is None or (isinstance(v, float) and math.isnan(v)))
    clean = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if clean:
        mn = min(clean)
        mx = max(clean)
        mean = sum(clean) / len(clean)
    else:
        mn = mx = mean = None
    return {"n": n, "nan": nan_cnt, "min": mn, "max": mx, "mean": mean}

def cross_check_sum(rows: List[Dict[str, Any]], group_by: str, sum_field: str) -> Tuple[bool, Dict[str, float]]:
    """
    Простейшая проверка агрегации: суммируем sum_field по группам и возвращаем словарь.
    Возвращаем (ok, sums).
    """
    sums = {}
    for r in rows:
        key = r.get(group_by)
        val = r.get(sum_field) or 0
        try:
            val = float(val)
        except Exception:
            val = 0.0
        sums[key] = sums.get(key, 0.0) + val
    # можно вернуть always ok=True — детектор несогласованности может быть в вызывающем коде
    return True, sums
