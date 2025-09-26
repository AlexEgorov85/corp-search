# coding: utf-8
"""
Промты для DataAnalysisAgent — все на русском.

Здесь основные шаблоны:
- synthesize_findings_prompt: превращает структурированные метрики/агрегаты в человекочитаемый вывод
- explain_discrepancy_prompt: попросить LLM объяснить показываемые несогласованности
"""

import textwrap
from typing import Dict, Any

def synthesize_findings_prompt(question: str, metrics_summary: Dict[str, Any], top_n: int = 10) -> str:
    """
    Формируем краткий промпт для LLM: на вход метрики_summary (json-serializable)
    Ожидаем на выходе — краткий поясняющий текст, выводы, приоритетные next steps.
    """
    tpl = textwrap.dedent("""\
        Вы — аналитическая служба. Дана задача пользователя:
        {question}

        Ниже — структурированная сводка рассчитанных метрик (JSON):
        {metrics}

        Задача: на русском языке кратко (3-8 предложений) сформулировать:
         - ключевые наблюдения (top-3)
         - возможные причины (коротко)
         - рекомендации по дальнейшим шагам (2-3 конкретных шага)

        Требования: верните ТОЛЬКО чистый текст (без JSON), емко и по делу.
    """)
    return tpl.format(question=question, metrics=metrics_summary)

def explain_discrepancy_prompt(context_summary: str, discrepancy_details: Dict[str, Any]) -> str:
    tpl = textwrap.dedent("""\
        Дано краткое описание контекста:
        {context}

        И детали обнаруженной несогласованности/аномалии:
        {discrepancy}

        Поясните на русском возможные причины этой аномалии и предложите 3 возможных гипотезы и способы проверки.
    """)
    return tpl.format(context=context_summary, discrepancy=discrepancy_details)
