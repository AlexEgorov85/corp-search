# src/graph/state_utils.py
# coding: utf-8
"""
Утилиты для преобразования legacy state dict <-> GraphContext.
Сохраняем минимальную обратную совместимость: state_from_dict/ state_to_dict.
"""
from __future__ import annotations
from typing import Dict, Any

from src.graph.context import GraphContext


def state_from_dict(d: Dict[str, Any]) -> GraphContext:
    """
    Преобразовать legacy dict -> GraphContext (валидировано).
    Используется внутри узлов при входе.
    """
    return GraphContext.from_state_dict(d)


def state_to_dict(ctx: GraphContext) -> Dict[str, Any]:
    """
    Преобразовать GraphContext -> legacy dict (для совместимости с остальной системой).
    Делегируем ctx.to_legacy_state().
    """
    return ctx.to_legacy_state()
