"""
Утилиты для работы с графом выполнения.
Содержит общие вспомогательные функции, используемые несколькими узлами.
"""

from __future__ import annotations
from typing import Any, Dict
import logging


LOG = logging.getLogger(__name__)


def build_tool_registry_snapshot(agent_registry) -> Dict[str, Any]:
    """
    Строит корректный snapshot инструментов из AgentRegistry.
    Гарантирует, что все операции присутствуют, даже если агент не инициализирован.

    Используется в узлах:
      - planner_node (для генерации плана)
      - reasoner_node (для выбора инструментов)

    Args:
        agent_registry: экземпляр AgentRegistry

    Returns:
        Dict[str, Any]: сериализуемый словарь вида:
        {
          "AgentName": {
            "title": "...",
            "description": "...",
            "operations": {
              "op1": {
                "kind": "direct",
                "description": "...",
                "params": {...},
                "outputs": {...}
              }
            }
          }
        }
    """
    if agent_registry is None:
        LOG.warning("build_tool_registry_snapshot: agent_registry is None")
        return {}

    snapshot = {}
    for name, entry in agent_registry.tool_registry.items():
        if not isinstance(entry, dict):
            snapshot[name] = {}
            continue
        try:
            operations = agent_registry._resolve_operations(name, entry)
        except Exception as e:
            LOG.warning("Не удалось разрешить операции для агента %s: %s", name, e)
            operations = {}
        safe_meta = {
            "title": entry.get("title", ""),
            "description": entry.get("description", ""),
            "operations": {}
        }
        for op_name, op_meta in operations.items():
            safe_meta["operations"][op_name] = {
                "kind": op_meta.get("kind", "direct"),
                "description": op_meta.get("description", ""),
                "params": op_meta.get("params", {}),
                "outputs": op_meta.get("outputs", {})
            }
        snapshot[name] = safe_meta
    return snapshot