# src/agents/PlannerAgent/rules.py

from typing import Any, Dict, List
import logging

LOG = logging.getLogger(__name__)

def _has_cycles(subquestions: List[Dict]) -> bool:
    if not isinstance(subquestions, list):
        return False
    graph = {}
    for sq in subquestions:
        if not isinstance(sq, dict) or "id" not in sq:
            continue
        deps = sq.get("depends_on", [])
        if not isinstance(deps, list):
            return False
        graph[sq["id"]] = deps

    visited = set()
    rec_stack = set()

    def dfs(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True
        rec_stack.remove(node)
        return False

    if not graph:
        return False
    return any(dfs(node) for node in graph)

# Обновлённые правила валидации
DECOMPOSITION_RULES = [
    {
        "id": "has_reasoning",
        "target": "decomposition",
        "condition": lambda d, tools: isinstance(d, dict) and "reasoning" in d and len(d["reasoning"]) == 5,
        "message": "Декомпозиция должна содержать массив reasoning из 5 элементов (P1–P5)",
        "severity": "error"
    },
    {
        "id": "has_planning",
        "target": "decomposition",
        "condition": lambda d, tools: isinstance(d.get("planning"), dict),
        "message": "Поле planning должно быть объектом",
        "severity": "error"
    },
    {
        "id": "subq_structure",
        "target": "subquestion",
        "condition": lambda sq, tools: all(k in sq for k in ["id", "text", "depends_on", "confidence", "reason", "explanation"]),
        "message": "Подвопрос должен содержать id, text, depends_on, confidence, reason, explanation",
        "severity": "error"
    },
    {
        "id": "no_cycles",
        "target": "decomposition",
        "condition": lambda d, tools: not _has_cycles(d.get("subquestions", [])),
        "message": "Обнаружены циклические зависимости между подвопросами",
        "severity": "error"
    }
]

def validate_decomposition(decomposition: Any, tool_registry: dict) -> tuple[bool, List[dict]]:
    issues = []
    for rule in DECOMPOSITION_RULES:
        try:
            if rule["target"] == "decomposition":
                is_ok = rule["condition"](decomposition, tool_registry)
                if not is_ok:
                    issues.append({
                        "rule_id": rule["id"],
                        "message": rule["message"],
                        "severity": rule["severity"]
                    })
            elif rule["target"] == "subquestion":
                subqs = decomposition.get("subquestions", [])
                for sq in subqs:
                    if not isinstance(sq, dict):
                        continue
                    is_ok = rule["condition"](sq, tool_registry)
                    if not is_ok:
                        issues.append({
                            "rule_id": rule["id"],
                            "message": f"{rule['message']} (подвопрос: {sq.get('text', 'N/A')})",
                            "severity": rule["severity"]
                        })
        except Exception as e:
            issues.append({
                "rule_id": rule["id"],
                "message": f"Ошибка выполнения правила: {e}",
                "severity": "error"
            })
    is_valid = len([i for i in issues if i["severity"] == "error"]) == 0
    return is_valid, issues