# src/agents/PlannerAgent/validation/decomposition_rules.py

from typing import Any, Dict, List
import logging

LOG = logging.getLogger(__name__)

def _is_atomic_question(text: str) -> bool:
    """Простая эвристика: атомарный вопрос содержит один глагол/цель."""
    # В реальности можно использовать NLP, но для MVP — простая проверка
    return "?" not in text[:-1]  # нет вложенных вопросов

def _has_cycles(subquestions: List[Dict]) -> bool:
    """Проверка на циклические зависимости (DFS)."""
    if not isinstance(subquestions, list):
        return False  # Если не список — не может быть циклов
    graph = {}
    for sq in subquestions:
        if not isinstance(sq, dict) or "id" not in sq:
            continue
        # Проверяем, что depends_on — список
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
    # Проверяем, что graph не пустой
    if not graph:
        return False
    return any(dfs(node) for node in graph)

# Правила валидации ДЕКОМПОЗИЦИИ
DECOMPOSITION_RULES = [
    {
        "id": "subq_structure",
        "target": "decomposition",
        "condition": lambda d, tools: isinstance(d, dict) and "subquestions" in d and isinstance(d["subquestions"], list),
        "message": "Декомпозиция должна быть объектом с полем 'subquestions' (массив)",
        "severity": "error"
    },
    {
        "id": "subq_atomic",
        "target": "subquestion",
        "condition": lambda sq, tools: _is_atomic_question(sq["text"]),
        "message": "Подвопрос должен быть атомарным (один вопрос)",
        "severity": "error"
    },
    {
        "id": "no_cycles",
        "target": "decomposition",
        "condition": lambda d, tools: not _has_cycles(d["subquestions"]),
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
                    if "id" not in sq or "text" not in sq:
                        continue
                    is_ok = rule["condition"](sq, tool_registry)
                    if not is_ok:
                        issues.append({
                            "rule_id": rule["id"],
                            "message": f"{rule['message']} (подвопрос: {sq['text']})",
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

