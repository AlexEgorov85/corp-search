# Корпоративный поиск — ReActGraph (версия v2)

В этой версии добавлен skeleton ReActGraph на основе LangGraph и расширенный набор агентов/tools:
  - Реализация графа ReAct: src/graph/react_graph.py
  - Router и Reasoner (узлы) с подробными промптами в папке src/prompts
  - Склетон агентов: SQLAgent, FAISSAgent, DataAnalysisAgent (src/agents)
  - Конфигурации LLM и агентов: src/utils/settings.py и src/utils/config.py
  - Примеры промптов (русский язык) и контрактов ввода/вывода для агентов
  - Тесты (интеграционный skeleton)

Как использовать:
  - Установите зависимости (см. requirements.txt)
  - Укажите путь к локальной модели phi-3 в src/utils/settings.py или через переменные окружения LLM_MODEL_PATH
  - Для запуска тестов используйте pytest (часть агентов — заглушки, тесты проверяют flow)

Цель релиза: дать рабочий скелет ReActGraph, который можно расширять реальными реализациями агентов.
