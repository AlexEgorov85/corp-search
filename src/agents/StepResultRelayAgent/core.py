# src/agents/StepResultRelayAgent/core.py
from src.agents.base import BaseAgent

class StepResultRelayAgent(BaseAgent):
    def __init__(self, descriptor, config=None):
        super().__init__(descriptor, config)