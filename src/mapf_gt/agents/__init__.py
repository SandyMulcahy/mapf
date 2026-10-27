from .base import Agent
from .truthful import TruthfulAgent

AGENTS: dict[str, type[Agent]] = {"truthful": TruthfulAgent}

__all__ = ["AGENTS", "Agent", "TruthfulAgent"]
