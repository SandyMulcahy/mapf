from .base import Agent
from .game_theoretic import GameTheoreticAgent

AGENTS: dict[str, type[Agent]] = {"game_theoretic": GameTheoreticAgent}

__all__ = ["AGENTS", "Agent", "GameTheoreticAgent"]
