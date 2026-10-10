"""The game-theoretic agent. Interface only: the algorithm goes here."""

from __future__ import annotations

from typing import Any

from .base import Agent


class GameTheoreticAgent(Agent):
    """One decentralised agent that chooses its move by game-theoretic reasoning.

    With the default settings in ``pogema_adapter.to_grid_config`` the
    observation is POGEMA's ``POMAPF`` dictionary, a local window of side
    ``2 * obs_radius + 1`` centred on the agent. Print one observation before
    relying on its keys; they come from POGEMA, not from this repo.
    """

    name = "game_theoretic"

    def __init__(self, agent_id: int):
        super().__init__(agent_id)

    def reset(self, observation: Any) -> None:
        raise NotImplementedError

    def act(self, observation: Any) -> int:
        raise NotImplementedError
