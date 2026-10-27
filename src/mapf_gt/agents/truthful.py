"""The game-theoretic agent. Interface only: the algorithm goes here."""

from __future__ import annotations

from typing import Any

from .base import Agent


class TruthfulAgent(Agent):
    """
    With the default settings in ``pogema_adapter.to_grid_config`` the
    observation is POGEMA's ``POMAPF`` dictionary, a local window of side
    ``2 * obs_radius + 1`` centred on the agent. Print one observation before
    relying on its keys; they come from POGEMA, not from this repo.
    """

    name = "truthful"

    def __init__(self, agent_id: int): #creates agent using id
        super().__init__(agent_id)

    def reset(self, observation: Any) -> None:
        #observe, set internal state
        raise NotImplementedError

    def act(self, observation: Any) -> int:
        #observation for this step, output 
        raise NotImplementedError

"""
OBSERVATION
obstacles: a (2r+1)x(2r+1) binary grid centred on the agent
agents: the same-sized grid marking other agents nearby
xy: the agent's own position
target_xy: the agent's goal
"""