"""Interface for decentralised agents: one object per robot.

Each agent sees only its own observation and returns only its own action.
Nothing in this interface gives an agent access to the other agents' objects
or to the full environment, which is what keeps the method decentralised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    name: str = "agent"

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def reset(self, observation: Any) -> None:
        """Called once per episode with the first observation."""

    @abstractmethod
    def act(self, observation: Any) -> int:
        """Return a POGEMA action index for this timestep.

        Do not hard-code which integer means which move. Use
        ``mapf_gt.maps.pogema_adapter.action_for_move`` or ``move_table`` so
        the mapping always comes from the installed POGEMA version.
        """
