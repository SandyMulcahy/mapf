"""The clock and the radio. STUB.

``TickRunner`` delivers messages, enforces range and ordering, and steps the
POGEMA environment. It never decides anyone's move.

Event ordering for one tick (see ``run_tick``):

1. Bid & reveal: every agent's ``start_tick`` in id order; each Bid is
   delivered to every agent within ``comm_range``.
2. Move (event-driven PIBT): repeat until every agent is DONE
     (A) if a message is pending, deliver the oldest one (FIFO);
     (B) otherwise activate the runnable agent with the highest
         ``current_priority``.
   No runnable agent while some are unfinished, or more than
   ``activation_cap`` activations, raises ``ProtocolStall`` (a bug, since
   waiting only ever points to strictly higher priority).
3. Execute: collect ``action()`` from every agent, check the moves are
   collision-free, then ``env.step``.
4. Pay & update: compute payments from the tick log, call ``end_tick``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from ..maps.movingai import Cell, Instance
from .context import Context, Message, MessageLog, Priority

if TYPE_CHECKING:
    from ..agents.bidding import BiddingAgent


class ProtocolStall(RuntimeError):
    """Phase 2 could not finish. Means the implementation is wrong."""


@dataclass(frozen=True)
class ProtocolConfig:
    comm_range: int = 2  # Manhattan distance; r >= 2 is needed for correctness
    alpha: float = 1.0  # weight on waiting time in the priority
    emergency_wait: int | None = None  # W*; None disables the emergency tier
    activation_cap: int = 100_000  # per tick, guards against livelock bugs


@dataclass(frozen=True)
class TickRecord:
    """Everything that happened in one tick. Index k describes the move from
    ``paths[i][k]`` to ``paths[i][k + 1]``."""

    t: int
    positions: tuple[Cell, ...]  # before the move
    bids: tuple[float, ...]
    priorities: tuple[Priority, ...]
    emergency: tuple[bool, ...]  # in the emergency tier this tick
    reserved: tuple[Cell, ...]  # cell each agent reserved for the next tick
    payments: tuple[float, ...]
    messages: tuple[MessageLog, ...] = ()
    activations: int = 0
    max_push_chain: int = 0
    decide_s: float = 0.0  # time spent in phases 1, 2 and 4 (not env.step)


@dataclass(frozen=True)
class TickOutcome:
    observations: Any
    record: TickRecord
    terminated: Sequence[bool]
    truncated: Sequence[bool]


class TickRunner:
    """One per episode. ``rollout.run_protocol`` calls ``run_tick`` once per step."""

    def __init__(
        self,
        env: Any,
        agents: Sequence["BiddingAgent"],
        instance: Instance,
        config: ProtocolConfig | None = None,
    ):
        self.env = env
        self.agents = list(agents)
        self.instance = instance
        self.config = config or ProtocolConfig()
        self.t = 0
        self.log: list[MessageLog] = []

    # ------------------------------------------------------------ main entry

    def run_tick(self, observations: Any) -> TickOutcome:
        """Run phases 1-4 for one timestep and return what happened."""
        raise NotImplementedError

    # --------------------------------------------------------------- helpers

    def positions(self) -> list[Cell]:
        """Current (row, col) of every agent; the runner may know this, agents may not."""
        raise NotImplementedError

    def within_range(self, agent_id: int) -> list[int]:
        """Ids of agents within ``comm_range`` of ``agent_id`` (excluding itself)."""
        raise NotImplementedError

    def context_for(self, agent_id: int) -> Context:
        return Context(self, agent_id)

    def deliver(self, message: Message) -> list[Message]:
        """Range-check, log and hand ``message`` to its receiver(s); return replies."""
        raise NotImplementedError

    def assert_collision_free(self, reserved: Sequence[Cell]) -> None:
        """Raise if the reserved cells contain a vertex or swap conflict."""
        raise NotImplementedError
