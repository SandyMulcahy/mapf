"""Agent that takes part in the bidding protocol. STUB.

Driven by ``protocol.TickRunner``, not by ``rollout.run_agents``: each tick
the runner calls, in order,

    start_tick -> (runnable / step / on_message, repeatedly) -> action -> end_tick

Bid strategy: b_i = multiplier * urgency * d_i, where d_i is the delay the
agent expects if it does not get its preferred cell (k-step lookahead).
multiplier = 1 is truthful; the evolutionary strategies (0.25x, 0.5x, 2x, 4x,
...) are other multipliers or small subclasses.

Tie-break rule (must match ``validate.reference_pibt_step``): candidate cells
are [stay] + ``GridMap.neighbours(pos)``, stable-sorted by distance to goal,
so ties keep that order.
"""

from __future__ import annotations

from typing import Any

from ..maps.movingai import Cell
from ..protocol.context import Context, Message, Priority, PublicView, Status
from .base import Agent


class BiddingAgent(Agent):
    name = "bidding"

    def __init__(self, agent_id: int, urgency: float, multiplier: float = 1.0):
        super().__init__(agent_id)
        # private: never exposed through PublicView
        self.urgency = urgency
        self.multiplier = multiplier
        self.total_paid = 0.0

        # public (read by neighbours through Context)
        self.pos: Cell | None = None
        self.goal: Cell | None = None
        self.w = 0
        self.priority: Priority | None = None
        self.current_priority: Priority | None = None

        # per-tick protocol state
        self.status = Status.IDLE
        self.candidates: list[Cell] = []
        self.reserved: Cell | None = None
        self.pusher: int | None = None
        self.bid = 0.0

    @property
    def strategy(self) -> str:
        """Label used in results, e.g. 'truthful' or '2x'."""
        return "truthful" if self.multiplier == 1.0 else f"{self.multiplier:g}x"

    # ------------------------------------------------------------ per episode

    def reset(self, observation: Any) -> None:
        """Cache goal and distances; w = 0, total_paid = 0."""
        raise NotImplementedError

    # --------------------------------------------------------------- per tick

    def start_tick(self, observation: Any, ctx: Context) -> Message:
        """Phase 1: compute d_i and bid, set priority, clear per-tick state,
        build the candidate list. Returns the Bid broadcast."""
        raise NotImplementedError

    def runnable(self, ctx: Context) -> bool:
        """Not DONE, not AWAITING_REPLY, and no unfinished agent in range has a
        higher priority than ``current_priority``."""
        raise NotImplementedError

    def step(self, ctx: Context) -> list[Message]:
        """Phase 2: one planning attempt (reserve a free cell, or push its occupant)."""
        raise NotImplementedError

    def on_message(self, ctx: Context, message: Message) -> list[Message]:
        """Phase 2: handle a Push (inherit priority) or a Reply (valid / invalid)."""
        raise NotImplementedError

    def action(self) -> int:
        """Phase 3: POGEMA action that moves from ``pos`` to ``reserved``."""
        raise NotImplementedError

    def end_tick(self, observation: Any, payment: float) -> None:
        """Phase 4: update position, add payment, w += 1 (or 0 at goal)."""
        raise NotImplementedError

    def public_view(self) -> PublicView:
        """What the runner shows neighbours through ``Context``."""
        raise NotImplementedError

    # ------------------------------------------------- base Agent interface

    def act(self, observation: Any) -> int:
        raise TypeError(
            "BiddingAgent is driven by protocol.TickRunner; use rollout.run_protocol"
        )
