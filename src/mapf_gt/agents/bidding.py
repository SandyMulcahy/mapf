"""Agent that takes part in the bidding protocol.

Driven by ``protocol.TickRunner``, not by ``rollout.run_agents``: each tick
the runner calls, in order,

    start_tick -> (runnable / step / on_message, repeatedly) -> action -> end_tick

Bid: b_i = multiplier * urgency * d_i. For now d_i = 1, so a truthful agent
(multiplier = 1) bids its urgency. The k-step lookahead for d_i comes later.
Priority: (tier, b_i + alpha * w_i, id), where tier = 1 once w_i reaches the
emergency threshold (if one is set).

Move step: decentralised PIBT. The agent tries cells best first; if the cell
it wants is occupied by an agent that has not planned yet, it reserves the
cell tentatively and pushes that agent, which inherits its priority and plans
at once. A valid reply confirms the cell; an invalid one means try the next.

Tie-break rule, the same as the vendored pypibt with its shuffle switched
off (and as ``validate.reference_pibt_step`` and ``solvers.UrgencyPIBT``):
candidate cells are [stay, left, right, up, down], keeping only free cells,
stable-sorted by distance to goal, so ties keep that order.

The agent knows the static map, its start and its goal. Everything about
other agents comes through ``Context`` or messages.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Callable, Sequence

from ..maps.movingai import Cell, GridMap, Instance
from ..protocol.context import (
    Bid,
    Context,
    Done,
    Message,
    Priority,
    PublicView,
    Push,
    Reply,
    Status,
)
from .base import Agent

# pypibt's neighbour order: left, right, up, down, as (d_row, d_col).
PIBT_NEIGHBOUR_ORDER: tuple[Cell, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))


def pibt_candidates(grid: GridMap, cell: Cell) -> list[Cell]:
    """[stay] + free neighbours in pypibt's order, before sorting by distance."""
    row, col = cell
    out = [tuple(cell)]
    for d_row, d_col in PIBT_NEIGHBOUR_ORDER:
        nxt = (row + d_row, col + d_col)
        if grid.is_free(nxt):
            out.append(nxt)
    return out


def _distances_to(grid: GridMap, goal: Cell) -> dict[Cell, int]:
    """BFS distance from every reachable free cell to ``goal``."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in grid.neighbours(cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


class BiddingAgent(Agent):
    name = "bidding"

    def __init__(
        self,
        agent_id: int,
        urgency: float,
        grid: GridMap,
        start: Cell,
        goal: Cell,
        multiplier: float = 1.0,
    ):
        super().__init__(agent_id)
        # private: never exposed through PublicView
        self.urgency = urgency
        self.multiplier = multiplier
        self.total_paid = 0.0

        # static knowledge
        self.grid = grid
        self.start = tuple(start)
        self.goal = tuple(goal)
        self._dist = _distances_to(grid, self.goal)
        self._unreachable = grid.height * grid.width
        self._offset = (0, 0)  # map cell = observation xy + offset

        # public (read by neighbours through Context)
        self.pos: Cell = self.start
        self.w = 0
        self.priority: Priority | None = None
        self.current_priority: Priority | None = None

        # per-tick protocol state
        self.status = Status.IDLE
        self.candidates: list[Cell] = []
        self.reserved: Cell | None = None
        self.pusher: int | None = None
        self.awaiting: int | None = None
        self.bid = 0.0

    @property
    def strategy(self) -> str:
        """Label used in results, e.g. 'truthful' or '2x'."""
        return "truthful" if self.multiplier == 1.0 else f"{self.multiplier:g}x"

    def distance(self, cell: Cell) -> int:
        """Shortest-path distance from ``cell`` to this agent's goal."""
        return self._dist.get(tuple(cell), self._unreachable)

    def _to_map(self, xy: Sequence[int]) -> Cell:
        return (int(xy[0]) + self._offset[0], int(xy[1]) + self._offset[1])

    # ------------------------------------------------------------ per episode

    def reset(self, observation: Any) -> None:
        """Work out how observation coordinates map onto the map; w = 0, total_paid = 0.

        POGEMA's ``POMAPF`` observation gives ``xy`` relative to the start in
        some versions; the offset makes this work either way, and the goal
        check catches anything else.
        """
        xy = observation["xy"]
        self._offset = (self.start[0] - int(xy[0]), self.start[1] - int(xy[1]))
        target = self._to_map(observation["target_xy"])
        if target != self.goal:
            raise ValueError(
                f"agent {self.agent_id}: observation puts the goal at {target}, "
                f"expected {self.goal}; check POGEMA's coordinate convention"
            )
        self.pos = self.start
        self.w = 0
        self.total_paid = 0.0
        self._clear_tick()

    def _clear_tick(self) -> None:
        self.status = Status.IDLE
        self.candidates = []
        self.reserved = None
        self.pusher = None
        self.awaiting = None

    # --------------------------------------------------------------- per tick

    def start_tick(self, observation: Any, ctx: Context) -> Message:
        """Phase 1: compute the bid and priority, clear per-tick state, build
        the candidate list. Returns the Bid broadcast."""
        self.pos = self._to_map(observation["xy"])
        self._clear_tick()

        delay = 1.0  # d_i; replace with the k-step lookahead
        self.bid = self.multiplier * self.urgency * delay
        config = ctx.config
        emergency = config.emergency_wait is not None and self.w >= config.emergency_wait
        self.priority = (1 if emergency else 0, self.bid + config.alpha * self.w, self.agent_id)
        self.current_priority = self.priority

        self.candidates = sorted(pibt_candidates(self.grid, self.pos), key=self.distance)
        return Bid(self.agent_id, None, self.pos, self.goal, self.priority)

    def runnable(self, ctx: Context) -> bool:
        """Not DONE, not AWAITING_REPLY, and no unfinished agent in range has a
        higher priority than ``current_priority``."""
        if self.status != Status.IDLE:
            return False
        return not any(
            view.status != Status.DONE and view.current_priority > self.current_priority
            for view in ctx.neighbours()
        )

    def step(self, ctx: Context) -> list[Message]:
        """Phase 2: one planning attempt (reserve a free cell, or push its occupant)."""
        while self.candidates:
            cell = self.candidates.pop(0)
            if not ctx.cell_free(cell):
                continue
            self.reserved = cell
            occupant = ctx.occupant(cell)
            if occupant is None or occupant == self.agent_id:
                return self._finish(valid=True)
            if ctx.view(occupant).reserved is not None:
                return self._finish(valid=True)  # it has already planned elsewhere
            self.status = Status.AWAITING_REPLY
            self.awaiting = occupant
            return [Push(self.agent_id, occupant, self.current_priority)]
        self.reserved = self.pos  # nowhere to go: stay
        return self._finish(valid=False)

    def _finish(self, valid: bool) -> list[Message]:
        self.status = Status.DONE
        out: list[Message] = [Done(self.agent_id, None, self.reserved)]
        if self.pusher is not None:
            out.append(Reply(self.agent_id, self.pusher, valid))
        return out

    def on_message(self, ctx: Context, message: Message) -> list[Message]:
        """Phase 2: handle a Push (inherit priority) or a Reply (valid / invalid)."""
        if isinstance(message, Push):
            if self.status != Status.IDLE or self.reserved is not None:
                raise RuntimeError(f"agent {self.agent_id} pushed after it had planned")
            self.pusher = message.sender
            self.current_priority = message.priority
            return self.step(ctx)
        if isinstance(message, Reply):
            if self.status != Status.AWAITING_REPLY or message.sender != self.awaiting:
                raise RuntimeError(f"agent {self.agent_id} got an unexpected reply")
            self.awaiting = None
            if message.valid:
                return self._finish(valid=True)
            self.status = Status.IDLE
            return self.step(ctx)  # the pushed agent stays put: try the next cell
        return []  # Bid and Done carry information only

    def action(self, ctx: Context) -> int:
        """Phase 3: POGEMA action that moves from ``pos`` to ``reserved``."""
        delta = (self.reserved[0] - self.pos[0], self.reserved[1] - self.pos[1])
        return ctx.moves[delta]

    def end_tick(self, observation: Any, payment: float) -> None:
        """Phase 4: update position, add payment, w += 1 (or 0 at goal)."""
        self.total_paid += payment
        self.pos = self._to_map(observation["xy"])
        self.w = 0 if self.pos == self.goal else self.w + 1

    def public_view(self) -> PublicView:
        """What the runner shows neighbours through ``Context``."""
        return PublicView(
            id=self.agent_id,
            pos=self.pos,
            goal=self.goal,
            w=self.w,
            priority=self.priority,
            current_priority=self.current_priority,
            status=self.status,
            reserved=self.reserved,
        )

    # ------------------------------------------------- base Agent interface

    def act(self, observation: Any) -> int:
        raise TypeError(
            "BiddingAgent is driven by protocol.TickRunner; use rollout.run_protocol"
        )


# ------------------------------------------------------------------ helpers


def draw_urgencies(n: int, seed: int, low: float = 1.0, high: float = 10.0) -> list[float]:
    """n urgencies drawn uniformly from [low, high], reproducible from ``seed``."""
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(n)]


def make_bidding_agents(
    instance: Instance,
    urgencies: Sequence[float],
    multipliers: Sequence[float] | None = None,
) -> Callable[[int], BiddingAgent]:
    """``make_agent`` for ``rollout.run_protocol``. All truthful unless
    ``multipliers`` is given."""
    if len(urgencies) != instance.num_agents:
        raise ValueError(f"need {instance.num_agents} urgencies, got {len(urgencies)}")
    multipliers = multipliers or [1.0] * instance.num_agents

    def make_agent(i: int) -> BiddingAgent:
        return BiddingAgent(
            i,
            urgencies[i],
            instance.grid,
            instance.starts[i],
            instance.goals[i],
            multipliers[i],
        )

    return make_agent
