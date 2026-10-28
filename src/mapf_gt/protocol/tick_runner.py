"""The clock and the radio.

``TickRunner`` delivers messages, enforces range and ordering, and steps the
POGEMA environment. It never decides anyone's move.

Event ordering for one tick (see ``run_tick``):

1. Bid & reveal: every agent's ``start_tick`` in id order; each Bid is
   delivered to every agent within ``comm_range``.
2. Move (event-driven PIBT): repeat until every agent is DONE
     (A) if a message is pending, deliver the oldest one (FIFO);
     (B) otherwise activate the runnable agent with the highest
         ``current_priority``.
   A Push makes the receiver plan straight away, and a Reply makes the pusher
   carry on, so one push chain resolves completely (depth first) before the
   next root starts, exactly as in centralised PIBT.
   No runnable agent while some are unfinished, a pending push with no
   messages left, or more than ``activation_cap`` activations, raises
   ``ProtocolStall`` (a bug, since waiting only ever points to strictly
   higher priority).
3. Execute: collect ``action()`` from every agent, check the moves are
   collision-free (``ProtocolStall`` if not), then ``env.step``.
4. Pay & update: compute payments from the tick log, call ``end_tick``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

from ..maps.movingai import Cell, Instance
from ..maps.pogema_adapter import agent_positions, move_table
from .context import (
    Context,
    Message,
    MessageLog,
    Priority,
    PublicView,
    Push,
    Status,
    payload_size,
)

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
    reads: int = 0  # Context reads (each is one request/response with a neighbour)


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
        self.moves = move_table(env.unwrapped.grid_config)
        self.t = 0
        self.log: list[MessageLog] = []  # this tick's deliveries
        self.reads = 0
        self._positions: list[Cell] = []
        self._at: dict[Cell, int] = {}
        self._range_cache: dict[int, list[int]] = {}
        self._depth: dict[int, int] = {}

    # ------------------------------------------------------------ main entry

    def run_tick(self, observations: Any) -> TickOutcome:
        """Run phases 1-4 for one timestep and return what happened."""
        started = time.perf_counter()
        self._begin_tick()

        # Phase 1: bid and reveal.
        for agent in sorted(self.agents, key=lambda a: a.agent_id):
            bid = agent.start_tick(observations[agent.agent_id], self.context_for(agent.agent_id))
            if tuple(agent.pos) != self._positions[agent.agent_id]:
                raise ProtocolStall(
                    f"agent {agent.agent_id} thinks it is at {agent.pos}, "
                    f"POGEMA has it at {self._positions[agent.agent_id]}"
                )
            self.deliver(bid)

        # Phase 2: event-driven PIBT.
        activations = self._negotiate()

        # Phase 3: execute.
        reserved = [tuple(agent.reserved) for agent in self.agents]
        self.assert_collision_free(reserved)
        actions = [agent.action(self.context_for(agent.agent_id)) for agent in self.agents]
        deciding = time.perf_counter() - started
        observations, _, terminated, truncated, _ = self.env.step(actions)

        # Phase 4: pay and update.
        started = time.perf_counter()
        record = TickRecord(
            t=self.t,
            positions=tuple(self._positions),
            bids=tuple(float(agent.bid) for agent in self.agents),
            priorities=tuple(agent.priority for agent in self.agents),
            emergency=tuple(agent.priority[0] == 1 for agent in self.agents),
            reserved=tuple(reserved),
            payments=tuple(0.0 for _ in self.agents),
            messages=tuple(self.log),
            activations=activations,
            max_push_chain=max(self._depth.values(), default=0),
            reads=self.reads,
        )
        from .payments import critical_bids  # payments imports this module

        payments = critical_bids(record, self.config)
        for agent in self.agents:
            agent.end_tick(observations[agent.agent_id], payments[agent.agent_id])
        deciding += time.perf_counter() - started
        record = replace(record, payments=tuple(payments), decide_s=deciding)

        self.t += 1
        return TickOutcome(observations, record, terminated, truncated)

    def _begin_tick(self) -> None:
        self._positions = list(agent_positions(self.env))
        self._at = {cell: i for i, cell in enumerate(self._positions)}
        self._range_cache = {}
        self._depth = {}
        self.log = []
        self.reads = 0

    def _negotiate(self) -> int:
        queue: deque[Message] = deque()
        activations = 0
        while True:
            if queue:
                message = queue.popleft()
                if isinstance(message, Push):
                    self._depth[message.receiver] = self._depth.get(message.sender, 0) + 1
                queue.extend(self.deliver(message))
                activations += 1
            else:
                unfinished = [a for a in self.agents if a.status != Status.DONE]
                if not unfinished:
                    return activations
                waiting = [a.agent_id for a in unfinished if a.status == Status.AWAITING_REPLY]
                if waiting:
                    raise ProtocolStall(f"t={self.t}: agents {waiting} wait for replies that never came")
                runnable = [a for a in unfinished if a.runnable(self.context_for(a.agent_id))]
                if not runnable:
                    ids = [a.agent_id for a in unfinished]
                    raise ProtocolStall(f"t={self.t}: agents {ids} unfinished but none runnable")
                root = max(runnable, key=lambda a: a.current_priority)
                queue.extend(root.step(self.context_for(root.agent_id)))
                activations += 1
            if activations > self.config.activation_cap:
                raise ProtocolStall(f"t={self.t}: more than {self.config.activation_cap} activations")

    # --------------------------------------------------------------- helpers

    def positions(self) -> list[Cell]:
        """Current (row, col) of every agent; the runner may know this, agents may not."""
        return list(self._positions)

    def agent_at(self, cell: Cell) -> int | None:
        """Agent standing on ``cell`` at the start of this tick, if any."""
        return self._at.get(tuple(cell))

    def public(self, agent_id: int) -> PublicView:
        return self.agents[agent_id].public_view()

    def within_range(self, agent_id: int) -> list[int]:
        """Ids of agents within ``comm_range`` (Manhattan) of ``agent_id``, excluding itself."""
        if agent_id not in self._range_cache:
            r = self.config.comm_range
            row, col = self._positions[agent_id]
            found = []
            for d_row in range(-r, r + 1):
                reach = r - abs(d_row)
                for d_col in range(-reach, reach + 1):
                    j = self._at.get((row + d_row, col + d_col))
                    if j is not None and j != agent_id:
                        found.append(j)
            self._range_cache[agent_id] = sorted(found)
        return self._range_cache[agent_id]

    def context_for(self, agent_id: int) -> Context:
        return Context(self, agent_id)

    def deliver(self, message: Message) -> list[Message]:
        """Range-check, log and hand ``message`` to its receiver(s); return replies."""
        in_range = self.within_range(message.sender)
        if message.receiver is None:
            receivers = in_range
        elif message.receiver in in_range:
            receivers = [message.receiver]
        else:
            raise ProtocolStall(
                f"t={self.t}: {type(message).__name__} from {message.sender} to "
                f"{message.receiver} is out of range"
            )
        replies: list[Message] = []
        for j in receivers:
            self.log.append(
                MessageLog(
                    tick=self.t,
                    seq=len(self.log),
                    kind=type(message).__name__.lower(),
                    sender=message.sender,
                    receiver=j,
                    size=payload_size(message),
                )
            )
            replies.extend(self.agents[j].on_message(self.context_for(j), message))
        return replies

    def assert_collision_free(self, reserved: Sequence[Cell]) -> None:
        """Raise ``ProtocolStall`` if the reserved cells contain a vertex or swap conflict."""
        seen: dict[Cell, int] = {}
        for i, cell in enumerate(reserved):
            if cell in seen:
                raise ProtocolStall(f"t={self.t}: agents {seen[cell]} and {i} both reserved {cell}")
            seen[cell] = i
        for i, (frm, to) in enumerate(zip(self._positions, reserved)):
            j = self._at.get(to)
            if j is not None and j != i and frm != to and reserved[j] == frm:
                raise ProtocolStall(f"t={self.t}: agents {i} and {j} swap {frm} and {to}")
