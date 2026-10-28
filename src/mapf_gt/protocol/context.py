"""What an agent may see and do while the protocol runs. STUB.

Agents never hold references to each other. Each time the runner activates an
agent it hands over a fresh ``Context`` for that agent. Reads go through the
context (and are counted as messages); everything else is a ``Message`` the
agent returns, which the runner delivers.

The context enforces two rules, so the protocol stays decentralised:
  * range: an agent only sees agents within ``comm_range`` (Manhattan) of it
  * privacy: only ``PublicView`` fields are visible; urgency, bid multiplier
    and payments stay on the agent object
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..maps.movingai import Cell

if TYPE_CHECKING:
    from .tick_runner import TickRunner

# (tier, value, agent_id), compared lexicographically, higher wins.
# tier = 1 for the emergency tier (w_i >= W*), else 0; value = b_i + alpha * w_i.
Priority = tuple[int, float, int]


class Status(str, Enum):
    IDLE = "idle"  # may plan when runnable
    AWAITING_REPLY = "awaiting_reply"  # pushed someone, waiting for valid/invalid
    DONE = "done"  # has reserved its cell for this tick


@dataclass(frozen=True)
class PublicView:
    """The part of an agent's state its neighbours are allowed to read."""

    id: int
    pos: Cell
    goal: Cell
    w: int  # consecutive ticks not at goal
    priority: Priority  # own priority this tick
    current_priority: Priority  # own, or inherited from a pusher
    status: Status
    reserved: Cell | None  # cell reserved for next tick, once DONE


# ------------------------------------------------------------------ messages


@dataclass(frozen=True)
class Message:
    """Base class. ``receiver=None`` means broadcast to everyone in range."""

    sender: int
    receiver: int | None


@dataclass(frozen=True)
class Bid(Message):
    """Phase 1: announce position, goal and priority to everyone in range."""

    pos: Cell
    goal: Cell
    priority: Priority


@dataclass(frozen=True)
class Push(Message):
    """Phase 2: ask the receiver to vacate its cell; it inherits ``priority``."""

    priority: Priority


@dataclass(frozen=True)
class Reply(Message):
    """Phase 2: answer to a Push. valid = the pushed agent found somewhere to go."""

    valid: bool


@dataclass(frozen=True)
class Done(Message):
    """Phase 2: broadcast that the sender has reserved ``cell`` for next tick."""

    cell: Cell


@dataclass(frozen=True)
class MessageLog:
    """One delivery, as recorded by the runner (a broadcast logs one per receiver)."""

    tick: int
    seq: int  # delivery order within the tick
    kind: str  # "bid" | "push" | "reply" | "done" | "read"
    sender: int
    receiver: int
    size: int  # rough payload size, for communication metrics


# ------------------------------------------------------------------- context


def payload_size(message: Message) -> int:
    """Number of scalar values a message carries, for communication metrics."""
    if isinstance(message, Bid):
        return 2 + 2 + 3  # pos, goal, priority
    if isinstance(message, Push):
        return 3  # priority
    if isinstance(message, Done):
        return 2  # cell
    return 1  # Reply: valid


class Context:
    """Handle given to one agent for one activation.

    Every read goes through the runner, only returns agents within
    ``comm_range``, and only exposes ``PublicView`` fields. Each call counts as
    one read for the communication metrics.
    """

    def __init__(self, runner: "TickRunner", agent_id: int):
        self._runner = runner
        self._me = agent_id

    @property
    def me(self) -> int:
        return self._me

    @property
    def config(self):
        """The protocol's public parameters (alpha, emergency tier, range)."""
        return self._runner.config

    @property
    def moves(self) -> dict[Cell, int]:
        """(d_row, d_col) -> POGEMA action index, from the installed POGEMA."""
        return self._runner.moves

    def neighbours(self) -> list[PublicView]:
        """Public views of every other agent within ``comm_range``."""
        self._runner.reads += 1
        return [self._runner.public(j) for j in self._runner.within_range(self._me)]

    def view(self, agent_id: int) -> PublicView:
        """Public view of one agent. Raises if it is out of range."""
        self._runner.reads += 1
        if agent_id != self._me and agent_id not in self._runner.within_range(self._me):
            raise PermissionError(f"agent {self._me} cannot see agent {agent_id}")
        return self._runner.public(agent_id)

    def occupant(self, cell: Cell) -> int | None:
        """Id of the agent currently on ``cell`` (if in range), else None."""
        self._runner.reads += 1
        cell = tuple(cell)
        j = self._runner.agent_at(cell)
        if j is None or j == self._me or j in self._runner.within_range(self._me):
            return j
        return None

    def cell_free(self, cell: Cell) -> bool:
        """True if no other agent in range has reserved ``cell`` (including a
        tentative reservation while it waits for a reply), and moving there
        would not swap with the agent on it.

        Anyone who could reserve ``cell`` is within distance 2 of me, so this
        is exact when ``comm_range >= 2``.
        """
        self._runner.reads += 1
        cell = tuple(cell)
        others = [self._runner.public(j) for j in self._runner.within_range(self._me)]
        if any(view.reserved == cell for view in others):
            return False
        me = self._runner.public(self._me)
        for view in others:
            if view.pos == cell and view.reserved == me.pos:
                return False  # it is moving into my cell: a swap
        return True
