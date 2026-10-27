"""Per-tick payments. STUB.

Myerson's payment for agent i with allocation x_i(.) (others' bids fixed):

    p_i = b_i * x_i(b_i) - integral_0^{b_i} x_i(z) dz

x_i only changes where b_i crosses a neighbour's priority, so the integral is
a short sum over those thresholds. With a binary allocation (got the cell or
not) this reduces to the critical bid max(0, P_loser - alpha * w_i). Agents
that were only pushed (inherited priority) pay nothing.

Every threshold comes from priorities within range, so in the write-up this
calculation belongs to the winning agent; the runner does it centrally only
because that is cheaper in simulation.
"""

from __future__ import annotations

from typing import Sequence

from .tick_runner import ProtocolConfig, TickRecord


def critical_bids(record: TickRecord, config: ProtocolConfig) -> list[float]:
    """Payment owed by each agent for the tick described by ``record``."""
    raise NotImplementedError


def myerson_payment(bid: float, thresholds: Sequence[tuple[float, float]]) -> float:
    """Myerson payment for a step allocation.

    ``thresholds`` is [(z, x_i just above z), ...] in increasing z, with the
    allocation at z = 0 given first.
    """
    raise NotImplementedError
