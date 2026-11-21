"""Tie-break order for candidate cells, shared by the agent, the oracle and
the validator.

Candidates start in pypibt's order (stay, left, right, up, down; free cells
only) and are then stable-sorted by distance to goal, so ties keep that order.

With ``config.random_tiebreak`` the candidates are first shuffled, as in
original PIBT, which makes ties random. The shuffle is keyed on
(``config.tiebreak_seed``, tick, agent id), not drawn from a shared stream:
the decentralised agents plan in a different order from the centralised
oracle, and both must still make the same draw. The key holds no urgency or
bid, and anyone who knows the seed can recompute it.

With ``config.tiebreak_prefer_free`` (PIBT's own tie-break, Okumura et al.
2019, §4.3), among equally short cells those not occupied by another agent at
the start of the tick come first, so an agent avoids pushing someone when an
equally good free cell exists. Applied after the shuffle and before the
distance sort, so the order is (distance, occupied, shuffled/pypibt order).
"""

from __future__ import annotations

import random
from typing import Callable, Collection, Sequence, TypeVar

T = TypeVar("T")


def tiebreak_permutation(n: int, t: int, agent_id: int, config) -> list[int]:
    """The order in which to take n candidates before the distance sort."""
    order = list(range(n))
    if config.random_tiebreak:
        random.Random(f"{config.tiebreak_seed}:{t}:{agent_id}").shuffle(order)
    return order


def tiebreak_order(
    candidates: Sequence[T],
    t: int,
    agent_id: int,
    config,
    occupied: Collection[T] = (),
) -> list[T]:
    """Candidates in tie-break order, before the distance sort: the keyed
    permutation, then (if ``tiebreak_prefer_free``) cells in ``occupied``
    (cells held by other agents) moved behind the free ones, stably."""
    order = tiebreak_permutation(len(candidates), t, agent_id, config)
    out = [candidates[k] for k in order]
    if config.tiebreak_prefer_free and occupied:
        out.sort(key=lambda cell: cell in occupied)  # stable: free first
    return out


def order_candidates(
    candidates: Sequence[T],
    distance: Callable[[T], float],
    t: int,
    agent_id: int,
    config,
    occupied: Collection[T] = (),
) -> list[T]:
    """Candidates in the order the agent tries them: ``tiebreak_order``, then
    stable-sorted by ``distance``."""
    return sorted(tiebreak_order(candidates, t, agent_id, config, occupied), key=distance)
