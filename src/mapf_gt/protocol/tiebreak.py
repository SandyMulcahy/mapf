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
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def tiebreak_permutation(n: int, t: int, agent_id: int, config) -> list[int]:
    """The order in which to take n candidates before the distance sort."""
    order = list(range(n))
    if config.random_tiebreak:
        random.Random(f"{config.tiebreak_seed}:{t}:{agent_id}").shuffle(order)
    return order


def order_candidates(
    candidates: Sequence[T],
    distance: Callable[[T], float],
    t: int,
    agent_id: int,
    config,
) -> list[T]:
    """Candidates in the order the agent tries them: shuffled if configured,
    then stable-sorted by ``distance``."""
    order = tiebreak_permutation(len(candidates), t, agent_id, config)
    return sorted((candidates[k] for k in order), key=distance)
