"""PIBT with true urgencies as priorities: the cooperative oracle baseline.

Centralised, using the vendored pypibt's own ``PIBT.step`` (unmodified) with
two choices made from outside:

* priorities are the protocol's, computed from TRUE urgencies:
  (tier, c_i + alpha * w_i, i), where w_i counts consecutive steps not at goal
  (0 at goal) and tier = 1 once w_i >= emergency_wait;
* pypibt's random tie-break shuffle is switched off, so ties keep its
  neighbour order (stay, left, right, up, down).

With truthful bids and d_i = 1, the decentralised bidding protocol should
produce exactly the same paths (see tests/test_protocol.py).
"""

from __future__ import annotations

import time
from typing import Sequence

from ..maps.movingai import Cell, Instance
from ._pypibt import PIBT as _PIBT
from .base import Solution, Solver


class _NoShuffle:
    """Stands in for pypibt's random generator: leaves candidate order alone."""

    def shuffle(self, items) -> None:
        pass


class UrgencyPIBT(Solver):
    name = "urgency_pibt"

    def __init__(
        self,
        urgencies: Sequence[float],
        alpha: float = 1.0,
        emergency_wait: int | None = None,
        max_timestep: int = 256,
    ):
        self.urgencies = list(urgencies)
        self.alpha = alpha
        self.emergency_wait = emergency_wait
        self.max_timestep = max_timestep

    def priorities(self, waits: Sequence[int]) -> list[tuple[int, float, int]]:
        out = []
        for i, (urgency, w) in enumerate(zip(self.urgencies, waits)):
            emergency = self.emergency_wait is not None and w >= self.emergency_wait
            out.append((1 if emergency else 0, urgency + self.alpha * w, i))
        return out

    def rollout(self, instance: Instance) -> list[list[Cell]]:
        """Run until every agent is at its goal or ``max_timestep`` steps pass.

        Returns the paths either way, so unsolved runs can still be compared.
        """
        if len(self.urgencies) != instance.num_agents:
            raise ValueError(f"need {instance.num_agents} urgencies, got {len(self.urgencies)}")
        planner = _PIBT(
            ~instance.grid.obstacles,  # pypibt: True = free
            list(instance.starts),
            list(instance.goals),
        )
        planner.rng = _NoShuffle()
        goals = [tuple(goal) for goal in instance.goals]
        config = [tuple(cell) for cell in instance.starts]
        waits = [0] * instance.num_agents
        configs = [config]
        while config != goals and len(configs) <= self.max_timestep:
            config = [tuple(int(v) for v in cell) for cell in planner.step(config, self.priorities(waits))]
            configs.append(config)
            waits = [0 if cell == goal else w + 1 for cell, goal, w in zip(config, goals, waits)]
        return [[c[i] for c in configs] for i in range(instance.num_agents)]

    def solve(self, instance: Instance, time_limit_s: float = 30.0) -> Solution | None:
        started = time.perf_counter()
        paths = self.rollout(instance)
        runtime = time.perf_counter() - started
        if [path[-1] for path in paths] != [tuple(goal) for goal in instance.goals]:
            return None  # ran out of timesteps
        return Solution(paths, runtime, self.name)
