"""PIBT with true urgencies as priorities: the cooperative oracle baseline.

Centralised, using the vendored pypibt's own ``PIBT.step`` (unmodified) with
two choices made from outside:

* priorities are the protocol's (``protocol/priority.py``, so the same
  rule switches apply), computed from TRUE urgencies. With the defaults:
  (tier, c_i * (d_i + alpha * eps_i), i), where d_i is the same public heatmap
  delay the agents compute (``protocol.heatmap_delay``, from the agents within
  ``comm_range``), eps_i counts timesteps since the agent was last at its goal
  (0 at the goal), and tier = 1 once eps_i >= emergency_wait;
* pypibt's random tie-break shuffle is switched off, so ties keep its
  neighbour order (stay, left, right, up, down).

With truthful reports the decentralised protocol uses the same priorities. It
produces exactly the same paths when every agent can hear every other
(communication range covering the map); with a shorter range, chains in
different places run concurrently and may differ (``validate.pibt_divergence``).
"""

from __future__ import annotations

import time
from typing import Sequence

from ..maps.movingai import Cell, Instance
from ..protocol.heatmap_delay import distances_to, heatmap_delay
from ..protocol.priority import next_age, next_since_goal, priority
from ..protocol.tick_runner import ProtocolConfig
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
        config: ProtocolConfig | None = None,
        max_timestep: int = 256,
    ):
        self.urgencies = list(urgencies)
        self.config = config or ProtocolConfig()
        self.max_timestep = max_timestep

    def delays(self, instance: Instance, positions: Sequence[Cell]) -> list[float]:
        """Each agent's heatmap delay, from the agents within ``comm_range``."""
        r = self.config.comm_range
        out = []
        for i, (pos, goal) in enumerate(zip(positions, instance.goals)):
            neighbours = [
                (other, instance.goals[j])
                for j, other in enumerate(positions)
                if j != i and abs(other[0] - pos[0]) + abs(other[1] - pos[1]) <= r
            ]
            out.append(heatmap_delay(instance.grid, pos, goal, neighbours, self.config))
        return out

    def priorities(
        self,
        delays: Sequence[float],
        eps: Sequence[int],
        since_goal: Sequence[int] | None = None,
    ) -> list[tuple[int, float, int]]:
        since_goal = since_goal if since_goal is not None else eps
        return [
            priority(urgency, d, e, g, i, self.config)
            for i, (urgency, d, e, g) in enumerate(zip(self.urgencies, delays, eps, since_goal))
        ]

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
        dist = [distances_to(instance.grid, goal) for goal in goals]
        far = instance.grid.height * instance.grid.width
        eps = [0] * instance.num_agents
        since_goal = [0] * instance.num_agents
        configs = [config]
        while config != goals and len(configs) <= self.max_timestep:
            priorities = self.priorities(self.delays(instance, config), eps, since_goal)
            nxt = [tuple(int(v) for v in cell) for cell in planner.step(config, priorities)]
            eps = [
                next_age(e, new == goal, d.get(new, far) < d.get(old, far), self.config)
                for old, new, goal, d, e in zip(config, nxt, goals, dist, eps)
            ]
            since_goal = [next_since_goal(g, new == goal) for new, goal, g in zip(nxt, goals, since_goal)]
            config = nxt
            configs.append(config)
        return [[c[i] for c in configs] for i in range(instance.num_agents)]

    def solve(self, instance: Instance, time_limit_s: float = 30.0) -> Solution | None:
        started = time.perf_counter()
        paths = self.rollout(instance)
        runtime = time.perf_counter() - started
        if [path[-1] for path in paths] != [tuple(goal) for goal in instance.goals]:
            return None  # ran out of timesteps
        return Solution(paths, runtime, self.name)
