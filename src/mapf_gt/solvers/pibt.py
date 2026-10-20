"""PIBT (Okumura et al.), using the vendored pypibt."""

from __future__ import annotations

import time

from ..maps.movingai import Instance
from ._pypibt import PIBT as _PIBT
from .base import Solution, Solver


class PIBT(Solver):
    name = "pibt"

    def __init__(self, seed: int = 0, max_timestep: int = 1000):
        self.seed = seed
        self.max_timestep = max_timestep

    def solve(self, instance: Instance, time_limit_s: float = 30.0) -> Solution | None:
        started = time.perf_counter()
        planner = _PIBT(
            ~instance.grid.obstacles,  # pypibt: True = free
            list(instance.starts),  # both use (row, col)
            list(instance.goals),
            seed=self.seed,
        )
        configs = planner.run(max_timestep=self.max_timestep)
        runtime = time.perf_counter() - started
        if list(configs[-1]) != list(instance.goals):
            return None  # ran out of timesteps
        paths = [
            [tuple(int(v) for v in config[i]) for config in configs]
            for i in range(instance.num_agents)
        ]
        return Solution(paths, runtime, self.name)