"""LaCAM* in pure Python, using the vendored py-lacam (PIBT branch).

No compiling needed, but it is far slower than the C++ lacam3 and lacks the
lacam3 improvements. Use it to try ideas and on small instances; use
``LaCAM`` (the C++ wrapper) for results you report.

Like lacam3 it is anytime: it finds a solution, then keeps improving it until
the time limit or until it proves the solution optimal.
"""

from __future__ import annotations

import time

import numpy as np

from ..maps.movingai import Instance
from ._pylacam import Config, LaCAM as _LaCAM
from .base import Solution, Solver


class PyLaCAM(Solver):
    name = "pylacam"

    def __init__(self, seed: int = 0, refine: bool = True):
        """``refine=False`` stops at the first solution (plain LaCAM)."""
        self.seed = seed
        self.refine = refine

    def solve(self, instance: Instance, time_limit_s: float = 30.0) -> Solution | None:
        free = ~instance.grid.obstacles  # py-lacam marks free cells True
        started = time.perf_counter()
        configs = _LaCAM().solve(
            grid=np.ascontiguousarray(free),
            starts=Config(list(instance.starts)),  # both use (row, col)
            goals=Config(list(instance.goals)),
            time_limit_ms=int(time_limit_s * 1000),
            flg_star=self.refine,
            seed=self.seed,
            verbose=0,
        )
        runtime = time.perf_counter() - started
        if not configs:
            return None
        paths = [
            [tuple(int(v) for v in config[agent]) for config in configs]
            for agent in range(instance.num_agents)
        ]
        return Solution(paths, runtime, self.name)
