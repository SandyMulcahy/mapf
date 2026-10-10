"""Interface for centralised planners: whole instance in, whole plan out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..maps.movingai import Cell, Instance


@dataclass(frozen=True)
class Solution:
    paths: list[list[Cell]]  # paths[agent][t] = (row, col)
    runtime_s: float
    solver: str


class Solver(ABC):
    name: str = "solver"

    @abstractmethod
    def solve(self, instance: Instance, time_limit_s: float = 30.0) -> Solution | None:
        """Return a plan, or None if none was found within the time limit."""
