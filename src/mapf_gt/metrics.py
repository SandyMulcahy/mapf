"""Standard MAPF metrics, computed the same way for every method."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .maps.movingai import Cell, Instance
from .validate import find_problems

Path = Sequence[Cell]


def path_cost(path: Path, goal: Cell) -> int:
    """Timesteps until the agent reaches its goal and never leaves again.

    Waiting at the goal at the end of a path is free, but waiting there and
    then moving away is not. If the agent never arrives, the cost is the full
    length of the path.
    """
    t = len(path) - 1
    if tuple(path[-1]) != tuple(goal):
        return t
    while t > 0 and tuple(path[t - 1]) == tuple(goal):
        t -= 1
    return t


def sum_of_costs(paths: Sequence[Path], goals: Sequence[Cell]) -> int:
    return sum(path_cost(path, goal) for path, goal in zip(paths, goals))


def makespan(paths: Sequence[Path], goals: Sequence[Cell]) -> int:
    return max(path_cost(path, goal) for path, goal in zip(paths, goals))


@dataclass(frozen=True)
class Result:
    """One row of experiment output."""

    success: bool
    sum_of_costs: int | None
    makespan: int | None
    agents_at_goal: int
    runtime_s: float
    problems: tuple[str, ...] = ()

    def as_row(self) -> dict:
        row = asdict(self)
        row["problems"] = "; ".join(self.problems)
        return row


def evaluate(
    instance: Instance, paths: Sequence[Path] | None, runtime_s: float
) -> Result:
    """Validate ``paths`` and compute metrics. ``None`` means no solution."""
    if paths is None:
        return Result(False, None, None, 0, runtime_s, ("no solution returned",))

    at_goal = sum(
        1
        for path, goal in zip(paths, instance.goals)
        if len(path) > 0 and tuple(path[-1]) == tuple(goal)
    )
    problems = find_problems(instance, paths)
    if problems:
        return Result(False, None, None, at_goal, runtime_s, tuple(problems))
    return Result(
        success=True,
        sum_of_costs=sum_of_costs(paths, instance.goals),
        makespan=makespan(paths, instance.goals),
        agents_at_goal=at_goal,
        runtime_s=runtime_s,
    )
