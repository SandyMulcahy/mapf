"""Independent check that a set of paths is a legal MAPF solution.

Every result, from any solver or agent, should pass through here before its
metrics are recorded. That way a bug in a solver, a wrapper or the simulator
cannot quietly produce results that look better than they are.

Rules checked (the standard one-shot MAPF model of Stern et al., 2019):
  * each path starts at the agent's start and ends at its goal
  * each step is a wait or a move to a free neighbouring cell
  * no vertex conflict: two agents in the same cell at the same time
  * no edge conflict: two agents swapping cells in one step
  * agents stay at their goal after finishing, and still block that cell
Following (moving into a cell another agent is leaving) is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .maps.movingai import Cell, Instance

Path = Sequence[Cell]


@dataclass(frozen=True)
class Conflict:
    """A collision between two agents.

    For a vertex conflict ``cells`` is the shared cell. For an edge conflict
    it is ``(from, to)`` for the first agent, who swapped with the second.
    """

    kind: str  # "vertex" or "edge"
    agents: tuple[int, int]
    cells: tuple[Cell, ...]
    t: int

    def describe(self) -> str:
        a, b = self.agents
        if self.kind == "vertex":
            return f"vertex conflict: agents {a} and {b} at {self.cells[0]}, t={self.t}"
        frm, to = self.cells
        return f"edge conflict: agents {a} and {b} swap {frm} and {to}, t={self.t}"


def position_at(path: Path, t: int) -> Cell:
    """Where an agent is at time ``t``; it stays at its last cell afterwards."""
    return path[t] if t < len(path) else path[-1]


def find_conflicts(paths: Sequence[Path]) -> list[Conflict]:
    """Every vertex and edge conflict, in time order.

    Agents that have finished stay at their last cell and still collide.
    """
    paths = [[tuple(cell) for cell in path] for path in paths]
    conflicts: list[Conflict] = []
    horizon = max((len(path) for path in paths), default=0)
    for t in range(horizon):
        occupied: dict[Cell, int] = {}
        for agent, path in enumerate(paths):
            cell = position_at(path, t)
            if cell in occupied:
                conflicts.append(Conflict("vertex", (occupied[cell], agent), (cell,), t))
            else:
                occupied[cell] = agent
        if t == 0:
            continue
        for a in range(len(paths)):
            a_from, a_to = position_at(paths[a], t - 1), position_at(paths[a], t)
            if a_from == a_to:
                continue
            for b in range(a + 1, len(paths)):
                b_from, b_to = position_at(paths[b], t - 1), position_at(paths[b], t)
                if a_from == b_to and a_to == b_from:
                    conflicts.append(Conflict("edge", (a, b), (a_from, a_to), t))
    return conflicts


def find_problems(
    instance: Instance, paths: Sequence[Path], require_goals: bool = True
) -> list[str]:
    """Return a description of every rule broken. An empty list means valid.

    Set ``require_goals=False`` to check a rollout that ran out of time: the
    movement and collision rules are still enforced, but agents need not have
    arrived.
    """
    problems: list[str] = []
    grid = instance.grid

    if len(paths) != instance.num_agents:
        return [f"expected {instance.num_agents} paths, got {len(paths)}"]

    for agent, path in enumerate(paths):
        if len(path) == 0:
            problems.append(f"agent {agent}: empty path")
            continue
        path = [tuple(cell) for cell in path]
        if path[0] != instance.starts[agent]:
            problems.append(
                f"agent {agent}: starts at {path[0]}, expected {instance.starts[agent]}"
            )
        if require_goals and path[-1] != instance.goals[agent]:
            problems.append(
                f"agent {agent}: ends at {path[-1]}, expected {instance.goals[agent]}"
            )
        for t, cell in enumerate(path):
            if not grid.is_free(cell):
                problems.append(f"agent {agent}: t={t} cell {cell} is not free")
            if t > 0:
                prev = path[t - 1]
                distance = abs(cell[0] - prev[0]) + abs(cell[1] - prev[1])
                if distance > 1:
                    problems.append(
                        f"agent {agent}: t={t} jumps from {prev} to {cell}"
                    )
    if problems:
        return problems  # collision checks are meaningless on broken paths

    return [conflict.describe() for conflict in find_conflicts(paths)]


def is_valid(instance: Instance, paths: Sequence[Path]) -> bool:
    return not find_problems(instance, paths)
