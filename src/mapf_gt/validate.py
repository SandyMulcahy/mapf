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

from typing import Sequence

from .maps.movingai import Cell, Instance

Path = Sequence[Cell]


def position_at(path: Path, t: int) -> Cell:
    """Where an agent is at time ``t``; it stays at its last cell afterwards."""
    return path[t] if t < len(path) else path[-1]


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

    paths = [[tuple(cell) for cell in path] for path in paths]
    horizon = max(len(path) for path in paths)
    for t in range(horizon):
        occupied: dict[Cell, int] = {}
        for agent, path in enumerate(paths):
            cell = position_at(path, t)
            if cell in occupied:
                problems.append(
                    f"vertex conflict: agents {occupied[cell]} and {agent} "
                    f"at {cell}, t={t}"
                )
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
                    problems.append(
                        f"edge conflict: agents {a} and {b} swap "
                        f"{a_from} and {a_to}, t={t}"
                    )
    return problems


def is_valid(instance: Instance, paths: Sequence[Path]) -> bool:
    return not find_problems(instance, paths)
