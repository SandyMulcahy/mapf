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

For runs of the bidding protocol, ``find_protocol_problems`` also checks the
protocol's own invariants (range, payments, agreement with centralised PIBT),
rebuilding everything from the recorded run rather than trusting the runner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .maps.movingai import Cell, GridMap, Instance

if TYPE_CHECKING:
    from .rollout import RunResult

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


# --------------------------------------------------------------- protocol


def distances_to(grid: GridMap, goal: Cell) -> dict[Cell, int]:
    """BFS distance from every reachable free cell to ``goal``."""
    dist = {tuple(goal): 0}
    queue = deque([tuple(goal)])
    while queue:
        cell = queue.popleft()
        for nxt in grid.neighbours(cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def reference_pibt_step(
    grid: GridMap,
    positions: Sequence[Cell],
    goals: Sequence[Cell],
    priorities: Sequence,
    dist_cache: dict[Cell, dict[Cell, int]] | None = None,
) -> list[Cell]:
    """One step of centralised PIBT, deterministic, as the protocol must match.

    Same algorithm as the vendored pypibt ``PIBT.step``, but without its
    random tie-break: candidates are [stay] + ``grid.neighbours(pos)``,
    stable-sorted by distance to goal. ``priorities`` can be anything
    comparable (the protocol uses (tier, value, id) tuples); higher plans first.
    """
    positions = [tuple(cell) for cell in positions]
    n = len(positions)
    cache = dist_cache if dist_cache is not None else {}
    for goal in goals:
        if tuple(goal) not in cache:
            cache[tuple(goal)] = distances_to(grid, goal)
    unreachable = grid.height * grid.width

    occupied_now = {cell: i for i, cell in enumerate(positions)}
    occupied_next: dict[Cell, int] = {}
    nxt: list[Cell | None] = [None] * n

    def plan(i: int) -> bool:
        dist = cache[tuple(goals[i])]
        here = positions[i]
        candidates = sorted(
            [here, *grid.neighbours(here)], key=lambda v: dist.get(v, unreachable)
        )
        for v in candidates:
            if v in occupied_next:
                continue
            j = occupied_now.get(v)
            if j is not None and nxt[j] == here:  # would swap with j
                continue
            nxt[i] = v
            occupied_next[v] = i
            if j is not None and j != i and nxt[j] is None and not plan(j):
                continue  # j failed and now stays on v, so v is taken; try the next cell
            return True
        nxt[i] = here
        occupied_next[here] = i
        return False

    for i in sorted(range(n), key=lambda i: priorities[i], reverse=True):
        if nxt[i] is None:
            plan(i)
    return [cell for cell in nxt]  # type: ignore[misc]


def find_protocol_problems(run: "RunResult", check_pibt: bool = True) -> list[str]:
    """Every protocol invariant the run breaks. An empty list means it passed.

    Checks, from the recorded run only:
      * the paths obey the MAPF rules (goals not required if the run timed out)
      * the protocol did not stall
      * every message was delivered within ``comm_range``
      * each agent's reserved cell is where POGEMA actually put it
        (catches POGEMA silently resolving a collision)
      * payments are >= 0 and never exceed the bid (individual rationality)
      * with ``check_pibt``, the reserved cells equal centralised PIBT run
        with the same priorities
    """
    instance = run.instance
    problems = list(find_problems(instance, run.paths, require_goals=run.reached_goals))
    if run.stall is not None:
        problems.append(f"protocol stalled: {run.stall}")

    r = run.config.comm_range
    dist_cache: dict[Cell, dict[Cell, int]] = {}
    for k, tick in enumerate(run.ticks):
        here = [tuple(position_at(path, k)) for path in run.paths]
        after = [tuple(position_at(path, k + 1)) for path in run.paths]

        if list(map(tuple, tick.positions)) != here:
            problems.append(f"t={k}: recorded positions differ from the paths")

        for msg in tick.messages:
            (r0, c0), (r1, c1) = here[msg.sender], here[msg.receiver]
            if abs(r0 - r1) + abs(c0 - c1) > r:
                problems.append(
                    f"t={k}: {msg.kind} from {msg.sender} to {msg.receiver} "
                    f"is beyond range {r}"
                )

        for i, (want, got) in enumerate(zip(tick.reserved, after)):
            if tuple(want) != got:
                problems.append(f"t={k}: agent {i} reserved {tuple(want)} but is at {got}")

        for i, (paid, bid) in enumerate(zip(tick.payments, tick.bids)):
            if paid < -1e-9:
                problems.append(f"t={k}: agent {i} has negative payment {paid}")
            if paid > bid + 1e-9:
                problems.append(f"t={k}: agent {i} pays {paid} > its bid {bid}")

        if check_pibt:
            expected = reference_pibt_step(
                instance.grid, here, instance.goals, tick.priorities, dist_cache
            )
            for i, (want, got) in enumerate(zip(expected, tick.reserved)):
                if tuple(want) != tuple(got):
                    problems.append(
                        f"t={k}: agent {i} reserved {tuple(got)}, centralised PIBT "
                        f"gives {tuple(want)}"
                    )
    return problems
