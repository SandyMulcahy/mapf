"""Conflict-Based Search (Sharon et al., 2015), minimising sum of costs.

A plain, readable implementation with no improvements (no bypassing, no
prioritised conflicts, no high-level heuristics). It is here to give optimal
costs on small instances. For runtime comparisons or more than a couple of
dozen agents, use a C++ solver such as EECBS.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass, field

from ..maps.movingai import Cell, GridMap, Instance
from ..validate import position_at
from .base import Solution, Solver

VertexConstraint = tuple[Cell, int]  # may not be at cell at time t
EdgeConstraint = tuple[Cell, Cell, int]  # may not move from -> to, arriving at t


class _OutOfTime(Exception):
    pass


def distances_to(grid: GridMap, goal: Cell) -> dict[Cell, int]:
    """Shortest path length from every reachable cell to ``goal`` (BFS)."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in grid.neighbours(cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def space_time_astar(
    grid: GridMap,
    start: Cell,
    goal: Cell,
    heuristic: dict[Cell, int],
    vertex_constraints: frozenset[VertexConstraint] | set[VertexConstraint],
    edge_constraints: frozenset[EdgeConstraint] | set[EdgeConstraint],
    deadline: float = float("inf"),
) -> list[Cell] | None:
    """Shortest path for one agent that respects the constraints.

    The agent must be able to rest at the goal forever, so it may only finish
    after the last constraint on the goal cell.
    """
    if start not in heuristic:
        return None  # goal unreachable from start

    last_goal_block = max(
        (t for cell, t in vertex_constraints if cell == goal), default=-1
    )
    last_constraint = max(
        [t for _, t in vertex_constraints] + [t for _, _, t in edge_constraints],
        default=0,
    )
    # After the last constraint nothing changes, so a path never needs to be
    # longer than that plus a visit to every reachable cell.
    horizon = last_constraint + len(heuristic) + 1

    counter = itertools.count()
    open_list = [(heuristic[start], heuristic[start], next(counter), start, 0)]
    parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {(start, 0): None}
    closed: set[tuple[Cell, int]] = set()
    expansions = 0

    while open_list:
        _, _, _, cell, t = heapq.heappop(open_list)
        if (cell, t) in closed:
            continue
        closed.add((cell, t))

        if cell == goal and t > last_goal_block:
            path = []
            node: tuple[Cell, int] | None = (cell, t)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            return path[::-1]

        expansions += 1
        if expansions % 2048 == 0 and time.perf_counter() > deadline:
            raise _OutOfTime
        if t >= horizon:
            continue

        for nxt in [cell, *grid.neighbours(cell)]:
            state = (nxt, t + 1)
            if state in closed or state in parent:
                continue
            if state in vertex_constraints or (cell, nxt, t + 1) in edge_constraints:
                continue
            parent[state] = (cell, t)
            h = heuristic[nxt]
            heapq.heappush(open_list, (t + 1 + h, h, next(counter), nxt, t + 1))
    return None


def first_conflict(paths: list[list[Cell]]):
    """Earliest conflict as ("vertex", a, b, cell, t) or ("edge", a, b, u, v, t)."""
    horizon = max(len(path) for path in paths)
    for t in range(horizon):
        occupied: dict[Cell, int] = {}
        for agent, path in enumerate(paths):
            cell = position_at(path, t)
            if cell in occupied:
                return ("vertex", occupied[cell], agent, cell, t)
            occupied[cell] = agent
        if t == 0:
            continue
        moves: dict[tuple[Cell, Cell], int] = {}
        for agent, path in enumerate(paths):
            frm, to = position_at(path, t - 1), position_at(path, t)
            if frm == to:
                continue
            if (to, frm) in moves:
                return ("edge", moves[(to, frm)], agent, to, frm, t)
            moves[(frm, to)] = agent
    return None


@dataclass(order=True)
class _Node:
    cost: int
    tiebreak: int
    paths: list[list[Cell]] = field(compare=False)
    vertex: dict[int, frozenset[VertexConstraint]] = field(compare=False)
    edge: dict[int, frozenset[EdgeConstraint]] = field(compare=False)


class CBS(Solver):
    name = "cbs"

    def solve(self, instance: Instance, time_limit_s: float = 30.0) -> Solution | None:
        started = time.perf_counter()
        deadline = started + time_limit_s
        try:
            paths = self._search(instance, deadline)
        except _OutOfTime:
            return None
        if paths is None:
            return None
        return Solution(paths, time.perf_counter() - started, self.name)

    def _search(self, instance: Instance, deadline: float) -> list[list[Cell]] | None:
        grid = instance.grid
        n = instance.num_agents
        heuristics = [distances_to(grid, goal) for goal in instance.goals]
        counter = itertools.count()

        def plan(agent, vertex, edge):
            return space_time_astar(
                grid,
                instance.starts[agent],
                instance.goals[agent],
                heuristics[agent],
                vertex,
                edge,
                deadline,
            )

        empty_v: frozenset[VertexConstraint] = frozenset()
        empty_e: frozenset[EdgeConstraint] = frozenset()
        root_paths = []
        for agent in range(n):
            path = plan(agent, empty_v, empty_e)
            if path is None:
                return None
            root_paths.append(path)

        def cost(paths):
            return sum(len(path) - 1 for path in paths)

        open_list = [_Node(cost(root_paths), next(counter), root_paths, {}, {})]
        while open_list:
            if time.perf_counter() > deadline:
                raise _OutOfTime
            node = heapq.heappop(open_list)
            conflict = first_conflict(node.paths)
            if conflict is None:
                return node.paths

            if conflict[0] == "vertex":
                _, a, b, cell, t = conflict
                branches = [(a, (cell, t), None), (b, (cell, t), None)]
            else:
                _, a, b, u, v, t = conflict  # a moved u -> v, b moved v -> u
                branches = [(a, None, (u, v, t)), (b, None, (v, u, t))]

            for agent, new_vertex, new_edge in branches:
                vertex = dict(node.vertex)
                edge = dict(node.edge)
                if new_vertex is not None:
                    vertex[agent] = vertex.get(agent, empty_v) | {new_vertex}
                if new_edge is not None:
                    edge[agent] = edge.get(agent, empty_e) | {new_edge}
                path = plan(agent, vertex.get(agent, empty_v), edge.get(agent, empty_e))
                if path is None:
                    continue
                paths = list(node.paths)
                paths[agent] = path
                heapq.heappush(
                    open_list, _Node(cost(paths), next(counter), paths, vertex, edge)
                )
        return None
