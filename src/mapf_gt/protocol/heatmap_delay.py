"""Public delay estimate d_i from a heatmap of where neighbours will be.

d_i is the cost of losing this tick:

    d_i = (best k-step cost if forced to stay now)
        - (best k-step cost if it moves to its best cell now)

floored at ``config.min_delay``. A k-step path pi = (pi_1, ..., pi_k) from the
agent's cell costs

    sum_{t=1..k} (1 + lambda * H_t(pi_t))  +  dist(pi_k, goal)

where H_t(v) is the predicted number of other agents on cell v at step t
(``predict_heatmap``) and each step moves to an adjacent free cell or stays.
With an empty heatmap and an open path this gives d_i = 1.

Everything here uses public information only (the map, and positions and
goals of agents within range), so any neighbour can recompute an agent's d_i.
Parameters come from ``ProtocolConfig``: ``delay_horizon`` (k),
``heat_model``, ``heat_follow_prob`` (p), ``heat_weight`` (lambda) and ``min_delay``.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from ..maps.movingai import Cell, GridMap

Heatmap = list[dict[Cell, float]]  # heat[t - 1][cell] = expected agents on cell at step t

# pypibt's neighbour order: left, right, up, down, as (d_row, d_col).
_ORDER: tuple[Cell, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))


def predict_heatmap(
    grid: GridMap,
    neighbours: Sequence[tuple[Cell, Cell]],
    config,
) -> Heatmap:
    """Predicted occupancy of other agents for steps 1..k.

    ``neighbours`` is [(position, goal), ...] for the agents within range
    (not including the agent itself). Returns a list of length
    ``config.delay_horizon``; entry t - 1 maps a cell to the expected number
    of these agents on it at step t. Cells not in a dict have heat 0.

    Each neighbour is an independent Markov chain over its moves (stay or an
    adjacent free cell); heat is the sum of their occupancy probabilities.
    ``config.heat_model`` picks the move probabilities:

    * ``"none"``: no heat at all (d_i is the plain one-step cost of waiting).
    * ``"symmetric"`` (Keskin-style): every move equally likely; goals unused.
    * ``"goal"``: probability p = ``config.heat_follow_prob`` split evenly over
      the moves that minimise distance to the neighbour's goal, 1 - p split
      evenly over the other moves (all to the shortest moves if there are none).

    Neighbours on their goal get no special case in either model.
    """
    k = config.delay_horizon
    model = config.heat_model
    if model == "none":
        return [{} for _ in range(k)]
    if model == "symmetric":
        transition = _symmetric_moves
    elif model == "goal":
        transition = _goal_moves
    else:
        raise ValueError(f"unknown heat_model {model!r}")

    heat: Heatmap = [{} for _ in range(k)]
    for pos, goal in neighbours:
        dist = distances_to(grid, goal) if model == "goal" else None
        here = {tuple(pos): 1.0}
        for t in range(k):
            nxt: dict[Cell, float] = {}
            for cell, prob in here.items():
                for move, share in transition(grid, cell, dist, config):
                    nxt[move] = nxt.get(move, 0.0) + prob * share
            for cell, prob in nxt.items():
                heat[t][cell] = heat[t].get(cell, 0.0) + prob
            here = nxt
    return heat


def _symmetric_moves(grid: GridMap, cell: Cell, dist, config) -> list[tuple[Cell, float]]:
    """Every move (stay or a free neighbour) with equal probability."""
    moves = _moves(grid, cell)
    return [(move, 1.0 / len(moves)) for move in moves]


def _goal_moves(grid: GridMap, cell: Cell, dist: dict[Cell, int], config) -> list[tuple[Cell, float]]:
    """p over the moves closest to the goal, 1 - p over the rest, each split evenly."""
    moves = _moves(grid, cell)
    unreachable = grid.height * grid.width
    best = min(dist.get(move, unreachable) for move in moves)
    shortest = [move for move in moves if dist.get(move, unreachable) == best]
    others = [move for move in moves if dist.get(move, unreachable) != best]
    if not others:
        return [(move, 1.0 / len(shortest)) for move in shortest]
    p = config.heat_follow_prob
    return [(move, p / len(shortest)) for move in shortest] + [
        (move, (1.0 - p) / len(others)) for move in others
    ]


def heatmap_delay(
    grid: GridMap,
    pos: Cell,
    goal: Cell,
    neighbours: Sequence[tuple[Cell, Cell]],
    config,
) -> float:
    """d_i for the agent at ``pos`` heading to ``goal`` (see module docstring)."""
    pos, goal = tuple(pos), tuple(goal)
    k = config.delay_horizon
    heat = predict_heatmap(grid, sorted((tuple(p), tuple(g)) for p, g in neighbours), config)
    dist = distances_to(grid, goal)
    unreachable = grid.height * grid.width

    def step_cost(t: int, cell: Cell) -> float:
        return 1.0 + config.heat_weight * heat[t - 1].get(cell, 0.0)

    # value[cell] = best cost of steps t+1..k from ``cell`` plus remaining distance.
    value: dict[Cell, float] = {}
    reach = _cells_within(grid, pos, k)
    for cell in reach:
        value[cell] = float(dist.get(cell, unreachable))
    for t in range(k, 1, -1):  # fill steps k, k-1, ..., 2
        value = {
            cell: min(step_cost(t, nxt) + value[nxt] for nxt in _moves(grid, cell) if nxt in value)
            for cell in reach
        }

    def cost_if_first(cell: Cell) -> float:
        return step_cost(1, cell) + value[cell]

    best = min(_moves(grid, pos), key=lambda v: dist.get(v, unreachable))
    return max(config.min_delay, cost_if_first(pos) - cost_if_first(best))


# ------------------------------------------------------------------ helpers


def _moves(grid: GridMap, cell: Cell) -> list[Cell]:
    """[stay] + free neighbours in pypibt's order."""
    row, col = cell
    out = [cell]
    for d_row, d_col in _ORDER:
        nxt = (row + d_row, col + d_col)
        if grid.is_free(nxt):
            out.append(nxt)
    return out


def _cells_within(grid: GridMap, start: Cell, k: int) -> list[Cell]:
    """Free cells reachable from ``start`` in at most k moves."""
    seen = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if seen[cell] == k:
            continue
        for nxt in _moves(grid, cell)[1:]:
            if nxt not in seen:
                seen[nxt] = seen[cell] + 1
                queue.append(nxt)
    return list(seen)


_DIST_CACHE: dict[tuple[int, Cell], tuple[GridMap, dict[Cell, int]]] = {}


def distances_to(grid: GridMap, goal: Cell) -> dict[Cell, int]:
    """BFS distance from every reachable free cell to ``goal`` (cached per map and goal)."""
    goal = tuple(goal)
    hit = _DIST_CACHE.get((id(grid), goal))
    if hit is not None and hit[0] is grid:
        return hit[1]
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in grid.neighbours(cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    if len(_DIST_CACHE) > 4096:
        _DIST_CACHE.clear()
    _DIST_CACHE[(id(grid), goal)] = (grid, dist)
    return dist
