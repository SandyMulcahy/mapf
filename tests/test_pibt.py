import numpy as np

from conftest import DATA, instance_from_rows
from mapf_gt.maps import load_instance
from mapf_gt.metrics import evaluate
from mapf_gt.solvers import CBS, PIBT
from mapf_gt.solvers._pypibt import PIBT as RawPIBT


def test_valid_and_never_better_than_optimal():
    for stem, k in (("toy-walls-8", 6), ("toy-empty-8", 8)):
        instance = load_instance(DATA / f"{stem}.map", DATA / f"{stem}.scen", k)
        optimal = evaluate(instance, CBS().solve(instance, 10).paths, 0.0)
        solution = PIBT().solve(instance)
        assert solution is not None
        result = evaluate(instance, solution.paths, solution.runtime_s)
        assert result.success, result.problems
        assert result.sum_of_costs >= optimal.sum_of_costs


def test_gets_stuck_where_lacam_does_not():
    # Known PIBT limitation, kept as a test so it is never forgotten.
    # The agents must swap ends of a corridor using the bay above the middle.
    # PIBT only looks one step ahead: agent 0 reaches the middle, agent 1
    # sits in the dead end, and neither ever steps back into the bay. Its
    # guarantee only holds on biconnected graphs, and this one isn't.
    # LaCAM solves the same instance (see test_pylacam.py).
    instance = instance_from_rows(
        ["@.@", "...", "@@@"], [(1, 0), (1, 2)], [(1, 2), (1, 0)]
    )
    for seed in range(5):
        assert PIBT(seed=seed, max_timestep=100).solve(instance) is None


def test_same_seed_gives_same_paths():
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 8)
    first = PIBT(seed=3).solve(instance)
    second = PIBT(seed=3).solve(instance)
    assert first.paths == second.paths


def test_unsolvable_returns_none_after_max_timestep():
    # A corridor with no room to pass: PIBT never finishes, so it must give up.
    instance = instance_from_rows(["..."], [(0, 0), (0, 2)], [(0, 2), (0, 0)])
    assert PIBT(max_timestep=50).solve(instance) is None


def test_higher_priority_wins_the_contested_cell():
    # Agents at either end of a corridor both want the middle-left cell (0, 1)
    # next step. Whichever has the higher priority gets it; the other waits.
    # This is the hook a bid-based priority will plug into.
    free = np.ones((1, 4), dtype=bool)
    starts = [(0, 0), (0, 2)]
    goals = [(0, 3), (0, 0)]
    for winner, priorities in ((0, [1.0, 0.0]), (1, [0.0, 1.0])):
        planner = RawPIBT(free, list(starts), list(goals), seed=0)
        nxt = planner.step(list(starts), priorities)
        assert nxt[winner] == (0, 1)
        assert nxt[1 - winner] == starts[1 - winner]
