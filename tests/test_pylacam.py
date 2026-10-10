from conftest import DATA, instance_from_rows
from mapf_gt.maps import load_instance
from mapf_gt.metrics import evaluate
from mapf_gt.solvers import CBS, PyLaCAM


def test_valid_and_never_better_than_optimal():
    for stem, k in (("toy-walls-8", 6), ("toy-empty-8", 8)):
        instance = load_instance(DATA / f"{stem}.map", DATA / f"{stem}.scen", k)
        optimal = evaluate(instance, CBS().solve(instance, 10).paths, 0.0)
        for refine in (False, True):
            solution = PyLaCAM(refine=refine).solve(instance, time_limit_s=1)
            assert solution is not None
            result = evaluate(instance, solution.paths, solution.runtime_s)
            assert result.success, result.problems
            assert result.sum_of_costs >= optimal.sum_of_costs


def test_easy_instance_is_solved_optimally():
    instance = instance_from_rows(
        ["@.@", "...", "@@@"], [(1, 0), (1, 2)], [(1, 2), (1, 0)]
    )
    solution = PyLaCAM().solve(instance, time_limit_s=2)
    assert evaluate(instance, solution.paths, 0.0).sum_of_costs == 7


def test_unsolvable_returns_none():
    instance = instance_from_rows(["..."], [(0, 0), (0, 2)], [(0, 2), (0, 0)])
    assert PyLaCAM().solve(instance, time_limit_s=1) is None
