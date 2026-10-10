from conftest import instance_from_rows
from mapf_gt.metrics import evaluate, makespan, path_cost, sum_of_costs


def test_trailing_waits_at_goal_are_free():
    assert path_cost([(0, 0), (0, 1), (0, 1), (0, 1)], (0, 1)) == 1


def test_leaving_the_goal_and_returning_is_charged():
    assert path_cost([(0, 0), (0, 1), (0, 0), (0, 1)], (0, 1)) == 3


def test_agent_already_at_goal_costs_nothing():
    assert path_cost([(0, 1)], (0, 1)) == 0
    assert path_cost([(0, 1), (0, 1)], (0, 1)) == 0


def test_agent_that_never_arrives_pays_full_length():
    assert path_cost([(0, 0), (0, 0), (0, 0)], (0, 1)) == 2


def test_sum_of_costs_and_makespan():
    paths = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 1)]]
    goals = [(0, 2), (1, 1)]
    assert sum_of_costs(paths, goals) == 3
    assert makespan(paths, goals) == 2


def test_evaluate_valid_invalid_and_missing():
    instance = instance_from_rows(["...", "..."], [(0, 0), (1, 0)], [(0, 2), (1, 1)])
    good = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1)]]
    result = evaluate(instance, good, runtime_s=0.5)
    assert result.success
    assert (result.sum_of_costs, result.makespan, result.agents_at_goal) == (3, 2, 2)

    unfinished = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 0)]]
    result = evaluate(instance, unfinished, runtime_s=0.5)
    assert not result.success
    assert result.sum_of_costs is None
    assert result.agents_at_goal == 1
    assert result.problems

    assert not evaluate(instance, None, runtime_s=1.0).success
