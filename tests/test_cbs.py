from conftest import DATA, instance_from_rows
from mapf_gt.maps import load_instance
from mapf_gt.metrics import sum_of_costs
from mapf_gt.solvers import CBS
from mapf_gt.solvers.cbs import distances_to
from mapf_gt.validate import find_problems


def solve(instance, time_limit_s=20.0):
    solution = CBS().solve(instance, time_limit_s)
    assert solution is not None
    assert find_problems(instance, solution.paths) == []
    return solution


def test_single_agent_takes_a_shortest_path():
    instance = instance_from_rows(["...", ".@.", "..."], [(0, 0)], [(2, 2)])
    solution = solve(instance)
    assert sum_of_costs(solution.paths, instance.goals) == 4


def test_independent_agents_cost_the_sum_of_shortest_paths():
    instance = instance_from_rows(
        ["....", "....", "...."], [(0, 0), (2, 0)], [(0, 3), (2, 3)]
    )
    solution = solve(instance)
    assert sum_of_costs(solution.paths, instance.goals) == 6


def test_swap_with_one_passing_bay_costs_seven():
    # A corridor with a single bay above its middle cell. The agents must swap
    # ends. Worked out by hand: one agent steps into the bay and back (4
    # moves), the other waits once and walks through (3), so the optimum is 7.
    instance = instance_from_rows(
        ["@.@", "...", "@@@"], [(1, 0), (1, 2)], [(1, 2), (1, 0)]
    )
    solution = solve(instance)
    assert sum_of_costs(solution.paths, instance.goals) == 7


def test_agent_resting_on_its_goal_must_move_aside():
    # Agent 0 starts on its goal in the middle of the corridor, and agent 1
    # has to get past. Agent 0 must step into the bay and come back.
    instance = instance_from_rows(
        ["@.@", "...", "@@@"], [(1, 1), (1, 0)], [(1, 1), (1, 2)]
    )
    solution = solve(instance)
    assert sum_of_costs(solution.paths, instance.goals) == 4


def test_unsolvable_swap_returns_none_at_the_time_limit():
    instance = instance_from_rows(["..."], [(0, 0), (0, 2)], [(0, 2), (0, 0)])
    assert CBS().solve(instance, time_limit_s=0.5) is None


def test_unreachable_goal_returns_none():
    instance = instance_from_rows([".@."], [(0, 0)], [(0, 2)])
    assert CBS().solve(instance, time_limit_s=1.0) is None


def test_cost_is_never_below_the_independent_lower_bound():
    for stem in ("toy-empty-8", "toy-walls-8"):
        for num_agents in (2, 4, 6):
            instance = load_instance(
                DATA / f"{stem}.map", DATA / f"{stem}.scen", num_agents
            )
            solution = solve(instance)
            bound = sum(
                distances_to(instance.grid, goal)[start]
                for start, goal in zip(instance.starts, instance.goals)
            )
            assert sum_of_costs(solution.paths, instance.goals) >= bound
