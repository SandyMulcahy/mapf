from conftest import instance_from_rows
from mapf_gt.validate import find_problems, is_valid

ROWS = ["...", ".@.", "..."]


def two_agents():
    return instance_from_rows(ROWS, [(0, 0), (0, 2)], [(0, 2), (0, 0)])


def test_valid_solution_passes():
    instance = instance_from_rows(ROWS, [(0, 0), (2, 0)], [(0, 2), (2, 2)])
    paths = [[(0, 0), (0, 1), (0, 2)], [(2, 0), (2, 1), (2, 2)]]
    assert is_valid(instance, paths)


def test_vertex_conflict_is_found():
    instance = instance_from_rows(ROWS, [(0, 0), (0, 2)], [(0, 1), (1, 2)])
    paths = [[(0, 0), (0, 1)], [(0, 2), (0, 1), (0, 2), (1, 2)]]
    assert any("vertex conflict" in p for p in find_problems(instance, paths))


def test_swap_is_an_edge_conflict():
    instance = instance_from_rows(["..."], [(0, 0), (0, 1)], [(0, 1), (0, 0)])
    paths = [[(0, 0), (0, 1)], [(0, 1), (0, 0)]]
    assert any("edge conflict" in p for p in find_problems(instance, paths))


def test_following_is_allowed():
    instance = instance_from_rows(["...."], [(0, 0), (0, 1)], [(0, 1), (0, 2)])
    paths = [[(0, 0), (0, 1)], [(0, 1), (0, 2)]]
    assert is_valid(instance, paths)


def test_finished_agent_still_blocks_its_goal():
    # Agent 0 stops at (0, 1) after one step; agent 1 walks through it later.
    instance = instance_from_rows(["....", "...."], [(0, 0), (1, 1)], [(0, 1), (0, 2)])
    paths = [[(0, 0), (0, 1)], [(1, 1), (1, 1), (0, 1), (0, 2)]]
    assert any("vertex conflict" in p for p in find_problems(instance, paths))


def test_obstacle_jump_and_wrong_endpoints_are_found():
    instance = two_agents()
    through_wall = [[(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)], [(0, 2), (0, 1), (0, 0)]]
    assert any("not free" in p for p in find_problems(instance, through_wall))

    jump = [[(0, 0), (0, 2)], [(0, 2), (0, 1), (0, 0)]]
    assert any("jumps" in p for p in find_problems(instance, jump))

    wrong_goal = [[(0, 0), (0, 1)], [(0, 2), (1, 2)]]
    assert any("ends at" in p for p in find_problems(instance, wrong_goal))

    wrong_start = [[(0, 1), (0, 2)], [(0, 2), (0, 1), (0, 0)]]
    assert any("starts at" in p for p in find_problems(instance, wrong_start))


def test_wrong_number_of_paths():
    assert find_problems(two_agents(), [[(0, 0)]])


def test_goals_can_be_optional_for_unfinished_rollouts():
    instance = two_agents()
    paths = [[(0, 0), (1, 0)], [(0, 2), (1, 2)]]
    assert find_problems(instance, paths)
    assert not find_problems(instance, paths, require_goals=False)
