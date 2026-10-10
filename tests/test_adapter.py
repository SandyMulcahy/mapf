"""Tests that need POGEMA installed. They are skipped if it is missing."""

import pytest

pytest.importorskip("pogema")

from conftest import DATA, instance_from_rows  # noqa: E402
from mapf_gt.agents import Agent  # noqa: E402
from mapf_gt.maps import load_instance  # noqa: E402
from mapf_gt.maps.pogema_adapter import (  # noqa: E402
    action_for_move,
    agent_positions,
    make_env,
    move_table,
    to_grid_config,
)
from mapf_gt.rollout import replay_plan, run_agents  # noqa: E402
from mapf_gt.solvers import CBS  # noqa: E402
from mapf_gt.validate import find_problems  # noqa: E402

# Not square, obstacle off the diagonal: catches (row, col) mix-ups.
ROWS = [".....", "..@..", "....."]


def small_instance():
    return instance_from_rows(ROWS, [(0, 0), (2, 4)], [(0, 4), (2, 0)])


def test_grid_config_keeps_map_starts_and_goals():
    instance = small_instance()
    config = to_grid_config(instance, obs_radius=3)
    assert config.num_agents == 2
    assert config.obs_radius == 3
    assert [list(cell) for cell in instance.starts] == [list(c) for c in config.agents_xy]
    assert [list(cell) for cell in instance.goals] == [list(c) for c in config.targets_xy]
    assert config.map[1][2] == 1
    assert sum(sum(row) for row in config.map) == 1


def test_env_places_agents_at_the_starts():
    instance = small_instance()
    env = make_env(instance)
    env.reset()
    assert agent_positions(env) == list(instance.starts)


def test_every_action_moves_the_agent_as_the_move_table_says():
    instance = instance_from_rows(["...", "...", "..."], [(1, 1)], [(0, 0)])
    config = to_grid_config(instance)
    table = move_table(config)
    assert set(table) == {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    for (d_row, d_col), action in table.items():
        env = make_env(instance)
        env.reset()
        env.step([action])
        assert agent_positions(env) == [(1 + d_row, 1 + d_col)]
        assert action_for_move(config, (1, 1), (1 + d_row, 1 + d_col)) == action


def test_observation_is_a_local_window():
    instance = small_instance()
    env = make_env(instance, obs_radius=2)
    observations, _ = env.reset()
    assert len(observations) == instance.num_agents
    first = observations[0]
    assert first["obstacles"].shape == (5, 5)
    assert first["agents"].shape == (5, 5)


def test_replaying_a_cbs_plan_reproduces_it_exactly():
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 4)
    solution = CBS().solve(instance, 20.0)
    assert solution is not None
    executed = replay_plan(instance, solution.paths)
    horizon = max(len(path) for path in solution.paths)
    for planned, actual in zip(solution.paths, executed):
        padded = list(planned) + [planned[-1]] * (horizon - len(planned))
        assert actual == padded
    assert find_problems(instance, executed) == []


class _Waiter(Agent):
    def act(self, observation):
        return 0


def test_run_agents_records_one_position_per_step():
    instance = small_instance()
    rollout = run_agents(instance, _Waiter, max_episode_steps=5)
    assert not rollout.reached_goals
    assert rollout.steps == 5
    assert all(len(path) == 6 for path in rollout.paths)
    assert find_problems(instance, rollout.paths, require_goals=False) == []
