"""Run agents, or replay a centralised plan, inside POGEMA.

Both return trajectories in the same shape as a solver's paths, so
decentralised agents and centralised planners go through the same validator
and the same metrics.

Pass ``record=True`` to keep POGEMA's animation history, then hand the
returned environment to ``pogema_adapter.save_animation``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .agents.base import Agent
from .maps.movingai import Cell, Instance
from .maps.pogema_adapter import action_for_move, agent_positions, make_env
from .validate import position_at


@dataclass(frozen=True)
class Rollout:
    paths: list[list[Cell]]  # paths[agent][t] = (row, col), as executed
    runtime_s: float  # total time spent inside agents' act()
    steps: int
    reached_goals: bool
    env: Any = field(default=None, repr=False, compare=False)


def _start(instance: Instance, record: bool, env_config: dict):
    env = make_env(instance, **env_config)
    if record:
        env.enable_animation()  # must come before reset()
    observations, _ = env.reset()
    return env, observations


def run_agents(
    instance: Instance,
    make_agent: Callable[[int], Agent],
    record: bool = False,
    **env_config: Any,
) -> Rollout:
    """Step POGEMA with one agent object per robot until all are at goal.

    Stops when every agent is on its goal at the same time, or when POGEMA
    ends the episode (``max_episode_steps``).
    """
    env, observations = _start(instance, record, env_config)
    agents = [make_agent(i) for i in range(instance.num_agents)]
    for agent, observation in zip(agents, observations):
        agent.reset(observation)

    goals = list(instance.goals)
    paths = [[cell] for cell in agent_positions(env)]
    deciding = 0.0
    steps = 0
    while [path[-1] for path in paths] != goals:
        started = time.perf_counter()
        actions = [agent.act(obs) for agent, obs in zip(agents, observations)]
        deciding += time.perf_counter() - started

        observations, _, terminated, truncated, _ = env.step(actions)
        steps += 1
        for path, cell in zip(paths, agent_positions(env)):
            path.append(cell)
        if all(terminated) or all(truncated):
            break

    reached = [path[-1] for path in paths] == goals
    return Rollout(paths, deciding, steps, reached, env)


def execute_plan(
    instance: Instance,
    paths: Sequence[Sequence[Cell]],
    record: bool = False,
    **env_config: Any,
) -> tuple[list[list[Cell]], Any]:
    """Execute a centralised plan in POGEMA; return (what happened, env)."""
    horizon = max(len(path) for path in paths)
    env_config.setdefault("max_episode_steps", max(horizon, 1))
    env, _ = _start(instance, record, env_config)
    config = env.unwrapped.grid_config

    executed = [[cell] for cell in agent_positions(env)]
    for t in range(1, horizon):
        actions = [
            action_for_move(config, position_at(path, t - 1), position_at(path, t))
            for path in paths
        ]
        env.step(actions)
        for path, cell in zip(executed, agent_positions(env)):
            path.append(cell)
    return executed, env


def replay_plan(
    instance: Instance, paths: Sequence[Sequence[Cell]], **env_config: Any
) -> list[list[Cell]]:
    """Execute a centralised plan in POGEMA and return what actually happened.

    If the returned trajectories differ from ``paths``, the simulator and the
    planner disagree about the movement rules, and the comparison is not fair
    until that is resolved.
    """
    executed, _ = execute_plan(instance, paths, **env_config)
    return executed


def first_divergence(
    planned: Sequence[Sequence[Cell]], executed: Sequence[Sequence[Cell]]
) -> tuple[int, int, Cell, Cell] | None:
    """First (agent, t, planned cell, executed cell) where the two differ."""
    horizon = max(len(path) for path in [*planned, *executed])
    for t in range(horizon):
        for agent, (plan, actual) in enumerate(zip(planned, executed)):
            want, got = position_at(plan, t), position_at(actual, t)
            if tuple(want) != tuple(got):
                return agent, t, tuple(want), tuple(got)
    return None
