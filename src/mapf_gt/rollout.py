"""Run agents, or replay a centralised plan, inside POGEMA.

Both return trajectories in the same shape as a solver's paths, so
decentralised agents and centralised planners go through the same validator
and the same metrics.

Pass ``record=True`` to keep POGEMA's animation history, then hand the
returned environment to ``pogema_adapter.save_animation``.

Two ways to run agents:
  * ``run_agents``: baselines whose agents only implement ``act``.
  * ``run_protocol``: bidding agents driven by ``protocol.TickRunner``. It
    returns a ``RunResult``, the single record of a run that ``metrics``,
    ``validate`` and ``visualise`` read; nothing re-runs the protocol to
    measure or draw it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .agents.base import Agent
from .maps.movingai import Cell, Instance
from .maps.pogema_adapter import action_for_move, agent_positions, make_env
from .protocol.tick_runner import (
    ProtocolConfig,
    ProtocolStall,
    TickRecord,
    TickRunner,
)
from .validate import position_at

if TYPE_CHECKING:
    from .agents.bidding import BiddingAgent


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


# --------------------------------------------------------------- protocol


@dataclass(frozen=True)
class RunResult:
    """Everything recorded about one protocol run.

    ``ticks[k]`` describes the move from ``paths[i][k]`` to ``paths[i][k + 1]``.
    ``stall`` is set if the protocol raised ``ProtocolStall``; the run stops
    there and ``paths`` end at the last completed tick.
    """

    instance: Instance
    paths: list[list[Cell]]
    runtime_s: float  # time in phases 1, 2 and 4, summed over ticks
    steps: int
    reached_goals: bool
    seed: int
    urgencies: tuple[float, ...]
    strategies: tuple[str, ...]
    config: ProtocolConfig
    ticks: tuple[TickRecord, ...]
    stall: str | None = None
    env: Any = field(default=None, repr=False, compare=False)

    def as_rollout(self) -> Rollout:
        """The same run in the baseline shape, for ``metrics.evaluate``."""
        return Rollout(self.paths, self.runtime_s, self.steps, self.reached_goals, self.env)

    def total_payments(self) -> list[float]:
        totals = [0.0] * self.instance.num_agents
        for tick in self.ticks:
            for i, paid in enumerate(tick.payments):
                totals[i] += paid
        return totals

    def emergency_firings(self) -> int:
        """Agent-ticks spent in the emergency tier."""
        return sum(sum(tick.emergency) for tick in self.ticks)


def run_protocol(
    instance: Instance,
    make_agent: Callable[[int], "BiddingAgent"],
    *,
    config: ProtocolConfig | None = None,
    seed: int = 0,
    record: bool = False,
    **env_config: Any,
) -> RunResult:
    """Run bidding agents through ``TickRunner`` until all are at goal.

    ``make_agent(i)`` builds agent i with its urgency and strategy; draw
    urgencies from ``seed`` there so a run can be replayed exactly. ``seed``
    is also passed to POGEMA unless ``env_config`` sets its own.
    """
    config = config or ProtocolConfig()
    env_config.setdefault("seed", seed)
    env, observations = _start(instance, record, env_config)
    agents = [make_agent(i) for i in range(instance.num_agents)]
    for agent, observation in zip(agents, observations):
        agent.reset(observation)
    runner = TickRunner(env, agents, instance, config)

    goals = list(instance.goals)
    paths = [[cell] for cell in agent_positions(env)]
    ticks: list[TickRecord] = []
    stall: str | None = None
    while [path[-1] for path in paths] != goals:
        try:
            outcome = runner.run_tick(observations)
        except ProtocolStall as exc:
            stall = str(exc)
            break
        observations = outcome.observations
        ticks.append(outcome.record)
        for path, cell in zip(paths, agent_positions(env)):
            path.append(cell)
        if all(outcome.terminated) or all(outcome.truncated):
            break

    return RunResult(
        instance=instance,
        paths=paths,
        runtime_s=sum(tick.decide_s for tick in ticks),
        steps=len(ticks),
        reached_goals=[path[-1] for path in paths] == goals,
        seed=seed,
        urgencies=tuple(float(agent.urgency) for agent in agents),
        strategies=tuple(agent.strategy for agent in agents),
        config=config,
        ticks=tuple(ticks),
        stall=stall,
        env=env,
    )
