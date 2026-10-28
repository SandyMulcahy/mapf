"""The bidding protocol: decentralised moves must equal centralised PIBT.

Needs POGEMA; skipped if it is missing.
"""

import random

import pytest

pytest.importorskip("pogema")

from conftest import DATA, instance_from_rows  # noqa: E402
from mapf_gt.agents.bidding import draw_urgencies, make_bidding_agents  # noqa: E402
from mapf_gt.maps import Instance, load_instance, load_map  # noqa: E402
from mapf_gt.protocol import ProtocolConfig  # noqa: E402
from mapf_gt.rollout import run_protocol  # noqa: E402
from mapf_gt.validate import find_protocol_problems  # noqa: E402


def run(instance, seed=0, **config):
    urgencies = draw_urgencies(instance.num_agents, seed)
    return run_protocol(
        instance,
        make_bidding_agents(instance, urgencies),
        config=ProtocolConfig(**config),
        seed=seed,
        max_episode_steps=200,
    )


@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_scenarios_pass_every_protocol_check(k):
    instance = load_instance(DATA / "toy-empty-8.map", DATA / "toy-empty-8.scen", k)
    result = run(instance)
    assert result.stall is None
    assert find_protocol_problems(result) == []
    assert result.reached_goals


@pytest.mark.parametrize("seed", range(4))
def test_dense_random_instances_match_centralised_pibt(seed):
    # Dense enough for long push chains; reaching goals is not required here,
    # only that every tick equals centralised PIBT with the same priorities.
    grid = load_map(DATA / "toy-walls-8.map")
    free = [(r, c) for r in range(grid.height) for c in range(grid.width) if grid.is_free((r, c))]
    rng = random.Random(seed)
    instance = Instance(grid, tuple(rng.sample(free, 30)), tuple(rng.sample(free, 30)))
    result = run(instance, seed=seed, emergency_wait=6)
    assert result.stall is None
    assert find_protocol_problems(result) == []
    assert max(tick.max_push_chain for tick in result.ticks) > 1


def test_higher_bid_wins_the_contested_cell():
    # Both want the middle of the corridor; the higher urgency goes first.
    instance = instance_from_rows(["...", "@.@"], [(0, 0), (0, 2)], [(1, 1), (0, 0)])
    for urgencies, winner in (([9.0, 1.0], 0), ([1.0, 9.0], 1)):
        result = run_protocol(instance, make_bidding_agents(instance, urgencies), max_episode_steps=20)
        first = result.ticks[0]
        assert first.reserved[winner] == (0, 1)


def test_messages_stay_within_range():
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 6)
    result = run(instance, comm_range=2)
    for tick in result.ticks:
        for message in tick.messages:
            (r0, c0), (r1, c1) = tick.positions[message.sender], tick.positions[message.receiver]
            assert abs(r0 - r1) + abs(c0 - c1) <= 2


# ------------------------------------------------- truthful = urgency oracle


def oracle_paths(instance, urgencies, max_steps=200, **config):
    from mapf_gt.solvers import UrgencyPIBT

    alpha = config.get("alpha", 1.0)
    emergency = config.get("emergency_wait")
    return UrgencyPIBT(urgencies, alpha, emergency, max_timestep=max_steps).rollout(instance)


@pytest.mark.parametrize("map_name", ["toy-empty-8", "toy-walls-8"])
@pytest.mark.parametrize("k", [2, 4, 6, 8])
@pytest.mark.parametrize("config", [{}, {"alpha": 2.0, "emergency_wait": 5}])
def test_truthful_run_equals_urgency_pibt(map_name, k, config):
    # Truthful bids (d = 1) are the true urgencies, so the decentralised
    # protocol must move exactly like centralised pypibt with those
    # priorities: same paths, step for step, solved or not.
    instance = load_instance(DATA / f"{map_name}.map", DATA / f"{map_name}.scen", k)
    for seed in range(3):
        urgencies = draw_urgencies(k, seed)
        result = run_protocol(
            instance,
            make_bidding_agents(instance, urgencies),
            config=ProtocolConfig(**config),
            seed=seed,
            max_episode_steps=200,
        )
        assert result.stall is None
        assert result.paths == oracle_paths(instance, urgencies, 200, **config)


@pytest.mark.parametrize("seed", range(4))
def test_truthful_run_equals_urgency_pibt_when_dense(seed):
    grid = load_map(DATA / "toy-walls-8.map")
    free = [(r, c) for r in range(grid.height) for c in range(grid.width) if grid.is_free((r, c))]
    rng = random.Random(seed)
    instance = Instance(grid, tuple(rng.sample(free, 30)), tuple(rng.sample(free, 30)))
    urgencies = draw_urgencies(30, seed)
    result = run_protocol(
        instance,
        make_bidding_agents(instance, urgencies),
        config=ProtocolConfig(emergency_wait=6),
        seed=seed,
        max_episode_steps=100,
    )
    assert result.paths == oracle_paths(instance, urgencies, 100, emergency_wait=6)
