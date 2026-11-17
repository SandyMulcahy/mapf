"""The bidding protocol: event-driven decentralised PIBT.

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
def test_dense_random_instances_are_safe(seed):
    # Dense enough for long, concurrent push chains; reaching goals is not
    # required here, only that every protocol check passes.
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


# ---------------------------------------------- event-driven waiting rule


@pytest.mark.parametrize("seed", range(3))
def test_agents_wait_for_every_higher_neighbour_unless_pushed(seed):
    # Checked from the message log alone: before an agent sends anything in
    # the move phase, it has either been pushed or heard Done from every
    # neighbour that outranks it.
    grid = load_map(DATA / "toy-walls-8.map")
    free = [(r, c) for r in range(grid.height) for c in range(grid.width) if grid.is_free((r, c))]
    rng = random.Random(seed)
    instance = Instance(grid, tuple(rng.sample(free, 20)), tuple(rng.sample(free, 20)))
    result = run(instance, seed=seed)
    r = result.config.comm_range
    for tick in result.ticks:
        pos, prio = tick.positions, tick.priorities
        n = len(pos)
        higher = {
            i: {
                j for j in range(n)
                if j != i
                and abs(pos[i][0] - pos[j][0]) + abs(pos[i][1] - pos[j][1]) <= r
                and prio[j] > prio[i]
            }
            for i in range(n)
        }
        heard = {i: set() for i in range(n)}
        pushed = set()
        acted = set()
        for message in tick.messages:
            if message.kind == "bid":
                continue
            sender, receiver = message.sender, message.receiver
            if sender not in acted and sender not in pushed:
                assert higher[sender] <= heard[sender], (tick.t, sender)
            acted.add(sender)
            if message.kind == "done":
                heard[receiver].add(sender)
            elif message.kind == "push":
                pushed.add(receiver)


# ------------------------------------------- reduces to centralised PIBT


def oracle_paths(instance, urgencies, max_steps=200, **config):
    from mapf_gt.solvers import UrgencyPIBT

    config = ProtocolConfig(**{"comm_range": 14, **config})
    return UrgencyPIBT(urgencies, config, max_timestep=max_steps).rollout(instance)


@pytest.mark.parametrize("map_name", ["toy-empty-8", "toy-walls-8"])
@pytest.mark.parametrize("k", [2, 4, 6, 8])
@pytest.mark.parametrize("config", [{}, {"alpha": 2.0, "emergency_wait": 5}])
def test_whole_map_range_equals_urgency_pibt(map_name, k, config):
    # With a range covering the whole 8x8 map every agent waits for every
    # higher-priority agent, so chains run one at a time in priority order:
    # truthful runs must then move exactly like centralised pypibt with the
    # true urgencies, step for step, solved or not.
    instance = load_instance(DATA / f"{map_name}.map", DATA / f"{map_name}.scen", k)
    for seed in range(3):
        urgencies = draw_urgencies(k, seed)
        result = run_protocol(
            instance,
            make_bidding_agents(instance, urgencies),
            config=ProtocolConfig(comm_range=14, **config),
            seed=seed,
            max_episode_steps=200,
        )
        assert result.stall is None
        assert result.paths == oracle_paths(instance, urgencies, 200, **config)


@pytest.mark.parametrize("seed", range(4))
def test_whole_map_range_equals_urgency_pibt_when_dense(seed):
    grid = load_map(DATA / "toy-walls-8.map")
    free = [(r, c) for r in range(grid.height) for c in range(grid.width) if grid.is_free((r, c))]
    rng = random.Random(seed)
    instance = Instance(grid, tuple(rng.sample(free, 30)), tuple(rng.sample(free, 30)))
    urgencies = draw_urgencies(30, seed)
    result = run_protocol(
        instance,
        make_bidding_agents(instance, urgencies),
        config=ProtocolConfig(comm_range=14, emergency_wait=6),
        seed=seed,
        max_episode_steps=100,
    )
    assert result.paths == oracle_paths(instance, urgencies, 100, emergency_wait=6)


@pytest.mark.parametrize("seed", range(3))
def test_divergence_from_centralised_pibt_is_measured(seed):
    from mapf_gt.validate import pibt_divergence

    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 8)
    result = run(instance, seed=seed)
    differ, total = pibt_divergence(result)
    assert total == len(result.ticks) * instance.num_agents
    assert 0 <= differ <= total


# ------------------------------------------------- delay and eps (Option 1)


def test_delay_is_one_on_an_open_path_and_minimal_at_goal():
    from mapf_gt.protocol.heatmap_delay import heatmap_delay

    grid = load_map(DATA / "toy-empty-8.map")
    config = ProtocolConfig()
    assert heatmap_delay(grid, (0, 0), (0, 5), [], config) == pytest.approx(1.0)
    assert heatmap_delay(grid, (0, 5), (0, 5), [], config) == config.min_delay


def test_delay_grows_when_waiting_is_congested(monkeypatch):
    from mapf_gt.protocol import heatmap_delay as module

    grid = load_map(DATA / "toy-empty-8.map")
    config = ProtocolConfig()
    calm = module.heatmap_delay(grid, (0, 0), (0, 5), [], config)
    # Heat on the agent's own cell at step 1 makes staying, i.e. losing, costlier.
    monkeypatch.setattr(
        module, "predict_heatmap", lambda g, n, c: [{(0, 0): 2.0}] + [{} for _ in range(c.delay_horizon - 1)]
    )
    assert module.heatmap_delay(grid, (0, 0), (0, 5), [], config) == pytest.approx(calm + 2.0)


def test_eps_counts_ticks_since_last_at_goal():
    grid = load_map(DATA / "toy-empty-8.map")
    agent = make_bidding_agents(
        Instance(grid, ((0, 0),), ((0, 2),)), [5.0]
    )(0)
    agent.reset({"xy": (0, 0), "target_xy": (0, 2)})
    agent.end_tick({"xy": (0, 0)}, 0.0)  # waited
    assert agent.eps == 1
    agent.end_tick({"xy": (0, 1)}, 0.0)  # closer, but not there: still ageing
    assert agent.eps == 2
    agent.end_tick({"xy": (0, 2)}, 0.0)  # at goal
    assert agent.eps == 0
    agent.end_tick({"xy": (0, 2)}, 0.0)  # waiting at goal
    assert agent.eps == 0
    agent.end_tick({"xy": (1, 2)}, 0.0)  # pushed off goal
    assert agent.eps == 1


def test_truthful_report_is_the_urgency_and_priority_uses_the_delay():
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 4)
    urgencies = draw_urgencies(4, 0)
    result = run(instance)
    first = result.ticks[0]
    assert first.bids == pytest.approx(tuple(urgencies))
    for i in range(4):
        tier, value, agent_id = first.priorities[i]
        assert agent_id == i
        assert value == pytest.approx(first.bids[i] * first.delays[i])  # eps = 0, so b * (d + 0)
