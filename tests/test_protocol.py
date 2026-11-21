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


# ------------------------------------------- priority rules (protocol/priority.py)

RULE_A = {"priority_rule": "additive", "age_rule": "since_progress", "emergency_wait": 5}
RULE_D = {"priority_rule": "additive", "age_rule": "since_progress",
          "emergency_wait": 20, "emergency_counter": "since_goal"}


def test_default_rule_is_urgency_weighted_ageing():
    from mapf_gt.protocol.priority import priority

    config = ProtocolConfig(alpha=0.5)
    assert priority(4.0, 1.0, 6, 6, 3, config) == (0, 4.0 * (1.0 + 0.5 * 6), 3)
    additive = ProtocolConfig(alpha=0.5, priority_rule="additive")
    assert priority(4.0, 1.0, 6, 6, 3, additive) == (0, 4.0 * 1.0 + 0.5 * 6, 3)


def test_emergency_tier_can_count_ticks_since_goal():
    from mapf_gt.protocol.priority import priority

    config = ProtocolConfig(emergency_wait=10, emergency_counter="since_goal")
    assert priority(1.0, 1.0, 0, 10, 0, config)[0] == 1  # age reset, but stuck 10 ticks
    assert priority(1.0, 1.0, 10, 9, 0, config)[0] == 0


def test_since_progress_age_resets_when_closer():
    grid = load_map(DATA / "toy-empty-8.map")
    agent = make_bidding_agents(Instance(grid, ((0, 0),), ((0, 2),)), [5.0])(0)
    agent.reset({"xy": (0, 0), "target_xy": (0, 2)})
    agent._config = ProtocolConfig(age_rule="since_progress")
    agent.end_tick({"xy": (0, 0)}, 0.0)  # waited
    assert (agent.eps, agent.since_goal) == (1, 1)
    agent.end_tick({"xy": (0, 1)}, 0.0)  # closer: age resets, since_goal does not
    assert (agent.eps, agent.since_goal) == (0, 2)


@pytest.mark.parametrize("rule", [RULE_A, RULE_D], ids=["A", "D"])
@pytest.mark.parametrize("k", [4, 8])
def test_whole_map_range_equals_urgency_pibt_for_other_rules(rule, k):
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", k)
    for seed in range(3):
        urgencies = draw_urgencies(k, seed)
        result = run_protocol(
            instance,
            make_bidding_agents(instance, urgencies),
            config=ProtocolConfig(comm_range=14, **rule),
            seed=seed,
            max_episode_steps=200,
        )
        assert result.paths == oracle_paths(instance, urgencies, 200, **rule)


# ------------------------------------------ heatmap models (predict_heatmap)

HEAT_MODELS = ["symmetric", "goal"]


def _heat(model, neighbours, map_name="toy-walls-8", **config):
    from mapf_gt.protocol.heatmap_delay import predict_heatmap

    grid = load_map(DATA / f"{map_name}.map")
    return grid, predict_heatmap(grid, neighbours, ProtocolConfig(heat_model=model, **config))


def test_no_heat_model_predicts_nothing():
    _, heat = _heat("none", [((0, 0), (7, 7))])
    assert heat == [{}, {}, {}]


@pytest.mark.parametrize("model", HEAT_MODELS)
def test_heatmap_has_k_steps_of_nonnegative_heat_on_free_cells(model):
    neighbours = [((0, 0), (7, 7)), ((2, 2), (0, 3)), ((4, 4), (4, 4))]
    grid, heat = _heat(model, neighbours, delay_horizon=5)
    assert len(heat) == 5
    for step in heat:
        assert all(grid.is_free(cell) and value >= 0 for cell, value in step.items())
        assert sum(step.values()) == pytest.approx(len(neighbours))  # mass is conserved


@pytest.mark.parametrize("model", HEAT_MODELS)
def test_step_one_heat_is_within_one_cell_of_a_neighbour(model):
    neighbours = [((0, 0), (7, 7)), ((4, 4), (0, 0))]
    _, heat = _heat(model, neighbours)
    for (row, col) in heat[0]:
        assert min(abs(row - r) + abs(col - c) for (r, c), _ in neighbours) <= 1


def test_symmetric_heat_ignores_goals():
    _, a = _heat("symmetric", [((3, 3), (0, 0))])
    _, b = _heat("symmetric", [((3, 3), (7, 7))])
    assert a == b
    assert a[0] == {cell: pytest.approx(0.2) for cell in [(3, 3), (3, 2), (3, 4), (2, 3), (4, 3)]}


def test_goal_heat_follows_the_shortest_path_with_probability_p():
    # Empty map, goal straight to the right: one shortest move (right) gets p,
    # the other four moves (stay, left, up, down) share 1 - p.
    _, heat = _heat("goal", [((3, 3), (3, 6))], map_name="toy-empty-8", heat_follow_prob=0.8)
    assert heat[0][(3, 4)] == pytest.approx(0.8)
    for cell in [(3, 3), (3, 2), (2, 3), (4, 3)]:
        assert heat[0][cell] == pytest.approx(0.05)


def test_goal_heat_splits_p_over_tied_shortest_moves():
    # Goal diagonal: right and down are equally short, so each gets p / 2.
    _, heat = _heat("goal", [((3, 3), (5, 5))], map_name="toy-empty-8", heat_follow_prob=0.6)
    assert heat[0][(3, 4)] == pytest.approx(0.3)
    assert heat[0][(4, 3)] == pytest.approx(0.3)
    assert heat[0][(3, 3)] == pytest.approx(0.4 / 3)


def test_goal_heat_with_uniform_p_equals_symmetric_off_goal():
    # p = 1/5 with one shortest move out of five gives every move 1/5.
    neighbours = [((3, 3), (3, 6))]
    _, goal = _heat("goal", neighbours, map_name="toy-empty-8", delay_horizon=1, heat_follow_prob=0.2)
    _, symmetric = _heat("symmetric", neighbours, map_name="toy-empty-8", delay_horizon=1)
    assert goal[0] == pytest.approx(symmetric[0])


@pytest.mark.parametrize("model", HEAT_MODELS)
def test_heat_on_the_route_raises_the_delay_and_ignores_urgency(model):
    from mapf_gt.protocol.heatmap_delay import heatmap_delay

    grid = load_map(DATA / "toy-empty-8.map")
    config = ProtocolConfig(heat_model=model)
    calm = heatmap_delay(grid, (3, 0), (3, 7), [], config)
    # A neighbour ahead heading towards us changes the delay; the inputs hold no urgency.
    busy = heatmap_delay(grid, (3, 0), (3, 7), [((3, 2), (3, 0))], config)
    assert busy != pytest.approx(calm)


@pytest.mark.parametrize("model", HEAT_MODELS)
def test_heat_models_run_cleanly_and_match_the_oracle(model):
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 6)
    for seed in range(2):
        urgencies = draw_urgencies(6, seed)
        result = run_protocol(
            instance,
            make_bidding_agents(instance, urgencies),
            config=ProtocolConfig(comm_range=14, heat_model=model),
            seed=seed,
            max_episode_steps=200,
        )
        assert find_protocol_problems(result) == []
        assert result.paths == oracle_paths(instance, urgencies, 200, heat_model=model)


# ----------------------------------------- random tie-break (protocol/tiebreak.py)


def test_tiebreak_is_pypibt_order_unless_switched_on():
    from mapf_gt.protocol.tiebreak import tiebreak_permutation

    assert tiebreak_permutation(5, 3, 1, ProtocolConfig()) == [0, 1, 2, 3, 4]
    on = ProtocolConfig(random_tiebreak=True)
    draw = tiebreak_permutation(5, 3, 1, on)
    assert sorted(draw) == [0, 1, 2, 3, 4]
    assert tiebreak_permutation(5, 3, 1, on) == draw  # same key, same draw
    others = {
        tuple(tiebreak_permutation(5, t, i, ProtocolConfig(random_tiebreak=True, tiebreak_seed=s)))
        for t in range(4) for i in range(4) for s in range(2)
    }
    assert len(others) > 1  # seed, tick and agent all feed the key


def test_random_tiebreak_only_reorders_equally_short_cells():
    from mapf_gt.protocol.tiebreak import order_candidates

    grid = load_map(DATA / "toy-empty-8.map")
    from mapf_gt.agents.bidding import pibt_candidates

    # Goal diagonal from (3, 3): right and down tie at distance 3, the rest are 5.
    dist = lambda v: abs(v[0] - 5) + abs(v[1] - 5)  # noqa: E731
    firsts = set()
    for t in range(20):
        config = ProtocolConfig(random_tiebreak=True)
        order = order_candidates(pibt_candidates(grid, (3, 3)), dist, t, 0, config)
        assert [dist(v) for v in order] == sorted(dist(v) for v in order)
        firsts.add(order[0])
    assert firsts == {(3, 4), (4, 3)}


@pytest.mark.parametrize("map_name", ["toy-empty-8", "toy-walls-8"])
@pytest.mark.parametrize("tiebreak_seed", [0, 1, 2])
def test_random_tiebreak_matches_the_oracle_and_reference_pibt(map_name, tiebreak_seed):
    rule = {"random_tiebreak": True, "tiebreak_seed": tiebreak_seed}
    instance = load_instance(DATA / f"{map_name}.map", DATA / f"{map_name}.scen", 8)
    urgencies = draw_urgencies(8, tiebreak_seed)
    result = run_protocol(
        instance,
        make_bidding_agents(instance, urgencies),
        config=ProtocolConfig(comm_range=14, **rule),
        seed=tiebreak_seed,
        max_episode_steps=200,
    )
    assert find_protocol_problems(result, check_pibt=True) == []
    assert result.paths == oracle_paths(instance, urgencies, 200, **rule)


def test_random_tiebreak_changes_some_paths():
    instance = load_instance(DATA / "toy-empty-8.map", DATA / "toy-empty-8.scen", 8)
    urgencies = draw_urgencies(8, 0)
    fixed = oracle_paths(instance, urgencies, 200)
    shuffled = [oracle_paths(instance, urgencies, 200, random_tiebreak=True, tiebreak_seed=s) for s in range(3)]
    assert any(paths != fixed for paths in shuffled)


# ------------------------------------- keskin radius-2 kernel, sliding along the path


def _peak(step):
    return max(step, key=step.get)


def test_keskin_kernel_slides_one_step_per_tick_with_open_space_values():
    _, heat = _heat("keskin", [((3, 3), (3, 7))], map_name="toy-empty-8", delay_horizon=3)
    z = 1 + 4 * (2 / 3) + 8 * (1 / 3)  # 19/3 in open space
    assert [_peak(step) for step in heat] == [(3, 4), (3, 5), (3, 6)]
    assert heat[0][(3, 4)] == pytest.approx(1 / z)
    assert heat[0][(3, 5)] == pytest.approx((2 / 3) / z)
    assert heat[0][(4, 5)] == pytest.approx((1 / 3) / z)
    assert (3, 7) not in heat[0]  # 3 moves from the step-1 centre
    for step in heat:
        assert sum(step.values()) == pytest.approx(1.0)  # 1 per neighbour per step


def test_keskin_centre_breaks_ties_in_pypibt_order_and_stops_at_goal():
    # Goal diagonal: right and down tie, pypibt order (left, right, up, down) takes right.
    _, heat = _heat("keskin", [((3, 3), (5, 5))], map_name="toy-empty-8")
    assert [_peak(step) for step in heat] == [(3, 4), (3, 5), (4, 5)]
    # One move from the goal: the centre reaches it at t = 1 and stays.
    _, near = _heat("keskin", [((3, 3), (3, 4))], map_name="toy-empty-8")
    assert [_peak(step) for step in near] == [(3, 4)] * 3
    assert near[0] == near[1] == near[2]


def test_keskin_uses_goals_and_concentrates_near_walls():
    _, a = _heat("keskin", [((3, 3), (0, 0))], map_name="toy-empty-8")
    _, b = _heat("keskin", [((3, 3), (7, 7))], map_name="toy-empty-8")
    assert a != b
    _, corner = _heat("keskin", [((0, 0), (0, 0))], map_name="toy-empty-8")
    _, middle = _heat("keskin", [((3, 3), (3, 3))], map_name="toy-empty-8")
    assert sum(corner[0].values()) == pytest.approx(1.0)
    assert corner[0][(0, 0)] > middle[0][(3, 3)]  # fewer cells, so more heat on each


def test_keskin_distance_is_through_free_cells():
    # toy-walls-8 row 1 is ".@@..@@.": (0, 1) and (2, 1) are 2 apart in Manhattan
    # distance but 4 moves apart round the wall at (1, 1), so no heat crosses it.
    grid, heat = _heat("keskin", [((0, 1), (0, 1))])  # on its goal: centre stays
    assert not grid.is_free((1, 1))
    assert (2, 1) not in heat[0]
    assert all(grid.is_free(cell) for step in heat for cell in step)


def test_keskin_heat_model_runs_cleanly_and_matches_the_oracle():
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 6)
    for seed in range(2):
        urgencies = draw_urgencies(6, seed)
        result = run_protocol(
            instance,
            make_bidding_agents(instance, urgencies),
            config=ProtocolConfig(comm_range=14, heat_model="keskin"),
            seed=seed,
            max_episode_steps=200,
        )
        assert find_protocol_problems(result) == []
        assert result.paths == oracle_paths(instance, urgencies, 200, heat_model="keskin")


# ------------------------------------------------ deceptive bidders (multipliers)


def test_draw_multipliers_picks_a_seeded_fraction_of_liars():
    from mapf_gt.agents.bidding import draw_multipliers

    assert draw_multipliers(8, 0.0, 4.0, 1) == [1.0] * 8
    assert draw_multipliers(8, 1.0, 4.0, 1) == [4.0] * 8
    drawn = draw_multipliers(8, 0.25, 4.0, 1)
    assert sorted(drawn) == [1.0] * 6 + [4.0] * 2
    assert draw_multipliers(8, 0.25, 4.0, 1) == drawn  # same seed, same liars
    assert len({tuple(draw_multipliers(8, 0.25, 4.0, s)) for s in range(10)}) > 1
    with pytest.raises(ValueError):
        draw_multipliers(8, 1.5, 4.0, 1)


def test_liars_bid_their_multiplier_times_urgency_and_run_cleanly():
    from mapf_gt.agents.bidding import draw_multipliers

    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 6)
    urgencies = draw_urgencies(6, 2)
    multipliers = draw_multipliers(6, 0.5, 4.0, 2)
    result = run_protocol(
        instance,
        make_bidding_agents(instance, urgencies, multipliers),
        config=ProtocolConfig(),
        seed=2,
        max_episode_steps=200,
    )
    assert find_protocol_problems(result) == []
    for tick in result.ticks:
        assert list(tick.bids) == pytest.approx([m * c for m, c in zip(multipliers, urgencies)])


# --------------------------------------- prefer-free tie-break (PIBT, protocol/tiebreak.py)


def test_prefer_free_puts_unoccupied_cells_first_among_ties_only():
    from mapf_gt.agents.bidding import pibt_candidates
    from mapf_gt.protocol.tiebreak import order_candidates

    grid = load_map(DATA / "toy-empty-8.map")
    dist = lambda v: abs(v[0] - 5) + abs(v[1] - 5)  # noqa: E731  goal (5, 5): right and down tie
    cells = pibt_candidates(grid, (3, 3))
    plain = order_candidates(cells, dist, 0, 0, ProtocolConfig(), occupied={(3, 4)})
    assert plain[0] == (3, 4)  # pypibt order: right before down
    free = order_candidates(cells, dist, 0, 0, ProtocolConfig(tiebreak_prefer_free=True), occupied={(3, 4)})
    assert free[:2] == [(4, 3), (3, 4)]  # occupied right moves behind free down
    # A strictly shorter cell still comes first even if it is occupied.
    near = lambda v: abs(v[0] - 3) + abs(v[1] - 5)  # noqa: E731  goal (3, 5): only right is shortest
    assert order_candidates(cells, near, 0, 0, ProtocolConfig(tiebreak_prefer_free=True), occupied={(3, 4)})[0] == (3, 4)


@pytest.mark.parametrize("map_name", ["toy-empty-8", "toy-walls-8"])
@pytest.mark.parametrize("random_tiebreak", [False, True])
def test_prefer_free_matches_the_oracle_and_reference_pibt(map_name, random_tiebreak):
    rule = {"tiebreak_prefer_free": True, "random_tiebreak": random_tiebreak, "tiebreak_seed": 1}
    instance = load_instance(DATA / f"{map_name}.map", DATA / f"{map_name}.scen", 8)
    for seed in range(2):
        urgencies = draw_urgencies(8, seed)
        result = run_protocol(
            instance,
            make_bidding_agents(instance, urgencies),
            config=ProtocolConfig(comm_range=14, **rule),
            seed=seed,
            max_episode_steps=200,
        )
        assert find_protocol_problems(result, check_pibt=True) == []
        assert result.paths == oracle_paths(instance, urgencies, 200, **rule)


def test_prefer_free_changes_some_paths():
    instance = load_instance(DATA / "toy-empty-8.map", DATA / "toy-empty-8.scen", 8)
    changed = False
    for seed in range(4):
        urgencies = draw_urgencies(8, seed)
        changed |= oracle_paths(instance, urgencies, 200) != oracle_paths(
            instance, urgencies, 200, tiebreak_prefer_free=True
        )
    assert changed
