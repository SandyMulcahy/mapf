"""Run the bidding protocol on one instance, validate it and (optionally) watch it.

    python experiments/run_protocol.py \\
        --map data/small/toy-walls-8.map --scen data/small/toy-walls-8.scen -k 6

Agents are truthful unless --liars is given: then that fraction of agents,
chosen from --seed, bid --multiplier times their urgency. Urgencies and liars
are drawn from --seed, so a run can be replayed exactly. Payments are
currently placeholders (all zero), so lying costs nothing yet.

    python experiments/run_protocol.py --map ... --scen ... -k 40 --seed 1 \\
        --rule E --heat-model goal --heat-p 0.8 --liars 0.25 --multiplier 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compare_rules import RULES  # rules A, D, E (same directory)

from mapf_gt.agents.bidding import draw_multipliers, draw_urgencies, make_bidding_agents
from mapf_gt.maps import load_instance
from mapf_gt.metrics import evaluate, path_cost
from mapf_gt.protocol.heatmap_delay import distances_to
from mapf_gt.protocol import ProtocolConfig
from mapf_gt.rollout import run_protocol
from mapf_gt.validate import find_protocol_problems, pibt_divergence


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--scen", type=Path, required=True)
    parser.add_argument("-k", "--agents", type=int, default=4, help="number of agents")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--range", type=int, default=2, dest="comm_range",
                        help="communication range (Manhattan); needs >= 2")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight on waiting time")
    parser.add_argument("--rule", default="E", choices=list(RULES),
                        help="priority/ageing rule (see compare_rules.py); default E")
    parser.add_argument("--emergency", type=int, default=None, metavar="W",
                        help="wait after which an agent enters the emergency tier "
                             "(overrides the rule's own setting)")
    parser.add_argument("--liars", type=float, default=0.0, metavar="FRACTION",
                        help="fraction of agents that bid --multiplier x their urgency")
    parser.add_argument("--multiplier", type=float, default=4.0,
                        help="bid multiplier used by the liars (default 4)")
    parser.add_argument("--random-tiebreak", action="store_true",
                        help="random tie-break between equally short cells, keyed on --seed")
    parser.add_argument("--prefer-free", action="store_true",
                        help="PIBT tie-break: among equally short cells, prefer ones no other agent holds")
    parser.add_argument("--heat-model", default="none",
                        choices=["none", "symmetric", "goal", "keskin"],
                        help="heatmap used for the delay d_i (protocol/heatmap_delay.py)")
    parser.add_argument("--heat-p", type=float, default=0.8, metavar="P",
                        help="goal model: chance a neighbour follows its shortest path")
    parser.add_argument("--horizon", type=int, default=3, metavar="K",
                        help="steps the heatmap delay looks ahead")
    parser.add_argument("--heat-weight", type=float, default=1.0, metavar="LAMBDA",
                        help="cost per expected agent on a cell")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--no-show", action="store_true", help="skip the window")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    instance = load_instance(args.map, args.scen, args.agents)
    urgencies = draw_urgencies(instance.num_agents, args.seed)
    multipliers = draw_multipliers(instance.num_agents, args.liars, args.multiplier, args.seed)
    rule = dict(RULES[args.rule])
    if args.emergency is not None:
        rule["emergency_wait"] = args.emergency
    config = ProtocolConfig(
        comm_range=args.comm_range,
        alpha=args.alpha,
        random_tiebreak=args.random_tiebreak,
        tiebreak_seed=args.seed,
        tiebreak_prefer_free=args.prefer_free,
        heat_model=args.heat_model,
        heat_follow_prob=args.heat_p,
        delay_horizon=args.horizon,
        heat_weight=args.heat_weight,
        **rule,
    )
    run = run_protocol(
        instance,
        make_bidding_agents(instance, urgencies, multipliers),
        config=config,
        seed=args.seed,
        max_episode_steps=args.max_steps,
    )

    result = evaluate(instance, run.paths, run.runtime_s)
    problems = find_protocol_problems(run)
    messages = sum(len(tick.messages) for tick in run.ticks)
    print(f"steps            {run.steps}")
    print(f"reached goals    {run.reached_goals} ({result.agents_at_goal}/{instance.num_agents})")
    print(f"sum of costs     {result.sum_of_costs}")
    print(f"messages / tick  {messages / max(run.steps, 1):.1f}")
    print(f"longest push     {max((t.max_push_chain for t in run.ticks), default=0)}")
    print(f"emergency ticks  {run.emergency_firings()}")
    print(f"total paid       {sum(run.total_payments()):.2f}")
    differ, total = pibt_divergence(run)
    print(f"differs from PIBT {differ}/{total} moves ({100 * differ / max(total, 1):.2f}%)")
    if run.stall:
        print(f"STALL: {run.stall}")
    liars = [i for i, m in enumerate(multipliers) if m != 1.0]
    if liars:
        def summary(group):
            """'a/n at goal, mean delay x': delay = extra timesteps over the
            shortest path, averaged over the agents that reached their goal."""
            arrived = [i for i in group if tuple(run.paths[i][-1]) == tuple(instance.goals[i])]
            extra = [
                path_cost(run.paths[i], instance.goals[i])
                - distances_to(instance.grid, instance.goals[i])[tuple(instance.starts[i])]
                for i in arrived
            ]
            delay = f"{sum(extra) / len(extra):.2f}" if extra else "-"
            return f"{len(arrived)}/{len(group)} at goal, mean delay {delay}"

        honest = [i for i in range(instance.num_agents) if i not in liars]
        print(f"liars            {len(liars)} agents bid x{args.multiplier}: {liars}")
        print(f"  liars          {summary(liars)}")
        if honest:
            print(f"  truthful       {summary(honest)}")
        print("  (delay = extra steps over the shortest path, agents at goal only)")
    print("protocol checks  " + ("passed" if not problems else f"{len(problems)} problems"))
    for problem in problems[:20]:
        print("  " + problem)

    if not args.no_show:
        from mapf_gt.visualise import show_run

        show_run(run, title=f"{args.map.stem}, k={instance.num_agents}, seed={args.seed}")
    return 0 if not problems and not run.stall else 1


if __name__ == "__main__":
    sys.exit(main())
