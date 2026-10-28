"""Run the bidding protocol on one instance, validate it and (optionally) watch it.

    python experiments/run_protocol.py \\
        --map data/small/toy-walls-8.map --scen data/small/toy-walls-8.scen -k 6

All agents are truthful. Urgencies are drawn from --seed, so a run can be
replayed exactly. Payments are currently placeholders (all zero).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mapf_gt.agents.bidding import draw_urgencies, make_bidding_agents
from mapf_gt.maps import load_instance
from mapf_gt.metrics import evaluate
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
    parser.add_argument("--emergency", type=int, default=None, metavar="W",
                        help="wait after which an agent enters the emergency tier")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--no-show", action="store_true", help="skip the window")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    instance = load_instance(args.map, args.scen, args.agents)
    urgencies = draw_urgencies(instance.num_agents, args.seed)
    config = ProtocolConfig(
        comm_range=args.comm_range, alpha=args.alpha, emergency_wait=args.emergency
    )
    run = run_protocol(
        instance,
        make_bidding_agents(instance, urgencies),
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
    print("protocol checks  " + ("passed" if not problems else f"{len(problems)} problems"))
    for problem in problems[:20]:
        print("  " + problem)

    if not args.no_show:
        from mapf_gt.visualise import show_run

        show_run(run, title=f"{args.map.stem}, k={instance.num_agents}, seed={args.seed}")
    return 0 if not problems and not run.stall else 1


if __name__ == "__main__":
    sys.exit(main())
