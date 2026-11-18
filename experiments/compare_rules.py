"""Compare priority/ageing rules for the bidding protocol across maps and densities.

    python experiments/compare_rules.py                     # defaults below
    python experiments/compare_rules.py --rules A E --agents 10 20 --seeds 1 2 3

Each (map, agents, seed) uses scenario file <map>-random-<seed>.scen and urgency
draw <seed>, so every rule sees exactly the same instances. All agents are
truthful. Writes one CSV row per run and prints, for each rule, the number of
runs solved summed over the maps, per agent count (x/x/x/x).

Rules (see protocol/priority.py):
  A  plan's original rule: b*d + alpha*age, age resets when the agent gets
     closer; emergency tier on age >= 5
  D  as A, but the emergency tier counts ticks since last at goal (>= 20)
  E  urgency-weighted ageing b*(d + alpha*age), age = ticks since last at
     goal; no emergency tier (the current default)
"""

from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

from mapf_gt.agents.bidding import draw_urgencies, make_bidding_agents
from mapf_gt.maps import load_instance
from mapf_gt.metrics import sum_of_costs
from mapf_gt.protocol import ProtocolConfig
from mapf_gt.rollout import run_protocol

RULES = {
    "A": dict(priority_rule="additive", age_rule="since_progress", emergency_wait=5),
    "D": dict(priority_rule="additive", age_rule="since_progress",
              emergency_wait=20, emergency_counter="since_goal"),
    "E": dict(priority_rule="weighted", age_rule="since_goal", emergency_wait=None),
}
MAPS = ["empty-32-32", "random-32-32-20", "room-32-32-4", "maze-32-32-4", "warehouse-10-20-10-2-1"]
FIELDS = ["rule", "map", "agents", "seed", "solved", "at_goal", "steps",
          "sum_of_costs", "emergency_agent_ticks", "stall"]


def run_one(job):
    data, rule, name, k, seed, alpha, comm_range, max_steps = job
    instance = load_instance(
        data / "maps" / f"{name}.map",
        data / "scen-random" / "scen-random" / f"{name}-random-{seed}.scen",
        k,
    )
    config = ProtocolConfig(alpha=alpha, comm_range=comm_range, **RULES[rule])
    run = run_protocol(
        instance,
        make_bidding_agents(instance, draw_urgencies(k, seed)),
        config=config,
        seed=seed,
        max_episode_steps=max_steps,
    )
    return {
        "rule": rule,
        "map": name,
        "agents": k,
        "seed": seed,
        "solved": run.reached_goals,
        "at_goal": sum(tuple(p[-1]) == tuple(g) for p, g in zip(run.paths, instance.goals)),
        "steps": run.steps,
        "sum_of_costs": sum_of_costs(run.paths, instance.goals) if run.reached_goals else "",
        "emergency_agent_ticks": run.emergency_firings(),
        "stall": run.stall or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/movingai"))
    parser.add_argument("--maps", nargs="+", default=MAPS)
    parser.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    parser.add_argument("--agents", nargs="+", type=int, default=[10, 20, 40, 60])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--range", type=int, default=2, dest="comm_range")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/compare_rules.csv"))
    args = parser.parse_args()

    jobs = [
        (args.data, rule, name, k, seed, args.alpha, args.comm_range, args.max_steps)
        for name in args.maps for k in args.agents for seed in args.seeds for rule in args.rules
    ]
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=1)
    print(f"{len(rows)} runs in {time.perf_counter() - started:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")

    total = len(args.maps) * len(args.seeds)
    print(f"\nRuns solved, summed over {len(args.maps)} maps x {len(args.seeds)} seeds "
          f"(out of {total}), for {'/'.join(map(str, args.agents))} agents:")
    for rule in args.rules:
        solved = [
            sum(r["solved"] for r in rows if r["rule"] == rule and r["agents"] == k)
            for k in args.agents
        ]
        print(f"  {rule}  " + "/".join(map(str, solved)))
    stalls = sum(1 for r in rows if r["stall"])
    if stalls:
        print(f"\nWARNING: {stalls} runs stalled (protocol bug); see the stall column")


if __name__ == "__main__":
    main()
