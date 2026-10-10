"""Run one experiment config and write one CSV row per (instance, agents, method).

    python experiments/run.py experiments/configs/cbs_small.yaml

Centralised solvers are evaluated straight from their plans. Decentralised
agents are rolled out in POGEMA. Both go through the same validator and
metrics, so the columns mean the same thing for every method.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from mapf_gt.agents import AGENTS
from mapf_gt.maps import load_instance
from mapf_gt.metrics import evaluate
from mapf_gt.solvers import SOLVERS
from mapf_gt.validate import find_problems

FIELDS = [
    "experiment", "map", "scen", "num_agents", "method", "success",
    "sum_of_costs", "makespan", "agents_at_goal", "runtime_s", "problems",
]


def run_method(method: str, instance, config: dict):
    time_limit = float(config.get("time_limit_s", 30))
    if method in SOLVERS:
        solution = SOLVERS[method]().solve(instance, time_limit)
        if solution is None:
            return evaluate(instance, None, time_limit)
        return evaluate(instance, solution.paths, solution.runtime_s)

    if method in AGENTS:
        from mapf_gt.rollout import run_agents  # needs POGEMA

        rollout = run_agents(instance, AGENTS[method], **config.get("pogema", {}))
        result = evaluate(instance, rollout.paths, rollout.runtime_s)
        if not rollout.reached_goals:
            # Report only rule violations, not the missing goals themselves.
            broken = find_problems(instance, rollout.paths, require_goals=False)
            problems = tuple(broken) or ("did not reach all goals",)
            result = type(result)(
                False, None, None, result.agents_at_goal, result.runtime_s, problems
            )
        return result

    known = sorted([*SOLVERS, *AGENTS])
    raise ValueError(f"unknown method {method!r}; choose from {known}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    name = config.get("name", args.config.stem)
    maps_dir = Path(config["maps_dir"])
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{name}.csv"

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for item in config["instances"]:
            for num_agents in config["agent_counts"]:
                instance = load_instance(
                    maps_dir / item["map"], maps_dir / item["scen"], num_agents
                )
                for method in config["methods"]:
                    result = run_method(method, instance, config)
                    row = {
                        "experiment": name,
                        "map": item["map"],
                        "scen": item["scen"],
                        "num_agents": num_agents,
                        "method": method,
                        **result.as_row(),
                    }
                    writer.writerow(row)
                    handle.flush()
                    print(
                        f"{item['scen']:<24} k={num_agents:<3} {method:<16} "
                        f"success={result.success} soc={result.sum_of_costs} "
                        f"time={result.runtime_s:.3f}s"
                    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
