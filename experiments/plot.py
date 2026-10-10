"""Plot success rate and mean sum of costs against number of agents.

    python experiments/plot.py results/cbs_small.csv

Mean sum of costs is taken over solved instances only, so compare it between
methods only where their success rates are similar.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    solved = defaultdict(list)  # (method, k) -> [0/1, ...]
    costs = defaultdict(list)  # (method, k) -> [soc, ...] for solved runs
    with args.csv_path.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], int(row["num_agents"]))
            ok = row["success"] == "True"
            solved[key].append(1 if ok else 0)
            if ok:
                costs[key].append(int(row["sum_of_costs"]))

    methods = sorted({method for method, _ in solved})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        ks = sorted(k for m, k in solved if m == method)
        left.plot(
            ks,
            [sum(solved[method, k]) / len(solved[method, k]) for k in ks],
            marker="o",
            label=method,
        )
        cost_ks = [k for k in ks if costs[method, k]]
        right.plot(
            cost_ks,
            [sum(costs[method, k]) / len(costs[method, k]) for k in cost_ks],
            marker="o",
            label=method,
        )
    left.set(xlabel="agents", ylabel="success rate", ylim=(-0.05, 1.05))
    right.set(xlabel="agents", ylabel="mean sum of costs (solved only)")
    left.legend()
    fig.tight_layout()
    out_path = args.csv_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
