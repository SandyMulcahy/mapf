# mapf-game-theory

A decentralised, game-theoretic algorithm for Multi-Agent Path Finding (MAPF),
benchmarked against centralised planners on the MovingAI MAPF benchmark.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests in `tests/test_adapter.py` need POGEMA and are skipped without it.
Everything else (loader, CBS, validator, metrics) runs without POGEMA.

## Layout

```
src/mapf_gt/
  maps/movingai.py        read and write MovingAI .map and .scen files
  maps/pogema_adapter.py  MovingAI instance -> POGEMA environment
  agents/base.py          interface for decentralised agents
  agents/game_theoretic.py   the algorithm (interface only so far)
  solvers/cbs.py          optimal CBS in pure Python, for small instances
  solvers/lacam.py        wrapper for a compiled LaCAM binary (untested)
  rollout.py              run agents, or replay a plan, inside POGEMA
  validate.py             independent collision and rule checker
  metrics.py              success, sum of costs, makespan
experiments/              YAML configs, run.py, plot.py
data/small/               two toy 8x8 maps used by the tests
data/download.py          fetches the full MovingAI benchmark
external/                 C++ baselines as git submodules
tests/
```

## Conventions

- **Coordinates are (row, col) everywhere.** MovingAI files use (x, y), which
  is (col, row); the swap happens only in `maps/movingai.py`.
- **Agents are taken from a scenario in file order.** "Map X, scenario Y, k
  agents" then means the same problem as in published results.
- **Agents stay at their goal and keep blocking it** (`on_target="nothing"` in
  POGEMA). POGEMA's default removes agents on arrival, which is an easier
  problem than the one CBS and LaCAM solve.
- **Cost** is the time at which an agent reaches its goal for the last time.
  Sum of costs adds this over agents; makespan takes the maximum.
- **Every result is validated** by `validate.py` before metrics are recorded,
  whichever solver or simulator produced it.

## Running an experiment

```
python experiments/run.py experiments/configs/cbs_small.yaml
python experiments/plot.py results/cbs_small.csv
```

For the real benchmark, run `python data/download.py` and point a config's
`maps_dir` at the downloaded files.

## Comparing fairly

The decentralised agents see a local window; CBS and LaCAM see the whole map.
State the observation radius alongside every result. Per-step decision time
for decentralised agents is not the same quantity as a centralised planner's
one-off planning time, so report them separately.
