## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
## Layout

```
src/mapf_gt/
  maps/movingai.py        read and write MovingAI .map and .scen files
  maps/pogema_adapter.py  MovingAI instance -> POGEMA environment
  agents/base.py          interface for decentralised agents
  agents/game_theoretic.py   the algorithm (interface only so far)
  solvers/cbs.py          optimal CBS in pure Python, for small instances
  solvers/pylacam.py      wrapper for Python implementation ofLaCAM
  rollout.py              run agents, or replay a plan, inside POGEMA
  validate.py             independent collision and rule checker
  visualise.py            live matplotlib view of agents moving
  metrics.py              success, sum of costs, makespan
experiments/              YAML configs, run.py, plot.py
data/small/               two toy 8x8 maps used by the tests
data/download.py          fetches the full MovingAI benchmark
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


## Watching agents move

```
python experiments/visualise.py --map data/small/toy-walls-8.map \--scen data/small/toy-walls-8.scen -k 6 --method cbs
```

This solves the instance, runs the plan in POGEMA and opens a window showing
goals, trails and each agent's field of view. Space pauses, the arrow keys
step one timestep, and Home restarts.

- `--focus 2` draws only agent 2's field of view.
- `--raw` draws the solver's plan without POGEMA. Use it to debug a solver:
  conflicts are ringed in red, which POGEMA alone never shows because it
  blocks colliding moves.
- `--save run.svg` also writes POGEMA's own animation (`.svg`, `.html`,
  `.mp4` or `.tex`); add `--focus` for its egocentric view.
- `--no-show` skips the window, e.g. when only saving.

From Python: `mapf_gt.visualise.show_paths(instance, paths)`.

