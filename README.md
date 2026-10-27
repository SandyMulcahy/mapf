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
  agents/truthful.py      the game-theoretic agent (interface only)
  agents/bidding.py       agent for the bidding protocol (stub)
  protocol/context.py     what an agent may see and send: public view, messages (stub)
  protocol/tick_runner.py the clock and the radio: event ordering for one tick (stub)
  protocol/payments.py    per-tick Myerson / critical-bid payments (stub)
  solvers/cbs.py          optimal CBS in pure Python, for small instances
  solvers/pibt.py         wrapper for a Python implementation of PIBT
  solvers/pylacam.py      wrapper for a Python implementation of LaCAM
  rollout.py              run agents or the protocol, or replay a plan, inside POGEMA
  validate.py             independent collision, rule and protocol checker
  visualise.py            live matplotlib view of agents moving, with an auction view
  metrics.py              success, sum of costs, makespan
  evolution.py            evolutionary strategy experiments (not implemented yet)
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
- **Agents never hold references to each other.** Protocol agents read
  neighbours only through a `protocol.Context`, which enforces the
  communication range and hides private state (urgency, bid strategy,
  payments). Everything else they do is a message delivered by the runner.
- **One run, one record.** `rollout.run_protocol` returns a `RunResult`;
  metrics, validation and visualisation read only that, and never re-run the
  protocol.

## The bidding protocol

Each tick, `protocol.TickRunner` runs four phases in order:

1. **Bid and reveal.** Each agent computes its bid and priority
   `(tier, bid + alpha * wait, id)` and broadcasts it to agents in range.
2. **Move (event-driven PIBT).** Pending messages are delivered first, in the
   order they were sent. When none are pending, the runnable agent with the
   highest priority plans: it reserves a free cell, or pushes the agent on it,
   which inherits its priority. An agent is runnable once every
   higher-priority agent in range has finished.
3. **Execute.** The reserved moves are checked for collisions, then POGEMA
   takes one step.
4. **Pay and update.** Payments are computed from the tick log, and each
   agent's waiting count is updated.

The runner only routes messages and keeps time; it never decides a move.
`TickRunner`, `Context`, `payments` and `BiddingAgent` are currently stubs,
so `run_protocol` will raise `NotImplementedError` until they are written.

`validate.find_protocol_problems(run)` checks a protocol run from its record:

- the MAPF rules hold;
- the protocol did not stall;
- every message stayed within range;
- each agent ended up where it reserved, which catches POGEMA silently
  resolving a collision;
- payments are never negative and never exceed the bid;
- each tick matches centralised PIBT run with the same priorities
  (`validate.reference_pibt_step`).

For the PIBT check to pass, agents must use the same tie-break as the
reference: candidates are "stay" followed by `GridMap.neighbours(pos)`,
stable-sorted by distance to goal.

## Running an experiment

```
python experiments/run.py experiments/configs/cbs_small.yaml
python experiments/plot.py results/cbs_small.csv
```

## Watching agents move

```
python experiments/visualise.py --map data/small/toy-walls-8.map \
    --scen data/small/toy-walls-8.scen -k 6 --method cbs
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

From Python: `mapf_gt.visualise.show_paths(instance, paths)` for any plan, or
`mapf_gt.visualise.show_run(run)` for a protocol run. The auction view
colours agents by urgency, labels each with its current bid, rings agents in
the emergency tier in gold, and shows payments above the agents who paid.