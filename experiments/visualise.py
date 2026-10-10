"""Solve one instance and watch the agents move.

    python experiments/visualise.py \\
        --map data/small/toy-walls-8.map --scen data/small/toy-walls-8.scen \\
        -k 6 --method cbs

By default the plan is executed in POGEMA and the window shows what POGEMA
did. Use --raw to draw the solver's plan exactly as returned (useful when
debugging a solver, since POGEMA itself never lets agents collide).

--save also writes POGEMA's own animation: .svg, .html, .mp4 or .tex.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mapf_gt.agents import AGENTS
from mapf_gt.maps import load_instance
from mapf_gt.metrics import evaluate
from mapf_gt.solvers import SOLVERS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="keys: space = pause/play, left/right = step, home = restart",
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--scen", type=Path, required=True)
    parser.add_argument("-k", "--agents", type=int, default=4, help="number of agents")
    parser.add_argument("--method", default="cbs", choices=sorted([*SOLVERS, *AGENTS]))
    parser.add_argument("--time-limit", type=float, default=30.0, help="seconds")
    parser.add_argument("--obs-radius", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256,
                        help="episode length for decentralised agents")
    parser.add_argument("--raw", action="store_true",
                        help="draw the solver's plan without running it in POGEMA")
    parser.add_argument("--focus", type=int, default=None,
                        help="only draw this agent's field of view")
    parser.add_argument("--save", type=Path, default=None,
                        help="also save POGEMA's animation (.svg/.html/.mp4/.tex)")
    parser.add_argument("--no-show", action="store_true", help="don't open a window")
    parser.add_argument("--no-goals", action="store_true")
    parser.add_argument("--no-trails", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="hide fields of view")
    parser.add_argument("--no-conflicts", action="store_true")
    parser.add_argument("--speed", type=int, default=60, help="ms per frame")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.raw and args.save:
        sys.exit("--save needs POGEMA to run the episode, so it can't be used with --raw")

    instance = load_instance(args.map, args.scen, args.agents)
    env_config = {"obs_radius": args.obs_radius}
    env = None

    if args.method in SOLVERS:
        try:
            solution = SOLVERS[args.method]().solve(instance, args.time_limit)
        except FileNotFoundError as missing:
            print(missing)
            return 1
        if solution is None:
            print(f"{args.method} found no solution within {args.time_limit}s")
            return 1
        paths = solution.paths
        result = evaluate(instance, paths, solution.runtime_s)
        if not args.raw:
            from mapf_gt.rollout import execute_plan, first_divergence

            executed, env = execute_plan(
                instance, paths, record=args.save is not None, **env_config
            )
            gap = first_divergence(paths, executed)
            if gap:
                agent, t, want, got = gap
                print(f"warning: POGEMA did not follow the plan. Agent {agent} at "
                      f"t={t} should be at {want} but is at {got}.")
            paths = executed
    else:
        if args.raw:
            sys.exit("--raw only applies to centralised solvers")
        from mapf_gt.rollout import run_agents

        rollout = run_agents(
            instance, AGENTS[args.method], record=args.save is not None,
            max_episode_steps=args.max_steps, **env_config,
        )
        paths, env = rollout.paths, rollout.env
        result = evaluate(instance, paths, rollout.runtime_s)

    print(f"{args.method}: success={result.success} sum_of_costs={result.sum_of_costs} "
          f"makespan={result.makespan} runtime={result.runtime_s:.3f}s")

    if args.save is not None:
        from mapf_gt.maps.pogema_adapter import save_animation

        save_animation(env, args.save, focus=args.focus)
        print(f"wrote {args.save}")

    if not args.no_show:
        from mapf_gt.visualise import GridAnimator

        source = "plan" if args.raw else "POGEMA"
        GridAnimator(
            instance, paths,
            goals=not args.no_goals,
            trails=not args.no_trails,
            conflicts=not args.no_conflicts,
            obs_radius=None if args.no_view else args.obs_radius,
            focus=args.focus,
            interval_ms=args.speed,
            title=f"{args.method} ({source}), {instance.name}",
        ).show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
