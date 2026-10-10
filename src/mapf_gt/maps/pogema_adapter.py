"""Turn a MovingAI instance into a POGEMA environment.

POGEMA is imported inside the functions, so the rest of the package (loader,
CBS, validator, metrics) works without it installed.

Defaults, and why
-----------------
on_target="nothing"
    Agents stay on the map at their goal and keep blocking that cell. This is
    the standard MAPF model that CBS and LaCAM solve. POGEMA's own default,
    "finish", removes agents when they arrive, which makes the problem easier
    and the results not comparable.
observation_type="POMAPF", obs_radius=5
    Each agent sees only a local window around itself.
collision_system="soft"
    Check this against the POGEMA docs for your version. Whatever it is set
    to, every trajectory is re-checked by ``mapf_gt.validate``.
"""

from __future__ import annotations

from typing import Any

from .movingai import Cell, Instance

# POGEMA's documented move order; only used if the config does not expose it.
_FALLBACK_MOVES = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]


def to_grid_config(
    instance: Instance,
    *,
    obs_radius: int = 5,
    max_episode_steps: int = 256,
    on_target: str = "nothing",
    observation_type: str = "POMAPF",
    collision_system: str = "soft",
    seed: int = 0,
    **extra: Any,
):
    """Build a ``pogema.GridConfig`` with fixed starts and goals."""
    from pogema import GridConfig

    return GridConfig(
        map=instance.grid.obstacles.astype(int).tolist(),  # 1 = obstacle
        agents_xy=[list(cell) for cell in instance.starts],  # (row, col)
        targets_xy=[list(cell) for cell in instance.goals],
        num_agents=instance.num_agents,
        obs_radius=obs_radius,
        max_episode_steps=max_episode_steps,
        on_target=on_target,
        observation_type=observation_type,
        collision_system=collision_system,
        seed=seed,
        **extra,
    )


def make_env(instance: Instance, **config: Any):
    """Create a POGEMA environment for ``instance``."""
    from pogema import pogema_v0

    return pogema_v0(grid_config=to_grid_config(instance, **config))


def move_table(grid_config) -> dict[Cell, int]:
    """Map a (d_row, d_col) step to POGEMA's action index. (0, 0) is wait."""
    moves = getattr(grid_config, "MOVES", None) or _FALLBACK_MOVES
    return {(int(d_row), int(d_col)): index for index, (d_row, d_col) in enumerate(moves)}


def action_for_move(grid_config, frm: Cell, to: Cell) -> int:
    """POGEMA action that takes an agent from ``frm`` to ``to`` in one step."""
    delta = (to[0] - frm[0], to[1] - frm[1])
    table = move_table(grid_config)
    if delta not in table:
        raise ValueError(f"{frm} -> {to} is not a single grid move")
    return table[delta]


def save_animation(env, path, focus: int | None = None, **options: Any) -> None:
    """Save POGEMA's own animation of a recorded episode.

    The format follows the file extension: .svg, .html, .mp4 (needs
    ``pip install "pogema[video]"``) or .tex (TikZ, for LaTeX). ``focus``
    draws the episode from one agent's point of view, with its field of view.
    The environment must have been run with ``record=True``.
    """
    from pathlib import Path

    suffix = Path(path).suffix.lower()
    savers = {
        ".svg": "save_animation",
        ".html": "save_html_animation",
        ".mp4": "save_video_animation",
        ".tex": "save_tikz",
    }
    if suffix not in savers:
        raise ValueError(f"can't save {suffix!r}; use one of {sorted(savers)}")
    if focus is not None:
        options["egocentric_idx"] = focus
    getattr(env, savers[suffix])(str(path), **options)


def agent_positions(env) -> list[Cell]:
    """Current (row, col) of every agent, in map coordinates.

    POGEMA pads the map by ``obs_radius`` on every side internally;
    ``ignore_borders=True`` removes that offset.
    """
    grid = env.unwrapped.grid
    return [
        (int(row), int(col)) for row, col in grid.get_agents_xy(ignore_borders=True)
    ]
