"""Watch agents move around a grid in a live matplotlib window.

Draws any list of paths: a solver's plan, a plan as POGEMA executed it, or a
decentralised rollout. On top of the agents it can show their goals, the
trail behind each one, the local window each agent observes, and any vertex
or edge conflicts the validator finds.

Keys while the window is open:
    space         pause or play
    left / right  step back or forward one timestep (pauses)
    home          back to the start
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from .maps.movingai import Cell, Instance
from .validate import Conflict, find_conflicts, position_at

# POGEMA's palette, so agents keep the same colour in its exported files.
POGEMA_COLOURS = (
    "#c1433c", "#2e6f9e", "#6e81af", "#00b9c8", "#72D5C8", "#0ea08c", "#8F7B66",
)
OBSTACLE = "#84A1AE"
CONFLICT = "#e8000b"


def agent_colour(agent: int) -> str:
    return POGEMA_COLOURS[agent % len(POGEMA_COLOURS)]


class GridAnimator:
    """Builds the figure and draws one frame at a time.

    ``substeps`` frames are drawn per timestep, sliding agents between cells.
    Use ``show()`` for a live window, or ``draw_frame(i)`` to render a frame
    yourself (for example in a notebook or a test).
    """

    def __init__(
        self,
        instance: Instance,
        paths: Sequence[Sequence[Cell]],
        *,
        goals: bool = True,
        trails: bool = True,
        conflicts: bool = True,
        obs_radius: int | None = None,
        focus: int | None = None,
        substeps: int = 4,
        interval_ms: int = 60,
        title: str = "",
    ):
        if len(paths) != instance.num_agents:
            raise ValueError(f"expected {instance.num_agents} paths, got {len(paths)}")
        if focus is not None and not 0 <= focus < instance.num_agents:
            raise ValueError(f"focus must be an agent index below {instance.num_agents}")

        self.instance = instance
        self.paths = [[tuple(cell) for cell in path] for path in paths]
        self.horizon = max(len(path) for path in self.paths) - 1
        self.substeps = max(1, substeps)
        self.total_frames = self.horizon * self.substeps + 1
        self.interval_ms = interval_ms
        self.title = title
        self.focus = focus
        self.show_trails = trails
        self.obs_radius = obs_radius

        self.conflicts: dict[int, list[Conflict]] = {}
        if conflicts:
            for conflict in find_conflicts(self.paths):
                self.conflicts.setdefault(conflict.t, []).append(conflict)

        self.frame = 0
        self.playing = True
        self._hold = 0  # frames to linger on the last frame before looping

        self._build_figure(goals)

    # ----------------------------------------------------------------- setup

    def _build_figure(self, goals: bool) -> None:
        grid = self.instance.grid
        n = self.instance.num_agents
        height, width = grid.height, grid.width

        scale = 7.0 / max(height, width)
        self.fig, self.ax = plt.subplots(
            figsize=(max(4.0, width * scale + 0.4), max(4.0, height * scale + 0.8))
        )
        ax = self.ax
        ax.imshow(
            grid.obstacles.astype(int),
            cmap=ListedColormap(["white", OBSTACLE]),
            vmin=0,
            vmax=1,
            origin="upper",
            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
            interpolation="nearest",
        )
        if max(height, width) <= 64:
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="#d0d0d0", linewidth=0.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        self.fig.tight_layout(rect=(0, 0, 1, 0.94))

        # Marker sizes are in points, so work out how big one cell is.
        bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        cell_pts = min(bbox.width * 72 / width, bbox.height * 72 / height)
        self._cell_pts = cell_pts
        colours = [agent_colour(i) for i in range(n)]

        if goals:
            goal_xy = np.array([(c, r) for r, c in self.instance.goals])
            ax.scatter(
                goal_xy[:, 0], goal_xy[:, 1],
                s=max(0.7 * cell_pts, 5) ** 2, marker="s", facecolors="none",
                edgecolors=colours, linewidths=max(0.8, cell_pts / 12), zorder=2,
            )

        self.trail_lines = []
        if self.show_trails:
            for i in range(n):
                (line,) = ax.plot(
                    [], [], color=colours[i], alpha=0.45,
                    linewidth=max(1.0, cell_pts / 8), solid_capstyle="round", zorder=3,
                )
                self.trail_lines.append(line)

        self.view_patches = []
        if self.obs_radius is not None:
            side = 2 * self.obs_radius + 1
            watched = [self.focus] if self.focus is not None else range(n)
            for i in watched:
                patch = Rectangle(
                    (0, 0), side, side, facecolor=colours[i], alpha=0.08,
                    edgecolor=colours[i], linewidth=1.0, zorder=1,
                )
                ax.add_patch(patch)
                self.view_patches.append((i, patch))

        self.agent_dots = ax.scatter(
            np.zeros(n), np.zeros(n), s=max(0.75 * cell_pts, 5) ** 2, c=colours,
            edgecolors="black", linewidths=0.5, zorder=4,
        )
        self.labels = []
        if cell_pts >= 14 and n <= 50:
            for i in range(n):
                self.labels.append(
                    ax.text(0, 0, str(i), ha="center", va="center", color="white",
                            fontsize=max(5, min(10, cell_pts / 2.5)), zorder=5)
                )

        self.conflict_rings = ax.scatter(
            [], [], s=max(1.3 * cell_pts, 9) ** 2, facecolors="none",
            edgecolors=CONFLICT, linewidths=max(1.5, cell_pts / 8), zorder=6,
        )
        (self.conflict_edges,) = ax.plot(
            [], [], color=CONFLICT, linewidth=max(1.5, cell_pts / 12), zorder=6
        )
        self.status = self.fig.suptitle("", fontsize=10)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # --------------------------------------------------------------- drawing

    def positions(self, t: float) -> np.ndarray:
        """(n, 2) array of (x, y) = (col, row), interpolated between steps."""
        t0 = int(math.floor(t))
        frac = t - t0
        out = np.empty((self.instance.num_agents, 2))
        for i, path in enumerate(self.paths):
            r0, c0 = position_at(path, t0)
            r1, c1 = position_at(path, t0 + 1) if frac else (r0, c0)
            out[i] = (c0 + frac * (c1 - c0), r0 + frac * (r1 - r0))
        return out

    def draw_frame(self, frame: int):
        frame = int(np.clip(frame, 0, self.total_frames - 1))
        self.frame = frame
        t = frame / self.substeps
        step = int(math.floor(t))
        xy = self.positions(t)

        self.agent_dots.set_offsets(xy)
        for i, label in enumerate(self.labels):
            label.set_position(xy[i])

        for i, line in enumerate(self.trail_lines):
            history = [(c, r) for r, c in (position_at(self.paths[i], s) for s in range(step + 1))]
            history.append(tuple(xy[i]))
            cols, rows = zip(*history)
            line.set_data(cols, rows)

        for i, patch in self.view_patches:
            x, y = xy[i]
            patch.set_xy((x - self.obs_radius - 0.5, y - self.obs_radius - 0.5))

        # Conflicts belong to integer timesteps; show them while sitting on one.
        now = self.conflicts.get(step, []) if t == step else []
        rings = [(c.cells[0][1], c.cells[0][0]) for c in now if c.kind == "vertex"]
        self.conflict_rings.set_offsets(np.array(rings) if rings else np.empty((0, 2)))
        xs: list[float] = []
        ys: list[float] = []
        for c in now:
            if c.kind == "edge":
                (r0, c0), (r1, c1) = c.cells
                xs += [c0, c1, np.nan]
                ys += [r0, r1, np.nan]
        self.conflict_edges.set_data(xs, ys)

        at_goal = sum(
            1 for i, goal in enumerate(self.instance.goals)
            if position_at(self.paths[i], step) == goal
        )
        total_conflicts = sum(len(v) for v in self.conflicts.values())
        parts = [
            self.title,
            f"t = {step} / {self.horizon}",
            f"at goal {at_goal}/{self.instance.num_agents}",
        ]
        if total_conflicts:
            parts.append(f"conflicts {total_conflicts}" + (" (now!)" if now else ""))
        if not self.playing:
            parts.append("paused")
        self.status.set_text("   ".join(p for p in parts if p))
        return []

    # ---------------------------------------------------------- live window

    def _tick(self, _):
        if self.playing:
            if self.frame < self.total_frames - 1:
                self.frame += 1
            elif self._hold < 20:
                self._hold += 1
            else:
                self._hold = 0
                self.frame = 0
        return self.draw_frame(self.frame)

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.playing = not self.playing
        elif event.key in ("right", "left"):
            self.playing = False
            step = round(self.frame / self.substeps) + (1 if event.key == "right" else -1)
            self.frame = int(np.clip(step, 0, self.horizon)) * self.substeps
        elif event.key == "home":
            self.frame = 0
        self.draw_frame(self.frame)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """Open the window and play on a loop until it is closed."""
        self.draw_frame(0)
        self._animation = FuncAnimation(  # keep a reference or it stops
            self.fig, self._tick, interval=self.interval_ms, cache_frame_data=False
        )
        plt.show()


def show_paths(instance: Instance, paths: Sequence[Sequence[Cell]], **options) -> None:
    """One-line helper: ``show_paths(instance, solution.paths)``."""
    GridAnimator(instance, paths, **options).show()
