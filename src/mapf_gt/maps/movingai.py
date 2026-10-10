"""Read and write MovingAI MAPF benchmark files (.map and .scen).

Coordinates
-----------
MovingAI files store positions as (x, y) = (column, row), origin top-left.
Everything else in this package, and POGEMA, uses (row, col). The conversion
happens in this module and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

Cell = tuple[int, int]  # (row, col)

# Terrain characters from the MovingAI format. MAPF treats trees and water
# as obstacles, since agents move on a plain 4-connected grid.
PASSABLE = frozenset(".GS")
IMPASSABLE = frozenset("@OTW")

_MOVES: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True, eq=False)
class GridMap:
    """A 4-connected grid. ``obstacles[row, col]`` is True for blocked cells."""

    name: str
    obstacles: np.ndarray

    @property
    def height(self) -> int:
        return int(self.obstacles.shape[0])

    @property
    def width(self) -> int:
        return int(self.obstacles.shape[1])

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not bool(self.obstacles[cell[0], cell[1]])

    def neighbours(self, cell: Cell) -> list[Cell]:
        """Free cells one step up, down, left or right of ``cell``."""
        row, col = cell
        out = []
        for d_row, d_col in _MOVES:
            nxt = (row + d_row, col + d_col)
            if self.is_free(nxt):
                out.append(nxt)
        return out


@dataclass(frozen=True)
class ScenarioEntry:
    """One line of a .scen file: a single agent's start and goal."""

    bucket: int
    map_name: str
    map_width: int
    map_height: int
    start: Cell  # (row, col)
    goal: Cell  # (row, col)
    optimal_length: float  # as given in the file; see load_scenario


@dataclass(frozen=True, eq=False)
class Instance:
    """A one-shot MAPF problem: agent i moves from starts[i] to goals[i]."""

    grid: GridMap
    starts: tuple[Cell, ...]
    goals: tuple[Cell, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.starts) != len(self.goals):
            raise ValueError("starts and goals must have the same length")
        if not self.starts:
            raise ValueError("an instance needs at least one agent")
        for label, cells in (("start", self.starts), ("goal", self.goals)):
            for agent, cell in enumerate(cells):
                if not self.grid.is_free(cell):
                    raise ValueError(
                        f"agent {agent} {label} {cell} is blocked or outside the map"
                    )
            if len(set(cells)) != len(cells):
                raise ValueError(f"two agents share the same {label} cell")

    @property
    def num_agents(self) -> int:
        return len(self.starts)


def load_map(path: str | Path) -> GridMap:
    """Parse a MovingAI .map file."""
    path = Path(path)
    lines = path.read_text().splitlines()

    header: dict[str, str] = {}
    body_start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "map":
            body_start = index + 1
            break
        if stripped:
            key, _, value = stripped.partition(" ")
            header[key] = value.strip()
    if body_start is None:
        raise ValueError(f"{path}: no 'map' line found")

    try:
        height, width = int(header["height"]), int(header["width"])
    except KeyError as missing:
        raise ValueError(f"{path}: header is missing {missing}") from None

    rows = [line for line in lines[body_start:] if line.strip()]
    if len(rows) != height:
        raise ValueError(f"{path}: header says {height} rows, file has {len(rows)}")

    obstacles = np.zeros((height, width), dtype=bool)
    for row, line in enumerate(rows):
        line = line.rstrip("\r\n")
        if len(line) != width:
            raise ValueError(
                f"{path}: row {row} has {len(line)} cells, header says {width}"
            )
        for col, char in enumerate(line):
            if char in IMPASSABLE:
                obstacles[row, col] = True
            elif char not in PASSABLE:
                raise ValueError(f"{path}: unknown terrain {char!r} at row {row}")
    return GridMap(name=path.name, obstacles=obstacles)


def load_scenario(path: str | Path) -> list[ScenarioEntry]:
    """Parse a MovingAI .scen file, converting (x, y) to (row, col).

    ``optimal_length`` is copied from the file. In the published benchmark it
    was computed with diagonal moves allowed, so it is not the 4-connected
    shortest path length. Do not use it as a MAPF lower bound.
    """
    path = Path(path)
    entries: list[ScenarioEntry] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.lower().startswith("version"):
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{path}:{number}: expected 9 fields, got {len(parts)}")
        bucket, map_name, map_width, map_height, sx, sy, gx, gy, length = parts
        entries.append(
            ScenarioEntry(
                bucket=int(bucket),
                map_name=map_name,
                map_width=int(map_width),
                map_height=int(map_height),
                start=(int(sy), int(sx)),  # (x, y) in the file -> (row, col)
                goal=(int(gy), int(gx)),
                optimal_length=float(length),
            )
        )
    return entries


def make_instance(
    grid: GridMap,
    entries: Sequence[ScenarioEntry],
    num_agents: int,
    name: str = "",
) -> Instance:
    """Build an instance from the first ``num_agents`` scenario entries.

    Taking agents in file order is the benchmark convention: it is what makes
    "map X, scenario Y, k agents" refer to the same problem in every paper.
    """
    if not 1 <= num_agents <= len(entries):
        raise ValueError(
            f"asked for {num_agents} agents, scenario has {len(entries)} entries"
        )
    chosen = entries[:num_agents]
    for entry in chosen:
        if (entry.map_height, entry.map_width) != (grid.height, grid.width):
            raise ValueError(
                f"scenario is for a {entry.map_height}x{entry.map_width} map, "
                f"but {grid.name} is {grid.height}x{grid.width}"
            )
    return Instance(
        grid=grid,
        starts=tuple(entry.start for entry in chosen),
        goals=tuple(entry.goal for entry in chosen),
        name=name or f"{grid.name}:{num_agents}",
    )


def load_instance(
    map_path: str | Path, scen_path: str | Path, num_agents: int
) -> Instance:
    """Load a map and scenario from disk and take the first ``num_agents``."""
    grid = load_map(map_path)
    entries = load_scenario(scen_path)
    name = f"{Path(scen_path).stem}:{num_agents}"
    return make_instance(grid, entries, num_agents, name=name)


def write_map(grid: GridMap, path: str | Path) -> None:
    """Write a grid in MovingAI .map format."""
    lines = ["type octile", f"height {grid.height}", f"width {grid.width}", "map"]
    for row in grid.obstacles:
        lines.append("".join("@" if blocked else "." for blocked in row))
    Path(path).write_text("\n".join(lines) + "\n")


def write_scenario(
    instance: Instance,
    path: str | Path,
    map_name: str | None = None,
    lengths: Iterable[float] | None = None,
) -> None:
    """Write an instance in MovingAI .scen format, converting to (x, y)."""
    grid = instance.grid
    map_name = map_name or grid.name
    lengths = list(lengths) if lengths is not None else [0.0] * instance.num_agents
    lines = ["version 1"]
    for (s_row, s_col), (g_row, g_col), length in zip(
        instance.starts, instance.goals, lengths
    ):
        fields = [0, map_name, grid.width, grid.height, s_col, s_row, g_col, g_row]
        lines.append("\t".join(str(f) for f in fields) + f"\t{length:.8f}")
    Path(path).write_text("\n".join(lines) + "\n")
