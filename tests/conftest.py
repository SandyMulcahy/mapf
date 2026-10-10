"""Shared helpers for building small maps in tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mapf_gt.maps import GridMap, Instance

DATA = Path(__file__).parent.parent / "data" / "small"


def grid_from_rows(rows: list[str], name: str = "test") -> GridMap:
    """'.' is free and '@' is blocked; rows[0] is the top row."""
    obstacles = np.array([[char == "@" for char in row] for row in rows], dtype=bool)
    return GridMap(name=name, obstacles=obstacles)


def instance_from_rows(rows, starts, goals) -> Instance:
    return Instance(grid_from_rows(rows), tuple(starts), tuple(goals))
