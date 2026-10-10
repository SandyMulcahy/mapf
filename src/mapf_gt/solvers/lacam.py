"""Wrapper that runs a compiled LaCAM binary (github.com/Kei18/lacam3).

NOT YET TESTED against a real binary. The command-line flags and the output
format below are written from the lacam3 README and have not been run. Check
both against the version you build, and add a test once it works.

Build lacam3 under external/ (see external/README.md), then either pass the
binary path or set the LACAM_BIN environment variable.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from ..maps.movingai import Cell, Instance, write_map, write_scenario
from .base import Solution, Solver

_PAIR = re.compile(r"\((\d+),(\d+)\)")
DEFAULT_BINARY = "external/lacam3/build/main"


def parse_lacam_output(text: str, num_agents: int) -> list[list[Cell]] | None:
    """Turn LaCAM's result file into per-agent (row, col) paths.

    After a ``solution=`` line, each line is one timestep, written as
    ``t:(x,y),(x,y),...`` with one (x, y) pair per agent.
    """
    if not re.search(r"^solved=1\s*$", text, flags=re.MULTILINE):
        return None
    _, _, body = text.partition("solution=")
    paths: list[list[Cell]] = [[] for _ in range(num_agents)]
    for line in body.splitlines():
        pairs = _PAIR.findall(line)
        if not pairs:
            continue
        if len(pairs) != num_agents:
            raise ValueError(f"expected {num_agents} positions, got {len(pairs)}")
        for agent, (x, y) in enumerate(pairs):
            paths[agent].append((int(y), int(x)))  # (x, y) -> (row, col)
    return paths if all(paths) else None


class LaCAM(Solver):
    name = "lacam"

    def __init__(self, binary: str | Path | None = None):
        self.binary = Path(binary or os.environ.get("LACAM_BIN", DEFAULT_BINARY))

    def solve(self, instance: Instance, time_limit_s: float = 30.0) -> Solution | None:
        if not self.binary.exists():
            raise FileNotFoundError(
                f"LaCAM binary not found at {self.binary}. Build lacam3 and set "
                "LACAM_BIN, or pass binary=... (see external/README.md)."
            )
        with tempfile.TemporaryDirectory() as tmp:
            map_path = Path(tmp) / "instance.map"
            scen_path = Path(tmp) / "instance.scen"
            out_path = Path(tmp) / "result.txt"
            write_map(instance.grid, map_path)
            write_scenario(instance, scen_path, map_name=map_path.name)
            command = [
                str(self.binary),
                "-m", str(map_path),
                "-i", str(scen_path),
                "-N", str(instance.num_agents),
                "-t", str(int(max(1, time_limit_s))),
                "-o", str(out_path),
                "-v", "0",
            ]
            started = time.perf_counter()
            try:
                subprocess.run(
                    command, check=True, capture_output=True, timeout=time_limit_s + 5
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return None
            runtime = time.perf_counter() - started
            if not out_path.exists():
                return None
            paths = parse_lacam_output(out_path.read_text(), instance.num_agents)
        if paths is None:
            return None
        return Solution(paths, runtime, self.name)
