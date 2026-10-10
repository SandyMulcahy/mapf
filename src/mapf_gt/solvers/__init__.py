from .base import Solution, Solver
from .cbs import CBS
from .lacam import LaCAM

SOLVERS: dict[str, type[Solver]] = {"cbs": CBS, "lacam": LaCAM}

__all__ = ["CBS", "LaCAM", "SOLVERS", "Solution", "Solver"]
