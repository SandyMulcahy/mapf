from .base import Solution, Solver
from .cbs import CBS
from .lacam import LaCAM
from .pylacam import PyLaCAM

SOLVERS: dict[str, type[Solver]] = {"cbs": CBS, "lacam": LaCAM, "pylacam": PyLaCAM}

__all__ = ["CBS", "LaCAM", "PyLaCAM", "SOLVERS", "Solution", "Solver"]
