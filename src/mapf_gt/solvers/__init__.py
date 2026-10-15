from .base import Solution, Solver
from .cbs import CBS
from .pylacam import PyLaCAM

SOLVERS: dict[str, type[Solver]] = {"cbs": CBS, "pylacam": PyLaCAM}

__all__ = ["CBS", "LaCAM", "PyLaCAM", "SOLVERS", "Solution", "Solver"]
