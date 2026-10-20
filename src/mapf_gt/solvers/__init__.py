from .base import Solution, Solver
from .cbs import CBS
from .pylacam import PyLaCAM
from .pibt import PIBT

SOLVERS: dict[str, type[Solver]] = {"cbs": CBS, "pylacam": PyLaCAM,"pibt": PIBT}

__all__ = ["CBS", "LaCAM", "PIBT", "PyLaCAM", "SOLVERS", "Solution", "Solver"]
