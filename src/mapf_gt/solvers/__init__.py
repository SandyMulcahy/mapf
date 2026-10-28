from .base import Solution, Solver
from .cbs import CBS
from .pylacam import PyLaCAM
from .pibt import PIBT
from .urgency_pibt import UrgencyPIBT

# UrgencyPIBT is not listed: it needs urgencies, so it cannot be built with no arguments.
SOLVERS: dict[str, type[Solver]] = {"cbs": CBS, "pylacam": PyLaCAM,"pibt": PIBT}

__all__ = ["CBS", "LaCAM", "PIBT", "PyLaCAM", "SOLVERS", "Solution", "Solver", "UrgencyPIBT"]
