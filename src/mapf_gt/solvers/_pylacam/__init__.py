"""Vendored copy of py-lacam (LaCAM* with PIBT) by Keisuke Okumura.

Source: https://github.com/Kei18/py-lacam, branch "pibt",
commit 9ff5e0e89b40a3b1b6f90977a280fbfd8aa05a78. MIT licence, see LICENCE.txt.

Copied rather than installed because the package depends on JupyterLab 3 and
other tools this repo doesn't need. Only change: lacam.py logs through the
standard library instead of loguru. Use it through mapf_gt.solvers.PyLaCAM.
"""

from .lacam import LaCAM
from .mapf_utils import Config

__all__ = ["Config", "LaCAM"]
