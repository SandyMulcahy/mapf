"""LaCAM wrapper. The parser test uses a real lacam3 output file; the solver
test runs only if a lacam3 binary is available (LACAM_BIN or the default)."""

from pathlib import Path

import pytest

from conftest import DATA
from mapf_gt.maps import load_instance
from mapf_gt.metrics import evaluate
from mapf_gt.solvers import CBS, LaCAM
from mapf_gt.solvers.lacam import parse_lacam_output

FIXTURE = Path(__file__).parent / "fixtures" / "lacam3_result.txt"


def test_parser_reads_a_real_result_file():
    # Produced by: build/main -m toy-walls-8.map -i toy-walls-8.scen -N 3
    instance = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 3)
    paths = parse_lacam_output(FIXTURE.read_text(), 3)
    assert paths is not None
    assert [path[0] for path in paths] == list(instance.starts)  # (x, y) swapped
    assert [path[-1] for path in paths] == list(instance.goals)
    result = evaluate(instance, paths, 0.0)
    assert result.success
    assert result.sum_of_costs == 19  # the file says soc=19


def test_parser_returns_none_when_unsolved():
    text = FIXTURE.read_text().replace("solved=1", "solved=0")
    assert parse_lacam_output(text, 3) is None


def test_solver_matches_cbs_on_small_instances():
    solver = LaCAM()
    if not solver.binary.exists():
        pytest.skip("lacam3 binary not built")
    for stem, k in (("toy-walls-8", 6), ("toy-empty-8", 8)):
        instance = load_instance(DATA / f"{stem}.map", DATA / f"{stem}.scen", k)
        lacam = solver.solve(instance, time_limit_s=3)
        assert lacam is not None
        result = evaluate(instance, lacam.paths, lacam.runtime_s)
        assert result.success
        optimal = evaluate(instance, CBS().solve(instance, 10).paths, 0.0)
        assert result.sum_of_costs >= optimal.sum_of_costs
