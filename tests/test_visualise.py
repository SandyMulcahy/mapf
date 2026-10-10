import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window during tests

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import DATA, instance_from_rows  # noqa: E402
from mapf_gt.validate import find_conflicts  # noqa: E402
from mapf_gt.visualise import GridAnimator  # noqa: E402

ROWS = ["....", "....", "...."]


def test_find_conflicts_reports_vertex_and_edge_with_times():
    paths = [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 2), (0, 1), (0, 0)],  # meets agent 0 at (0, 1) at t=1
        [(2, 0), (2, 1)],
        [(2, 1), (2, 0)],  # swaps with agent 2 at t=1
    ]
    conflicts = find_conflicts(paths)
    kinds = {(c.kind, c.agents, c.t) for c in conflicts}
    assert ("vertex", (0, 1), 1) in kinds
    assert ("edge", (2, 3), 1) in kinds


def test_frames_and_interpolation():
    instance = instance_from_rows(ROWS, [(0, 0)], [(0, 3)])
    animator = GridAnimator(instance, [[(0, 0), (0, 1), (0, 2), (0, 3)]], substeps=4)
    assert animator.total_frames == 3 * 4 + 1
    assert animator.positions(0).tolist() == [[0.0, 0.0]]
    assert animator.positions(1.5).tolist() == [[1.5, 0.0]]  # (x=col, y=row)
    assert animator.positions(10).tolist() == [[3.0, 0.0]]  # stays at the end


def test_every_frame_draws_with_all_extras():
    instance = instance_from_rows(ROWS, [(0, 0), (2, 3)], [(0, 3), (2, 0)])
    paths = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(2, 3), (2, 2), (2, 1), (2, 0)],
    ]
    animator = GridAnimator(instance, paths, obs_radius=1, substeps=2)
    for frame in range(animator.total_frames):
        animator.draw_frame(frame)
    assert len(animator.view_patches) == 2
    assert np.allclose(animator.agent_dots.get_offsets(), [[3, 0], [0, 2]])
    assert "at goal 2/2" in animator.status.get_text()


def test_conflicts_are_highlighted_at_their_timestep():
    instance = instance_from_rows(["...."], [(0, 0), (0, 2)], [(0, 1), (0, 3)])
    paths = [[(0, 0), (0, 1)], [(0, 2), (0, 1), (0, 2), (0, 3)]]
    animator = GridAnimator(instance, paths, substeps=1)
    animator.draw_frame(1)
    assert len(animator.conflict_rings.get_offsets()) == 1
    assert "now!" in animator.status.get_text()
    animator.draw_frame(3)
    assert len(animator.conflict_rings.get_offsets()) == 0


def test_focus_only_draws_one_field_of_view():
    instance = instance_from_rows(ROWS, [(0, 0), (2, 3)], [(0, 3), (2, 0)])
    paths = [[(0, 0)], [(2, 3)]]
    animator = GridAnimator(instance, paths, obs_radius=2, focus=1)
    assert [i for i, _ in animator.view_patches] == [1]
    with pytest.raises(ValueError):
        GridAnimator(instance, paths, focus=5)


def test_command_line_runs_without_a_window(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    import visualise

    code = visualise.main([
        "--map", str(DATA / "toy-walls-8.map"),
        "--scen", str(DATA / "toy-walls-8.scen"),
        "-k", "3", "--raw", "--no-show",
    ])
    assert code == 0


def test_saving_pogemas_animation(tmp_path):
    pytest.importorskip("pogema")
    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    import visualise

    for name in ("run.svg", "run.html"):
        out = tmp_path / name
        code = visualise.main([
            "--map", str(DATA / "toy-walls-8.map"),
            "--scen", str(DATA / "toy-walls-8.scen"),
            "-k", "3", "--no-show", "--save", str(out), "--focus", "0",
        ])
        assert code == 0
        assert out.stat().st_size > 1000
