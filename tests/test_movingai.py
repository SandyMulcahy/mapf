import pytest

from conftest import DATA, grid_from_rows
from mapf_gt.maps import (
    Instance,
    load_instance,
    load_map,
    load_scenario,
    make_instance,
    write_map,
    write_scenario,
)

# 2 rows by 5 columns, deliberately not square, with one obstacle at
# row 0, column 3. Any (x, y) / (row, col) mix-up fails on this map.
MAP_TEXT = "type octile\nheight 2\nwidth 5\nmap\n...@.\n.....\n"
# start x=4, y=0  -> (row 0, col 4);  goal x=0, y=1 -> (row 1, col 0)
SCEN_TEXT = "version 1\n0\twide.map\t5\t2\t4\t0\t0\t1\t5.00000000\n"


def write_pair(tmp_path):
    map_path = tmp_path / "wide.map"
    scen_path = tmp_path / "wide.scen"
    map_path.write_text(MAP_TEXT)
    scen_path.write_text(SCEN_TEXT)
    return map_path, scen_path


def test_map_shape_and_obstacles(tmp_path):
    map_path, _ = write_pair(tmp_path)
    grid = load_map(map_path)
    assert (grid.height, grid.width) == (2, 5)
    assert not grid.is_free((0, 3))
    assert grid.is_free((1, 3))
    assert int(grid.obstacles.sum()) == 1


def test_scenario_converts_xy_to_row_col(tmp_path):
    _, scen_path = write_pair(tmp_path)
    (entry,) = load_scenario(scen_path)
    assert entry.start == (0, 4)
    assert entry.goal == (1, 0)
    assert (entry.map_width, entry.map_height) == (5, 2)
    assert entry.optimal_length == 5.0


def test_swapped_coordinates_would_be_rejected(tmp_path):
    # (4, 0) read as (row, col) is outside this 2-row map, so a loader that
    # forgot to swap could not build the instance at all.
    map_path, _ = write_pair(tmp_path)
    grid = load_map(map_path)
    with pytest.raises(ValueError):
        Instance(grid, ((4, 0),), ((0, 1),))


def test_out_of_bounds_cells_are_not_free():
    grid = grid_from_rows(["..", ".."])
    assert not grid.is_free((-1, 0))
    assert not grid.is_free((0, 2))
    assert sorted(grid.neighbours((0, 0))) == [(0, 1), (1, 0)]


def test_all_obstacle_characters(tmp_path):
    path = tmp_path / "terrain.map"
    path.write_text("type octile\nheight 1\nwidth 7\nmap\n.GS@OTW\n")
    grid = load_map(path)
    assert grid.obstacles[0].tolist() == [False, False, False, True, True, True, True]


def test_bad_map_is_rejected(tmp_path):
    path = tmp_path / "bad.map"
    path.write_text("type octile\nheight 2\nwidth 3\nmap\n...\n..\n")
    with pytest.raises(ValueError):
        load_map(path)


def test_make_instance_takes_agents_in_file_order():
    grid = load_map(DATA / "toy-empty-8.map")
    entries = load_scenario(DATA / "toy-empty-8.scen")
    instance = make_instance(grid, entries, 3)
    assert instance.num_agents == 3
    assert instance.starts == tuple(entry.start for entry in entries[:3])
    assert instance.goals == tuple(entry.goal for entry in entries[:3])


def test_make_instance_rejects_too_many_agents_or_wrong_map(tmp_path):
    grid = load_map(DATA / "toy-empty-8.map")
    entries = load_scenario(DATA / "toy-empty-8.scen")
    with pytest.raises(ValueError):
        make_instance(grid, entries, len(entries) + 1)
    map_path, _ = write_pair(tmp_path)
    with pytest.raises(ValueError):
        make_instance(load_map(map_path), entries, 1)


def test_instance_rejects_blocked_or_shared_cells():
    grid = grid_from_rows(["..@", "..."])
    with pytest.raises(ValueError):
        Instance(grid, ((0, 2),), ((1, 0),))
    with pytest.raises(ValueError):
        Instance(grid, ((0, 0), (0, 0)), ((1, 0), (1, 1)))
    with pytest.raises(ValueError):
        Instance(grid, ((0, 0), (0, 1)), ((1, 0), (1, 0)))


def test_write_then_load_round_trip(tmp_path):
    original = load_instance(DATA / "toy-walls-8.map", DATA / "toy-walls-8.scen", 5)
    write_map(original.grid, tmp_path / "copy.map")
    write_scenario(original, tmp_path / "copy.scen", map_name="copy.map")
    copy = load_instance(tmp_path / "copy.map", tmp_path / "copy.scen", 5)
    assert (copy.grid.obstacles == original.grid.obstacles).all()
    assert copy.starts == original.starts
    assert copy.goals == original.goals
