"""
Tests for trajectory_logger: schema, t=0 rows, per-world particle counts.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from scienceagent.executor import (
    SimulationExecutor,
    CircleExecutor,
    SpeciesExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
)
from scienceagent.trajectory_logger import (
    COLUMNS,
    TrajectoryLogger,
    make_run_id,
)


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _check_common_schema(rows: list[dict], run_id: str, source: str = "agent") -> None:
    assert rows, "expected at least one row"
    for r in rows:
        assert set(r.keys()) == set(
            COLUMNS
        ), f"unexpected columns: {set(r.keys()) - set(COLUMNS)}"
        assert r["run_id"] == run_id
        assert r["source"] == source
        assert r["round"] == "1"


@pytest.fixture
def run_id():
    return make_run_id("test-model")


# ---------------------------------------------------------------------------
# 2-particle worlds
def test_two_particle_schema_and_t0(tmp_path, run_id):
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    logger = TrajectoryLogger("gravity", executor, csv_path, run_id)

    exp_input = {
        "p1": 1.5,
        "p2": 2.0,
        "pos2": [3.0, -1.0],
        "velocity2": [0.1, 0.2],
        "measurement_times": [0.5, 1.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(
        round_num=1, source="agent", exp_input=exp_input, exp_output=out
    )

    rows = _read_csv(csv_path)
    _check_common_schema(rows, run_id)

    # 2 t=0 rows + 2 particles × 2 measurement_times = 6 rows total.
    assert len(rows) == 6
    t0_rows = [r for r in rows if float(r["time"]) == 0.0]
    assert len(t0_rows) == 2
    p0_t0 = next(r for r in t0_rows if r["particle_id"] == "0")
    p1_t0 = next(r for r in t0_rows if r["particle_id"] == "1")
    assert (float(p0_t0["x"]), float(p0_t0["y"])) == (0.0, 0.0)
    assert (float(p0_t0["vx"]), float(p0_t0["vy"])) == (0.0, 0.0)
    assert (float(p1_t0["x"]), float(p1_t0["y"])) == (3.0, -1.0)
    assert (float(p1_t0["vx"]), float(p1_t0["vy"])) == (0.1, 0.2)
    # Scalar params populated; circle params blank.
    assert float(p1_t0["p1"]) == 1.5
    assert float(p1_t0["p2"]) == 2.0
    assert p1_t0["ring_radius"] == ""
    assert p1_t0["initial_tangential_velocity"] == ""


def test_two_particle_appends_across_calls(tmp_path, run_id):
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    logger = TrajectoryLogger("gravity", executor, csv_path, run_id)

    base = {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [3.0, 0.0],
        "velocity2": [0.0, 0.0],
        "measurement_times": [0.5],
    }
    [out1] = executor.run([base])
    [out2] = executor.run([base])

    logger.log_experiment(1, "agent", base, out1, exp_idx_in_round=0)
    logger.log_experiment(2, "agent", base, out2, exp_idx_in_round=0)

    rows = _read_csv(csv_path)
    exp_ids = {r["experiment_id"] for r in rows}
    assert len(exp_ids) == 2
    assert all(eid.startswith(run_id) for eid in exp_ids)


# ---------------------------------------------------------------------------
# Circle world (11 particles)
def test_circle_schema(tmp_path, run_id):
    executor = CircleExecutor()
    csv_path = tmp_path / "circle.csv"
    logger = TrajectoryLogger("circle", executor, csv_path, run_id)

    exp_input = {
        "ring_radius": 5.0,
        "initial_tangential_velocity": 0.3,
        "measurement_times": [0.5, 1.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)

    rows = _read_csv(csv_path)
    _check_common_schema(rows, run_id)

    # 11 particles × (1 t=0 + 2 measurement_times) = 33 rows.
    assert len(rows) == 33
    t0 = [r for r in rows if float(r["time"]) == 0.0]
    assert len(t0) == 11
    centre = next(r for r in t0 if r["particle_id"] == "0")
    assert (float(centre["x"]), float(centre["y"])) == (0.0, 0.0)
    # Ring particles lie on r=5 circle.
    for r in t0:
        if r["particle_id"] == "0":
            continue
        rr = math.hypot(float(r["x"]), float(r["y"]))
        assert abs(rr - 5.0) < 1e-9
    # Scalar params populated for circle, p1/p2 blank.
    assert float(centre["ring_radius"]) == 5.0
    assert float(centre["initial_tangential_velocity"]) == 0.3
    assert centre["p1"] == ""
    assert centre["p2"] == ""


# ---------------------------------------------------------------------------
# Species (6 particles)
def test_species_schema(tmp_path, run_id):
    executor = SpeciesExecutor()
    csv_path = tmp_path / "species.csv"
    logger = TrajectoryLogger("species", executor, csv_path, run_id)

    exp_input = {
        "positions": [[0, 0], [3, 0], [-3, 0], [0, 3], [0, -3], [4, 4]],
        "velocities": [[0, 0]] * 6,
        "measurement_times": [0.5, 1.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)

    rows = _read_csv(csv_path)
    _check_common_schema(rows, run_id)

    # 6 particles × (1 t=0 + 2 times) = 18 rows.
    assert len(rows) == 18
    t0 = [r for r in rows if float(r["time"]) == 0.0]
    assert len(t0) == 6
    p5_t0 = next(r for r in t0 if r["particle_id"] == "5")
    assert (float(p5_t0["x"]), float(p5_t0["y"])) == (4.0, 4.0)
    # All four scalar params blank for species.
    for col in ("p1", "p2", "ring_radius", "initial_tangential_velocity"):
        assert p5_t0[col] == ""


# ---------------------------------------------------------------------------
# Three species (35 particles)
def test_three_species_schema(tmp_path, run_id):
    executor = ThreeSpeciesExecutor()
    csv_path = tmp_path / "three_species.csv"
    logger = TrajectoryLogger("three_species", executor, csv_path, run_id)

    exp_input = {
        "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
        "probe_velocities": [[0, 0]] * 5,
        "measurement_times": [0.5],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)

    rows = _read_csv(csv_path)
    _check_common_schema(rows, run_id)

    # 35 particles × (1 t=0 + 1 time) = 70 rows.
    assert len(rows) == 70
    t0 = [r for r in rows if float(r["time"]) == 0.0]
    assert len(t0) == 35
    # Background velocities = 0 at t=0.
    for r in t0:
        pid = int(r["particle_id"])
        if pid < 30:
            assert (float(r["vx"]), float(r["vy"])) == (0.0, 0.0)
    # Probe 30 came from input.
    p30 = next(r for r in t0 if r["particle_id"] == "30")
    assert (float(p30["x"]), float(p30["y"])) == (5.0, 0.0)


# ---------------------------------------------------------------------------
# Dark matter (25 agent-visible particles)
def test_dark_matter_schema(tmp_path, run_id):
    executor = DarkMatterExecutor()
    csv_path = tmp_path / "dark_matter.csv"
    logger = TrajectoryLogger("dark_matter", executor, csv_path, run_id)

    exp_input = {
        "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
        "probe_velocities": [[0, 0]] * 5,
        "measurement_times": [1.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)

    rows = _read_csv(csv_path)
    _check_common_schema(rows, run_id)

    # 25 agent-visible particles × (1 t=0 + 1 time) = 50 rows; dark hidden.
    assert len(rows) == 50
    pids = {int(r["particle_id"]) for r in rows}
    assert pids == set(range(25))

    # Visible particle 0 t=0 velocity matches executor's _visible_velocities[0].
    t0 = [r for r in rows if float(r["time"]) == 0.0]
    p0 = next(r for r in t0 if r["particle_id"] == "0")
    expected_vel = executor._visible_velocities[0]
    assert float(p0["vx"]) == pytest.approx(expected_vel[0])
    assert float(p0["vy"]) == pytest.approx(expected_vel[1])


# ---------------------------------------------------------------------------
# End-to-end: pandas round-trip for MSE-fitting consumers.
def test_pandas_roundtrip(tmp_path, run_id):
    pd = pytest.importorskip("pandas")
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    logger = TrajectoryLogger("gravity", executor, csv_path, run_id)

    exp_input = {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [3.0, 0.0],
        "velocity2": [0.0, 0.5],
        "measurement_times": [0.5, 1.0, 2.0],
    }
    [out] = executor.run([exp_input])
    logger.log_experiment(1, "agent", exp_input, out)

    df = pd.read_csv(csv_path)
    assert list(df.columns) == list(COLUMNS)
    # Initial-conditions slice for MSE fitting.
    ic = df[df.time == 0.0].sort_values("particle_id")
    assert len(ic) == 2
    assert ic.iloc[1][["x", "y", "vx", "vy"]].tolist() == [3.0, 0.0, 0.0, 0.5]
    # Per-experiment scalar params load as floats.
    assert df["p1"].dropna().unique().tolist() == [1.0]
    assert df["p2"].dropna().unique().tolist() == [1.0]
