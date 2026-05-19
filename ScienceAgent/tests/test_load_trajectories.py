"""
Tests for load_trajectories: CSV -> numpy arrays.

Each test logs an experiment for one world via TrajectoryLogger, then loads
the CSV back and verifies shapes, time ordering, and that initial
conditions round-trip correctly.
"""

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
from scienceagent.trajectory_logger import TrajectoryLogger, make_run_id
from scienceagent.load_trajectories import (
    ExperimentTrajectory,
    default_csv_path,
    load_trajectories,
)


def _log(
    world, executor, exp_input, csv_path, run_id, round_num=1, exp_idx=0, source="agent"
):
    logger = TrajectoryLogger(world, executor, csv_path, run_id)
    [out] = executor.run([exp_input])
    logger.log_experiment(round_num, source, exp_input, out, exp_idx_in_round=exp_idx)
    return out


# ---------------------------------------------------------------------------
# 2-particle (gravity) round-trip
def test_two_particle_roundtrip(tmp_path):
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    run_id = make_run_id("rt-gravity")
    exp_input = {
        "p1": 1.5,
        "p2": 2.0,
        "pos2": [3.0, -1.0],
        "velocity2": [0.1, 0.2],
        "measurement_times": [0.5, 1.0, 1.5],
    }
    _log("gravity", executor, exp_input, csv_path, run_id)

    [exp] = load_trajectories(csv_path)
    assert isinstance(exp, ExperimentTrajectory)
    assert exp.run_id == run_id
    assert exp.round == 1
    assert exp.source == "agent"

    # 2 particles, 4 times (t=0, 0.5, 1.0, 1.5).
    assert exp.times.shape == (4,)
    assert exp.times[0] == 0.0
    assert np.all(np.diff(exp.times) > 0)
    assert exp.positions.shape == (4, 2, 2)
    assert exp.velocities.shape == (4, 2, 2)

    # t=0 round-trip from input.
    np.testing.assert_array_equal(exp.initial_positions[0], [0.0, 0.0])
    np.testing.assert_array_equal(exp.initial_velocities[0], [0.0, 0.0])
    np.testing.assert_array_equal(exp.initial_positions[1], [3.0, -1.0])
    np.testing.assert_array_equal(exp.initial_velocities[1], [0.1, 0.2])

    # Convenience: initial_* equals positions[0] / velocities[0].
    np.testing.assert_array_equal(exp.initial_positions, exp.positions[0])
    np.testing.assert_array_equal(exp.initial_velocities, exp.velocities[0])

    assert exp.params == {"p1": 1.5, "p2": 2.0}


# ---------------------------------------------------------------------------
# Circle (11 particles) — checks geometry
def test_circle_roundtrip(tmp_path):
    executor = CircleExecutor()
    csv_path = tmp_path / "circle.csv"
    run_id = make_run_id("rt-circle")
    _log(
        "circle",
        executor,
        {
            "ring_radius": 5.0,
            "initial_tangential_velocity": 0.3,
            "measurement_times": [0.5, 1.0],
        },
        csv_path,
        run_id,
    )

    [exp] = load_trajectories(csv_path)
    assert exp.positions.shape == (3, 11, 2)
    # Centre at origin; ring on r=5 circle.
    np.testing.assert_array_equal(exp.initial_positions[0], [0.0, 0.0])
    radii = np.linalg.norm(exp.initial_positions[1:], axis=1)
    np.testing.assert_allclose(radii, np.full(10, 5.0), atol=1e-9)
    assert exp.params == {"ring_radius": 5.0, "initial_tangential_velocity": 0.3}


# ---------------------------------------------------------------------------
# Three species (35 particles)
def test_three_species_roundtrip(tmp_path):
    executor = ThreeSpeciesExecutor()
    csv_path = tmp_path / "three_species.csv"
    run_id = make_run_id("rt-3sp")
    _log(
        "three_species",
        executor,
        {
            "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
            "probe_velocities": [[0, 0]] * 5,
            "measurement_times": [0.5, 1.0],
        },
        csv_path,
        run_id,
    )

    [exp] = load_trajectories(csv_path)
    assert exp.positions.shape == (3, 35, 2)
    # Background velocities all zero at t=0.
    np.testing.assert_array_equal(exp.initial_velocities[:30], np.zeros((30, 2)))
    # Probe 0 (sim index 30) starts at (5, 0).
    np.testing.assert_array_equal(exp.initial_positions[30], [5.0, 0.0])
    assert exp.params == {}  # no scalar params for this world


# ---------------------------------------------------------------------------
# Dark matter (25 agent-visible particles, dark hidden)
def test_dark_matter_roundtrip(tmp_path):
    executor = DarkMatterExecutor()
    csv_path = tmp_path / "dark_matter.csv"
    run_id = make_run_id("rt-dm")
    _log(
        "dark_matter",
        executor,
        {
            "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
            "probe_velocities": [[0, 0]] * 5,
            "measurement_times": [1.0],
        },
        csv_path,
        run_id,
    )

    [exp] = load_trajectories(csv_path)
    assert exp.positions.shape == (2, 25, 2)
    # Visible velocities at t=0 match the executor's stored values.
    np.testing.assert_allclose(
        exp.initial_velocities[:20],
        executor._visible_velocities,
    )


# ---------------------------------------------------------------------------
# Multiple experiments + run_id filtering
def test_multiple_experiments_and_run_id_filter(tmp_path):
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    base = {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [3.0, 0.0],
        "velocity2": [0.0, 0.0],
        "measurement_times": [0.5],
    }

    run_id_a = make_run_id("run-a")
    _log("gravity", executor, base, csv_path, run_id_a, round_num=1)
    _log("gravity", executor, base, csv_path, run_id_a, round_num=2)

    run_id_b = make_run_id("run-b")
    _log("gravity", executor, base, csv_path, run_id_b, round_num=1)

    all_exps = load_trajectories(csv_path)
    assert len(all_exps) == 3

    a_only = load_trajectories(csv_path, run_id=run_id_a)
    assert len(a_only) == 2
    assert {e.run_id for e in a_only} == {run_id_a}
    assert sorted(e.round for e in a_only) == [1, 2]

    b_only = load_trajectories(csv_path, run_id=[run_id_b])
    assert len(b_only) == 1
    assert b_only[0].run_id == run_id_b


# ---------------------------------------------------------------------------
# World-name resolution falls back to default path
def test_world_name_resolution(tmp_path, monkeypatch):
    """When given a bare world name, the loader resolves it via default_csv_path."""
    fake_repo_root = tmp_path
    target = fake_repo_root / "results" / "trajectories" / "gravity.csv"

    monkeypatch.setattr(
        "scienceagent.load_trajectories.default_csv_path",
        lambda world: fake_repo_root / "results" / "trajectories" / f"{world}.csv",
    )

    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    run_id = make_run_id("rt-resolve")
    _log(
        "gravity",
        executor,
        {
            "p1": 1.0,
            "p2": 1.0,
            "pos2": [3.0, 0.0],
            "velocity2": [0.0, 0.0],
            "measurement_times": [0.5],
        },
        target,
        run_id,
    )

    exps = load_trajectories("gravity")
    assert len(exps) == 1


# ---------------------------------------------------------------------------
# Missing-file error path
def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trajectories(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# MSE-fitting style smoke check: loaded data is directly consumable.
def test_mse_fit_consumability(tmp_path):
    """Demonstrate that load_trajectories output flows into a typical MSE
    expression with no further reshaping."""
    executor = SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
    )
    csv_path = tmp_path / "gravity.csv"
    run_id = make_run_id("rt-mse")
    _log(
        "gravity",
        executor,
        {
            "p1": 1.0,
            "p2": 1.0,
            "pos2": [3.0, 0.0],
            "velocity2": [0.0, 0.0],
            "measurement_times": [0.5, 1.0, 1.5],
        },
        csv_path,
        run_id,
    )

    [exp] = load_trajectories(csv_path)

    # A trivial "law" that just returns the initial position at every time.
    def stub_law(initial_positions, initial_velocities, times, **params):
        T = len(times)
        return np.broadcast_to(initial_positions, (T, *initial_positions.shape)).copy()

    pred = stub_law(
        exp.initial_positions, exp.initial_velocities, exp.times, **exp.params
    )
    assert pred.shape == exp.positions.shape
    mse = float(((pred - exp.positions) ** 2).mean())
    assert math.isfinite(mse) and mse >= 0.0
