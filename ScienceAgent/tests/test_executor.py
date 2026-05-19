"""
Tests for SimulationExecutor — no LLM required.
"""

import numpy as np
import pytest
from scienceagent.executor import SimulationExecutor


@pytest.fixture
def gravity_executor():
    return SimulationExecutor(
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
    )


def test_executor_returns_correct_keys(gravity_executor):
    results = gravity_executor.run(
        [
            {
                "p1": 1.0,
                "p2": 1.0,
                "pos2": [3.0, 0.0],
                "velocity2": [0.0, 0.0],
                "measurement_times": [0.0, 0.5, 1.0],
            }
        ]
    )
    assert len(results) == 1
    r = results[0]
    for key in ("measurement_times", "pos1", "pos2", "velocity1", "velocity2"):
        assert key in r, f"Missing key: {key}"


def test_executor_correct_number_of_measurements(gravity_executor):
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = gravity_executor.run(
        [
            {
                "p1": 1.0,
                "p2": 1.0,
                "pos2": [3.0, 0.0],
                "velocity2": [0.0, 0.0],
                "measurement_times": times,
            }
        ]
    )
    r = results[0]
    assert len(r["pos2"]) == len(times)
    assert len(r["velocity2"]) == len(times)


def test_particle1_stays_near_origin(gravity_executor):
    """Particle 1 has very high inertia — should barely move."""
    results = gravity_executor.run(
        [
            {
                "p1": 1.0,
                "p2": 1.0,
                "pos2": [3.0, 0.0],
                "velocity2": [0.0, 0.5],
                "measurement_times": [0.5, 1.0, 2.0],
            }
        ]
    )
    pos1 = np.array(results[0]["pos1"])
    # Should stay within 0.01 units of origin
    assert np.all(
        np.linalg.norm(pos1, axis=1) < 0.01
    ), f"Particle 1 moved unexpectedly: {pos1}"


def test_attractive_force_pulls_particle2(gravity_executor):
    """Gravity-like field: particle 2 starting at rest should be attracted toward origin."""
    results = gravity_executor.run(
        [
            {
                "p1": 1.0,
                "p2": 1.0,
                "pos2": [4.0, 0.0],
                "velocity2": [0.0, 0.0],
                "measurement_times": [0.5, 1.0, 2.0],
            }
        ]
    )
    pos2 = np.array(results[0]["pos2"])
    initial_dist = np.linalg.norm([4.0, 0.0])
    final_dist = np.linalg.norm(pos2[-1])
    assert (
        final_dist < initial_dist
    ), f"Expected attraction: final dist {final_dist:.3f} should be < initial {initial_dist:.3f}"


def test_stronger_source_gives_stronger_force(gravity_executor):
    """Doubling p1 should produce a larger displacement of p2."""
    base = gravity_executor.run(
        [
            {
                "p1": 1.0,
                "p2": 1.0,
                "pos2": [4.0, 0.0],
                "velocity2": [0.0, 0.0],
                "measurement_times": [1.0],
            }
        ]
    )[0]
    strong = gravity_executor.run(
        [
            {
                "p1": 2.0,
                "p2": 1.0,
                "pos2": [4.0, 0.0],
                "velocity2": [0.0, 0.0],
                "measurement_times": [1.0],
            }
        ]
    )[0]

    base_displacement = np.linalg.norm(np.array(base["pos2"][0]) - [4.0, 0.0])
    strong_displacement = np.linalg.norm(np.array(strong["pos2"][0]) - [4.0, 0.0])
    assert (
        strong_displacement > base_displacement
    ), f"Stronger source should displace more: {strong_displacement:.4f} vs {base_displacement:.4f}"


def test_batch_experiments(gravity_executor):
    """Multiple experiments in one call should all return results."""
    exps = [
        {
            "p1": 1.0,
            "p2": 1.0,
            "pos2": [3.0, 0.0],
            "velocity2": [0.0, 0.0],
            "measurement_times": [1.0],
        },
        {
            "p1": 1.5,
            "p2": 0.5,
            "pos2": [5.0, 1.0],
            "velocity2": [0.1, 0.0],
            "measurement_times": [0.5, 1.0],
        },
    ]
    results = gravity_executor.run(exps)
    assert len(results) == 2
    assert len(results[1]["pos2"]) == 2
