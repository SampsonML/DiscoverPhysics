"""
Tests for the optional Gaussian observation noise on particle positions
and velocities.

Covers:
  - noise_std=0 is bit-identical to the pre-noise behavior (hard bypass)
  - noise_std>0 perturbs positions (std noise_std) and velocities
    (std noise_std / Δt, Δt = median sampling cadence)
  - velocity noise comes from an independent RNG stream, so position noise
    is bit-identical to the position-only implementation for a given seed
  - velocity noise magnitude scales with the sampling cadence
  - fresh noise on every call (agent cannot average it out trivially)
  - noise_disabled() context manager restores noise_std afterwards
  - the trajectory evaluator's ground-truth run is unaffected by noise_std
"""

import numpy as np
import pytest

from scienceagent.executor import (
    SimulationExecutor,
    SpeciesExecutor,
    CircleExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
)
from scienceagent.evaluator import Evaluator

_GRAVITY_OPS = [{"type": "laplacian", "params": {"strength": 1.0}}]
_EXP = {
    "p1": 1.0,
    "p2": 1.0,
    "pos2": [3.0, 0.0],
    "velocity2": [0.0, 0.0],
    "measurement_times": [1.0, 2.0, 3.0],
}


def _make_executor(noise_std=0.0, noise_seed=None):
    return SimulationExecutor(
        operators=_GRAVITY_OPS,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )


def test_noise_std_zero_matches_clean_run():
    """noise_std=0 must be bit-identical to the pre-noise code path."""
    clean = _make_executor(noise_std=0.0)
    clean_result = clean.run([_EXP])[0]

    # Re-run with a different seed but still noise_std=0 — should still match
    bypassed = _make_executor(noise_std=0.0, noise_seed=12345)
    bypassed_result = bypassed.run([_EXP])[0]

    np.testing.assert_array_equal(
        np.asarray(clean_result["pos2"]), np.asarray(bypassed_result["pos2"])
    )
    np.testing.assert_array_equal(
        np.asarray(clean_result["pos1"]), np.asarray(bypassed_result["pos1"])
    )


def test_noise_perturbs_positions_and_velocities():
    clean = _make_executor(noise_std=0.0).run([_EXP])[0]
    noisy = _make_executor(noise_std=0.1, noise_seed=0).run([_EXP])[0]

    clean_pos2 = np.asarray(clean["pos2"])
    noisy_pos2 = np.asarray(noisy["pos2"])
    assert not np.allclose(clean_pos2, noisy_pos2), "noise should perturb positions"

    # Perturbations should be within a few sigma
    diff = noisy_pos2 - clean_pos2
    assert np.abs(diff).max() < 1.0, "noise should be small relative to scale"

    # Velocities are now noised too (scaled by 1/Δt, drawn from an
    # independent RNG stream).
    for key in ("velocity1", "velocity2"):
        clean_v = np.asarray(clean[key])
        noisy_v = np.asarray(noisy[key])
        assert clean_v.shape == noisy_v.shape
        assert not np.allclose(
            clean_v, noisy_v
        ), f"{key} should now be perturbed by velocity noise"


def test_velocity_noise_uses_independent_stream():
    """Velocity-noise draws must not disturb the position-noise sequence.

    Swapping in a completely different velocity RNG must leave the recorded
    positions bit-identical for the same seed — this is what guarantees that
    previously-recorded position-only seeded runs still reproduce exactly.
    """
    ex = _make_executor(noise_std=0.1, noise_seed=0)
    pos = np.asarray(ex.run([_EXP])[0]["pos2"])

    ex2 = _make_executor(noise_std=0.1, noise_seed=0)
    ex2._vel_noise_rng = np.random.default_rng(999_999)
    pos2 = np.asarray(ex2.run([_EXP])[0]["pos2"])
    np.testing.assert_array_equal(pos, pos2)


def test_velocity_noise_scales_with_cadence():
    """σ_v = noise_std / Δt → a finer cadence yields larger velocity noise."""
    coarse = dict(_EXP, measurement_times=[1.0, 2.0, 3.0, 4.0, 5.0])  # Δt=1.0
    fine = dict(_EXP, measurement_times=[0.25, 0.5, 0.75, 1.0, 1.25])  # Δt=0.25

    clean_c = _make_executor(0.0).run([coarse])[0]
    noisy_c = _make_executor(0.1, noise_seed=0).run([coarse])[0]
    clean_f = _make_executor(0.0).run([fine])[0]
    noisy_f = _make_executor(0.1, noise_seed=0).run([fine])[0]

    # noisy - clean isolates the injected velocity-noise array (the simulation
    # is deterministic, so the true velocities cancel).
    std_c = np.std(
        np.asarray(noisy_c["velocity2"]) - np.asarray(clean_c["velocity2"])
    )
    std_f = np.std(
        np.asarray(noisy_f["velocity2"]) - np.asarray(clean_f["velocity2"])
    )
    # Δt 1.0 → 0.25 is 4× finer ⇒ ~4× larger velocity-noise std.
    assert std_f > std_c * 2.0


def test_noise_is_fresh_across_calls():
    """Two consecutive calls with the same executor should yield different noise."""
    ex = _make_executor(noise_std=0.1, noise_seed=42)
    a = np.asarray(ex.run([_EXP])[0]["pos2"])
    b = np.asarray(ex.run([_EXP])[0]["pos2"])
    assert not np.array_equal(
        a, b
    ), "fresh noise should be sampled on every call so the agent cannot average it out"


def test_noise_seed_is_reproducible_across_executors():
    a = _make_executor(noise_std=0.1, noise_seed=7).run([_EXP])[0]["pos2"]
    b = _make_executor(noise_std=0.1, noise_seed=7).run([_EXP])[0]["pos2"]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_noise_disabled_context_manager():
    ex = _make_executor(noise_std=0.1, noise_seed=99)
    assert ex.noise_std == 0.1
    with ex.noise_disabled():
        assert ex.noise_std == 0.0
        result = ex.run([_EXP])[0]
        clean = _make_executor(noise_std=0.0).run([_EXP])[0]
        np.testing.assert_array_equal(
            np.asarray(result["pos2"]), np.asarray(clean["pos2"])
        )
        # noise_disabled() zeros noise_std, so velocities are clean too.
        np.testing.assert_array_equal(
            np.asarray(result["velocity2"]), np.asarray(clean["velocity2"])
        )
    assert ex.noise_std == 0.1, "noise_std should be restored after the context"


def test_noise_disabled_restores_on_exception():
    ex = _make_executor(noise_std=0.05)
    with pytest.raises(RuntimeError):
        with ex.noise_disabled():
            assert ex.noise_std == 0.0
            raise RuntimeError("boom")
    assert ex.noise_std == 0.05


def test_evaluator_ground_truth_is_noise_free():
    """The trajectory evaluator must compare against clean ground truth."""
    noisy_executor = _make_executor(noise_std=0.5, noise_seed=1)
    clean_executor = _make_executor(noise_std=0.0)

    test_cases = [_EXP]
    evaluator_noisy = Evaluator(noisy_executor, test_cases=test_cases)
    evaluator_clean = Evaluator(clean_executor, test_cases=test_cases)

    # A trivial law that just returns the input position unchanged — useful as a
    # fixed reference: any difference between the two evaluators must come from
    # the ground-truth trajectories, not from the predicted side.
    law_source = (
        "def discovered_law(pos1, pos2, p1, p2, velocity2, duration):\n"
        "    return pos2, velocity2\n"
    )

    res_noisy = evaluator_noisy.evaluate(law_source, verbose=False)
    res_clean = evaluator_clean.evaluate(law_source, verbose=False)

    assert res_noisy["mean_pos_error"] == pytest.approx(res_clean["mean_pos_error"])
    # noise_std restored after the evaluator returns
    assert noisy_executor.noise_std == 0.5


@pytest.mark.parametrize(
    "cls",
    [
        SimulationExecutor,
        SpeciesExecutor,
        CircleExecutor,
        ThreeSpeciesExecutor,
        DarkMatterExecutor,
    ],
)
def test_all_executors_accept_noise_kwargs(cls):
    """Smoke test: every executor class accepts noise_std/noise_seed in __init__."""
    ex = cls(noise_std=0.1, noise_seed=0)
    assert ex.noise_std == 0.1
    assert hasattr(ex, "noise_disabled")
