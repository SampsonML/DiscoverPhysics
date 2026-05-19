"""
Circle world tests: fractional Laplacian (alpha=0.75) with 11 particles.

Layout: 1 center particle + 10 particles equally spaced on a ring.

Tests:
  - Momentum conservation over many steps
  - Center particle feels near-zero net force (ring symmetry)
  - MSE between fractional (alpha=0.75) and standard Laplacian trajectories
    quantifies how "non-standard" the fractional world is relative to ordinary
    gravity — a meaningful baseline for the agent discovery task.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from physchool.worlds.field_sampler import FieldSampler

# ── World parameters ────────────────────────────────────────────────────────

N_RING = 10
N_TOTAL = 11  # 1 center + 10 ring
ALPHA = 0.75  # fractional Laplacian exponent
RING_RADIUS = 5.0  # distance from center to ring particles
DOMAIN = 50.0
CENTER = DOMAIN / 2


def _ring_positions():
    """Return (11, 2) array: center particle first, then 10 ring particles."""
    angles = np.linspace(0, 2 * np.pi, N_RING, endpoint=False)
    ring = np.column_stack(
        [
            CENTER + RING_RADIUS * np.cos(angles),
            CENTER + RING_RADIUS * np.sin(angles),
        ]
    )
    center = np.array([[CENTER, CENTER]])
    return np.vstack([center, ring])


def make_circle_sampler(ops=None, dt=0.01, grid_size=128):
    """11-particle circle world with fractional Laplacian (alpha=0.75) by default."""
    if ops is None:
        ops = [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": ALPHA},
            }
        ]
    masses = np.ones(N_TOTAL)
    return FieldSampler(
        particle_inertia=masses,
        particle_source=masses,
        particle_force=masses,
        initial_positions=_ring_positions(),
        initial_velocities=np.zeros((N_TOTAL, 2)),
        spatial_dimensions=2,
        temporal_order=0,
        grid_size=(grid_size, grid_size),
        domain_size=DOMAIN,
        operators=ops,
        n_particles=N_TOTAL,
        force_law="gradient",
        dt=dt,
        source_coupling=masses,
        force_coupling=1.0,
        periodic_boundaries=False,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCircleWorldMomentum:
    """Total momentum should be conserved (starts at zero, stays near zero)."""

    def test_momentum_conserved_short(self):
        """After 50 steps, total momentum should remain near zero."""
        s = make_circle_sampler()
        masses = np.asarray(s.particle_inertia)

        p0 = np.sum(s.velocities * masses[:, None], axis=0)
        for _ in range(50):
            s.step()
        p1 = np.sum(s.velocities * masses[:, None], axis=0)

        dp = np.linalg.norm(p1 - p0)
        # Scale by total impulse: max_force * dt * n_steps
        f0 = s.step()
        force_scale = float(np.max(np.sqrt(np.sum(np.asarray(f0) ** 2, axis=-1))))
        impulse_scale = force_scale * s.dt * 50
        assert (
            dp < impulse_scale * 0.05
        ), f"|Δp| = {dp:.2e} exceeds 5% of impulse scale {impulse_scale:.2e}"

    def test_momentum_conserved_long(self):
        """After 200 steps the total momentum should remain near zero."""
        s = make_circle_sampler()
        masses = np.asarray(s.particle_inertia)

        p0 = np.sum(s.velocities * masses[:, None], axis=0)
        for _ in range(200):
            s.step()
        p1 = np.sum(s.velocities * masses[:, None], axis=0)

        dp = np.linalg.norm(p1 - p0)
        total_displacement = (
            np.mean(np.sqrt(np.sum(s.velocities**2, axis=-1))) * 200 * s.dt
        )
        assert dp < 0.01 * max(
            total_displacement, 1e-8
        ), f"|Δp| = {dp:.2e} is too large after 200 steps"


class TestCircleWorldSymmetry:
    """Ring symmetry → center particle should feel zero net force."""

    def test_center_particle_near_zero_force(self):
        """Net force on center particle should be < 2% of mean ring force."""
        s = make_circle_sampler(grid_size=128)
        forces = np.asarray(s.step())

        center_force = float(np.sqrt(np.sum(forces[0] ** 2)))
        ring_forces = np.sqrt(np.sum(forces[1:] ** 2, axis=-1))
        mean_ring_force = float(np.mean(ring_forces))

        assert center_force < mean_ring_force * 0.02, (
            f"Center force {center_force:.2e} should be <2% of "
            f"mean ring force {mean_ring_force:.2e}"
        )

    def test_ring_forces_point_inward(self):
        """Each ring particle's force should have a positive component toward center."""
        s = make_circle_sampler(grid_size=128)
        forces = np.asarray(s.step())
        positions = s.positions

        inward = positions[0] - positions[1:]  # vectors toward center
        inward /= np.linalg.norm(inward, axis=-1, keepdims=True)
        dot_products = np.sum(forces[1:] * inward, axis=-1)

        assert np.all(
            dot_products > 0
        ), f"Some ring particles' forces do not point inward: {dot_products}"


class TestCircleWorldMSE:
    """
    MSE between fractional Laplacian (alpha=0.75) and standard Laplacian trajectories.

    'True' positions:      fractional world (alpha=0.75)
    'Predicted' positions: standard Laplacian world (alpha=1.0, same ICs)

    A large MSE means the fractional world is hard to predict with ordinary gravity —
    which is the signal the discovery agent must learn to detect.
    """

    def _run(self, ops, n_steps=200, dt=0.005):
        s = make_circle_sampler(ops=ops, dt=dt)
        for _ in range(n_steps):
            s.step()
        return s.positions.copy()

    def test_fractional_differs_from_laplacian(self):
        """Fractional (alpha=0.75) and standard Laplacian should diverge noticeably.

        1000 steps × dt=0.005 = 5.0 time units gives mean displacement ~0.21 and
        MSE well above the 1e-3 threshold.
        """
        ops_frac = [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": ALPHA},
            }
        ]
        ops_lap = [{"type": "laplacian", "params": {"strength": 1.0}}]

        pos_true = self._run(ops_frac, n_steps=1000)
        pos_pred = self._run(ops_lap, n_steps=1000)

        mse = float(np.mean((pos_true - pos_pred) ** 2))
        assert mse > 1e-3, (
            f"Fractional and Laplacian worlds should produce different trajectories "
            f"(MSE={mse:.2e} is too small)"
        )

    def test_mse_grows_with_steps(self):
        """Divergence between fractional and Laplacian worlds should grow over time."""
        ops_frac = [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": ALPHA},
            }
        ]
        ops_lap = [{"type": "laplacian", "params": {"strength": 1.0}}]

        mse_values = []
        for n_steps in [200, 400, 1000]:
            pos_true = self._run(ops_frac, n_steps=n_steps)
            pos_pred = self._run(ops_lap, n_steps=n_steps)
            mse_values.append(float(np.mean((pos_true - pos_pred) ** 2)))

        assert (
            mse_values[0] < mse_values[1] < mse_values[2]
        ), f"MSE should grow over time: {mse_values}"

    def test_same_model_gives_zero_mse(self):
        """Two runs with identical operators should give identical trajectories (MSE=0)."""
        ops = [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": ALPHA},
            }
        ]

        pos_a = self._run(ops)
        pos_b = self._run(ops)

        mse = float(np.mean((pos_a - pos_b) ** 2))
        assert mse < 1e-20, f"Identical runs should give zero MSE, got {mse:.2e}"
