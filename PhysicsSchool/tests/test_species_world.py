"""
Species world tests: standard Laplacian with 6 particles of 2 hidden species.

Species A (particles 0,1,2): source_coupling = 1.0
Species B (particles 3,4,5): source_coupling = 3.0

Tests:
  - Species B particles generate stronger fields (larger acceleration on a test particle)
  - Same-species particles at identical distances produce identical accelerations
  - Momentum conservation over many steps
  - MSE between uniform-source and species models shows the species difference is detectable
"""

import jax.numpy as jnp
import numpy as np
import pytest
from physchool.worlds.field_sampler import FieldSampler

# ── World parameters ────────────────────────────────────────────────────────

N_PARTICLES = 6
SOURCE_A = 1.0
SOURCE_B = 3.0
DOMAIN = 50.0
CENTER = DOMAIN / 2
GRID = 128


def _source_couplings():
    s = np.ones(N_PARTICLES)
    s[3:] = SOURCE_B
    return s


def make_species_sampler(positions_rel, velocities=None, dt=0.005, grid_size=GRID):
    """Create a 6-particle species world from positions relative to center."""
    positions = np.array(positions_rel, dtype=np.float64) + CENTER
    if velocities is None:
        velocities = np.zeros((N_PARTICLES, 2))
    else:
        velocities = np.array(velocities, dtype=np.float64)

    masses = np.ones(N_PARTICLES)
    source = _source_couplings()
    return FieldSampler(
        particle_inertia=masses,
        particle_source=masses,
        particle_force=masses,
        initial_positions=positions,
        initial_velocities=velocities,
        spatial_dimensions=2,
        temporal_order=0,
        grid_size=(grid_size, grid_size),
        domain_size=DOMAIN,
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        n_particles=N_PARTICLES,
        force_law="gradient",
        dt=dt,
        source_coupling=source,
        force_coupling=1.0,
        periodic_boundaries=True,
    )


def make_uniform_sampler(positions_rel, velocities=None, dt=0.005, grid_size=GRID):
    """Same layout but all particles have source_coupling = 1.0 (no species)."""
    positions = np.array(positions_rel, dtype=np.float64) + CENTER
    if velocities is None:
        velocities = np.zeros((N_PARTICLES, 2))
    else:
        velocities = np.array(velocities, dtype=np.float64)

    masses = np.ones(N_PARTICLES)
    return FieldSampler(
        particle_inertia=masses,
        particle_source=masses,
        particle_force=masses,
        initial_positions=positions,
        initial_velocities=velocities,
        spatial_dimensions=2,
        temporal_order=0,
        grid_size=(grid_size, grid_size),
        domain_size=DOMAIN,
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        n_particles=N_PARTICLES,
        force_law="gradient",
        dt=dt,
        source_coupling=np.ones(N_PARTICLES),
        force_coupling=1.0,
        periodic_boundaries=True,
    )


# ── Default layout ──────────────────────────────────────────────────────────


def _default_positions():
    """Asymmetric layout so species differences produce measurable effects."""
    return [
        [0.0, 0.0],  # particle 0 (A) at center
        [5.0, 0.0],  # particle 1 (A)
        [-5.0, 0.0],  # particle 2 (A)
        [0.0, 5.0],  # particle 3 (B)
        [0.0, -5.0],  # particle 4 (B)
        [4.0, 4.0],  # particle 5 (B)
    ]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSpeciesForces:
    """Species B should generate ~3x stronger fields than species A."""

    def test_species_b_stronger_force(self):
        """A test particle near a species-B source should accelerate ~3x more
        than near a species-A source at the same distance.

        Background particles are placed identically in both setups (far from
        the source-probe pair) so the only difference is the source coupling.
        """
        FAR = 20.0  # far enough on 50x50 domain that background is negligible

        # Setup A: particle 0 (A, source=1) at origin, particle 1 (A) probes at (3,0)
        # Background particles at symmetric far positions (identical in both setups
        # except for the swapped source/background particle).
        pos_a = [
            [0.0, 0.0],  # 0 (A, source=1) — SOURCE
            [3.0, 0.0],  # 1 (A) — PROBE
            [-FAR, FAR],  # 2 (A)
            [FAR, FAR],  # 3 (B)
            [-FAR, -FAR],  # 4 (B)
            [FAR, -FAR],  # 5 (B)
        ]
        sim_a = make_species_sampler(pos_a)
        f_a = np.asarray(sim_a.step())
        accel_from_a = np.linalg.norm(f_a[1])

        # Setup B: particle 3 (B, source=3) at origin, particle 1 (A) probes at (3,0)
        # Particle 0 (A) takes the far-away slot that particle 3 had.
        pos_b = [
            [FAR, FAR],  # 0 (A) — moved to where 3 was
            [3.0, 0.0],  # 1 (A) — PROBE (same position)
            [-FAR, FAR],  # 2 (A)
            [0.0, 0.0],  # 3 (B, source=3) — SOURCE
            [-FAR, -FAR],  # 4 (B)
            [FAR, -FAR],  # 5 (B)
        ]
        sim_b = make_species_sampler(pos_b)
        f_b = np.asarray(sim_b.step())
        accel_from_b = np.linalg.norm(f_b[1])

        ratio = accel_from_b / accel_from_a
        assert 2.5 < ratio < 3.5, f"Force ratio should be ~3.0, got {ratio:.2f}"

    def test_same_species_same_force(self):
        """Two species-A sources at equal distance from a test particle
        should produce equal forces."""
        FAR = 20.0

        # Particle 0 (A) as source at origin, particle 1 (A) as probe at (3,0)
        pos_0 = [
            [0.0, 0.0],
            [3.0, 0.0],
            [-FAR, FAR],
            [FAR, FAR],
            [-FAR, -FAR],
            [FAR, -FAR],
        ]
        sim_0 = make_species_sampler(pos_0)
        f_0 = np.asarray(sim_0.step())
        force_0 = np.linalg.norm(f_0[1])

        # Particle 2 (A) as source at origin, particle 1 (A) as probe at (3,0)
        pos_2 = [
            [-FAR, FAR],
            [3.0, 0.0],
            [0.0, 0.0],
            [FAR, FAR],
            [-FAR, -FAR],
            [FAR, -FAR],
        ]
        sim_2 = make_species_sampler(pos_2)
        f_2 = np.asarray(sim_2.step())
        force_2 = np.linalg.norm(f_2[1])

        rel_diff = abs(force_0 - force_2) / max(force_0, 1e-10)
        assert (
            rel_diff < 0.05
        ), f"Same-species sources should give equal forces, rel diff = {rel_diff:.4f}"


class TestSpeciesAsymmetry:
    """Newton's 3rd law is broken: source_coupling differs but particle_force
    is uniform, so species B pushes harder than it is pushed back."""

    def test_force_asymmetry(self):
        """In a 2-particle isolation, force on A from B should be 3x larger
        than force on B from A (because B has 3x source coupling)."""
        FAR = 20.0
        # Particle 0 (A) and particle 3 (B) close; others far away
        pos = [
            [-2.0, 0.0],  # 0 (A, source=1)
            [FAR, FAR],  # 1 (A)
            [-FAR, FAR],  # 2 (A)
            [2.0, 0.0],  # 3 (B, source=3)
            [-FAR, -FAR],  # 4 (B)
            [FAR, -FAR],  # 5 (B)
        ]
        sim = make_species_sampler(pos)
        forces = np.asarray(sim.step())

        f_on_a = np.linalg.norm(forces[0])  # force on A from B's field
        f_on_b = np.linalg.norm(forces[3])  # force on B from A's field

        # B has 3x source coupling, so it generates 3x stronger field
        # → force on A should be ~3x force on B
        ratio = f_on_a / f_on_b
        assert (
            2.5 < ratio < 3.5
        ), f"Force asymmetry ratio should be ~3.0, got {ratio:.2f}"

    def test_momentum_not_conserved(self):
        """Total momentum should drift (Newton's 3rd law is broken)."""
        s = make_species_sampler(_default_positions())
        p0 = np.sum(s.velocities, axis=0)

        for _ in range(200):
            s.step()
        p1 = np.sum(s.velocities, axis=0)

        dp = np.linalg.norm(p1 - p0)
        assert (
            dp > 1e-4
        ), f"Momentum should drift due to broken Newton's 3rd law, got dp={dp:.2e}"


class TestSpeciesMSE:
    """Species world should diverge from a uniform (all source=1) world."""

    def _run(self, sampler_fn, n_steps=200, dt=0.005):
        s = sampler_fn(_default_positions(), dt=dt)
        for _ in range(n_steps):
            s.step()
        return s.positions.copy()

    def test_species_differs_from_uniform(self):
        """Trajectories with species should differ from uniform source model."""
        pos_species = self._run(make_species_sampler, n_steps=500)
        pos_uniform = self._run(make_uniform_sampler, n_steps=500)

        mse = float(np.mean((pos_species - pos_uniform) ** 2))
        assert mse > 1e-3, f"Species and uniform worlds should diverge (MSE={mse:.2e})"

    def test_mse_grows_with_time(self):
        """Divergence should grow over time."""
        mse_values = []
        for n_steps in [100, 200, 500]:
            pos_s = self._run(make_species_sampler, n_steps=n_steps)
            pos_u = self._run(make_uniform_sampler, n_steps=n_steps)
            mse_values.append(float(np.mean((pos_s - pos_u) ** 2)))

        assert (
            mse_values[0] < mse_values[1] < mse_values[2]
        ), f"MSE should grow over time: {mse_values}"

    def test_identical_runs_zero_mse(self):
        """Two runs with same species config should be identical."""
        pos_a = self._run(make_species_sampler)
        pos_b = self._run(make_species_sampler)
        mse = float(np.mean((pos_a - pos_b) ** 2))
        assert mse < 1e-20, f"Identical runs should give zero MSE, got {mse:.2e}"
