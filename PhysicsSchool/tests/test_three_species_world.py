"""
Three-species world tests: standard Laplacian with 30 + 5 particles.

Species A (particles 0-9):   source_coupling = 1.0
Species B (particles 10-19): source_coupling = 3.0
Species C (particles 20-29): source_coupling = -2.0
Probes   (particles 30-34): source_coupling = 0.0

Tests:
  - Species B generates stronger fields than A; species C generates repulsive fields
  - Probes do not disturb the background (source_coupling = 0)
  - Momentum does NOT conserve (heterogeneous source couplings break Newton's 3rd)
  - MSE between three-species and uniform-source models is detectable
"""

import numpy as np
import pytest
from physchool.worlds.field_sampler import FieldSampler

# ── World parameters ────────────────────────────────────────────────────────

N_BACKGROUND = 30
N_PROBES = 5
N_TOTAL = 35
SOURCE_A = 1.0
SOURCE_B = 3.0
SOURCE_C = -2.0
DOMAIN = 50.0
CENTER = DOMAIN / 2
GRID = 128


def _source_couplings():
    s = np.zeros(N_TOTAL)
    s[0:10] = SOURCE_A
    s[10:20] = SOURCE_B
    s[20:30] = SOURCE_C
    # probes 30-34 stay at 0
    return s


def _background_positions():
    """Fixed random background positions (same seed as ThreeSpeciesExecutor)."""
    rng = np.random.RandomState(42)
    return rng.uniform(-10, 10, (N_BACKGROUND, 2))


def _default_probe_positions():
    """5 probes at known locations relative to center."""
    return np.array(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [-5.0, 0.0],
            [0.0, -5.0],
            [7.0, 7.0],
        ]
    )


def _all_positions():
    bg = _background_positions() + CENTER
    probes = _default_probe_positions() + CENTER
    return np.vstack([bg, probes])


def make_three_species_sampler(source_coupling=None, dt=0.005, grid_size=GRID):
    """Create the 35-particle three-species world."""
    if source_coupling is None:
        source_coupling = _source_couplings()
    positions = _all_positions()
    velocities = np.zeros((N_TOTAL, 2))
    masses = np.ones(N_TOTAL)
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
        n_particles=N_TOTAL,
        force_law="gradient",
        dt=dt,
        source_coupling=source_coupling,
        force_coupling=1.0,
        periodic_boundaries=False,
    )


def make_uniform_sampler(dt=0.005, grid_size=GRID):
    """Same layout but all background particles have source_coupling = 1.0."""
    source = np.zeros(N_TOTAL)
    source[0:30] = 1.0  # uniform background
    # probes still 0
    return make_three_species_sampler(
        source_coupling=source, dt=dt, grid_size=grid_size
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestThreeSpeciesForces:
    """Species B should generate ~3x stronger fields than A; C should repel."""

    def _isolated_force_on_probe(self, source_idx):
        """Place one background particle at origin, one probe at (3, 0),
        all others far away with zeroed source coupling. Return force on probe."""
        FAR = 20.0
        positions = np.full((N_TOTAL, 2), FAR) + CENTER
        positions[source_idx] = [CENTER, CENTER]  # source at origin
        probe_idx = 30
        positions[probe_idx] = [CENTER + 3.0, CENTER]  # probe at (3, 0)

        # Zero all couplings except the one source under test
        source = np.zeros(N_TOTAL)
        source[source_idx] = _source_couplings()[source_idx]

        masses = np.ones(N_TOTAL)
        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=np.zeros((N_TOTAL, 2)),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=(GRID, GRID),
            domain_size=DOMAIN,
            operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
            n_particles=N_TOTAL,
            force_law="gradient",
            dt=0.005,
            source_coupling=source,
            force_coupling=1.0,
            periodic_boundaries=False,
        )
        forces = np.asarray(sim.step())
        return forces[probe_idx]

    def test_species_b_stronger_than_a(self):
        """Force from a species-B source should be ~3x that from species-A."""
        f_a = self._isolated_force_on_probe(0)  # particle 0, species A
        f_b = self._isolated_force_on_probe(10)  # particle 10, species B
        ratio = np.linalg.norm(f_b) / np.linalg.norm(f_a)
        assert 2.5 < ratio < 3.5, f"B/A force ratio should be ~3.0, got {ratio:.2f}"

    def test_species_c_repulsive(self):
        """Force from a species-C source should point AWAY from the source."""
        f_c = self._isolated_force_on_probe(20)  # particle 20, species C
        # Probe is at +x from source, so repulsion means force is in +x direction
        assert f_c[0] > 0, f"Species C should repel; got force x-component {f_c[0]:.4e}"

    def test_species_c_magnitude(self):
        """Species C magnitude should be ~2x species A (|source_coupling| = 2)."""
        f_a = self._isolated_force_on_probe(0)
        f_c = self._isolated_force_on_probe(20)
        ratio = np.linalg.norm(f_c) / np.linalg.norm(f_a)
        assert 1.5 < ratio < 2.5, f"|C|/|A| force ratio should be ~2.0, got {ratio:.2f}"

    def test_species_a_attractive(self):
        """Force from species A should point toward the source (attractive)."""
        f_a = self._isolated_force_on_probe(0)
        # Probe is at +x from source, so attraction means force is in -x direction
        assert (
            f_a[0] < 0
        ), f"Species A should attract; got force x-component {f_a[0]:.4e}"


class TestProbeNeutrality:
    """Probes (source_coupling=0) should not affect background trajectories."""

    def test_probe_does_not_affect_background(self):
        """Background trajectories should be identical with or without probes."""
        # Run with probes
        sim_with = make_three_species_sampler()
        for _ in range(100):
            sim_with.step()
        bg_pos_with = sim_with.positions[:N_BACKGROUND].copy()

        # Run without probes: set probe source to 0 (already 0) and
        # place them far away so force_coupling doesn't matter
        sim_without = make_three_species_sampler()
        sim_without.positions[30:] = CENTER + 100.0  # move probes far away
        for _ in range(100):
            sim_without.step()
        bg_pos_without = sim_without.positions[:N_BACKGROUND].copy()

        # Background should match (probes had source=0 so no effect either way,
        # but verify the dynamics agree)
        mse = float(np.mean((bg_pos_with - bg_pos_without) ** 2))
        assert mse < 1e-10, f"Probes should not affect background, MSE={mse:.2e}"


class TestThreeSpeciesMomentum:
    """Momentum should NOT be conserved due to heterogeneous source couplings."""

    def test_momentum_drifts(self):
        """Total momentum should drift away from zero."""
        sim = make_three_species_sampler()
        p0 = np.sum(sim.velocities, axis=0)
        for _ in range(300):
            sim.step()
        p1 = np.sum(sim.velocities, axis=0)
        dp = np.linalg.norm(p1 - p0)
        assert (
            dp > 1e-4
        ), f"Momentum should drift (broken Newton's 3rd), got dp={dp:.2e}"


class TestThreeSpeciesMSE:
    """Three-species world should diverge from uniform-source world."""

    def _run(self, sampler_fn, n_steps=200, dt=0.005):
        s = sampler_fn(dt=dt)
        for _ in range(n_steps):
            s.step()
        return s.positions.copy()

    def test_species_differs_from_uniform(self):
        pos_species = self._run(make_three_species_sampler, n_steps=500)
        pos_uniform = self._run(make_uniform_sampler, n_steps=500)
        mse = float(np.mean((pos_species - pos_uniform) ** 2))
        assert mse > 1e-3, f"Species and uniform worlds should diverge (MSE={mse:.2e})"

    def test_mse_grows_with_time(self):
        mse_values = []
        for n_steps in [100, 200, 500]:
            pos_s = self._run(make_three_species_sampler, n_steps=n_steps)
            pos_u = self._run(make_uniform_sampler, n_steps=n_steps)
            mse_values.append(float(np.mean((pos_s - pos_u) ** 2)))
        assert (
            mse_values[0] < mse_values[1] < mse_values[2]
        ), f"MSE should grow over time: {mse_values}"

    def test_identical_runs_zero_mse(self):
        pos_a = self._run(make_three_species_sampler)
        pos_b = self._run(make_three_species_sampler)
        mse = float(np.mean((pos_a - pos_b) ** 2))
        assert mse < 1e-20, f"Identical runs should give zero MSE, got {mse:.2e}"
