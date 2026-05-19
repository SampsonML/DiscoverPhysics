"""
Dark matter world tests: standard Laplacian with 20 visible + 10 hidden + 5 probes.

Visible  (particles 0-19):  source_coupling = 1.0, reported to agent
Dark     (particles 20-29): source_coupling = 5.0, hidden from agent
Probes   (particles 30-34): source_coupling = 0.0, reported to agent

Tests:
  - Agent output contains exactly 25 particles (dark matter is hidden)
  - Dark matter generates detectable excess force toward the halo center
  - Probes do not affect dynamics (source_coupling = 0)
  - Trajectories with dark matter diverge from visible-only simulation
"""

import numpy as np
import pytest
from physchool.worlds.field_sampler import FieldSampler

# ── World parameters ────────────────────────────────────────────────────────

N_VISIBLE = 20
N_DARK = 10
N_PROBES = 5
N_TOTAL = 35
SOURCE_VIS = 1.0
SOURCE_DARK = 5.0
DOMAIN = 50.0
CENTER = DOMAIN / 2
GRID = 128


def _positions_and_couplings():
    """Return (positions, velocities, source_couplings) for the full 35-particle system.
    Must match DarkMatterExecutor.__init__ exactly (same seed, same layout)."""
    rng = np.random.RandomState(123)
    # Visible: orbiting at larger radii (ring between r=8 and r=15)
    vis_angles = rng.uniform(0, 2 * np.pi, N_VISIBLE)
    vis_radii = rng.uniform(8, 15, N_VISIBLE)
    vis_pos_rel = np.column_stack(
        [
            vis_radii * np.cos(vis_angles),
            vis_radii * np.sin(vis_angles),
        ]
    )
    # Dark matter: tightly clustered (σ = 1.0)
    dark_pos_rel = rng.normal(0, 1.0, (N_DARK, 2))

    probe_pos = np.array(
        [
            [5, 0],
            [0, 5],
            [-5, 0],
            [0, -5],
            [7, 7],
        ],
        dtype=np.float64,
    )

    positions = np.vstack(
        [vis_pos_rel + CENTER, dark_pos_rel + CENTER, probe_pos + CENTER]
    )

    # Tangential velocities for visible particles (match executor)
    # Per-particle enclosed mass
    vis_r = np.linalg.norm(vis_pos_rel, axis=1)
    dark_r = np.linalg.norm(dark_pos_rel, axis=1)
    M_enclosed = np.zeros(N_VISIBLE)
    for i in range(N_VISIBLE):
        ri = vis_r[i]
        M_enclosed[i] = (
            np.sum(dark_r < ri) * SOURCE_DARK
            + np.sum(vis_r < ri) * SOURCE_VIS
            - SOURCE_VIS  # exclude self
        )
    v_circ = np.sqrt(np.maximum(M_enclosed, 0.0) / (2 * np.pi))
    r_safe = np.maximum(vis_r, 1e-6)
    tangent = np.column_stack([-vis_pos_rel[:, 1], vis_pos_rel[:, 0]]) / r_safe[:, None]
    vis_vel = v_circ[:, None] * tangent

    velocities = np.vstack([vis_vel, np.zeros((N_DARK, 2)), np.zeros((N_PROBES, 2))])

    source = np.zeros(N_TOTAL)
    source[0:N_VISIBLE] = SOURCE_VIS
    source[N_VISIBLE : N_VISIBLE + N_DARK] = SOURCE_DARK
    return positions, velocities, source


def _make_sampler(source_coupling=None, dt=0.005):
    positions, velocities, default_source = _positions_and_couplings()
    if source_coupling is None:
        source_coupling = default_source
    masses = np.ones(N_TOTAL)
    return FieldSampler(
        particle_inertia=masses,
        particle_source=masses,
        particle_force=masses,
        initial_positions=positions.copy(),
        initial_velocities=velocities.copy(),
        spatial_dimensions=2,
        temporal_order=0,
        grid_size=(GRID, GRID),
        domain_size=DOMAIN,
        operators=[{"type": "laplacian", "params": {"strength": 1.0}}],
        n_particles=N_TOTAL,
        force_law="gradient",
        dt=dt,
        source_coupling=source_coupling,
        force_coupling=1.0,
        periodic_boundaries=False,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDarkMatterHiding:
    """Agent-facing executor must hide the dark matter particles."""

    def test_agent_sees_25_particles(self):
        """DarkMatterExecutor.run() should return only 25 particles."""
        from scienceagent.executor import DarkMatterExecutor

        ex = DarkMatterExecutor()
        result = ex.run(
            [
                {
                    "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
                    "probe_velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                    "measurement_times": [1.0],
                }
            ]
        )[0]
        assert len(result["positions"][0]) == 25
        assert len(result["background_initial_positions"]) == 20

    def test_full_output_has_35_particles(self):
        """DarkMatterExecutor.run_full() should return all 35 particles."""
        from scienceagent.executor import DarkMatterExecutor

        ex = DarkMatterExecutor()
        result = ex.run_full(
            [
                {
                    "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
                    "probe_velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                    "measurement_times": [1.0],
                }
            ]
        )[0]
        assert len(result["positions"][0]) == 35
        assert len(result["dark_initial_positions"]) == 10
        assert len(result["field_snapshots"]) == 1


class TestDarkMatterForces:
    """Dark matter halo should create detectable excess attraction toward center."""

    def test_excess_force_toward_center(self):
        """A probe near the dark halo center should feel stronger force
        than explained by visible matter alone."""
        positions, velocities, source_with_dark = _positions_and_couplings()

        # With dark matter
        sim_dm = _make_sampler(source_coupling=source_with_dark)
        forces_dm = np.asarray(sim_dm.step())
        probe_force_dm = np.linalg.norm(forces_dm[30])  # probe 0 at (5, 0)

        # Without dark matter (zero out dark source coupling)
        source_no_dark = source_with_dark.copy()
        source_no_dark[N_VISIBLE : N_VISIBLE + N_DARK] = 0.0
        sim_no = _make_sampler(source_coupling=source_no_dark)
        forces_no = np.asarray(sim_no.step())
        probe_force_no = np.linalg.norm(forces_no[30])

        # Dark matter should significantly boost the force
        assert probe_force_dm > 2.0 * probe_force_no, (
            f"Force with dark matter ({probe_force_dm:.4f}) should be "
            f"much larger than without ({probe_force_no:.4f})"
        )

    def test_dark_matter_pulls_toward_halo(self):
        """Force on a probe should have a component pointing toward the
        dark matter cluster center (near domain center)."""
        sim = _make_sampler()
        forces = np.asarray(sim.step())

        # Probe 0 is at (CENTER+5, CENTER). Dark halo is near (CENTER, CENTER).
        # Force should have a negative x-component (pointing toward center).
        assert (
            forces[30, 0] < 0
        ), f"Probe at +x should be pulled toward center, got Fx={forces[30, 0]:.4e}"


class TestDarkMatterMSE:
    """Trajectories with dark matter should diverge from visible-only simulation."""

    def _run(self, source_coupling, n_steps=300, dt=0.005):
        sim = _make_sampler(source_coupling=source_coupling, dt=dt)
        for _ in range(n_steps):
            sim.step()
        return sim.positions.copy()

    def test_dark_matter_changes_trajectories(self):
        _, _, source_dm = _positions_and_couplings()
        source_no_dm = source_dm.copy()
        source_no_dm[N_VISIBLE : N_VISIBLE + N_DARK] = 0.0

        pos_dm = self._run(source_dm, n_steps=500)
        pos_no = self._run(source_no_dm, n_steps=500)

        # Compare visible particles only
        mse = float(np.mean((pos_dm[:N_VISIBLE] - pos_no[:N_VISIBLE]) ** 2))
        assert (
            mse > 0.1
        ), f"Dark matter should change visible trajectories significantly (MSE={mse:.2e})"

    def test_identical_runs_zero_mse(self):
        _, _, source = _positions_and_couplings()
        pos_a = self._run(source)
        pos_b = self._run(source)
        mse = float(np.mean((pos_a - pos_b) ** 2))
        assert mse < 1e-20, f"Identical runs should give zero MSE, got {mse:.2e}"


def _total_angular_momentum(positions, velocities, centre):
    """Compute total scalar angular momentum L = Σ (r × v)_z about centre."""
    r = positions - centre
    # In 2D: L_z = x*vy - y*vx
    return float(np.sum(r[:, 0] * velocities[:, 1] - r[:, 1] * velocities[:, 0]))


class TestDarkMatterAngularMomentum:
    """Total angular momentum should be approximately conserved.

    The system has a central potential (dark halo near origin) and the
    visible particles orbit CCW. Torques between particles in a
    symmetric-ish configuration should nearly cancel, so total L_z
    should not drift much over moderate timescales.
    """

    def test_angular_momentum_conserved_short(self):
        """After 200 steps, total L_z should stay within 5% of its initial value."""
        sim = _make_sampler()
        centre = np.array([CENTER, CENTER])
        L0 = _total_angular_momentum(sim.positions, sim.velocities, centre)

        for _ in range(200):
            sim.step()

        L1 = _total_angular_momentum(sim.positions, sim.velocities, centre)
        rel_change = abs(L1 - L0) / abs(L0)
        assert rel_change < 0.05, (
            f"Angular momentum drifted by {rel_change:.2%} "
            f"(L0={L0:.4f}, L1={L1:.4f})"
        )

    def test_angular_momentum_conserved_long(self):
        """After 1000 steps, total L_z should stay within 15% of initial."""
        sim = _make_sampler()
        centre = np.array([CENTER, CENTER])
        L0 = _total_angular_momentum(sim.positions, sim.velocities, centre)

        for _ in range(1000):
            sim.step()

        L1 = _total_angular_momentum(sim.positions, sim.velocities, centre)
        rel_change = abs(L1 - L0) / abs(L0)
        assert rel_change < 0.15, (
            f"Angular momentum drifted by {rel_change:.2%} after 1000 steps "
            f"(L0={L0:.4f}, L1={L1:.4f})"
        )

    def test_initial_angular_momentum_positive(self):
        """Visible particles orbit CCW, so total L_z should be positive at t=0."""
        sim = _make_sampler()
        centre = np.array([CENTER, CENTER])
        L0 = _total_angular_momentum(sim.positions, sim.velocities, centre)
        assert L0 > 0, f"Expected positive (CCW) angular momentum, got L0={L0:.4f}"
