import jax.numpy as jnp
import numpy as np
import pytest
from physchool.worlds.field_sampler import FieldSampler


class TestOrbits:
    """Test particle trajectories against analytical solutions."""

    def test_two_body_circular_orbit_2d(self):
        """Two equal-mass particles in circular orbit should maintain separation."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        domain = 50.0
        center = domain / 2
        separation = 3.0
        half = separation / (2 * np.sqrt(2))
        positions = np.array(
            [
                [center - half, center - half],
                [center + half, center + half],
            ]
        )
        masses = np.ones(2)

        # For 2D Poisson: F = m/(2πr), centripetal: F = mv²/r_orbit
        # r_orbit = separation/2, so v = sqrt(m/(4π))
        v_orbital = np.sqrt(1.0 / (4 * np.pi))

        # Velocity perpendicular to separation vector
        pos_rel = positions[0] - np.array([center, center])
        vel_dir = np.array([-pos_rel[1], pos_rel[0]])
        vel_dir = vel_dir / np.linalg.norm(vel_dir)
        velocities = np.array([v_orbital * vel_dir, -v_orbital * vel_dir])

        s = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions.copy(),
            initial_velocities=velocities.copy(),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=(128, 128),
            domain_size=domain,
            operators=ops,
            n_particles=2,
            force_law="gradient",
            dt=0.01,
            source_coupling=masses,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        omega = v_orbital / (separation / 2)
        period = 2 * np.pi / omega

        # Run for half an orbit
        n_steps = int(0.5 * period / s.dt)
        for _ in range(n_steps):
            s.step()

        # Separation should be roughly maintained
        final_sep = np.sqrt(np.sum((s.positions[1] - s.positions[0]) ** 2))
        rel_error = abs(final_sep - separation) / separation
        assert (
            rel_error < 0.15
        ), f"Separation drifted: {separation:.3f} → {final_sep:.3f} ({rel_error:.1%})"

    def test_orbit_not_trivially_static(self):
        """Particles should actually move during the orbit test."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        domain = 50.0
        center = domain / 2
        positions = np.array([[center - 1, center], [center + 1, center]])
        masses = np.ones(2)
        v = np.sqrt(1.0 / (4 * np.pi))
        velocities = np.array([[0, v], [0, -v]])

        s = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions.copy(),
            initial_velocities=velocities.copy(),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=(64, 64),
            domain_size=domain,
            operators=ops,
            n_particles=2,
            force_law="gradient",
            dt=0.01,
            source_coupling=masses,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        pos_init = s.positions.copy()
        for _ in range(50):
            s.step()

        displacement = np.sqrt(np.sum((s.positions - pos_init) ** 2))
        assert displacement > 0.01, "Particles should have moved"

    def test_radial_infall(self):
        """Two particles starting at rest should fall toward each other."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        domain = 50.0
        center = domain / 2
        sep = 4.0
        positions = np.array([[center - sep / 2, center], [center + sep / 2, center]])
        masses = np.ones(2)

        s = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions.copy(),
            initial_velocities=np.zeros((2, 2)),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=(64, 64),
            domain_size=domain,
            operators=ops,
            n_particles=2,
            force_law="gradient",
            dt=0.01,
            source_coupling=masses,
            force_coupling=1.0,
            periodic_boundaries=False,
        )

        for _ in range(50):
            s.step()

        final_sep = np.sqrt(np.sum((s.positions[1] - s.positions[0]) ** 2))
        assert final_sep < sep, "Particles should fall toward each other"

    def test_no_force_no_acceleration(self):
        """With zero source, particles should move in straight lines."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        positions = np.array([[25.0, 25.0], [30.0, 30.0]])
        velocities = np.array([[0.1, 0.2], [-0.1, 0.0]])

        s = FieldSampler(
            particle_inertia=np.ones(2),
            particle_source=np.ones(2),
            particle_force=np.ones(2),
            initial_positions=positions.copy(),
            initial_velocities=velocities.copy(),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=(64, 64),
            domain_size=50.0,
            operators=ops,
            n_particles=2,
            force_law="gradient",
            dt=0.01,
            source_coupling=np.zeros(2),
            force_coupling=1.0,  # zero sources
            periodic_boundaries=False,
        )

        n_steps = 30
        for _ in range(n_steps):
            s.step()

        # Should be at positions + velocities * t (straight line)
        expected = positions + velocities * n_steps * s.dt
        assert np.allclose(
            s.positions, expected, atol=1e-6
        ), "Free particles should travel in straight lines"
