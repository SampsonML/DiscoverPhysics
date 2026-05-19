import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physchool.worlds.field_sampler import FieldSampler


def make_solver(operators, grid_size=64, domain_size=2 * np.pi):
    """Helper: create a constraint solver (n=0) with given operators."""
    return FieldSampler(
        particle_inertia=np.ones(2),
        particle_source=np.ones(2),
        particle_force=np.ones(2),
        initial_positions=np.array([[1.0, 1.0], [3.0, 3.0]]),
        initial_velocities=np.zeros((2, 2)),
        spatial_dimensions=2,
        temporal_order=0,
        grid_size=(grid_size, grid_size),
        domain_size=domain_size,
        operators=operators,
        n_particles=2,
        force_law="gradient",
        dt=0.01,
    )


def grid_coords(sampler):
    """Return (X, Y) meshgrid arrays for the sampler's domain."""
    x = jnp.linspace(0, sampler.domain_size, sampler.grid_size[0], endpoint=False)
    y = jnp.linspace(0, sampler.domain_size, sampler.grid_size[1], endpoint=False)
    return jnp.meshgrid(x, y, indexing="ij")


class TestPoissonSolver:
    """Test solve_field_equation for the constraint case (n=0)."""

    def test_laplacian_sinusoidal(self):
        """∇²φ = sin(kx) → φ = -sin(kx)/k²"""
        s = make_solver([{"type": "laplacian", "params": {"strength": 1.0}}])
        X, Y = grid_coords(s)
        k = 2.0
        sources = jnp.sin(k * X)
        field = s.solve_field_equation(sources)
        expected = -jnp.sin(k * X) / k**2
        assert jnp.allclose(field, expected, atol=1e-6)

    def test_laplacian_wrong_answer_fails(self):
        """Sanity: the solver does NOT give φ = +sin(kx)/k² (wrong sign)."""
        s = make_solver([{"type": "laplacian", "params": {"strength": 1.0}}])
        X, Y = grid_coords(s)
        k = 2.0
        sources = jnp.sin(k * X)
        field = s.solve_field_equation(sources)
        wrong = jnp.sin(k * X) / k**2  # wrong sign
        assert not jnp.allclose(field, wrong, atol=1e-3)

    def test_fractional_laplacian(self):
        """(-∇²)^α φ = sin(kx) → φ = -sin(kx)/k^(2α)"""
        alpha = 0.5
        ops = [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": alpha},
            }
        ]
        s = make_solver(ops)
        X, Y = grid_coords(s)
        k = 2.0
        sources = jnp.sin(k * X)
        field = s.solve_field_equation(sources)
        expected = -sources / k ** (2 * alpha)
        assert jnp.allclose(field, expected, atol=1e-6)

    def test_different_alpha_gives_different_field(self):
        """Different fractional powers should produce different fields."""
        X = None
        fields = []
        for alpha in [0.25, 0.75]:
            ops = [
                {
                    "type": "fractional_laplacian",
                    "params": {"strength": 1.0, "alpha": alpha},
                }
            ]
            s = make_solver(ops)
            if X is None:
                X, Y = grid_coords(s)
            field = s.solve_field_equation(jnp.sin(2.0 * X))
            fields.append(field)
        assert not jnp.allclose(fields[0], fields[1], atol=1e-3)

    def test_screening_zero_mean(self):
        """Screened Poisson solution should have zero mean (DC removed)."""
        ops = [
            {"type": "laplacian", "params": {"strength": 1.0}},
            {"type": "screening", "params": {"strength": 1.0, "screening_length": 1.0}},
        ]
        s = make_solver(ops)
        X, Y = grid_coords(s)
        sources = jnp.exp(-((X - np.pi) ** 2 + (Y - np.pi) ** 2))
        field = s.solve_field_equation(sources)
        assert jnp.abs(jnp.mean(field)) < 1e-6

    def test_screening_more_localized_than_laplacian(self):
        """Yukawa field should decay faster than Poisson field."""
        ops_poisson = [{"type": "laplacian", "params": {"strength": 1.0}}]
        ops_yukawa = [
            {"type": "laplacian", "params": {"strength": 1.0}},
            {"type": "screening", "params": {"strength": 1.0, "screening_length": 0.5}},
        ]
        for ops, label in [(ops_poisson, "poisson"), (ops_yukawa, "yukawa")]:
            s = make_solver(ops)
        X, Y = grid_coords(s)
        sources = jnp.zeros_like(X).at[32, 32].set(1.0)

        f_poisson = make_solver(ops_poisson).solve_field_equation(sources)
        f_yukawa = make_solver(ops_yukawa).solve_field_equation(sources)

        # Yukawa field should be smaller far from source
        far_poisson = jnp.abs(f_poisson[0, 0])
        far_yukawa = jnp.abs(f_yukawa[0, 0])
        assert (
            far_yukawa < far_poisson * 0.5
        ), "Yukawa field should decay faster than Poisson"

    def test_strength_scales_field(self):
        """Doubling the operator strength should halve the field amplitude."""
        X = None
        fields = []
        for strength in [1.0, 2.0]:
            ops = [{"type": "laplacian", "params": {"strength": strength}}]
            s = make_solver(ops)
            if X is None:
                X, Y = grid_coords(s)
            field = s.solve_field_equation(jnp.sin(2.0 * X))
            fields.append(field)
        ratio = jnp.max(jnp.abs(fields[0])) / jnp.max(jnp.abs(fields[1]))
        assert jnp.allclose(ratio, 2.0, rtol=1e-4)
