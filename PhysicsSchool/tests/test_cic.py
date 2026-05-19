from physchool.worlds.utils import cic_paint, cic_read
import jax.numpy as jnp
import jax


class TestCICPainting:
    def test_periodic_boundaries(self):
        """Test particle near boundary with periodic BC"""
        positions = jnp.array([[9.8, 9.9]])
        values = jnp.array([1.0])
        grid = cic_paint(positions, values, (10, 10), 10.0, periodic=True)

        # Should wrap around to beginning of grid
        assert grid[0, 0] > 0  # Wrapped contribution
        assert grid[9, 9] > 0  # Main contribution
        assert jnp.allclose(jnp.sum(grid), 1.0)

    def test_non_periodic_boundaries(self):
        """Test particle near boundary with non-periodic BC"""
        positions = jnp.array([[9.8, 9.9]])
        values = jnp.array([1.0])
        grid = cic_paint(positions, values, (10, 10), 10.0, periodic=False)

        # Should NOT wrap around
        assert jnp.allclose(grid[0, 0], 0.0)  # No wrapped contribution
        assert grid[9, 9] > 0  # Main contribution

        # Total mass might be less due to clipping at boundary
        total = jnp.sum(grid)
        assert total <= 1.0  # Some mass lost at boundary
        assert total > 0.0  # But some mass should be captured

    def test_particle_outside_domain_non_periodic(self):
        """Test particle outside domain with non-periodic BC"""
        # Particle completely outside domain
        positions = jnp.array([[-1.0, 5.0]])
        values = jnp.array([1.0])
        grid = cic_paint(positions, values, (10, 10), 10.0, periodic=False)

        # Should contribute nothing to grid
        assert jnp.allclose(jnp.sum(grid), 0.0)

        # Particle partially outside domain
        positions = jnp.array([[0.1, 5.0]])
        values = jnp.array([1.0])
        grid = cic_paint(positions, values, (10, 10), 10.0, periodic=False)
        print("grid = ", grid)

        # Should have partial contribution
        assert grid[0, 5] > 0
        # assert jnp.sum(grid) < 1.0  # Some mass lost

    def test_mass_conservation_both_bc(self):
        """Test mass conservation for both boundary conditions"""
        key = jax.random.PRNGKey(0)
        n_particles = 100

        for n_dims in [2, 3]:
            pos_key, val_key = jax.random.split(key)
            # Keep particles well within domain
            positions = jax.random.uniform(
                pos_key, (n_particles, n_dims), minval=1.0, maxval=9.0
            )
            values = jax.random.uniform(val_key, (n_particles,), minval=-1, maxval=1)

            grid_shape = (32,) * n_dims

            # Both should conserve mass for interior particles
            for periodic in [True, False]:
                grid = cic_paint(positions, values, grid_shape, 10.0, periodic=periodic)

                total_input = jnp.sum(values)
                total_output = jnp.sum(grid) * (10.0 / 32) ** n_dims

                assert jnp.allclose(total_input, total_output)

    def test_gradient_smoothness_both_bc(self):
        """Test that CIC produces smooth, differentiable fields for both BCs"""
        for periodic in [True, False]:

            def paint_at_position(x):
                positions = jnp.array([[x, 5.0]])
                values = jnp.array([1.0])
                grid = cic_paint(positions, values, (10, 10), 10.0, periodic=periodic)
                return jnp.sum(grid * grid)

            # Should be differentiable
            grad_fn = jax.grad(paint_at_position)
            grad = grad_fn(5.0)
            assert jnp.isfinite(grad)


class TestCICReadWrite:
    """Test CIC read/write operations are consistent"""

    def test_read_interpolation(self):
        """Test CIC read correctly interpolates field values"""
        # Create a simple gradient field
        grid = jnp.zeros((10, 10))
        # Set a single point to 1
        grid = grid.at[5, 5].set(1.0)

        # Read at exact grid point
        pos_exact = jnp.array([[5.0, 5.0]])
        value_exact = cic_read(grid, pos_exact, (10, 10), 10.0, periodic=True)
        assert jnp.allclose(value_exact[0], 1.0)

        # Read at offset position - should interpolate
        pos_offset = jnp.array([[5.5, 5.5]])
        value_offset = cic_read(grid, pos_offset, (10, 10), 10.0, periodic=True)
        # Should get 0.25 (equally weighted between 4 cells, only one has value 1)
        assert jnp.allclose(value_offset[0], 0.25, rtol=1e-5)

        # Read further away
        pos_far = jnp.array([[7.0, 5.0]])
        value_far = cic_read(grid, pos_far, (10, 10), 10.0, periodic=True)
        assert jnp.allclose(value_far[0], 0.0)

    def test_read_periodic_boundaries(self):
        """Test CIC read handles periodic boundaries correctly"""
        grid = jnp.zeros((10, 10))
        grid = grid.at[0, 0].set(1.0)

        # Position that wraps around
        positions = jnp.array([[9.8, 9.9]])

        # With periodic BC, should read from wrapped position
        value_periodic = cic_read(grid, positions, (10, 10), 10.0, periodic=True)
        assert value_periodic[0] > 0  # Should get contribution from [0,0]

        # Without periodic BC, should get nothing
        value_nonperiodic = cic_read(grid, positions, (10, 10), 10.0, periodic=False)
        assert jnp.allclose(value_nonperiodic[0], 0.0)

    def test_read_multiple_particles(self):
        """Test reading values for multiple particles at once"""
        # Create a known field
        x = jnp.linspace(0, 10, 10, endpoint=False)
        y = jnp.linspace(0, 10, 10, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        grid = jnp.sin(2 * jnp.pi * X / 10)  # Sinusoidal in x

        # Multiple particle positions
        positions = jnp.array(
            [
                [0.0, 5.0],  # x=0, sin(0) = 0
                [2.5, 5.0],  # x=2.5, sin(π/2) = 1
                [5.0, 5.0],  # x=5, sin(π) = 0
                [7.5, 5.0],  # x=7.5, sin(3π/2) = -1
            ]
        )

        values = cic_read(grid, positions, (10, 10), 10.0, periodic=True)

        # Check approximate values (won't be exact due to interpolation)
        assert jnp.abs(values[0]) < 0.1  # Near 0
        assert values[1] > 0.9  # Near 1
        assert jnp.abs(values[2]) < 0.1  # Near 0
        assert values[3] < -0.9  # Near -1

    def test_3d_read(self):
        """Test 3D CIC reading"""
        # Create 3D grid with a peak
        grid = jnp.zeros((8, 8, 8))
        grid = grid.at[4, 4, 4].set(1.0)

        # Read at exact point
        pos = jnp.array([[4.0, 4.0, 4.0]])
        value = cic_read(grid, pos, (8, 8, 8), 8.0, periodic=True)
        assert jnp.allclose(value[0], 1.0)

        # Read at offset - should interpolate
        pos_offset = jnp.array([[4.5, 4.5, 4.5]])
        value_offset = cic_read(grid, pos_offset, (8, 8, 8), 8.0, periodic=True)
        # Should get 1/8 (one corner out of 8 has value)
        assert jnp.allclose(value_offset[0], 0.125, rtol=1e-5)
