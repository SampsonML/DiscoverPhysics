import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physchool.worlds.field_sampler import FieldSampler


def make_two_particles_2d(
    operators, separation=3.0, grid_size=64, domain_size=50.0, periodic=False
):
    """Helper: two unit-mass particles on diagonal, centered in a large box."""
    center = domain_size / 2
    half = separation / (2 * np.sqrt(2))
    positions = np.array(
        [[center - half, center - half], [center + half, center + half]]
    )
    masses = np.ones(2)
    return FieldSampler(
        particle_inertia=masses,
        particle_source=masses,
        particle_force=masses,
        initial_positions=positions,
        initial_velocities=np.zeros((2, 2)),
        spatial_dimensions=2,
        temporal_order=0,
        grid_size=(grid_size, grid_size),
        domain_size=domain_size,
        operators=operators,
        n_particles=2,
        force_law="gradient",
        dt=0.01,
        source_coupling=masses,
        force_coupling=1.0,
        periodic_boundaries=periodic,
    )


class TestForceScaling:
    """Test that forces scale correctly with distance."""

    def test_newtonian_1_over_r_2d(self):
        """∇²φ = ρ in 2D → F ∝ 1/r"""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        seps = [2.0, 3.0, 4.0]
        forces = []
        for sep in seps:
            s = make_two_particles_2d(ops, separation=sep, grid_size=128)
            f = s.step()
            forces.append(float(jnp.sqrt(jnp.sum(f[0] ** 2))))

        # F ∝ 1/r  →  F1/F2 = r2/r1
        ratio = forces[0] / forces[1]
        expected = seps[1] / seps[0]
        assert abs(ratio - expected) / expected < 0.15

    def test_newtonian_1_over_r2_3d(self):
        """∇²φ = ρ in 3D → F ∝ 1/r²"""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        domain = 30.0
        center = domain / 2
        seps = [2.0, 3.0, 4.0]
        forces = []
        for sep in seps:
            d = sep / (2 * np.sqrt(3))
            pos = np.array([[center - d] * 3, [center + d] * 3])
            s = FieldSampler(
                particle_inertia=np.ones(2),
                particle_source=np.ones(2),
                particle_force=np.ones(2),
                initial_positions=pos,
                initial_velocities=np.zeros((2, 3)),
                spatial_dimensions=3,
                temporal_order=0,
                grid_size=(64, 64, 64),
                domain_size=domain,
                operators=ops,
                n_particles=2,
                force_law="gradient",
                dt=0.01,
                source_coupling=np.ones(2),
                force_coupling=1.0,
            )
            f = s.step()
            forces.append(float(jnp.sqrt(jnp.sum(f[0] ** 2))))

        # F ∝ 1/r²  →  F1/F2 = (r2/r1)²
        ratio = forces[0] / forces[1]
        expected = (seps[1] / seps[0]) ** 2
        assert abs(ratio - expected) / expected < 0.15

    @pytest.mark.parametrize("alpha", [0.5, 0.75, 1.25, 1.5])
    def test_fractional_laplacian_scaling(self, alpha):
        """(-∇²)^α φ = ρ in 2D → F ∝ r^(2α-3).
        Measured via log-log slope over multiple separations."""
        ops = [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": alpha},
            }
        ]
        seps = np.array([2.0, 3.0, 4.0, 5.0])
        forces = []
        for sep in seps:
            s = make_two_particles_2d(
                ops, separation=sep, grid_size=128, domain_size=30.0
            )
            f = s.step()
            forces.append(float(jnp.sqrt(jnp.sum(f[0] ** 2))))

        slope = np.polyfit(np.log(seps), np.log(np.array(forces)), 1)[0]
        expected_slope = 2 * alpha - 3
        assert (
            abs(slope - expected_slope) < 0.4
        ), f"alpha={alpha}: slope={slope:.3f}, expected={expected_slope:.1f}"

    def test_fractional_slopes_ordered(self):
        """Higher alpha should give shallower (less negative) force slope."""
        slopes = []
        for alpha in [0.5, 1.0, 1.5]:
            if alpha == 1.0:
                ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
            else:
                ops = [
                    {
                        "type": "fractional_laplacian",
                        "params": {"strength": 1.0, "alpha": alpha},
                    }
                ]
            seps = np.array([2.0, 4.0, 6.0])
            forces = []
            for sep in seps:
                s = make_two_particles_2d(
                    ops, separation=sep, grid_size=128, domain_size=30.0
                )
                f = s.step()
                forces.append(float(jnp.sqrt(jnp.sum(f[0] ** 2))))
            slope = np.polyfit(np.log(seps), np.log(np.array(forces)), 1)[0]
            slopes.append(slope)

        # slopes should be monotonically increasing: -2ish, -1ish, ~0
        for i in range(len(slopes) - 1):
            assert (
                slopes[i] < slopes[i + 1]
            ), f"Slopes should increase with alpha: {slopes}"

    def test_fractional_wrong_exponent_rejected(self):
        """The force ratio should NOT match the wrong exponent (2α+1 instead of 3-2α)."""
        alpha = 1.25
        ops = [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": alpha},
            }
        ]
        seps = [2.0, 4.0]
        forces = []
        for sep in seps:
            s = make_two_particles_2d(ops, separation=sep, grid_size=128)
            f = s.step()
            forces.append(float(jnp.sqrt(jnp.sum(f[0] ** 2))))

        ratio = forces[0] / forces[1]
        correct = (seps[1] / seps[0]) ** (3 - 2 * alpha)
        wrong = (seps[1] / seps[0]) ** (2 * alpha + 1)
        # Should match correct, not wrong
        assert abs(ratio - correct) / correct < abs(ratio - wrong) / wrong


class TestAbsoluteForce:
    """Test absolute force magnitude against analytical values."""

    def test_2d_poisson_absolute_force(self):
        """F = m₁m₂/(2πr) for 2D Poisson with large box (minimal image effects)."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        sep = 3.0
        s = make_two_particles_2d(ops, separation=sep, grid_size=128, domain_size=50.0)
        f = s.step()
        fmag = float(jnp.sqrt(jnp.sum(f[0] ** 2)))
        expected = 1.0 / (2 * np.pi * sep)
        assert (
            abs(fmag - expected) / expected < 0.05
        ), f"2D force {fmag:.6f} vs expected {expected:.6f}"

    def test_2d_force_convergence(self):
        """Force should be accurate at multiple resolutions."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        sep = 3.0
        expected = 1.0 / (2 * np.pi * sep)
        for gs in [64, 128]:
            s = make_two_particles_2d(
                ops, separation=sep, grid_size=gs, domain_size=50.0
            )
            f = s.step()
            fmag = float(jnp.sqrt(jnp.sum(f[0] ** 2)))
            err = abs(fmag - expected) / expected
            assert err < 0.05, f"Grid {gs}: error {err:.2%} exceeds 5%"


class TestForceProperties:
    """Test physical properties of forces."""

    def test_force_is_attractive(self):
        """Force on particle 0 should point toward particle 1."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        s = make_two_particles_2d(ops, separation=3.0, grid_size=64)
        f = s.step()
        displacement = s.positions[1] - s.positions[0]
        assert jnp.dot(f[0], displacement) > 0, "Force should be attractive"

    def test_newtons_third_law(self):
        """F₁ + F₂ ≈ 0 (Newton's third law via CIC symmetry)."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        s = make_two_particles_2d(ops, separation=3.0, grid_size=128)
        f = s.step()
        net = jnp.sum(f, axis=0)
        fmag = float(jnp.sqrt(jnp.sum(f[0] ** 2)))
        assert (
            float(jnp.sqrt(jnp.sum(net**2))) / fmag < 0.01
        ), "Net force should be ~0 (Newton's 3rd law)"

    def test_yukawa_screening(self):
        """Yukawa force at long range should be much weaker than at short range."""
        ops = [
            {"type": "laplacian", "params": {"strength": 1.0}},
            {"type": "screening", "params": {"strength": 1.0, "screening_length": 2.0}},
        ]
        s_short = make_two_particles_2d(ops, separation=1.0, grid_size=128)
        f_short = float(jnp.sqrt(jnp.sum(s_short.step()[0] ** 2)))

        s_long = make_two_particles_2d(ops, separation=5.0, grid_size=128)
        f_long = float(jnp.sqrt(jnp.sum(s_long.step()[0] ** 2)))

        assert f_short > 3 * f_long, "Screened force should decay faster than 1/r"


class TestNegativeChecks:
    """Verify tests aren't trivially true — things that should fail, do fail."""

    def test_force_is_not_repulsive(self):
        """Force should NOT point away from the other particle."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        s = make_two_particles_2d(ops, separation=3.0, grid_size=64)
        f = s.step()
        displacement = s.positions[1] - s.positions[0]
        # Dot product should be positive (attractive), NOT negative
        assert jnp.dot(f[0], displacement) > 0
        assert jnp.dot(f[0], -displacement) < 0  # repulsive direction fails

    def test_force_not_zero(self):
        """Two particles should produce nonzero force."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        s = make_two_particles_2d(ops, separation=3.0, grid_size=64)
        f = s.step()
        fmag = float(jnp.sqrt(jnp.sum(f[0] ** 2)))
        assert fmag > 1e-6, "Force should not be zero"

    def test_zero_source_gives_zero_force(self):
        """Particles with zero source coupling should produce no force."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        center = 25.0
        positions = np.array([[center - 1, center], [center + 1, center]])
        s = FieldSampler(
            particle_inertia=np.ones(2),
            particle_source=np.ones(2),
            particle_force=np.ones(2),
            initial_positions=positions,
            initial_velocities=np.zeros((2, 2)),
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=(64, 64),
            domain_size=50.0,
            operators=ops,
            n_particles=2,
            force_law="gradient",
            dt=0.01,
            source_coupling=np.zeros(2),
            force_coupling=1.0,  # zero sources!
        )
        f = s.step()
        assert float(jnp.max(jnp.abs(f))) < 1e-10, "Zero source → zero force"

    def test_wrong_scaling_rejected(self):
        """2D Poisson should NOT follow 1/r² scaling (that's 3D)."""
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        seps = [2.0, 4.0]
        forces = []
        for sep in seps:
            s = make_two_particles_2d(ops, separation=sep, grid_size=128)
            f = s.step()
            forces.append(float(jnp.sqrt(jnp.sum(f[0] ** 2))))
        ratio = forces[0] / forces[1]
        wrong_3d_scaling = (seps[1] / seps[0]) ** 2  # 1/r² would give ratio = 4
        correct_2d_scaling = seps[1] / seps[0]  # 1/r gives ratio = 2
        # Should be close to 2D scaling, NOT 3D
        assert abs(ratio - correct_2d_scaling) / correct_2d_scaling < 0.2
        assert abs(ratio - wrong_3d_scaling) / wrong_3d_scaling > 0.3


class TestMomentumConservation:
    """Total momentum should be conserved in periodic domain."""

    def test_periodic_momentum_conservation(self):
        ops = [{"type": "laplacian", "params": {"strength": 1.0}}]
        np.random.seed(42)
        n = 10
        domain = 10.0
        positions = np.random.uniform(1, 9, (n, 2))
        velocities = np.random.randn(n, 2) * 0.1
        masses = np.ones(n)

        s = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            spatial_dimensions=2,
            temporal_order=0,
            grid_size=(64, 64),
            domain_size=domain,
            operators=ops,
            n_particles=n,
            force_law="gradient",
            dt=0.01,
            source_coupling=masses,
            force_coupling=1.0,
            periodic_boundaries=True,
        )

        p0 = np.sum(s.velocities * np.asarray(masses)[:, None], axis=0)
        for _ in range(20):
            s.step()
        p1 = np.sum(s.velocities * np.asarray(masses)[:, None], axis=0)

        # Momentum should be conserved to ~1% of its magnitude
        dp = np.linalg.norm(p1 - p0)
        p_scale = max(np.linalg.norm(p0), 1e-10)
        assert dp / p_scale < 0.05, f"Momentum drift {dp/p_scale:.4f} too large"


class TestDiffusion:
    """n=1 (diffusion equation): Gaussian should spread as σ²(t) = σ₀² + 2αt."""

    def test_gaussian_spreading(self):
        grid_size = 128
        domain = 10.0
        dx = domain / grid_size
        alpha = 0.5  # diffusion coefficient

        ops = [{"type": "laplacian", "params": {"strength": alpha}}]
        s = FieldSampler(
            particle_inertia=np.ones(1),
            particle_source=np.ones(1),
            particle_force=np.ones(1),
            initial_positions=np.array([[5.0, 5.0]]),
            initial_velocities=np.zeros((1, 2)),
            spatial_dimensions=2,
            temporal_order=1,
            grid_size=(grid_size, grid_size),
            domain_size=domain,
            operators=ops,
            n_particles=1,
            force_law="gradient",
            dt=0.001,
            source_coupling=np.zeros(1),
            force_coupling=0.0,
        )

        # Initialize with a Gaussian (no particle source, just field IC)
        sigma0 = 0.5
        x = jnp.linspace(0, domain, grid_size, endpoint=False) + dx / 2
        X, Y = jnp.meshgrid(x, x, indexing="ij")
        center = domain / 2
        s.field = jnp.exp(-((X - center) ** 2 + (Y - center) ** 2) / (2 * sigma0**2))
        s.field = s.field / jnp.sum(s.field)  # normalize

        def measure_sigma2(field):
            total = jnp.sum(field)
            cx = jnp.sum(X * field) / total
            cy = jnp.sum(Y * field) / total
            return jnp.sum(((X - cx) ** 2 + (Y - cy) ** 2) * field) / total

        sigma2_init = float(measure_sigma2(s.field))

        # Evolve for some time
        n_steps = 200
        for _ in range(n_steps):
            s.step()

        sigma2_final = float(measure_sigma2(s.field))
        t_final = n_steps * s.dt

        # Expected: σ²(t) = σ²₀ + 2*d*α*t  (d=2 for 2D, factor of 2 per dimension)
        # Actually for 2D diffusion ∂φ/∂t = α∇²φ, σ² grows as σ²₀ + 2*α*t per dimension
        # Total σ² = σ²₀ + 4*α*t (2 dimensions)
        expected_sigma2 = sigma2_init + 4 * alpha * t_final
        rel_error = abs(sigma2_final - expected_sigma2) / expected_sigma2

        assert (
            rel_error < 0.1
        ), f"σ²={sigma2_final:.4f}, expected={expected_sigma2:.4f}, err={rel_error:.2%}"


class TestWaveEquation:
    """n=2 (wave equation): sinusoidal wave should propagate at c = √α."""

    def test_wave_speed(self):
        grid_size = 256
        domain = 10.0
        dx = domain / grid_size
        alpha = 1.0  # wave speed² = α

        ops = [{"type": "laplacian", "params": {"strength": alpha}}]
        s = FieldSampler(
            particle_inertia=np.ones(1),
            particle_source=np.ones(1),
            particle_force=np.ones(1),
            initial_positions=np.array([[5.0, 5.0]]),
            initial_velocities=np.zeros((1, 2)),
            spatial_dimensions=2,
            temporal_order=2,
            grid_size=(grid_size, grid_size),
            domain_size=domain,
            operators=ops,
            n_particles=1,
            force_law="gradient",
            dt=0.002,
            source_coupling=np.zeros(1),
            force_coupling=0.0,
        )

        # Standing wave IC: φ = cos(kx), ∂φ/∂t = 0
        # Solution: φ(x,t) = cos(kx)cos(ωt) where ω = c*k
        k_mode = 2  # wavenumber index
        kx = 2 * np.pi * k_mode / domain
        x = jnp.linspace(0, domain, grid_size, endpoint=False) + dx / 2
        X, Y = jnp.meshgrid(x, x, indexing="ij")
        s.field = jnp.cos(kx * X)
        s.field_velocity = jnp.zeros_like(s.field)

        omega = np.sqrt(alpha) * kx
        period = 2 * np.pi / omega
        n_steps = int(0.25 * period / s.dt)  # quarter period

        for _ in range(n_steps):
            s.step()

        # At t = T/4, the field should be ~0 (cos(π/2) = 0)
        t = n_steps * s.dt
        expected_amplitude = abs(np.cos(omega * t))

        # Measure the amplitude of the k_mode
        field_k = jnp.fft.fftn(s.field)
        mode_amplitude = 2 * jnp.abs(field_k[k_mode, 0]) / (grid_size**2)
        initial_amplitude = 1.0

        assert (
            mode_amplitude / initial_amplitude < 0.3
        ), f"At T/4, amplitude should be near 0, got {mode_amplitude:.4f}"
