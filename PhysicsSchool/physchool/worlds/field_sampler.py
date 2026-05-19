import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from physchool.worlds.utils import cic_paint, cic_read

# in case jax floats are overflowing, set all jax floats to float64
jax.config.update("jax_enable_x64", True)


class FieldSampler:
    """
    Simulated world with one scalar field and particles.

    General form:
        ∂ⁿφ/∂tⁿ = L[φ] + S(particles)

    where:
        n ∈ {0, 1, 2} : temporal derivative order
        L : linear combination of spatial operators (applied in Fourier space)
        S : source term from particles (CIC-painted onto grid)

    Particles feel forces from the field gradient: F = -∇φ.
    """

    @staticmethod
    def recommend_grid_params(
        positions,
        domain_size,
        spatial_dimensions=2,
        min_cells_per_sep=3,
        max_grid_size=1024,
    ):
        """Recommend grid_size and source_smoothing based on particle distribution.

        Ensures the grid resolves inter-particle separations and the source
        smoothing prevents aliasing without over-smoothing forces.

        Returns (grid_size_tuple, source_smoothing).
        """
        from scipy.spatial import cKDTree

        if len(positions) < 2:
            gs = 64
            return tuple([gs] * spatial_dimensions), domain_size / gs

        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        min_sep = np.median(dists[:, 1])

        needed_dx = min_sep / min_cells_per_sep
        gs = int(np.ceil(domain_size / needed_dx))
        gs = max(64, min(max_grid_size, gs))
        # Round up to even number for FFT efficiency
        gs += gs % 2

        dx = domain_size / gs
        smoothing = max(dx, min_sep * 0.3)

        return tuple([gs] * spatial_dimensions), smoothing

    def __init__(
        self,
        particle_inertia,
        particle_source,
        particle_force,
        initial_positions=None,
        initial_velocities=None,
        spatial_dimensions=2,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=10.0,
        operators=None,
        n_particles=100,
        force_law="gradient",
        dt=0.01,
        source_coupling=1.0,
        force_coupling=1.0,
        periodic_boundaries=True,
        source_smoothing=None,
        force_softening=None,
    ):
        self.spatial_dimensions = spatial_dimensions
        self.temporal_order = temporal_order
        self.grid_size = tuple(grid_size)
        self.domain_size = float(domain_size)
        self.dx = domain_size / grid_size[0]
        self.n_particles = n_particles
        self.dt = dt
        self.time = 0.0
        self.periodic_boundaries = periodic_boundaries

        # Particle properties (all arrays)
        self.particle_inertia = jnp.asarray(particle_inertia)
        self.particle_source = jnp.asarray(particle_source)
        self.particle_force = jnp.asarray(particle_force)
        self.source_coupling = jnp.asarray(source_coupling)
        self.force_coupling = float(force_coupling)
        self.force_law = force_law

        # Operators default to Laplacian
        if operators is None:
            self.operators = [{"type": "laplacian", "params": {"strength": 1.0}}]
        else:
            self.operators = operators

        # Source smoothing: Gaussian kernel applied in Fourier space.
        # Prevents grid artifacts for singular Green's functions (e.g. fractional Laplacian).
        # Default: 1*dx. Increase to ~2-3*dx for operators with alpha < 1.
        self.source_smoothing = (
            source_smoothing if source_smoothing is not None else self.dx
        )

        # Force softening length: regularizes the Green's function at short range.
        # Adds epsilon^2 to k^2 in the denominator, equivalent to softening the
        # potential at separations below ~epsilon.  Default: None (no softening).
        self.force_softening = force_softening

        # Precompute k-vectors and operator kernel in Fourier space
        self._kvec = self._build_kvec()
        self._k2 = sum(k**2 for k in self._kvec)
        self._L_k = self._build_operator_kernel()
        self._smoothing_kernel = jnp.exp(-self._k2 * self.source_smoothing**2 / 2)

        # Initialize field(s)
        self.field = jnp.zeros(self.grid_size)
        if self.temporal_order == 2:
            self.field_velocity = jnp.zeros(self.grid_size)

        # Initialize particles
        if initial_positions is None:
            self.positions = (
                np.random.rand(n_particles, spatial_dimensions) * domain_size
            )
        else:
            self.positions = np.asarray(initial_positions, dtype=np.float64)
        if initial_velocities is None:
            self.velocities = np.random.randn(n_particles, spatial_dimensions) * 0.1
        else:
            self.velocities = np.asarray(initial_velocities, dtype=np.float64)

    # ── Precomputation ──────────────────────────────────────────────────

    def _build_kvec(self):
        kvec = []
        for i, s in enumerate(self.grid_size):
            k = jnp.fft.fftfreq(s, self.dx) * 2 * jnp.pi
            shape = [1] * len(self.grid_size)
            shape[i] = s
            kvec.append(k.reshape(shape))
        return kvec

    def _build_operator_kernel(self):
        """Build L(k) once from the operator list. Reused every step."""
        k2 = self._k2
        # Apply force softening: replace k² with k² + (2π/ε)² in the operator.
        # This regularizes the Green's function at separations below ~ε.
        if self.force_softening is not None and self.force_softening > 0:
            eps_k2 = (2 * jnp.pi / self.force_softening) ** 2
            k2_eff = k2 + eps_k2
        else:
            k2_eff = k2

        L_k = jnp.zeros_like(k2)

        for op in self.operators:
            op_type = op["type"]
            params = op["params"]
            strength = params.get("strength", 1.0)

            if op_type == "laplacian":
                # ∇² → -k²
                L_k = L_k + strength * (-k2_eff)
            elif op_type == "fractional_laplacian":
                # (-∇²)^α → -k^(2α)
                alpha = params["alpha"]
                L_k = L_k - strength * jnp.power(k2_eff + 1e-10, alpha)
            elif op_type == "helmholtz":
                # ∇² - m² → -(k² + m²)
                m2 = params["mass_squared"]
                L_k = L_k + strength * (-k2_eff - m2)
            elif op_type == "screening":
                # ∇² - 1/λ² → -(k² + 1/λ²)
                lam = params["screening_length"]
                L_k = L_k + strength * (-k2_eff - 1.0 / lam**2)
            elif op_type == "identity":
                L_k = L_k + strength

        return L_k

    # ── Public API ───────────────────────────────────────────────────────

    def solve_field_equation(self, sources):
        """Solve L[φ] = sources in Fourier space. Returns real-space field.

        No source smoothing applied — use this for analytically constructed sources.
        Particle sources go through _step_constraint which applies smoothing.
        """
        sources_k = jnp.fft.fftn(sources)
        L_k_safe = jnp.where(jnp.abs(self._L_k) < 1e-10, 1.0, self._L_k)
        field_k = sources_k / L_k_safe
        field_k = field_k.at[tuple([0] * self.spatial_dimensions)].set(0.0)
        return jnp.real(jnp.fft.ifftn(field_k))

    # ── Time stepping ───────────────────────────────────────────────────

    def step(self):
        """Advance the system by one timestep. Returns forces on particles."""
        dt = self.dt
        sources = self._paint_sources()

        if self.temporal_order == 0:
            forces = self._step_constraint(sources)
        elif self.temporal_order == 1:
            forces = self._step_diffusion(sources, dt)
        elif self.temporal_order == 2:
            forces = self._step_wave(sources, dt)
        else:
            raise ValueError(
                f"temporal_order must be 0, 1, or 2, got {self.temporal_order}"
            )

        # Update particles (leapfrog-ish: kick-drift)
        accelerations = forces / self.particle_inertia[:, None]
        self.velocities = self.velocities + dt * np.asarray(accelerations)
        self.positions = self.positions + dt * self.velocities

        if self.periodic_boundaries:
            self.positions = np.mod(self.positions, self.domain_size)

        self.time += dt
        return forces

    def _step_constraint(self, sources):
        """
        n=0: Solve L[φ] = S and compute forces in one FFT pass.

        FFT flow: sources → FFT → field_k = S_k / L_k → ik·field_k → IFFT → grad → forces
        Total: 1 FFT + n_dim IFFTs (no redundant transforms).
        """
        sources_k = jnp.fft.fftn(sources) * self._smoothing_kernel

        # Solve: field_k = sources_k / L_k (with zero-mode removed)
        L_k_safe = jnp.where(jnp.abs(self._L_k) < 1e-10, 1.0, self._L_k)
        field_k = sources_k / L_k_safe
        field_k = field_k.at[tuple([0] * self.spatial_dimensions)].set(0.0)

        # Store real-space field for diagnostics
        self.field = jnp.real(jnp.fft.ifftn(field_k))

        # Compute forces directly from field_k (no extra FFT)
        return self._forces_from_field_k(field_k)

    def _step_diffusion(self, sources, dt):
        """
        n=1: ∂φ/∂t = L[φ] + S

        Apply L in Fourier space, step forward in time, then compute forces.
        FFT flow: field → FFT → L_k·field_k → IFFT (for RHS) + ik·field_k → IFFT (for forces)
        We share the single FFT of the field.
        """
        field_k = jnp.fft.fftn(self.field)

        # RHS in Fourier space: L_k * field_k + sources_k
        sources_k = jnp.fft.fftn(sources) * self._smoothing_kernel
        rhs_k = self._L_k * field_k + sources_k

        # Euler step in Fourier space, then back to real space
        new_field_k = field_k + dt * rhs_k
        self.field = jnp.real(jnp.fft.ifftn(new_field_k))

        # Forces from the updated field
        return self._forces_from_field_k(new_field_k)

    def _step_wave(self, sources, dt):
        """
        n=2: ∂²φ/∂t² = L[φ] + S

        Symplectic Euler (kick-drift on the field):
          field_velocity += dt * (L[φ] + S)
          field += dt * field_velocity
        """
        field_k = jnp.fft.fftn(self.field)
        sources_k = jnp.fft.fftn(sources) * self._smoothing_kernel

        # Acceleration of the field
        accel_k = self._L_k * field_k + sources_k
        accel = jnp.real(jnp.fft.ifftn(accel_k))

        # Kick-drift
        self.field_velocity = self.field_velocity + dt * accel
        self.field = self.field + dt * self.field_velocity

        # Forces from updated field
        new_field_k = jnp.fft.fftn(self.field)
        return self._forces_from_field_k(new_field_k)

    # ── Source painting & force computation ──────────────────────────────

    def _paint_sources(self):
        return cic_paint(
            self.positions,
            self.source_coupling,
            self.grid_size,
            self.domain_size,
            periodic=self.periodic_boundaries,
        )

    def _forces_from_field_k(self, field_k):
        """Compute F = -force_coupling * particle_force * ∇φ at particle positions."""
        if self.force_law != "gradient":
            return -self.force_coupling * self.particle_force[:, None] * self.velocities

        forces = []
        for dim in range(self.spatial_dimensions):
            grad_field = jnp.real(jnp.fft.ifftn(1j * self._kvec[dim] * field_k))
            grad_at_particles = cic_read(
                grad_field,
                self.positions,
                self.grid_size,
                self.domain_size,
                periodic=self.periodic_boundaries,
            )
            forces.append(
                -self.force_coupling * self.particle_force * grad_at_particles
            )

        return jnp.stack(forces, axis=-1)
