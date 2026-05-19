"""Direct N-body simulator for the PhysicsSchool environment.

Designed as a drop-in alternative to ``FieldSampler`` for force laws that don't
correspond to the Green's function of a linear PDE (running couplings,
non-conservative central forces, etc.).  Instead of solving a field equation on
a grid, this module computes the O(N^2) pairwise force directly from a
user-supplied force law.

The integrator is selectable so that callers can trade accuracy for speed:

============  ===================================  =====================
Name          Description                          Order  /  force evals
============  ===================================  =====================
``euler``     Symplectic Euler (kick-drift)        1st  /  1
``leapfrog``  Velocity Verlet (drift-kick-drift)   2nd  /  1
``yoshida4``  Yoshida 4th-order symplectic         4th  /  3
``yoshida6``  Yoshida 6th-order symplectic         6th  /  7
``rk4``       Classical Runge-Kutta                4th  /  4
``dopri5``    Adaptive Dormand-Prince (jax.odeint) 5(4) /  ~6
============  ===================================  =====================

All integrators (except ``dopri5``) advance the system by a fixed ``dt`` and
are JIT-compiled together with the force law.  ``dopri5`` integrates directly
to a list of recording times using JAX's adaptive Dormand-Prince solver.

The user supplies the force (and optionally potential) law.  Both must be
JAX-traceable callables with the conventions

    ``force_law(r_mag, q_i, q_j, m_i, m_j) -> F_mag``
    ``potential_law(r_mag, q_i, q_j, m_i, m_j) -> V``

where ``q`` are per-particle "charges" / source couplings (signed) and ``m``
are inertias.  The simulator interprets ``F_mag`` along ``r_j - r_i``, so a
*positive* magnitude is an attractive force and repulsive interactions arise
naturally when ``q_i * q_j < 0``.  ``F_mag = dV/dr`` is the pair convention
for the potential.

A standard library of force / potential laws lives in
``physchool.worlds.force_laws``; analytic reference solutions used to
benchmark the simulator live in ``physchool.worlds.analytic_orbits``.
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

# Run all sims in float64
jax.config.update("jax_enable_x64", True)


def _wrap_for_time(fn: Callable, role: str) -> Callable:
    """Return a JAX-traceable callable with signature ``(pos, t)``.

    Accepts the legacy single-argument convention ``fn(pos)`` and wraps it so
    the integrator can always pass time uniformly.  Arity is detected via
    :func:`inspect.signature`; if that fails (e.g. an unusual decorator), the
    function is assumed to follow the new ``(pos, t)`` convention.

    ``role`` is used only in the error message when the function takes more
    than two positional arguments.
    """
    try:
        n_params = len(
            [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        )
    except (ValueError, TypeError):
        n_params = 2

    if n_params == 1:
        return lambda pos, t, _f=fn: _f(pos)
    if n_params == 2:
        return fn
    raise ValueError(
        f"{role} callable must accept either (positions,) or "
        f"(positions, time); got {n_params}-arg signature."
    )


# ── Pairwise force / energy computation ────────────────────────────────────


def _pairwise_displacements(positions, softening):
    """Return r_ij = r_j - r_i, ||r_ij||, and softened ||r_ij|| for all pairs."""
    diff = positions[None, :, :] - positions[:, None, :]  # (n, n, dim)
    r2 = jnp.sum(diff * diff, axis=-1)  # (n, n)
    r_mag = jnp.sqrt(r2)
    r_eff = jnp.sqrt(r2 + softening**2)
    return diff, r_mag, r_eff


def make_acceleration_fn(force_law: Callable, softening: float) -> Callable:
    """Build a JIT-compatible function returning accelerations.

    Returned function has signature
    ``accelerations(positions, source_charges, force_charges, masses)``,
    where each charge / mass argument is a 1-D array of length N.

    Convention (matches ``physchool.worlds.field_sampler``):

    * ``source_charges[j]`` controls how strongly particle j *generates*
      the field — the analogue of ``FieldSampler.source_coupling[j]``.
    * ``force_charges[i]`` controls how strongly particle i *responds*
      to the field gradient — the analogue of ``FieldSampler.particle_force[i]``.

    The pairwise force on i from j is therefore evaluated as
    ``F_mag(r_ij) = force_law(r_ij, q_recv = force_charges[i],
                              q_send = source_charges[j], m_i, m_j)``,
    which makes the force matrix asymmetric in i↔j — exactly what's needed
    for test particles: a ``force_charge = 0`` particle produces no field
    on the other particles' equations of motion, and a ``source_charge = 0``
    particle feels only forces sourced by everyone else.
    """
    soft = float(softening)

    def accelerations(positions, source_charges, force_charges, masses):
        n = positions.shape[0]
        diff, r_mag, r_eff = _pairwise_displacements(positions, soft)
        # Mask out the diagonal (self-interaction) safely.
        eye = jnp.eye(n, dtype=bool)
        r_eff_safe = jnp.where(eye, 1.0, r_eff)
        r_mag_safe = jnp.where(eye, 1.0, r_mag)

        # Receiver on the i-axis, sender on the j-axis:
        #   F[i, j] = force_law(r, q_recv = force_charges[i],
        #                          q_send = source_charges[j], ...)
        Q_recv = force_charges[:, None]
        Q_send = source_charges[None, :]
        M_i, M_j = masses[:, None], masses[None, :]
        F_mag = force_law(r_eff_safe, Q_recv, Q_send, M_i, M_j)
        F_mag = jnp.where(eye, 0.0, F_mag)

        # Direction: attractive force (F_mag > 0) on i pulls toward j,
        # i.e. along +diff/r.  Repulsion handled naturally via signed F_mag.
        F_vec = F_mag[..., None] * diff / r_mag_safe[..., None]
        F_total = jnp.sum(F_vec, axis=1)
        return F_total / masses[:, None]

    return accelerations


def make_potential_fn(potential_law: Callable, softening: float) -> Callable:
    """Build a JIT-compatible total potential energy function (sum over i<j).

    Returned function has signature
    ``potential_energy(positions, source_charges, masses)``.

    Only ``source_charges`` enter here: the field's stored energy is
    determined by what generates the field, not by what responds to it.
    Test particles (``force_charges`` non-zero, ``source_charges = 0``)
    therefore contribute no PE — consistent with the test-particle limit
    in which their KE is *not* exchanged with the field.
    """
    soft = float(softening)

    def potential_energy(positions, source_charges, masses):
        n = positions.shape[0]
        _, _, r_eff = _pairwise_displacements(positions, soft)
        eye = jnp.eye(n, dtype=bool)
        r_eff_safe = jnp.where(eye, 1.0, r_eff)
        Q_i, Q_j = source_charges[:, None], source_charges[None, :]
        M_i, M_j = masses[:, None], masses[None, :]
        V_pair = potential_law(r_eff_safe, Q_i, Q_j, M_i, M_j)
        # Upper triangle only to count each pair once.
        upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
        return jnp.sum(jnp.where(upper, V_pair, 0.0))

    return potential_energy


# ── Symplectic-integrator coefficient sets ─────────────────────────────────

# Yoshida 4th-order (PRSL A 150, 262 (1990)).  drift-kick-drift-kick-drift-kick-drift
# pattern with seven sub-steps (4 drifts, 3 kicks).
_W4 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA4_C = (0.5 * _W4, 0.5 * (1.0 - _W4), 0.5 * (1.0 - _W4), 0.5 * _W4)
_YOSHIDA4_D = (_W4, 1.0 - 2.0 * _W4, _W4)

# Yoshida 6th-order, "solution A" coefficients from H. Yoshida (1990).
_Y6_W = (
    -0.117767998417887e1,
    0.235573213359357e0,
    0.784513610477560e0,
)


def _yoshida6_coefs():
    w1, w2, w3 = _Y6_W
    w0 = 1.0 - 2.0 * (w1 + w2 + w3)
    d = (w3, w2, w1, w0, w1, w2, w3)
    c = []
    c.append(0.5 * d[0])
    for i in range(1, len(d)):
        c.append(0.5 * (d[i - 1] + d[i]))
    c.append(0.5 * d[-1])
    return tuple(c), d


_YOSHIDA6_C, _YOSHIDA6_D = _yoshida6_coefs()


# ── Step functions ─────────────────────────────────────────────────────────


def _make_step(integrator: str, accel_fn: Callable, dt: float):
    """Return ``step(state) -> new_state`` JIT-compiled for the given integrator.

    State is the tuple ``(positions, velocities, time)``; ``time`` is a
    JAX-traced scalar so that time-dependent ``accel_fn`` evaluations stay
    inside the JIT trace.  ``accel_fn`` has signature ``(pos, t) -> a``.

    Higher-order symplectic integrators (Yoshida) advance time in fractional
    sub-steps consistent with their drift coefficients, so a time-modulated
    coupling stays Nth-order accurate on smooth ``G(t)``.
    """

    def euler_step(state):
        pos, vel, t = state
        a = accel_fn(pos, t)
        vel = vel + dt * a  # kick
        pos = pos + dt * vel  # drift
        return (pos, vel, t + dt)

    def leapfrog_step(state):
        pos, vel, t = state
        a = accel_fn(pos, t)
        vel_half = vel + 0.5 * dt * a
        pos = pos + dt * vel_half
        t_new = t + dt
        a_new = accel_fn(pos, t_new)
        vel = vel_half + 0.5 * dt * a_new
        return (pos, vel, t_new)

    def make_yoshida(c_seq, d_seq):
        c_arr = jnp.asarray(c_seq)
        d_arr = jnp.asarray(d_seq)
        n_drift = len(c_seq)
        n_kick = len(d_seq)
        assert n_drift == n_kick + 1

        def yoshida_step(state):
            pos, vel, t = state
            for i in range(n_kick):
                pos = pos + c_arr[i] * dt * vel
                t = t + c_arr[i] * dt
                vel = vel + d_arr[i] * dt * accel_fn(pos, t)
            pos = pos + c_arr[-1] * dt * vel
            t = t + c_arr[-1] * dt
            return (pos, vel, t)

        return yoshida_step

    def rk4_step(state):
        pos, vel, t = state
        # ODE: dpos/dt = vel ; dvel/dt = a(pos, t)
        k1_p, k1_v = vel, accel_fn(pos, t)
        k2_p = vel + 0.5 * dt * k1_v
        k2_v = accel_fn(pos + 0.5 * dt * k1_p, t + 0.5 * dt)
        k3_p = vel + 0.5 * dt * k2_v
        k3_v = accel_fn(pos + 0.5 * dt * k2_p, t + 0.5 * dt)
        k4_p = vel + dt * k3_v
        k4_v = accel_fn(pos + dt * k3_p, t + dt)
        pos = pos + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        vel = vel + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return (pos, vel, t + dt)

    if integrator == "euler":
        step_fn = euler_step
    elif integrator == "leapfrog":
        step_fn = leapfrog_step
    elif integrator == "yoshida4":
        step_fn = make_yoshida(_YOSHIDA4_C, _YOSHIDA4_D)
    elif integrator == "yoshida6":
        step_fn = make_yoshida(_YOSHIDA6_C, _YOSHIDA6_D)
    elif integrator == "rk4":
        step_fn = rk4_step
    elif integrator == "dopri5":
        # Adaptive integrator handled separately in ``run``.
        return None
    else:
        raise ValueError(f"Unknown integrator: {integrator!r}")

    return jax.jit(step_fn)


# ── Public class ───────────────────────────────────────────────────────────


class NBodySampler:
    """Direct O(N^2) N-body simulator with selectable integrator."""

    SUPPORTED_INTEGRATORS = (
        "euler",
        "leapfrog",
        "yoshida4",
        "yoshida6",
        "rk4",
        "dopri5",
    )

    def __init__(
        self,
        masses,
        initial_positions,
        initial_velocities,
        force_law: Callable,
        potential_law: Optional[Callable] = None,
        charges=None,
        source_charges=None,
        force_charges=None,
        integrator: str = "leapfrog",
        dt: float = 0.01,
        softening: float = 0.0,
        spatial_dimensions: Optional[int] = None,
        external_acceleration=None,
        force_modulation: Optional[Callable] = None,
        force_position_modifier: Optional[Callable] = None,
        force_anisotropy=None,
        initial_time: float = 0.0,
    ):
        if integrator not in self.SUPPORTED_INTEGRATORS:
            raise ValueError(
                f"Integrator {integrator!r} not in {self.SUPPORTED_INTEGRATORS}"
            )
        self.integrator = integrator
        self.dt = float(dt)
        self.softening = float(softening)
        self.force_law = force_law
        self.potential_law = potential_law

        self.masses = jnp.asarray(masses, dtype=jnp.float64)
        # Resolve the source / force charge arrays from the user-supplied options.
        # Three calling conventions are supported, in order of precedence:
        #   1. (source_charges, force_charges) — the new API; both required if either
        #      is given.  Use this whenever you want a *FieldSampler-style* split
        #      between "how strongly do I source the field" and "how strongly do I
        #      respond to it" (test particles, multi-species worlds, ...).
        #   2. ``charges`` — legacy alias used for both source and force, matching
        #      the gravity-like Newtonian convention where mass plays both roles.
        #   3. neither — both default to ``masses`` (Newtonian convention).
        if source_charges is not None or force_charges is not None:
            if source_charges is None or force_charges is None:
                raise ValueError(
                    "Pass both `source_charges` and `force_charges`, or neither."
                )
            if charges is not None:
                raise ValueError(
                    "Cannot pass `charges` together with `source_charges` / "
                    "`force_charges`; pick one calling convention."
                )
            self.source_charges = jnp.asarray(source_charges, dtype=jnp.float64)
            self.force_charges = jnp.asarray(force_charges, dtype=jnp.float64)
        elif charges is not None:
            arr = jnp.asarray(charges, dtype=jnp.float64)
            self.source_charges = arr
            self.force_charges = arr
        else:
            self.source_charges = self.masses
            self.force_charges = self.masses

        if (
            self.source_charges.shape != self.masses.shape
            or self.force_charges.shape != self.masses.shape
        ):
            raise ValueError(
                "source_charges, force_charges, and masses must all share the same shape"
            )
        # Backwards-compatible alias on the source array (the role ``charges``
        # historically played for analyses that assumed source = response).
        self.charges = self.source_charges

        self.positions = jnp.asarray(initial_positions, dtype=jnp.float64)
        self.velocities = jnp.asarray(initial_velocities, dtype=jnp.float64)
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.shape[0] != self.masses.shape[0]:
            raise ValueError("masses, positions, velocities must agree on N")
        self.n_particles = self.positions.shape[0]
        self.spatial_dimensions = (
            spatial_dimensions
            if spatial_dimensions is not None
            else self.positions.shape[1]
        )
        self.time = float(initial_time)

        # JIT-compiled helpers closed over the user force/potential laws.
        self._accel_fn = jax.jit(make_acceleration_fn(force_law, softening))
        if potential_law is not None:
            self._potential_fn = jax.jit(make_potential_fn(potential_law, softening))
        else:
            self._potential_fn = None

        # Optional external acceleration applied to every particle each step.
        # Three shapes are accepted:
        #   * a constant ``(D,)`` array — uniform body-force (e.g. the ether
        #     world's northward push), broadcast to every particle.
        #   * a callable ``f(positions) -> (N, D)`` — position-dependent
        #     time-independent body force (e.g. a Hubble flow ``H · r``).
        #   * a callable ``f(positions, time) -> (N, D)`` — fully time- and
        #     position-dependent body force.
        # ``None`` means no external term.  Callables must be JAX-traceable
        # so they JIT alongside the integrator.
        self._external_accel_fn: Optional[Callable] = None
        self.external_acceleration = None
        if external_acceleration is not None:
            if callable(external_acceleration):
                self.external_acceleration = external_acceleration
                self._external_accel_fn = _wrap_for_time(
                    external_acceleration, role="external_acceleration"
                )
            else:
                ext = jnp.asarray(external_acceleration, dtype=jnp.float64)
                if ext.shape != (self.spatial_dimensions,):
                    raise ValueError(
                        f"external_acceleration must have shape "
                        f"({self.spatial_dimensions},), got {ext.shape}"
                    )
                self.external_acceleration = ext
                # Wrap as a (pos, t) -> (N, D) callable for uniform handling.
                self._external_accel_fn = lambda pos, _t, _ext=ext: jnp.broadcast_to(
                    _ext[None, :], pos.shape
                )

        # Optional multiplicative time-modulation on the pairwise force law.
        # ``force_modulation(t) -> scalar`` (or any shape broadcastable
        # against the (N, D) acceleration tensor).  Used by worlds whose
        # underlying coupling oscillates / drifts in time without changing
        # its spatial form (e.g. ``G(t)·F_static(r)``).  Must be JAX-
        # traceable.  ``None`` → constant coupling.
        self.force_modulation = force_modulation
        self._force_modulation_fn = force_modulation

        # Optional per-particle multiplicative spatial modulation.
        # ``force_position_modifier(positions, t) -> (N,)`` returns a
        # scalar multiplier for each particle based on its current
        # position (and optionally time).  Applied *after* the global
        # ``force_modulation`` and *before* ``external_acceleration``.
        # Used by worlds whose coupling constant varies across space.
        # Must be JAX-traceable.
        # ``None`` → no spatial modulation.
        self.force_position_modifier = force_position_modifier
        self._force_position_modifier_fn = force_position_modifier

        # Optional per-component anisotropic multiplier on pairwise accel.
        # ``force_anisotropy`` is a ``(D,)`` array that scales each spatial
        # component independently — e.g. ``[2.0, 1.0]`` doubles the
        # x-component of the pairwise force while leaving y unchanged.
        # Applied *before* any modulation / position modifier / external
        # term.  ``None`` → isotropic (no scaling).
        if force_anisotropy is not None:
            self.force_anisotropy = jnp.asarray(force_anisotropy, dtype=jnp.float64)
            if self.force_anisotropy.shape != (self.spatial_dimensions,):
                raise ValueError(
                    f"force_anisotropy must have shape ({self.spatial_dimensions},), "
                    f"got {self.force_anisotropy.shape}"
                )
        else:
            self.force_anisotropy = None

        # Step functions assume charges and masses are fixed, so close over
        # them.  The compiled accel takes ``(pos, t)`` even when neither
        # the modulation nor the external term actually depend on ``t`` —
        # this keeps the integrator state shape uniform across worlds and
        # lets JAX trace away the no-op time argument.
        src_fixed = self.source_charges
        frc_fixed = self.force_charges
        masses_fixed = self.masses
        ext_fn = self._external_accel_fn
        mod_fn = self._force_modulation_fn
        pos_mod_fn = self._force_position_modifier_fn
        aniso = self.force_anisotropy

        def _accel_pos_t(pos, t):
            a = self._accel_fn(pos, src_fixed, frc_fixed, masses_fixed)
            if aniso is not None:
                a = a * aniso[None, :]
            if mod_fn is not None:
                a = mod_fn(t) * a
            if pos_mod_fn is not None:
                a = pos_mod_fn(pos, t)[:, None] * a
            if ext_fn is not None:
                a = a + ext_fn(pos, t)
            return a

        accel_pos_only = jax.jit(_accel_pos_t)

        if integrator != "dopri5":
            self._step_fn = _make_step(integrator, accel_pos_only, self.dt)
        else:
            self._step_fn = None

    # ── Diagnostics ─────────────────────────────────────────────────────

    def kinetic_energy(self, velocities=None):
        v = self.velocities if velocities is None else velocities
        return 0.5 * jnp.sum(self.masses * jnp.sum(v * v, axis=-1))

    def potential_energy(self, positions=None):
        if self._potential_fn is None:
            raise ValueError("potential_law was not provided")
        p = self.positions if positions is None else positions
        return self._potential_fn(p, self.source_charges, self.masses)

    def total_energy(self, positions=None, velocities=None):
        return self.kinetic_energy(velocities) + self.potential_energy(positions)

    def linear_momentum(self, velocities=None):
        v = self.velocities if velocities is None else velocities
        return jnp.sum(self.masses[:, None] * v, axis=0)

    def angular_momentum(self, positions=None, velocities=None):
        """Total angular momentum about the origin.

        Returns a scalar in 2D (z-component) and a 3-vector in 3D.
        """
        p = self.positions if positions is None else positions
        v = self.velocities if velocities is None else velocities
        if p.shape[1] == 2:
            # L_z = sum_i m_i (x v_y - y v_x)
            return jnp.sum(self.masses * (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]))
        elif p.shape[1] == 3:
            r_cross_v = jnp.cross(p, v)
            return jnp.sum(self.masses[:, None] * r_cross_v, axis=0)
        else:
            raise ValueError("angular momentum only defined for 2D or 3D")

    def center_of_mass(self, positions=None):
        p = self.positions if positions is None else positions
        return jnp.sum(self.masses[:, None] * p, axis=0) / jnp.sum(self.masses)

    # ── Stepping ────────────────────────────────────────────────────────

    def step(self):
        """Advance the system by ``dt``."""
        if self.integrator == "dopri5":
            raise RuntimeError(
                "Single-step interface not available for adaptive 'dopri5'; "
                "use ``run`` instead."
            )
        t_in = jnp.asarray(self.time, dtype=jnp.float64)
        new_pos, new_vel, new_t = self._step_fn(
            (self.positions, self.velocities, t_in)
        )
        self.positions = new_pos
        self.velocities = new_vel
        self.time = float(new_t)

    def run(self, n_steps: int, record_every: int = 1, t_eval=None):
        """Advance the system and return recorded trajectory.

        Parameters
        ----------
        n_steps : int
            Number of fixed-``dt`` steps for the symplectic / RK4 integrators.
            Ignored for ``dopri5`` if ``t_eval`` is given.
        record_every : int
            For fixed-``dt`` integrators, record every Nth step (plus the
            initial state).
        t_eval : array-like, optional
            For ``dopri5``, the absolute times at which to record.  Must be
            increasing and start at ``self.time``.  When supplied for
            fixed-``dt`` integrators, behavior is unchanged (``t_eval`` is
            ignored — recording still uses ``record_every``).

        Returns
        -------
        dict
            Keys: ``'times'`` (T,), ``'positions'`` (T, N, D), ``'velocities'``
            (T, N, D).
        """
        if self.integrator == "dopri5":
            return self._run_dopri5(n_steps, t_eval=t_eval)

        step_fn = self._step_fn
        rec_n = int(record_every)

        def chunk(state, _):
            state = jax.lax.fori_loop(0, rec_n, lambda _i, s: step_fn(s), state)
            return state, state

        n_records = n_steps // record_every
        t0 = jnp.asarray(self.time, dtype=jnp.float64)
        state0 = (self.positions, self.velocities, t0)
        # Initial state is recorded explicitly so users always see t=t0.
        final_state, recorded = jax.lax.scan(chunk, state0, jnp.arange(n_records))

        positions = jnp.concatenate([self.positions[None], recorded[0]], axis=0)
        velocities = jnp.concatenate([self.velocities[None], recorded[1]], axis=0)
        # ``recorded[2]`` is the post-chunk time at each scan step; prepend t0.
        times = jnp.concatenate([t0[None], recorded[2]], axis=0)

        # Update internal state to the end of the run.
        self.positions = final_state[0]
        self.velocities = final_state[1]
        self.time = float(final_state[2])

        return {
            "times": times,
            "positions": positions,
            "velocities": velocities,
        }

    # ── Adaptive Dopri5 path ────────────────────────────────────────────

    def _run_dopri5(
        self, n_steps: int, t_eval=None, rtol: float = 1e-10, atol: float = 1e-12
    ):
        if t_eval is None:
            t_eval = self.time + jnp.arange(n_steps + 1) * self.dt
        else:
            t_eval = jnp.asarray(t_eval, dtype=jnp.float64)

        n = self.n_particles
        d = self.spatial_dimensions
        accel = self._accel_fn
        src = self.source_charges
        frc = self.force_charges
        masses = self.masses
        ext_fn = self._external_accel_fn
        mod_fn = self._force_modulation_fn
        pos_mod_fn = self._force_position_modifier_fn
        aniso = self.force_anisotropy

        @jax.jit
        def rhs(y, t):
            pos = y[: n * d].reshape((n, d))
            vel = y[n * d :].reshape((n, d))
            a = accel(pos, src, frc, masses)
            if aniso is not None:
                a = a * aniso[None, :]
            if mod_fn is not None:
                a = mod_fn(t) * a
            if pos_mod_fn is not None:
                a = pos_mod_fn(pos, t)[:, None] * a
            if ext_fn is not None:
                a = a + ext_fn(pos, t)
            return jnp.concatenate([vel.reshape(-1), a.reshape(-1)])

        y0 = jnp.concatenate([self.positions.reshape(-1), self.velocities.reshape(-1)])

        ys = odeint(rhs, y0, t_eval, rtol=rtol, atol=atol, mxstep=50_000)
        positions = ys[:, : n * d].reshape((-1, n, d))
        velocities = ys[:, n * d :].reshape((-1, n, d))

        self.positions = positions[-1]
        self.velocities = velocities[-1]
        self.time = float(t_eval[-1])

        return {
            "times": t_eval,
            "positions": positions,
            "velocities": velocities,
        }
