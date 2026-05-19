"""Pairwise force and potential laws for ``NBodySampler``.

Each callable here has the signature

    ``force_law(r_mag, q_i, q_j, m_i, m_j) -> F_mag``
    ``potential_law(r_mag, q_i, q_j, m_i, m_j) -> V``

where ``q_i, q_j`` are per-particle "charges" / source couplings (signed),
``m_i, m_j`` are inertias, and the simulator uses the convention

    Force on particle i from j  =  F_mag * (r_j - r_i) / |r_j - r_i|

so a *positive* ``F_mag`` corresponds to an *attractive* force.  Repulsive
forces fall out automatically when ``q_i * q_j < 0``.  The companion potential
is normalised so that ``F_mag = dV/dr`` (i.e. attractive → V increases with
separation; bound systems sit in the V<0 region for short-ranged kernels).

All implementations use ``jax.numpy`` so they JIT-compile cleanly.  The 2D
Yukawa kernel calls ``scipy.special.k0/k1`` via ``jax.pure_callback`` because
JAX has no native modified-Bessel functions; this still traces through JIT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

# ── 3D-style "Newtonian" gravity (uses charges as gravitational mass) ──────


def gravity_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """Newtonian inverse-square attractive force, ``F = G q_i q_j / r^2``."""
    return G * q_i * q_j / r_mag**2


def gravity_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """Companion potential, ``V = -G q_i q_j / r``."""
    return -G * q_i * q_j / r_mag


def screened_gravity_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 5.0):
    """3D-style exponentially screened inverse-square force.

    ``F = G q_i q_j exp(-r/lam) / r^2``.  Used by the screening
    stress-test; the correct *2D* screened-Poisson kernel lives in
    ``yukawa_2d_force`` and is the one that matches ``FieldSampler``.
    """
    return G * q_i * q_j / r_mag**2 * jnp.exp(-r_mag / lam)


# ── Coulomb (electrostatic) — physics-standard sign convention ────────────

# In real-world electrostatics (and in the codebase's pair convention where
# F_mag > 0 along (r_j - r_i) is attractive), Coulomb's law is
#     F = -k q_i q_j / r²,
# i.e. opposite-sign charges attract (q_i q_j < 0 → F_mag > 0) and like-sign
# charges repel (q_i q_j > 0 → F_mag < 0). This is the *opposite* sign from
# ``gravity_force`` above, which models same-sign-attractive Newtonian
# gravity (where "charges" are masses, always positive).


def coulomb_force(r_mag, q_i, q_j, m_i, m_j, k: float = 1.0):
    """Coulomb 1/r² force, ``F = -k q_i q_j / r²`` (opposite-sign attracts)."""
    return -k * q_i * q_j / r_mag**2


def coulomb_potential(r_mag, q_i, q_j, m_i, m_j, k: float = 1.0):
    """Companion potential, ``V = k q_i q_j / r`` (F_mag = dV/dr)."""
    return k * q_i * q_j / r_mag


def running_coupling_force(r_mag, q_i, q_j, m_i, m_j,
                          alpha0: float = 0.3, beta: float = 0.5,
                          r0: float = 5.0):
    """Running-coupling 1/r force: ``F = g(r) q_i q_j / r``.

    ``g(r) = α₀ / (1 + β ln(r/r₀))`` — one-loop RG-like logarithmic
    running with independent coupling strength (α₀) and running speed
    (β).  At ``r ≪ r₀`` the effective coupling grows; at ``r ≫ r₀`` it
    decays logarithmically.  The 1/r base is the natural 2D force
    (Laplacian Green's function).
    """
    g = alpha0 / (1.0 + beta * jnp.log(r_mag / r0))
    return g * q_i * q_j / r_mag


def running_coupling_potential(r_mag, q_i, q_j, m_i, m_j,
                               alpha0: float = 0.3, beta: float = 0.5,
                               r0: float = 5.0):
    """Potential for the running-coupling 1/r force.
    ``V(r) = -(α₀/β) q_i q_j ln(1 + β ln(r/r₀))``

    Satisfies ``F_mag = -dV/dr``.
    """
    return -(alpha0 / beta) * q_i * q_j * jnp.log(
        1.0 + beta * jnp.log(r_mag / r0)
    )


# ── 2D Poisson (Laplacian Green's function in 2D) ─────────────────────────

# In 2D, the Green's function of ∇² is logarithmic, G(r) = ln(r) / (2π).
# A point source of charge q at the origin produces field
#     φ(r) = q ln(r) / (2π)
# which exerts an attractive force of magnitude q_i q_j / (2π r) on a second
# particle of charge q_j at separation r (for like-signed charges).
# This is the kernel that ``FieldSampler`` with ``operators=[{type:laplacian}]``
# is computing on the grid.

_INV_TWO_PI = 1.0 / (2.0 * jnp.pi)


def poisson_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """2D Poisson force: ``F = G q_i q_j / (2π r)``."""
    return G * q_i * q_j * _INV_TWO_PI / r_mag


def poisson_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    """2D Poisson potential, ``V = G q_i q_j ln(r) / (2π)``.

    The logarithm has no natural ``r → ∞`` reference, so ``V(r=1) = 0`` by
    construction.  Only differences in PE are physically meaningful.
    """
    return G * q_i * q_j * _INV_TWO_PI * jnp.log(r_mag)


# ── 2D Yukawa / screened Poisson (modified Helmholtz) ─────────────────────

# In 2D, the Green's function of (-∇² + 1/λ²) is
#     G(r) = K_0(r/λ) / (2π)
# where K_0 is the modified Bessel function of the second kind.  The force
# magnitude is then |dG/dr| = K_1(r/λ) / (2π λ).  We expose ``scipy.special``
# inside JAX via pure_callback (no native JAX modified-Bessel-K).


def _k0_jax(x):
    """K_0 for jnp arrays via scipy callback (pure → JIT-compatible)."""
    shape_dtype = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(
        lambda xx: np.asarray(scipy.special.k0(xx), dtype=xx.dtype),
        shape_dtype,
        x,
    )


def _k1_jax(x):
    """K_1 for jnp arrays via scipy callback."""
    shape_dtype = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(
        lambda xx: np.asarray(scipy.special.k1(xx), dtype=xx.dtype),
        shape_dtype,
        x,
    )


def yukawa_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 2.0):
    """2D screened Poisson (Yukawa) force.

    ``F = G q_i q_j K_1(r/λ) / (2π λ)``.  At ``r << λ`` this reduces to the
    2D Poisson kernel; at ``r >> λ`` it falls as exp(-r/λ) / sqrt(r λ).
    """
    return G * q_i * q_j * _INV_TWO_PI * _k1_jax(r_mag / lam) / lam


def yukawa_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 2.0):
    """2D Yukawa potential, ``V = -G q_i q_j K_0(r/λ) / (2π)``.

    Vanishes as ``r → ∞`` (unlike the 2D Poisson log potential), so total PE
    is well-defined.
    """
    return -G * q_i * q_j * _INV_TWO_PI * _k0_jax(r_mag / lam)


# ── 2D Riesz / fractional Laplacian Green's function ─────────────────────

# ``FieldSampler`` implements the operator ``-(-∇²)^α`` with Fourier symbol
# ``-|k|^(2α)`` (see ``field_sampler._build_operator_kernel``).  In d=2 the
# Green's function of this operator is the Riesz potential
#     G_α(r) = c_α · r^(2α - 2)
# with the dimensional prefactor
#     c_α = Γ(1 - α) / (2^(2α) π Γ(α)).
# The pairwise force / potential follow from F = -∇φ:
#     F_mag(r) = G · c_α · (2 - 2α) · q_i q_j / r^(3 - 2α)
#     V_pair(r) = -G · c_α · q_i q_j / r^(2 - 2α)
# At α = 1/2 the prefactor c_α = 1/(2π) and these reduce to the (2π-scaled)
# Newtonian inverse-square law that ``FieldSampler`` produces for that
# operator, so the two engines agree up to integrator-precision.


def _riesz_2d_prefactor(alpha: float) -> float:
    """``c_α = Γ(1-α) / (2^(2α) π Γ(α))`` for the 2D Riesz potential."""
    return float(
        scipy.special.gamma(1.0 - alpha)
        / (2.0 ** (2.0 * alpha) * np.pi * scipy.special.gamma(alpha))
    )


def riesz_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, alpha: float = 0.5):
    """2D fractional-Laplacian (Riesz) force, ``F ∝ q_i q_j / r^(3 - 2α)``."""
    c = _riesz_2d_prefactor(alpha)
    return G * c * (2.0 - 2.0 * alpha) * q_i * q_j / r_mag ** (3.0 - 2.0 * alpha)


def riesz_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, alpha: float = 0.5):
    """2D Riesz pair potential, ``V = -G c_α q_i q_j / r^(2 - 2α)``.

    Vanishes as ``r → ∞`` for α < 1.  At α = 1/2: ``c_α = 1/(2π)`` and the
    kernel matches the FieldSampler "fractional_laplacian" output.
    """
    c = _riesz_2d_prefactor(alpha)
    return -G * c * q_i * q_j / r_mag ** (2.0 - 2.0 * alpha)


# ── Kaluza-Klein extra-dimension kernel (2D visible + 1 compact extra) ────
#
# Physics
# -------
# The visible world is 2D, but a single extra spatial dimension is
# compactified on a circle of circumference L = 2π·R.  Both source and
# probe sit at the same point on the compact circle (y = 0), so the
# 3D Newtonian potential of the source generates an infinite tower of
# *images* at y_n = n·L for n ∈ ℤ.  Summing the 3D potential over all
# images and projecting back to the visible 2D slice gives the in-plane
# force on the probe:
#
#     F(r) = (G·L)/(4π) · q_i · q_j · Σ_n  r / (r² + (n·L)²)^(3/2)
#
# The bare image sum has these limits, controlling the experiments the
# agent must run to reveal the extra dimension:
#
#   * r ≪ R   →   only the n=0 term survives:
#                    F → G·L · q_i q_j / (4π r²)   ⟹   3D inverse-square
#                                                       ("Newtonian
#                                                       gravity in 3+1")
#   * r ≫ R   →   the sum becomes an integral 2/(L·r):
#                    F → G · q_i q_j / (2π r)      ⟹   2D Poisson 1/r
#                                                       (looks like the
#                                                       standard 2D
#                                                       "gravity" world)
#
# Choosing the prefactor (G·L)/(4π) makes the long-range coefficient match
# ``poisson_2d_force`` with the same ``G`` exactly, so an agent that only
# explores r ≫ R will infer "this is just 2D gravity".  The 3D inverse-
# square law only emerges if they probe r ≲ R, where the n=0 image
# dominates and the (n·L)² regularisation in the denominator is negligible.
#
# Implementation note
# -------------------
# The infinite tower is truncated to ±``n_images`` images.  Each tail term
# falls off as 1/n³ in the *force* sum, so 20 images (default) keeps the
# truncation error below ~10⁻⁴ relative to the converged value at every r
# of interest.  The companion potential is the antiderivative of the same
# truncated sum; its overall constant depends on ``n_images`` (the bare
# sum is logarithmically divergent), but only differences in V are
# physical and the integrator never relies on its absolute value.


def extra_dimensions_2d_force(
    r_mag,
    q_i,
    q_j,
    m_i,
    m_j,
    G: float = 1.0,
    R_compact: float = 0.5,
    n_images: int = 20,
):
    """Kaluza-Klein image-sum force for 2D + 1 compactified extra dim.

    Long-range limit (r ≫ R): ``F → G q_i q_j / (2π r)`` — matches the
    standard 2D Poisson kernel with the same ``G``.

    Short-range limit (r ≪ R): ``F → G·L q_i q_j / (4π r²)`` with
    ``L = 2π R`` — full 3D Newtonian inverse-square law in the bulk.
    """
    L = 2.0 * jnp.pi * R_compact
    n_arr = jnp.arange(-n_images, n_images + 1)
    y_n = n_arr * L  # (2N+1,)
    r_expanded = r_mag[..., None]  # (..., 1)
    denom = (r_expanded * r_expanded + y_n * y_n) ** 1.5
    F_geom = jnp.sum(r_expanded / denom, axis=-1)
    return G * L * q_i * q_j * F_geom / (4.0 * jnp.pi)


def extra_dimensions_2d_potential(
    r_mag,
    q_i,
    q_j,
    m_i,
    m_j,
    G: float = 1.0,
    R_compact: float = 0.5,
    n_images: int = 20,
):
    """Companion potential for ``extra_dimensions_2d_force``.

    Antiderivative of the truncated image sum:
    ``V = -(G L q_i q_j)/(4π) · Σ_n 1/√(r² + (nL)²)``.

    The bare infinite sum diverges logarithmically; the truncation adds an
    r-independent constant per pair.  Only differences in V are physical,
    and the simulator never relies on V's absolute value (it is used only
    in the optional energy-diagnostic methods).
    """
    L = 2.0 * jnp.pi * R_compact
    n_arr = jnp.arange(-n_images, n_images + 1)
    y_n = n_arr * L
    r_expanded = r_mag[..., None]
    V_geom = -jnp.sum(
        1.0 / jnp.sqrt(r_expanded * r_expanded + y_n * y_n), axis=-1
    )
    return G * L * q_i * q_j * V_geom / (4.0 * jnp.pi)
