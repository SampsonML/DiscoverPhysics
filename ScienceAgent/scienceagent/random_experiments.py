"""
Random experiment samplers for the "random-experiments" benchmark mode.

In this mode the LLM does NOT propose experiments. The benchmark harness
draws one experiment per round uniformly from the documented parameter
ranges (see PhysicsSchool/prompts/run_experiments*.md) and the agent is
asked to analyse the resulting trajectory data before submitting a
<final_law>.

Each world has its own sampler because the executor contract differs
(two-particle worlds take p1/p2/pos2/velocity2, probe worlds take
probe_positions/probe_velocities, etc.).
"""

import numpy as np
from typing import Callable

# Fixed 10-point measurement grid over the first 5 s of every run.
# Matches the default `measurement_times` shown in the example requests
# inside the prompt files.
_DEFAULT_TIMES = [round(0.5 * i, 2) for i in range(1, 11)]


def _round_list(arr, decimals=4):
    return [round(float(x), decimals) for x in np.asarray(arr).ravel()]


def _round_pairs(arr, decimals=4):
    return [[round(float(x), decimals), round(float(y), decimals)] for x, y in arr]


def sample_two_particle(rng: np.random.Generator) -> dict:
    """gravity / yukawa / fractional / diffusion / wave

    Rejection-samples pos2 until it is ≥ 0.5 away from the origin so the
    mobile particle never starts on top of the fixed source particle.
    """
    for _ in range(50):
        pos2 = rng.uniform(-10.0, 10.0, 2)
        if np.linalg.norm(pos2) >= 0.5:
            break
    return {
        "p1": round(float(rng.uniform(0.1, 10.0)), 4),
        "p2": round(float(rng.uniform(0.1, 10.0)), 4),
        "pos2": _round_list(pos2),
        "velocity2": _round_list(rng.uniform(-5.0, 5.0, 2)),
        "measurement_times": list(_DEFAULT_TIMES),
    }


def _sample_probes(rng: np.random.Generator, n: int = 5) -> dict:
    return {
        "probe_positions": _round_pairs(rng.uniform(-10.0, 10.0, (n, 2))),
        "probe_velocities": _round_pairs(rng.uniform(-5.0, 5.0, (n, 2))),
        "measurement_times": list(_DEFAULT_TIMES),
    }


def _sample_probes_with_masses(rng: np.random.Generator, n: int = 5) -> dict:
    # Probe masses drawn from the same {1, 2, 4} class set the ether/hubble
    # worlds use for their orbiters — exposing the mass dependence is the whole
    # point of these worlds, so a fixed default mass would mask it.
    mass_choices = np.array([1.0, 2.0, 4.0])
    return {
        "probe_positions": _round_pairs(rng.uniform(-10.0, 10.0, (n, 2))),
        "probe_velocities": _round_pairs(rng.uniform(-5.0, 5.0, (n, 2))),
        "probe_masses": _round_list(rng.choice(mass_choices, size=n)),
        "measurement_times": list(_DEFAULT_TIMES),
    }


def sample_dark_matter(rng: np.random.Generator) -> dict:
    return _sample_probes(rng, n=5)


def sample_three_species(rng: np.random.Generator) -> dict:
    return _sample_probes(rng, n=5)


def sample_ether(rng: np.random.Generator) -> dict:
    return _sample_probes_with_masses(rng, n=5)


def sample_hubble(rng: np.random.Generator) -> dict:
    return _sample_probes_with_masses(rng, n=5)


def sample_coulomb_hard(rng: np.random.Generator, n: int = 10) -> dict:
    """coulomb_hard: 10 mobile particles, signed charges in [-3, 3]."""
    return {
        "positions": _round_pairs(rng.uniform(-10.0, 10.0, (n, 2))),
        "velocities": _round_pairs(rng.uniform(-3.0, 3.0, (n, 2))),
        "charges": _round_list(rng.uniform(-3.0, 3.0, n)),
        "measurement_times": list(_DEFAULT_TIMES),
    }


def sample_species(rng: np.random.Generator) -> dict:
    """species world: 6 particles, agent controls all initial conditions."""
    return {
        "positions": _round_pairs(rng.uniform(-10.0, 10.0, (6, 2))),
        "velocities": _round_pairs(rng.uniform(-5.0, 5.0, (6, 2))),
        "measurement_times": list(_DEFAULT_TIMES),
    }


def sample_circle(rng: np.random.Generator) -> dict:
    """circle world: only ring_radius and initial tangential velocity are free."""
    return {
        "ring_radius": round(float(rng.uniform(2.0, 10.0)), 4),
        "initial_tangential_velocity": round(float(rng.uniform(-3.0, 3.0)), 4),
        "measurement_times": list(_DEFAULT_TIMES),
    }


_SAMPLERS: dict[str, Callable[[np.random.Generator], dict]] = {
    "gravity": sample_two_particle,
    "yukawa": sample_two_particle,
    "fractional": sample_two_particle,
    "diffusion": sample_two_particle,
    "wave": sample_two_particle,
    "oscillator": sample_two_particle,
    "extra_dimensions": sample_two_particle,
    "coulomb_easy": sample_two_particle,
    "coulomb_hard": sample_coulomb_hard,
    "dark_matter": sample_dark_matter,
    "three_species": sample_three_species,
    "ether": sample_ether,
    "hubble": sample_hubble,
    "species": sample_species,
    "circle": sample_circle,
}


def make_random_generator(world: str, seed=None) -> Callable[[], dict]:
    """Return a zero-arg callable yielding successive random experiments.

    Args:
        world: World name; must be a key of the internal sampler registry.
        seed: Optional RNG seed. The returned callable is stateful — each
            invocation advances the shared numpy Generator, so calling it
            `max_rounds` times produces a deterministic sequence.
    """
    if world not in _SAMPLERS:
        raise ValueError(
            f"No random sampler registered for world '{world}'. "
            f"Available: {sorted(_SAMPLERS)}"
        )
    rng = np.random.default_rng(seed)
    sampler = _SAMPLERS[world]

    def generator() -> dict:
        return sampler(rng)

    return generator
