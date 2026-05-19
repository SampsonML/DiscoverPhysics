"""
Evaluator: compares the agent's discovered_law against ground-truth trajectories.
The saved metric `mean_pos_error` is the mean particle MSE — squared L2 position
error per particle, averaged over (particle, time, case). Pass thresholds are
likewise on squared distance.
"""

import functools
import re
import time
import numpy as np
from typing import Callable, Optional

from scienceagent.executor import (
    SimulationExecutor,
    SpeciesExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
    NBodyEtherExecutor,
    NBodyHubbleExecutor,
)

MAX_FIT_PARAMETERS = 5
FIT_MAXITER = 50
FIT_TIME_BUDGET_S = 180.0  # wall-clock cap on a single scipy.optimize.minimize call

# Wall-clock cap for a single discovered_law call. Stops a pathological law
# (Python for-loop with tiny dt, stiff ODE, infinite loop) from hanging the
# scoring loop or the fit objective indefinitely.
LAW_CALL_TIMEOUT_S = 10.0


# Caps on training-data subsampling for the fit. The agent's discovered_law
# may be a pure-Python integrator that takes seconds per call; without these
# caps scipy.minimize can call it hundreds of times and stall evaluation for
# hours. Used by `_subsample_training` for n-body / multi-trajectory fits.
FIT_MAX_TRAJECTORIES = 4  # cap training trajectories used by the loss
FIT_MAX_TIMES_PER_TRAJ = 5  # cap measurement times within each trajectory


class _FitTimeBudgetExceeded(Exception):
    """Raised inside the optimizer objective when the wall-clock budget is hit."""


class _LawCallTimeout(Exception):
    """Raised when a single discovered_law call exceeds LAW_CALL_TIMEOUT_S."""


def _wrap_with_timeout(fn: Callable, timeout_s: float = LAW_CALL_TIMEOUT_S) -> Callable:
    """
    Wrap `fn` so any single call exceeding `timeout_s` raises _LawCallTimeout.
    Uses SIGALRM (Unix-only, main-thread-only). Falls back to the unwrapped
    function on Windows or off-main-thread, where SIGALRM is unavailable.
    """
    import signal
    import threading

    if not hasattr(signal, "SIGALRM"):
        return fn
    if threading.current_thread() is not threading.main_thread():
        return fn

    def _handler(signum, frame):
        raise _LawCallTimeout(f"discovered_law exceeded {timeout_s:g}s")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    return wrapper


# Default held-out test cases: (p1, p2, pos2, velocity2, measurement_times)
_DEFAULT_TEST_CASES = [
    {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [3.0, 0.0],
        "velocity2": [0.0, 0.5],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        "p1": 2.0,
        "p2": 1.0,
        "pos2": [5.0, 0.0],
        "velocity2": [0.0, 0.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        "p1": 1.0,
        "p2": 2.0,
        "pos2": [-4.0, 2.0],
        "velocity2": [0.3, -0.3],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class Evaluator:
    """
    Evaluates a discovered law function against the simulator.

    Args:
        executor: The same SimulationExecutor used during discovery (same world).
        test_cases: List of experiment dicts. Defaults to _DEFAULT_TEST_CASES.
    """

    def __init__(
        self,
        executor: SimulationExecutor,
        test_cases: list[dict] = None,
    ):
        self.executor = executor
        self.test_cases = test_cases or _DEFAULT_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
    ) -> dict:
        """
        Execute the agent's law and compare against ground truth.

        Args:
            law_source: Python source string containing the `discovered_law` function.
            verbose: If True, print per-case results.
            training_trajectories: Optional list of training samples (as
                produced by `_extract_training_trajectories`). When the law
                source also defines `fit_parameters()`, scipy.optimize tunes
                the declared free parameters against these trajectories
                before the MSE scoring below runs.

        Returns:
            dict with keys:
                mean_pos_error  — mean particle MSE: mean of squared L2 position error
                                  across all (particle, time, case) tuples
                max_pos_error   — max squared L2 position error
                per_case        — list of per-case mean particle MSE
                passed          — bool, True if mean_pos_error < 0.01 (squared-distance threshold)
                fit             — info dict from the parameter fit, or None
        """
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source,
            discovered_law,
            training_trajectories,
            _two_particle_loss,
            verbose,
        )

        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []  # collected for plotting

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_pos1 = np.asarray(gt["pos1"])
            gt_pos2 = np.asarray(gt["pos2"])
            try:
                # Call discovered_law at each measurement time to build a full trajectory
                pred_traj = []
                case_errors = []
                vel = list(case["velocity2"])
                for j, t in enumerate(case["measurement_times"]):
                    p2_out, v2_out = discovered_law(
                        pos1=[0.0, 0.0],
                        pos2=case["pos2"],
                        p1=case["p1"],
                        p2=case["p2"],
                        velocity2=case["velocity2"],
                        duration=t,
                    )
                    p2_out = np.asarray(p2_out)
                    # If the law returns a trajectory array, take the last point
                    if p2_out.ndim == 2:
                        p2_out = p2_out[-1]
                    pred_traj.append(p2_out.tolist())
                    diff = p2_out - np.asarray(gt_pos2[j])
                    err = float(
                        np.dot(diff, diff)
                    )  # squared L2 = particle MSE for this timestep
                    case_errors.append(err)

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                all_errors.extend(case_errors)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "p1": case["p1"],
                        "p2": case["p2"],
                        "gt1": gt_pos1.tolist(),
                        "gt": gt_pos2.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )

                if verbose:
                    print(f"  Case {i+1}: mean_particle_mse = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "p1": case["p1"],
                        "p2": case["p2"],
                        "gt1": gt_pos1.tolist(),
                        "gt": gt_pos2.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.01

        if verbose:
            print(f"\n  Mean particle MSE: {mean_total:.4f}")
            print(f"  Max  particle MSE: {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


# seperate circle world for now as particle amounts are hard-coded into the evals
_CIRCLE_TEST_CASES = [
    # Ring with tangential velocity — tests orbital / spiral dynamics
    {
        "ring_radius": 5.0,
        "initial_tangential_velocity": 0.3,
        "measurement_times": [2.0, 4.0, 6.0, 8.0, 10.0],
    },
]


class CircleEvaluator:
    """
    Evaluates a discovered law for the 11-particle circle world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  — list of 11 [x, y] coords relative to center at t=0
        velocities — list of 11 [vx, vy] at t=0
        duration   — float, time to simulate
        return     — list/array of 11 [x, y] positions at t=duration

    The evaluator calls discovered_law once per measurement time (duration=t),
    computes squared L2 error per particle (particle MSE), and averages across
    all particles, times, and test cases.
    """

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _CIRCLE_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source, discovered_law, training_trajectories, _circle_loss, verbose
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])  # (T, 11, 2)
            gt_velocities = np.asarray(gt["velocities"])  # (T, 11, 2)
            init_pos = gt_positions[0] if len(gt_positions) > 0 else None

            # Initial conditions: positions and velocities at t=0
            # Run the simulator one step from t=0 to get t=0 state
            zero_result = self.executor.run(
                [{**case, "measurement_times": [case["measurement_times"][0]]}]
            )
            # Actually just reconstruct from ring_radius and v_tang
            ring_radius = float(case.get("ring_radius", 5.0))
            v_tang = float(case.get("initial_tangential_velocity", 0.0))
            angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            ring_pos = np.column_stack(
                [ring_radius * np.cos(angles), ring_radius * np.sin(angles)]
            )
            init_positions = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
            ring_vel = np.column_stack(
                [-v_tang * np.sin(angles), v_tang * np.cos(angles)]
            )
            init_velocities = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)  # (11, 2)
                    pred_traj.append(pos_out.tolist())

                    # Per-particle squared L2 error (particle MSE for this timestep)
                    diff = pos_out - gt_positions[j]
                    errs = np.sum(diff * diff, axis=-1)  # (11,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "ring_radius": case["ring_radius"],
                        "v_tang": case["initial_tangential_velocity"],
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(
                        f"  Case {i+1} (r={case['ring_radius']}, v_t={case['initial_tangential_velocity']}): "
                        f"mean_particle_mse = {mean_err:.4f}"
                    )

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "ring_radius": case["ring_radius"],
                        "v_tang": case["initial_tangential_velocity"],
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist() if gt_positions is not None else [],
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = (
            mean_total < 0.25
        )  # squared-distance threshold: 11-particle problem is harder

        if verbose:
            print(f"\n  Mean particle MSE (all particles): {mean_total:.4f}")
            print(f"  Max  particle MSE:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


# this is not working at all right now
_SPECIES_TEST_CASES = [
    # Asymmetric layout: species differences should cause divergent trajectories
    {
        "positions": [[0, 0], [4, 0], [-4, 0], [0, 4], [0, -4], [3, 3]],
        "velocities": [[0, 0], [0, 0.3], [0, -0.3], [0.3, 0], [-0.3, 0], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    # Closer particles — stronger forces reveal species difference faster
    {
        "positions": [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2], [2, 2]],
        "velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
]


class SpeciesEvaluator:
    """
    Evaluates a discovered law for the 6-particle species world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  -- list of 6 [x, y] coords relative to center at t=0
        velocities -- list of 6 [vx, vy] at t=0
        duration   -- float, time to simulate
        return     -- list/array of 6 [x, y] positions at t=duration
    """

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _SPECIES_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])  # (T, 6, 2)
            init_positions = case["positions"]
            init_velocities = case["velocities"]

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)  # (6, 2)
                    pred_traj.append(pos_out.tolist())

                    diff = pos_out - gt_positions[j]
                    errs = np.sum(diff * diff, axis=-1)  # squared L2 per particle (6,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i+1}: mean_particle_mse = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.09

        if verbose:
            print(f"\n  Mean particle MSE (all particles): {mean_total:.4f}")
            print(f"  Max  particle MSE:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


class CoulombHardEvaluator:
    """
    Evaluates a discovered law for the 10-particle signed-Coulomb world.

    Mirrors ``SpeciesEvaluator`` (all particles mobile, scored on every
    final position) but the discovered_law signature carries the
    agent-supplied per-particle signed charges:

        discovered_law(positions, velocities, charges, duration)
            -> 10 final [x, y] positions

    Each test case must provide ``charges`` alongside ``positions`` /
    ``velocities`` / ``measurement_times``; the executor echoes the
    trajectory under ``positions`` (shape (T, 10, 2)).
    """

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or []

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])  # (T, 10, 2)
            init_positions = case["positions"]
            init_velocities = case["velocities"]
            init_charges = case["charges"]

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        charges=init_charges,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)  # (10, 2)
                    pred_traj.append(pos_out.tolist())

                    diff = pos_out - gt_positions[j]
                    errs = np.sum(diff * diff, axis=-1)  # squared L2 per particle
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i+1}: mean_particle_mse = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.09

        if verbose:
            print(f"\n  Mean particle MSE (all particles): {mean_total:.4f}")
            print(f"  Max  particle MSE:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


_THREE_SPECIES_TEST_CASES = [
    {
        "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
        "probe_velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    {
        "probe_positions": [[3, 3], [-3, 3], [-3, -3], [3, -3], [0, 0]],
        "probe_velocities": [[0.2, 0], [0, 0.2], [-0.2, 0], [0, -0.2], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
]


class ThreeSpeciesEvaluator:
    """
    Evaluates a discovered law for the 35-particle three-species world.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  -- list of 35 [x, y] coords relative to center at t=0
        velocities -- list of 35 [vx, vy] at t=0
        duration   -- float, time to simulate
        return     -- list/array of 35 [x, y] positions at t=duration
    """

    def __init__(self, executor: ThreeSpeciesExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _THREE_SPECIES_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])  # (T, 35, 2)
            bg_init = np.asarray(gt["background_initial_positions"])  # (30, 2)

            # Reconstruct initial conditions for all 35 particles
            probe_pos = np.asarray(case["probe_positions"])  # (5, 2)
            probe_vel = np.asarray(case["probe_velocities"])  # (5, 2)
            init_positions = np.vstack([bg_init, probe_pos]).tolist()
            init_velocities = np.vstack(
                [np.zeros((self.executor.N_BACKGROUND, 2)), probe_vel]
            ).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)  # (35, 2)
                    pred_traj.append(pos_out.tolist())

                    diff = pos_out - gt_positions[j]
                    errs = np.sum(diff * diff, axis=-1)  # squared L2 per particle (35,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i+1}: mean_particle_mse = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = (
            mean_total < 0.25
        )  # squared-distance threshold: 35-particle problem is harder

        if verbose:
            print(f"\n  Mean particle MSE (all particles): {mean_total:.4f}")
            print(f"  Max  particle MSE:                 {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


_DARK_MATTER_TEST_CASES = [
    {
        # Probes at large radii with tangential velocities (CCW visible orbits)
        "probe_positions": [[12, 0], [0, 14], [-11, 0], [0, -13], [10, 10]],
        "probe_velocities": [[0, 2.0], [-2.0, 0], [0, -1.5], [2.0, 0], [-1.5, 1.5]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        # Probes at moderate-large radii, CW visible orbits
        "probe_positions": [[9, 5], [-7, 10], [-10, -6], [6, -11], [0, 15]],
        "probe_velocities": [
            [0.5, -2.0],
            [2.0, 0.5],
            [-0.5, 2.0],
            [-2.0, -0.5],
            [2.5, 0],
        ],
        "visible_velocity_sign": -1.0,  # CW orbits
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class DarkMatterEvaluator:
    """
    Evaluates a discovered law for the dark matter world.

    The agent's law operates on 25 particles (20 visible + 5 probes).
    Scoring is based ONLY on the 5 probe particles (agent indices 20-24),
    whose exact initial positions and velocities the agent knows.

    The discovered_law signature is:
        discovered_law(positions, velocities, duration) -> positions_final

    where:
        positions  -- list of 25 [x, y] coords relative to center at t=0
        velocities -- list of 25 [vx, vy] at t=0
        duration   -- float, time to simulate
        return     -- list/array of 25 [x, y] positions at t=duration
    """

    def __init__(self, executor: DarkMatterExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _DARK_MATTER_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)

        # Run through the NORMAL executor (agent-facing, 25-particle output)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        # Also run full simulation for plotting (all 35 + field)
        full_truths = self.executor.run_full(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        n_vis = self.executor.N_VISIBLE  # 20
        # Probe indices in agent-facing output (25 particles: 0-19 visible, 20-24 probes)
        probe_slice = slice(n_vis, n_vis + self.executor.N_PROBES)  # 20:25

        for i, (case, gt, gt_full) in enumerate(
            zip(self.test_cases, ground_truths, full_truths)
        ):
            gt_positions = np.asarray(gt["positions"])  # (T, 25, 2)
            bg_init = np.asarray(gt["background_initial_positions"])  # (20, 2)

            # Reconstruct agent-visible initial conditions
            vis_vel_sign = float(case.get("visible_velocity_sign", 1.0))
            vis_vel = vis_vel_sign * self.executor._visible_velocities  # (20, 2)
            probe_pos = np.asarray(case["probe_positions"])
            probe_vel = np.asarray(case["probe_velocities"])
            init_positions = np.vstack([bg_init, probe_pos]).tolist()
            init_velocities = np.vstack([vis_vel, probe_vel]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=t,
                    )
                    pos_out = np.asarray(pos_out)  # (25, 2)
                    pred_traj.append(pos_out.tolist())

                    # Score only on the 5 probe particles — per-particle squared L2 (MSE)
                    diff = pos_out[probe_slice] - gt_positions[j, probe_slice]
                    errs = np.sum(diff * diff, axis=-1)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),  # (T, 25, 2) agent-visible
                        "gt_full": gt_full["positions"],  # (T, 35, 2) all particles
                        "field_snapshots": gt_full["field_snapshots"],
                        "dark_initial": gt_full["dark_initial_positions"],
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i+1}: mean_probe_mse = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "gt_full": gt_full["positions"],
                        "field_snapshots": gt_full["field_snapshots"],
                        "dark_initial": gt_full["dark_initial_positions"],
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.25

        if verbose:
            print(f"\n  Mean probe MSE (probes only): {mean_total:.4f}")
            print(f"  Max  probe MSE:               {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


_ETHER_TEST_CASES = [
    # Far-out probes at rest with mixed masses. Central force is weak at this
    # range, so the parabolic ether-drift dominates and dynamics stay regular.
    # Mass mix exercises whether the agent's law treats the drift as
    # mass-dependent (it shouldn't — drift acceleration is uniform).
    {
        "probe_positions": [[15, 0], [0, 18], [-15, 0], [0, -16], [12, 12]],
        "probe_velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "probe_masses": [1.0, 2.0, 4.0, 1.0, 2.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    # Probes with approximately-circular tangential velocities at r=10
    # (v_circ ≈ √(50/2π) ≈ 2.82 in 2D Laplacian gravity), so they orbit
    # cleanly around the anchor while drifting north — tests that the agent
    # captured both the orbital force law and the drift superposition.
    {
        "probe_positions": [[10, 0], [0, 10], [-10, 0], [0, -10], [9, 9]],
        "probe_velocities": [[0, 2.8], [-2.8, 0], [0, -2.8], [2.8, 0], [-2.0, 2.0]],
        "probe_masses": [1.0, 1.0, 1.0, 1.0, 1.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class EtherEvaluator:
    """
    Evaluates a discovered law for the 26-particle ether world.

    The agent's law operates on all 26 particles (1 anchor + 20 ring
    orbiters + 5 probes). Scoring is restricted to the 5 probe particles
    (indices 21-24, i.e. the slice 21:26) because the agent fully controls
    their initial conditions; the ring-orbiter velocities and masses are
    fixed by the world and are reconstructed by this evaluator from the
    executor's stored layout.

    The discovered_law signature is:
        discovered_law(positions, velocities, masses, duration) -> positions_final

    where:
        positions  -- list of 26 [x, y] coords relative to centre at t=0
        velocities -- list of 26 [vx, vy] at t=0
        masses     -- list of 26 per-particle masses (truth values)
        duration   -- float, time to simulate
        return     -- list/array of 26 [x, y] positions at t=duration
    """

    PROBE_SLICE = slice(21, 26)

    def __init__(self, executor: NBodyEtherExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _ETHER_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
        **kwargs,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source,
            discovered_law,
            training_trajectories,
            _ether_loss,
            verbose,
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        bg_pos = np.asarray(self.executor._bg_positions_rel)  # (21, 2)
        bg_vel = np.asarray(self.executor._bg_velocities)  # (21, 2)
        bg_mass = np.asarray(self.executor._bg_masses)  # (21,)

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])  # (T, 26, 2)

            probe_pos = np.asarray(case["probe_positions"])  # (5, 2)
            probe_vel = np.asarray(case["probe_velocities"])  # (5, 2)
            probe_mass = np.asarray(
                case.get(
                    "probe_masses",
                    [self.executor.DEFAULT_PROBE_MASS] * self.executor.N_PROBES,
                ),
                dtype=float,
            )

            init_positions = np.vstack([bg_pos, probe_pos]).tolist()
            init_velocities = np.vstack([bg_vel, probe_vel]).tolist()
            init_masses = np.concatenate([bg_mass, probe_mass]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        masses=init_masses,
                        duration=float(t),
                    )
                    pos_out = np.asarray(pos_out)  # (26, 2)
                    pred_traj.append(pos_out.tolist())

                    diff = pos_out[self.PROBE_SLICE] - gt_positions[j, self.PROBE_SLICE]
                    errs = np.sum(diff * diff, axis=-1)  # squared L2 per probe (5,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i+1}: mean_probe_mse = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.25

        if verbose:
            print(f"\n  Mean probe MSE (probes only): {mean_total:.4f}")
            print(f"  Max  probe MSE:               {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


_HUBBLE_TEST_CASES = [
    # Mixed radii with stable circular tangential velocities inside r_crit
    # so probes orbit cleanly (no close encounters with the anchor), plus
    # outer probes at rest that escape outward under the Hubble flow.
    # Hubble-corrected v_circ = √(Q/(2π) − H·r²):
    #   r=6  → v ≈ 2.48,  r=10 → v ≈ 1.72.
    # Outer probes at r=15 and r=18 are well outside r_crit≈12.6.
    # Mass mix exercises mass-independence of both terms.
    {
        "probe_positions": [[6, 0], [0, 10], [15, 0], [-15, 0], [0, 18]],
        "probe_velocities": [[0, 2.48], [-1.72, 0], [0, 0], [0, 0], [0, 0]],
        "probe_masses": [1.0, 2.0, 4.0, 1.0, 2.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    # Symmetric probes at r=10 in 4 cardinal directions on circular orbits,
    # plus one escape probe at r=16. Tests rotational symmetry of the
    # Hubble flow and the agent's reproduction of the radial structure
    # against a clean orbital benchmark.
    {
        "probe_positions": [[10, 0], [0, 10], [-10, 0], [0, -10], [16, 0]],
        "probe_velocities": [[0, 1.72], [-1.72, 0], [0, -1.72], [1.72, 0], [0, 0]],
        "probe_masses": [1.0, 1.0, 1.0, 1.0, 1.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class HubbleEvaluator:
    """
    Evaluates a discovered law for the 26-particle Hubble-flow world.

    Identical scoring shape to ``EtherEvaluator`` (probe-slice 21:26 of
    the 26-particle output), but the test cases are designed to span a
    range of radii — both inside and outside the critical radius
    ``r_crit ≈ 12.6`` where the Hubble outward push balances central
    inward gravity. A correct law must reproduce both bound orbits at
    small ``r`` and outward escape at large ``r``.

    The discovered_law signature mirrors ether:
        discovered_law(positions, velocities, masses, duration) -> positions_final
    """

    PROBE_SLICE = slice(21, 26)

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _HUBBLE_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
        **kwargs,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source,
            discovered_law,
            training_trajectories,
            _ether_loss,  # same shape: probe slice 21:26 of 26-particle output
            verbose,
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)

        per_case_errors = []
        all_errors = []
        trajectories = []

        bg_pos = np.asarray(self.executor._bg_positions_rel)  # (21, 2)
        bg_vel = np.asarray(self.executor._bg_velocities)  # (21, 2)
        bg_mass = np.asarray(self.executor._bg_masses)  # (21,)

        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])  # (T, 26, 2)

            probe_pos = np.asarray(case["probe_positions"])  # (5, 2)
            probe_vel = np.asarray(case["probe_velocities"])  # (5, 2)
            probe_mass = np.asarray(
                case.get(
                    "probe_masses",
                    [self.executor.DEFAULT_PROBE_MASS] * self.executor.N_PROBES,
                ),
                dtype=float,
            )

            init_positions = np.vstack([bg_pos, probe_pos]).tolist()
            init_velocities = np.vstack([bg_vel, probe_vel]).tolist()
            init_masses = np.concatenate([bg_mass, probe_mass]).tolist()

            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        masses=init_masses,
                        duration=float(t),
                    )
                    pos_out = np.asarray(pos_out)  # (26, 2)
                    pred_traj.append(pos_out.tolist())

                    diff = pos_out[self.PROBE_SLICE] - gt_positions[j, self.PROBE_SLICE]
                    errs = np.sum(diff * diff, axis=-1)  # squared L2 per probe (5,)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())

                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i+1}: mean_probe_mse = {mean_err:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Case {i+1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )

        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        # Looser threshold than ether: outer probes outside r_crit accelerate
        # rapidly, so squared position errors grow large; 1.0 is the prior
        # distance threshold squared (1.0² = 1.0). May need re-tuning under
        # the new MSE metric since mean(||Δ||²) ≥ mean(||Δ||)² (Jensen).
        passed = mean_total < 1.0

        if verbose:
            print(f"\n  Mean probe MSE (probes only): {mean_total:.4f}")
            print(f"  Max  probe MSE:               {max_total:.4f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


# -----------------
# utility functions
def clean_law_source(source: str) -> str:
    """Strip markdown fences and prose before the first code line."""
    import re as _re

    source = _re.sub(r"^```[a-zA-Z]*\n?", "", source.strip(), flags=_re.MULTILINE)
    source = source.replace("```", "")
    lines = source.splitlines()
    code_start = next(
        (
            i
            for i, l in enumerate(lines)
            if l.startswith("def ") or l.startswith("import ") or l.startswith("from ")
        ),
        0,
    )
    return "\n".join(lines[code_start:])


def _compile_law(source: str) -> Callable:
    """Compile and return the discovered_law function from a source string.

    The returned callable is wrapped with a per-call wall-clock timeout so a
    pathological law (tiny-dt for-loop, stiff ODE, infinite loop) cannot hang
    scoring or the fit objective. Every consumer — direct scoring loops and
    loss_fn calls inside the optimizer — inherits the cap automatically.
    """
    source = clean_law_source(source)
    namespace = {}
    exec(compile(source, "<discovered_law>", "exec"), namespace)
    if "discovered_law" not in namespace:
        raise ValueError("Source does not define a function named `discovered_law`")
    return _wrap_with_timeout(namespace["discovered_law"])


def _compile_fit_parameters(source: str) -> Optional[Callable]:
    """
    Compile and return the optional fit_parameters() function if the law
    source defines one. Returns None when absent.
    """
    source = clean_law_source(source)
    namespace = {}
    try:
        exec(compile(source, "<fit_parameters>", "exec"), namespace)
    except Exception:
        return None
    return namespace.get("fit_parameters")


def _extract_training_trajectories(conversation_log: list) -> list:
    """
    Walk a DiscoveryAgent.conversation_log and collect (input_case, output)
    pairs from all successful experiment rounds. Each returned dict has the
    experiment's input keys plus the executor's output arrays.

    Used as the fit set for evaluator-side parameter optimisation: the agent
    already paid the simulator cost during discovery, so we reuse those
    trajectories rather than generating fresh ones.
    """
    training = []
    if not conversation_log:
        return training
    for entry in conversation_log:
        if entry.get("action") != "experiment":
            continue
        inputs = entry.get("experiment_input")
        outputs = entry.get("experiment_output")
        if not inputs or not outputs:
            continue
        for inp, out in zip(inputs, outputs):
            if out is None or not isinstance(out, dict):
                continue
            training.append({"input": inp, "output": out})
    return training


def _subsample_training(
    training: list,
    max_trajectories: int = FIT_MAX_TRAJECTORIES,
    max_times: int = FIT_MAX_TIMES_PER_TRAJ,
) -> list:
    """Down-select training data so that fitting a slow law stays bounded.

    Picks evenly-spaced trajectories, then evenly-spaced measurement times
    within each, deterministically. Aligned per-time arrays in the output
    dict (positions, pos1, pos2, velocities, ...) are sliced to match.
    """
    if not training:
        return training

    n_traj = len(training)
    if n_traj > max_trajectories:
        idx = np.linspace(0, n_traj - 1, max_trajectories).round().astype(int).tolist()
        training = [training[i] for i in idx]

    pruned = []
    for sample in training:
        in_dict = sample.get("input", {}) or {}
        out_in = sample.get("output", {}) or {}
        times = list(
            out_in.get(
                "measurement_times",
                in_dict.get("measurement_times", []),
            )
        )
        if not times or len(times) <= max_times:
            pruned.append(sample)
            continue

        sel = sorted(
            set(np.linspace(0, len(times) - 1, max_times).round().astype(int).tolist())
        )
        out = dict(out_in)
        out["measurement_times"] = [times[i] for i in sel]
        for key in (
            "positions",
            "pos1",
            "pos2",
            "velocities",
            "velocity1",
            "velocity2",
        ):
            arr = out.get(key)
            if arr is None or len(arr) != len(times):
                continue
            out[key] = [arr[i] for i in sel]
        pruned.append({"input": in_dict, "output": out})

    return pruned


def _two_particle_loss(law: Callable, training: list) -> float:
    """Mean-squared position error on 2-particle training trajectories."""
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        obs_pos2 = np.asarray(out.get("pos2", []))
        times = out.get("measurement_times", case.get("measurement_times", []))
        if obs_pos2.ndim != 2 or len(times) == 0:
            continue
        for t, obs in zip(times, obs_pos2):
            pred, _ = law(
                pos1=[0.0, 0.0],
                pos2=case["pos2"],
                p1=case["p1"],
                p2=case["p2"],
                velocity2=case["velocity2"],
                duration=float(t),
            )
            pred = np.asarray(pred)
            if pred.ndim == 2:
                pred = pred[-1]
            diff = pred - np.asarray(obs)
            total_sq += float(np.dot(diff, diff))
            count += 1
    if count == 0:
        return float("inf")
    return total_sq / count


def _circle_loss(law: Callable, training: list) -> float:
    """Mean-squared position error on circle-world training trajectories."""
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        obs_positions = np.asarray(out.get("positions", []))
        times = out.get("measurement_times", case.get("measurement_times", []))
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        ring_radius = float(case.get("ring_radius", 5.0))
        v_tang = float(case.get("initial_tangential_velocity", 0.0))
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        ring_pos = np.column_stack(
            [ring_radius * np.cos(angles), ring_radius * np.sin(angles)]
        )
        init_positions = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
        ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
        init_velocities = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()
        for t, obs in zip(times, obs_positions):
            pred = law(
                positions=init_positions,
                velocities=init_velocities,
                duration=float(t),
            )
            pred = np.asarray(pred)
            if pred.shape != obs.shape:
                return float("inf")
            diff = pred - obs
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


def _three_species_loss(law: Callable, training: list) -> float:
    """Mean-squared position error on three-species training trajectories.

    The discovered law operates on all 35 particles (30 fixed background
    sources of three hidden species + 5 neutral probes). Loss aggregates
    squared distance across every particle, every measurement time, every
    sample — matching ThreeSpeciesEvaluator's all-particles convention.

    Each training sample provides the full 35-particle initial state in
    `input["init_positions"]` and `input["init_velocities"]`; the
    measurement times and observed positions come from `output`. The
    helper `_one_three_species_sample` in mse_fitting.py builds these
    samples from CSV t=0 + measurement rows.
    """
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if init_positions is None or init_velocities is None:
            continue
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=float(t),
                    )
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = pred - obs_positions[j]
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


def _coulomb_hard_loss(law: Callable, training: list) -> float:
    """Mean-squared position error on coulomb_hard training trajectories.

    The discovered law operates on all 10 mobile particles with signature
    ``law(positions, velocities, charges, duration)``. Loss aggregates
    squared distance across every particle, every measurement time.
    Each training sample carries the per-particle ``init_charges`` packed
    by ``mse_fitting._one_coulomb_hard_sample`` from the CSV charge column.
    """
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        init_charges = case.get("init_charges")
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if init_positions is None or init_velocities is None or init_charges is None:
            continue
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(
                        positions=init_positions,
                        velocities=init_velocities,
                        charges=init_charges,
                        duration=float(t),
                    )
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = pred - obs_positions[j]
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


# Probe slice in the dark-matter agent-facing 25-particle output. Mirrors
# DarkMatterEvaluator: scoring is restricted to the 5 probes whose initial
# conditions the agent set, since the visible-particle initial velocities
# are unknown to the agent and dark matter is hidden.
_DARK_MATTER_PROBE_SLICE = slice(20, 25)


def _dark_matter_loss(law: Callable, training: list) -> float:
    """Mean-squared probe-position error on dark-matter training trajectories.

    The discovered law operates on 25 agent-visible particles (20 visible
    + 5 probes). Loss aggregates squared distance across the 5 probe
    particles only, matching DarkMatterEvaluator's probe-only scoring.
    """
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if init_positions is None or init_velocities is None:
            continue
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=float(t),
                    )
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = (
                pred[_DARK_MATTER_PROBE_SLICE]
                - obs_positions[j, _DARK_MATTER_PROBE_SLICE]
            )
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


# Probe slice in the ether agent-facing 26-particle output. Mirrors
# EtherEvaluator: scoring is restricted to the 5 probes (indices 21:26)
# whose initial conditions the agent set, since the orbiter ring is fixed
# by the world.
_ETHER_PROBE_SLICE = slice(21, 26)


def _ether_loss(law: Callable, training: list) -> float:
    """Mean-squared probe-position error on ether-world training trajectories.

    The discovered law operates on all 26 particles
    (1 anchor + 20 ring orbiters + 5 probes) with signature
    ``law(positions, velocities, masses, duration)``.

    Each training sample's ``input`` may carry the full 26-particle initial
    state directly (``init_positions`` / ``init_velocities`` / ``init_masses``
    — the format produced by ``mse_fitting._one_ether_sample`` from CSV
    rows). Otherwise this loss reconstructs the full state from the agent's
    raw experiment input (``probe_positions`` / ``probe_velocities`` /
    optional ``probe_masses``) plus the executor-fixed background that the
    output always carries (``background_initial_positions/_velocities``,
    ``particle_masses``). This dual path lets the same loss function serve
    both the post-discovery ``_maybe_fit`` (conversation-log inputs) and the
    mid-round ``<run_mse_fit>`` (CSV-derived inputs) flows.
    """
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if obs_positions.ndim != 3 or len(times) == 0:
            continue

        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        init_masses = case.get("init_masses")

        if init_positions is None or init_velocities is None or init_masses is None:
            # Conversation-log path: stitch agent probe state onto the
            # executor's fixed background state from the experiment output.
            bg_pos = out.get("background_initial_positions")
            bg_vel = out.get("background_initial_velocities")
            particle_masses = out.get("particle_masses")
            if bg_pos is None or bg_vel is None or particle_masses is None:
                continue
            probe_pos = case.get("probe_positions")
            probe_vel = case.get("probe_velocities")
            if probe_pos is None or probe_vel is None:
                continue
            bg_pos = np.asarray(bg_pos, dtype=float)
            bg_vel = np.asarray(bg_vel, dtype=float)
            init_positions = np.vstack(
                [bg_pos, np.asarray(probe_pos, dtype=float)]
            ).tolist()
            init_velocities = np.vstack(
                [bg_vel, np.asarray(probe_vel, dtype=float)]
            ).tolist()
            init_masses = list(particle_masses)

        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(
                        positions=init_positions,
                        velocities=init_velocities,
                        masses=init_masses,
                        duration=float(t),
                    )
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = pred[_ETHER_PROBE_SLICE] - obs_positions[j, _ETHER_PROBE_SLICE]
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


def _validate_fit_spec(spec) -> list:
    """
    Normalise the user-returned fit_parameters spec into a list of
    (name, init, (lo, hi)) tuples, raising ValueError on malformed input.
    """
    if not isinstance(spec, dict):
        raise ValueError("fit_parameters() must return a dict")
    if len(spec) > MAX_FIT_PARAMETERS:
        raise ValueError(
            f"fit_parameters() declares {len(spec)} parameters; "
            f"max allowed is {MAX_FIT_PARAMETERS}"
        )
    out = []
    for name, entry in spec.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"fit_parameters()['{name}'] must be a dict with 'init' and 'bounds'"
            )
        if "init" not in entry or "bounds" not in entry:
            raise ValueError(
                f"fit_parameters()['{name}'] must provide both 'init' and 'bounds'"
            )
        bounds = entry["bounds"]
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            raise ValueError(
                f"fit_parameters()['{name}']['bounds'] must be a 2-element sequence"
            )
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise ValueError(
                f"fit_parameters()['{name}']: lower bound must be below upper bound"
            )
        init = float(entry["init"])
        if not lo <= init <= hi:
            # Clamp init into the declared bounds rather than erroring.
            init = min(max(init, lo), hi)
        out.append((name, init, (lo, hi)))
    return out


def _fit_law_parameters(
    discovered_law: Callable,
    fit_spec_list: list,
    training: list,
    loss_fn: Callable,
    maxiter: int = FIT_MAXITER,
    time_budget_s: float = FIT_TIME_BUDGET_S,
) -> dict:
    """
    Run scipy.optimize.minimize (L-BFGS-B, bounded) with a wall-clock budget.
    If `time_budget_s` is exceeded mid-fit, returns the best parameters seen
    so far rather than raising — slow agent code never blocks a benchmark cell
    indefinitely. Same code path is used for mid-round and final-eval fitting,
    so the budget covers both.
    """
    if not fit_spec_list:
        return {}
    import time
    from scipy.optimize import minimize

    names = [s[0] for s in fit_spec_list]
    x0 = [s[1] for s in fit_spec_list]
    bounds = [s[2] for s in fit_spec_list]

    state = {
        "best_x": list(x0),
        "best_loss": float("inf"),
        "deadline": time.monotonic() + time_budget_s,
    }

    def _objective(x):
        if time.monotonic() > state["deadline"]:
            raise _FitTimeBudgetExceeded()
        kwargs = dict(zip(names, x.tolist()))
        bound_law = functools.partial(discovered_law, **kwargs)
        try:
            loss = loss_fn(bound_law, training)
        except Exception:
            return 1e12
        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_x"] = list(x)
        return loss

    try:
        result = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        return dict(zip(names, result.x.tolist()))
    except _FitTimeBudgetExceeded:
        print(
            f"  [fit time budget {time_budget_s:.0f}s exceeded; "
            f"using best-so-far params (loss={state['best_loss']:.4g})]"
        )
        return dict(zip(names, state["best_x"]))


def _maybe_fit(
    law_source: str,
    discovered_law: Callable,
    training_trajectories: Optional[list],
    loss_fn: Callable,
    verbose: bool,
) -> tuple:
    """
    If law_source defines fit_parameters() AND training_trajectories is
    non-empty, run scipy.optimize and return (bound_law, fit_info).
    Otherwise return (discovered_law, None).

    fit_info dict keys:
        declared_params  — dict {name: {init, bounds}} before fitting
        fitted_params    — dict {name: value} after fitting (or init if fit failed)
        loss_before      — training-set loss at init values
        loss_after       — training-set loss after fitting
        error            — error string if something went wrong, else None
    """
    fit_fn = _compile_fit_parameters(law_source)
    if fit_fn is None:
        return discovered_law, None
    if not training_trajectories:
        if verbose:
            print("  [fit skipped: no training trajectories available]")
        return discovered_law, {"error": "no_training_trajectories"}

    try:
        raw_spec = fit_fn()
        fit_spec_list = _validate_fit_spec(raw_spec)
    except Exception as e:
        if verbose:
            print(f"  [fit skipped: invalid fit_parameters() — {e}]")
        return discovered_law, {"error": f"invalid_spec: {e}"}

    if not fit_spec_list:
        return discovered_law, None

    declared = {
        name: {"init": init, "bounds": list(bounds)}
        for name, init, bounds in fit_spec_list
    }

    # Cap training data so a slow agent law can't make scipy.minimize stall
    # for hours on hundreds of training samples.
    training = _subsample_training(training_trajectories)
    if verbose and len(training) < len(training_trajectories):
        print(
            f"  [fit using {len(training)}/{len(training_trajectories)} "
            f"training trajectories (cap: {FIT_MAX_TRAJECTORIES} traj × "
            f"{FIT_MAX_TIMES_PER_TRAJ} times)]"
        )

    init_kwargs = {name: init for name, init, _ in fit_spec_list}
    init_law = functools.partial(discovered_law, **init_kwargs)
    try:
        loss_before = loss_fn(init_law, training)
    except Exception:
        loss_before = float("inf")

    fit_t0 = time.monotonic()
    try:
        fitted = _fit_law_parameters(discovered_law, fit_spec_list, training, loss_fn)
    except Exception as e:
        if verbose:
            print(f"  [fit failed: {e}; falling back to init values]")
        return (
            functools.partial(discovered_law, **init_kwargs),
            {
                "declared_params": declared,
                "fitted_params": init_kwargs,
                "loss_before": loss_before,
                "loss_after": loss_before,
                "error": f"optimizer_failure: {e}",
            },
        )
    fit_elapsed = time.monotonic() - fit_t0
    if verbose and fit_elapsed >= FIT_TIME_BUDGET_S - 1.0:
        print(
            f"  [fit hit {FIT_TIME_BUDGET_S:.0f}s wall-clock budget; "
            f"using best-so-far parameters]"
        )

    bound_law = functools.partial(discovered_law, **fitted)
    try:
        loss_after = loss_fn(bound_law, training)
    except Exception:
        loss_after = float("inf")

    if verbose:
        pretty = ", ".join(f"{k}={v:.4g}" for k, v in fitted.items())
        print(f"  Fitted parameters: {pretty}")
        print(f"  Training-set loss: {loss_before:.4g} → {loss_after:.4g}")

    return bound_law, {
        "declared_params": declared,
        "fitted_params": fitted,
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "error": None,
    }


# ---------------------------------------------------------------
# Explanation judge: scores the agent's prose description of the
# physical system against the world's ground-truth optimal_explanation.
# Uses a fixed strong LLM judge (default
# claude-opus-4-6) for reproducibility
# across agent models.  The default is intentionally chosen to be
# disjoint from the models being benchmarked so that no agent grades its
# own explanations — keeping the explanation metric a fair test.

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert physicist grading how well a student's prose description of a "
    "simulated physical system matches the ground-truth description. You are precise, "
    "fair, and reward semantic correctness over surface phrasing — paraphrases and "
    "equivalent formulations (e.g. 'inverse-square-like' ≈ '∇²φ' in 2D) should receive "
    "credit, but missing or wrong physical content should not."
)

_GENERIC_SCORING_GUIDE = """\
10 — captures every essential element correctly, with correct quantitative or relational claims where applicable.
 7–9 — captures the operator and qualitative structure but misses or muddles a quantitative detail or one structural feature.
 4–6 — partially correct: identifies the general physics regime but misses key structural features (e.g. fails to identify multiple species).
 1–3 — incorrect operator or fundamentally wrong physical picture, with only superficial correctness.
   0 — empty, irrelevant, or completely wrong."""


_JUDGE_USER_TEMPLATE = """Compare the student's description against the ground-truth description of the physical system.

<ground_truth>
{ground_truth}
</ground_truth>

<student>
{student}
</student>

Score the student description on a 0–10 integer scale based on how well it captures:
  1. The correct field equation / governing operator (e.g. Laplacian, fractional Laplacian, Helmholtz, diffusion, wave).
  2. The temporal character (static vs. time-evolving; instantaneous vs. retarded).
  3. The force law / coupling structure (how particles couple to the field, including p1/p2 roles).
  4. Any structural features unique to this world: hidden species and their relative coupling strengths and signs, neutral probes, hidden/dark sources, screening lengths, etc.

Use the world-specific rubric below to calibrate the bands. A 10/10 represents the best explanation achievable given the experimental capabilities — reward semantically-equivalent phrasings and numeric estimates within the tolerance specified by the rubric.

<scoring_rubric>
{rubric}
</scoring_rubric>

Respond with 1–3 sentences of justification, then your final integer score inside <score>...</score> tags. Example: "<score>7</score>"."""


class ExplanationJudge:
    """
    LLM-judge-based scorer comparing an agent's prose explanation of a discovered
    physical system against a ground-truth optimal_explanation.

    The judge is independent from the trajectory evaluator and returns a scalar
    score in [0, 1].
    """

    def __init__(
        self,
        judge_model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
    ):
        self.judge_model = judge_model
        self.max_tokens = max_tokens

    def score(
        self,
        agent_explanation: Optional[str],
        optimal_explanation: str,
        rubric: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Args:
            agent_explanation: The agent's prose explanation (may be None or empty).
            optimal_explanation: The ground-truth explanation from the world config.
            rubric: Per-world scoring rubric calibrated to that problem's
                experimental capabilities. If None/empty, falls back to a
                generic 5-band guide.
            verbose: If True, print the score and judge reasoning.

        Returns:
            dict with keys:
                score        — float in [0, 1]
                raw_score    — int in [0, 10] (or None if unparseable)
                reasoning    — full judge reply text
                error        — error message if the judge call failed, else None
        """
        if not agent_explanation or not agent_explanation.strip():
            result = {
                "score": 0.0,
                "raw_score": 0,
                "reasoning": "No <explanation> tag was submitted by the agent.",
                "error": None,
            }
            if verbose:
                print(f"  Explanation score: 0.00  (no explanation submitted)")
            return result

        if not optimal_explanation:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": "No optimal_explanation defined for this world.",
                "error": "missing_ground_truth",
            }
            if verbose:
                print("  Explanation score: skipped (no ground truth defined)")
            return result

        prompt = _JUDGE_USER_TEMPLATE.format(
            ground_truth=optimal_explanation.strip(),
            student=agent_explanation.strip(),
            rubric=(
                rubric.strip() if rubric and rubric.strip() else _GENERIC_SCORING_GUIDE
            ),
        )

        try:
            from scienceagent import llm_client

            reply = llm_client.complete(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                system=_JUDGE_SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": "",
                "error": f"Judge call failed: {e}",
            }
            if verbose:
                print(f"  Explanation score: ERROR — {e}")
            return result

        raw_score = _parse_judge_score(reply)
        if raw_score is None:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": reply,
                "error": "Could not parse <score> tag from judge reply.",
            }
            if verbose:
                print("  Explanation score: ERROR — unparseable judge reply")
            return result

        score = max(0.0, min(1.0, raw_score / 10.0))
        result = {
            "score": float(score),
            "raw_score": int(raw_score),
            "reasoning": reply,
            "error": None,
        }
        if verbose:
            print(f"  Explanation score: {score:.2f}  (raw {raw_score}/10)")
            print(f"  Judge reasoning: {reply.strip()}")
        return result


def _parse_judge_score(reply: str) -> Optional[float]:
    """Extract a 0–10 score from a judge reply.

    Tries a cascade of formats in order of strictness so that a judge
    model which ignores the strict ``<score>...</score>`` instruction
    but still emits a clear score in prose still gets credit:

      1. ``<score>N</score>``                        — the prompted format
      2. ``<score>N/10</score>``                     — common tag variant
      3. ``Score: N`` / ``**Score:** N`` / ``Final score: N`` (optional ``/10``)

    Returns ``None`` if no recognizable form parses to a value in [0, 10].
    """
    if not reply:
        return None

    patterns = (
        # 1. Strict tagged form (the prompted output).
        r"<score>\s*(\d+(?:\.\d+)?)\s*</score>",
        # 2. Tagged with a ``/10`` suffix some models append.
        r"<score>\s*(\d+(?:\.\d+)?)\s*/\s*10\s*</score>",
        # 3. Prose "Score: N", "**Score:** N", "Final Score: N",
        #    optionally followed by "/10".  Allows surrounding markdown.
        r"(?:final\s+)?score\s*:?\s*\*{0,2}\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?\b",
    )

    for pat in patterns:
        match = re.search(pat, reply, re.IGNORECASE)
        if not match:
            continue
        try:
            val = float(match.group(1))
        except ValueError:
            continue
        if 0.0 <= val <= 10.0:
            return val

    return None
