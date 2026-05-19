"""
Tests for evaluator-side parameter fitting. The evaluator reuses the
agent's training trajectories; if the submitted law source defines an
optional fit_parameters() function, scipy.optimize tunes the declared
free parameters before MSE scoring.
"""

import numpy as np
import pytest

from scienceagent.evaluator import (
    MAX_FIT_PARAMETERS,
    _compile_fit_parameters,
    _extract_training_trajectories,
    _fit_law_parameters,
    _maybe_fit,
    _two_particle_loss,
    _validate_fit_spec,
)

# -----------------------------------------------------------------------------
# _compile_fit_parameters


def test_compile_fit_parameters_present():
    source = """
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    return pos2, velocity2

def fit_parameters():
    return {"alpha": {"init": 0.5, "bounds": [0.1, 1.0]}}
"""
    fn = _compile_fit_parameters(source)
    assert fn is not None
    assert fn() == {"alpha": {"init": 0.5, "bounds": [0.1, 1.0]}}


def test_compile_fit_parameters_absent():
    source = """
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    return pos2, velocity2
"""
    assert _compile_fit_parameters(source) is None


def test_compile_fit_parameters_syntax_error_returns_none():
    # Broken law source should not raise at compile time; fitting is skipped.
    assert _compile_fit_parameters("def not valid python(") is None


# -----------------------------------------------------------------------------
# _validate_fit_spec


def test_validate_spec_accepts_well_formed():
    spec = _validate_fit_spec(
        {
            "alpha": {"init": 0.5, "bounds": [0.1, 1.0]},
            "D": {"init": 1.0, "bounds": (0.01, 10.0)},
        }
    )
    assert len(spec) == 2
    names = [s[0] for s in spec]
    assert names == ["alpha", "D"]


def test_validate_spec_rejects_too_many_params():
    spec = {
        f"p{i}": {"init": 1.0, "bounds": [0, 2]} for i in range(MAX_FIT_PARAMETERS + 1)
    }
    with pytest.raises(ValueError, match="max allowed"):
        _validate_fit_spec(spec)


def test_validate_spec_rejects_bad_bounds():
    with pytest.raises(ValueError, match="lower bound"):
        _validate_fit_spec({"a": {"init": 1.0, "bounds": [2.0, 1.0]}})


def test_validate_spec_requires_init_and_bounds():
    with pytest.raises(ValueError):
        _validate_fit_spec({"a": {"init": 1.0}})
    with pytest.raises(ValueError):
        _validate_fit_spec({"a": {"bounds": [0, 1]}})


def test_validate_spec_clamps_init_into_bounds():
    # init outside bounds is clamped rather than erroring — easier on the agent.
    spec = _validate_fit_spec({"a": {"init": 100.0, "bounds": [0, 1]}})
    assert spec[0][1] == 1.0


def test_validate_spec_rejects_non_dict():
    with pytest.raises(ValueError):
        _validate_fit_spec("not a dict")


# -----------------------------------------------------------------------------
# _extract_training_trajectories


def test_extract_training_trajectories_filters_non_experiments():
    log = [
        {"action": "warning", "system_message": "ignore me"},
        {
            "action": "experiment",
            "experiment_input": [
                {"p1": 1.0, "p2": 1.0, "pos2": [3, 0], "velocity2": [0, 0]}
            ],
            "experiment_output": [
                {"measurement_times": [1.0], "pos1": [[0, 0]], "pos2": [[2.5, 0]]}
            ],
        },
        {"action": "final_law", "final_law": "..."},
    ]
    out = _extract_training_trajectories(log)
    assert len(out) == 1
    assert out[0]["input"]["p1"] == 1.0
    assert out[0]["output"]["pos2"] == [[2.5, 0]]


def test_extract_training_trajectories_handles_empty_log():
    assert _extract_training_trajectories([]) == []
    assert _extract_training_trajectories(None) == []


# -----------------------------------------------------------------------------
# _fit_law_parameters: end-to-end convergence test on a synthetic law.
# The "true" law is p2_final = pos2 + k * duration * (pos1 - pos2), with k=0.3.
# The agent submits a law with k as a free parameter initialised far from 0.3;
# scipy.optimize should recover k within the maxiter=150 budget.


def _make_training_set(k_true: float) -> list:
    samples = []
    for case in [
        {"p1": 1.0, "p2": 1.0, "pos2": [3.0, 0.0], "velocity2": [0.0, 0.0]},
        {"p1": 1.0, "p2": 1.0, "pos2": [0.0, 4.0], "velocity2": [0.0, 0.0]},
        {"p1": 1.0, "p2": 1.0, "pos2": [-2.0, 2.0], "velocity2": [0.0, 0.0]},
    ]:
        times = [1.0, 2.0, 3.0]
        pos2_traj = []
        p2_start = np.asarray(case["pos2"])
        for t in times:
            p2_t = p2_start + k_true * t * (np.zeros(2) - p2_start)
            pos2_traj.append(p2_t.tolist())
        samples.append(
            {
                "input": case,
                "output": {"measurement_times": times, "pos2": pos2_traj},
            }
        )
    return samples


def _linear_drag_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    k = params.get("k", 1.0)
    p1 = np.asarray(pos1)
    p2_arr = np.asarray(pos2)
    final = p2_arr + k * duration * (p1 - p2_arr)
    return final.tolist(), [0.0, 0.0]


def test_fit_recovers_known_parameter():
    training = _make_training_set(k_true=0.3)
    fitted = _fit_law_parameters(
        _linear_drag_law,
        [("k", 1.0, (0.01, 2.0))],
        training,
        _two_particle_loss,
    )
    assert abs(fitted["k"] - 0.3) < 0.02, f"fit should recover k≈0.3, got {fitted['k']}"


def test_fit_lowers_loss_vs_init():
    training = _make_training_set(k_true=0.3)
    import functools

    init_loss = _two_particle_loss(functools.partial(_linear_drag_law, k=1.0), training)
    fitted = _fit_law_parameters(
        _linear_drag_law,
        [("k", 1.0, (0.01, 2.0))],
        training,
        _two_particle_loss,
    )
    final_loss = _two_particle_loss(
        functools.partial(_linear_drag_law, **fitted), training
    )
    assert final_loss < init_loss / 10  # should be dramatically better


# -----------------------------------------------------------------------------
# _maybe_fit: integration with law source strings


def test_maybe_fit_returns_original_law_when_no_fit_parameters():
    source = """
def discovered_law(pos1, pos2, p1, p2, velocity2, duration):
    return pos2, velocity2
"""
    import functools
    from scienceagent.evaluator import _compile_law

    law = _compile_law(source)
    bound, info = _maybe_fit(
        source, law, training_trajectories=[], loss_fn=_two_particle_loss, verbose=False
    )
    assert bound is law
    assert info is None


def test_maybe_fit_skips_when_no_training_data():
    source = """
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    return pos2, velocity2

def fit_parameters():
    return {"k": {"init": 0.5, "bounds": [0, 2]}}
"""
    from scienceagent.evaluator import _compile_law

    law = _compile_law(source)
    _, info = _maybe_fit(
        source, law, training_trajectories=[], loss_fn=_two_particle_loss, verbose=False
    )
    assert info is not None
    assert info["error"] == "no_training_trajectories"


def test_maybe_fit_rejects_too_many_parameters():
    spec_entries = ", ".join(
        f'"p{i}": {{"init": 1.0, "bounds": [0, 2]}}'
        for i in range(MAX_FIT_PARAMETERS + 1)
    )
    source = f"""
def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    return pos2, velocity2

def fit_parameters():
    return {{{spec_entries}}}
"""
    from scienceagent.evaluator import _compile_law

    law = _compile_law(source)
    training = _make_training_set(0.3)
    bound, info = _maybe_fit(
        source,
        law,
        training_trajectories=training,
        loss_fn=_two_particle_loss,
        verbose=False,
    )
    # Fitting was skipped because spec was invalid.
    assert bound is law
    assert info is not None
    assert "invalid_spec" in info["error"]


def test_maybe_fit_end_to_end_reduces_training_loss():
    source = """
import numpy as np

def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):
    k = params.get("k", 1.0)
    p1 = np.asarray(pos1)
    p2 = np.asarray(pos2)
    final = p2 + k * duration * (p1 - p2)
    return final.tolist(), [0.0, 0.0]

def fit_parameters():
    return {"k": {"init": 1.5, "bounds": [0.01, 2.0]}}
"""
    from scienceagent.evaluator import _compile_law

    law = _compile_law(source)
    training = _make_training_set(0.3)
    _, info = _maybe_fit(
        source,
        law,
        training_trajectories=training,
        loss_fn=_two_particle_loss,
        verbose=False,
    )
    assert info is not None
    assert info["error"] is None
    assert info["loss_after"] < info["loss_before"] / 10
    assert abs(info["fitted_params"]["k"] - 0.3) < 0.02
