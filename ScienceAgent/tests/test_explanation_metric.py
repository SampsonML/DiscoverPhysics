"""
Tests for the explanation metric: optimal_explanation fields on each world
and the ExplanationJudge scoring class. The judge LLM call is mocked so
these tests do not require an API key.
"""

from unittest.mock import patch

import pytest

from scienceagent.evaluator import ExplanationJudge, _parse_judge_score
from scienceagent.worlds import WORLDS, get_world


@pytest.mark.parametrize("name", list(WORLDS.keys()))
def test_each_world_defines_optimal_explanation(name):
    entry = WORLDS[name]
    assert "optimal_explanation" in entry, f"{name} missing optimal_explanation"
    text = entry["optimal_explanation"]
    assert (
        isinstance(text, str) and len(text.strip()) > 50
    ), f"{name} optimal_explanation should be a non-trivial prose string"


@pytest.mark.parametrize("name", list(WORLDS.keys()))
def test_each_world_defines_explanation_rubric(name):
    entry = WORLDS[name]
    assert "explanation_rubric" in entry, f"{name} missing explanation_rubric"
    text = entry["explanation_rubric"]
    assert (
        isinstance(text, str) and len(text.strip()) > 100
    ), f"{name} explanation_rubric should be a non-trivial multi-band rubric"
    # Sanity: rubric should mention each of the expected score bands.
    for band in ("10", "7", "4", "1", "0"):
        assert band in text, f"{name} rubric missing band marker '{band}'"


def test_get_world_returns_optimal_explanation():
    cfg = get_world("gravity")
    assert "optimal_explanation" in cfg
    assert cfg["optimal_explanation"]


def test_get_world_returns_explanation_rubric():
    cfg = get_world("gravity")
    assert "explanation_rubric" in cfg
    assert cfg["explanation_rubric"]


def test_parse_judge_score_basic():
    assert _parse_judge_score("Reasoning here. <score>7</score>") == 7.0
    assert _parse_judge_score("<score> 10 </score>") == 10.0
    assert _parse_judge_score("no tag here") is None
    assert _parse_judge_score("") is None
    assert _parse_judge_score(None) is None


def test_judge_returns_zero_for_empty_explanation():
    judge = ExplanationJudge()
    result = judge.score(
        agent_explanation=None,
        optimal_explanation="The system obeys a 2D Laplacian field.",
        verbose=False,
    )
    assert result["score"] == 0.0
    assert result["raw_score"] == 0
    assert result["error"] is None

    result = judge.score(
        agent_explanation="   ",
        optimal_explanation="The system obeys a 2D Laplacian field.",
        verbose=False,
    )
    assert result["score"] == 0.0


def test_judge_skips_when_no_ground_truth():
    judge = ExplanationJudge()
    result = judge.score(
        agent_explanation="Some agent description.",
        optimal_explanation="",
        verbose=False,
    )
    assert result["score"] is None
    assert result["error"] == "missing_ground_truth"


def test_judge_parses_mocked_reply():
    judge = ExplanationJudge(judge_model="claude-opus-4-6")
    fake_reply = (
        "The student correctly identifies the Laplacian operator and the force law, "
        "but misses the screening structure. <score>6</score>"
    )
    with patch("scienceagent.llm_client.complete", return_value=fake_reply) as mock:
        result = judge.score(
            agent_explanation="Two particles interact through a Laplacian field.",
            optimal_explanation="Two particles interact through a screened Helmholtz field.",
            verbose=False,
        )
        mock.assert_called_once()
        kwargs = mock.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-6"
        assert "temperature" not in kwargs

    assert result["raw_score"] == 6
    assert result["score"] == pytest.approx(0.6)
    assert result["error"] is None


def test_judge_injects_rubric_into_prompt():
    judge = ExplanationJudge()
    rubric = "10 — must identify the custom screening operator XYZ_123."
    with patch(
        "scienceagent.llm_client.complete",
        return_value="Good. <score>8</score>",
    ) as mock:
        judge.score(
            agent_explanation="Some description.",
            optimal_explanation="Ground truth about screening.",
            rubric=rubric,
            verbose=False,
        )
        prompt = mock.call_args.kwargs["messages"][0]["content"]
        assert "XYZ_123" in prompt, "rubric text must appear in the judge prompt"
        assert "<scoring_rubric>" in prompt


def test_judge_falls_back_to_generic_guide_without_rubric():
    judge = ExplanationJudge()
    with patch(
        "scienceagent.llm_client.complete",
        return_value="OK. <score>5</score>",
    ) as mock:
        judge.score(
            agent_explanation="Some description.",
            optimal_explanation="Ground truth.",
            rubric=None,
            verbose=False,
        )
        prompt = mock.call_args.kwargs["messages"][0]["content"]
        # The generic guide's wording should appear when no rubric is passed.
        assert "captures every essential element correctly" in prompt


def test_judge_handles_unparseable_reply():
    judge = ExplanationJudge()
    fake_reply = "I refuse to score this."
    with patch("scienceagent.llm_client.complete", return_value=fake_reply):
        result = judge.score(
            agent_explanation="Some description.",
            optimal_explanation="Ground truth.",
            verbose=False,
        )
    assert result["score"] is None
    assert result["error"] is not None
    assert "parse" in result["error"].lower()


def test_judge_handles_llm_failure():
    judge = ExplanationJudge()
    with patch(
        "scienceagent.llm_client.complete",
        side_effect=RuntimeError("API down"),
    ):
        result = judge.score(
            agent_explanation="Some description.",
            optimal_explanation="Ground truth.",
            verbose=False,
        )
    assert result["score"] is None
    assert "API down" in result["error"]
