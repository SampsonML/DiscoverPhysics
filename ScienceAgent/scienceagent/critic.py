"""
CriticAgent: reviews the science agent's experiments for rule compliance
and information gain.

Runs as a separate single-turn LLM call after each experiment round
(starting from round 2), injecting feedback into the conversation.
"""

from scienceagent import llm_client

# likely these instructions are important, and for now I have not experimented with them much
_CRITIC_SYSTEM_PROMPT = """\
You are a physics expert and supervisor reviewing a science agent's experimental process in a simulated physics universe.

You will be given:
1. The RULES the science agent must follow (its system prompt)
2. The full conversation history so far
3. The current round number

Your job is to provide brief, actionable feedback on two things:

A) RULE COMPLIANCE — Is the agent following its instructions?
   Check for:
   - Only one action per round (run_experiment or final_law, not both)
   - Not reproducing/quoting raw experiment data in its responses
   - Parameter ranges respected (positions [-10,10], velocities [-5,5], properties [0.1,10], duration >= 5.0, measurement_times <= 10 entries)
   - JSON format without inline comments
   - Using the correct number of measurement times

B) EXPERIMENT QUALITY — Is the agent making good scientific progress?
   Check for:
   - Is this experiment providing new information not seen in previous rounds?
   - Is the agent systematically varying parameters to understand the physics?
   - Is the agent repeating configurations it has already tested?
   - Is the agent exploring the parameter space efficiently?

Format your response as:
**Rule Compliance:** [PASS or specific violations found]
**Experiment Quality:** [Brief assessment and suggestions]

Keep your feedback to 3-5 sentences total. Be direct and specific.\
"""


class CriticAgent:
    """
    Supervisor agent that reviews the science agent's experiment rounds.

    Args:
        model: LLM model string for the critic (default: Haiku).
        max_tokens: Max tokens per critic LLM call.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
    ):
        self.model = model
        self.max_tokens = max_tokens

    def review(
        self,
        agent_system_prompt: str,
        messages: list[dict],
        round_num: int,
    ) -> str:
        """
        Review the agent's latest round and return feedback.

        Args:
            agent_system_prompt: The system prompt the science agent operates under.
            messages: Full conversation history (list of role/content dicts).
            round_num: Current round number.

        Returns:
            Feedback string from the critic.
        """
        transcript = _format_transcript(messages)

        user_msg = (
            f"Round {round_num} has just completed.\n\n"
            f"<agent_rules>\n{agent_system_prompt}\n</agent_rules>\n\n"
            f"<conversation_history>\n{transcript}\n</conversation_history>\n\n"
            "Review the agent's latest round for rule compliance and experiment quality."
        )

        return llm_client.complete(
            model=self.model,
            messages=[{"role": "user", "content": user_msg}],
            system=_CRITIC_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )


def _format_transcript(messages: list[dict]) -> str:
    """Format conversation history into a readable transcript for the critic."""
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        lines.append(f"[{role}]\n{msg['content']}")
    return "\n\n".join(lines)
