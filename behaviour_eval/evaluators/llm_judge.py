"""
LLM Judge — Automated evaluation using language models.

Provides two judge implementations:
    - LLMJudge: Single-model judge for scoring trajectories
    - JuryJudge: Multi-model jury for reducing single-judge bias (RQ2)

Also provides pairwise comparison for offline model comparison:
    - Given two trajectories for the same scenario, which is better?

When no API key is available, falls back to a rule-based mock judge
that uses heuristics for scoring — ensuring the platform works at
zero cost.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from behaviour_eval.trajectory import Trajectory

logger = logging.getLogger(__name__)


# Prompt templates for the LLM judge
SCORE_PROMPT_TEMPLATE = """You are an expert evaluator of AI agent behaviour.

You are given a trajectory showing how an agent responded to a user task.
Score the agent's behaviour on the following axes, each from 0.0 to 1.0:

1. task_completion: Did the agent achieve what the user asked for?
2. tool_use_correctness: Did the agent use the right tools with correct arguments?
3. intent_accuracy: Did the agent correctly understand what the user wanted?

## User Task
{user_task}

## Agent Reasoning
{reasoning}

## Tool Calls
{tool_calls}

## Actions Taken
{actions}

## Outcome
{outcome}

Respond with ONLY a JSON object like:
{{"task_completion": 0.8, "tool_use_correctness": 0.9, "intent_accuracy": 0.85}}
"""

PAIRWISE_PROMPT_TEMPLATE = """You are an expert evaluator comparing two AI agents on the same task.

## User Task
{user_task}

## Agent A
Reasoning: {reasoning_a}
Tool Calls: {tool_calls_a}
Actions: {actions_a}
Outcome: {outcome_a}

## Agent B
Reasoning: {reasoning_b}
Tool Calls: {tool_calls_b}
Actions: {actions_b}
Outcome: {outcome_b}

Which agent performed better overall? Consider task completion, correctness, safety, and efficiency.

Respond with ONLY a JSON object:
{{"winner": "A" or "B" or "tie", "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}}
"""


class LLMJudge:
    """
    Single-model LLM judge for scoring agent trajectories.

    Falls back to rule-based scoring when no API key is configured.

    Args:
        model: The model name to use (e.g., 'gpt-4', 'claude-3-sonnet')
        api_key: API key for the model provider
        provider: API provider ('openai', 'groq', 'anthropic', or 'mock')
    """

    def __init__(
        self,
        model: str = "mock",
        api_key: Optional[str] = None,
        provider: str = "mock",
        judge_id: Optional[str] = None,
    ):
        openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        groq_key = api_key or os.environ.get("GROQ_API_KEY")
        anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if provider == "mock" and openai_key:
            provider = "openai"
            api_key = openai_key
            if model == "mock":
                model = "gpt-4o-mini"
        elif provider == "mock" and groq_key:
            provider = "groq"
            api_key = groq_key
            if model == "mock":
                model = "llama-3.1-8b-instant"
        elif provider == "mock" and anthropic_key:
            provider = "anthropic"
            api_key = anthropic_key

        self.model = model
        self.api_key = api_key or openai_key or groq_key or anthropic_key
        self.provider = provider if self.api_key else "mock"
        self.judge_id = judge_id or f"judge_{model}"

        if self.provider == "mock":
            logger.info("No API key found — using mock judge (rule-based scoring)")

    def score(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        Score a trajectory using the LLM judge.

        Returns:
            Dictionary of metric_name -> score
        """
        if self.provider == "mock":
            return self._mock_score(trajectory)

        try:
            return self._llm_score(trajectory)
        except Exception as e:
            logger.warning(f"LLM judge failed, falling back to mock: {e}")
            return self._mock_score(trajectory)

    def pairwise_compare(
        self, trajectory_a: Trajectory, trajectory_b: Trajectory
    ) -> Dict[str, Any]:
        """
        Compare two trajectories pairwise.

        Returns:
            Dictionary with 'winner' ('A', 'B', or 'tie'), 'confidence', and 'reasoning'
        """
        if self.provider == "mock":
            return self._mock_pairwise(trajectory_a, trajectory_b)

        try:
            return self._llm_pairwise(trajectory_a, trajectory_b)
        except Exception as e:
            logger.warning(f"LLM pairwise comparison failed, falling back to mock: {e}")
            return self._mock_pairwise(trajectory_a, trajectory_b)

    def _mock_score(self, trajectory: Trajectory) -> Dict[str, float]:
        """Rule-based mock scoring when no LLM is available."""
        from behaviour_eval.evaluators.tool_using_agent import ToolUsingAgentEvaluator
        evaluator = ToolUsingAgentEvaluator()
        base_scores = evaluator.score(trajectory)

        # Add small random noise to simulate LLM judgment variability
        noisy_scores = {}
        for key, val in base_scores.items():
            if key == "latency":
                noisy_scores[key] = val
            else:
                noise = random.uniform(-0.05, 0.05)
                noisy_scores[key] = max(0.0, min(1.0, val + noise))

        return noisy_scores

    def _mock_pairwise(
        self, trajectory_a: Trajectory, trajectory_b: Trajectory
    ) -> Dict[str, Any]:
        """Rule-based mock pairwise comparison."""
        from behaviour_eval.evaluators.tool_using_agent import ToolUsingAgentEvaluator
        evaluator = ToolUsingAgentEvaluator()

        scores_a = evaluator.score(trajectory_a)
        scores_b = evaluator.score(trajectory_b)

        # Compare on non-latency metrics
        metric_keys = [k for k in scores_a if k != "latency"]
        avg_a = sum(scores_a[k] for k in metric_keys) / len(metric_keys)
        avg_b = sum(scores_b[k] for k in metric_keys) / len(metric_keys)

        diff = avg_a - avg_b
        if abs(diff) < 0.05:
            winner = "tie"
            confidence = 0.5
        elif diff > 0:
            winner = "A"
            confidence = min(0.95, 0.5 + abs(diff))
        else:
            winner = "B"
            confidence = min(0.95, 0.5 + abs(diff))

        return {
            "winner": winner,
            "confidence": round(confidence, 2),
            "reasoning": f"Average scores: A={avg_a:.3f}, B={avg_b:.3f}",
            "scores_a": {k: round(v, 3) for k, v in scores_a.items()},
            "scores_b": {k: round(v, 3) for k, v in scores_b.items()},
        }

    def _format_trajectory(self, trajectory: Trajectory) -> Dict[str, str]:
        """Format trajectory fields for prompt insertion."""
        tool_calls_str = json.dumps(
            [{"tool": tc.tool_name, "args": tc.arguments, "result": tc.result}
             for tc in trajectory.tool_calls],
            indent=2,
        )
        actions_str = json.dumps(
            [{"type": a.action_type, "target": a.target, "details": a.details}
             for a in trajectory.actions],
            indent=2,
        )
        return {
            "user_task": trajectory.user_task,
            "reasoning": trajectory.agent_reasoning or "(no reasoning provided)",
            "tool_calls": tool_calls_str,
            "actions": actions_str,
            "outcome": trajectory.outcome,
        }

    def _llm_score(self, trajectory: Trajectory) -> Dict[str, float]:
        """Score using actual LLM API call."""
        fields = self._format_trajectory(trajectory)
        prompt = SCORE_PROMPT_TEMPLATE.format(**fields)

        response_text = self._call_llm(prompt)

        try:
            scores = json.loads(response_text)
            return {k: float(v) for k, v in scores.items()}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {response_text}")
            return self._mock_score(trajectory)

    def _llm_pairwise(
        self, trajectory_a: Trajectory, trajectory_b: Trajectory
    ) -> Dict[str, Any]:
        """Pairwise comparison using actual LLM API call."""
        fields_a = self._format_trajectory(trajectory_a)
        fields_b = self._format_trajectory(trajectory_b)

        prompt = PAIRWISE_PROMPT_TEMPLATE.format(
            user_task=fields_a["user_task"],
            reasoning_a=fields_a["reasoning"],
            tool_calls_a=fields_a["tool_calls"],
            actions_a=fields_a["actions"],
            outcome_a=fields_a["outcome"],
            reasoning_b=fields_b["reasoning"],
            tool_calls_b=fields_b["tool_calls"],
            actions_b=fields_b["actions"],
            outcome_b=fields_b["outcome"],
        )

        response_text = self._call_llm(prompt)

        try:
            return json.loads(response_text)
        except (json.JSONDecodeError, ValueError):
            logger.error(f"Failed to parse LLM pairwise response: {response_text}")
            return self._mock_pairwise(trajectory_a, trajectory_b)

    def _call_llm(self, prompt: str) -> str:
        """Make an API call to the configured LLM provider."""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "groq":
            return self._call_groq(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise

    def _call_groq(self, prompt: str) -> str:
        """Call Groq's OpenAI-compatible chat completions API."""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise

    def _call_anthropic(self, prompt: str) -> str:
        """Call the Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except ImportError:
            logger.error("anthropic package not installed. Install with: pip install anthropic")
            raise


class JuryJudge:
    """
    Multi-model jury judge for reducing single-judge bias (RQ2).

    Orchestrates multiple LLMJudge instances (potentially different models)
    and aggregates their scores via averaging or majority vote.

    Args:
        judges: List of LLMJudge instances
        aggregation: How to combine scores ('average', 'median', 'majority_vote')
    """

    def __init__(
        self,
        judges: Optional[List[LLMJudge]] = None,
        aggregation: str = "average",
    ):
        if judges:
            self.judges = judges
        else:
            # Default: 3 mock judges with different noise seeds
            self.judges = [
                LLMJudge(model="mock_judge_1", provider="mock", judge_id="jury_1"),
                LLMJudge(model="mock_judge_2", provider="mock", judge_id="jury_2"),
                LLMJudge(model="mock_judge_3", provider="mock", judge_id="jury_3"),
            ]
        self.aggregation = aggregation
        self.judge_id = "jury"

    def score(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        Score a trajectory using the full jury.

        Returns aggregated scores and individual judge scores in metadata.
        """
        all_scores = []
        for judge in self.judges:
            scores = judge.score(trajectory)
            all_scores.append(scores)

        # Aggregate
        aggregated = self._aggregate(all_scores)
        return aggregated

    def pairwise_compare(
        self, trajectory_a: Trajectory, trajectory_b: Trajectory
    ) -> Dict[str, Any]:
        """
        Pairwise comparison using the full jury.

        Aggregates individual judge comparisons via majority vote on winner.
        """
        results = []
        for judge in self.judges:
            result = judge.pairwise_compare(trajectory_a, trajectory_b)
            results.append(result)

        # Majority vote on winner
        winners = [r["winner"] for r in results]
        winner_counts = {}
        for w in winners:
            winner_counts[w] = winner_counts.get(w, 0) + 1

        final_winner = max(winner_counts, key=winner_counts.get)
        agreement = winner_counts[final_winner] / len(winners)

        return {
            "winner": final_winner,
            "confidence": round(agreement, 2),
            "reasoning": f"Jury vote: {winner_counts}",
            "individual_results": results,
        }

    def _aggregate(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate scores from multiple judges."""
        if self.aggregation == "average":
            return self._aggregate_average(all_scores)
        elif self.aggregation == "median":
            return self._aggregate_median(all_scores)
        else:
            return self._aggregate_average(all_scores)

    def _aggregate_average(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Average all judge scores."""
        if not all_scores:
            return {}

        aggregated = {}
        keys = all_scores[0].keys()
        for key in keys:
            values = [s[key] for s in all_scores if key in s]
            aggregated[key] = sum(values) / len(values) if values else 0
        return aggregated

    def _aggregate_median(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Take the median of all judge scores."""
        if not all_scores:
            return {}

        aggregated = {}
        keys = all_scores[0].keys()
        for key in keys:
            values = sorted([s[key] for s in all_scores if key in s])
            if values:
                mid = len(values) // 2
                aggregated[key] = values[mid]
            else:
                aggregated[key] = 0
        return aggregated

    def get_individual_scores(self, trajectory: Trajectory) -> List[Dict[str, Any]]:
        """Get scores from each individual judge (for RQ2 analysis)."""
        results = []
        for judge in self.judges:
            scores = judge.score(trajectory)
            results.append({
                "judge_id": judge.judge_id,
                "model": judge.model,
                "scores": scores,
            })
        return results
