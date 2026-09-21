"""
Evaluator Registry — Archetype-based behaviour scoring.

Provides a registry of evaluators keyed to agent archetypes.
Each archetype defines a metric pack — a set of scoring functions
that are relevant to that type of agent.

The pattern is generalised for arbitrary behaviour axes rather than test pass/fail.

Usage:
    registry = EvaluatorRegistry()
    registry.register("tool_using_agent", ToolUsingAgentEvaluator())
    scores = registry.evaluate("tool_using_agent", trajectory)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from behaviour_eval.trajectory import Trajectory

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for behaviour evaluators.

    Each evaluator implements a metric pack for a specific agent archetype.
    The `score` method takes a trajectory and returns a dictionary of
    metric_name -> score (float in [0, 1] except for raw metrics like latency).
    """

    @property
    @abstractmethod
    def archetype(self) -> str:
        """The archetype this evaluator is designed for."""
        pass

    @property
    @abstractmethod
    def metric_names(self) -> List[str]:
        """List of metric names this evaluator produces."""
        pass

    @abstractmethod
    def score(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        Score a trajectory along all axes in this metric pack.

        Args:
            trajectory: The Trajectory to evaluate

        Returns:
            Dictionary mapping metric names to scores
        """
        pass

    def score_batch(self, trajectories: List[Trajectory]) -> List[Dict[str, float]]:
        """Score a batch of trajectories."""
        return [self.score(t) for t in trajectories]


class EvaluatorRegistry:
    """
    Registry mapping archetype names to evaluator instances.

    Projects select an archetype (e.g., "tool_using_agent") and the
    registry returns the appropriate evaluator with its metric pack.
    """

    def __init__(self):
        self._evaluators: Dict[str, BaseEvaluator] = {}

    def register(self, archetype: str, evaluator: BaseEvaluator):
        """Register an evaluator for an archetype."""
        self._evaluators[archetype] = evaluator
        logger.info(f"Registered evaluator for archetype '{archetype}' with metrics: {evaluator.metric_names}")

    def get(self, archetype: str) -> Optional[BaseEvaluator]:
        """Get the evaluator for an archetype."""
        return self._evaluators.get(archetype)

    def evaluate(self, archetype: str, trajectory: Trajectory) -> Dict[str, float]:
        """
        Evaluate a trajectory using the evaluator for the given archetype.

        Args:
            archetype: The agent archetype
            trajectory: The Trajectory to evaluate

        Returns:
            Dictionary mapping metric names to scores

        Raises:
            ValueError: If no evaluator is registered for the archetype
        """
        evaluator = self._evaluators.get(archetype)
        if evaluator is None:
            raise ValueError(f"No evaluator registered for archetype '{archetype}'. "
                           f"Available: {list(self._evaluators.keys())}")
        return evaluator.score(trajectory)

    def list_archetypes(self) -> List[str]:
        """List all registered archetypes."""
        return list(self._evaluators.keys())

    def list_metrics(self, archetype: str) -> List[str]:
        """List metrics for a given archetype."""
        evaluator = self._evaluators.get(archetype)
        if evaluator is None:
            return []
        return evaluator.metric_names


# Global registry instance
_default_registry = None

def get_default_registry() -> EvaluatorRegistry:
    """Get or create the default evaluator registry with built-in evaluators."""
    global _default_registry
    if _default_registry is None:
        _default_registry = EvaluatorRegistry()
        # Register built-in evaluators
        from behaviour_eval.evaluators.tool_using_agent import ToolUsingAgentEvaluator
        _default_registry.register("tool_using_agent", ToolUsingAgentEvaluator())
    return _default_registry
