"""
Behaviour SDK — Instrumentation for capturing agent interactions.

Provides a lightweight SDK that wraps an agent function and captures
its interactions as Trajectory objects. Trajectories are buffered
and flushed asynchronously to the ingest service.

Usage:
    sdk = BehaviourSDK(project_id="my-project", model_id="gpt-4")

    # Manual capture
    trajectory = sdk.capture(
        user_task="Book a hotel for March 15",
        reasoning="User wants a hotel booking...",
        tool_calls=[...],
        actions=[...],
        outcome="success"
    )
    sdk.flush()

    # Decorator-based capture
    @sdk.capture_behaviour
    def my_agent(task: str) -> dict:
        # agent logic here
        return {"reasoning": "...", "tool_calls": [...], ...}
"""

import time
import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from behaviour_eval.trajectory import (
    Action,
    Trajectory,
    TrajectoryOutcome,
    ToolCall,
)

logger = logging.getLogger(__name__)


class BehaviourSDK:
    """
    SDK for instrumenting AI agents and capturing their behaviour as trajectories.

    The SDK maintains a buffer of captured trajectories and can flush them
    to a TrajectoryStore (from ingest.py) either manually or automatically
    when the buffer reaches a threshold.

    Args:
        project_id: The project identifier for grouping trajectories
        model_id: The model/agent identifier
        store: Optional TrajectoryStore instance for persistence
        buffer_size: Number of trajectories to buffer before auto-flushing
    """

    def __init__(
        self,
        project_id: str = "default",
        model_id: str = "unknown",
        store: Optional[Any] = None,
        buffer_size: int = 50,
    ):
        self.project_id = project_id
        self.model_id = model_id
        self.store = store
        self.buffer_size = buffer_size
        self._buffer: List[Trajectory] = []

    def capture(
        self,
        user_task: str,
        reasoning: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        outcome: str = TrajectoryOutcome.UNKNOWN.value,
        outcome_details: str = "",
        expected_state: Optional[Dict[str, Any]] = None,
        actual_state: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scenario_id: Optional[str] = None,
    ) -> Trajectory:
        """
        Capture a single agent interaction as a Trajectory.

        Args:
            user_task: The user's original request
            reasoning: The agent's self-reported reasoning (may be None)
            tool_calls: List of tool call dicts (converted to ToolCall objects)
            actions: List of action dicts (converted to Action objects)
            outcome: Outcome of the interaction
            outcome_details: Additional details about the outcome
            expected_state: What the correct final state should be
            actual_state: What the actual final state is
            latency_ms: End-to-end latency
            metadata: Additional metadata
            scenario_id: Optional scenario identifier for batch comparison

        Returns:
            The captured Trajectory object
        """
        tc_objects = []
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, ToolCall):
                    tc_objects.append(tc)
                else:
                    tc_objects.append(ToolCall.from_dict(tc))

        action_objects = []
        if actions:
            for a in actions:
                if isinstance(a, Action):
                    action_objects.append(a)
                else:
                    action_objects.append(Action.from_dict(a))

        trajectory = Trajectory(
            project_id=self.project_id,
            model_id=self.model_id,
            user_task=user_task,
            agent_reasoning=reasoning,
            tool_calls=tc_objects,
            actions=action_objects,
            outcome=outcome,
            outcome_details=outcome_details,
            expected_state=expected_state,
            actual_state=actual_state,
            latency_ms=latency_ms,
            metadata=metadata or {},
            scenario_id=scenario_id,
        )

        self._buffer.append(trajectory)
        logger.debug(f"Captured trajectory {trajectory.trajectory_id}")

        # Auto-flush if buffer is full
        if len(self._buffer) >= self.buffer_size:
            self.flush()

        return trajectory

    def flush(self) -> int:
        """
        Flush all buffered trajectories to the store.

        Returns:
            Number of trajectories flushed
        """
        if not self._buffer:
            return 0

        count = len(self._buffer)

        if self.store is not None:
            for trajectory in self._buffer:
                try:
                    self.store.save(trajectory)
                except Exception as e:
                    logger.error(f"Failed to save trajectory {trajectory.trajectory_id}: {e}")
        else:
            logger.warning(f"No store configured — {count} trajectories will be lost")

        self._buffer.clear()
        logger.info(f"Flushed {count} trajectories")
        return count

    def get_buffer(self) -> List[Trajectory]:
        """Return the current buffer contents (without flushing)."""
        return list(self._buffer)

    def capture_behaviour(self, func: Callable) -> Callable:
        """
        Decorator that wraps an agent function and captures its behaviour.

        The decorated function should return a dict with keys:
            - reasoning (str, optional)
            - tool_calls (list of dicts, optional)
            - actions (list of dicts, optional)
            - outcome (str, optional)
            - outcome_details (str, optional)
            - expected_state (dict, optional)
            - actual_state (dict, optional)

        The first positional argument is treated as the user_task.

        Example:
            @sdk.capture_behaviour
            def my_agent(task: str) -> dict:
                return {
                    "reasoning": "I need to book a hotel...",
                    "tool_calls": [{"tool_name": "calendar.create", ...}],
                    "actions": [{"action_type": "create", "target": "event", ...}],
                    "outcome": "success",
                }
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_task = args[0] if args else kwargs.get("task", "unknown")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if isinstance(result, dict):
                    self.capture(
                        user_task=str(user_task),
                        reasoning=result.get("reasoning"),
                        tool_calls=result.get("tool_calls"),
                        actions=result.get("actions"),
                        outcome=result.get("outcome", TrajectoryOutcome.SUCCESS.value),
                        outcome_details=result.get("outcome_details", ""),
                        expected_state=result.get("expected_state"),
                        actual_state=result.get("actual_state"),
                        latency_ms=elapsed_ms,
                        scenario_id=result.get("scenario_id"),
                    )
                else:
                    self.capture(
                        user_task=str(user_task),
                        outcome=TrajectoryOutcome.SUCCESS.value,
                        latency_ms=elapsed_ms,
                    )

                return result

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                self.capture(
                    user_task=str(user_task),
                    outcome=TrajectoryOutcome.ERROR.value,
                    outcome_details=str(e),
                    latency_ms=elapsed_ms,
                )
                raise

        return wrapper
