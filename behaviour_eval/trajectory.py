"""
Trajectory Schema — Core data structures for capturing agent interactions.

Each interaction between a user and an agent is captured as a Trajectory,
which contains the user's task, the agent's reasoning (if available),
tool calls made, actions taken, and the outcome.

This module provides:
    - ToolCall: A single tool invocation by the agent
    - Action: A real-world action taken by the agent
    - Trajectory: The complete interaction record
    - Serialisation/deserialisation to/from JSON and dict
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    """Types of real-world actions an agent can take."""
    CREATE = "create"
    UPDATE = "update"
    CANCEL = "cancel"
    CONFIRM = "confirm"
    DELETE = "delete"
    READ = "read"
    NOTIFY = "notify"


class TrajectoryOutcome(str, Enum):
    """Outcome of a trajectory."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """
    Represents a single tool invocation by the agent.

    Attributes:
        tool_name: Name of the tool invoked (e.g., 'calendar.create_event')
        arguments: Dictionary of arguments passed to the tool
        result: The result returned by the tool (can be any JSON-serialisable value)
        timestamp: When the tool was called
        duration_ms: How long the tool call took in milliseconds
        error: Error message if the tool call failed, None otherwise
    """
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    timestamp: str = ""
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(**data)


@dataclass
class Action:
    """
    Represents a real-world action taken by the agent.

    Attributes:
        action_type: The type of action (create, update, cancel, etc.)
        target: What the action targets (e.g., 'calendar_event', 'alarm')
        details: Human-readable description of what was done
        is_destructive: Whether this action is destructive (delete, cancel)
        confirmation_required: Whether the agent should have confirmed before acting
        confirmation_obtained: Whether the agent actually obtained confirmation
        timestamp: When the action was taken
    """
    action_type: str
    target: str
    details: str
    is_destructive: bool = False
    confirmation_required: bool = False
    confirmation_obtained: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        # Auto-detect destructiveness
        if self.action_type in (ActionType.DELETE.value, ActionType.CANCEL.value):
            self.is_destructive = True
            self.confirmation_required = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(**data)


@dataclass
class Trajectory:
    """
    The complete record of a single agent interaction.

    A trajectory captures everything that happened when an agent
    responded to a user task: the reasoning, tool calls, actions,
    and outcome.

    Attributes:
        trajectory_id: Unique identifier for this trajectory
        project_id: The project this trajectory belongs to
        model_id: The model/agent that produced this trajectory
        user_task: The original user task/request
        agent_reasoning: The agent's self-reported reasoning (for RQ1, may be None)
        tool_calls: List of tool invocations made by the agent
        actions: List of real-world actions taken
        outcome: The outcome of the trajectory
        outcome_details: Additional details about the outcome
        expected_state: What the correct final state should be (for evaluation)
        actual_state: What the actual final state is
        latency_ms: End-to-end latency in milliseconds
        timestamp: When the trajectory was created
        metadata: Additional key-value metadata
        scores: Evaluation scores (populated by evaluators)
    """
    trajectory_id: str = ""
    project_id: str = "default"
    model_id: str = "unknown"
    user_task: str = ""
    agent_reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    outcome: str = TrajectoryOutcome.UNKNOWN.value
    outcome_details: str = ""
    expected_state: Optional[Dict[str, Any]] = None
    actual_state: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    scenario_id: Optional[str] = None

    def __post_init__(self):
        if not self.trajectory_id:
            self.trajectory_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    @property
    def has_reasoning(self) -> bool:
        """Whether this trajectory includes agent reasoning (relevant for RQ1)."""
        return self.agent_reasoning is not None and len(self.agent_reasoning.strip()) > 0

    @property
    def has_destructive_actions(self) -> bool:
        """Whether any actions in this trajectory are destructive."""
        return any(a.is_destructive for a in self.actions)

    @property
    def destructive_actions_confirmed(self) -> bool:
        """Whether all destructive actions were properly confirmed."""
        destructive = [a for a in self.actions if a.is_destructive]
        if not destructive:
            return True
        return all(a.confirmation_obtained for a in destructive)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary."""
        d = {
            "trajectory_id": self.trajectory_id,
            "project_id": self.project_id,
            "model_id": self.model_id,
            "user_task": self.user_task,
            "agent_reasoning": self.agent_reasoning,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "actions": [a.to_dict() for a in self.actions],
            "outcome": self.outcome,
            "outcome_details": self.outcome_details,
            "expected_state": self.expected_state,
            "actual_state": self.actual_state,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "scores": self.scores,
            "scenario_id": self.scenario_id,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Deserialise from a dictionary."""
        tool_calls = [ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])]
        actions = [Action.from_dict(a) for a in data.get("actions", [])]
        return cls(
            trajectory_id=data.get("trajectory_id", ""),
            project_id=data.get("project_id", "default"),
            model_id=data.get("model_id", "unknown"),
            user_task=data.get("user_task", ""),
            agent_reasoning=data.get("agent_reasoning"),
            tool_calls=tool_calls,
            actions=actions,
            outcome=data.get("outcome", TrajectoryOutcome.UNKNOWN.value),
            outcome_details=data.get("outcome_details", ""),
            expected_state=data.get("expected_state"),
            actual_state=data.get("actual_state"),
            latency_ms=data.get("latency_ms"),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
            scores=data.get("scores", {}),
            scenario_id=data.get("scenario_id"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Trajectory":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))
