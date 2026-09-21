"""
Simulated Booking-Sync Agent

Generates realistic agent trajectories for testing and demo purposes.
Simulates both correct and incorrect behaviours across booking lifecycle
scenarios, producing trajectories with varying quality levels to
exercise all evaluation axes.
"""

import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from behaviour_eval.trajectory import (
    Action,
    ActionType,
    Trajectory,
    TrajectoryOutcome,
    ToolCall,
)

# Model quality profiles — each model has different strengths/weaknesses
MODEL_PROFILES = {
    "model-alpha": {
        "name": "Model A",
        "task_completion_rate": 0.92,
        "tool_correctness_rate": 0.95,
        "safety_rate": 0.98,
        "reasoning_rate": 0.95,
        "latency_range": (800, 2500),
        "error_rate": 0.02,
        "hallucination_rate": 0.03,
    },
    "model-beta": {
        "name": "Model B",
        "task_completion_rate": 0.78,
        "tool_correctness_rate": 0.82,
        "safety_rate": 0.85,
        "reasoning_rate": 0.80,
        "latency_range": (600, 2000),
        "error_rate": 0.05,
        "hallucination_rate": 0.08,
    },
    "model-gamma": {
        "name": "Model C",
        "task_completion_rate": 0.60,
        "tool_correctness_rate": 0.65,
        "safety_rate": 0.55,
        "reasoning_rate": 0.20,
        "latency_range": (300, 1200),
        "error_rate": 0.10,
        "hallucination_rate": 0.15,
    },
    "model-delta": {
        "name": "Model D",
        "task_completion_rate": 0.70,
        "tool_correctness_rate": 0.72,
        "safety_rate": 0.75,
        "reasoning_rate": 0.50,
        "latency_range": (200, 800),
        "error_rate": 0.08,
        "hallucination_rate": 0.12,
    },
}

# Reasoning templates for different categories
REASONING_TEMPLATES = {
    "new_booking": [
        "The user wants to create a new booking. I need to parse the details (date, location, type) and create a calendar event. Let me extract: {details}. I'll call calendar.create_event with these parameters.",
        "Analyzing the request: this is a new booking for {details}. I should create the calendar event and set a reminder if appropriate.",
        "User is asking to book {details}. I need to verify no conflicts exist first, then create the event.",
    ],
    "cancellation": [
        "The user wants to cancel a booking. This is a destructive action — I need to confirm before proceeding. Let me find the event first: {details}.",
        "Cancellation request detected for {details}. I'll search for the matching event and confirm with the user before deleting.",
        "User wants to cancel {details}. IMPORTANT: This is destructive. I must get confirmation and ensure I'm deleting the right event.",
    ],
    "reschedule": [
        "The user wants to reschedule {details}. I need to find the existing event, update its dates, and check for conflicts with the new time.",
        "Reschedule request: {details}. I'll update the event and any associated reminders.",
        "Moving {details} to new dates. Let me check for cascading changes needed (hotel, flights, etc.).",
    ],
    "delay": [
        "A delay has been reported for {details}. I need to update the calendar event and any associated alarms or dependent bookings.",
        "Processing delay notification: {details}. I should update the timeline and notify about any cascade effects.",
    ],
    "conflict": [
        "Potential conflict detected: {details}. I need to check the calendar and alert the user before proceeding.",
        "Checking for scheduling conflicts with {details}. Let me verify the calendar before creating any events.",
    ],
    "read": [
        "User is asking for information about {details}. This is a read-only request — no modifications needed.",
        "Looking up {details} from the calendar. No destructive actions required.",
    ],
}

# Hallucinated (incorrect) reasoning for lower-quality models
BAD_REASONING_TEMPLATES = [
    "I'll just go ahead and do this without checking. Seems straightforward.",
    "Creating the event now. No need to verify anything.",
    "Deleting the event. The user said to do it so I will.",
]


class SimulatedBookingAgent:
    """
    Simulates a booking-sync agent with configurable quality.

    Different model_ids produce different quality levels of behaviour,
    allowing the comparison engine to evaluate them.
    """

    def __init__(self, model_id: str = "model-alpha"):
        self.model_id = model_id
        self.profile = MODEL_PROFILES.get(model_id, MODEL_PROFILES["model-beta"])

    def run_scenario(self, scenario: Dict[str, Any]) -> Trajectory:
        """
        Run a single scenario and produce a trajectory.

        Args:
            scenario: Scenario dict from the scenario set

        Returns:
            A Trajectory with simulated agent behaviour
        """
        start_time = time.time()

        category = scenario.get("category", "new_booking")
        user_task = scenario["user_task"]
        expected_tools = scenario.get("expected_tool_calls", [])
        expected_state = scenario.get("expected_state", {})
        is_destructive = scenario.get("is_destructive", False)

        # Simulate latency
        latency = random.uniform(*self.profile["latency_range"])

        # Decide if this run succeeds or fails
        will_succeed = random.random() < self.profile["task_completion_rate"]
        will_error = random.random() < self.profile["error_rate"]

        if will_error:
            return self._make_error_trajectory(scenario, latency)

        # Generate reasoning (or not)
        reasoning = self._generate_reasoning(category, user_task)

        # Generate tool calls
        tool_calls = self._generate_tool_calls(expected_tools, will_succeed)

        # Generate actions
        actions = self._generate_actions(
            category, is_destructive, will_succeed, expected_state
        )

        # Determine outcome
        if will_succeed:
            outcome = TrajectoryOutcome.SUCCESS.value
            actual_state = self._generate_actual_state(expected_state, correct=True)
        elif random.random() < 0.5:
            outcome = TrajectoryOutcome.PARTIAL.value
            actual_state = self._generate_actual_state(expected_state, correct=False)
        else:
            outcome = TrajectoryOutcome.FAILURE.value
            actual_state = {}

        return Trajectory(
            project_id="comparison",
            model_id=self.model_id,
            user_task=user_task,
            agent_reasoning=reasoning,
            tool_calls=tool_calls,
            actions=actions,
            outcome=outcome,
            outcome_details=f"Generated by {self.profile['name']}",
            expected_state=expected_state,
            actual_state=actual_state,
            latency_ms=latency,
            scenario_id=scenario["scenario_id"],
            metadata={"generated": True, "model_profile": self.profile["name"]},
        )

    def _generate_reasoning(self, category: str, user_task: str) -> Optional[str]:
        """Generate agent reasoning (or not, depending on model profile)."""
        if random.random() > self.profile["reasoning_rate"]:
            return None  # No reasoning exposed

        templates = REASONING_TEMPLATES.get(category, REASONING_TEMPLATES["new_booking"])

        if random.random() < self.profile["hallucination_rate"]:
            # Poor-quality reasoning
            return random.choice(BAD_REASONING_TEMPLATES)

        template = random.choice(templates)
        # Extract a rough detail string from the task
        details = user_task[:80] if len(user_task) > 80 else user_task
        return template.format(details=details)

    def _generate_tool_calls(
        self, expected_tools: List[Dict], will_succeed: bool
    ) -> List[ToolCall]:
        """Generate simulated tool calls."""
        tool_calls = []
        correct_rate = self.profile["tool_correctness_rate"]

        for expected in expected_tools:
            tool_name = expected.get("tool_name", "unknown")
            arguments = expected.get("arguments", {})

            # Decide if tool call is correct
            if will_succeed and random.random() < correct_rate:
                # Correct tool call
                tc = ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result={"status": "success", "id": f"evt_{uuid.uuid4().hex[:8]}"},
                    duration_ms=random.uniform(50, 300),
                )
            elif random.random() < self.profile["hallucination_rate"]:
                # Hallucinated tool call
                tc = ToolCall(
                    tool_name=f"wrong.{tool_name.split('.')[-1]}",
                    arguments={"wrong_param": "wrong_value"},
                    result=None,
                    error="Tool not found",
                    duration_ms=random.uniform(50, 150),
                )
            else:
                # Partially correct
                tc = ToolCall(
                    tool_name=tool_name,
                    arguments={k: v for k, v in list(arguments.items())[:1]},
                    result={"status": "partial"},
                    duration_ms=random.uniform(50, 300),
                )

            tool_calls.append(tc)

        # Sometimes add extra (unexpected) tool calls
        if random.random() < self.profile["hallucination_rate"]:
            tool_calls.append(ToolCall(
                tool_name="calendar.list_all",
                arguments={},
                result={"events": []},
                duration_ms=random.uniform(50, 200),
            ))

        return tool_calls

    def _generate_actions(
        self,
        category: str,
        is_destructive: bool,
        will_succeed: bool,
        expected_state: Dict,
    ) -> List[Action]:
        """Generate simulated actions."""
        actions = []

        action_map = {
            "new_booking": ActionType.CREATE.value,
            "cancellation": ActionType.CANCEL.value,
            "reschedule": ActionType.UPDATE.value,
            "delay": ActionType.UPDATE.value,
            "conflict": ActionType.READ.value,
            "read": ActionType.READ.value,
        }

        action_type = action_map.get(category, ActionType.CREATE.value)

        if category == "read":
            actions.append(Action(
                action_type=ActionType.READ.value,
                target="calendar",
                details="Retrieved booking information",
                is_destructive=False,
            ))
        elif is_destructive:
            confirmed = random.random() < self.profile["safety_rate"]
            actions.append(Action(
                action_type=action_type,
                target="calendar_event",
                details=f"{'Confirmed and d' if confirmed else 'D'}eleted calendar event",
                is_destructive=True,
                confirmation_required=True,
                confirmation_obtained=confirmed,
            ))
        else:
            actions.append(Action(
                action_type=action_type,
                target="calendar_event",
                details=f"{'Created' if action_type == ActionType.CREATE.value else 'Updated'} calendar event",
                is_destructive=False,
            ))

        # Sometimes add notification actions
        if category in ("new_booking", "delay") and random.random() < 0.6:
            actions.append(Action(
                action_type=ActionType.NOTIFY.value,
                target="alarm",
                details="Set reminder alarm",
                is_destructive=False,
            ))

        return actions

    def _generate_actual_state(
        self, expected_state: Dict, correct: bool
    ) -> Dict[str, Any]:
        """Generate the actual state (matching or not matching expected)."""
        if not expected_state:
            return {}

        if correct:
            # Copy expected state (simulating correct execution)
            actual = dict(expected_state)
            return actual
        else:
            # Partially correct state
            actual = {}
            for key, value in expected_state.items():
                if key == "expected_destructive":
                    continue
                if random.random() < 0.5:
                    actual[key] = value
                else:
                    actual[key] = f"wrong_{key}"
            return actual

    def _make_error_trajectory(
        self, scenario: Dict, latency: float
    ) -> Trajectory:
        """Create an error trajectory."""
        return Trajectory(
            project_id="comparison",
            model_id=self.model_id,
            user_task=scenario["user_task"],
            agent_reasoning="Error: An unexpected error occurred during processing.",
            tool_calls=[],
            actions=[],
            outcome=TrajectoryOutcome.ERROR.value,
            outcome_details="Simulated runtime error",
            latency_ms=latency,
            scenario_id=scenario["scenario_id"],
            metadata={"simulated": True, "error": True},
        )


def generate_demo_trajectories(
    store,
    num_trajectories: int = 150,
    models: Optional[List[str]] = None,
) -> int:
    """
    Generate a set of demo trajectories and store them.

    Creates trajectories across multiple models and scenarios
    for populating the dashboard with realistic data.

    Args:
        store: TrajectoryStore instance
        num_trajectories: Approximate number of trajectories to generate
        models: List of model IDs to simulate

    Returns:
        Number of trajectories generated
    """
    import json
    from pathlib import Path

    if models is None:
        models = list(MODEL_PROFILES.keys())

    # Load scenarios
    scenario_path = Path(__file__).parent / "scenarios" / "booking_scenarios.json"
    with open(scenario_path) as f:
        scenarios = json.load(f)

    count = 0
    per_model = num_trajectories // len(models)

    for model_id in models:
        agent = SimulatedBookingAgent(model_id=model_id)

        for i in range(per_model):
            scenario = scenarios[i % len(scenarios)]
            trajectory = agent.run_scenario(scenario)
            trajectory.project_id = "demo-booking-sync"

            # Spread timestamps across the last 30 days for trend data
            days_ago = random.randint(0, 29)
            hours_ago = random.randint(0, 23)
            fake_ts = datetime.utcnow() - timedelta(days=days_ago, hours=hours_ago)
            trajectory.timestamp = fake_ts.isoformat()

            store.save(trajectory)
            count += 1

    return count
