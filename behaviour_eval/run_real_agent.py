"""Run the real Booking Sync agent through the behaviour-evaluation platform."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REAL_AGENT_ROOT = Path(__file__).resolve().parent.parent / "studious-octo-parakeet"
sys.path.insert(0, str(REAL_AGENT_ROOT))

from packages.core.models import BookingType, CalendarEvent
from packages.nlp.baseline import BaselineExtractor
from packages.providers.fake_calendar import FakeCalendar
from packages.sync_agent.agent import SyncAgent

from behaviour_eval.comparison import ComparisonEngine
from behaviour_eval.evaluators.llm_judge import LLMJudge
from behaviour_eval.ingest import TrajectoryStore
from behaviour_eval.sdk import BehaviourSDK

logger = logging.getLogger(__name__)

MODEL_NAME = "SyncAgent-Real"
BASELINE_MODEL = "model-alpha"
PROJECT_ID = "booking-sync-real"
EVAL_NOW = datetime(2025, 1, 15, 9, 0, 0)

ACTION_TYPE_MAP = {
    "CREATE": "create",
    "UPDATE": "update",
    "DELETE": "delete",
    "NOOP": "read",
}


def run_real_agent(port: int | None = None) -> str:
    """Capture real SyncAgent trajectories and compare them with model-alpha."""
    del port
    store = TrajectoryStore()
    scenarios = _load_scenarios()
    sdk = BehaviourSDK(project_id=PROJECT_ID, model_id=MODEL_NAME, store=store)

    logger.info("Loaded %d booking scenarios.", len(scenarios))
    logger.info("Executing %s with fixed evaluation clock %s.", MODEL_NAME, EVAL_NOW.isoformat())

    for scenario in scenarios:
        calendar = _seed_calendar(scenario)
        agent = SyncAgent(calendar=calendar, extractor=BaselineExtractor())
        _capture_scenario(agent, calendar, scenario, sdk)

    sdk.flush()
    logger.info("Captured real SyncAgent trajectories.")

    _generate_baseline_trajectories(store, scenarios)

    judge = _build_judge()
    engine = ComparisonEngine(store=store, judge=judge)
    scenario_path = str(_scenario_path())
    run_id = engine.run_existing_comparison(
        scenarios=scenarios,
        models=[MODEL_NAME, BASELINE_MODEL],
        scenario_set=scenario_path,
        project_id=PROJECT_ID,
    )

    logger.info("Real agent evaluation complete. Comparison run: %s", run_id)
    return run_id


def _capture_scenario(
    agent: SyncAgent,
    calendar: FakeCalendar,
    scenario: Dict[str, Any],
    sdk: BehaviourSDK,
) -> None:
    user_task = scenario["user_task"]
    start = time.perf_counter()

    try:
        result = agent.handle_command(user_task, EVAL_NOW)
        latency_ms = (time.perf_counter() - start) * 1000
        trajectory = result.trajectory
        sdk.capture(
            user_task=user_task,
            reasoning=_format_reasoning(trajectory.get("reasoning")),
            tool_calls=_format_tool_calls(trajectory.get("tool_calls", [])),
            actions=_format_actions(trajectory.get("actions", [])),
            outcome=_normalise_outcome(result.status),
            outcome_details=result.message,
            expected_state={
                **scenario.get("expected_state", {}),
                "expected_tool_calls": scenario.get("expected_tool_calls", []),
            },
            actual_state=_calendar_state(calendar),
            latency_ms=latency_ms,
            metadata={
                "source_trajectory_id": trajectory.get("trajectory_id"),
                "agent_model": trajectory.get("agent", {}).get("model"),
                "category": scenario.get("category"),
            },
            scenario_id=scenario.get("scenario_id"),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("Real agent failed on scenario %s", scenario.get("scenario_id"))
        sdk.capture(
            user_task=user_task,
            reasoning=None,
            tool_calls=[],
            actions=[],
            outcome="error",
            outcome_details=str(exc),
            expected_state=scenario.get("expected_state", {}),
            actual_state=_calendar_state(calendar),
            latency_ms=latency_ms,
            metadata={"category": scenario.get("category")},
            scenario_id=scenario.get("scenario_id"),
        )


def _format_reasoning(reasoning: List[Dict[str, Any]] | None) -> str | None:
    if not reasoning:
        return None
    return "\n".join(f"Step {item.get('step')}: {item.get('text')}" for item in reasoning)


def _format_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for call in tool_calls:
        formatted.append(
            {
                "tool_name": call.get("name", "unknown"),
                "arguments": call.get("args", {}),
                "result": call.get("result"),
                "timestamp": call.get("ts", ""),
                "error": call.get("error"),
            }
        )
    return formatted


def _format_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for action in actions:
        raw_type = str(action.get("type", "NOOP")).upper()
        action_type = ACTION_TYPE_MAP.get(raw_type, raw_type.lower())
        destructive = bool(action.get("destructive", False))
        formatted.append(
            {
                "action_type": action_type,
                "target": action.get("target") or "calendar",
                "details": f"{raw_type} {action.get('target') or 'calendar'}",
                "is_destructive": destructive,
                "confirmation_required": destructive,
                "confirmation_obtained": bool(action.get("confirmed_by_user")) if destructive else False,
                "timestamp": action.get("ts", ""),
            }
        )
    return formatted


def _normalise_outcome(status: str) -> str:
    if status == "awaiting_confirmation":
        return "partial"
    if status == "refused":
        return "failure"
    return status


def _build_judge() -> LLMJudge:
    if os.environ.get("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY found; using OpenAI judge.")
        return LLMJudge(model=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini"), provider="openai")
    if os.environ.get("GROQ_API_KEY"):
        logger.info("GROQ_API_KEY found; using Groq judge.")
        return LLMJudge(
            model=os.environ.get("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant"),
            provider="groq",
        )
    logger.info("No supported API key found; using mock judge.")
    return LLMJudge()


def _generate_baseline_trajectories(store: TrajectoryStore, scenarios: List[Dict[str, Any]]) -> None:
    from behaviour_eval.simulate_agent import SimulatedBookingAgent

    agent = SimulatedBookingAgent(model_id=BASELINE_MODEL)
    for scenario in scenarios:
        trajectory = agent.run_scenario(scenario)
        trajectory.project_id = PROJECT_ID
        store.save(trajectory)
    logger.info("Generated %s baseline trajectories for %s.", len(scenarios), BASELINE_MODEL)


def _seed_calendar(scenario: Dict[str, Any]) -> FakeCalendar:
    calendar = FakeCalendar()
    task = scenario.get("user_task", "").lower()

    if any(token in task for token in ("grand palace", "hotel", "march 15", "march 16", "march 18")):
        calendar.seed(
            CalendarEvent(
                event_id="hotel_grand_palace",
                title="Hotel: Grand Palace",
                start_time=datetime(2025, 3, 15, 15, 0),
                end_time=datetime(2025, 3, 17, 11, 0),
                booking_ref="GP-315",
                booking_type=BookingType.HOTEL,
            )
        )

    if any(token in task for token in ("tokyo", "lhr", "nrt", "april 1", "april 10")):
        calendar.seed(
            CalendarEvent(
                event_id="flight_outbound",
                title="Flight: LHR to NRT",
                start_time=datetime(2025, 4, 1, 9, 0),
                booking_ref="LHRNRT-401",
                booking_type=BookingType.FLIGHT,
            )
        )
        calendar.seed(
            CalendarEvent(
                event_id="flight_return",
                title="Flight: NRT to LHR",
                start_time=datetime(2025, 4, 10, 13, 0),
                booking_ref="NRTLHR-410",
                booking_type=BookingType.FLIGHT,
            )
        )

    if any(token in task for token in ("dinner", "ristorante", "tonight")):
        calendar.seed(
            CalendarEvent(
                event_id="dinner_milano",
                title="Dinner: Ristorante Milano",
                start_time=datetime(2025, 2, 14, 19, 30),
                booking_type=BookingType.APPOINTMENT,
            )
        )

    if "seaside resort" in task:
        calendar.seed(
            CalendarEvent(
                event_id="hotel_seaside",
                title="Hotel: Seaside Resort",
                start_time=datetime(2025, 6, 5, 15, 0),
                end_time=datetime(2025, 6, 8, 11, 0),
                booking_ref="RSV-4821",
                booking_type=BookingType.HOTEL,
            )
        )

    if "shinjuku" in task:
        calendar.seed(
            CalendarEvent(
                event_id="hotel_shinjuku",
                title="Hotel: Shinjuku Inn",
                start_time=datetime(2025, 4, 1, 15, 0),
                end_time=datetime(2025, 4, 5, 11, 0),
                booking_type=BookingType.HOTEL,
            )
        )

    if "dentist" in task:
        calendar.seed(
            CalendarEvent(
                event_id="dentist",
                title="Dentist appointment",
                start_time=datetime(2025, 3, 16, 11, 0),
                booking_type=BookingType.APPOINTMENT,
            )
        )

    return calendar


def _calendar_state(calendar: FakeCalendar) -> Dict[str, Any]:
    return {
        "calendar_events": [
            {
                "event_id": event.event_id,
                "title": event.title,
                "start": event.start_time.isoformat(),
                "end": event.end_time.isoformat() if event.end_time else None,
                "booking_ref": event.booking_ref,
                "type": event.booking_type.value if event.booking_type else None,
            }
            for event in calendar.list_events()
        ],
        "mutations": [{"action": action, "event_id": event_id} for action, event_id in calendar.mutations],
    }


def _load_scenarios() -> List[Dict[str, Any]]:
    with _scenario_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scenario_path() -> Path:
    return Path(__file__).resolve().parent / "scenarios" / "booking_scenarios.json"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_real_agent()
