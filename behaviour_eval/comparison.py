"""
Offline Model Comparison & Recommendation Engine (§B.3.3)

Runs a fixed set of scenarios across candidate models, compares their
behaviours pairwise using an automated judge, aggregates the results
into per-model rankings, and issues a recommendation.

This follows a batch-evaluation pattern adapted for agentic behaviour comparison.

Usage:
    engine = ComparisonEngine(store, judge)
    run_id = engine.run_comparison(
        scenario_path="scenarios/booking_scenarios.json",
        models=["gpt-4", "gpt-3.5", "claude-3"],
    )
    results = engine.get_results(run_id)
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from behaviour_eval.trajectory import Trajectory, TrajectoryOutcome
from behaviour_eval.ingest import TrajectoryStore
from behaviour_eval.evaluators import EvaluatorRegistry, get_default_registry
from behaviour_eval.evaluators.llm_judge import LLMJudge, JuryJudge

logger = logging.getLogger(__name__)


class ScenarioSet:
    """
    A fixed set of scenarios for offline evaluation.

    Loads scenarios from a JSON file and provides iteration
    and filtering by category.
    """

    def __init__(self, scenarios: List[Dict[str, Any]]):
        self.scenarios = scenarios

    @classmethod
    def from_file(cls, path: str) -> "ScenarioSet":
        """Load scenarios from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)

    def filter_by_category(self, category: str) -> "ScenarioSet":
        """Filter scenarios by category."""
        filtered = [s for s in self.scenarios if s.get("category") == category]
        return ScenarioSet(filtered)

    @property
    def categories(self) -> List[str]:
        """Get unique categories."""
        return list(set(s.get("category", "unknown") for s in self.scenarios))

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    def __getitem__(self, idx):
        return self.scenarios[idx]


class OfflineBatch:
    """
    Runs a set of scenarios against a simulated agent to generate trajectories.

    In a production setting, this would call the actual agent. For this
    platform, it uses the simulated agent to generate realistic trajectories.
    """

    def __init__(self, agent_fn: Optional[Callable] = None):
        """
        Args:
            agent_fn: A callable that takes (scenario_dict, model_id) and
                      returns a Trajectory. If None, uses the simulated agent.
        """
        self.agent_fn = agent_fn

    def run(
        self,
        scenarios: ScenarioSet,
        model_id: str,
        project_id: str = "comparison",
    ) -> List[Trajectory]:
        """
        Run all scenarios with the given model and collect trajectories.

        Args:
            scenarios: The ScenarioSet to run
            model_id: The model identifier
            project_id: The project identifier

        Returns:
            List of Trajectory objects
        """
        trajectories = []

        for scenario in scenarios:
            try:
                if self.agent_fn:
                    trajectory = self.agent_fn(scenario, model_id)
                else:
                    # Use simulated agent
                    from behaviour_eval.simulate_agent import SimulatedBookingAgent
                    agent = SimulatedBookingAgent(model_id=model_id)
                    trajectory = agent.run_scenario(scenario)

                trajectory.project_id = project_id
                trajectory.scenario_id = scenario["scenario_id"]
                trajectories.append(trajectory)

            except Exception as e:
                logger.error(f"Error running scenario {scenario['scenario_id']} with {model_id}: {e}")
                # Create an error trajectory
                trajectory = Trajectory(
                    project_id=project_id,
                    model_id=model_id,
                    user_task=scenario["user_task"],
                    outcome=TrajectoryOutcome.ERROR.value,
                    outcome_details=str(e),
                    scenario_id=scenario["scenario_id"],
                )
                trajectories.append(trajectory)

        logger.info(f"Generated {len(trajectories)} trajectories for model {model_id}")
        return trajectories


class PairwiseComparison:
    """
    Compares trajectories from two models pairwise on matching scenarios.

    For each scenario, takes the trajectories from model A and model B,
    and uses the judge to determine which is better.
    """

    def __init__(self, judge: Optional[LLMJudge] = None):
        self.judge = judge or LLMJudge()

    def compare(
        self,
        trajectories_a: List[Trajectory],
        trajectories_b: List[Trajectory],
    ) -> Dict[str, Any]:
        """
        Run pairwise comparison for all matching scenarios.

        Returns:
            Dictionary with win counts and per-scenario results
        """
        # Index by scenario_id
        map_a = {t.scenario_id: t for t in trajectories_a if t.scenario_id}
        map_b = {t.scenario_id: t for t in trajectories_b if t.scenario_id}

        common_scenarios = set(map_a.keys()) & set(map_b.keys())

        results = {
            "model_a": trajectories_a[0].model_id if trajectories_a else "A",
            "model_b": trajectories_b[0].model_id if trajectories_b else "B",
            "total_comparisons": len(common_scenarios),
            "wins_a": 0,
            "wins_b": 0,
            "ties": 0,
            "per_scenario": {},
        }

        for scenario_id in common_scenarios:
            traj_a = map_a[scenario_id]
            traj_b = map_b[scenario_id]

            comparison = self.judge.pairwise_compare(traj_a, traj_b)

            if comparison["winner"] == "A":
                results["wins_a"] += 1
            elif comparison["winner"] == "B":
                results["wins_b"] += 1
            else:
                results["ties"] += 1

            results["per_scenario"][scenario_id] = comparison

        return results


class RankingEngine:
    """
    Aggregates pairwise comparison results into per-model rankings
    using a simplified Elo/Bradley-Terry approach.

    Computes:
        - Win rate for each model
        - Elo ratings (starting from 1000)
        - Confidence margins
    """

    INITIAL_ELO = 1000
    K_FACTOR = 32

    def __init__(self):
        self.elo_ratings: Dict[str, float] = {}

    def compute_rankings(
        self, pairwise_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute rankings from a list of pairwise comparison results.

        Args:
            pairwise_results: List of results from PairwiseComparison.compare()

        Returns:
            Rankings dict with per-model stats
        """
        # Initialize Elo ratings
        all_models = set()
        for result in pairwise_results:
            all_models.add(result["model_a"])
            all_models.add(result["model_b"])

        for model in all_models:
            if model not in self.elo_ratings:
                self.elo_ratings[model] = self.INITIAL_ELO

        # Win/loss tracking
        model_stats: Dict[str, Dict[str, int]] = {
            m: {"wins": 0, "losses": 0, "ties": 0, "total": 0}
            for m in all_models
        }

        # Process each pairwise result
        for result in pairwise_results:
            model_a = result["model_a"]
            model_b = result["model_b"]

            model_stats[model_a]["wins"] += result["wins_a"]
            model_stats[model_a]["losses"] += result["wins_b"]
            model_stats[model_a]["ties"] += result["ties"]
            model_stats[model_a]["total"] += result["total_comparisons"]

            model_stats[model_b]["wins"] += result["wins_b"]
            model_stats[model_b]["losses"] += result["wins_a"]
            model_stats[model_b]["ties"] += result["ties"]
            model_stats[model_b]["total"] += result["total_comparisons"]

            # Update Elo ratings
            self._update_elo(model_a, model_b, result)

        # Build ranking
        rankings = []
        for model in all_models:
            stats = model_stats[model]
            total = stats["total"]
            win_rate = stats["wins"] / total if total > 0 else 0

            rankings.append({
                "model_id": model,
                "elo_rating": round(self.elo_ratings[model], 1),
                "win_rate": round(win_rate, 4),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total_comparisons": total,
            })

        # Sort by Elo rating (descending)
        rankings.sort(key=lambda x: x["elo_rating"], reverse=True)

        return {
            "rankings": rankings,
            "model_count": len(rankings),
            "total_comparisons": sum(r["total_comparisons"] for r in rankings) // 2,
        }

    def _update_elo(
        self, model_a: str, model_b: str, result: Dict[str, Any]
    ):
        """Update Elo ratings based on pairwise comparison results."""
        ra = self.elo_ratings[model_a]
        rb = self.elo_ratings[model_b]

        # Expected scores
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 / (1.0 + 10 ** ((ra - rb) / 400.0))

        total = result["total_comparisons"]
        if total == 0:
            return

        # Actual scores (normalised)
        sa = (result["wins_a"] + 0.5 * result["ties"]) / total
        sb = (result["wins_b"] + 0.5 * result["ties"]) / total

        # Update ratings
        self.elo_ratings[model_a] = ra + self.K_FACTOR * (sa - ea)
        self.elo_ratings[model_b] = rb + self.K_FACTOR * (sb - eb)


class Recommender:
    """
    Issues a model recommendation based on ranking results.

    Only recommends when the margin is sufficient and sample size
    is large enough to be statistically meaningful.
    """

    def __init__(
        self,
        min_margin: float = 0.1,
        min_comparisons: int = 10,
        min_elo_gap: float = 30,
    ):
        """
        Args:
            min_margin: Minimum win rate margin to recommend
            min_comparisons: Minimum number of comparisons needed
            min_elo_gap: Minimum Elo gap for recommendation
        """
        self.min_margin = min_margin
        self.min_comparisons = min_comparisons
        self.min_elo_gap = min_elo_gap

    def recommend(self, rankings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Issue a recommendation based on the rankings.

        Returns:
            Dictionary with recommendation and reasoning
        """
        model_rankings = rankings["rankings"]
        total = rankings["total_comparisons"]

        if not model_rankings:
            return {
                "recommended": None,
                "confidence": "none",
                "reasoning": "No models to compare",
            }

        if total < self.min_comparisons:
            return {
                "recommended": None,
                "confidence": "insufficient_data",
                "reasoning": f"Need at least {self.min_comparisons} comparisons, "
                           f"but only have {total}",
            }

        top = model_rankings[0]

        if len(model_rankings) == 1:
            return {
                "recommended": top["model_id"],
                "confidence": "low",
                "reasoning": "Only one model evaluated",
            }

        second = model_rankings[1]
        elo_gap = top["elo_rating"] - second["elo_rating"]
        win_margin = top["win_rate"] - second["win_rate"]

        if elo_gap >= self.min_elo_gap and win_margin >= self.min_margin:
            confidence = "high" if elo_gap >= self.min_elo_gap * 2 else "medium"
            return {
                "recommended": top["model_id"],
                "confidence": confidence,
                "reasoning": (
                    f"{top['model_id']} outperforms {second['model_id']} with "
                    f"Elo gap of {elo_gap:.1f} and win rate margin of {win_margin:.1%}"
                ),
                "elo_gap": round(elo_gap, 1),
                "win_margin": round(win_margin, 4),
                "rankings": model_rankings,
            }

        return {
            "recommended": top["model_id"],
            "confidence": "low",
            "reasoning": (
                f"Top model is {top['model_id']} but margin is not decisive "
                f"(Elo gap: {elo_gap:.1f}, win margin: {win_margin:.1%}). "
                f"Consider running more comparisons."
            ),
            "elo_gap": round(elo_gap, 1),
            "win_margin": round(win_margin, 4),
            "rankings": model_rankings,
        }


class ComparisonEngine:
    """
    Orchestrates the full offline comparison pipeline:
        1. Load scenarios
        2. Run each model on each scenario
        3. Score all trajectories
        4. Compare pairwise
        5. Rank and recommend

    This is the main entry point for offline model comparison.
    """

    def __init__(
        self,
        store: Optional[TrajectoryStore] = None,
        judge: Optional[LLMJudge] = None,
        registry: Optional[EvaluatorRegistry] = None,
        use_jury: bool = False,
    ):
        self.store = store or TrajectoryStore()
        self.judge = judge or LLMJudge()
        self.registry = registry or get_default_registry()
        self.jury = JuryJudge() if use_jury else None
        self.batch = OfflineBatch()
        self.comparer = PairwiseComparison(self.judge)
        self.ranker = RankingEngine()
        self.recommender = Recommender()

    def run_comparison(
        self,
        scenario_path: str,
        models: List[str],
        archetype: str = "tool_using_agent",
    ) -> str:
        """
        Run a full offline comparison.

        Args:
            scenario_path: Path to the scenarios JSON file
            models: List of model identifiers to compare
            archetype: The evaluator archetype to use

        Returns:
            The run_id for this comparison
        """
        run_id = f"comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting comparison run {run_id} with models: {models}")

        # Save run metadata
        self.store.save_comparison_run(
            run_id=run_id,
            scenario_set=scenario_path,
            models=models,
            status="running",
        )

        try:
            # 1. Load scenarios
            scenarios = ScenarioSet.from_file(scenario_path)
            logger.info(f"Loaded {len(scenarios)} scenarios from {scenario_path}")

            # 2. Run each model
            model_trajectories: Dict[str, List[Trajectory]] = {}
            for model_id in models:
                logger.info(f"Running scenarios for model: {model_id}")
                trajectories = self.batch.run(scenarios, model_id, project_id=run_id)

                # 3. Score all trajectories
                evaluator = self.registry.get(archetype)
                for traj in trajectories:
                    if evaluator:
                        scores = evaluator.score(traj)
                        traj.scores = scores
                    # Save to store
                    self.store.save(traj)

                model_trajectories[model_id] = trajectories

            # 4. Pairwise comparison
            pairwise_results = []
            model_list = list(models)
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    model_a = model_list[i]
                    model_b = model_list[j]
                    logger.info(f"Comparing {model_a} vs {model_b}")
                    result = self.comparer.compare(
                        model_trajectories[model_a],
                        model_trajectories[model_b],
                    )
                    pairwise_results.append(result)

            # 5. Rank and recommend
            rankings = self.ranker.compute_rankings(pairwise_results)
            recommendation = self.recommender.recommend(rankings)

            # Compile full results
            full_results = {
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "scenarios_count": len(scenarios),
                "models": models,
                "pairwise_results": pairwise_results,
                "rankings": rankings,
                "recommendation": recommendation,
                "per_model_avg_scores": {},
            }

            # Compute per-model average scores
            for model_id, trajectories in model_trajectories.items():
                all_scores = [t.scores for t in trajectories if t.scores]
                if all_scores:
                    avg_scores = {}
                    for key in all_scores[0]:
                        values = [s[key] for s in all_scores if key in s]
                        avg_scores[key] = round(sum(values) / len(values), 4)
                    full_results["per_model_avg_scores"][model_id] = avg_scores

            # Update run status
            self.store.save_comparison_run(
                run_id=run_id,
                scenario_set=scenario_path,
                models=models,
                results=full_results,
                recommendation=json.dumps(recommendation),
                status="completed",
            )

            logger.info(f"Comparison run {run_id} completed. Recommendation: {recommendation}")
            return run_id

        except Exception as e:
            logger.error(f"Comparison run {run_id} failed: {e}")
            self.store.save_comparison_run(
                run_id=run_id,
                scenario_set=scenario_path,
                models=models,
                status=f"failed: {str(e)}",
            )
            raise

    def run_existing_comparison(
        self,
        scenarios: List[Dict[str, Any]],
        models: List[str],
        scenario_set: str = "stored",
        project_id: Optional[str] = None,
        archetype: str = "tool_using_agent",
    ) -> str:
        """
        Compare trajectories that have already been captured in the store.

        This is used when one of the candidates is a real agent that cannot be
        generated by the simulated OfflineBatch runner. Each model must have at
        least one stored trajectory for at least one common scenario_id.
        """
        run_id = f"comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting stored comparison run {run_id} with models: {models}")
        self.store.save_comparison_run(
            run_id=run_id,
            scenario_set=scenario_set,
            models=models,
            status="running",
        )

        try:
            scenario_ids = {s["scenario_id"] for s in scenarios if s.get("scenario_id")}
            evaluator = self.registry.get(archetype)
            model_trajectories: Dict[str, List[Trajectory]] = {}

            for model_id in models:
                trajectories = self.store.list_trajectories(
                    project_id=project_id,
                    model_id=model_id,
                    limit=10000,
                )
                filtered = [t for t in trajectories if t.scenario_id in scenario_ids]
                latest_by_scenario: Dict[str, Trajectory] = {}
                for trajectory in sorted(filtered, key=lambda t: t.timestamp):
                    latest_by_scenario[trajectory.scenario_id] = trajectory

                if not latest_by_scenario:
                    raise ValueError(f"No stored trajectories found for model {model_id}")

                scored = list(latest_by_scenario.values())
                for traj in scored:
                    if evaluator and not traj.scores:
                        traj.scores = evaluator.score(traj)
                        self.store.save(traj)
                model_trajectories[model_id] = scored

            pairwise_results = []
            model_list = list(models)
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    model_a = model_list[i]
                    model_b = model_list[j]
                    logger.info(f"Comparing stored trajectories: {model_a} vs {model_b}")
                    result = self.comparer.compare(
                        model_trajectories[model_a],
                        model_trajectories[model_b],
                    )
                    if result["total_comparisons"] == 0:
                        raise ValueError(f"No common scenarios for {model_a} and {model_b}")
                    pairwise_results.append(result)

            rankings = self.ranker.compute_rankings(pairwise_results)
            recommendation = self.recommender.recommend(rankings)
            full_results = {
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "scenarios_count": len(scenarios),
                "models": models,
                "pairwise_results": pairwise_results,
                "rankings": rankings,
                "recommendation": recommendation,
                "per_model_avg_scores": {},
            }

            for model_id, trajectories in model_trajectories.items():
                all_scores = [t.scores for t in trajectories if t.scores]
                if all_scores:
                    avg_scores = {}
                    for key in all_scores[0]:
                        values = [s[key] for s in all_scores if key in s]
                        avg_scores[key] = round(sum(values) / len(values), 4)
                    full_results["per_model_avg_scores"][model_id] = avg_scores

            self.store.save_comparison_run(
                run_id=run_id,
                scenario_set=scenario_set,
                models=models,
                results=full_results,
                recommendation=json.dumps(recommendation),
                status="completed",
            )
            logger.info(f"Stored comparison run {run_id} completed. Recommendation: {recommendation}")
            return run_id
        except Exception as e:
            logger.error(f"Stored comparison run {run_id} failed: {e}")
            self.store.save_comparison_run(
                run_id=run_id,
                scenario_set=scenario_set,
                models=models,
                status=f"failed: {str(e)}",
            )
            raise

    def get_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of a comparison run."""
        runs = self.store.get_comparison_runs()
        for run in runs:
            if run["run_id"] == run_id:
                return run
        return None

    def get_win_rate_matrix(self, run_id: str) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get a win rate matrix for visualisation.

        Returns a dict of model_a -> model_b -> win_rate_of_a_vs_b
        """
        run = self.get_results(run_id)
        if not run or not run["results"]:
            return None

        results = run["results"]
        models = results["models"]
        matrix = {m: {m2: 0.5 for m2 in models} for m in models}

        for pw in results.get("pairwise_results", []):
            model_a = pw["model_a"]
            model_b = pw["model_b"]
            total = pw["total_comparisons"]
            if total > 0:
                matrix[model_a][model_b] = pw["wins_a"] / total
                matrix[model_b][model_a] = pw["wins_b"] / total

        return matrix
