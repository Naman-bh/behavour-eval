"""
RQ2: Reliability of Automated Judgment

Measures how reliably automated evaluators approximate human judgment.

This module:
    1. Loads human-labelled ground truth (50-100 annotated trajectories)
    2. Compares single LLM judge scores vs. human labels
    3. Compares jury (multi-model) scores vs. human labels
    4. Computes Cohen's κ, accuracy, and other agreement metrics
    5. Tests: (a) pairwise > absolute, (b) jury > single judge

Key findings expected:
    - Agreement level between automated and human scores (Cohen's κ)
    - Pairwise preference tracks human preference more closely
    - Jury-based evaluation reduces single-judge bias
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from behaviour_eval.trajectory import Trajectory

logger = logging.getLogger(__name__)


def cohens_kappa(labels_a: List[int], labels_b: List[int]) -> float:
    """
    Compute Cohen's kappa inter-rater agreement.

    Args:
        labels_a: First rater's categorical labels
        labels_b: Second rater's categorical labels

    Returns:
        Kappa value in [-1, 1]. >0.6 is substantial, >0.8 is near-perfect.
    """
    assert len(labels_a) == len(labels_b), "Label lists must have same length"
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Get all unique categories
    categories = sorted(set(labels_a) | set(labels_b))
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        matrix[cat_to_idx[a]][cat_to_idx[b]] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement (by chance)
    pe = sum(
        (sum(matrix[i][j] for j in range(k)) * sum(matrix[j][i] for j in range(k)))
        for i in range(k)
    ) / (n * n)

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)


def accuracy(predicted: List[Any], actual: List[Any]) -> float:
    """Compute accuracy (fraction of matching predictions)."""
    if not predicted:
        return 0.0
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return round(correct / len(predicted), 4)


def discretize_score(score: float, thresholds: Tuple[float, ...] = (0.3, 0.7)) -> int:
    """
    Convert a continuous score [0,1] to a categorical label.

    Default thresholds: <0.3 = 0 (bad), 0.3-0.7 = 1 (ok), >0.7 = 2 (good)
    """
    for i, threshold in enumerate(thresholds):
        if score < threshold:
            return i
    return len(thresholds)


def load_human_labels(path: str) -> List[Dict[str, Any]]:
    """Load human-annotated trajectory labels from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def generate_synthetic_human_labels(
    trajectories: List[Trajectory],
    sample_size: int = 80,
    annotators: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic human labels for testing the RQ2 pipeline.

    In a real study, these would be actual human annotations.
    This function simulates annotator behaviour with realistic
    inter-annotator disagreement.
    """
    import random

    sampled = random.sample(trajectories, min(sample_size, len(trajectories)))
    labels = []

    for traj in sampled:
        # Base quality from the trajectory scores
        if traj.scores:
            base_tc = traj.scores.get("task_completion", 0.5)
            base_tc_correct = traj.scores.get("tool_use_correctness", 0.5)
            base_safety = traj.scores.get("destructive_action_safety", 0.5)
            base_intent = traj.scores.get("intent_accuracy", 0.5)
        else:
            base_tc = 0.5
            base_tc_correct = 0.5
            base_safety = 0.5
            base_intent = 0.5

        annotator_scores = []
        for a in range(annotators):
            # Add realistic noise to simulate annotator disagreement
            noise = random.gauss(0, 0.12)
            annotator_scores.append({
                "annotator_id": f"annotator_{a+1}",
                "task_completion": max(0, min(1, base_tc + noise)),
                "tool_use_correctness": max(0, min(1, base_tc_correct + random.gauss(0, 0.1))),
                "destructive_action_safety": max(0, min(1, base_safety + random.gauss(0, 0.08))),
                "intent_accuracy": max(0, min(1, base_intent + random.gauss(0, 0.1))),
                "overall_quality": max(0, min(1, (base_tc + base_tc_correct + base_safety + base_intent) / 4 + noise)),
            })

        # Consensus = average of annotators
        consensus = {}
        metrics = ["task_completion", "tool_use_correctness", "destructive_action_safety",
                   "intent_accuracy", "overall_quality"]
        for metric in metrics:
            values = [a[metric] for a in annotator_scores]
            consensus[metric] = round(sum(values) / len(values), 4)

        labels.append({
            "trajectory_id": traj.trajectory_id,
            "user_task": traj.user_task,
            "model_id": traj.model_id,
            "annotator_scores": annotator_scores,
            "consensus": consensus,
        })

    return labels


def evaluate_single_judge(
    trajectories: List[Trajectory],
    human_labels: List[Dict[str, Any]],
    judge,
    metrics: List[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single judge's agreement with human labels.

    Args:
        trajectories: List of Trajectory objects
        human_labels: List of human label dicts (with 'trajectory_id' and 'consensus')
        judge: An LLMJudge or JuryJudge instance
        metrics: Which metrics to evaluate

    Returns:
        Agreement results per metric
    """
    if metrics is None:
        metrics = ["task_completion", "tool_use_correctness",
                   "destructive_action_safety", "intent_accuracy"]

    # Build lookup
    traj_map = {t.trajectory_id: t for t in trajectories}
    label_map = {l["trajectory_id"]: l for l in human_labels}

    results = {}

    for metric in metrics:
        human_vals = []
        judge_vals = []
        human_cats = []
        judge_cats = []

        for tid, label in label_map.items():
            if tid not in traj_map:
                continue

            traj = traj_map[tid]
            human_score = label["consensus"].get(metric, 0.5)

            # Get judge score
            judge_scores = judge.score(traj)
            judge_score = judge_scores.get(metric, 0.5)

            human_vals.append(human_score)
            judge_vals.append(judge_score)

            # Discretise for Cohen's kappa
            human_cats.append(discretize_score(human_score))
            judge_cats.append(discretize_score(judge_score))

        if not human_vals:
            results[metric] = {"error": "No matching data"}
            continue

        # Compute metrics
        kappa = cohens_kappa(human_cats, judge_cats)
        acc = accuracy(judge_cats, human_cats)

        # Mean absolute error
        mae = sum(abs(h - j) for h, j in zip(human_vals, judge_vals)) / len(human_vals)

        # Pearson correlation (simplified)
        mean_h = sum(human_vals) / len(human_vals)
        mean_j = sum(judge_vals) / len(judge_vals)
        cov = sum((h - mean_h) * (j - mean_j) for h, j in zip(human_vals, judge_vals)) / len(human_vals)
        std_h = math.sqrt(sum((h - mean_h) ** 2 for h in human_vals) / len(human_vals))
        std_j = math.sqrt(sum((j - mean_j) ** 2 for j in judge_vals) / len(judge_vals))
        pearson = cov / (std_h * std_j) if (std_h > 0 and std_j > 0) else 0

        results[metric] = {
            "cohens_kappa": kappa,
            "accuracy": acc,
            "mean_absolute_error": round(mae, 4),
            "pearson_correlation": round(pearson, 4),
            "sample_size": len(human_vals),
        }

    return results


def evaluate_pairwise_vs_absolute(
    trajectories: List[Trajectory],
    human_labels: List[Dict[str, Any]],
    judge,
) -> Dict[str, Any]:
    """
    Test hypothesis: pairwise preference is more reliable than absolute scoring.

    Compares judge agreement with human preferences in pairwise setup
    vs. agreement with human absolute scores.
    """
    import random

    label_map = {l["trajectory_id"]: l for l in human_labels}
    traj_map = {t.trajectory_id: t for t in trajectories}

    # Generate pairwise comparisons from human labels
    labeled_tids = [tid for tid in label_map if tid in traj_map]

    if len(labeled_tids) < 4:
        return {"error": "Not enough labeled trajectories for pairwise analysis"}

    # Create pairs
    pairs = []
    random.shuffle(labeled_tids)
    for i in range(0, len(labeled_tids) - 1, 2):
        pairs.append((labeled_tids[i], labeled_tids[i + 1]))

    pairwise_agreements = 0
    absolute_agreements = 0
    total_pairs = 0

    for tid_a, tid_b in pairs:
        traj_a = traj_map[tid_a]
        traj_b = traj_map[tid_b]

        human_a = label_map[tid_a]["consensus"]["overall_quality"]
        human_b = label_map[tid_b]["consensus"]["overall_quality"]

        # Human pairwise preference
        if human_a > human_b + 0.05:
            human_pref = "A"
        elif human_b > human_a + 0.05:
            human_pref = "B"
        else:
            human_pref = "tie"

        # Judge pairwise comparison
        judge_pairwise = judge.pairwise_compare(traj_a, traj_b)
        judge_pref = judge_pairwise["winner"]

        # Judge absolute comparison
        judge_scores_a = judge.score(traj_a)
        judge_scores_b = judge.score(traj_b)
        metric_keys = [k for k in judge_scores_a if k != "latency"]
        avg_a = sum(judge_scores_a.get(k, 0) for k in metric_keys) / len(metric_keys) if metric_keys else 0
        avg_b = sum(judge_scores_b.get(k, 0) for k in metric_keys) / len(metric_keys) if metric_keys else 0

        if avg_a > avg_b + 0.05:
            absolute_pref = "A"
        elif avg_b > avg_a + 0.05:
            absolute_pref = "B"
        else:
            absolute_pref = "tie"

        # Check agreement
        if judge_pref == human_pref:
            pairwise_agreements += 1
        if absolute_pref == human_pref:
            absolute_agreements += 1
        total_pairs += 1

    pairwise_accuracy = pairwise_agreements / total_pairs if total_pairs > 0 else 0
    absolute_accuracy = absolute_agreements / total_pairs if total_pairs > 0 else 0

    return {
        "total_pairs": total_pairs,
        "pairwise_accuracy": round(pairwise_accuracy, 4),
        "absolute_accuracy": round(absolute_accuracy, 4),
        "pairwise_better": pairwise_accuracy > absolute_accuracy,
        "margin": round(pairwise_accuracy - absolute_accuracy, 4),
        "hypothesis_supported": pairwise_accuracy > absolute_accuracy,
    }


def generate_rq2_report(
    trajectories: List[Trajectory],
    human_labels: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generate the complete RQ2 report.

    If human_labels is None, generates synthetic ones for demo purposes.
    """
    from behaviour_eval.evaluators.llm_judge import LLMJudge, JuryJudge

    # Generate synthetic human labels if needed
    if human_labels is None:
        human_labels = generate_synthetic_human_labels(trajectories)

    single_judge = LLMJudge(model="mock_single", provider="mock", judge_id="single_judge")
    jury_judge = JuryJudge()

    # 1. Single judge vs human
    single_agreement = evaluate_single_judge(
        trajectories, human_labels, single_judge
    )

    # 2. Jury vs human
    jury_agreement = evaluate_single_judge(
        trajectories, human_labels, jury_judge
    )

    # 3. Pairwise vs absolute (for single judge)
    pairwise_analysis = evaluate_pairwise_vs_absolute(
        trajectories, human_labels, single_judge
    )

    # 4. Pairwise vs absolute (for jury)
    pairwise_analysis_jury = evaluate_pairwise_vs_absolute(
        trajectories, human_labels, jury_judge
    )

    # Compute summary statistics
    single_kappas = [
        v.get("cohens_kappa", 0)
        for v in single_agreement.values()
        if isinstance(v, dict) and "cohens_kappa" in v
    ]
    jury_kappas = [
        v.get("cohens_kappa", 0)
        for v in jury_agreement.values()
        if isinstance(v, dict) and "cohens_kappa" in v
    ]

    avg_single_kappa = sum(single_kappas) / len(single_kappas) if single_kappas else 0
    avg_jury_kappa = sum(jury_kappas) / len(jury_kappas) if jury_kappas else 0

    jury_better = avg_jury_kappa > avg_single_kappa

    # Interpret kappa
    def interpret_kappa(k):
        if k < 0:
            return "poor (worse than chance)"
        elif k < 0.2:
            return "slight"
        elif k < 0.4:
            return "fair"
        elif k < 0.6:
            return "moderate"
        elif k < 0.8:
            return "substantial"
        else:
            return "near-perfect"

    return {
        "title": "RQ2: Reliability of Automated Judgment",
        "summary": (
            f"Average Cohen's κ: single judge = {avg_single_kappa:.3f} "
            f"({interpret_kappa(avg_single_kappa)}), "
            f"jury = {avg_jury_kappa:.3f} ({interpret_kappa(avg_jury_kappa)}). "
            f"Jury {'outperforms' if jury_better else 'does not outperform'} single judge. "
            f"Pairwise comparison {'is' if pairwise_analysis.get('hypothesis_supported') else 'is not'} "
            f"more reliable than absolute scoring."
        ),
        "single_judge_agreement": single_agreement,
        "jury_agreement": jury_agreement,
        "pairwise_vs_absolute_single": pairwise_analysis,
        "pairwise_vs_absolute_jury": pairwise_analysis_jury,
        "hypotheses": {
            "h1_pairwise_better_than_absolute": pairwise_analysis.get("hypothesis_supported", False),
            "h1_margin": pairwise_analysis.get("margin", 0),
            "h2_jury_better_than_single": jury_better,
            "h2_kappa_improvement": round(avg_jury_kappa - avg_single_kappa, 4),
        },
        "overall_assessment": {
            "avg_single_kappa": round(avg_single_kappa, 4),
            "avg_jury_kappa": round(avg_jury_kappa, 4),
            "single_kappa_interpretation": interpret_kappa(avg_single_kappa),
            "jury_kappa_interpretation": interpret_kappa(avg_jury_kappa),
            "human_label_count": len(human_labels),
        },
    }
