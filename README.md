# Agentic Behaviour-Evaluation Platform

A self-hosted platform that evaluates the behaviour of non-deterministic, multi-step, tool-using AI agents.

## Overview

AI agents that take real-world actions are being deployed faster than the tools to understand how they behave. Conventional testing does not capture whether an agent chooses correct and safe actions. This project proposes a self-hosted platform that evaluates the behaviour of such agents.

The platform captures how the agent responds to user tasks, scores that behaviour along axes appropriate to a tool-using agent, and runs offline comparisons across candidate models to recommend better ones.

## Research Questions
1. **RQ1**: Is self-reported reasoning a faithful observability signal?
2. **RQ2**: How reliably does automated judgment approximate human judgment?

## Components

- `behaviour_eval/trajectory.py`: Core data schema for capturing agent interactions.
- `behaviour_eval/sdk.py`: SDK for instrumenting AI agents.
- `behaviour_eval/evaluators/`: Archetype-based scoring registries (e.g., Tool-Using Agent).
- `behaviour_eval/comparison.py`: Offline comparison engine generating Elo rankings.
- `behaviour_eval/dashboard/`: Web-based visualisation dashboard.
- `behaviour_eval/research/`: Automated pipelines for answering the research questions.

## Running the Platform

To install dependencies:
```bash
pip install -e .
```

To run a demo of the pipeline (which generates data, scores it, and launches the dashboard):
```bash
python -m behaviour_eval.cli demo
```

To evaluate the real `SyncAgent` from `studious-octo-parakeet`, compare it against
the synthetic `model-alpha` baseline, and launch the dashboard:
```bash
python -m behaviour_eval.cli run-real
```

The real-agent run uses the mock judge by default. To use an OpenAI judge instead,
set `OPENAI_API_KEY` first. You can also set `OPENAI_JUDGE_MODEL`; otherwise the
runner uses `gpt-4o-mini`. Groq is also supported through its OpenAI-compatible
endpoint: set `GROQ_API_KEY` and optionally `GROQ_JUDGE_MODEL` (default:
`llama-3.1-8b-instant`).

To run just the dashboard on an existing database:
```bash
python -m behaviour_eval.cli dashboard --port 8080
```
