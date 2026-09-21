"""
Agentic Behaviour-Evaluation Platform

A self-hosted platform that captures, scores, and compares AI agent behaviour.
Built as a generalised evaluation harness for
evaluating tool-using agents across multiple behaviour axes.

Components:
    - trajectory: Core data schema for agent interactions
    - sdk: Instrumentation SDK for capturing agent behaviour
    - ingest: Event normalisation and storage service
    - evaluators: Archetype-based behaviour scoring registry
    - comparison: Offline model comparison and recommendation
    - research: RQ1 (reasoning observability) and RQ2 (judge reliability)
    - dashboard: Web-based visualisation and drill-down
"""

__version__ = "1.0.0"
__project__ = "Agentic Behaviour-Evaluation Platform"
