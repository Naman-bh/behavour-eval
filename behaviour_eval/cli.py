import os
import sys
import argparse
import logging
from pathlib import Path

# Configure basic logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("behaviour_eval.cli")

def main():
    parser = argparse.ArgumentParser(description="Agentic Behaviour-Evaluation Platform CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    subparsers.add_parser("ingest", help="Run the ingest service (interactive test mode)")

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start the dashboard web server")
    dash_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")

    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Run an offline model comparison")
    comp_parser.add_argument("--scenarios", type=str, required=True, help="Path to scenarios JSON")
    comp_parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model IDs")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Generate synthetic data, run comparison, and start dashboard")
    demo_parser.add_argument("--models", type=str, default="model-alpha,model-beta,model-gamma,model-delta", help="Comma-separated list of models to simulate")
    demo_parser.add_argument("--count", type=int, default=100, help="Number of trajectories to generate")

    # Run real command
    real_parser = subparsers.add_parser("run-real", help="Run real agent evaluation")
    real_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")

    args = parser.parse_args()

    if args.command == "run-real":
        from behaviour_eval.run_real_agent import run_real_agent
        logger.info("Starting real agent pipeline...")
        
        # Reset DB before run
        from behaviour_eval.ingest import TrajectoryStore
        store = TrajectoryStore()
        logger.info("Clearing database...")
        store.clear()
        
        run_real_agent()
        
        logger.info("Real agent pipeline complete!")
        
        from behaviour_eval.dashboard.server import app
        logger.info(f"Starting dashboard on http://0.0.0.0:{args.port}")
        app.run(host="0.0.0.0", port=args.port, debug=False)

    elif args.command == "dashboard":
        from behaviour_eval.dashboard.server import run_server
        run_server(port=args.port)

    elif args.command == "compare":
        from behaviour_eval.comparison import ComparisonEngine
        models = [m.strip() for m in args.models.split(",")]
        engine = ComparisonEngine()
        engine.run_comparison(args.scenarios, models)

    elif args.command == "demo":
        logger.info("Starting demo pipeline...")
        
        # 1. Clear database
        from behaviour_eval.ingest import TrajectoryStore
        store = TrajectoryStore()
        logger.info("Clearing database...")
        store.clear()

        # 2. Generate synthetic data
        from behaviour_eval.simulate_agent import generate_demo_trajectories
        models = [m.strip() for m in args.models.split(",")]
        logger.info(f"Generating {args.count} synthetic trajectories for models: {models}...")
        generate_demo_trajectories(store, num_trajectories=args.count, models=models)

        # 3. Score all trajectories
        logger.info("Scoring trajectories using tool_using_agent evaluator...")
        from behaviour_eval.evaluators import get_default_registry
        registry = get_default_registry()
        evaluator = registry.get("tool_using_agent")
        
        trajectories = store.list_trajectories(limit=1000)
        for t in trajectories:
            t.scores = evaluator.score(t)
            store.save(t)
        
        # 4. Run an offline comparison batch
        logger.info("Running offline model comparison batch...")
        from behaviour_eval.comparison import ComparisonEngine
        engine = ComparisonEngine(store=store)
        scenario_path = Path(__file__).parent / "scenarios" / "booking_scenarios.json"
        engine.run_comparison(str(scenario_path), models)

        logger.info("Demo data generation complete!")
        
        # 5. Start dashboard
        from behaviour_eval.dashboard.server import run_server
        run_server(port=8080)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
