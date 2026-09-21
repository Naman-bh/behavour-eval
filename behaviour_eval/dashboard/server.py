import os
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from behaviour_eval.ingest import TrajectoryStore
from behaviour_eval.comparison import ComparisonEngine
from behaviour_eval.research.rq1_reasoning import generate_rq1_report
from behaviour_eval.research.rq2_judge_reliability import generate_rq2_report

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")

# Make store available to app
store = TrajectoryStore()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/projects")
def api_projects():
    # Simplification: just return projects that exist in DB
    conn = store._get_conn()
    rows = conn.execute("SELECT DISTINCT project_id FROM trajectories").fetchall()
    conn.close()
    return jsonify({"projects": [r["project_id"] for r in rows]})


@app.route("/api/stats")
def api_stats():
    project_id = request.args.get("project")
    if project_id == "all":
        project_id = None
    stats = store.get_stats(project_id)
    return jsonify(stats)


@app.route("/api/trajectories")
def api_trajectories():
    project_id = request.args.get("project")
    if project_id == "all":
        project_id = None
    model_id = request.args.get("model")
    outcome = request.args.get("outcome")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    trajectories = store.list_trajectories(
        project_id=project_id,
        model_id=model_id,
        outcome=outcome,
        limit=limit,
        offset=offset,
    )
    return jsonify({"trajectories": [t.to_dict() for t in trajectories]})


@app.route("/api/trajectory/<trajectory_id>")
def api_trajectory(trajectory_id):
    trajectory = store.get(trajectory_id)
    if not trajectory:
        return jsonify({"error": "Trajectory not found"}), 404
    return jsonify(trajectory.to_dict())


@app.route("/api/comparisons")
def api_comparisons():
    runs = store.get_comparison_runs()
    return jsonify({"runs": runs})


@app.route("/api/comparison/<run_id>")
def api_comparison(run_id):
    engine = ComparisonEngine(store=store)
    result = engine.get_results(run_id)
    if not result:
        return jsonify({"error": "Comparison run not found"}), 404
    
    matrix = engine.get_win_rate_matrix(run_id)
    result["win_rate_matrix"] = matrix
    return jsonify(result)


@app.route("/api/rq1/reasoning_analysis")
def api_rq1():
    # Use all trajectories for RQ1 analysis
    trajectories = store.list_trajectories(limit=1000)
    report = generate_rq1_report(trajectories)
    return jsonify(report)


@app.route("/api/rq2/judge_reliability")
def api_rq2():
    # Use all trajectories for RQ2 analysis
    trajectories = store.list_trajectories(limit=100)
    report = generate_rq2_report(trajectories)
    return jsonify(report)


def run_server(host="0.0.0.0", port=8080):
    logger.info(f"Starting dashboard on http://{host}:{port}")
    # Disable werkzeug logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    run_server()
