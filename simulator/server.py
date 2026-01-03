# mmbn_sim/server.py
#!/usr/bin/env python3
from __future__ import annotations

import threading
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from simcore.serialize import view_state
from simcore.tree import TreeStore

app = Flask(__name__)
_LOCK = threading.Lock()
_TREE = TreeStore()


def _payload() -> Dict[str, Any]:
    cursor = _TREE.cursor
    return {
        "p1": view_state(cursor, "P1"),
        "p2": view_state(cursor, "P2"),
        "tree": {
            "root_id": _TREE.root_id,
            "current_id": _TREE.current_id,
            "cursor_base_id": _TREE.cursor_base_id,
        },
        "mcts": _TREE.mcts_summary_current(),
        "plan": _TREE.plan_json(),
    }


@app.get("/")
def index():
    return render_template("sim.html")


@app.get("/api/state")
def api_state():
    with _LOCK:
        return jsonify(_payload())


@app.post("/api/reset")
def api_reset():
    with _LOCK:
        _TREE.reset()
        return jsonify(_payload())


@app.post("/api/ui_action")
def api_ui_action():
    data = request.get_json(silent=True) or {}
    actor = data.get("actor", None)
    action = data.get("action", None)

    if actor not in ("P1", "P2"):
        return jsonify({"error": "actor must be 'P1' or 'P2'"}), 400
    if not isinstance(action, str):
        return jsonify({"error": "action must be a string"}), 400

    with _LOCK:
        try:
            if action == "TOGGLE_HOLD":
                _TREE.cursor.cursor_toggle_hold(actor)
            else:
                _TREE.cursor_apply_control(actor, action)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        return jsonify(_payload())


@app.post("/api/commit")
def api_commit():
    with _LOCK:
        _TREE.commit_cursor_step()
        return jsonify(_payload())


# ---------------- MCTS API ----------------

@app.post("/api/mcts/run")
def api_mcts_run():
    data = request.get_json(silent=True) or {}
    iterations = data.get("iterations", 400)
    max_depth = data.get("max_depth", 10)
    seed = data.get("seed", 0)

    try:
        iterations = int(iterations)
        max_depth = int(max_depth)
        seed = int(seed)
    except Exception:
        return jsonify({"error": "iterations/max_depth/seed must be ints"}), 400

    iterations = max(0, min(20000, iterations))
    max_depth = max(1, min(120, max_depth))

    with _LOCK:
        try:
            summary = _TREE.run_mcts_from_current(
                iterations=iterations,
                max_depth=max_depth,
                seed=seed,
                target_cust=None,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        payload = _payload()
        payload["mcts_last_run"] = {
            "iterations": iterations,
            "max_depth": max_depth,
            "seed": seed,
            "summary": summary,
        }
        return jsonify(payload)


@app.post("/api/mcts/apply")
def api_mcts_apply():
    data = request.get_json(silent=True) or {}
    who = data.get("who", None)  # "P1"|"P2"|"BOTH"
    if who not in ("P1", "P2", "BOTH"):
        return jsonify({"error": "who must be 'P1', 'P2', or 'BOTH'"}), 400

    with _LOCK:
        summary = _TREE.mcts_summary_current()
        p1_best = summary["p1_maximin"]["best"]
        p2_best = summary["p2_minimax"]["best"]

        try:
            if who in ("P1", "BOTH"):
                _TREE.cursor_apply_control("P1", p1_best)
            if who in ("P2", "BOTH"):
                _TREE.cursor_apply_control("P2", p2_best)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        payload = _payload()
        payload["mcts_applied"] = {"who": who, "p1": p1_best, "p2": p2_best}
        return jsonify(payload)


# ---------------- Planning + Replay API ----------------

@app.post("/api/plan/run")
def api_plan_run():
    data = request.get_json(silent=True) or {}
    target_cust = data.get("target_cust", 64)
    iters_per_step = data.get("iters_per_step", 800)
    lookahead_depth = data.get("lookahead_depth", 16)
    seed = data.get("seed", 0)

    try:
        target_cust = int(target_cust)
        iters_per_step = int(iters_per_step)
        lookahead_depth = int(lookahead_depth)
        seed = int(seed)
    except Exception:
        return jsonify({"error": "target_cust/iters_per_step/lookahead_depth/seed must be ints"}), 400

    target_cust = max(1, min(256, target_cust))
    iters_per_step = max(0, min(20000, iters_per_step))
    lookahead_depth = max(1, min(120, lookahead_depth))

    with _LOCK:
        try:
            plan = _TREE.plan_to_cust(
                target_cust=target_cust,
                iters_per_step=iters_per_step,
                lookahead_depth=lookahead_depth,
                seed=seed,
                time_penalty=0.002,
                jitter_eps=1e-4,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        payload = _payload()
        payload["plan_last_run"] = {
            "target_cust": target_cust,
            "iters_per_step": iters_per_step,
            "lookahead_depth": lookahead_depth,
            "seed": seed,
            "steps": len(plan.steps),
        }
        return jsonify(payload)


@app.post("/api/plan/replay_reset")
def api_plan_replay_reset():
    with _LOCK:
        _TREE.replay_reset()
        return jsonify(_payload())


@app.post("/api/plan/replay_step")
def api_plan_replay_step():
    with _LOCK:
        _TREE.replay_step()
        return jsonify(_payload())


@app.post("/api/plan/replay_set_index")
def api_plan_replay_set_index():
    data = request.get_json(silent=True) or {}
    idx = data.get("index", 0)
    try:
        idx = int(idx)
    except Exception:
        return jsonify({"error": "index must be int"}), 400

    with _LOCK:
        _TREE.replay_set_index(idx)
        return jsonify(_payload())


# ---------------- Tree API ----------------

@app.get("/api/tree/subtree")
def api_tree_subtree():
    node_id = request.args.get("node_id", "").strip()
    depth = request.args.get("depth", "4").strip()
    if not node_id:
        node_id = "ROOT"

    with _LOCK:
        try:
            if node_id == "ROOT":
                node_id = _TREE.root_id
            payload = _TREE.subtree(node_id=node_id, depth=int(depth))
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(payload)


@app.post("/api/tree/set_current")
def api_tree_set_current():
    data = request.get_json(silent=True) or {}
    node_id = data.get("node_id", None)
    if not isinstance(node_id, str) or not node_id.strip():
        return jsonify({"error": "node_id must be a non-empty string"}), 400

    with _LOCK:
        try:
            _TREE.set_current(node_id.strip())
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        return jsonify(_payload())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
