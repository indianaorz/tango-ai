# mmbn_sim/server.py
#!/usr/bin/env python3
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

from simcore.actions import ACT_HOLD_OFF, ACT_HOLD_ON, ACT_NOOP
from simcore.serialize import view_state
from simcore.tree import TreeStore

app = Flask(__name__)
_LOCK = threading.Lock()
_TREE = TreeStore()


@app.get("/")
def index():
    return render_template("sim.html")


@app.get("/api/state")
def api_state():
    with _LOCK:
        cursor = _TREE.cursor
        payload = {
            "p1": view_state(cursor, "P1"),
            "p2": view_state(cursor, "P2"),
            "tree": {
                "root_id": _TREE.root_id,
                "current_id": _TREE.current_id,
                "cursor_base_id": _TREE.cursor_base_id,
            },
        }
        return jsonify(payload)


@app.post("/api/reset")
def api_reset():
    with _LOCK:
        _TREE.reset()
        cursor = _TREE.cursor
        return jsonify(
            {
                "p1": view_state(cursor, "P1"),
                "p2": view_state(cursor, "P2"),
                "tree": {
                    "root_id": _TREE.root_id,
                    "current_id": _TREE.current_id,
                    "cursor_base_id": _TREE.cursor_base_id,
                },
            }
        )


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
            # Backward-compatible aliases (optional)
            if action == "TOGGLE_HOLD":
                _TREE.cursor.cursor_toggle_hold(actor)
            else:
                _TREE.cursor_apply_control(actor, action)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        cursor = _TREE.cursor
        return jsonify(
            {
                "p1": view_state(cursor, "P1"),
                "p2": view_state(cursor, "P2"),
                "tree": {
                    "root_id": _TREE.root_id,
                    "current_id": _TREE.current_id,
                    "cursor_base_id": _TREE.cursor_base_id,
                },
            }
        )


@app.post("/api/commit")
def api_commit():
    with _LOCK:
        _TREE.commit_cursor_step()
        cursor = _TREE.cursor
        return jsonify(
            {
                "p1": view_state(cursor, "P1"),
                "p2": view_state(cursor, "P2"),
                "tree": {
                    "root_id": _TREE.root_id,
                    "current_id": _TREE.current_id,
                    "cursor_base_id": _TREE.cursor_base_id,
                },
            }
        )


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

        cursor = _TREE.cursor
        return jsonify(
            {
                "p1": view_state(cursor, "P1"),
                "p2": view_state(cursor, "P2"),
                "tree": {
                    "root_id": _TREE.root_id,
                    "current_id": _TREE.current_id,
                    "cursor_base_id": _TREE.cursor_base_id,
                },
            }
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
